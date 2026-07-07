// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Slack interactive auth as two client-driven flow factories (`daemon-interactive-auth-spec`).
//!
//! Slack has two account modes, each a distinct [`AuthFlowFactory`] (honest one-flow-kind-per-factory
//! capability discovery):
//!
//! - **bot/app** (family `slack`, [`AuthFlowKind::OAuth2Pkce`]): a browser-redirect authorization-code
//!   flow. `begin` mints the `https://slack.com/oauth/v2/authorize` URL against the client-owned
//!   `redirect_uri` (+ a CSRF `state`); the completing step exchanges the `code` at
//!   `oauth.v2.access` through the SSRF-safe [`EgressClient`] (the SDK's own OAuth surface is
//!   axum-coupled, so we own this plain form POST), yielding the workspace bot token. Modelled on
//!   `daemon-oauth`'s descriptor flow. *(Slack's v2 exchange is authorization-code with a
//!   `client_secret`, not PKCE; the redirect challenge shape is identical, so `OAuth2Pkce` is the
//!   closest capability hint in the fixed [`AuthFlowKind`] set.)*
//! - **user** (family `slack-user`, [`AuthFlowKind::UserToken`]): a no-browser [`AuthChallenge::Form`]
//!   collecting a browser-extracted `xoxc` token + `xoxd` cookie, stored for the `slacko` stealth
//!   client. No exchange, no network.
//!
//! Both complete with [`CredentialSlotKind::Derived`] (a transport account, not a provider key).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::Deserialize;

use daemon_api::{
    ApiError, AuthChallenge, AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput,
};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};
use daemon_protocol::TransportId;

use crate::account::{StoredCredential, FAMILY};

/// The user-mode (stealth) auth family.
pub const USER_FAMILY: &str = "slack-user";

/// The Slack OAuth authorization endpoint.
const AUTHORIZE_URL: &str = "https://slack.com/oauth/v2/authorize";
/// The Slack OAuth token-exchange endpoint (`oauth.v2.access`).
const TOKEN_ENDPOINT: &str = "https://slack.com/api/oauth.v2.access";
/// A sensible default bot scope set (used when the operator supplies none).
const DEFAULT_BOT_SCOPE: &str =
    "app_mentions:read,channels:history,channels:read,chat:write,groups:read,im:history,im:read";
/// The per-request deadline on the token exchange.
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);

/// `auth_begin` param: the Slack app client id (bot family, required).
pub const PARAM_CLIENT_ID: &str = "client_id";
/// `auth_begin` param: the Slack app client secret (bot family, required).
pub const PARAM_CLIENT_SECRET: &str = "client_secret";
/// `auth_begin` param: the comma-delimited bot scope list (bot family, optional).
pub const PARAM_SCOPE: &str = "scope";
/// Form field: the browser `xoxc` token (user family, required).
pub const FIELD_XOXC: &str = "xoxc_token";
/// Form field: the browser `xoxd` cookie (user family, required).
pub const FIELD_XOXD: &str = "xoxd_cookie";
/// Form field / `auth_begin` param: a human label for the account instance (user family, optional).
pub const FIELD_LABEL: &str = "label";

/// A best-effort, process-locally-unique, hard-to-guess CSRF `state` (counter + time + stack seed,
/// hex). The `state` is a short-lived anti-CSRF nonce echoed on the redirect, not the auth secret.
fn fresh_state() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stack = &n as *const _ as usize as u128;
    format!("{n:016x}{nanos:032x}{stack:016x}")
}

/// Require a non-empty begin param.
fn require<'p>(params: &'p BTreeMap<String, String>, key: &str) -> Result<&'p str, ApiError> {
    params
        .get(key)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::Other(format!("slack auth: missing `{key}`")))
}

// ---------------------------------------------------------------------------
// bot/app OAuth
// ---------------------------------------------------------------------------

/// The bot/app OAuth factory: one flow per app install, exchanging through the shared egress client.
pub struct SlackBotAuthFlowFactory {
    http: EgressClient,
    token_endpoint: String,
}

impl SlackBotAuthFlowFactory {
    /// A factory with a fresh SSRF-safe [`EgressClient`]. Fails only when the TLS backend cannot
    /// initialise (a boot-environment defect).
    pub fn new() -> Result<Self, ApiError> {
        let http = EgressClient::new(EgressConfig {
            user_agent: Some("daemon".to_string()),
            timeout: Some(TOKEN_EXCHANGE_TIMEOUT),
        })
        .map_err(|e| ApiError::Other(format!("slack auth: building egress client: {e}")))?;
        Ok(Self {
            http,
            token_endpoint: TOKEN_ENDPOINT.to_string(),
        })
    }
}

#[async_trait]
impl AuthFlowFactory for SlackBotAuthFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::OAuth2Pkce,
            display_name: "Slack (bot/app)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_CLIENT_ID.to_string(),
                    label: "Slack app client id".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_CLIENT_SECRET.to_string(),
                    label: "Slack app client secret".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_SCOPE.to_string(),
                    label: "Bot scopes (comma-delimited, optional)".to_string(),
                    required: false,
                },
            ],
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let client_id = require(params, PARAM_CLIENT_ID)?.to_string();
        let client_secret = require(params, PARAM_CLIENT_SECRET)?.to_string();
        let scope = params
            .get(PARAM_SCOPE)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| DEFAULT_BOT_SCOPE.to_string());
        let state = fresh_state();

        let mut url = url::Url::parse(AUTHORIZE_URL)
            .map_err(|e| ApiError::Other(format!("slack auth: invalid authorize url: {e}")))?;
        url.query_pairs_mut()
            .append_pair("client_id", &client_id)
            .append_pair("scope", &scope)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("state", &state);

        Ok(Box::new(BotOAuthFlow {
            http: self.http.clone(),
            token_endpoint: self.token_endpoint.clone(),
            authorization_url: url.into(),
            client_id,
            client_secret,
            redirect_uri: redirect_uri.to_string(),
            state,
        }))
    }
}

/// A parked bot OAuth flow: the continuation state held across the browser hop.
struct BotOAuthFlow {
    http: EgressClient,
    token_endpoint: String,
    authorization_url: String,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    state: String,
}

/// The `oauth.v2.access` response shape we consume (a subset of Slack's fields).
#[derive(Debug, Deserialize)]
struct OAuthAccessResponse {
    ok: bool,
    error: Option<String>,
    access_token: Option<String>,
    bot_user_id: Option<String>,
    team: Option<TeamRef>,
}

/// The `team` object of an `oauth.v2.access` response.
#[derive(Debug, Deserialize)]
struct TeamRef {
    id: String,
    name: Option<String>,
}

impl BotOAuthFlow {
    /// Parse + validate the captured callback (full URL or bare query): surface an IdP `error`,
    /// enforce the `state` echo, and extract the `code`.
    fn callback_code(&self, callback: &str) -> Result<String, ApiError> {
        let query = callback.split_once('?').map(|(_, q)| q).unwrap_or(callback);
        let mut code = None;
        let mut state = None;
        let mut error = None;
        for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
            match k.as_ref() {
                "code" => code = Some(v.into_owned()),
                "state" => state = Some(v.into_owned()),
                "error" => error = Some(v.into_owned()),
                _ => {}
            }
        }
        if let Some(error) = error {
            return Err(ApiError::Other(format!(
                "slack auth: authorization failed: {error}"
            )));
        }
        if state.as_deref() != Some(self.state.as_str()) {
            return Err(ApiError::Other(
                "slack auth: state mismatch on callback (possible CSRF); restart the flow".into(),
            ));
        }
        code.filter(|c| !c.is_empty())
            .ok_or_else(|| ApiError::Other("slack auth: callback carries no `code`".into()))
    }

    /// Exchange `code` at `oauth.v2.access` and build the [`AuthOutcome`] the node persists.
    async fn complete(&self, callback: &str) -> Result<AuthOutcome, ApiError> {
        let code = self.callback_code(callback)?;
        let request = EgressRequest::post_form(
            &self.token_endpoint,
            &[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("code", code.as_str()),
                ("redirect_uri", self.redirect_uri.as_str()),
            ],
        );
        // `Redirects::None` — a token endpoint never legitimately redirects (kills redirect-SSRF).
        let response = self
            .http
            .execute(request, Redirects::None)
            .await
            .map_err(|e| ApiError::Other(format!("slack auth: token exchange: {e}")))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ApiError::Other(format!("slack auth: reading token response: {e}")))?;
        if !status.is_success() {
            return Err(ApiError::Other(format!(
                "slack auth: token endpoint returned {status}: {body}"
            )));
        }
        let parsed: OAuthAccessResponse = serde_json::from_str(&body)
            .map_err(|e| ApiError::Other(format!("slack auth: token response not JSON: {e}")))?;
        if !parsed.ok {
            return Err(ApiError::Other(format!(
                "slack auth: oauth.v2.access failed: {}",
                parsed.error.unwrap_or_else(|| "unknown error".into())
            )));
        }
        let bot_token = parsed
            .access_token
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                ApiError::Other("slack auth: oauth.v2.access carries no access_token".into())
            })?;
        let team = parsed
            .team
            .ok_or_else(|| ApiError::Other("slack auth: oauth.v2.access carries no team".into()))?;
        let bot_user_id = parsed.bot_user_id.unwrap_or_default();
        let label = team.name.clone().unwrap_or_else(|| team.id.clone());
        let credential = StoredCredential::Bot {
            bot_token,
            team_id: team.id.clone(),
            bot_user_id,
        };
        let credential_blob = credential
            .to_blob()
            .map_err(|e| ApiError::Other(format!("slack auth: serializing credential: {e}")))?;
        Ok(AuthOutcome {
            credential_blob,
            credential_ref: format!("{FAMILY}/{}", team.id),
            account_label: label,
            transport_instance: TransportId::new(format!("{FAMILY}/{}", team.id)),
            slot: CredentialSlotKind::Derived,
        })
    }
}

#[async_trait]
impl PendingAuthFlow for BotOAuthFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        AuthChallenge::Redirect {
            authorization_url: self.authorization_url.clone(),
        }
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Callback(callback) = input else {
            return Err(ApiError::Other(
                "slack auth: bot/app flow expects the captured redirect callback".into(),
            ));
        };
        Ok(AuthStepOutcome::Completed(self.complete(&callback).await?))
    }
}

// ---------------------------------------------------------------------------
// user (stealth) token form
// ---------------------------------------------------------------------------

/// The user-mode (stealth) auth factory: a no-browser form collecting the xoxc/xoxd pair.
pub struct SlackUserAuthFlowFactory;

#[async_trait]
impl AuthFlowFactory for SlackUserAuthFlowFactory {
    fn family(&self) -> &str {
        USER_FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: USER_FAMILY.to_string(),
            flow_kind: AuthFlowKind::UserToken,
            display_name: "Slack (user / stealth)".to_string(),
            // The tokens are collected via the initial Form challenge, not begin params.
            params_schema: Vec::new(),
        }
    }

    async fn begin(
        &self,
        _params: &BTreeMap<String, String>,
        _redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        Ok(Box::new(UserTokenFlow))
    }
}

/// A user (stealth) token flow: presents a form, completes on the filled fields (no network).
struct UserTokenFlow;

/// The form challenge presented by [`UserTokenFlow`] (shared with the completion validation).
fn user_form() -> AuthChallenge {
    AuthChallenge::Form {
        title: "Slack stealth login (browser-extracted tokens)".to_string(),
        fields: vec![
            AuthParamField {
                key: FIELD_XOXC.to_string(),
                label: "xoxc token (from the browser session)".to_string(),
                required: true,
            },
            AuthParamField {
                key: FIELD_XOXD.to_string(),
                label: "xoxd cookie (the `d` cookie value)".to_string(),
                required: true,
            },
            AuthParamField {
                key: FIELD_LABEL.to_string(),
                label: "Account label (optional)".to_string(),
                required: false,
            },
        ],
    }
}

#[async_trait]
impl PendingAuthFlow for UserTokenFlow {
    fn initial_challenge(&self) -> AuthChallenge {
        user_form()
    }

    async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
        let AuthStepInput::Fields(fields) = input else {
            return Err(ApiError::Other(
                "slack auth: user flow expects the filled token fields".into(),
            ));
        };
        let xoxc = fields
            .get(FIELD_XOXC)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::Other(format!("slack auth: missing `{FIELD_XOXC}`")))?;
        let xoxd = fields
            .get(FIELD_XOXD)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::Other(format!("slack auth: missing `{FIELD_XOXD}`")))?;
        let label = fields
            .get(FIELD_LABEL)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| "user".to_string());

        let credential = StoredCredential::User {
            xoxc_token: xoxc.clone(),
            xoxd_cookie: xoxd.clone(),
        };
        let credential_blob = credential
            .to_blob()
            .map_err(|e| ApiError::Other(format!("slack auth: serializing credential: {e}")))?;
        Ok(AuthStepOutcome::Completed(AuthOutcome {
            credential_blob,
            credential_ref: format!("{USER_FAMILY}/{label}"),
            account_label: label.clone(),
            transport_instance: TransportId::new(format!("{FAMILY}/{label}")),
            slot: CredentialSlotKind::Derived,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The authorization URL a redirect flow presents as its initial challenge.
    fn redirect_url(flow: &dyn PendingAuthFlow) -> String {
        match flow.initial_challenge() {
            AuthChallenge::Redirect { authorization_url } => authorization_url,
            other => panic!("expected a redirect challenge, got {other:?}"),
        }
    }

    fn query_param(url: &str, key: &str) -> Option<String> {
        url::Url::parse(url)
            .unwrap()
            .query_pairs()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.into_owned())
    }

    #[tokio::test]
    async fn bot_begin_builds_authorize_url_and_step_exchanges_code() {
        // Mock `oauth.v2.access` — obviously-fake tokens, no live network.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/oauth.v2.access"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "access_token": "xoxb-fake-bot-token",
                "bot_user_id": "U0BOT",
                "team": { "id": "T0TEAM", "name": "Acme" }
            })))
            .mount(&server)
            .await;

        let factory = SlackBotAuthFlowFactory {
            http: EgressClient::new(EgressConfig::default()).unwrap(),
            token_endpoint: format!("{}/api/oauth.v2.access", server.uri()),
        };
        let params: BTreeMap<String, String> = [
            (PARAM_CLIENT_ID.to_string(), "cid-123".to_string()),
            (PARAM_CLIENT_SECRET.to_string(), "csecret-456".to_string()),
        ]
        .into_iter()
        .collect();
        let flow = factory
            .begin(&params, "http://127.0.0.1:7777/cb")
            .await
            .unwrap();

        let url = redirect_url(flow.as_ref());
        assert!(
            url.starts_with("https://slack.com/oauth/v2/authorize?"),
            "{url}"
        );
        assert_eq!(query_param(&url, "client_id").as_deref(), Some("cid-123"));
        assert_eq!(
            query_param(&url, "redirect_uri").as_deref(),
            Some("http://127.0.0.1:7777/cb")
        );
        let state = query_param(&url, "state").expect("state present");

        // Complete with the captured redirect (echoing the minted state).
        let callback = format!("http://127.0.0.1:7777/cb?code=the-code&state={state}");
        let outcome = match flow.step(AuthStepInput::Callback(callback)).await.unwrap() {
            AuthStepOutcome::Completed(o) => o,
            AuthStepOutcome::Challenge(c) => panic!("expected completion, got {c:?}"),
        };
        assert_eq!(outcome.transport_instance, TransportId::new("slack/T0TEAM"));
        assert_eq!(outcome.account_label, "Acme");
        assert_eq!(outcome.slot, CredentialSlotKind::Derived);
        let cred = StoredCredential::from_blob(&outcome.credential_blob).unwrap();
        assert_eq!(
            cred,
            StoredCredential::Bot {
                bot_token: "xoxb-fake-bot-token".into(),
                team_id: "T0TEAM".into(),
                bot_user_id: "U0BOT".into(),
            }
        );
    }

    #[tokio::test]
    async fn bot_step_rejects_state_mismatch() {
        let factory = SlackBotAuthFlowFactory::new().unwrap();
        let params: BTreeMap<String, String> = [
            (PARAM_CLIENT_ID.to_string(), "cid".to_string()),
            (PARAM_CLIENT_SECRET.to_string(), "sec".to_string()),
        ]
        .into_iter()
        .collect();
        let flow = factory.begin(&params, "http://127.0.0.1/cb").await.unwrap();
        let result = flow
            .step(AuthStepInput::Callback(
                "http://127.0.0.1/cb?code=x&state=not-the-minted-state".into(),
            ))
            .await;
        assert!(
            matches!(result, Err(ApiError::Other(_))),
            "state mismatch must fail"
        );
    }

    #[tokio::test]
    async fn user_form_flow_completes_from_filled_fields() {
        let factory = SlackUserAuthFlowFactory;
        assert_eq!(factory.provider_info().flow_kind, AuthFlowKind::UserToken);
        let flow = factory.begin(&BTreeMap::new(), "").await.unwrap();
        // The initial challenge is a form collecting the stealth tokens.
        assert!(matches!(
            flow.initial_challenge(),
            AuthChallenge::Form { .. }
        ));

        let fields: BTreeMap<String, String> = [
            (FIELD_XOXC.to_string(), "xoxc-fake".to_string()),
            (FIELD_XOXD.to_string(), "xoxd-fake".to_string()),
            (FIELD_LABEL.to_string(), "acme-user".to_string()),
        ]
        .into_iter()
        .collect();
        let outcome = match flow.step(AuthStepInput::Fields(fields)).await.unwrap() {
            AuthStepOutcome::Completed(o) => o,
            AuthStepOutcome::Challenge(c) => panic!("expected completion, got {c:?}"),
        };
        assert_eq!(
            outcome.transport_instance,
            TransportId::new("slack/acme-user")
        );
        let cred = StoredCredential::from_blob(&outcome.credential_blob).unwrap();
        assert_eq!(
            cred,
            StoredCredential::User {
                xoxc_token: "xoxc-fake".into(),
                xoxd_cookie: "xoxd-fake".into(),
            }
        );
    }

    #[tokio::test]
    async fn user_step_requires_both_tokens() {
        let flow = UserTokenFlow;
        let only_xoxc: BTreeMap<String, String> =
            [(FIELD_XOXC.to_string(), "xoxc-fake".to_string())]
                .into_iter()
                .collect();
        assert!(flow.step(AuthStepInput::Fields(only_xoxc)).await.is_err());
    }
}
