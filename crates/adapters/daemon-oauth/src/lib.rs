// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-oauth` — the generic OAuth2 / OIDC authorization-code + PKCE interactive-auth family
//! (`daemon-interactive-auth-spec.md` §7, the A2 roadmap item).
//!
//! A production [`AuthFlowFactory`] for [`AuthFlowKind::OAuth2Pkce`], registered by the daemon
//! binary next to the Matrix SSO factory so a decoupled client can drive a browser login against
//! any authorization-code IdP over the wire `AuthApi`:
//!
//! - `auth_begin` (→ [`OAuth2PkceFlowFactory::begin`]): mints a fresh PKCE `code_verifier` +
//!   S256 `code_challenge` + CSRF `state`, builds the authorization URL against the
//!   *client-owned* `redirect_uri`, and parks `{ verifier, state, token_endpoint, … }` as the
//!   pending flow.
//! - `auth_complete` (→ [`PendingAuthFlow::complete`]): parses `code` + `state` from the captured
//!   callback, **validates `state`** (reject on mismatch), exchanges `code` + `code_verifier` at
//!   the token endpoint (an `application/x-www-form-urlencoded` POST through the one SSRF-safe
//!   [`daemon_egress::EgressClient`], redirects never followed), and hands the raw token-response
//!   JSON back as the credential blob the node persists.
//!
//! **Params-driven, not config-driven:** the family is generic (`"oauth2"`); the client supplies
//! `authorization_endpoint` / `token_endpoint` / `client_id` (+ optional `scopes`,
//! `client_secret`, `account_label`) in `auth_begin.params`, so any IdP works with zero node
//! config. A curated per-provider config table can layer on later without touching the wire.
//!
//! **Account identity derivation** (label → `credential_ref` `oauth2/<label>` and
//! `transport_instance` `oauth2/<label>`), in order:
//! 1. the explicit `account_label` param, when supplied;
//! 2. the `sub` claim of an `id_token` in the token response, when present — decoded **without
//!    signature verification**, which is sound here because the value only *labels* the stored
//!    credential (nothing authenticates against it; the tokens themselves came over TLS from the
//!    token endpoint we called);
//! 3. a stable short hash of `(token_endpoint, client_id)` — deterministic, so re-running the
//!    flow for the same app+IdP overwrites the same credential slot instead of minting siblings.
//!
//! The token endpoint may legitimately live on a private/loopback host (a self-hosted IdP), so
//! the exchange uses [`Redirects::None`] — the daemon-egress mode for trusted, non-redirecting
//! peers: no redirect is ever followed (killing the redirect-SSRF and credential-leak vectors)
//! and the operator-configured host is not subjected to the public-host gate.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use sha2::{Digest, Sha256};

use daemon_api::{ApiError, AuthFlowKind, AuthParamField, AuthProviderInfo};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_host::{AuthFlowFactory, AuthOutcome, PendingAuthFlow};
use daemon_protocol::TransportId;

/// The transport/provider family this factory serves (`auth_begin.family`).
pub const FAMILY: &str = "oauth2";

/// The `auth_begin` param naming the IdP's authorization endpoint URL (required).
pub const PARAM_AUTHORIZATION_ENDPOINT: &str = "authorization_endpoint";
/// The `auth_begin` param naming the IdP's token endpoint URL (required).
pub const PARAM_TOKEN_ENDPOINT: &str = "token_endpoint";
/// The `auth_begin` param naming the OAuth2 client id (required).
pub const PARAM_CLIENT_ID: &str = "client_id";
/// The `auth_begin` param carrying the space-delimited scope list (optional).
pub const PARAM_SCOPES: &str = "scopes";
/// The `auth_begin` param carrying a confidential-client secret (optional; PKCE public clients
/// omit it).
pub const PARAM_CLIENT_SECRET: &str = "client_secret";
/// The `auth_begin` param naming the account label / credential slot explicitly (optional; see
/// the crate docs for the derivation order when absent).
pub const PARAM_ACCOUNT_LABEL: &str = "account_label";

/// The per-request deadline on the token exchange.
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);

/// `n` OS-entropy bytes as unpadded base64url — the PKCE verifier/state alphabet (RFC 7636's
/// unreserved set). 32 bytes → 43 chars, the spec's minimum verifier length.
fn random_urlsafe(n: usize) -> Result<String, ApiError> {
    let mut buf = vec![0u8; n];
    // Fail closed: PKCE material minted from a broken entropy source would be guessable.
    getrandom::getrandom(&mut buf)
        .map_err(|e| ApiError::Other(format!("oauth2: OS entropy unavailable: {e}")))?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

/// The S256 code challenge for `verifier`: `base64url(sha256(ascii(verifier)))` (RFC 7636 §4.2).
fn s256_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// The `sub` claim of a JWT `id_token` in `tokens`, decoded WITHOUT signature verification — used
/// only to label the stored credential (crate docs), never as an authentication decision.
fn id_token_sub(tokens: &serde_json::Value) -> Option<String> {
    let jwt = tokens.get("id_token")?.as_str()?;
    let payload = jwt.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(claims.get("sub")?.as_str()?.to_string())
}

/// The deterministic fallback label for `(token_endpoint, client_id)`: `acct-<16 hex>` over a
/// SHA-256 prefix, so the same app+IdP always lands in the same credential slot.
fn derived_label(token_endpoint: &str, client_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token_endpoint.as_bytes());
    hasher.update(b"|");
    hasher.update(client_id.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(5 + 16);
    out.push_str("acct-");
    for byte in digest.iter().take(8) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Fetch a required, non-empty param.
fn require<'p>(params: &'p BTreeMap<String, String>, key: &str) -> Result<&'p str, ApiError> {
    params
        .get(key)
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::Other(format!("oauth2 auth: missing `{key}`")))
}

/// The generic OAuth2 authorization-code + PKCE factory: registered with the node so a client can
/// drive any authorization-code IdP over the wire `AuthApi`. Stateless beyond the shared egress
/// client; every flow's endpoints/identity arrive in the begin params.
pub struct OAuth2PkceFlowFactory {
    http: EgressClient,
}

impl OAuth2PkceFlowFactory {
    /// A factory over a fresh [`EgressClient`]. Fails only when the TLS backend cannot initialize
    /// (a boot-environment defect) — surfaced, not defaulted.
    pub fn new() -> Result<Self, ApiError> {
        let http = EgressClient::new(EgressConfig {
            user_agent: Some("daemon".to_string()),
            timeout: Some(TOKEN_EXCHANGE_TIMEOUT),
        })
        .map_err(|e| ApiError::Other(format!("oauth2: building egress client: {e}")))?;
        Ok(Self { http })
    }
}

#[async_trait]
impl AuthFlowFactory for OAuth2PkceFlowFactory {
    fn family(&self) -> &str {
        FAMILY
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: FAMILY.to_string(),
            flow_kind: AuthFlowKind::OAuth2Pkce,
            display_name: "OAuth2 / OIDC (PKCE)".to_string(),
            params_schema: vec![
                AuthParamField {
                    key: PARAM_AUTHORIZATION_ENDPOINT.to_string(),
                    label: "Authorization endpoint URL".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_TOKEN_ENDPOINT.to_string(),
                    label: "Token endpoint URL".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_CLIENT_ID.to_string(),
                    label: "Client id".to_string(),
                    required: true,
                },
                AuthParamField {
                    key: PARAM_SCOPES.to_string(),
                    label: "Scopes (space-delimited, optional)".to_string(),
                    required: false,
                },
                AuthParamField {
                    key: PARAM_CLIENT_SECRET.to_string(),
                    label: "Client secret (confidential clients only)".to_string(),
                    required: false,
                },
                AuthParamField {
                    key: PARAM_ACCOUNT_LABEL.to_string(),
                    label: "Account label (optional)".to_string(),
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
        let authorization_endpoint = require(params, PARAM_AUTHORIZATION_ENDPOINT)?;
        let token_endpoint = require(params, PARAM_TOKEN_ENDPOINT)?.to_string();
        let client_id = require(params, PARAM_CLIENT_ID)?.to_string();
        let scopes = params.get(PARAM_SCOPES).cloned().unwrap_or_default();

        let verifier = random_urlsafe(32)?;
        let state = random_urlsafe(16)?;
        let challenge = s256_challenge(&verifier);

        let mut url = url::Url::parse(authorization_endpoint).map_err(|e| {
            ApiError::Other(format!(
                "oauth2 auth: invalid `{PARAM_AUTHORIZATION_ENDPOINT}`: {e}"
            ))
        })?;
        {
            let mut q = url.query_pairs_mut();
            q.append_pair("response_type", "code")
                .append_pair("client_id", &client_id)
                .append_pair("redirect_uri", redirect_uri)
                .append_pair("state", &state)
                .append_pair("code_challenge", &challenge)
                .append_pair("code_challenge_method", "S256");
            if !scopes.is_empty() {
                q.append_pair("scope", &scopes);
            }
        }

        Ok(Box::new(OAuth2PendingFlow {
            http: self.http.clone(),
            authorization_url: url.into(),
            token_endpoint,
            client_id,
            client_secret: params.get(PARAM_CLIENT_SECRET).cloned(),
            account_label: params.get(PARAM_ACCOUNT_LABEL).cloned(),
            redirect_uri: redirect_uri.to_string(),
            verifier,
            state,
        }))
    }
}

/// A parked PKCE flow: the secret continuation state (`verifier` + expected `state` + the token
/// endpoint identity) held between `begin` and `complete`.
struct OAuth2PendingFlow {
    http: EgressClient,
    authorization_url: String,
    token_endpoint: String,
    client_id: String,
    client_secret: Option<String>,
    account_label: Option<String>,
    redirect_uri: String,
    verifier: String,
    state: String,
}

impl OAuth2PendingFlow {
    /// Parse + validate the captured callback (full redirect URL or bare query): surface an IdP
    /// `error`, enforce the `state` echo, and extract the authorization `code`.
    fn callback_code(&self, callback: &str) -> Result<String, ApiError> {
        let query = callback.split_once('?').map(|(_, q)| q).unwrap_or(callback);
        let mut code = None;
        let mut state = None;
        let mut error = None;
        let mut error_description = None;
        for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
            match k.as_ref() {
                "code" => code = Some(v.into_owned()),
                "state" => state = Some(v.into_owned()),
                "error" => error = Some(v.into_owned()),
                "error_description" => error_description = Some(v.into_owned()),
                _ => {}
            }
        }
        if let Some(error) = error {
            let detail = error_description.unwrap_or_default();
            return Err(ApiError::Other(format!(
                "oauth2 auth: authorization failed: {error} {detail}"
            )));
        }
        // Mandatory CSRF gate: the redirect must echo the exact state minted at begin.
        if state.as_deref() != Some(self.state.as_str()) {
            return Err(ApiError::Other(
                "oauth2 auth: state mismatch on callback (possible CSRF); restart the flow".into(),
            ));
        }
        code.filter(|c| !c.is_empty())
            .ok_or_else(|| ApiError::Other("oauth2 auth: callback carries no `code`".into()))
    }
}

#[async_trait]
impl PendingAuthFlow for OAuth2PendingFlow {
    fn authorization_url(&self) -> &str {
        &self.authorization_url
    }

    fn flow_kind(&self) -> AuthFlowKind {
        AuthFlowKind::OAuth2Pkce
    }

    async fn complete(self: Box<Self>, callback: &str) -> Result<AuthOutcome, ApiError> {
        let code = self.callback_code(callback)?;

        // RFC 6749 §4.1.3 + RFC 7636 §4.5: exchange code + verifier at the token endpoint. The
        // redirect_uri is echoed for the server-side binding check; a confidential client adds its
        // secret. `Redirects::None` — a token endpoint never legitimately redirects.
        let mut pairs: Vec<(&str, &str)> = vec![
            ("grant_type", "authorization_code"),
            ("code", &code),
            ("redirect_uri", &self.redirect_uri),
            ("client_id", &self.client_id),
            ("code_verifier", &self.verifier),
        ];
        if let Some(secret) = self.client_secret.as_deref() {
            pairs.push(("client_secret", secret));
        }
        let response = self
            .http
            .execute(
                EgressRequest::post_form(&self.token_endpoint, &pairs),
                Redirects::None,
            )
            .await
            .map_err(|e| ApiError::Other(format!("oauth2 auth: token exchange: {e}")))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ApiError::Other(format!("oauth2 auth: reading token response: {e}")))?;
        if !status.is_success() {
            return Err(ApiError::Other(format!(
                "oauth2 auth: token endpoint returned {status}: {body}"
            )));
        }
        let tokens: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ApiError::Other(format!("oauth2 auth: token response not JSON: {e}")))?;
        if tokens
            .get("access_token")
            .and_then(|t| t.as_str())
            .is_none()
        {
            return Err(ApiError::Other(
                "oauth2 auth: token response carries no access_token".into(),
            ));
        }

        // Identity derivation order (crate docs): explicit label → id_token sub → stable hash.
        let label = self
            .account_label
            .clone()
            .filter(|l| !l.is_empty())
            .or_else(|| id_token_sub(&tokens))
            .unwrap_or_else(|| derived_label(&self.token_endpoint, &self.client_id));

        Ok(AuthOutcome {
            // The raw token-response JSON (access/refresh/expiry/id_token) is the opaque blob the
            // node persists in the CredentialStore; consumers re-parse what they need.
            credential_blob: body,
            credential_ref: format!("{FAMILY}/{label}"),
            account_label: label.clone(),
            transport_instance: TransportId::new(format!("{FAMILY}/{label}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Unwrap an expected completion error (`AuthOutcome` deliberately has no `Debug` — it carries
    /// the secret blob — so `expect_err` is unavailable).
    fn completion_err(result: Result<AuthOutcome, ApiError>, ctx: &str) -> ApiError {
        match result {
            Err(e) => e,
            Ok(_) => panic!("{ctx}: expected an error"),
        }
    }

    fn base_params() -> BTreeMap<String, String> {
        params(&[
            (
                PARAM_AUTHORIZATION_ENDPOINT,
                "https://idp.example/authorize",
            ),
            (PARAM_TOKEN_ENDPOINT, "https://idp.example/token"),
            (PARAM_CLIENT_ID, "my-client"),
            (PARAM_SCOPES, "openid profile"),
        ])
    }

    /// Extract a query param from a URL.
    fn query_param(url: &str, key: &str) -> Option<String> {
        let parsed = url::Url::parse(url).unwrap();
        parsed
            .query_pairs()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.into_owned())
    }

    #[test]
    fn verifier_and_state_have_pkce_shape() {
        let verifier = random_urlsafe(32).unwrap();
        assert_eq!(verifier.len(), 43, "32 bytes -> 43 base64url chars");
        assert!(verifier
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert_ne!(random_urlsafe(32).unwrap(), verifier, "fresh per call");
        // The S256 challenge is the base64url sha256 of the ASCII verifier.
        assert_eq!(
            s256_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "RFC 7636 appendix B vector"
        );
    }

    #[tokio::test]
    async fn begin_builds_a_pkce_authorization_url() {
        let factory = OAuth2PkceFlowFactory::new().unwrap();
        let flow = factory
            .begin(&base_params(), "http://127.0.0.1:7777/cb")
            .await
            .unwrap();
        let url = flow.authorization_url();
        assert!(url.starts_with("https://idp.example/authorize?"), "{url}");
        assert_eq!(query_param(url, "response_type").as_deref(), Some("code"));
        assert_eq!(query_param(url, "client_id").as_deref(), Some("my-client"));
        assert_eq!(
            query_param(url, "redirect_uri").as_deref(),
            Some("http://127.0.0.1:7777/cb")
        );
        assert_eq!(
            query_param(url, "code_challenge_method").as_deref(),
            Some("S256")
        );
        assert_eq!(query_param(url, "scope").as_deref(), Some("openid profile"));
        assert_eq!(
            query_param(url, "code_challenge").map(|c| c.len()),
            Some(43),
            "S256 challenge is 43 base64url chars"
        );
        assert!(query_param(url, "state").is_some());
        assert_eq!(flow.flow_kind(), AuthFlowKind::OAuth2Pkce);
    }

    #[tokio::test]
    async fn begin_requires_endpoints_and_client_id() {
        let factory = OAuth2PkceFlowFactory::new().unwrap();
        for missing in [
            PARAM_AUTHORIZATION_ENDPOINT,
            PARAM_TOKEN_ENDPOINT,
            PARAM_CLIENT_ID,
        ] {
            let mut p = base_params();
            p.remove(missing);
            let err = factory.begin(&p, "http://127.0.0.1/cb").await;
            assert!(err.is_err(), "missing `{missing}` must fail begin");
        }
    }

    /// A callback whose `state` does not echo the minted one is rejected BEFORE any network I/O
    /// (the CSRF gate), as is an IdP error and a missing code.
    #[tokio::test]
    async fn complete_rejects_bad_callbacks_before_any_exchange() {
        let factory = OAuth2PkceFlowFactory::new().unwrap();
        let begin = || async {
            factory
                .begin(&base_params(), "http://127.0.0.1:7777/cb")
                .await
                .unwrap()
        };

        let flow = begin().await;
        let err = completion_err(
            flow.complete("http://127.0.0.1:7777/cb?code=abc&state=WRONG")
                .await,
            "state mismatch must be rejected",
        );
        assert!(err.to_string().contains("state mismatch"), "{err}");

        let flow = begin().await;
        let err = completion_err(
            flow.complete("http://127.0.0.1:7777/cb?error=access_denied&error_description=nope")
                .await,
            "an IdP error must be surfaced",
        );
        assert!(err.to_string().contains("access_denied"), "{err}");

        let flow = begin().await;
        let url = flow.authorization_url().to_string();
        let state = query_param(&url, "state").unwrap();
        let err = completion_err(
            flow.complete(&format!("http://127.0.0.1:7777/cb?state={state}"))
                .await,
            "a code-less callback must be rejected",
        );
        assert!(err.to_string().contains("no `code`"), "{err}");
    }

    #[test]
    fn id_token_sub_reads_the_unverified_payload() {
        let payload = URL_SAFE_NO_PAD.encode(r#"{"sub":"alice@idp"}"#);
        let tokens = serde_json::json!({ "id_token": format!("h.{payload}.sig") });
        assert_eq!(id_token_sub(&tokens).as_deref(), Some("alice@idp"));
        assert_eq!(id_token_sub(&serde_json::json!({})), None);
        assert_eq!(
            id_token_sub(&serde_json::json!({ "id_token": "garbage" })),
            None
        );
    }

    #[test]
    fn derived_label_is_stable_and_scoped() {
        let a = derived_label("https://idp.example/token", "client-1");
        assert_eq!(a, derived_label("https://idp.example/token", "client-1"));
        assert!(a.starts_with("acct-") && a.len() == 5 + 16, "{a}");
        assert_ne!(a, derived_label("https://idp.example/token", "client-2"));
        assert_ne!(a, derived_label("https://other.example/token", "client-1"));
    }
}
