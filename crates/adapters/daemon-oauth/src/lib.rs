// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-oauth` — ONE descriptor-driven OAuth2 / OIDC authorization-code + PKCE engine covering
//! every authorization-code variant (`daemon-interactive-auth-spec.md` §7).
//!
//! A single [`DescriptorFlowFactory`] implements the whole begin/complete flow; an
//! [`OAuthFlowDescriptor`] parameterizes it (endpoints, client id, scopes, callback param name,
//! CSRF `state` on/off, exchange style, credential shape). PKCE (S256) is **always on**. The daemon
//! registers one factory per descriptor next to the Matrix SSO factory, so a decoupled (possibly
//! remote) client drives a browser-redirect login over the wire `AuthApi`:
//!
//! - `auth_begin` (→ [`DescriptorFlowFactory::begin`]): mints a fresh PKCE `code_verifier` + S256
//!   `code_challenge` (+ a CSRF `state` when the descriptor enables it), builds the authorization
//!   URL against the *client-owned* callback (under the descriptor's callback param — RFC
//!   `redirect_uri` or OpenRouter's `callback_url`), and parks the continuation.
//! - `auth_complete` (→ [`PendingAuthFlow::complete`]): validates `state` (when enabled), exchanges
//!   `code` + `code_verifier` through the one SSRF-safe [`daemon_egress::EgressClient`] (redirects
//!   never followed) — either an RFC 6749 `application/x-www-form-urlencoded` POST or a JSON POST —
//!   and returns the credential blob + slot the node persists.
//!
//! ## The descriptors
//!
//! - [`generic_oauth2`] — the operator-facing generic `oauth2` family: endpoints + client_id arrive
//!   in `auth_begin.params`, `use_state` on, RFC form-post exchange, the raw token JSON stored under
//!   a `oauth2/<label>` ref (identity: explicit `account_label` → `id_token` `sub` → stable hash).
//!   Byte-identical to the pre-refactor factory (no behavior change for operators).
//! - [`openrouter`] — a curated provider-bound family (`"provider/openrouter"`, EMPTY params
//!   schema — the node owns every parameter). Deliberately non-RFC: authorize at
//!   `https://openrouter.ai/auth?callback_url=<redirect>&code_challenge=<S256>&code_challenge_method=S256`
//!   (no `client_id`, no `state` — PKCE binds the flow), exchange is a JSON POST to
//!   `https://openrouter.ai/api/v1/auth/keys` with `{code, code_verifier, code_challenge_method}`
//!   returning `{"key": "..."}`. The minted `key` is an ordinary OpenRouter API key, stored BARE
//!   under the bound profile's credential slot ([`CredentialSlotKind::ProviderKeyForProfile`]).
//! - [`huggingface`] — a curated provider-bound family (`"provider/huggingface"`, EMPTY params
//!   schema), gated on an operator-supplied `client_id` in node config: standard OIDC
//!   (`https://huggingface.co/oauth/authorize` + `/oauth/token`), `inference-api` scope, `use_state`
//!   on, RFC form-post exchange, provider-key credential shape.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use sha2::{Digest, Sha256};

use daemon_api::{ApiError, AuthFlowKind, AuthParamField, AuthProviderInfo};
use daemon_egress::{EgressClient, EgressConfig, EgressRequest, Redirects};
use daemon_host::{AuthFlowFactory, AuthOutcome, CredentialSlotKind, PendingAuthFlow};
use daemon_protocol::TransportId;

/// The generic operator-facing family (`auth_begin.family`).
pub const FAMILY: &str = "oauth2";
/// The OpenRouter provider-bound family — the auth family a `ProviderDescriptor.sign_in`
/// advertisement points at. Its `params_schema` is empty (the node owns every parameter).
pub const OPENROUTER_FAMILY: &str = "provider/openrouter";
/// The Hugging Face provider-bound family (registered only when an operator supplies a client id).
pub const HUGGINGFACE_FAMILY: &str = "provider/huggingface";

/// The `auth_begin` param naming the IdP's authorization endpoint URL (generic family).
pub const PARAM_AUTHORIZATION_ENDPOINT: &str = "authorization_endpoint";
/// The `auth_begin` param naming the IdP's token endpoint URL (generic family).
pub const PARAM_TOKEN_ENDPOINT: &str = "token_endpoint";
/// The `auth_begin` param naming the OAuth2 client id (generic family).
pub const PARAM_CLIENT_ID: &str = "client_id";
/// The `auth_begin` param carrying the space-delimited scope list (optional).
pub const PARAM_SCOPES: &str = "scopes";
/// The `auth_begin` param carrying a confidential-client secret (optional; PKCE public clients omit).
pub const PARAM_CLIENT_SECRET: &str = "client_secret";
/// The `auth_begin` param naming the account label / credential slot explicitly (optional).
pub const PARAM_ACCOUNT_LABEL: &str = "account_label";

/// The per-request deadline on the token exchange.
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);

/// A descriptor field resolved either from a node-fixed value or a client `begin` param.
#[derive(Clone, Debug)]
pub enum Source {
    /// A node-owned constant (curated families).
    Fixed(&'static str),
    /// Read from `auth_begin.params[key]` (the generic family).
    Param(&'static str),
}

impl Source {
    /// Resolve a required value: a fixed constant, or a non-empty begin param.
    fn require(&self, params: &BTreeMap<String, String>) -> Result<String, ApiError> {
        match self {
            Source::Fixed(v) => Ok((*v).to_string()),
            Source::Param(key) => require(params, key).map(str::to_string),
        }
    }

    /// Resolve an optional value: a fixed constant, or a (possibly absent) begin param.
    fn resolve_opt(&self, params: &BTreeMap<String, String>) -> Option<String> {
        match self {
            Source::Fixed("") => None,
            Source::Fixed(v) => Some((*v).to_string()),
            Source::Param(key) => params.get(*key).filter(|s| !s.is_empty()).cloned(),
        }
    }
}

/// The callback query parameter the authorization URL carries the client-owned redirect under.
#[derive(Clone, Copy, Debug)]
pub enum CallbackParam {
    /// RFC 6749 `redirect_uri` (also emits `response_type=code`).
    RedirectUri,
    /// OpenRouter's `callback_url` (no `response_type`).
    CallbackUrl,
}

/// How the authorization `code` + PKCE `code_verifier` are exchanged for the credential.
#[derive(Clone, Debug)]
pub enum ExchangeStyle {
    /// RFC 6749 §4.1.3 `application/x-www-form-urlencoded` POST.
    FormPost,
    /// A JSON POST of `{code, code_verifier, code_challenge_method: "S256"}`; the response field
    /// named `key_field` carries the minted secret (OpenRouter: `"key"`).
    JsonPost { key_field: &'static str },
}

/// What the completed flow yields as the stored credential + how the node slots it.
#[derive(Clone, Debug)]
pub enum CredentialShape {
    /// Store the raw token-response JSON as the blob under a family-derived `oauth2/<label>` ref
    /// ([`CredentialSlotKind::Derived`]); identity: explicit label → `id_token` `sub` → stable hash.
    RawTokenJson,
    /// Store the BARE minted key (the exchange's `key_field`) under the bound profile's credential
    /// slot ([`CredentialSlotKind::ProviderKeyForProfile`]); `account_label` is the fixed provider
    /// name. Requires a [`ExchangeStyle::JsonPost`] exchange.
    ProviderKey { account_label: &'static str },
}

/// The parameterization of the one OAuth engine — one begin/complete implementation covers every
/// authorization-code variant by binding these fields (PKCE S256 is always on).
#[derive(Clone, Debug)]
pub struct OAuthFlowDescriptor {
    /// The auth family this descriptor serves (`auth_begin.family`).
    pub family: &'static str,
    /// A human display name for capability discovery.
    pub display_name: &'static str,
    /// The IdP authorization endpoint.
    pub authorization_endpoint: Source,
    /// The IdP token endpoint (the exchange target).
    pub token_endpoint: Source,
    /// The OAuth2 client id, when the flow carries one (`None` = omit — OpenRouter binds via PKCE).
    pub client_id: Option<Source>,
    /// The `auth_begin` param carrying a confidential-client secret (`None` = public client).
    pub client_secret_param: Option<&'static str>,
    /// The scope list, when the flow carries one (`None` = omit the `scope` param).
    pub scopes: Option<Source>,
    /// Which query param the client-owned callback rides under.
    pub callback_param: CallbackParam,
    /// Whether to mint + enforce a CSRF `state` (RFC); `false` binds the flow via PKCE alone.
    pub use_state: bool,
    /// The token-exchange style.
    pub exchange: ExchangeStyle,
    /// The credential shape + slot mapping.
    pub credential: CredentialShape,
    /// The `params` fields a client collects (EMPTY for provider-bound families — the node owns
    /// every parameter, so the client calls `auth_begin { family, params: {} }`).
    pub params_schema: Vec<AuthParamField>,
}

/// The operator-facing generic `oauth2` descriptor: client-supplied endpoints/client_id, RFC
/// form-post exchange, raw-token-JSON credential. Byte-identical to the pre-refactor factory.
pub fn generic_oauth2() -> OAuthFlowDescriptor {
    OAuthFlowDescriptor {
        family: FAMILY,
        display_name: "OAuth2 / OIDC (PKCE)",
        authorization_endpoint: Source::Param(PARAM_AUTHORIZATION_ENDPOINT),
        token_endpoint: Source::Param(PARAM_TOKEN_ENDPOINT),
        client_id: Some(Source::Param(PARAM_CLIENT_ID)),
        client_secret_param: Some(PARAM_CLIENT_SECRET),
        scopes: Some(Source::Param(PARAM_SCOPES)),
        callback_param: CallbackParam::RedirectUri,
        use_state: true,
        exchange: ExchangeStyle::FormPost,
        credential: CredentialShape::RawTokenJson,
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

/// The curated OpenRouter descriptor (`"provider/openrouter"`, empty params schema): non-RFC PKCE
/// with a JSON key-mint exchange whose `key` is an ordinary OpenRouter API key, slotted as a
/// provider key under the bound profile.
pub fn openrouter() -> OAuthFlowDescriptor {
    OAuthFlowDescriptor {
        family: OPENROUTER_FAMILY,
        display_name: "OpenRouter",
        authorization_endpoint: Source::Fixed("https://openrouter.ai/auth"),
        token_endpoint: Source::Fixed("https://openrouter.ai/api/v1/auth/keys"),
        client_id: None,
        client_secret_param: None,
        scopes: None,
        callback_param: CallbackParam::CallbackUrl,
        use_state: false,
        exchange: ExchangeStyle::JsonPost { key_field: "key" },
        credential: CredentialShape::ProviderKey {
            account_label: "openrouter",
        },
        params_schema: Vec::new(),
    }
}

/// The curated Hugging Face descriptor (`"provider/huggingface"`, empty params schema), gated on an
/// operator-supplied `client_id`: standard OIDC authorization-code + PKCE with the `inference-api`
/// scope, RFC form-post exchange, provider-key credential shape. Registered only when the operator
/// has configured `oauth.huggingface_client_id` (so an unconfigured node never advertises it).
pub fn huggingface(client_id: String) -> OAuthFlowDescriptor {
    // The client id is operator config, resolved once at registration; leak it to a `'static`
    // so the descriptor's `Source::Fixed` can carry it (the process lives for the descriptor).
    let client_id: &'static str = Box::leak(client_id.into_boxed_str());
    OAuthFlowDescriptor {
        family: HUGGINGFACE_FAMILY,
        display_name: "Hugging Face",
        authorization_endpoint: Source::Fixed("https://huggingface.co/oauth/authorize"),
        token_endpoint: Source::Fixed("https://huggingface.co/oauth/token"),
        client_id: Some(Source::Fixed(client_id)),
        client_secret_param: None,
        scopes: Some(Source::Fixed("inference-api")),
        callback_param: CallbackParam::RedirectUri,
        use_state: true,
        exchange: ExchangeStyle::FormPost,
        credential: CredentialShape::ProviderKey {
            account_label: "huggingface",
        },
        params_schema: Vec::new(),
    }
}

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
/// only to label the stored credential, never as an authentication decision.
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

/// The one OAuth engine: a factory over an [`OAuthFlowDescriptor`] + the shared SSRF-safe egress
/// client. Stateless beyond the descriptor; every flow's runtime values are resolved at `begin`.
pub struct DescriptorFlowFactory {
    descriptor: Arc<OAuthFlowDescriptor>,
    http: EgressClient,
}

impl DescriptorFlowFactory {
    /// A factory over `descriptor` with a fresh [`EgressClient`]. Fails only when the TLS backend
    /// cannot initialize (a boot-environment defect) — surfaced, not defaulted.
    pub fn new(descriptor: OAuthFlowDescriptor) -> Result<Self, ApiError> {
        let http = EgressClient::new(EgressConfig {
            user_agent: Some("daemon".to_string()),
            timeout: Some(TOKEN_EXCHANGE_TIMEOUT),
        })
        .map_err(|e| ApiError::Other(format!("oauth2: building egress client: {e}")))?;
        Ok(Self {
            descriptor: Arc::new(descriptor),
            http,
        })
    }
}

#[async_trait]
impl AuthFlowFactory for DescriptorFlowFactory {
    fn family(&self) -> &str {
        self.descriptor.family
    }

    fn provider_info(&self) -> AuthProviderInfo {
        AuthProviderInfo {
            family: self.descriptor.family.to_string(),
            flow_kind: AuthFlowKind::OAuth2Pkce,
            display_name: self.descriptor.display_name.to_string(),
            params_schema: self.descriptor.params_schema.clone(),
        }
    }

    async fn begin(
        &self,
        params: &BTreeMap<String, String>,
        redirect_uri: &str,
    ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
        let d = &self.descriptor;
        let authorization_endpoint = d.authorization_endpoint.require(params)?;
        let token_endpoint = d.token_endpoint.require(params)?;
        let client_id = match &d.client_id {
            Some(src) => Some(src.require(params)?),
            None => None,
        };
        let client_secret = d
            .client_secret_param
            .and_then(|key| params.get(key).filter(|s| !s.is_empty()).cloned());
        let scopes = d.scopes.as_ref().and_then(|src| src.resolve_opt(params));

        let verifier = random_urlsafe(32)?;
        let challenge = s256_challenge(&verifier);
        let state = if d.use_state {
            Some(random_urlsafe(16)?)
        } else {
            None
        };

        let mut url = url::Url::parse(&authorization_endpoint).map_err(|e| {
            ApiError::Other(format!("oauth2 auth: invalid authorization endpoint: {e}"))
        })?;
        {
            let mut q = url.query_pairs_mut();
            // The client-owned callback under the descriptor's param; RFC flows also carry
            // `response_type=code` (OpenRouter's authorize endpoint takes neither that nor a state).
            match d.callback_param {
                CallbackParam::RedirectUri => {
                    q.append_pair("response_type", "code")
                        .append_pair("redirect_uri", redirect_uri);
                }
                CallbackParam::CallbackUrl => {
                    q.append_pair("callback_url", redirect_uri);
                }
            }
            if let Some(cid) = &client_id {
                q.append_pair("client_id", cid);
            }
            if let Some(state) = &state {
                q.append_pair("state", state);
            }
            q.append_pair("code_challenge", &challenge)
                .append_pair("code_challenge_method", "S256");
            if let Some(scopes) = &scopes {
                if !scopes.is_empty() {
                    q.append_pair("scope", scopes);
                }
            }
        }

        Ok(Box::new(DescriptorPendingFlow {
            descriptor: self.descriptor.clone(),
            http: self.http.clone(),
            authorization_url: url.into(),
            token_endpoint,
            client_id,
            client_secret,
            explicit_label: params.get(PARAM_ACCOUNT_LABEL).cloned(),
            redirect_uri: redirect_uri.to_string(),
            verifier,
            state,
        }))
    }
}

/// A parked descriptor flow: the secret continuation state (`verifier` + expected `state` + the
/// resolved endpoint identity + the descriptor) held between `begin` and `complete`.
struct DescriptorPendingFlow {
    descriptor: Arc<OAuthFlowDescriptor>,
    http: EgressClient,
    authorization_url: String,
    token_endpoint: String,
    client_id: Option<String>,
    client_secret: Option<String>,
    explicit_label: Option<String>,
    redirect_uri: String,
    verifier: String,
    state: Option<String>,
}

impl DescriptorPendingFlow {
    /// Parse + validate the captured callback (full redirect URL or bare query): surface an IdP
    /// `error`, enforce the `state` echo when the descriptor uses one, and extract the `code`.
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
        // Mandatory CSRF gate WHEN the descriptor mints a state: the redirect must echo it. A
        // PKCE-only descriptor (OpenRouter) binds the flow via the verifier and skips this.
        if let Some(expected) = &self.state {
            if state.as_deref() != Some(expected.as_str()) {
                return Err(ApiError::Other(
                    "oauth2 auth: state mismatch on callback (possible CSRF); restart the flow"
                        .into(),
                ));
            }
        }
        code.filter(|c| !c.is_empty())
            .ok_or_else(|| ApiError::Other("oauth2 auth: callback carries no `code`".into()))
    }

    /// Run the descriptor's token exchange, returning the parsed response JSON + its raw body.
    async fn exchange(&self, code: &str) -> Result<(serde_json::Value, String), ApiError> {
        let request = match &self.descriptor.exchange {
            ExchangeStyle::FormPost => {
                // RFC 6749 §4.1.3 + RFC 7636 §4.5. `client_id` is included when the flow carries one.
                let mut pairs: Vec<(&str, &str)> = vec![
                    ("grant_type", "authorization_code"),
                    ("code", code),
                    ("redirect_uri", &self.redirect_uri),
                    ("code_verifier", &self.verifier),
                ];
                if let Some(cid) = &self.client_id {
                    pairs.push(("client_id", cid));
                }
                if let Some(secret) = &self.client_secret {
                    pairs.push(("client_secret", secret));
                }
                EgressRequest::post_form(&self.token_endpoint, &pairs)
            }
            ExchangeStyle::JsonPost { .. } => {
                let body = serde_json::json!({
                    "code": code,
                    "code_verifier": self.verifier,
                    "code_challenge_method": "S256",
                });
                EgressRequest::post_json(&self.token_endpoint, &body)
                    .map_err(|e| ApiError::Other(format!("oauth2 auth: building exchange: {e}")))?
            }
        };
        // `Redirects::None` — a token endpoint never legitimately redirects (kills redirect-SSRF).
        let response = self
            .http
            .execute(request, Redirects::None)
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
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ApiError::Other(format!("oauth2 auth: token response not JSON: {e}")))?;
        Ok((json, body))
    }
}

#[async_trait]
impl PendingAuthFlow for DescriptorPendingFlow {
    fn authorization_url(&self) -> &str {
        &self.authorization_url
    }

    fn flow_kind(&self) -> AuthFlowKind {
        AuthFlowKind::OAuth2Pkce
    }

    async fn complete(self: Box<Self>, callback: &str) -> Result<AuthOutcome, ApiError> {
        let code = self.callback_code(callback)?;
        let (tokens, body) = self.exchange(&code).await?;

        match &self.descriptor.credential {
            CredentialShape::RawTokenJson => {
                if tokens
                    .get("access_token")
                    .and_then(|t| t.as_str())
                    .is_none()
                {
                    return Err(ApiError::Other(
                        "oauth2 auth: token response carries no access_token".into(),
                    ));
                }
                // Identity: explicit label → id_token sub → stable hash of (token_endpoint, client_id).
                let client_id = self.client_id.as_deref().unwrap_or_default();
                let label = self
                    .explicit_label
                    .clone()
                    .filter(|l| !l.is_empty())
                    .or_else(|| id_token_sub(&tokens))
                    .unwrap_or_else(|| derived_label(&self.token_endpoint, client_id));
                let family = self.descriptor.family;
                Ok(AuthOutcome {
                    // The raw token-response JSON is the opaque blob the node persists; consumers
                    // re-parse what they need.
                    credential_blob: body,
                    credential_ref: format!("{family}/{label}"),
                    account_label: label.clone(),
                    transport_instance: TransportId::new(format!("{family}/{label}")),
                    slot: CredentialSlotKind::Derived,
                })
            }
            CredentialShape::ProviderKey { account_label } => {
                let ExchangeStyle::JsonPost { key_field } = &self.descriptor.exchange else {
                    // A misconfigured descriptor (provider-key credential without a key-mint
                    // exchange) is a build defect, surfaced rather than mislabelling a blob.
                    return Err(ApiError::Other(
                        "oauth2 auth: provider-key credential requires a JSON key-mint exchange"
                            .into(),
                    ));
                };
                let key = tokens
                    .get(*key_field)
                    .and_then(|k| k.as_str())
                    .filter(|k| !k.is_empty())
                    .ok_or_else(|| {
                        ApiError::Other(format!(
                            "oauth2 auth: key-mint response carries no `{key_field}`"
                        ))
                    })?;
                let family = self.descriptor.family;
                Ok(AuthOutcome {
                    // The BARE minted key rides the exact downstream path as a pasted API key; the
                    // node slots it under the bound profile's credential slot (never the raw JSON).
                    credential_blob: key.to_string(),
                    // Informational default; `auth_complete` targets the bound profile slot instead.
                    credential_ref: format!("{family}/{account_label}"),
                    account_label: (*account_label).to_string(),
                    transport_instance: TransportId::new(format!("{family}/{account_label}")),
                    slot: CredentialSlotKind::ProviderKeyForProfile,
                })
            }
        }
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
        assert_eq!(
            s256_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            "RFC 7636 appendix B vector"
        );
    }

    #[tokio::test]
    async fn generic_begin_builds_a_pkce_authorization_url() {
        let factory = DescriptorFlowFactory::new(generic_oauth2()).unwrap();
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
    async fn generic_begin_requires_endpoints_and_client_id() {
        let factory = DescriptorFlowFactory::new(generic_oauth2()).unwrap();
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

    #[tokio::test]
    async fn openrouter_begin_is_pkce_only_with_callback_url() {
        let factory = DescriptorFlowFactory::new(openrouter()).unwrap();
        // The node owns every parameter — the client calls begin with an empty params map.
        let flow = factory
            .begin(&BTreeMap::new(), "http://127.0.0.1:7777/cb")
            .await
            .unwrap();
        let url = flow.authorization_url();
        assert!(url.starts_with("https://openrouter.ai/auth?"), "{url}");
        assert_eq!(
            query_param(url, "callback_url").as_deref(),
            Some("http://127.0.0.1:7777/cb"),
            "OpenRouter carries the callback under `callback_url`"
        );
        assert_eq!(
            query_param(url, "code_challenge_method").as_deref(),
            Some("S256")
        );
        assert_eq!(
            query_param(url, "code_challenge").map(|c| c.len()),
            Some(43)
        );
        // PKCE-only: no client_id, no state, no response_type, no redirect_uri.
        assert!(query_param(url, "client_id").is_none());
        assert!(query_param(url, "state").is_none());
        assert!(query_param(url, "response_type").is_none());
        assert!(query_param(url, "redirect_uri").is_none());
    }

    #[tokio::test]
    async fn huggingface_begin_is_rfc_with_inference_scope() {
        let factory = DescriptorFlowFactory::new(huggingface("hf-client-123".into())).unwrap();
        let flow = factory
            .begin(&BTreeMap::new(), "http://127.0.0.1:7777/cb")
            .await
            .unwrap();
        let url = flow.authorization_url();
        assert!(
            url.starts_with("https://huggingface.co/oauth/authorize?"),
            "{url}"
        );
        assert_eq!(query_param(url, "response_type").as_deref(), Some("code"));
        assert_eq!(
            query_param(url, "client_id").as_deref(),
            Some("hf-client-123")
        );
        assert_eq!(query_param(url, "scope").as_deref(), Some("inference-api"));
        assert!(query_param(url, "state").is_some(), "HF uses CSRF state");
        assert_eq!(
            query_param(url, "code_challenge_method").as_deref(),
            Some("S256")
        );
    }

    #[test]
    fn openrouter_family_constant_is_stable() {
        // The sibling wire stream's `ProviderDescriptor.sign_in` advertisement points at this exact
        // string; it must not drift.
        assert_eq!(OPENROUTER_FAMILY, "provider/openrouter");
        assert!(openrouter().params_schema.is_empty());
        assert!(huggingface("x".into()).params_schema.is_empty());
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
