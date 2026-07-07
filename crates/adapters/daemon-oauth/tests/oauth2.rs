// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Vertical tests for the descriptor-driven OAuth engine over a wiremock token endpoint: the
//! generic `oauth2` family's full begin → (browser hop) → complete form-post exchange (the PKCE
//! consistency proof — the `code_verifier` POSTed hashes (S256) to exactly the advertised
//! `code_challenge` — plus the identity-derivation order and the failure path), the CSRF `state`
//! gate for `use_state` true/false, and the provider-key JSON key-mint exchange (bare key stored,
//! slotted as a provider key). (The node-side orchestration — parking, credential write, profile
//! bind, provider-key slotting — is proven by the conformance suite; these own the protocol slice.)

use std::collections::BTreeMap;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use sha2::{Digest, Sha256};
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use daemon_api::{ApiError, AuthChallenge, AuthStepInput};
use daemon_host::{
    AuthFlowFactory, AuthOutcome, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow,
};
use daemon_oauth::{
    generic_oauth2, openrouter, DescriptorFlowFactory, FAMILY, OPENROUTER_FAMILY,
    PARAM_ACCOUNT_LABEL, PARAM_AUTHORIZATION_ENDPOINT, PARAM_CLIENT_ID, PARAM_SCOPES,
    PARAM_TOKEN_ENDPOINT,
};

/// The authorization URL a redirect flow presents as its initial challenge.
fn redirect_url(flow: &dyn PendingAuthFlow) -> String {
    match flow.initial_challenge() {
        AuthChallenge::Redirect { authorization_url } => authorization_url,
        other => panic!("expected a redirect challenge, got {other:?}"),
    }
}

/// Drive a single-redirect flow to completion via the captured callback (the `auth_complete` shape).
async fn complete(flow: &dyn PendingAuthFlow, callback: &str) -> Result<AuthOutcome, ApiError> {
    match flow
        .step(AuthStepInput::Callback(callback.to_string()))
        .await?
    {
        AuthStepOutcome::Completed(outcome) => Ok(outcome),
        AuthStepOutcome::Challenge(_) => panic!("expected the flow to complete in one step"),
    }
}

fn params_for(server_uri: &str, label: Option<&str>) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        PARAM_AUTHORIZATION_ENDPOINT.to_string(),
        format!("{server_uri}/authorize"),
    );
    p.insert(
        PARAM_TOKEN_ENDPOINT.to_string(),
        format!("{server_uri}/token"),
    );
    p.insert(PARAM_CLIENT_ID.to_string(), "my-client".to_string());
    p.insert(PARAM_SCOPES.to_string(), "openid profile".to_string());
    if let Some(label) = label {
        p.insert(PARAM_ACCOUNT_LABEL.to_string(), label.to_string());
    }
    p
}

fn query_param(url: &str, key: &str) -> Option<String> {
    url::Url::parse(url)
        .unwrap()
        .query_pairs()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

/// One form field out of an `application/x-www-form-urlencoded` body.
fn form_field(body: &[u8], key: &str) -> Option<String> {
    url::form_urlencoded::parse(body)
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

/// The happy path: begin mints the PKCE authorization URL, complete validates the state, exchanges
/// the code, and returns the token JSON as the blob under the explicit account label. The mock
/// captures the exchange request so the test proves (a) the RFC 6749/7636 form fields, and (b) the
/// PKCE bond: sha256(code_verifier) == the code_challenge the authorization URL advertised.
#[tokio::test]
async fn begin_complete_exchanges_the_code_with_a_bound_verifier() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .and(body_string_contains("grant_type=authorization_code"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-123",
            "token_type": "Bearer",
            "refresh_token": "rt-456",
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let factory = DescriptorFlowFactory::new(generic_oauth2()).expect("build factory");
    let flow = factory
        .begin(
            &params_for(&server.uri(), Some("work")),
            "http://127.0.0.1:7777/cb",
        )
        .await
        .expect("begin");
    let auth_url = redirect_url(flow.as_ref());
    let challenge = query_param(&auth_url, "code_challenge").expect("challenge advertised");
    let state = query_param(&auth_url, "state").expect("state minted");

    let outcome = complete(
        flow.as_ref(),
        &format!("http://127.0.0.1:7777/cb?code=the-code&state={state}"),
    )
    .await
    .expect("complete");

    // The identity: explicit label wins; refs/instance are family-qualified.
    assert_eq!(outcome.account_label, "work");
    assert_eq!(outcome.credential_ref, format!("{FAMILY}/work"));
    assert_eq!(outcome.transport_instance.as_str(), "oauth2/work");
    // The blob is the raw token response (what a consumer re-parses for access/refresh).
    let blob: serde_json::Value = serde_json::from_str(&outcome.credential_blob).unwrap();
    assert_eq!(blob["access_token"], "at-123");
    assert_eq!(blob["refresh_token"], "rt-456");

    // The captured exchange request: all RFC form fields present, redirect echoed, and the PKCE
    // bond holds — the POSTed verifier hashes to the advertised challenge.
    let requests = server.received_requests().await.expect("recording on");
    let exchange = requests
        .iter()
        .find(|r| r.url.path() == "/token")
        .expect("the token endpoint was called");
    let body = &exchange.body;
    assert_eq!(form_field(body, "code").as_deref(), Some("the-code"));
    assert_eq!(
        form_field(body, "redirect_uri").as_deref(),
        Some("http://127.0.0.1:7777/cb")
    );
    assert_eq!(form_field(body, "client_id").as_deref(), Some("my-client"));
    let verifier = form_field(body, "code_verifier").expect("verifier POSTed");
    assert_eq!(
        URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes())),
        challenge,
        "sha256(code_verifier) must equal the advertised code_challenge (the PKCE bond)"
    );
}

/// Identity derivation order, leg 2: with no explicit label, the `sub` claim of the token
/// response's `id_token` names the account.
#[tokio::test]
async fn id_token_sub_names_the_account_when_no_label_is_given() {
    let server = MockServer::start().await;
    let payload = URL_SAFE_NO_PAD.encode(r#"{"sub":"alice@idp"}"#);
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-123",
            "id_token": format!("header.{payload}.sig"),
        })))
        .mount(&server)
        .await;

    let factory = DescriptorFlowFactory::new(generic_oauth2()).expect("build factory");
    let flow = factory
        .begin(&params_for(&server.uri(), None), "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let state = query_param(&redirect_url(flow.as_ref()), "state").unwrap();
    let outcome = complete(
        flow.as_ref(),
        &format!("http://127.0.0.1:7777/cb?code=c&state={state}"),
    )
    .await
    .expect("complete");
    assert_eq!(outcome.account_label, "alice@idp");
    assert_eq!(outcome.credential_ref, "oauth2/alice@idp");
}

/// A failed exchange (IdP 400) is surfaced as an error carrying the endpoint's response — no
/// credential outcome is fabricated.
#[tokio::test]
async fn a_failed_token_exchange_is_surfaced() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(serde_json::json!({ "error": "invalid_grant" })),
        )
        .mount(&server)
        .await;

    let factory = DescriptorFlowFactory::new(generic_oauth2()).expect("build factory");
    let flow = factory
        .begin(&params_for(&server.uri(), None), "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let state = query_param(&redirect_url(flow.as_ref()), "state").unwrap();
    // `AuthOutcome` deliberately has no `Debug` (it carries the secret blob), so match by hand.
    let err = match complete(
        flow.as_ref(),
        &format!("http://127.0.0.1:7777/cb?code=c&state={state}"),
    )
    .await
    {
        Err(e) => e,
        Ok(_) => panic!("a 400 exchange must fail the completion"),
    };
    assert!(err.to_string().contains("invalid_grant"), "{err}");
}

/// The generic (`use_state: true`) family enforces the CSRF `state` echo BEFORE any network I/O: a
/// callback whose `state` does not match the minted one is rejected without touching the endpoint.
#[tokio::test]
async fn state_mismatch_is_rejected_before_any_exchange() {
    let factory = DescriptorFlowFactory::new(generic_oauth2()).expect("build factory");
    let flow = factory
        .begin(
            &params_for("https://idp.example", None),
            "http://127.0.0.1:7777/cb",
        )
        .await
        .expect("begin");
    let err = match complete(
        flow.as_ref(),
        "http://127.0.0.1:7777/cb?code=abc&state=WRONG",
    )
    .await
    {
        Err(e) => e,
        Ok(_) => panic!("a state mismatch must be rejected"),
    };
    assert!(err.to_string().contains("state mismatch"), "{err}");
}

/// The real OpenRouter descriptor's begin URL is PKCE-only against the fixed OpenRouter endpoint,
/// carries the callback under `callback_url`, and advertises the S256 challenge — with no
/// `client_id`/`state`/`response_type`. (The static shape; the exchange path is proven below.)
#[tokio::test]
async fn openrouter_begin_targets_the_fixed_endpoint_pkce_only() {
    let factory = DescriptorFlowFactory::new(openrouter()).expect("build factory");
    assert_eq!(factory.family(), OPENROUTER_FAMILY);
    assert!(
        factory.provider_info().params_schema.is_empty(),
        "the OpenRouter family owns every parameter (empty schema)"
    );
    let flow = factory
        .begin(&BTreeMap::new(), "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let url = redirect_url(flow.as_ref());
    let url = url.as_str();
    assert!(url.starts_with("https://openrouter.ai/auth?"), "{url}");
    assert_eq!(
        query_param(url, "callback_url").as_deref(),
        Some("http://127.0.0.1:7777/cb")
    );
    assert_eq!(
        query_param(url, "code_challenge_method").as_deref(),
        Some("S256")
    );
    assert!(query_param(url, "client_id").is_none());
    assert!(query_param(url, "state").is_none());
}

/// The provider-key JSON key-mint exchange (OpenRouter's shape) over a wiremock endpoint: the flow
/// JSON-POSTs `{code, code_verifier, code_challenge_method}` (the PKCE verifier hashing to the
/// advertised challenge), the mock returns `{"key": ...}`, and the outcome carries the BARE key as
/// the blob with the `ProviderKeyForProfile` slot (`use_state: false` → no CSRF echo required).
/// The token endpoint is injected via a param so the engine's real JSON key-mint path is exercised
/// against the mock (the real descriptor's fixed endpoint is proven above).
#[tokio::test]
async fn provider_key_json_mint_stores_the_bare_key_and_slot() {
    use daemon_oauth::{
        CallbackParam, CredentialShape, ExchangeStyle, OAuthFlowDescriptor, Source,
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/keys"))
        .and(body_string_contains("code_challenge_method"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "key": "sk-or-minted-test-key",
        })))
        .expect(1)
        .mount(&server)
        .await;

    // OpenRouter's descriptor shape, but the token endpoint points at the mock (a param) so the
    // engine's JSON key-mint path runs against wiremock.
    let descriptor = OAuthFlowDescriptor {
        family: "provider/openrouter",
        display_name: "OpenRouter (test)",
        authorization_endpoint: Source::Fixed("https://openrouter.ai/auth"),
        token_endpoint: Source::Param("token_endpoint"),
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
    };
    let factory = DescriptorFlowFactory::new(descriptor).expect("build factory");
    let mut params = BTreeMap::new();
    params.insert(
        "token_endpoint".to_string(),
        format!("{}/keys", server.uri()),
    );

    let flow = factory
        .begin(&params, "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let challenge =
        query_param(&redirect_url(flow.as_ref()), "code_challenge").expect("challenge advertised");

    // No `state` on the callback — the PKCE-only descriptor accepts it.
    let outcome = complete(flow.as_ref(), "http://127.0.0.1:7777/cb?code=or-code")
        .await
        .expect("complete");
    // The BARE minted key is the blob (not the JSON envelope), and it is slotted as a provider key.
    assert_eq!(outcome.credential_blob, "sk-or-minted-test-key");
    assert_eq!(outcome.slot, CredentialSlotKind::ProviderKeyForProfile);
    assert_eq!(outcome.account_label, "openrouter");

    // The JSON exchange body carried the PKCE verifier that hashes to the advertised challenge.
    let requests = server.received_requests().await.expect("recording on");
    let exchange = requests
        .iter()
        .find(|r| r.url.path() == "/keys")
        .expect("the key-mint endpoint was called");
    let body: serde_json::Value = serde_json::from_slice(&exchange.body).expect("JSON body");
    assert_eq!(body["code"], "or-code");
    assert_eq!(body["code_challenge_method"], "S256");
    let verifier = body["code_verifier"].as_str().expect("verifier posted");
    assert_eq!(
        URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes())),
        challenge,
        "sha256(code_verifier) must equal the advertised code_challenge (the PKCE bond)"
    );
}
