// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Vertical tests for the OAuth2 PKCE factory over a wiremock token endpoint: the full
//! begin → (browser hop) → complete exchange, including the PKCE consistency proof — the
//! `code_verifier` POSTed to the token endpoint hashes (S256) to exactly the `code_challenge`
//! the authorization URL advertised — plus the identity-derivation order and the failure path.
//! (The generic node-side orchestration — parking, credential write, profile bind — is proven by
//! the conformance suite's stub-factory test; these tests own the real family's protocol slice.)

use std::collections::BTreeMap;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use sha2::{Digest, Sha256};
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use daemon_host::AuthFlowFactory;
use daemon_oauth::{
    OAuth2PkceFlowFactory, FAMILY, PARAM_ACCOUNT_LABEL, PARAM_AUTHORIZATION_ENDPOINT,
    PARAM_CLIENT_ID, PARAM_SCOPES, PARAM_TOKEN_ENDPOINT,
};

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

    let factory = OAuth2PkceFlowFactory::new().expect("build factory");
    let flow = factory
        .begin(
            &params_for(&server.uri(), Some("work")),
            "http://127.0.0.1:7777/cb",
        )
        .await
        .expect("begin");
    let auth_url = flow.authorization_url().to_string();
    let challenge = query_param(&auth_url, "code_challenge").expect("challenge advertised");
    let state = query_param(&auth_url, "state").expect("state minted");

    let outcome = flow
        .complete(&format!(
            "http://127.0.0.1:7777/cb?code=the-code&state={state}"
        ))
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

    let factory = OAuth2PkceFlowFactory::new().expect("build factory");
    let flow = factory
        .begin(&params_for(&server.uri(), None), "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let state = query_param(flow.authorization_url(), "state").unwrap();
    let outcome = flow
        .complete(&format!("http://127.0.0.1:7777/cb?code=c&state={state}"))
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

    let factory = OAuth2PkceFlowFactory::new().expect("build factory");
    let flow = factory
        .begin(&params_for(&server.uri(), None), "http://127.0.0.1:7777/cb")
        .await
        .expect("begin");
    let state = query_param(flow.authorization_url(), "state").unwrap();
    // `AuthOutcome` deliberately has no `Debug` (it carries the secret blob), so match by hand.
    let err = match flow
        .complete(&format!("http://127.0.0.1:7777/cb?code=c&state={state}"))
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("a 400 exchange must fail the completion"),
    };
    assert!(err.to_string().contains("invalid_grant"), "{err}");
}
