// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Vertical tests for the RFC 8628 device-code engine over wiremock endpoints: begin mints the
//! vendor's `user_code` and presents the "visit URL, enter code" Message; each Poll runs one
//! token exchange classified per §3.5 (`authorization_pending` re-challenges, `slow_down` grows
//! the self-pacing interval so an eager client never hammers the endpoint, a terminal error
//! fails, `access_token` completes with the BARE token slotted as a provider key). (Node-side
//! orchestration — parking, credential write, bind — is the conformance suite's; these own the
//! protocol slice.)

use std::collections::BTreeMap;

use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use daemon_api::{AuthChallenge, AuthFlowKind, AuthStepInput};
use daemon_host::{AuthFlowFactory, AuthStepOutcome, CredentialSlotKind, PendingAuthFlow};
use daemon_oauth::{DeviceFlowDescriptor, DeviceFlowFactory, GITHUB_COPILOT_FAMILY};

/// A descriptor bound to the mock server. `interval: 0` in the mock's device-code response makes
/// every poll immediately due (the pacing tests override it). Endpoints leak: the descriptor
/// carries `&'static str` (process-lifetime constants in production), and a test process is fine
/// to leak two short strings per case.
fn descriptor(server_uri: &str) -> DeviceFlowDescriptor {
    DeviceFlowDescriptor {
        family: GITHUB_COPILOT_FAMILY,
        display_name: "GitHub Copilot",
        device_code_endpoint: Box::leak(format!("{server_uri}/device/code").into_boxed_str()),
        token_endpoint: Box::leak(format!("{server_uri}/token").into_boxed_str()),
        client_id: "test-client",
        scopes: Some("read:user"),
        account_label: "github_copilot",
    }
}

/// Mount the §3.1 device-authorization endpoint minting fixed codes with poll interval `interval`.
async fn mount_device_code(server: &MockServer, interval: u64) {
    Mock::given(method("POST"))
        .and(path("/device/code"))
        .and(body_string_contains("client_id=test-client"))
        .and(body_string_contains("scope=read%3Auser"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "device_code": "dev-123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.example/login/device",
            "expires_in": 900,
            "interval": interval,
        })))
        .mount(server)
        .await;
}

async fn begin(server: &MockServer, interval: u64) -> Box<dyn PendingAuthFlow> {
    mount_device_code(server, interval).await;
    let factory = DeviceFlowFactory::new(descriptor(&server.uri())).unwrap();
    factory
        .begin(&BTreeMap::new(), "http://unused.invalid/cb")
        .await
        .unwrap()
}

fn message_text(challenge: &AuthChallenge) -> &str {
    match challenge {
        AuthChallenge::Message { text } => text,
        other => panic!("expected a Message challenge, got {other:?}"),
    }
}

#[tokio::test]
async fn begin_presents_visit_url_and_user_code() {
    let server = MockServer::start().await;
    let flow = begin(&server, 0).await;
    let challenge = flow.initial_challenge();
    let text = message_text(&challenge);
    assert!(
        text.contains("https://github.example/login/device"),
        "{text}"
    );
    assert!(text.contains("ABCD-1234"), "{text}");

    let factory = DeviceFlowFactory::new(descriptor(&server.uri())).unwrap();
    let info = factory.provider_info();
    assert_eq!(info.family, GITHUB_COPILOT_FAMILY);
    assert_eq!(info.flow_kind, AuthFlowKind::DeviceCode);
    assert!(info.params_schema.is_empty(), "node owns every parameter");
}

#[tokio::test]
async fn pending_re_challenges_then_success_mints_a_provider_key() {
    let server = MockServer::start().await;
    let flow = begin(&server, 0).await;

    // First poll: the user has not approved yet — §3.5 authorization_pending re-challenges.
    Mock::given(method("POST"))
        .and(path("/token"))
        .and(body_string_contains("device_code=dev-123"))
        .and(body_string_contains(
            "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "error": "authorization_pending"
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    match flow.step(AuthStepInput::Poll).await.unwrap() {
        AuthStepOutcome::Challenge(c) => assert!(message_text(&c).contains("ABCD-1234")),
        AuthStepOutcome::Completed(_) => panic!("pending poll must re-challenge"),
    }

    // Second poll: approved — the BARE minted token is the credential, provider-key slotted.
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "gho_minted",
            "token_type": "bearer",
        })))
        .mount(&server)
        .await;
    match flow.step(AuthStepInput::Poll).await.unwrap() {
        AuthStepOutcome::Completed(outcome) => {
            assert_eq!(
                outcome.credential_blob, "gho_minted",
                "bare token, not JSON"
            );
            assert_eq!(outcome.account_label, "github_copilot");
            assert!(matches!(
                outcome.slot,
                CredentialSlotKind::ProviderKeyForProfile
            ));
        }
        AuthStepOutcome::Challenge(_) => panic!("approved poll must complete"),
    }
}

#[tokio::test]
async fn slow_down_throttles_subsequent_polls_off_the_network() {
    let server = MockServer::start().await;
    let flow = begin(&server, 0).await;

    // Exactly ONE token call is allowed: the slow_down response grows the pacing interval (+5s),
    // so the immediate re-poll below must be answered locally without touching the endpoint.
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "error": "slow_down" })),
        )
        .expect(1)
        .mount(&server)
        .await;

    for _ in 0..3 {
        match flow.step(AuthStepInput::Poll).await.unwrap() {
            AuthStepOutcome::Challenge(c) => assert!(message_text(&c).contains("ABCD-1234")),
            AuthStepOutcome::Completed(_) => panic!("slow_down must re-challenge"),
        }
    }
    server.verify().await;
}

#[tokio::test]
async fn terminal_errors_fail_the_flow() {
    let server = MockServer::start().await;
    let flow = begin(&server, 0).await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "error": "access_denied",
            "error_description": "the user declined"
        })))
        .mount(&server)
        .await;
    let Err(err) = flow.step(AuthStepInput::Poll).await else {
        panic!("a terminal vendor error must fail the flow");
    };
    let msg = format!("{err:?}");
    assert!(msg.contains("access_denied"), "{msg}");

    // Non-Poll input is a protocol misuse, refused without touching the network.
    let Err(err) = flow
        .step(AuthStepInput::Callback("http://x/cb?code=1".into()))
        .await
    else {
        panic!("non-Poll input must be refused");
    };
    assert!(format!("{err:?}").contains("Poll"), "{err:?}");
}
