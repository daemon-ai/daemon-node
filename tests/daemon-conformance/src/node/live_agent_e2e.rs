// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE TRACK D LIVE-AGENT E2E GATE: the whole "login and chat" chain proven end to end against a
//! mock upstream. A fresh node with a first-admin seeded by the Track A bootstrap authenticates over
//! SCRAM, provisions a daemon-api bearer, creates + selects a `DaemonApi` profile, and drives a real
//! session turn — which routes through the **real** `genai` OpenAI adapter (Track B's
//! `ProviderSelector::DaemonApi`) to a wiremock upstream and returns an assistant message. The turn
//! is asserted to have hit the OpenAI-compatible `/api/v1/chat/completions` path carrying the
//! provisioned `Authorization: Bearer` (credential → chat), and to have completed (`TurnFinished`).
//!
//! It complements (does not duplicate) the coverage A/B already landed:
//! - Track A's over-the-wire seeded-admin SCRAM + audit is `positive_e2e.rs`; this file reuses A's
//!   bootstrap + node re-wrap shape but does not re-assert the audit chain.
//! - Track B's adapter-path unit proof is `daemon-providers`' `daemon_api_openai_adapter_hits_api_v1_
//!   chat_completions`; this file re-uses the wiremock idiom but asserts the path **through a real
//!   session Submit**, not a direct provider call.
//! - The fail-closed guarantees are `negative_auth.rs`; this file adds only the one assertion those
//!   suites do not make — that the live turn/provider surface is itself behind the gate (a pre-auth
//!   caller cannot drive a turn, and the mock upstream then receives nothing).

use super::harness::*;
use super::wire_client::MuxConn;

use daemon_api::{ApiError, Outbound, ProfileSpec, ProviderSelector};
use daemon_auth::{AdminSeed, AuthStore};
use daemon_common::ReqId;
use daemon_host::{serve_api_unix_authenticated, AuthAudit, Authenticator};
use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};
use daemon_telemetry::TraceSigner;
use tokio::net::UnixStream;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Open + `Hello`-handshake a fresh mux connection to the authenticated socket.
async fn connect(path: &std::path::Path) -> MuxConn<UnixStream> {
    let stream = UnixStream::connect(path).await.expect("connect socket");
    MuxConn::handshake(stream).await.expect("hello handshake")
}

/// Mount the daemon-api gateway's OpenAI-compatible chat path returning `content` as the assistant
/// message over **SSE** (the engine drives interactive turns through the provider's streaming path,
/// so the upstream must speak `text/event-stream`, mirroring `daemon-providers`' streaming wire
/// test). Matches ONLY `POST /api/v1/chat/completions`, so a hit proves the `v1` segment is neither
/// dropped nor doubled and that the OpenAI wire (not Anthropic-native `/v1/messages`) was used.
async fn mount_gateway(server: &MockServer, content: &str) {
    // A JSON-escaped, quoted content literal for the SSE delta payload.
    let content_json = serde_json::Value::String(content.to_string());
    let sse = format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{content_json}}}}}]}}\n\n\
         data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\
         \"usage\":{{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}}}\n\n\
         data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(server)
        .await;
}

/// The `DaemonApi` profile a GUI would create: OpenRouter-style id, base pinned at the mock gateway.
fn gateway_profile(base: &str) -> ProfileSpec {
    ProfileSpec {
        base_url: Some(base.to_string()),
        ..ProfileSpec::new(
            "gateway",
            ProviderSelector::DaemonApi,
            "anthropic/claude-sonnet-4-5",
        )
    }
}

/// Scan a drained batch for the terminal `TurnFinished`, returning its final assistant text.
fn find_finish(items: Vec<Outbound>) -> Option<String> {
    for item in items {
        if let Outbound::Event(AgentEvent::TurnFinished { summary, .. }) = item {
            return Some(summary.final_text.unwrap_or_default());
        }
    }
    None
}

/// Poll the authenticated connection until the turn finishes; return its final text.
async fn drive_to_finish_mux(conn: &mut MuxConn<UnixStream>, session: &SessionId) -> String {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        let items = match conn
            .call(ApiRequest::Poll {
                session: session.clone(),
                max: 0,
            })
            .await
            .expect("poll")
        {
            ApiResponse::Drained(items) => items,
            other => panic!("expected Drained, got {other:?}"),
        };
        if let Some(text) = find_finish(items) {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the turn never reached TurnFinished");
}

/// Poll the local (unauthenticated) client until the turn finishes; return its final text.
async fn drive_to_finish_client(client: &ApiClient, session: &SessionId) -> String {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        let items = match client
            .call(ApiRequest::Poll {
                session: session.clone(),
                max: 0,
            })
            .await
            .expect("poll")
        {
            ApiResponse::Drained(items) => items,
            other => panic!("expected Drained, got {other:?}"),
        };
        if let Some(text) = find_finish(items) {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the local-trust turn never reached TurnFinished");
}

/// THE POSITIVE HAPPY PATH: seeded admin → SCRAM login → `CredentialSet` → `DaemonApi` profile →
/// `Submit` → the turn routes through the real OpenAI adapter to the mock upstream and returns an
/// assistant message; the request hit `/api/v1/chat/completions` carrying the provisioned bearer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_daemon_api_turn_over_authenticated_wire() {
    let server = MockServer::start().await;
    mount_gateway(&server, "routed through the daemon-api gateway").await;
    let base = format!("{}/api/v1/", server.uri());

    // A node whose per-session provider is the real OpenAI adapter pinned at the mock gateway, with a
    // working credential broker + profile store (the run_as_host wiring).
    let (node, _cred_store, handle) = assemble_daemon_api_gateway(base.clone());

    // Seed the first admin via Track A's bootstrap (empty store), then bind the identity store +
    // shared audit onto the node exactly as run_as_host does (deref-clone-rewrap).
    let admin_pw = "correct horse battery staple";
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .seed_first_admin_if_empty(AdminSeed::Explicit {
            username: "root".into(),
            password: admin_pw.into(),
        })
        .expect("seed")
        .expect("seeded on empty store");
    let audit_store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let signer = Arc::new(TraceSigner::generate());
    let audit = AuthAudit::shared(audit_store, signer);
    let node = Arc::new(
        (*node)
            .clone()
            .with_auth_store(store.clone())
            .with_auth_audit(audit.clone()),
    );
    let auth = Arc::new(Authenticator::new(store.clone()).with_audit(audit));

    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server_task = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    // SCRAM login as the seeded admin.
    let mut admin = connect(&path).await;
    let (view, _token) = admin
        .authenticate_scram("root", admin_pw)
        .await
        .expect("admin scram");
    assert_eq!(view.username, "root");

    // Provision the daemon-api bearer for the profile, and confirm it lists redacted (never leaked).
    assert!(matches!(
        admin
            .call(ApiRequest::CredentialSet {
                profile: "gateway".into(),
                secret: "sk-daemon-live-key".into(),
            })
            .await
            .expect("credential set"),
        ApiResponse::Ok
    ));
    match admin
        .call(ApiRequest::CredentialList)
        .await
        .expect("credential list")
    {
        ApiResponse::Credentials(creds) => {
            let g = creds
                .iter()
                .find(|c| c.profile == "gateway")
                .expect("gateway credential");
            assert!(g.present, "the set credential reports present");
            assert!(
                !g.hint.contains("sk-daemon-live-key"),
                "the secret is never returned in a listing"
            );
        }
        other => panic!("expected Credentials, got {other:?}"),
    }

    // Create + select the DaemonApi profile (the shape a GUI creates/edits).
    assert!(matches!(
        admin
            .call(ApiRequest::ProfileCreate {
                spec: gateway_profile(&base),
            })
            .await
            .expect("profile create"),
        ApiResponse::Ok
    ));
    assert!(matches!(
        admin
            .call(ApiRequest::ProfileSelect {
                id: "gateway".into(),
            })
            .await
            .expect("profile select"),
        ApiResponse::Ok
    ));
    match admin
        .call(ApiRequest::ProfileList)
        .await
        .expect("profile list")
    {
        ApiResponse::Profiles(list) => {
            let g = list
                .iter()
                .find(|p| p.id == "gateway")
                .expect("gateway profile");
            assert!(g.is_active, "gateway is the active default");
            assert_eq!(g.provider, ProviderSelector::DaemonApi);
        }
        other => panic!("expected Profiles, got {other:?}"),
    }

    // Drive a real turn: it resolves the DaemonApi provider and routes to the mock upstream.
    let session = SessionId::new("live-1");
    assert!(matches!(
        admin
            .call(ApiRequest::Submit {
                session: session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello gateway"),
                    request_id: ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .expect("submit"),
        ApiResponse::Ok
    ));
    let final_text = drive_to_finish_mux(&mut admin, &session).await;
    assert!(
        final_text.contains("routed through the daemon-api gateway"),
        "the assistant reply must come from the mock upstream via the real adapter, got {final_text:?}"
    );

    // The chain hit the OpenAI-compatible path AND carried the provisioned bearer (credential→chat).
    let received = server.received_requests().await.expect("captured requests");
    assert_eq!(received.len(), 1, "exactly one upstream chat call");
    assert_eq!(
        received[0].url.path(),
        "/api/v1/chat/completions",
        "the DaemonApi turn must post to the OpenAI-compatible path"
    );
    let auth_header = received[0]
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .expect("authorization header present");
    assert_eq!(
        auth_header, "Bearer sk-daemon-live-key",
        "the CredentialSet key must flow to the upstream as the provider bearer"
    );

    server_task.abort();
    handle.shutdown().await;
}

/// THE LOCAL-TRUST PATH: the default `local_trust=system` Unix socket still drives a full turn
/// **without any login** (no Auth regression on the local path) — here through the same real
/// DaemonApi adapter to the mock upstream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_trust_drives_a_daemon_api_turn_without_login() {
    let server = MockServer::start().await;
    mount_gateway(&server, "local-trust gateway reply").await;
    let base = format!("{}/api/v1/", server.uri());

    let (node, cred_store, handle) = assemble_daemon_api_gateway(base.clone());
    // Seed the profile's bearer directly on the shared store (the local path needs no login).
    cred_store
        .set("gateway", "sk-daemon-local-key")
        .expect("seed credential");

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind socket");
    // `serve_api_unix` is the local-trust entry point (binds the explicit system principal).
    let server_task = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // No authentication: create + select the profile and chat straight over the local socket.
    assert!(matches!(
        client
            .call(ApiRequest::ProfileCreate {
                spec: gateway_profile(&base),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::ProfileSelect {
                id: "gateway".into(),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let session = SessionId::new("local-1");
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello local"),
                    request_id: ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    let final_text = drive_to_finish_client(&client, &session).await;
    assert!(
        final_text.contains("local-trust gateway reply"),
        "the local-trust turn must complete through the gateway, got {final_text:?}"
    );
    let received = server.received_requests().await.expect("captured requests");
    assert!(
        received
            .iter()
            .any(|r| r.url.path() == "/api/v1/chat/completions"),
        "the local-trust turn hit the OpenAI-compatible path"
    );

    server_task.abort();
    handle.shutdown().await;
}

/// FAIL-CLOSED ON THE LIVE SURFACE: over the authenticated transport a pre-auth caller cannot set a
/// credential, create a profile, or submit a turn (all `Unauthenticated`), and the mock upstream
/// therefore receives NOTHING — the new provider/turn path is itself behind the Auth gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthenticated_caller_cannot_drive_a_daemon_api_turn() {
    let server = MockServer::start().await;
    mount_gateway(&server, "should never be reached").await;
    let base = format!("{}/api/v1/", server.uri());

    let (node, _cred_store, handle) = assemble_daemon_api_gateway(base.clone());
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .seed_first_admin_if_empty(AdminSeed::Explicit {
            username: "root".into(),
            password: "rootpw".into(),
        })
        .expect("seed")
        .expect("seeded on empty store");
    let audit_store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let audit = AuthAudit::shared(audit_store, Arc::new(TraceSigner::generate()));
    let node = Arc::new(
        (*node)
            .clone()
            .with_auth_store(store.clone())
            .with_auth_audit(audit.clone()),
    );
    let auth = Arc::new(Authenticator::new(store.clone()).with_audit(audit));

    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server_task = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    // A pre-auth connection: every mutating call over the authenticated transport is denied.
    let mut c = connect(&path).await;
    for req in [
        ApiRequest::CredentialSet {
            profile: "gateway".into(),
            secret: "sk".into(),
        },
        ApiRequest::ProfileCreate {
            spec: gateway_profile(&base),
        },
        ApiRequest::Submit {
            session: SessionId::new("nope"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        },
    ] {
        assert!(
            matches!(
                c.call(req).await.expect("pre-auth call"),
                ApiResponse::Error(ApiError::Unauthenticated(_))
            ),
            "a pre-auth call over the authenticated transport must be Unauthenticated"
        );
    }

    // Fail-closed means the turn never ran: the mock upstream saw zero requests.
    let received = server.received_requests().await.expect("captured requests");
    assert!(
        received.is_empty(),
        "no upstream call may happen for an unauthenticated caller, got {} request(s)",
        received.len()
    );

    server_task.abort();
    handle.shutdown().await;
}
