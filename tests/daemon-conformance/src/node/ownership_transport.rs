// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Auth 4 ownership over the authenticated mux transport (the cross-owner transcript-read gap this
//! track closes). Two authenticated peers (`alice`, `bob`, both `User`) connect over the
//! SASL-required Unix socket; `alice` opens a session (stamping herself owner), and `bob` must be
//! denied on BOTH forms of the `Subscribe` op:
//!
//! * the one-shot `Call` form → `SessionApi::log_after` (previously ungated), and
//! * the streaming `Open` form → the `pump_session_log` mux pump (previously ran with no bound
//!   principal, so it streamed any owner's live transcript to any authenticated peer).
//!
//! These drive the REAL socket path (unlike `ownership.rs`, which exercises the in-process trait
//! calls), because the pump only exists on the streaming transport.

use super::harness::*;
use daemon_api::{ApiError, WireS2C};
use daemon_auth::{AuthStore, Role};
use daemon_common::ReqId;
use daemon_host::{serve_api_unix_authenticated, Authenticator, MuxApiClient};
use daemon_protocol::{AgentCommand, UserMsg};

/// An in-memory identity store with two ordinary users (SCRAM material derived on creation).
fn two_user_store() -> Arc<AuthStore> {
    let store = Arc::new(AuthStore::open_in_memory().expect("open auth store"));
    store
        .create_user("alice", "alice-pw", &[Role::User])
        .expect("create alice");
    store
        .create_user("bob", "bob-pw", &[Role::User])
        .expect("create bob");
    store
}

fn start_turn(text: &str) -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new(text),
        request_id: ReqId(1),
    }
}

/// Serve the auth-required Unix socket over a freshly assembled node; returns the node, the server
/// task, the started resident-service handle, and the socket path.
async fn serve_authed() -> (
    Arc<NodeApiImpl>,
    tokio::task::JoinHandle<()>,
    daemon_host::SupervisorHandle,
    std::path::PathBuf,
) {
    let (node, handle) = assemble();
    let auth = Arc::new(Authenticator::new(two_user_store()));
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));
    (node, server, handle, path)
}

/// `alice` opens a session over the socket and drives it until her own `log_after` (the one-shot
/// `Subscribe` `Call`) returns a non-empty page — so ownership is stamped AND the live log has
/// entries a cross-owner read could leak.
async fn alice_owns_populated_session(path: &std::path::Path, session: &SessionId) {
    let mut alice = MuxApiClient::connect(path)
        .await
        .expect("alice connect + hello");
    alice
        .authenticate_scram("alice", "alice-pw")
        .await
        .expect("alice scram");
    match alice
        .call(ApiRequest::Submit {
            session: session.clone(),
            command: start_turn("owned transcript"),
            origin: None,
            profile: None,
        })
        .await
        .expect("alice submit")
    {
        ApiResponse::Ok | ApiResponse::Routed { .. } => {}
        other => panic!("expected Ok/Routed from alice submit, got {other:?}"),
    }
    // Wait until the owner can read a non-empty page (the log has entries to leak).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match alice
            .call(ApiRequest::Subscribe {
                session: session.clone(),
                after_seq: 0,
                max: 64,
            })
            .await
            .expect("alice log_after")
        {
            ApiResponse::LogPage(page) if !page.entries.is_empty() => break,
            ApiResponse::LogPage(_) => {}
            other => panic!("owner's own log_after must return a page, got {other:?}"),
        }
        assert!(
            Instant::now() < deadline,
            "alice's session never produced log entries"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// The one-shot `Subscribe` `Call` (→ `log_after`): a non-owner peer is `Forbidden`, while the
/// owner reads her own page. Before the fix `log_after` had NO ownership check, so `bob` read
/// `alice`'s transcript.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_owner_one_shot_log_after_is_denied() {
    let (_node, server, handle, path) = serve_authed().await;
    let session = SessionId::new("owned-oneshot");
    alice_owns_populated_session(&path, &session).await;

    let mut bob = MuxApiClient::connect(path.clone())
        .await
        .expect("bob connect + hello");
    bob.authenticate_scram("bob", "bob-pw")
        .await
        .expect("bob scram");
    let bob_read = bob
        .call(ApiRequest::Subscribe {
            session: session.clone(),
            after_seq: 0,
            max: 64,
        })
        .await
        .expect("bob log_after call");
    assert!(
        matches!(bob_read, ApiResponse::Error(ApiError::Forbidden(_))),
        "a non-owner one-shot log_after must be Forbidden, got {bob_read:?}"
    );

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
}

/// The streaming `Subscribe` `Open` (→ `pump_session_log`): a non-owner peer's stream ends with
/// `End { error: Forbidden }` and NEVER delivers an owner's log entry. Before the fix the pump ran
/// with no bound principal, so it streamed `alice`'s live transcript to `bob` (the live vuln).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_owner_mux_stream_pump_is_denied() {
    let (_node, server, handle, path) = serve_authed().await;
    let session = SessionId::new("owned-stream");
    alice_owns_populated_session(&path, &session).await;

    let mut bob = MuxApiClient::connect(path.clone())
        .await
        .expect("bob connect + hello");
    bob.authenticate_scram("bob", "bob-pw")
        .await
        .expect("bob scram");
    let id = bob
        .open(ApiRequest::Subscribe {
            session: session.clone(),
            after_seq: 0,
            max: 64,
        })
        .await
        .expect("bob open subscribe");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(
            Instant::now() < deadline,
            "bob's cross-owner stream neither delivered End{{Forbidden}} nor an entry within the deadline"
        );
        match bob.next().await.expect("bob stream frame") {
            WireS2C::End { id: rid, error } => {
                assert_eq!(rid, id, "End must carry the stream id");
                assert!(
                    matches!(error, Some(ApiError::Forbidden(_))),
                    "a non-owner stream must End with Forbidden, got {error:?}"
                );
                break;
            }
            // The vuln: the pump streamed the owner's transcript to a non-owner. An empty keepalive
            // page is harmless; keep waiting for End.
            WireS2C::Item {
                res: ApiResponse::LogPage(page),
                ..
            } => {
                assert!(
                    page.entries.is_empty(),
                    "SECURITY: the mux pump leaked {} owner log entries to a non-owner",
                    page.entries.len()
                );
            }
            _ => {}
        }
    }

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
}
