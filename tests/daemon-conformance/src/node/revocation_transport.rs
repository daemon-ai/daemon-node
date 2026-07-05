// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE CLUSTER F REVOCATION-EPOCH GATE (Part A — live connection teardown). A mux connection
//! snapshots its principal for its whole lifetime; a store-only `session_revoke` would leave that
//! live connection acting under the revoked identity. These tests prove the epoch teardown:
//!
//! - `revoked_session_tears_down_live_connection`: after `session_revoke`, a further `Call` on the
//!   already-open connection fails (connection closed / `Unauthenticated`).
//! - `revoked_session_ends_live_subscribe_stream`: a live `Subscribe` pump is torn down on revoke.
//! - `revoking_one_principal_leaves_another_live`: revocation is per-principal, not global.

use super::harness::*;
use daemon_api::{AccessControlApi, ApiError, WireS2C};
use daemon_auth::{AuthStore, Role};
use daemon_common::ReqId;
use daemon_host::{serve_api_unix_authenticated, Authenticator, MuxApiClient, SessionRevocations};
use daemon_protocol::{AgentCommand, UserMsg};
use std::time::Duration;

/// An in-memory identity store with one operator and one viewer (SCRAM material derived on create).
fn auth_store() -> Arc<AuthStore> {
    let store = Arc::new(AuthStore::open_in_memory().expect("open auth store"));
    store
        .create_user("operator", "op-pw", &[Role::Operator])
        .expect("create operator");
    store
        .create_user("viewer", "view-pw", &[Role::Viewer])
        .expect("create viewer");
    store
}

/// The stable user id for a username (needed to drive `session_revoke`, which is keyed by user id).
fn user_id(store: &AuthStore, username: &str) -> String {
    store.find_user(username).unwrap().unwrap().id
}

/// A node whose access-control surface is live (auth store) and whose live connections are revocable
/// (the shared registry), assembled exactly as `bins/daemon` layers these on post-assembly.
fn revocable_node(
    store: Arc<AuthStore>,
    revocations: Arc<SessionRevocations>,
) -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
    let (node, handle) = assemble();
    let node = Arc::new(
        (*node)
            .clone()
            .with_auth_store(store)
            .with_revocations(revocations),
    );
    (node, handle)
}

/// After `session_revoke`, the operator's already-open, already-authenticated mux connection can no
/// longer issue a `Call`: the transport tears it down (connection closed) or refuses it
/// (`Unauthenticated`). Pre-enforcement (guard captured but not checked) the second `Call` still
/// succeeds — the RED proof that the live handle outlived revocation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn revoked_session_tears_down_live_connection() {
    let store = auth_store();
    let revocations = SessionRevocations::new();
    let auth = Arc::new(Authenticator::new(store.clone()).with_revocations(revocations.clone()));
    let (node, handle) = revocable_node(store.clone(), revocations.clone());
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    let mut client = MuxApiClient::connect(&path).await.expect("connect + hello");
    client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram authenticates operator");

    // Baseline: the live connection serves a Call the operator holds.
    let pre = client
        .call(ApiRequest::Health)
        .await
        .expect("pre-revoke call");
    assert!(
        !matches!(pre, ApiResponse::Error(_)),
        "the operator must be able to call Health before revocation, got {pre:?}"
    );

    // Revoke the operator through the real access-control handler (store delete + epoch bump).
    let op_id = user_id(&store, "operator");
    as_system(node.session_revoke(op_id))
        .await
        .expect("session_revoke succeeds");

    // The already-open connection must no longer be usable: the next Call errors (connection torn
    // down) or is refused. Wrapped in a timeout so a hang (rather than a clean teardown) is a
    // failure, not an indefinite block.
    let post = tokio::time::timeout(Duration::from_secs(5), client.call(ApiRequest::Health))
        .await
        .expect("the post-revoke call must resolve promptly (no hang)");
    let torn_down = match post {
        Err(_) => true,                                               // connection closed
        Ok(ApiResponse::Error(ApiError::Unauthenticated(_))) => true, // refused
        _ => false,
    };
    assert!(
        torn_down,
        "a revoked live connection must be unusable (closed or Unauthenticated), got {post:?}"
    );

    server.abort();
    handle.shutdown().await;
}

/// A **genuinely live** `Subscribe` pump on a revoked connection is torn down promptly rather than
/// hanging: after confirming the stream is delivering (a `Submit` gives the session a merged log,
/// then an `Item` frame arrives), a `session_revoke` must make a subsequent frame read resolve
/// (`End`/EOF) within the timeout. This is the flake-critical no-hang guard for the pump-teardown
/// path (the highest-risk edit on this track); pre-enforcement the live pump keeps running and the
/// read blocks until the 20s keepalive, so this also discriminates RED from GREEN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn revoked_session_tears_down_live_subscribe_stream() {
    let store = auth_store();
    let revocations = SessionRevocations::new();
    let auth = Arc::new(Authenticator::new(store.clone()).with_revocations(revocations.clone()));
    let (node, handle) = revocable_node(store.clone(), revocations.clone());
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    let mut client = MuxApiClient::connect(&path).await.expect("connect + hello");
    client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram authenticates operator");

    // Give the session a live merged log so the subscribe pump stays open (an operator owns the
    // session it submits to; SessionControlAny also permits the subscribe).
    let sid = SessionId::new("revocation-stream");
    match client
        .call(ApiRequest::Submit {
            session: sid.clone(),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        })
        .await
        .expect("submit")
    {
        ApiResponse::Ok | ApiResponse::Routed { .. } => {}
        other => panic!("expected Ok/Routed from submit, got {other:?}"),
    }

    // Open the push subscription and confirm it is genuinely live (at least one Item frame), so the
    // pump is running at revoke time (not already ended).
    let id = client
        .open(ApiRequest::Subscribe {
            session: sid,
            after_seq: 0,
            max: 0,
        })
        .await
        .expect("open subscribe stream");
    let mut live = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match client.next().await.expect("stream frame") {
            WireS2C::Item { id: rid, .. } if rid == id => {
                live = true;
                break;
            }
            WireS2C::End { id: rid, .. } if rid == id => break,
            _ => continue,
        }
    }
    assert!(
        live,
        "the subscribe pump must deliver at least one Item before we revoke"
    );

    // Revoke; the live pump must be torn down — a subsequent frame read resolves within the timeout.
    let op_id = user_id(&store, "operator");
    as_system(node.session_revoke(op_id))
        .await
        .expect("session_revoke succeeds");

    let ended = tokio::time::timeout(Duration::from_secs(5), client.next()).await;
    assert!(
        ended.is_ok(),
        "the live subscribe stream must be torn down on revoke (End/EOF), not hang"
    );

    server.abort();
    handle.shutdown().await;
}

/// Revocation is per-principal: revoking the operator must not tear down the viewer's independent
/// live connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn revoking_one_principal_leaves_another_live() {
    let store = auth_store();
    let revocations = SessionRevocations::new();
    let auth = Arc::new(Authenticator::new(store.clone()).with_revocations(revocations.clone()));
    let (node, handle) = revocable_node(store.clone(), revocations.clone());
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    let mut op_client = MuxApiClient::connect(&path)
        .await
        .expect("operator connect");
    op_client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("operator scram");
    let mut viewer_client = MuxApiClient::connect(&path).await.expect("viewer connect");
    viewer_client
        .authenticate_scram("viewer", "view-pw")
        .await
        .expect("viewer scram");

    // Revoke the operator only.
    let op_id = user_id(&store, "operator");
    as_system(node.session_revoke(op_id))
        .await
        .expect("session_revoke operator");

    // The viewer's connection stays usable (Health -> ControlRead, which a Viewer holds).
    let viewer_call = tokio::time::timeout(
        Duration::from_secs(5),
        viewer_client.call(ApiRequest::Health),
    )
    .await
    .expect("viewer call resolves");
    assert!(
        matches!(viewer_call, Ok(ApiResponse::Health(_))),
        "an unrelated principal's live connection must survive, got {viewer_call:?}"
    );

    server.abort();
    handle.shutdown().await;
}
