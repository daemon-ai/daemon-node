// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE AUTH 3 CONVERGENCE GATE (deliverable 3): authentication + the authz-core capability gate
//! wired into the live socket transport. Proves the fail-closed handshake → auth → context →
//! authorize → dispatch path end-to-end over the Unix socket:
//!
//! - (a)/(c) Without local trust the Unix socket **requires SCRAM**: a pre-auth `Call` returns
//!   `Unauthenticated` and the connection stays unelevated; after a SCRAM `AuthOk` the same `Call`
//!   succeeds.
//! - (b) Under `[api].local_trust` the Unix socket binds [`RequestContext::system`]: an admin-tier
//!   `command_invoke` passes the access gate (the local admin path is restored, not regressed to
//!   deny by the fail-closed inversion).
//! - (d) A `Viewer` principal authenticated over SCRAM is `Forbidden` on a write op by the live
//!   capability gate, while a read op it holds is allowed.

use super::harness::*;
use daemon_api::{ApiError, CommandInvocation};
use daemon_auth::{AuthStore, Role};
use daemon_host::{serve_api_unix_authenticated, Authenticator, CommandRegistry, MuxApiClient};
use daemon_protocol::{AgentCommand, UserMsg};

/// An in-memory identity store with one operator and one viewer (both with SCRAM material derived
/// from their passwords on creation).
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

/// (a)+(c) The auth-required Unix socket: pre-auth denied + unelevated, then SCRAM unlocks dispatch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unix_without_local_trust_requires_scram_then_serves() {
    let (node, handle) = assemble();
    let auth = Arc::new(Authenticator::new(auth_store()));
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    let mut client = MuxApiClient::connect(&path).await.expect("connect + hello");

    // Pre-auth: a Call is refused with Unauthenticated.
    let pre = client
        .call(ApiRequest::Health)
        .await
        .expect("pre-auth call");
    assert!(
        matches!(pre, ApiResponse::Error(ApiError::Unauthenticated(_))),
        "a pre-auth Call must be Unauthenticated, got {pre:?}"
    );
    // The connection STAYS unelevated: a second pre-auth Call is still denied.
    let pre2 = client
        .call(ApiRequest::Health)
        .await
        .expect("pre-auth call 2");
    assert!(
        matches!(pre2, ApiResponse::Error(ApiError::Unauthenticated(_))),
        "the connection must stay unelevated after a denied Call, got {pre2:?}"
    );

    // Authenticate via SCRAM-SHA-256 as the operator.
    let view = client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram authenticates");
    assert_eq!(view.username, "operator");

    // The same Call now succeeds (operator holds ControlRead for Health).
    let post = client
        .call(ApiRequest::Health)
        .await
        .expect("post-auth call");
    assert!(
        !matches!(post, ApiResponse::Error(_)),
        "the same Call must succeed after AuthOk, got {post:?}"
    );

    server.abort();
    handle.shutdown().await;
}

/// (b) Under local trust the Unix socket binds `system()`; an admin-tier command's access gate
/// passes (the only remaining failure is the missing session, NOT an access denial).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unix_local_trust_binds_system_and_admin_command_is_allowed() {
    let (node, handle) = assemble();
    // Bind the built-in command catalog (the host does this post-assembly) so `command_invoke` is
    // live; `/approve` is an admin-tier command.
    node.set_commands(Arc::new(CommandRegistry::with_builtins()));
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    // `serve_api_unix` is the local-trust entry point (binds `RequestContext::system`).
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let mut client = MuxApiClient::connect(&path).await.expect("connect + hello");
    let res = client
        .call(ApiRequest::CommandInvoke {
            invocation: CommandInvocation {
                name: "approve".into(),
                args: String::new(),
                session: None,
                origin: None,
            },
        })
        .await
        .expect("command_invoke call");
    match res {
        // Access granted under system(): the command proceeded to its session check.
        ApiResponse::Error(ApiError::Other(msg)) => {
            assert!(
                msg.contains("active session"),
                "admin access must be granted under local trust (expected the missing-session \
                 error, not an access denial), got: {msg}"
            );
            assert!(
                !msg.contains("admin"),
                "must not be the access-denied error under local trust, got: {msg}"
            );
        }
        other => panic!("expected the missing-session error (access granted), got {other:?}"),
    }

    server.abort();
    handle.shutdown().await;
}

/// (d) A Viewer is Forbidden on a write op by the live gate; an own-read op it holds is allowed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn viewer_is_forbidden_on_write_but_allowed_on_read() {
    let (node, handle) = assemble();
    let auth = Arc::new(Authenticator::new(auth_store()));
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    let mut client = MuxApiClient::connect(&path).await.expect("connect + hello");
    client
        .authenticate_scram("viewer", "view-pw")
        .await
        .expect("scram authenticates viewer");

    // A write op: Submit maps to SessionWrite, which a Viewer does not hold -> Forbidden by the gate.
    let write = client
        .call(ApiRequest::Submit {
            session: SessionId::new("viewer-session"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
            origin: None,
            profile: None,
        })
        .await
        .expect("write call");
    assert!(
        matches!(write, ApiResponse::Error(ApiError::Forbidden(_))),
        "a Viewer write must be Forbidden by the live gate, got {write:?}"
    );

    // A read op the Viewer holds (Sessions -> SessionRead): not Forbidden.
    let read = client.call(ApiRequest::Sessions).await.expect("read call");
    assert!(
        !matches!(read, ApiResponse::Error(ApiError::Forbidden(_))),
        "a Viewer read (Sessions) must be allowed, got {read:?}"
    );

    server.abort();
    handle.shutdown().await;
}
