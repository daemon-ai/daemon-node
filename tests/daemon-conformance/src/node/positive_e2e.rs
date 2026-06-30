// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE AUTH 7 POSITIVE E2E GATE: one cohesive happy-path that drives the fully-integrated node over
//! the real authenticated transport (the shared `serve_mux` path, here the Unix socket with
//! `local_trust` disabled), exercising handshake -> SASL auth -> request-context -> capability gate
//! -> ownership -> dispatch with real wire frames (no internal shortcut):
//!
//! 1. an admin logs in over `SCRAM-SHA-256`, sees `access_admin`, and performs user CRUD via the
//!    `AccessControl` API (`UserList`/`UserCreate`);
//! 2. two normal users (`alice`, `bob`) log in over SCRAM and each opens a session;
//! 3. the roster is owner-scoped (alice sees only her own session, never bob's);
//! 4. a cross-user op is `Forbidden` (alice cannot poll/cancel/inspect bob's session);
//! 5. the operator-tier admin transcends ownership (`SessionSeeAll`/`SessionControlAny`);
//! 6. alice reconnects on a fresh connection via `AuthResume(token)` and is the same principal.

use super::harness::*;
use super::wire_client::MuxConn;

use daemon_api::{ApiError, ApiRequest, ApiResponse, SessionQuery, SessionScope};
use daemon_auth::{AuthStore, Role};
use daemon_common::ReqId;
use daemon_host::{serve_api_unix_authenticated, AuthAudit, Authenticator};
use daemon_protocol::{AgentCommand, UserMsg};
use daemon_telemetry::TraceSigner;
use tokio::net::UnixStream;

/// Open + `Hello`-handshake a fresh mux connection to the authenticated socket.
async fn connect(path: &std::path::Path) -> MuxConn<UnixStream> {
    let stream = UnixStream::connect(path).await.expect("connect socket");
    MuxConn::handshake(stream).await.expect("hello handshake")
}

fn start_turn(text: &str) -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new(text),
        request_id: ReqId(1),
    }
}

/// The owner-scoped roster (`SessionsQuery` under the connection's principal), as a set of ids.
async fn roster_ids(conn: &mut MuxConn<UnixStream>) -> Vec<SessionId> {
    let query = SessionQuery {
        scope: SessionScope::All,
        ..Default::default()
    };
    match conn
        .call(ApiRequest::SessionsQuery { query })
        .await
        .expect("sessions query")
    {
        ApiResponse::SessionPage(page) => page.sessions.into_iter().map(|i| i.session).collect(),
        other => panic!("expected SessionPage, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_auth_lifecycle_over_the_wire() {
    let (node, handle) = assemble();

    // One identity store backs BOTH the authenticator (login) and the node's AccessControl surface
    // (admin CRUD) — the integrated shape the binary wires. Seed an admin and one normal user.
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .create_user("root", "rootpw", &[Role::Admin])
        .expect("create admin");
    store
        .create_user("alice", "alicepw", &[Role::User])
        .expect("create alice");

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
    let server = tokio::spawn(serve_api_unix_authenticated(listener, node.clone(), auth));

    // 1. Admin logs in over SCRAM and holds `access_admin`.
    let mut admin = connect(&path).await;
    let (admin_view, _admin_token) = admin
        .authenticate_scram("root", "rootpw")
        .await
        .expect("admin scram");
    assert_eq!(admin_view.username, "root");
    assert!(
        admin_view
            .capabilities
            .contains(&"access_admin".to_string()),
        "the admin principal advertises access_admin, got {:?}",
        admin_view.capabilities
    );

    // 1b. Admin user CRUD over the AccessControl API: the seeded users are listed, and a new user is
    //     created over the wire (the admin path the GUI/CLI drives).
    match admin.call(ApiRequest::UserList).await.expect("user list") {
        ApiResponse::AccessUsers(users) => {
            assert!(users.iter().any(|u| u.username == "root"));
            assert!(users.iter().any(|u| u.username == "alice"));
        }
        other => panic!("expected AccessUsers, got {other:?}"),
    }
    match admin
        .call(ApiRequest::UserCreate {
            username: "bob".into(),
            password: "bobpw".into(),
            roles: vec!["user".into()],
        })
        .await
        .expect("user create")
    {
        ApiResponse::AccessUser(u) => {
            assert_eq!(u.username, "bob");
            assert_eq!(u.roles, vec!["user".to_string()]);
        }
        other => panic!("expected AccessUser, got {other:?}"),
    }

    // 2. Alice logs in over SCRAM: a normal user (write over her own surfaces, NOT access_admin).
    let mut alice = connect(&path).await;
    let (alice_view, alice_token) = alice
        .authenticate_scram("alice", "alicepw")
        .await
        .expect("alice scram");
    assert_eq!(alice_view.username, "alice");
    assert!(alice_view
        .capabilities
        .contains(&"session_write".to_string()));
    assert!(!alice_view
        .capabilities
        .contains(&"access_admin".to_string()));

    // Alice opens a session — the first interactive submit stamps her as owner.
    let s_alice = SessionId::new("e2e-alice");
    assert!(
        !matches!(
            alice
                .call(ApiRequest::Submit {
                    session: s_alice.clone(),
                    command: start_turn("alice topic"),
                    origin: None,
                    profile: None,
                })
                .await
                .expect("alice submit"),
            ApiResponse::Error(_)
        ),
        "alice's own submit must succeed"
    );

    // 2b. Bob (the just-created user) logs in over SCRAM and opens his own session.
    let mut bob = connect(&path).await;
    let (bob_view, _bob_token) = bob
        .authenticate_scram("bob", "bobpw")
        .await
        .expect("bob scram");
    assert_eq!(bob_view.username, "bob");
    let s_bob = SessionId::new("e2e-bob");
    assert!(
        !matches!(
            bob.call(ApiRequest::Submit {
                session: s_bob.clone(),
                command: start_turn("bob topic"),
                origin: None,
                profile: None,
            })
            .await
            .expect("bob submit"),
            ApiResponse::Error(_)
        ),
        "bob's own submit must succeed"
    );

    // 3. The roster is owner-scoped: alice sees her own session and NOT bob's.
    let alice_roster = roster_ids(&mut alice).await;
    assert!(
        alice_roster.contains(&s_alice) && !alice_roster.contains(&s_bob),
        "alice's roster must be owner-scoped, got {alice_roster:?}"
    );

    // 4. Cross-user is fail-closed: alice cannot poll, cancel, or inspect bob's session.
    assert!(
        matches!(
            alice
                .call(ApiRequest::Poll {
                    session: s_bob.clone(),
                    max: 0,
                })
                .await
                .expect("alice poll bob"),
            ApiResponse::Error(ApiError::Forbidden(_))
        ),
        "alice must be Forbidden polling bob's session"
    );
    assert!(
        matches!(
            alice
                .call(ApiRequest::Cancel {
                    session: s_bob.clone(),
                })
                .await
                .expect("alice cancel bob"),
            ApiResponse::Error(ApiError::Forbidden(_))
        ),
        "alice must be Forbidden cancelling bob's session"
    );
    match alice
        .call(ApiRequest::SessionGet {
            session: s_bob.clone(),
        })
        .await
        .expect("alice get bob")
    {
        // No existence oracle: a peer's `session_get` returns `None`, not a denial that confirms it.
        ApiResponse::SessionDetail(detail) => assert!(
            detail.is_none(),
            "alice must not inspect bob's session (got Some)"
        ),
        other => panic!("expected SessionDetail, got {other:?}"),
    }

    // 5. The operator-tier admin transcends ownership: it sees BOTH sessions and may control bob's.
    let admin_roster = roster_ids(&mut admin).await;
    assert!(
        admin_roster.contains(&s_alice) && admin_roster.contains(&s_bob),
        "the operator-tier admin (SessionSeeAll) sees every user's sessions, got {admin_roster:?}"
    );
    assert!(
        !matches!(
            admin
                .call(ApiRequest::Poll {
                    session: s_bob.clone(),
                    max: 0,
                })
                .await
                .expect("admin poll bob"),
            ApiResponse::Error(_)
        ),
        "the admin (SessionControlAny) may poll bob's session"
    );

    // 6. Reconnect fast-path: alice opens a NEW connection and resumes via her token (no password),
    //    rebinding the same principal — and the resumed connection is still owner-scoped.
    let mut alice2 = connect(&path).await;
    let resumed = alice2
        .authenticate_resume(&alice_token)
        .await
        .expect("alice resumes via token");
    assert_eq!(resumed.username, "alice");
    match alice2.call(ApiRequest::WhoAmI).await.expect("whoami") {
        ApiResponse::WhoAmI(view) => assert_eq!(view.username, "alice"),
        other => panic!("expected WhoAmI, got {other:?}"),
    }
    let resumed_roster = roster_ids(&mut alice2).await;
    assert!(
        resumed_roster.contains(&s_alice) && !resumed_roster.contains(&s_bob),
        "the resumed connection stays owner-scoped, got {resumed_roster:?}"
    );

    server.abort();
    handle.shutdown().await;
}
