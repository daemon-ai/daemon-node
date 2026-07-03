// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Auth 4 session-ownership conformance: owner stamped on every creation path; children inherit the
//! parent/job owner (delegation + cron); a peer can neither see (roster/get/search/tree) nor control
//! (poll/cancel/submit) another user's session; `SessionSeeAll` (reads) and `SessionControlAny`
//! (control) operator overrides; legacy `owner IS NULL` rows hidden from non-operators. The node is
//! driven IN-PROCESS under an explicit request principal (the transport gate is exercised by the
//! Auth 2/3 suites); the durable `store` handle lets us assert the stamped owner directly.

use super::harness::*;
use daemon_api::{ApiError, SessionApi, SessionQuery, SessionScope};
use daemon_auth::{Principal, Role};
use daemon_common::ReqId;
use daemon_host::{with_request_context, RequestContext};
use daemon_protocol::{AgentCommand, UserMsg};

/// Assemble a node retaining its shared durable store so a test can assert the stamped `owner`.
fn assemble_with_store() -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<dyn SessionStore>,
) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 1, [0x4a; 32], fast_host_config());
    (node, handle, store)
}

/// A request context bound to `name` (its own `user_id`) holding exactly `role`.
fn ctx(name: &str, role: Role) -> RequestContext {
    RequestContext::authenticated(Principal::from_roles(name, name, vec![role]), None)
}

fn start_turn(text: &str) -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new(text),
        request_id: ReqId(1),
    }
}

/// The owner stamped on a session's durable meta (`None` if no meta / unowned).
async fn owner_of(store: &Arc<dyn SessionStore>, session: &SessionId) -> Option<String> {
    store.session_meta(session).await.and_then(|m| m.owner)
}

/// An interactive submit stamps the caller as owner; a peer can neither control nor see the session,
/// while an operator (`SessionControlAny` / `SessionSeeAll`) transcends ownership.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_stamps_owner_then_peer_denied_operator_allowed() {
    let (node, handle, store) = assemble_with_store();
    let session = SessionId::new("s1");

    // Alice opens the session — owner is stamped from her principal.
    with_request_context(ctx("alice", Role::User), async {
        node.submit(session.clone(), start_turn("alpha")).await
    })
    .await
    .expect("alice opens her session");
    assert_eq!(
        owner_of(&store, &session).await.as_deref(),
        Some("alice"),
        "the first interactive submit stamps the caller as owner"
    );

    // Bob (a peer) can neither control nor (re)submit to Alice's session.
    let bob_poll = with_request_context(ctx("bob", Role::User), async {
        node.poll(session.clone(), 0).await
    })
    .await;
    assert!(
        matches!(bob_poll, Err(ApiError::Forbidden(_))),
        "a peer cannot poll another user's session, got {bob_poll:?}"
    );
    let bob_cancel = with_request_context(ctx("bob", Role::User), async {
        node.cancel(session.clone()).await
    })
    .await;
    assert!(
        matches!(bob_cancel, Err(ApiError::Forbidden(_))),
        "a peer cannot cancel another user's session, got {bob_cancel:?}"
    );
    let bob_submit = with_request_context(ctx("bob", Role::User), async {
        node.submit(session.clone(), start_turn("intrude")).await
    })
    .await;
    assert!(
        bob_submit.is_err(),
        "a peer cannot submit to another user's session"
    );

    // Alice (owner) and an operator (SessionControlAny) may both control it.
    with_request_context(ctx("alice", Role::User), async {
        node.poll(session.clone(), 0).await
    })
    .await
    .expect("the owner may poll");
    with_request_context(ctx("op", Role::Operator), async {
        node.poll(session.clone(), 0).await
    })
    .await
    .expect("an operator (SessionControlAny) crosses ownership");

    handle.shutdown().await;
}

/// Roster / `session_get` / `session_search` are scoped to the request principal: a peer sees only
/// its own sessions; an operator (`SessionSeeAll`) sees every user's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_get_search_are_owner_scoped() {
    let (node, handle, _store) = assemble_with_store();
    let s_alice = SessionId::new("s-alice");
    let s_bob = SessionId::new("s-bob");

    with_request_context(ctx("alice", Role::User), async {
        node.submit(s_alice.clone(), start_turn("alpha topic"))
            .await
    })
    .await
    .expect("alice opens");
    with_request_context(ctx("bob", Role::User), async {
        node.submit(s_bob.clone(), start_turn("beta topic")).await
    })
    .await
    .expect("bob opens");

    let all = || SessionQuery {
        scope: SessionScope::All,
        ..Default::default()
    };
    let has = |page: &daemon_api::SessionPage, id: &SessionId| {
        page.sessions.iter().any(|i| &i.session == id)
    };

    // Roster scope.
    let alice_roster = with_request_context(ctx("alice", Role::User), async {
        node.sessions_query(all()).await
    })
    .await;
    assert!(has(&alice_roster, &s_alice) && !has(&alice_roster, &s_bob));
    let bob_roster = with_request_context(ctx("bob", Role::User), async {
        node.sessions_query(all()).await
    })
    .await;
    assert!(has(&bob_roster, &s_bob) && !has(&bob_roster, &s_alice));
    let op_roster = with_request_context(ctx("op", Role::Operator), async {
        node.sessions_query(all()).await
    })
    .await;
    assert!(
        has(&op_roster, &s_alice) && has(&op_roster, &s_bob),
        "an operator (SessionSeeAll) sees every user's sessions"
    );

    // session_get is a read-of-one: a peer gets `None` (no existence oracle); owner/operator get it.
    let bob_get = with_request_context(ctx("bob", Role::User), async {
        node.session_get(s_alice.clone()).await
    })
    .await;
    assert!(bob_get.is_none(), "a peer cannot inspect another's session");
    assert!(with_request_context(ctx("alice", Role::User), async {
        node.session_get(s_alice.clone()).await
    })
    .await
    .is_some());
    assert!(with_request_context(ctx("op", Role::Operator), async {
        node.session_get(s_alice.clone()).await
    })
    .await
    .is_some());

    // session_search is owner-scoped too.
    let alice_hits = with_request_context(ctx("alice", Role::User), async {
        node.session_search("alpha".into(), 10).await
    })
    .await;
    assert!(alice_hits.iter().any(|h| h.session == s_alice));
    let bob_hits = with_request_context(ctx("bob", Role::User), async {
        node.session_search("alpha".into(), 10).await
    })
    .await;
    assert!(
        bob_hits.is_empty(),
        "search must not leak another user's session, got {bob_hits:?}"
    );

    handle.shutdown().await;
}

/// A legacy `owner IS NULL` session (created with no bound principal — the trusted in-process path)
/// is hidden from a non-operator peer and reachable only via the operator overrides.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_null_owner_hidden_from_peer_visible_to_operator() {
    let (node, handle, store) = assemble_with_store();
    let session = SessionId::new("s-legacy");

    // No request context: trusted local caller, owner stays NULL.
    node.submit(session.clone(), start_turn("legacy"))
        .await
        .expect("trusted in-process submit opens an unowned session");
    assert_eq!(
        owner_of(&store, &session).await,
        None,
        "a principal-less submit leaves the session unowned (legacy NULL)"
    );

    let all = SessionQuery {
        scope: SessionScope::All,
        ..Default::default()
    };
    // A peer cannot see or control it.
    let alice_roster = with_request_context(ctx("alice", Role::User), async {
        node.sessions_query(all.clone()).await
    })
    .await;
    assert!(
        !alice_roster.sessions.iter().any(|i| i.session == session),
        "a legacy NULL-owner session is hidden from a non-operator peer"
    );
    let alice_poll = with_request_context(ctx("alice", Role::User), async {
        node.poll(session.clone(), 0).await
    })
    .await;
    assert!(matches!(alice_poll, Err(ApiError::Forbidden(_))));

    // An operator sees and controls it.
    let op_roster = with_request_context(ctx("op", Role::Operator), async {
        node.sessions_query(all).await
    })
    .await;
    assert!(op_roster.sessions.iter().any(|i| i.session == session));
    with_request_context(ctx("op", Role::Operator), async {
        node.poll(session.clone(), 0).await
    })
    .await
    .expect("an operator controls a legacy session");

    handle.shutdown().await;
}

/// A delegated child INHERITS the delegating session's owner, and the owner scope follows the whole
/// subtree through the orchestration `tree()` (a foreign subtree is dropped whole).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delegation_child_inherits_owner_and_tree_is_scoped() {
    let (node, handle, store) = assemble_with_store();
    let parent = SessionId::new("d1");

    // Alice drives a durable session that delegates one child (the gate's orchestrator mock).
    with_request_context(ctx("alice", Role::User), async {
        node.assign(parent.clone()).await
    })
    .await
    .expect("alice assigns a durable session");
    assert_eq!(owner_of(&store, &parent).await.as_deref(), Some("alice"));

    // Wait for the delegation child to materialize.
    let deadline = Instant::now() + Duration::from_secs(10);
    let child = loop {
        let children = store.children_of(&parent).await;
        if let Some(c) = children.into_iter().next() {
            break c;
        }
        assert!(Instant::now() < deadline, "no delegation child appeared");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    // The child inherited Alice's ownership (the worker has no principal of its own).
    assert_eq!(
        owner_of(&store, &child).await.as_deref(),
        Some("alice"),
        "a delegated child inherits the parent's owner"
    );

    // The tree is owner-scoped: Alice sees her subtree; Bob sees none of it; the operator sees all.
    let alice_tree =
        with_request_context(ctx("alice", Role::User), async { node.tree(None).await }).await;
    assert!(
        alice_tree
            .nodes
            .iter()
            .any(|n| n.session.as_ref() == Some(&parent)),
        "the owner sees her own subtree"
    );
    let bob_tree =
        with_request_context(ctx("bob", Role::User), async { node.tree(None).await }).await;
    assert!(
        !bob_tree
            .nodes
            .iter()
            .any(|n| n.session.as_ref() == Some(&parent) || n.session.as_ref() == Some(&child)),
        "a peer sees none of another user's subtree, got {bob_tree:?}"
    );
    let op_tree =
        with_request_context(ctx("op", Role::Operator), async { node.tree(None).await }).await;
    assert!(op_tree
        .nodes
        .iter()
        .any(|n| n.session.as_ref() == Some(&parent)));

    handle.shutdown().await;
}

/// A cron-materialized session inherits the cron job's creator (captured at `cron_create`), so a
/// scheduled run is owned by — and visible only to — the user who scheduled it (and operators).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cron_session_inherits_cron_creator_owner() {
    let (node, handle, store) = assemble_with_store();

    let spec = daemon_api::CronSpec {
        name: "alice-job".into(),
        schedule: "0 9 * * *".into(),
        payload: b"do the thing".to_vec(),
        enabled: true,
        ..daemon_api::CronSpec::default()
    };
    // Alice creates the job (owner captured here), then triggers it (worker fires off-principal).
    let id = with_request_context(ctx("alice", Role::User), async {
        node.cron_create(spec).await
    })
    .await
    .expect("alice creates a cron job");
    node.cron_trigger(id.clone())
        .await
        .expect("trigger the job");

    // Find the materialized cron session via the recorded run.
    let deadline = Instant::now() + Duration::from_secs(10);
    let session = loop {
        if let Some(run) = node.cron_runs(id.clone()).await.into_iter().next() {
            if let Some(s) = run.session {
                break s;
            }
        }
        assert!(
            Instant::now() < deadline,
            "cron never recorded a run session"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    assert_eq!(
        owner_of(&store, &session).await.as_deref(),
        Some("alice"),
        "the cron session inherits the job creator's owner"
    );
    // The cron session (an EphemeralSubagent) is hidden from a peer even under the All scope.
    let bob_roster = with_request_context(ctx("bob", Role::User), async {
        node.sessions_query(SessionQuery {
            scope: SessionScope::All,
            ..Default::default()
        })
        .await
    })
    .await;
    assert!(
        !bob_roster.sessions.iter().any(|i| i.session == session),
        "a peer cannot see another user's cron session"
    );

    handle.shutdown().await;
}
