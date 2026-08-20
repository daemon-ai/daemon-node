// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Session-recall conformance: turn-boundary FTS indexing (the live pump's snapshot round-trip
//! AND the durable incarnation's post-seal index), background title generation replacing the
//! truncation seed, and the pure-local `session_recap` op (durable-snapshot + live-view sources,
//! Auth-4 scoped).

use super::harness::*;
use daemon_api::SessionApi;
use daemon_auth::{Principal, Role};
use daemon_common::ReqId;
use daemon_host::{with_request_context, RequestContext};
use daemon_protocol::{AgentCommand, Outbound, UserMsg};

fn ctx(name: &str, role: Role) -> RequestContext {
    RequestContext::authenticated(Principal::from_roles(name, name, vec![role]), None)
}

fn start_turn(text: &str) -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new(text),
        request_id: ReqId(1),
    }
}

/// Poll `probe` until it yields `Some` or the deadline passes (the turn-boundary bookkeeping is
/// asynchronous by design: TurnFinished -> internal snapshot -> spawned index/title task).
async fn wait_for<T, F, Fut>(what: &str, mut probe: F) -> T
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Option<T>>,
{
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(v) = probe().await {
            return v;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A live turn indexes the FULL coalesced conversation at the turn boundary: the ASSISTANT reply
/// text is searchable (the old submit-time index carried only the latest user turn), and the
/// pump's internal snapshot round-trip never leaks onto the client drain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_turn_indexes_full_conversation_for_search() {
    let (node, handle, store) = assemble_with_store_for_recall(None);
    let session = SessionId::new("s-live-index");

    with_request_context(ctx("alice", Role::User), async {
        node.submit(session.clone(), start_turn("tell me about the moon"))
            .await
    })
    .await
    .expect("alice opens");

    // "session done" is the MOCK ASSISTANT reply — only the post-turn coalesced body carries it.
    let hits = wait_for("the assistant text to be indexed", || {
        let store = store.clone();
        async move {
            let hits = store.search_sessions("session done", 10).await;
            (!hits.is_empty()).then_some(hits)
        }
    })
    .await;
    assert!(hits.iter().any(|h| h.session_id == session));

    // The user turn is in the same body.
    assert!(store
        .search_sessions("moon", 10)
        .await
        .iter()
        .any(|h| h.session_id == session));

    // The internal snapshot reply was swallowed by the pump: a client drain sees turn events but
    // no host-internal `Snapshot` frame.
    let outbound = with_request_context(ctx("alice", Role::User), async {
        node.poll(session.clone(), 0).await
    })
    .await
    .expect("owner polls");
    assert!(
        !outbound.iter().any(|o| matches!(
            o,
            Outbound::Event(daemon_protocol::AgentEvent::Snapshot { .. })
        )),
        "host-internal snapshot replies must not reach the client drain"
    );

    handle.shutdown().await;
}

/// After the first exchange, the background title generator replaces the truncation-seeded roster
/// title with the aux model's cleaned reply, and the FTS row's title column follows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn title_generation_replaces_the_seed_after_first_exchange() {
    // The aux replies with a quoted title — the cleanup must strip the quotes.
    let title_aux: Arc<dyn Provider> =
        Arc::new(MockProvider::completing("\"Docker Networking Help\""));
    let (node, handle, store) = assemble_with_store_for_recall(Some(fixed_title_aux(title_aux)));
    let session = SessionId::new("s-titled");

    with_request_context(ctx("alice", Role::User), async {
        node.submit(
            session.clone(),
            start_turn("please help me with docker networking setup"),
        )
        .await
    })
    .await
    .expect("alice opens");

    // The submit-time seed lands first (the truncated opening turn)…
    let seeded = store
        .session_meta(&session)
        .await
        .and_then(|m| m.title)
        .expect("note_activity seeds a title");
    assert_eq!(seeded, "please help me with docker networking setup");

    // …and the generated title replaces it once the first exchange completes.
    let title = wait_for("the generated title", || {
        let store = store.clone();
        let session = session.clone();
        async move {
            store
                .session_meta(&session)
                .await
                .and_then(|m| m.title)
                .filter(|t| t == "Docker Networking Help")
        }
    })
    .await;
    assert_eq!(title, "Docker Networking Help");

    // The FTS row's title column follows (the post-title re-index).
    let hit_title = wait_for("the FTS title refresh", || {
        let store = store.clone();
        let session = session.clone();
        async move {
            store
                .search_sessions("docker", 10)
                .await
                .into_iter()
                .find(|h| h.session_id == session)
                .map(|h| h.title)
                .filter(|t| t == "Docker Networking Help")
        }
    })
    .await;
    assert_eq!(hit_title, "Docker Networking Help");

    handle.shutdown().await;
}

/// The title aux is resolved at TITLE TIME, not boot (the wizard-first-run fix): an exchange
/// whose resolver finds no provider keeps the truncation seed WITHOUT consuming the
/// once-per-residency guard, and once a provider becomes resolvable (the operator commissioned a
/// profile), the next exchange inside the first-exchange gate (≤ 2 user turns) titles.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn title_generation_resolves_lazily_after_a_late_created_profile() {
    // The "profile store": empty at boot, filled mid-session — the resolver reads it live.
    let slot: Arc<std::sync::Mutex<Option<Arc<dyn Provider>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let resolver: daemon_host::TitleAuxResolver = {
        let slot = slot.clone();
        Arc::new(move |_| slot.lock().unwrap().clone())
    };
    let (node, handle, store) = assemble_with_store_for_recall(Some(resolver));
    let session = SessionId::new("s-late-titled");

    with_request_context(ctx("alice", Role::User), async {
        node.submit(
            session.clone(),
            start_turn("first question about docker networking"),
        )
        .await
    })
    .await
    .expect("alice opens");

    // The first exchange completes (its FTS index lands) with nothing resolvable: the seed stays.
    wait_for("the first exchange to be indexed", || {
        let store = store.clone();
        let session = session.clone();
        async move {
            let hits = store.search_sessions("docker", 10).await;
            hits.iter().any(|h| h.session_id == session).then_some(())
        }
    })
    .await;
    assert_eq!(
        store.session_meta(&session).await.and_then(|m| m.title),
        Some("first question about docker networking".into()),
        "an unresolvable aux leaves the truncation seed untouched"
    );

    // The operator "commissions a profile": the resolver now yields a provider…
    *slot.lock().unwrap() = Some(Arc::new(MockProvider::completing("\"Late Docker Title\"")));

    // …and the SECOND exchange (still within the ≤ 2-user-turn gate) titles — proving the
    // unresolvable first pass did not consume the once-per-residency guard.
    with_request_context(ctx("alice", Role::User), async {
        node.submit(session.clone(), start_turn("a follow-up question"))
            .await
    })
    .await
    .expect("alice follows up");
    let title = wait_for("the late-resolved generated title", || {
        let store = store.clone();
        let session = session.clone();
        async move {
            store
                .session_meta(&session)
                .await
                .and_then(|m| m.title)
                .filter(|t| t == "Late Docker Title")
        }
    })
    .await;
    assert_eq!(title, "Late Docker Title");

    handle.shutdown().await;
}

/// `session_recap` is pure-local and source-ordered: a durable session recaps from its snapshot
/// (tool names included), a live-only session from its resident conversation view; a peer gets
/// `None` (Auth 4, no existence oracle) while the owner and an operator read it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_recap_serves_durable_and_live_sessions_owner_scoped() {
    let (node, handle, store) = assemble_with_store_for_recall(None);

    // Durable: assign -> the orchestrator mock delegates once, then completes.
    let durable = SessionId::new("r-durable");
    with_request_context(ctx("alice", Role::User), async {
        node.assign(durable.clone()).await
    })
    .await
    .expect("alice assigns");
    let recap = wait_for("the durable recap", || {
        let node = node.clone();
        let durable = durable.clone();
        async move {
            with_request_context(ctx("alice", Role::User), async {
                node.session_recap(durable.clone()).await
            })
            .await
            .filter(|r| {
                r.last_reply
                    .as_deref()
                    .is_some_and(|reply| reply.contains("fleet done"))
            })
        }
    })
    .await;
    assert!(
        recap
            .top_tools
            .iter()
            .any(|(name, count)| name == "orchestrate" && *count > 0),
        "the durable snapshot's tool turns feed top_tools, got {recap:?}"
    );

    // The durable incarnation indexed the coalesced conversation at the turn boundary too: the
    // orchestrator's FINAL assistant text is searchable.
    let hits = wait_for("the durable session's FTS row", || {
        let store = store.clone();
        async move {
            let hits = store.search_sessions("fleet done", 10).await;
            (!hits.is_empty()).then_some(hits)
        }
    })
    .await;
    assert!(hits.iter().any(|h| h.session_id == durable));

    // Wire-opened interactive session (stage-5 cutover): `submit` is durable-first — the session
    // gets a durable row and its recap reads the committed snapshot (there is no separate
    // live-only source anymore; the resident-view leg now only serves Foreign residencies).
    let live = SessionId::new("r-live");
    with_request_context(ctx("alice", Role::User), async {
        node.submit(live.clone(), start_turn("what is the meaning of life"))
            .await
    })
    .await
    .expect("alice opens the interactive session");
    let live_recap = wait_for("the live recap", || {
        let node = node.clone();
        let live = live.clone();
        async move {
            with_request_context(ctx("alice", Role::User), async {
                node.session_recap(live.clone()).await
            })
            .await
            .filter(|r| r.assistant_turns > 0)
        }
    })
    .await;
    assert_eq!(live_recap.user_turns, 1);
    assert_eq!(
        live_recap.last_ask.as_deref(),
        Some("what is the meaning of life")
    );

    // Auth 4: a peer sees nothing (no existence oracle); an operator crosses ownership.
    let bob = with_request_context(ctx("bob", Role::User), async {
        node.session_recap(live.clone()).await
    })
    .await;
    assert!(bob.is_none(), "a peer must not recap another's session");
    assert!(with_request_context(ctx("op", Role::Operator), async {
        node.session_recap(live.clone()).await
    })
    .await
    .is_some());

    handle.shutdown().await;
}

/// Wrap a fixed provider as the trivial lazy title-aux resolver (tests that don't exercise the
/// late-resolution seam).
fn fixed_title_aux(provider: Arc<dyn Provider>) -> daemon_host::TitleAuxResolver {
    Arc::new(move |_| Some(provider.clone()))
}

/// Assemble exactly like [`assemble_over`] but with an optional title-generation aux resolver.
fn assemble_with_store_for_recall(
    title_aux: Option<daemon_host::TitleAuxResolver>,
) -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<dyn SessionStore>,
) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: store.clone(),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x5c; 32]),
        nesting_depth: 1,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
        processes: Default::default(),
        title_aux,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });
    (node, handle, store)
}
