// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// FOUNDATION: `AgentCommand::Observe` appends context **without** running a turn (the multi-party
/// accumulation seam, event-io §5.9). Idle: an Observe emits no `TurnStarted` and folds into the
/// conversation the next `StartTurn` runs on. Busy: an Observe injected mid-turn starts no turn of
/// its own and lands in the conversation (drained at the phase boundary) for the following turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observe_appends_context_without_starting_a_turn() {
    use async_trait::async_trait;
    use daemon_api::{Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
    use daemon_protocol::{AgentCommand, AgentEvent, ConvView, UserMsg};

    // Collect every drained event for `window` (used to assert presence/absence of `TurnStarted`).
    async fn collect_for(
        node: &Arc<NodeApiImpl>,
        session: &SessionId,
        window: Duration,
    ) -> Vec<AgentEvent> {
        let deadline = Instant::now() + window;
        let mut events = Vec::new();
        while Instant::now() < deadline {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(ev) = item {
                    events.push(ev);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        events
    }

    // Drain into `events` until one matches `pred`.
    async fn drain_until(
        node: &Arc<NodeApiImpl>,
        session: &SessionId,
        events: &mut Vec<AgentEvent>,
        pred: impl Fn(&AgentEvent) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(ev) = item {
                    events.push(ev);
                }
            }
            if events.iter().any(&pred) {
                return;
            }
            assert!(Instant::now() < deadline, "never saw the expected event");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn snapshot_view(events: &[AgentEvent], request_id: ReqId) -> ConvView {
        events
            .iter()
            .find_map(|e| match e {
                AgentEvent::Snapshot {
                    request_id: id,
                    view,
                    ..
                } if *id == request_id => Some(view.clone()),
                _ => None,
            })
            .expect("a Snapshot event is present")
    }

    let conv_has = |view: &ConvView, needle: &str| -> bool {
        view.turns.iter().any(|t| t.text.contains(needle))
    };
    let started_count = |events: &[AgentEvent]| {
        events
            .iter()
            .filter(|e| matches!(e, AgentEvent::TurnStarted { .. }))
            .count()
    };

    // ---------------------------- idle ----------------------------
    let (node, handle) = assemble();
    let idle = SessionId::new("obs-idle");

    node.submit(
        idle.clone(),
        AgentCommand::Observe {
            input: UserMsg::new("[alice] the launch code is 4242"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("idle observe");
    let idle_window = collect_for(&node, &idle, Duration::from_millis(300)).await;
    assert_eq!(
        started_count(&idle_window),
        0,
        "an idle Observe must not start a turn: {idle_window:?}"
    );

    node.submit(
        idle.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("what is the code?"),
            request_id: ReqId(2),
        },
    )
    .await
    .expect("start turn");
    let mut idle_events = Vec::new();
    drain_until(&node, &idle, &mut idle_events, |e| {
        matches!(e, AgentEvent::TurnFinished { .. })
    })
    .await;
    assert_eq!(
        started_count(&idle_events),
        1,
        "exactly one turn ran (the StartTurn, not the prior Observe)"
    );

    node.submit(
        idle.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(3),
        },
    )
    .await
    .expect("snapshot");
    drain_until(
        &node,
        &idle,
        &mut idle_events,
        |e| matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(3)),
    )
    .await;
    let view = snapshot_view(&idle_events, ReqId(3));
    assert!(
        conv_has(&view, "launch code is 4242"),
        "the idle Observe folded into the conversation the next turn ran on: {view:?}"
    );
    assert!(
        conv_has(&view, "what is the code?"),
        "the StartTurn input shares that same conversation"
    );
    handle.shutdown().await;

    // ---------------------------- busy ----------------------------
    // A tool that blocks until the test releases it, so we can inject an Observe while a turn is
    // genuinely in flight (the engine sits inside this tool awaiting the gate).
    struct GateTool {
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait]
    impl Tool for GateTool {
        fn name(&self) -> &str {
            "gate"
        }
        fn schema(&self) -> &str {
            "{}"
        }
        async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
            self.release.notified().await;
            ToolOutcome::text(call.call_id.clone(), true, "released")
        }
    }

    let release = Arc::new(tokio::sync::Notify::new());
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::delegating("gate", "turn-one-done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers,
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x33; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: vec![Arc::new(GateTool {
            release: release.clone(),
        }) as Arc<dyn Tool>],
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
        reaper: Default::default(),
    });
    let busy = SessionId::new("obs-busy");

    node.submit(
        busy.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("go"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("start busy turn");
    let mut busy_events = Vec::new();
    // Wait until the gate tool is in flight: turn one is genuinely busy.
    drain_until(&node, &busy, &mut busy_events, |e| {
        matches!(e, AgentEvent::ToolStarted { .. })
    })
    .await;
    // Inject an Observe mid-turn, give the actor a moment to fold it onto the control queue, then
    // release the gate so the turn finalizes (draining the observe at the boundary).
    node.submit(
        busy.clone(),
        AgentCommand::Observe {
            input: UserMsg::new("[bob] mid-turn fact: the sky is green"),
            request_id: ReqId(2),
        },
    )
    .await
    .expect("busy observe");
    tokio::time::sleep(Duration::from_millis(50)).await;
    release.notify_one();
    drain_until(&node, &busy, &mut busy_events, |e| {
        matches!(e, AgentEvent::TurnFinished { .. })
    })
    .await;
    assert_eq!(
        started_count(&busy_events),
        1,
        "the mid-turn Observe started no turn of its own: {busy_events:?}"
    );

    node.submit(
        busy.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(3),
        },
    )
    .await
    .expect("snapshot");
    drain_until(
        &node,
        &busy,
        &mut busy_events,
        |e| matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(3)),
    )
    .await;
    let view = snapshot_view(&busy_events, ReqId(3));
    assert!(
        conv_has(&view, "the sky is green"),
        "the busy Observe landed in the conversation (drained at the boundary) for the following turn: {view:?}"
    );
    handle.shutdown().await;
}

/// FOUNDATION (inbound gate, daemon-event-io-spec §5.9.1 — the reusable `daemon-ingest` helper,
/// the symmetric counterpart to §5.9.3's `daemon-delivery`): an adapter classifies whether a
/// message is *addressed*; the `Ingestor` owns the transport-agnostic command selection over
/// `submit_routed`. Proves, with no chat transport, against the real host: an ambient (non-
/// addressed) reception emits `Observe` (no `TurnStarted`), and the following addressed reception
/// opens exactly one turn whose conversation carries both the folded-in ambient context and the
/// addressed text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_gate_folds_ambient_then_addressed_turns() {
    use daemon_api::{NodeApi, Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_ingest::{Ingestor, Reception};
    use daemon_protocol::{AgentCommand, AgentEvent, ConvView, Origin, OriginScope, UserMsg};

    async fn drain_until(
        node: &Arc<NodeApiImpl>,
        session: &SessionId,
        events: &mut Vec<AgentEvent>,
        pred: impl Fn(&AgentEvent) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(ev) = item {
                    events.push(ev);
                }
            }
            if events.iter().any(&pred) {
                return;
            }
            assert!(Instant::now() < deadline, "never saw the expected event");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    let (node, handle) = assemble();
    let ing = Ingestor::new(node.clone() as Arc<dyn NodeApi>);
    let origin = Origin::new(
        "matrix/@bot:hs",
        OriginScope::Group {
            chat: "#room".into(),
            thread: None,
        },
    );

    // Ambient chatter -> Observe, no turn.
    let session = ing
        .receive(Reception {
            origin: origin.clone(),
            input: UserMsg::new("[alice] the launch code is 4242"),
            addressed: false,
        })
        .await
        .expect("ambient receive");
    let mut started = 0;
    let win = Instant::now() + Duration::from_millis(300);
    while Instant::now() < win {
        for item in node.poll(session.clone(), 0).await.expect("poll") {
            if let Outbound::Event(AgentEvent::TurnStarted { .. }) = item {
                started += 1;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        started, 0,
        "an ambient reception via the gate starts no turn"
    );

    // Addressed message -> StartTurn on the same session.
    let s2 = ing
        .receive(Reception {
            origin: origin.clone(),
            input: UserMsg::new("what is the code?"),
            addressed: true,
        })
        .await
        .expect("addressed receive");
    assert_eq!(s2, session, "same origin routes to the same session");

    let mut events = Vec::new();
    drain_until(&node, &session, &mut events, |e| {
        matches!(e, AgentEvent::TurnFinished { .. })
    })
    .await;
    let started_turns = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStarted { .. }))
        .count();
    assert_eq!(
        started_turns, 1,
        "the addressed reception ran exactly one turn"
    );

    node.submit(
        session.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(99),
        },
    )
    .await
    .expect("snapshot");
    drain_until(
        &node,
        &session,
        &mut events,
        |e| matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(99)),
    )
    .await;
    let view: ConvView = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::Snapshot {
                request_id, view, ..
            } if *request_id == ReqId(99) => Some(view.clone()),
            _ => None,
        })
        .expect("a Snapshot view");
    let conv_has = |needle: &str| view.turns.iter().any(|t| t.text.contains(needle));
    assert!(
        conv_has("launch code is 4242"),
        "the gate's Observe folded the ambient context into the conversation: {view:?}"
    );
    assert!(
        conv_has("what is the code?"),
        "the addressed turn shares that conversation"
    );
    handle.shutdown().await;
}

/// FOUNDATION (inbound gate, §5.9.1 — the busy path): with the default `BusyPolicy::Queue`, an
/// addressed reception that arrives while a turn is genuinely in flight is held and replayed as a
/// single follow-up `StartTurn` when the turn finishes (driven by the adapter's
/// `note_turn_started` / `note_turn_finished` hooks). Proves it end-to-end against the real host:
/// the queued message runs no turn until the first finishes, then opens its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_gate_queues_addressed_while_busy_then_flushes() {
    use async_trait::async_trait;
    use daemon_api::{NodeApi, Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
    use daemon_ingest::{Ingestor, Reception};
    use daemon_protocol::{AgentCommand, AgentEvent, ConvView, Origin, OriginScope, UserMsg};

    struct GateTool {
        release: Arc<tokio::sync::Notify>,
    }
    #[async_trait]
    impl Tool for GateTool {
        fn name(&self) -> &str {
            "gate"
        }
        fn schema(&self) -> &str {
            "{}"
        }
        async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
            self.release.notified().await;
            ToolOutcome::text(call.call_id.clone(), true, "released")
        }
    }

    async fn drain_until_count(
        node: &Arc<NodeApiImpl>,
        session: &SessionId,
        events: &mut Vec<AgentEvent>,
        pred: impl Fn(&AgentEvent) -> bool,
        n: usize,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(ev) = item {
                    events.push(ev);
                }
            }
            if events.iter().filter(|e| pred(e)).count() >= n {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "never reached {n} matching events"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    let is_tool_started = |e: &AgentEvent| matches!(e, AgentEvent::ToolStarted { .. });
    let is_turn_finished = |e: &AgentEvent| matches!(e, AgentEvent::TurnFinished { .. });

    let release = Arc::new(tokio::sync::Notify::new());
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::delegating("gate", "done")) as Arc<dyn Provider>
    }));
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers,
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x71; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: vec![Arc::new(GateTool {
            release: release.clone(),
        }) as Arc<dyn Tool>],
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
        reaper: Default::default(),
    });
    let ing = Ingestor::new(node.clone() as Arc<dyn NodeApi>);
    let origin = Origin::new(
        "matrix/@bot:hs",
        OriginScope::Group {
            chat: "#q".into(),
            thread: None,
        },
    );

    // Turn one opens and blocks inside the gate tool.
    let session = ing
        .receive(Reception {
            origin: origin.clone(),
            input: UserMsg::new("first"),
            addressed: true,
        })
        .await
        .expect("first addressed");
    let mut events = Vec::new();
    drain_until_count(&node, &session, &mut events, is_tool_started, 1).await;
    ing.note_turn_started(&session);

    // An addressed message arrives mid-turn: queued, not yet submitted.
    ing.receive(Reception {
        origin: origin.clone(),
        input: UserMsg::new("second"),
        addressed: true,
    })
    .await
    .expect("second addressed (busy)");

    // Finish turn one; the queued "second" then flushes as a follow-up StartTurn. (Turn two need
    // not re-gate — its request already carries turn one's tool result — but pre-arm the gate so
    // the flush completes regardless of the mock's branch.)
    release.notify_one();
    drain_until_count(&node, &session, &mut events, is_turn_finished, 1).await;
    ing.note_turn_finished(&session)
        .await
        .expect("flush queued");
    release.notify_one();
    drain_until_count(&node, &session, &mut events, is_turn_finished, 2).await;

    let started_turns = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStarted { .. }))
        .count();
    assert_eq!(started_turns, 2, "first turn + the flushed queued turn");

    node.submit(
        session.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(99),
        },
    )
    .await
    .expect("snapshot");
    drain_until_count(
        &node,
        &session,
        &mut events,
        |e| matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(99)),
        1,
    )
    .await;
    let view: ConvView = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::Snapshot {
                request_id, view, ..
            } if *request_id == ReqId(99) => Some(view.clone()),
            _ => None,
        })
        .expect("a Snapshot view");
    assert!(
        view.turns.iter().any(|t| t.text.contains("second")),
        "the queued message ran after the first turn finished: {view:?}"
    );
    handle.shutdown().await;
}

/// FOUNDATION (inbound gate, §5.9.1 — routing intact through the gate): `Ingestor::receive`
/// submits via `submit_routed`, so the §5.9 routing precedence still selects the agent. Proves
/// two addressed receptions for two `bound_accounts`-bound matrix instances route to two distinct
/// sessions, each run by the right profile (the echoing resolver reveals which).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_gate_routes_distinct_origins_to_bound_profiles() {
    use daemon_api::{BoundAccount, NodeApi, Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_host::{MemProfileStore, ProfileStore};
    use daemon_ingest::{Ingestor, Reception};
    use daemon_protocol::{AgentEvent, Origin, OriginScope, UserMsg};

    let store = Arc::new(MemProfileStore::new());
    store
        .create(
            ProfileSpec::new("alpha", ProviderSelector::GenAi, "model-a")
                .with_bound_accounts(vec![BoundAccount::new("matrix/@a:hs", "matrix/alpha/a")]),
        )
        .expect("create alpha");
    store
        .create(
            ProfileSpec::new("beta", ProviderSelector::GenAi, "model-b")
                .with_bound_accounts(vec![BoundAccount::new("matrix/@b:hs", "matrix/beta/b")]),
        )
        .expect("create beta");
    store.set_active("alpha").expect("set active");

    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}]", spec.id);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x72; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(store),
        provider_resolver: Some(resolver),
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
        reaper: Default::default(),
    });
    let ing = Ingestor::new(node.clone() as Arc<dyn NodeApi>);

    async fn final_text_for(node: &Arc<NodeApiImpl>, session: &SessionId) -> String {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(AgentEvent::TurnFinished { summary, .. }) = item {
                    return summary.final_text.unwrap_or_default();
                }
            }
            assert!(Instant::now() < deadline, "turn never finished");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    let addressed = |instance: &str| Reception {
        origin: Origin::new(
            instance,
            OriginScope::Group {
                chat: "#general".into(),
                thread: None,
            },
        ),
        input: UserMsg::new("hi"),
        addressed: true,
    };

    let sa = ing
        .receive(addressed("matrix/@a:hs"))
        .await
        .expect("route a");
    let sb = ing
        .receive(addressed("matrix/@b:hs"))
        .await
        .expect("route b");
    assert_ne!(sa, sb, "the two instances derive distinct sessions");

    let ta = final_text_for(&node, &sa).await;
    let tb = final_text_for(&node, &sb).await;
    assert!(
        ta.contains("[alpha]"),
        "@a:hs routed to alpha via the gate, got {ta:?}"
    );
    assert!(
        tb.contains("[beta]"),
        "@b:hs routed to beta via the gate, got {tb:?}"
    );
    handle.shutdown().await;
}
