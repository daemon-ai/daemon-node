// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// FOUNDATION (outbound delivery, daemon-event-io-spec §5.9.3 — the in-process PUSH half): the
/// host's per-session pump resolves each session's *current* `Primary` and pushes its outbound
/// entries to the registered [`DeliverySink`] owning that transport. Proves, with no chat
/// transport: (1) a sink registered for the routed instance receives the session's `TurnFinished`
/// entry (push delivery, not poll); and (2) `handover` is honored for free — once the matrix
/// `Primary` is demoted to `Spectator`, the matrix sink stops receiving and the new `gui` sink
/// starts (targets are re-read every event).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivery_sink_push_honors_handover() {
    use daemon_api::{DeliverySink, Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_host::{
        DeliveryHost, MemCredentialStore, MemProfileStore, ProfileStore, RoutingRegistry,
    };
    use daemon_protocol::{
        AgentCommand, AgentEvent, DeliveryTarget, Origin, OriginScope, SessionLogEntry,
        SessionPayload, SinkKind, TransportId, UserMsg,
    };
    use std::sync::Mutex;

    // A recording sink: captures every (target, entry) the host pushes to it.
    #[derive(Default)]
    struct RecordingSink {
        got: Mutex<Vec<SessionLogEntry>>,
    }
    impl RecordingSink {
        fn turn_finished_count(&self) -> usize {
            self.got
                .lock()
                .unwrap()
                .iter()
                .filter(|e| {
                    matches!(
                        &e.payload,
                        SessionPayload::Event(AgentEvent::TurnFinished { .. })
                    )
                })
                .count()
        }
    }
    #[async_trait::async_trait]
    impl DeliverySink for RecordingSink {
        async fn deliver(&self, _target: DeliveryTarget, entry: SessionLogEntry) {
            self.got.lock().unwrap().push(entry);
        }
    }

    let store = Arc::new(MemProfileStore::new());
    let mut spec = ProfileSpec::new("alpha", ProviderSelector::GenAi, "model-a");
    spec.system_prompt = "You are alpha.".into();
    store.create(spec).expect("create profile");
    store.set_active("alpha").expect("set active");

    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}]", spec.id);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("alpha"));

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x55; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(store),
        provider_resolver: Some(resolver),
        credential_store: Some(Arc::new(MemCredentialStore::new())),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: Some(routing),
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
    });

    // Register two in-process sinks: the matrix account and a GUI surface.
    let matrix_sink = Arc::new(RecordingSink::default());
    let gui_sink = Arc::new(RecordingSink::default());
    node.register_delivery_sink(TransportId::new("matrix/@a:hs"), matrix_sink.clone());
    node.register_delivery_sink(TransportId::new("gui"), gui_sink.clone());

    let origin_a = Origin::new(
        TransportId::new("matrix/@a:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );

    // Drive a routed turn and wait for the drain to reach TurnFinished.
    async fn drive_turn(node: &Arc<NodeApiImpl>, origin: Origin) -> SessionId {
        let session = node
            .submit_routed(
                origin,
                AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(1),
                },
            )
            .await
            .expect("routed submit");
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finished = false;
        while Instant::now() < deadline && !finished {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if matches!(item, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                    finished = true;
                }
            }
            if !finished {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        assert!(finished, "routed turn never reached TurnFinished");
        session
    }

    // Wait until `sink` has observed at least `want` TurnFinished pushes (the push rides the pump
    // and can lag the drain `poll` by a scheduling tick).
    async fn wait_finished(sink: &Arc<RecordingSink>, want: usize) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if sink.turn_finished_count() >= want {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        false
    }

    let session = drive_turn(&node, origin_a.clone()).await;

    // 1. The matrix sink (the routed Primary) received the turn's outbound TurnFinished via push.
    assert!(
        wait_finished(&matrix_sink, 1).await,
        "the matrix sink should receive the first turn's TurnFinished via push"
    );
    assert_eq!(
        gui_sink.turn_finished_count(),
        0,
        "gui is not yet the Primary"
    );

    // 2. Hand the Primary over to the GUI; the matrix account is demoted to Spectator.
    let gui = DeliveryTarget::new("gui", "panel-1", SinkKind::Primary);
    node.handover(session.clone(), gui).await.expect("handover");

    // Drive a second turn on the same session (now Primary = gui).
    let _ = node
        .submit_from(
            session.clone(),
            origin_a.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("again"),
                request_id: ReqId(2),
            },
        )
        .await;
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finished = false;
    while Instant::now() < deadline && !finished {
        for item in node.poll(session.clone(), 0).await.expect("poll") {
            if matches!(item, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                finished = true;
            }
        }
        if !finished {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
    assert!(finished, "second turn never reached TurnFinished");

    // 3. The new Primary (gui) received the second turn; the demoted matrix sink did NOT (it
    // stays at one — push delivery honored the handover by re-reading the live targets).
    assert!(
        wait_finished(&gui_sink, 1).await,
        "the gui sink (new Primary) should receive the second turn's TurnFinished"
    );
    assert_eq!(
        matrix_sink.turn_finished_count(),
        1,
        "the demoted matrix sink stops receiving after handover"
    );

    handle.shutdown().await;
}

/// FOUNDATION (outbound delivery, daemon-event-io-spec §5.9.3 — the reusable PULL half): the host
/// exposes owned-session discovery (`delivery_sessions`) and the reusable `daemon-delivery`
/// subscriber stitches discovery + `subscribe` + handover-stop into one loop. Proves: (1)
/// `delivery_sessions(instance)` returns exactly that instance's routed sessions; (2) a
/// `serve_delivery` subscription projects an owned session's merged-log entries (incl. its
/// `TurnFinished`); and (3) the subscription halts that session once it is handed over (the
/// transport is demoted from `Primary`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivery_sessions_discovery_and_pull_subscriber() {
    use daemon_api::{Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_delivery::{serve_delivery, Projector};
    use daemon_host::{MemCredentialStore, MemProfileStore, ProfileStore, RoutingRegistry};
    use daemon_protocol::{
        AgentCommand, AgentEvent, DeliveryTarget, Origin, OriginScope, SessionLogEntry,
        SessionPayload, SinkKind, TransportId, UserMsg,
    };
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recorder {
        seen: Mutex<Vec<(SessionId, SessionLogEntry)>>,
    }
    impl Recorder {
        fn has_turn_finished(&self) -> bool {
            self.seen.lock().unwrap().iter().any(|(_, e)| {
                matches!(
                    &e.payload,
                    SessionPayload::Event(AgentEvent::TurnFinished { .. })
                )
            })
        }
    }
    #[async_trait::async_trait]
    impl Projector for Recorder {
        async fn project(&self, session: SessionId, entry: SessionLogEntry) {
            self.seen.lock().unwrap().push((session, entry));
        }
    }

    let store = Arc::new(MemProfileStore::new());
    for (id, model) in [("alpha", "model-a"), ("beta", "model-b")] {
        let mut spec = ProfileSpec::new(id, ProviderSelector::GenAi, model);
        spec.system_prompt = format!("You are {id}.");
        store.create(spec).expect("create profile");
    }
    store.set_active("alpha").expect("set active");

    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}]", spec.id);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("alpha"))
        .bind_instance(TransportId::new("matrix/@b:hs"), ProfileRef::new("beta"));

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x55; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(store),
        provider_resolver: Some(resolver),
        credential_store: Some(Arc::new(MemCredentialStore::new())),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: Some(routing),
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
    });

    async fn drive_turn(node: &Arc<NodeApiImpl>, origin: Origin, req: u64) -> SessionId {
        let session = node
            .submit_routed(
                origin,
                AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(req),
                },
            )
            .await
            .expect("routed submit");
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finished = false;
        while Instant::now() < deadline && !finished {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if matches!(item, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                    finished = true;
                }
            }
            if !finished {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        assert!(finished, "routed turn never reached TurnFinished");
        session
    }

    let origin_a = Origin::new(
        TransportId::new("matrix/@a:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );
    let origin_b = Origin::new(
        TransportId::new("matrix/@b:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );
    let session_a = drive_turn(&node, origin_a.clone(), 1).await;
    let session_b = drive_turn(&node, origin_b.clone(), 2).await;

    // 1. Owned-session discovery is scoped to the instance's Primary (a single wire page here).
    let owned_a = node
        .delivery_sessions(TransportId::new("matrix/@a:hs"), None)
        .await;
    assert_eq!(
        owned_a.items,
        vec![session_a.clone()],
        "@a:hs owns exactly session_a"
    );
    assert_eq!(owned_a.next, None, "one owned session fits one page");
    let owned_b = node
        .delivery_sessions(TransportId::new("matrix/@b:hs"), None)
        .await;
    assert_eq!(
        owned_b.items,
        vec![session_b.clone()],
        "@b:hs owns exactly session_b"
    );

    // 2. The reusable pull subscriber discovers + projects @a:hs's owned session.
    let recorder = Arc::new(Recorder::default());
    let api: Arc<dyn daemon_api::NodeApi> = node.clone();
    let sub = serve_delivery(api, TransportId::new("matrix/@a:hs"), recorder.clone()).await;
    assert_eq!(sub.len(), 1, "exactly one owned session under delivery");

    // Wait until the backfilled history (incl. the first turn's TurnFinished) is projected.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !recorder.has_turn_finished() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        recorder.has_turn_finished(),
        "the pull subscriber projected the owned session's TurnFinished"
    );
    assert!(
        recorder
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|(s, _)| s == &session_a),
        "the subscription only projects the owned session"
    );

    // 3. Hand session_a over to a GUI; @a:hs is demoted, so it no longer owns the session, and a
    // subsequent live event makes the subscription halt (still-owns re-check fails).
    let gui = DeliveryTarget::new("gui", "panel-1", SinkKind::Primary);
    node.handover(session_a.clone(), gui)
        .await
        .expect("handover");
    assert!(
        node.delivery_sessions(TransportId::new("matrix/@a:hs"), None)
            .await
            .items
            .is_empty(),
        "after handover @a:hs owns no sessions"
    );
    // Drive another turn to push a live entry through the (now demoted) subscription.
    let _ = node
        .submit_from(
            session_a.clone(),
            origin_a.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("again"),
                request_id: ReqId(3),
            },
        )
        .await;
    // The subscription's per-session task must end (halt-on-demotion); bound the wait.
    let halted = tokio::time::timeout(Duration::from_secs(10), sub.join())
        .await
        .is_ok();
    assert!(halted, "the pull subscription halts once handed over");

    // The demoted subscription never projected for any session other than session_a.
    assert!(
        recorder
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|(s, _)| s == &session_a),
        "no foreign-session entries projected"
    );

    handle.shutdown().await;
}

/// FOUNDATION: profile-scoped §11 memory under per-room routing. M1 made provider/persona/tools
/// profile-aware per session, but §10 context and §11 memory were wired once from the launch
/// profile's home — so two rooms routed to two profiles shared one bank. This proves the resolved
/// `ProfileRef` now threads all the way into memory construction: routing two accounts to two
/// profiles opens two banks under distinct `<data_dir>/<profile>/` homes on disk, while a
/// profile-less (legacy) engine resolves the shared default home (the pre-routing behavior).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routed_profiles_get_isolated_memory_banks() {
    use daemon_api::{Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_core::{EngineProfile, MemoryBuilder, MemoryProvider, SystemPrompt, ToolRegistry};
    use daemon_host::{MemCredentialStore, MemProfileStore, ProfileStore, RoutingRegistry};
    use daemon_protocol::{AgentCommand, AgentEvent, Origin, OriginScope, TransportId, UserMsg};
    use std::sync::Mutex;

    // A recording memory builder: each construction roots a real per-profile bank dir under a
    // tmp root (the on-disk isolation we assert) and records the (profile, session, dir) it saw.
    let root = std::env::temp_dir().join(format!("daemon-mc-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    type Calls = Arc<Mutex<Vec<(Option<String>, String, std::path::PathBuf)>>>;
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let make_builder = |calls: Calls, root: std::path::PathBuf| -> MemoryBuilder {
        Arc::new(move |profile: Option<&ProfileRef>, session: &SessionId| {
            let pname = profile.map(|p| p.as_str().to_string());
            let dir = root.join(pname.clone().unwrap_or_else(|| "default".to_string()));
            std::fs::create_dir_all(&dir).expect("create per-profile bank dir");
            calls
                .lock()
                .unwrap()
                .push((pname, session.as_str().to_string(), dir));
            Vec::<Arc<dyn MemoryProvider>>::new()
        })
    };

    // Two accounts bound to two profiles.
    let store = Arc::new(MemProfileStore::new());
    for (id, model) in [("alpha", "model-a"), ("beta", "model-b")] {
        let mut spec = ProfileSpec::new(id, ProviderSelector::GenAi, model);
        spec.system_prompt = format!("You are {id}.");
        store.create(spec).expect("create profile");
    }
    store.set_active("alpha").expect("set active");

    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}]", spec.id);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("alpha"))
        .bind_instance(TransportId::new("matrix/@b:hs"), ProfileRef::new("beta"));

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x55; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: Some(make_builder(calls.clone(), root.clone())),
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(store),
        provider_resolver: Some(resolver),
        credential_store: Some(Arc::new(MemCredentialStore::new())),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: Some(routing),
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
    });

    async fn route(node: &Arc<NodeApiImpl>, origin: Origin) {
        let session = node
            .submit_routed(
                origin,
                AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(1),
                },
            )
            .await
            .expect("routed submit");
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finished = false;
        while Instant::now() < deadline && !finished {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(AgentEvent::TurnFinished { .. }) = item {
                    finished = true;
                }
            }
            if !finished {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        assert!(finished, "routed turn never reached TurnFinished");
    }

    let origin_a = Origin::new(
        TransportId::new("matrix/@a:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );
    let origin_b = Origin::new(
        TransportId::new("matrix/@b:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );
    route(&node, origin_a).await;
    route(&node, origin_b).await;

    let recorded = calls.lock().unwrap().clone();
    let alpha_dir = root.join("alpha");
    let beta_dir = root.join("beta");
    assert!(
        recorded
            .iter()
            .any(|(p, _, d)| p.as_deref() == Some("alpha") && d == &alpha_dir),
        "the alpha-routed session built its memory under its own home: {recorded:?}"
    );
    assert!(
        recorded
            .iter()
            .any(|(p, _, d)| p.as_deref() == Some("beta") && d == &beta_dir),
        "the beta-routed session built its memory under its own home: {recorded:?}"
    );
    assert!(
        alpha_dir.is_dir() && beta_dir.is_dir(),
        "both per-profile bank dirs exist on disk"
    );
    assert_ne!(
        alpha_dir, beta_dir,
        "two routed profiles -> two isolated banks"
    );

    handle.shutdown().await;

    // None/legacy: an `EngineProfile` with no profile ref resolves the builder with `None`, so
    // two such engines share the default home (the pre-routing single-profile behavior).
    let legacy = EngineProfile::new(
        Arc::new(|| Arc::new(MockProvider::completing("ok")) as Arc<dyn Provider>),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("legacy"),
    )
    .with_memory_builder(make_builder(calls.clone(), root.clone()));
    let _ = legacy.fresh(SessionId::new("s1"));
    let _ = legacy.fresh(SessionId::new("s2"));
    let default_dir = root.join("default");
    let legacy_calls: Vec<_> = calls
        .lock()
        .unwrap()
        .iter()
        .filter(|(p, _, _)| p.is_none())
        .cloned()
        .collect();
    assert_eq!(legacy_calls.len(), 2, "two legacy engines built memory");
    assert!(
        legacy_calls.iter().all(|(_, _, d)| d == &default_dir),
        "profile-less engines share the default home: {legacy_calls:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}
