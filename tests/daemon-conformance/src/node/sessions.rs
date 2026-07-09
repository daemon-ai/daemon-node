// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_surface_is_transport_agnostic_and_drives_a_session_to_completion() {
    as_system(control_surface_is_transport_agnostic_and_drives_a_session_to_completion_impl())
        .await;
}
async fn control_surface_is_transport_agnostic_and_drives_a_session_to_completion_impl() {
    let (node, handle) = assemble();

    // Serve the same surface over a Unix socket.
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Health over the socket: the four resident services are present.
    let health = match client.call(ApiRequest::Health).await.unwrap() {
        ApiResponse::Health(h) => h,
        other => panic!("expected Health, got {other:?}"),
    };
    assert!(
        health.services.len() >= 4,
        "expected the resident-service tree, got {:?}",
        health.services
    );

    // Assign a durable session over the socket and drive it to Completed via the real fleet
    // job worker (the resident JobOutboxDispatcher), polling the control surface.
    let session = SessionId::new("op-session");
    assert!(matches!(
        client
            .call(ApiRequest::Assign {
                session: session.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let resp = client.call(ApiRequest::Sessions).await.unwrap();
        if let ApiResponse::Sessions(list) = resp {
            if list
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed)
            {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the assigned session never reached Completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // A delegation child ran: fleet usage folded in (the §7 fan-in).
    let fleet = match client.call(ApiRequest::Fleet).await.unwrap() {
        ApiResponse::Fleet(f) => f,
        other => panic!("expected Fleet, got {other:?}"),
    };
    assert!(
        fleet.usage.api_calls > 0 && !fleet.children.is_empty(),
        "expected a delegation child to have run and folded usage, got {fleet:?}"
    );

    // Transport parity: the in-process trait call and the socket round-trip agree.
    let inproc_health = node.health().await;
    let socket_health = match client.call(ApiRequest::Health).await.unwrap() {
        ApiResponse::Health(h) => h,
        other => panic!("expected Health, got {other:?}"),
    };
    assert_eq!(
        inproc_health.all_ok, socket_health.all_ok,
        "health all_ok must agree across transports"
    );
    assert_eq!(
        sorted_names(&inproc_health),
        sorted_names(&socket_health),
        "the service set must agree across transports"
    );

    let inproc_sessions = node.sessions().await;
    let socket_sessions = match client.call(ApiRequest::Sessions).await.unwrap() {
        ApiResponse::Sessions(list) => list,
        other => panic!("expected Sessions, got {other:?}"),
    };
    assert!(
        inproc_sessions
            .iter()
            .any(|i| i.session == session && i.state == SessionState::Completed)
            && socket_sessions
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed),
        "both transports must observe the completed session"
    );

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

fn sorted_names(h: &daemon_api::HealthReport) -> Vec<String> {
    let mut names: Vec<String> = h.services.iter().map(|s| s.name.clone()).collect();
    names.sort();
    names
}

/// Session-action ops (Phase-3 A): `session_update_meta` is a durable read-modify-write of the
/// roster metadata that rename/pin/archive ride. Proves: (a) a rename surfaces on the roster
/// line; (b) a pinned conversation sorts *first* in `TopLevel` (ahead of activity order); (c) an
/// archived conversation drops out of `TopLevel` and surfaces only under `Archived`; (d) the
/// patch op round-trips over the socket (`ApiResponse::Ok`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_meta_rename_pin_archive_round_trip() {
    as_system(session_meta_rename_pin_archive_round_trip_impl()).await;
}
async fn session_meta_rename_pin_archive_round_trip_impl() {
    use daemon_api::{SessionApi, SessionMetaPatch, SessionQuery, SessionScope};
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Three live top-level conversations (opened oldest-first so activity order is a, b, c).
    let ids: Vec<SessionId> = (0..3).map(|n| SessionId::new(format!("act-{n}"))).collect();
    for id in &ids {
        as_system(node.submit(
            id.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
        ))
        .await
        .expect("submit opens a live session");
    }

    // (a) Rename act-0 over the socket; the new title surfaces on its roster line.
    assert!(matches!(
        client
            .call(ApiRequest::SessionUpdateMeta {
                session: ids[0].clone(),
                patch: SessionMetaPatch {
                    title: Some(Some("renamed".into())),
                    ..Default::default()
                },
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let line_title = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            ..Default::default()
        })
        .await
        .sessions
        .into_iter()
        .find(|i| i.session == ids[0])
        .and_then(|i| i.title);
    assert_eq!(
        line_title.as_deref(),
        Some("renamed"),
        "the rename must surface on the roster line"
    );

    // (b) Pin act-0 (the oldest); it must now sort first in TopLevel despite being least-recent.
    node.session_update_meta(
        ids[0].clone(),
        SessionMetaPatch {
            pinned: Some(true),
            ..Default::default()
        },
    )
    .await
    .expect("pin act-0");
    let top = node.sessions_query(SessionQuery::default()).await.sessions;
    assert_eq!(
        top.first().map(|i| &i.session),
        Some(&ids[0]),
        "a pinned conversation must sort first, got {top:?}"
    );
    assert!(
        top.first().map(|i| i.pinned).unwrap_or(false),
        "the first line carries the pinned flag"
    );

    // (c) Archive act-1; it leaves TopLevel and appears only under the Archived scope.
    node.session_update_meta(
        ids[1].clone(),
        SessionMetaPatch {
            archived: Some(true),
            ..Default::default()
        },
    )
    .await
    .expect("archive act-1");
    let top = node.sessions_query(SessionQuery::default()).await.sessions;
    assert!(
        !top.iter().any(|i| i.session == ids[1]),
        "an archived conversation must drop out of TopLevel"
    );
    let archived = node
        .sessions_query(SessionQuery {
            scope: SessionScope::Archived,
            ..Default::default()
        })
        .await
        .sessions;
    assert!(
        archived.iter().any(|i| i.session == ids[1] && i.archived),
        "the archived conversation must surface under the Archived scope"
    );

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_surface_runs_an_interactive_turn_to_finished() {
    as_system(session_surface_runs_an_interactive_turn_to_finished_impl()).await;
}
async fn session_surface_runs_an_interactive_turn_to_finished_impl() {
    use daemon_api::{Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};

    let (node, handle) = assemble();
    let session = SessionId::new("live-1");

    // Open + run a turn on the live session sub-surface (the same surface the FFI wraps).
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit StartTurn");

    // Drain events until TurnFinished arrives.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finished = false;
    while Instant::now() < deadline {
        let drained = node.poll(session.clone(), 0).await.expect("poll");
        if drained
            .iter()
            .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
        {
            finished = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(finished, "the interactive turn never reached TurnFinished");

    handle.shutdown().await;
}

/// THE PHASE 0 GUI-READINESS DEMO GATE: over a single Unix socket, a scripted client walks the
/// whole GUI bring-up flow end to end — set an Anthropic key, create + select a
/// `claude-opus-4-8` profile, list discoverable models, confirm the current model, then open an
/// interactive session and chat — and observes the streamed usage + context-fill + turn events.
/// This is the demo gate that says "the GUI can be built against this surface."
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phase0_gui_readiness_demo_gate() {
    use daemon_api::{ApiRequest, ApiResponse, Outbound, ProfileSpec, ProviderSelector};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};

    let (node, handle) = assemble_demo();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // 1. Set the provider API key for the "opus" profile, then confirm it lists (redacted).
    assert!(matches!(
        client
            .call(ApiRequest::CredentialSet {
                profile: "opus".into(),
                secret: "sk-ant-demo-abcd1234".into(),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    match client.call(ApiRequest::CredentialList).await.unwrap() {
        ApiResponse::Credentials(creds) => {
            let opus = creds
                .iter()
                .find(|c| c.profile == "opus")
                .expect("opus credential");
            assert!(opus.present, "the set credential should report present");
            assert_eq!(opus.hint, "…1234", "the listing is redacted to a tail hint");
            assert!(!opus.hint.contains("abcd"), "the secret is never returned");
        }
        other => panic!("expected Credentials, got {other:?}"),
    }

    // 2. Create the genai/claude-opus-4-8 profile and make it the active default (the genai
    // adapter is inferred from the model id — the daemon keeps no per-provider selector).
    let spec = ProfileSpec::new("opus", ProviderSelector::GenAi, "claude-opus-4-8");
    assert!(matches!(
        client
            .call(ApiRequest::ProfileCreate { spec })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::ProfileSelect { id: "opus".into() })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    match client.call(ApiRequest::ProfileList).await.unwrap() {
        ApiResponse::Profiles(list) => {
            let opus = list.iter().find(|p| p.id == "opus").expect("opus profile");
            assert!(opus.is_active, "opus should be the active default");
            assert_eq!(opus.provider, ProviderSelector::GenAi);
        }
        other => panic!("expected Profiles, got {other:?}"),
    }

    // 3. The model picker can discover claude-opus-4-8 (the static cloud catalog), walking the
    // wire pages (v25: Models is cursor-paged in descriptor-id order).
    {
        let mut all = Vec::new();
        let mut after: Option<String> = None;
        loop {
            match client
                .call(ApiRequest::Models {
                    after: after.take(),
                })
                .await
                .unwrap()
            {
                ApiResponse::Models(page) => {
                    all.extend(page.items);
                    match page.next {
                        Some(next) => after = Some(next),
                        None => break,
                    }
                }
                other => panic!("expected Models, got {other:?}"),
            }
        }
        let opus = all
            .iter()
            .find(|m| m.id == "claude-opus-4-8")
            .expect("claude-opus-4-8 in the catalog");
        assert_eq!(opus.provider, ProviderSelector::GenAi);
        assert_eq!(opus.context_length, Some(200_000));
    }

    // 4. The current model resolves to the active profile's opus.
    match client
        .call(ApiRequest::ModelCurrent { profile: None })
        .await
        .unwrap()
    {
        ApiResponse::ModelCurrent(Some(m)) => {
            assert_eq!(m.id, "claude-opus-4-8");
            assert_eq!(m.context_length, Some(200_000));
        }
        other => panic!("expected ModelCurrent(Some), got {other:?}"),
    }

    // 5. Open an interactive session and chat (the engine is built from the active opus profile).
    let session = SessionId::new("demo-1");
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello opus"),
                    request_id: ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // 6. Drain the stream: assert we observe a context-fill update, a turn finish carrying usage,
    //    and that the reply came from the resolved opus profile (proving per-session resolution).
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut saw_context = false;
    let mut finished = false;
    let mut final_text = String::new();
    while Instant::now() < deadline && !finished {
        let drained = match client
            .call(ApiRequest::Poll {
                session: session.clone(),
                max: 0,
            })
            .await
            .unwrap()
        {
            ApiResponse::Drained(items) => items,
            other => panic!("expected Drained, got {other:?}"),
        };
        for item in drained {
            if let Outbound::Event(event) = item {
                match event {
                    AgentEvent::Context { status, .. } => {
                        saw_context = true;
                        // The mock declares an 8k window; the HUD denominator flows through.
                        assert_eq!(status.max_tokens, Some(8192));
                    }
                    AgentEvent::TurnFinished { summary, .. } => {
                        finished = true;
                        final_text = summary.final_text.unwrap_or_default();
                    }
                    _ => {}
                }
            }
        }
        if !finished {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
    assert!(saw_context, "the turn never emitted a context-fill event");
    assert!(finished, "the interactive turn never reached TurnFinished");
    assert!(
        final_text.contains("[opus]") && final_text.contains("claude-opus-4-8"),
        "the reply should come from the resolved opus profile, got {final_text:?}"
    );

    server.abort();
    handle.shutdown().await;
}

/// PROFILES + SESSION OVERLAY: a per-session model override is **persisted** on the session's
/// `SessionOverlay` (host-level metadata) and **restored** when the live actor is respawned —
/// the unified resolution path means the engine is rebuilt from `bound profile + overlay`, not
/// from the bare profile. We drive it through the public `SetSessionModel`, observe the persisted
/// overlay in the store, then shut the live actor down and reopen the same routed session and
/// observe that the provider is resolved for the *overridden* model (the restore), not the
/// profile's default — proving the override survives a (live) respawn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_overlay_persists_and_restores_on_respawn() {
    as_system(session_overlay_persists_and_restores_on_respawn_impl()).await;
}
async fn session_overlay_persists_and_restores_on_respawn_impl() {
    use daemon_api::{Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_host::{
        decode_overlay, MemCredentialStore, MemProfileStore, ProfileStore, RoutingRegistry,
    };
    use daemon_protocol::{AgentCommand, AgentEvent, Origin, OriginScope, TransportId, UserMsg};
    use std::sync::Mutex;

    // A resolver that records every model id it is asked to build a provider for — our window
    // into which (profile, overlay)-resolved model each engine construction saw.
    type Seen = Arc<Mutex<Vec<String>>>;
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    let resolver: daemon_node::ProviderResolver = Arc::new(move |spec: &ProfileSpec| {
        seen2.lock().unwrap().push(spec.model.clone());
        let reply = spec.model.clone();
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    let pstore = Arc::new(MemProfileStore::new());
    pstore
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create profile");
    pstore.set_active("alpha").expect("set active");

    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("alpha"));

    let store = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: store.clone(),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x66; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(pstore),
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
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });

    let origin = Origin::new(
        TransportId::new("matrix/@a:hs"),
        OriginScope::Group {
            chat: "#general".into(),
            thread: None,
        },
    );

    // Open the session (binds it to `alpha`; builds its engine from the bare profile -> model-a).
    let session = node
        .submit_routed(
            origin.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("routed submit opens the session");
    assert_eq!(
        seen.lock().unwrap().first().map(String::as_str),
        Some("model-a"),
        "the first engine build resolves the profile's own model"
    );

    // Override the model for this session. This persists the overlay AND swaps the live provider.
    node.set_session_model(session.clone(), "model-x".to_string(), None)
        .await
        .expect("set_session_model");

    // The override is durably recorded as host-level session metadata (bound profile + overlay).
    let meta = store
        .session_meta(&session)
        .await
        .expect("session meta recorded");
    assert_eq!(
        meta.bound_profile.as_ref().map(|p| p.as_str()),
        Some("alpha"),
        "the session's bound profile is recorded"
    );
    let overlay = decode_overlay(&meta.overlay);
    assert_eq!(
        overlay.model.as_deref(),
        Some("model-x"),
        "the model override is persisted on the overlay"
    );

    // Tear the live actor down, then reopen the same routed session: `ensure` reads the persisted
    // overlay and rebuilds the engine from `alpha + {model: model-x}` — the restore.
    node.submit(session.clone(), AgentCommand::Shutdown)
        .await
        .expect("shutdown the live actor");
    let reopened = node
        .submit_routed(
            origin,
            AgentCommand::StartTurn {
                input: UserMsg::new("again"),
                request_id: ReqId(2),
            },
        )
        .await
        .expect("reopen the routed session");
    assert_eq!(
        reopened, session,
        "the same origin resolves the same session"
    );

    // Drive the reopened turn to completion so we know the rebuild happened.
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
    assert!(finished, "the reopened turn ran to completion");

    let recorded = seen.lock().unwrap().clone();
    assert_eq!(
        recorded.last().map(String::as_str),
        Some("model-x"),
        "the respawned engine resolved the *restored* overridden model, not the profile default: {recorded:?}"
    );
    assert!(
        recorded.iter().filter(|m| m.as_str() == "model-x").count() >= 2,
        "model-x was resolved both at override time and again on respawn: {recorded:?}"
    );

    handle.shutdown().await;
}

/// Steer / Snapshot / Interrupt drive over the Unix socket, and the snapshot projection agrees
/// with the in-process transport (the phase-9 control-surface parity gate).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steer_snapshot_interrupt_drive_over_socket_with_parity() {
    as_system(steer_snapshot_interrupt_drive_over_socket_with_parity_impl()).await;
}
async fn steer_snapshot_interrupt_drive_over_socket_with_parity_impl() {
    use daemon_api::{Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, ConvView, UserMsg};

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Drain the socket until `pred` matches one of the outbound items; returns all drained.
    async fn drain_socket_until(
        client: &ApiClient,
        session: &SessionId,
        pred: impl Fn(&Outbound) -> bool,
    ) -> Vec<Outbound> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = Vec::new();
        loop {
            match client
                .call(ApiRequest::Poll {
                    session: session.clone(),
                    max: 0,
                })
                .await
                .unwrap()
            {
                ApiResponse::Drained(v) => {
                    let hit = v.iter().any(&pred);
                    seen.extend(v);
                    if hit {
                        return seen;
                    }
                }
                other => panic!("expected Drained, got {other:?}"),
            }
            assert!(Instant::now() < deadline, "socket drain never matched");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn find_snapshot(items: &[Outbound], request_id: ReqId) -> Option<ConvView> {
        items.iter().find_map(|o| match o {
            Outbound::Event(AgentEvent::Snapshot {
                request_id: id,
                view,
                ..
            }) if *id == request_id => Some(view.clone()),
            _ => None,
        })
    }

    // --- socket transport: StartTurn -> Snapshot -> Steer -> Interrupt ---
    let socket_session = SessionId::new("socket-live");
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: socket_session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello there"),
                    request_id: ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    drain_socket_until(&client, &socket_session, |o| {
        matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. }))
    })
    .await;

    // Snapshot over the socket.
    client
        .call(ApiRequest::Submit {
            session: socket_session.clone(),
            command: AgentCommand::Snapshot {
                request_id: ReqId(2),
            },
            origin: None,
            profile: None,
        })
        .await
        .unwrap();
    let socket_items = drain_socket_until(&client, &socket_session, |o| {
        matches!(o, Outbound::Event(AgentEvent::Snapshot { request_id, .. }) if *request_id == ReqId(2))
    })
    .await;
    let socket_view = find_snapshot(&socket_items, ReqId(2)).expect("a snapshot view");
    assert!(socket_view
        .turns
        .iter()
        .any(|t| t.role == "user" && t.text == "hello there"));

    // Steer over the socket: acked via a Steered event.
    client
        .call(ApiRequest::Submit {
            session: socket_session.clone(),
            command: AgentCommand::Steer {
                text: "stay focused".into(),
                request_id: ReqId(3),
            },
            origin: None,
            profile: None,
        })
        .await
        .unwrap();
    drain_socket_until(&client, &socket_session, |o| {
        matches!(o, Outbound::Event(AgentEvent::Steered { request_id, accepted, .. }) if *request_id == ReqId(3) && *accepted)
    })
    .await;

    // Interrupt over the socket flows through and is accepted.
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: socket_session.clone(),
                command: AgentCommand::Interrupt {
                    reason: Some("stop".into()),
                },
                origin: None,
                profile: None,
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // --- in-process parity: the same StartTurn + Snapshot yields the same view shape ---
    let inproc_session = SessionId::new("inproc-live");
    node.submit(
        inproc_session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello there"),
            request_id: ReqId(1),
        },
    )
    .await
    .unwrap();
    let inproc_deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let drained = node.poll(inproc_session.clone(), 0).await.unwrap();
        if drained
            .iter()
            .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
        {
            break;
        }
        assert!(
            Instant::now() < inproc_deadline,
            "in-proc turn never finished"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    node.submit(
        inproc_session.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(2),
        },
    )
    .await
    .unwrap();
    let mut inproc_items = Vec::new();
    let snap_deadline = Instant::now() + Duration::from_secs(10);
    let inproc_view = loop {
        inproc_items.extend(node.poll(inproc_session.clone(), 0).await.unwrap());
        if let Some(view) = find_snapshot(&inproc_items, ReqId(2)) {
            break view;
        }
        assert!(
            Instant::now() < snap_deadline,
            "in-proc snapshot never arrived"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    assert_eq!(
        socket_view.turns, inproc_view.turns,
        "the snapshot projection must agree across transports"
    );

    // --- §5.4 delivery targets + handover, and the transport/meta lever ---
    {
        use daemon_protocol::{
            DeliveryTarget, Disposition, Origin, OriginScope, SessionPayload, SinkKind, TransportId,
        };

        // Opening the in-proc session via `submit` (the generic `api` origin) seeded a single
        // Primary reply sink.
        let seeded = node.delivery_targets(inproc_session.clone()).await;
        assert_eq!(seeded.len(), 1);
        assert_eq!(seeded[0].kind, SinkKind::Primary);

        // Handover re-points the Primary to a chat target; the prior Primary is demoted.
        node.handover(
            inproc_session.clone(),
            DeliveryTarget::new("telegram", "chat-42", SinkKind::Primary),
        )
        .await
        .unwrap();
        let after = node.delivery_targets(inproc_session.clone()).await;
        let primaries: Vec<_> = after
            .iter()
            .filter(|t| t.kind == SinkKind::Primary)
            .collect();
        assert_eq!(primaries.len(), 1, "exactly one Primary in force");
        assert_eq!(primaries[0].transport, TransportId::new("telegram"));
        assert_eq!(primaries[0].route.as_str(), "chat-42");
        assert!(
            after.iter().any(|t| t.kind == SinkKind::Spectator),
            "the prior Primary is demoted to Spectator"
        );

        // record_meta lands on the live merged log as a Transport entry (observable), without
        // entering the prompt/journal.
        let before = node.log_after(inproc_session.clone(), 0, 0).await.unwrap();
        node.record_meta(daemon_api::RecordMetaArgs {
            session: inproc_session.clone(),
            origin: Origin::new(
                "gui",
                OriginScope::Api {
                    key: "owner".into(),
                },
            ),
            kind: "attach".into(),
            body: vec![1, 2, 3],
        })
        .await
        .unwrap();
        let delta = node
            .log_after(inproc_session.clone(), before.head_seq, 0)
            .await
            .unwrap();
        let meta = delta
            .entries
            .iter()
            .find(|e| matches!(&e.payload, SessionPayload::Meta { .. }))
            .expect("the meta event is observable on the live log");
        assert_eq!(meta.disposition, Disposition::Transport);
    }

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}
