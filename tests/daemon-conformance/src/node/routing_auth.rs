// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// Routing-pin resolution (Phase-2 B1): a durable chat→session pin is consulted *first* in
/// `resolve()`, so a routed submit lands on the pinned session id (overriding the deterministic
/// naming). The pin round-trips through `routing_get`, surfaces as a `transport_rooms` room, and
/// `routing_unbind_chat` clears it — all without a restart (the hot-reload seam).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routing_pin_resolves_to_bound_session() {
    as_system(routing_pin_resolves_to_bound_session_impl()).await;
}
async fn routing_pin_resolves_to_bound_session_impl() {
    use daemon_api::SessionApi;
    use daemon_protocol::{AgentCommand, Origin, OriginScope, TransportId, UserMsg};

    let (node, handle) = assemble();
    let origin = Origin::new(
        "telegram",
        OriginScope::Dm {
            user: "alice".into(),
        },
    );
    let pinned = SessionId::new("pinned-chat");

    node.routing_bind_chat(origin.clone(), pinned.clone(), None)
        .await
        .expect("bind a chat→session pin");

    // The pin round-trips through the durable store.
    let got = node
        .routing_get(origin.clone())
        .await
        .expect("a pinned route");
    assert_eq!(
        got.session, pinned,
        "routing_get returns the pinned session"
    );

    // Resolve-first: a routed submit lands on the pinned session id.
    let resolved = node
        .submit_routed(
            origin.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
        )
        .await
        .expect("routed submit resolves through the pin");
    assert_eq!(
        resolved, pinned,
        "the pin must override the deterministic session naming"
    );

    // The pin surfaces as a room of its transport family.
    let rooms = node
        .transport_rooms(TransportId::new("telegram"), None)
        .await
        .items;
    assert!(
        rooms.iter().any(|r| r.session.as_ref() == Some(&pinned)),
        "the pinned chat must enumerate as a transport room, got {rooms:?}"
    );

    // Unbind clears the pin (hot-reload): the origin falls back to deterministic naming.
    node.routing_unbind_chat(origin.clone())
        .await
        .expect("unbind the pin");
    assert!(
        node.routing_get(origin.clone()).await.is_none(),
        "the pin must be gone after unbind"
    );

    handle.shutdown().await;
}

/// Per-turn transport hints, end to end (prompt-arch): the per-surface formatting hint is injected
/// on a turn keyed on **that submit's** origin — never the session. Two routed submits land on the
/// SAME pinned session: the Matrix-origin turn carries the Matrix hint in its outgoing request's
/// last user message, while the `api`-origin turn to the same session carries none. The hint is
/// request-only (`TurnInjection`): it never enters the durable conversation, so the session's
/// snapshot stays clean of hint bytes across both turns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_turn_transport_hint_keys_on_the_submit_not_the_session() {
    as_system(per_turn_transport_hint_keys_on_the_submit_not_the_session_impl()).await;
}
async fn per_turn_transport_hint_keys_on_the_submit_not_the_session_impl() {
    use daemon_api::{Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_host::{MemProfileStore, ProfileStore, RoutingRegistry};
    use daemon_protocol::{AgentCommand, AgentEvent, Origin, OriginScope, TransportId, UserMsg};
    use std::sync::Mutex as StdMutex;

    /// Records each request's last user message (where a `TurnInjection` rides), then completes.
    struct LastUserRecorder {
        seen: Arc<StdMutex<Vec<String>>>,
    }
    #[async_trait::async_trait]
    impl Provider for LastUserRecorder {
        fn capabilities(&self) -> daemon_core::Capabilities {
            daemon_core::Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: daemon_core::ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(
            &self,
            req: daemon_core::Request,
        ) -> Result<daemon_core::ModelOutput, daemon_core::Failure> {
            let last_user = req
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            self.seen.lock().unwrap().push(last_user);
            Ok(daemon_core::ModelOutput {
                text: "ok".into(),
                ..Default::default()
            })
        }
    }

    let store = Arc::new(MemProfileStore::new());
    store
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create profile");
    store.set_active("alpha").expect("set active");

    let seen: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let resolver: daemon_node::ProviderResolver = {
        let seen = seen.clone();
        Arc::new(move |_spec: &ProfileSpec| {
            let seen = seen.clone();
            let builder: daemon_core::ProviderBuilder = Arc::new(move || {
                Arc::new(LastUserRecorder { seen: seen.clone() }) as Arc<dyn Provider>
            });
            builder
        })
    };

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("alpha"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x57; 32]),
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
        routing: Some(RoutingRegistry::new()),
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
        // `[prompt].transport_hints` defaults on: the origin-aware TransportHintSource is attached.
        prompt: Default::default(),
    });

    // Both origins pin to the SAME session, so the second submit reuses the first's live engine —
    // proving the hint follows the per-turn submit origin, not a session-stable property.
    let shared = SessionId::new("shared-surface");
    let matrix_origin = Origin::new(
        TransportId::new("matrix/@bot:hs.org"),
        OriginScope::Group {
            chat: "!room:hs".into(),
            thread: None,
        },
    );
    let api_origin = Origin::new(
        TransportId::new("api"),
        OriginScope::Dm {
            user: "alice".into(),
        },
    );
    node.routing_bind_chat(matrix_origin.clone(), shared.clone(), None)
        .await
        .expect("pin the matrix origin");
    node.routing_bind_chat(api_origin.clone(), shared.clone(), None)
        .await
        .expect("pin the api origin to the same session");

    async fn route_and_finish(node: &Arc<NodeApiImpl>, origin: Origin) {
        node.submit_routed(
            origin,
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("routed submit");
    }
    async fn drain_until_finished(node: &Arc<NodeApiImpl>, session: &SessionId) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let finished = node
                .poll(session.clone(), 0)
                .await
                .expect("poll")
                .into_iter()
                .any(|item| matches!(item, Outbound::Event(AgentEvent::TurnFinished { .. })));
            if finished {
                return;
            }
            assert!(Instant::now() < deadline, "routed turn never finished");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    // Turn 1: a Matrix-origin submit — the outgoing request carries the Matrix surface hint.
    route_and_finish(&node, matrix_origin).await;
    drain_until_finished(&node, &shared).await;
    // Turn 2: an api-origin submit to the SAME session — no hint that turn.
    route_and_finish(&node, api_origin).await;
    drain_until_finished(&node, &shared).await;

    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "one request per turn, same session");
        assert!(
            seen[0].contains("Matrix room"),
            "turn 1 (matrix origin): the per-surface hint rides the request, got {:?}",
            seen[0]
        );
        assert!(
            !seen[1].contains("Matrix"),
            "turn 2 (api origin, same session): no hint — keyed on the submit, not the session, got {:?}",
            seen[1]
        );
    }

    // The durable conversation stays clean of hint bytes: the request-only injection never enters
    // the persisted turns (a snapshot of the shared session shows no surface hint on any user turn).
    node.submit(
        shared.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(99),
        },
    )
    .await
    .expect("snapshot request");
    let view = {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let snap = node
                .poll(shared.clone(), 0)
                .await
                .expect("poll")
                .into_iter()
                .find_map(|item| match item {
                    Outbound::Event(AgentEvent::Snapshot {
                        view,
                        request_id: ReqId(99),
                        ..
                    }) => Some(view),
                    _ => None,
                });
            if let Some(view) = snap {
                break view;
            }
            assert!(Instant::now() < deadline, "snapshot never arrived");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    assert!(
        view.turns.iter().all(|t| !t.text.contains("Matrix")),
        "the durable conversation must stay clean of hint bytes, got {:?}",
        view.turns
    );

    handle.shutdown().await;
}

/// THE ROUTING GATE (daemon-event-io-spec §5.9): a routed submit hands the host only an `Origin`
/// and the host's routing registry resolves it to a session + profile + delivery. Proves, with no
/// chat transport at all: (1) the account->profile baseline (two transport instances bound to two
/// profiles run two different agents), (2) the per-room override beating the instance default
/// (precedence), (3) the `Primary` is auto-seeded as the inverse of the opening origin so a reply
/// leaves the right account, and (4) `handover` demotes the prior `Primary` to `Spectator`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routed_submit_resolves_profile_and_delivery_per_origin() {
    as_system(routed_submit_resolves_profile_and_delivery_per_origin_impl()).await;
}
async fn routed_submit_resolves_profile_and_delivery_per_origin_impl() {
    use daemon_api::{Outbound, ProfileSpec, ProviderSelector, SessionApi};
    use daemon_common::ReqId;
    use daemon_host::{
        MemCredentialStore, MemProfileStore, OriginMatcher, ProfileStore, RoutingRegistry,
        ScopePattern, SessionBinding, TransportPattern,
    };
    use daemon_protocol::{
        AgentCommand, AgentEvent, DeliveryTarget, IsolationPolicy, Origin, OriginScope, SinkKind,
        TransportId, UserMsg,
    };

    // Three profiles, each echoing its id+model through the mock provider so the reply reveals
    // which agent ran the session.
    let store = Arc::new(MemProfileStore::new());
    for (id, model) in [
        ("alpha", "model-a"),
        ("beta", "model-b"),
        ("secops", "model-s"),
    ] {
        let spec = ProfileSpec::new(id, ProviderSelector::GenAi, model);
        store.create(spec).expect("create profile");
    }
    store.set_active("alpha").expect("set active");

    let resolver: daemon_node::ProviderResolver = Arc::new(|spec: &ProfileSpec| {
        let reply = format!("[{}] from {}", spec.id, spec.model);
        let builder: daemon_core::ProviderBuilder = Arc::new(move || {
            Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
        });
        builder
    });

    // Two accounts bound to two profiles (the baseline); a per-room override on account A's
    // #secops* rooms picks a third profile (precedence step 1 beats step 2).
    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("alpha"))
        .bind_instance(TransportId::new("matrix/@b:hs"), ProfileRef::new("beta"))
        .with_binding(
            SessionBinding::new(
                OriginMatcher {
                    transport: TransportPattern::Exact(TransportId::new("matrix/@a:hs")),
                    scope: ScopePattern::Group {
                        chat_glob: "#secops*".into(),
                    },
                },
                IsolationPolicy::PerChat,
            )
            .with_profile(ProfileRef::new("secops")),
        );

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
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });

    // Drive a routed submit for `origin` and return (resolved session, final text).
    async fn route_and_drain(node: &Arc<NodeApiImpl>, origin: Origin) -> (SessionId, String) {
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
        let mut final_text = String::new();
        let mut finished = false;
        while Instant::now() < deadline && !finished {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(AgentEvent::TurnFinished { summary, .. }) = item {
                    finished = true;
                    final_text = summary.final_text.unwrap_or_default();
                }
            }
            if !finished {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        assert!(finished, "routed turn never reached TurnFinished");
        (session, final_text)
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
    let origin_secops = Origin::new(
        TransportId::new("matrix/@a:hs"),
        OriginScope::Group {
            chat: "#secops-alerts".into(),
            thread: None,
        },
    );

    let (session_a, text_a) = route_and_drain(&node, origin_a.clone()).await;
    let (session_b, text_b) = route_and_drain(&node, origin_b.clone()).await;
    let (session_secops, text_secops) = route_and_drain(&node, origin_secops.clone()).await;

    // 1+2. Each origin ran the agent the registry selected (account baseline + room override).
    assert!(
        text_a.contains("[alpha]"),
        "account A -> alpha, got {text_a:?}"
    );
    assert!(
        text_b.contains("[beta]"),
        "account B -> beta, got {text_b:?}"
    );
    assert!(
        text_secops.contains("[secops]"),
        "account A #secops -> override profile, got {text_secops:?}"
    );
    assert_ne!(
        session_a, session_b,
        "distinct accounts -> distinct sessions"
    );
    assert_ne!(
        session_a, session_secops,
        "override room is its own session"
    );

    // 3. The Primary is the inverse of the opening origin (reply leaves the right account/room).
    let targets_a = node.delivery_targets(session_a.clone()).await;
    let primary_a = targets_a
        .iter()
        .find(|t| t.kind == SinkKind::Primary)
        .expect("session A has a Primary");
    assert_eq!(primary_a, &origin_a.primary_target());
    assert_eq!(primary_a.transport, TransportId::new("matrix/@a:hs"));

    // 4. Handover re-points the Primary; the prior matrix Primary is demoted to Spectator.
    let gui = DeliveryTarget::new("gui", "panel-1", SinkKind::Primary);
    node.handover(session_a.clone(), gui.clone())
        .await
        .expect("handover");
    let after = node.delivery_targets(session_a.clone()).await;
    let primaries: Vec<_> = after
        .iter()
        .filter(|t| t.kind == SinkKind::Primary)
        .collect();
    assert_eq!(primaries.len(), 1, "exactly one Primary after handover");
    assert_eq!(primaries[0].transport, TransportId::new("gui"));
    assert!(
        after
            .iter()
            .any(|t| t.transport == TransportId::new("matrix/@a:hs")
                && t.kind == SinkKind::Spectator),
        "the prior matrix Primary is demoted to Spectator, not dropped"
    );

    handle.shutdown().await;
}

/// FOUNDATION (account->profile binding, daemon-event-io-spec §5.9.4): a profile *declares* the
/// transport-instance accounts bound to it (`ProfileSpec.bound_accounts`), and the host derives
/// the routing registry's `instance_profiles` baseline (precedence step 2) from that profile
/// data — not a route-table column. Proves, with no chat transport: (1) two profiles' bound
/// accounts route their instances to the right agent with an EMPTY config routing table; (2) an
/// explicit config instance binding overrides the profile-derived one (operator wins); (3) the
/// `CredentialStore` is the system-of-record for the opaque account blob a binding names — it
/// lists back redacted, the secret never returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_accounts_derive_instance_profile_binding() {
    as_system(bound_accounts_derive_instance_profile_binding_impl()).await;
}
async fn bound_accounts_derive_instance_profile_binding_impl() {
    use daemon_api::{
        BoundAccount, CredentialApi, Outbound, ProfileSpec, ProviderSelector, SessionApi,
    };
    use daemon_common::ReqId;
    use daemon_host::{MemCredentialStore, MemProfileStore, ProfileStore, RoutingRegistry};
    use daemon_protocol::{AgentCommand, AgentEvent, Origin, OriginScope, TransportId, UserMsg};

    // An echoing resolver: the reply reveals which profile (agent) ran the session.
    fn echo_resolver() -> daemon_node::ProviderResolver {
        Arc::new(|spec: &ProfileSpec| {
            let reply = format!("[{}]", spec.id);
            let builder: daemon_core::ProviderBuilder = Arc::new(move || {
                Arc::new(MockProvider::completing(reply.clone())) as Arc<dyn Provider>
            });
            builder
        })
    }

    // Two profiles, each DECLARING its bound transport-instance account (+ the credential ref
    // naming where its opaque session blob lives). No config route table is constructed.
    fn profile_store() -> Arc<MemProfileStore> {
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
        store
    }

    async fn route_text(node: &Arc<NodeApiImpl>, origin: Origin) -> String {
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
        loop {
            for item in node.poll(session.clone(), 0).await.expect("poll") {
                if let Outbound::Event(AgentEvent::TurnFinished { summary, .. }) = item {
                    return summary.final_text.unwrap_or_default();
                }
            }
            assert!(Instant::now() < deadline, "routed turn never finished");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    let origin = |account: &str| {
        Origin::new(
            TransportId::new(format!("matrix/{account}")),
            OriginScope::Group {
                chat: "#general".into(),
                thread: None,
            },
        )
    };

    // 1. Derive instance->profile purely from profile data, with an EMPTY config routing table.
    let creds = Arc::new(MemCredentialStore::new());
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
        profiles: Some(profile_store()),
        provider_resolver: Some(echo_resolver()),
        credential_store: Some(creds),
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
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });

    let text_a = route_text(&node, origin("@a:hs")).await;
    let text_b = route_text(&node, origin("@b:hs")).await;
    assert!(
        text_a.contains("[alpha]"),
        "@a:hs derived from alpha.bound_accounts, got {text_a:?}"
    );
    assert!(
        text_b.contains("[beta]"),
        "@b:hs derived from beta.bound_accounts, got {text_b:?}"
    );

    // 3. The CredentialStore is the system-of-record for the opaque account blob the binding
    // names: set it under the credential ref and confirm it lists back redacted.
    node.credential_set("matrix/alpha/a".into(), "mxsession-secret-blob-7f3c".into())
        .await
        .expect("store the opaque account session blob");
    let listed = node.credential_list().await;
    let acct = listed
        .iter()
        .find(|c| c.profile == "matrix/alpha/a")
        .expect("the account blob is listed under its credential ref");
    assert!(acct.present, "the stored account blob reports present");
    assert_eq!(
        acct.hint, "…7f3c",
        "the account blob is redacted to a tail hint, never returned"
    );

    handle.shutdown().await;

    // 2. An explicit config instance binding overrides the profile-derived one (operator wins):
    // `bind_instance(@a:hs -> beta)` beats `alpha.bound_accounts` for that instance.
    let routing = RoutingRegistry::new()
        .bind_instance(TransportId::new("matrix/@a:hs"), ProfileRef::new("beta"));
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
        profiles: Some(profile_store()),
        provider_resolver: Some(echo_resolver()),
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
    let text_override = route_text(&node, origin("@a:hs")).await;
    assert!(
        text_override.contains("[beta]"),
        "config bind_instance(@a:hs -> beta) wins over profile-derived alpha, got {text_override:?}"
    );
    handle.shutdown().await;
}

/// GENERIC INTERACTIVE-AUTH (daemon-interactive-auth-spec, the family-agnostic `AuthApi` seam): a
/// stub factory (standing in for a real SSO/OAuth2 family — no browser, no network) proves the
/// whole client-driven login orchestration through the node surface:
/// (1) `auth_providers` lists the registered family for client-side discovery;
/// (2) `auth_begin` parks a flow and returns the authorization URL minted against the
///     *client-supplied* `redirect_uri`;
/// (3) `auth_complete` runs the family completion, writes the resulting blob through the node's
///     `CredentialStore` (visible, redacted, via `credential_list`), and honors the optional
///     profile bind (`bound_accounts` gains the account);
/// (4) a consumed `flow_id` cannot be completed twice, and a cancelled flow cannot complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interactive_auth_generic_begin_complete_binds_and_lists() {
    use async_trait::async_trait;
    use daemon_api::{
        ApiError, AuthApi, AuthBeginRequest, AuthBindRequest, AuthChallenge, AuthCompleteRequest,
        AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput, CredentialApi, ProfileSpec,
        ProviderSelector,
    };
    use daemon_host::{
        AuthFlowFactory, AuthOutcome, AuthStepOutcome, MemCredentialStore, MemProfileStore,
        PendingAuthFlow, ProfileStore,
    };
    use daemon_protocol::TransportId;
    use std::collections::BTreeMap;

    // A parked flow: a single-redirect flow that echoes the captured callback into the blob so the
    // test can prove it flowed through, and reports a fixed identity (a real family derives these
    // from the IdP response).
    struct StubFlow {
        url: String,
    }
    #[async_trait]
    impl PendingAuthFlow for StubFlow {
        fn initial_challenge(&self) -> AuthChallenge {
            AuthChallenge::Redirect {
                authorization_url: self.url.clone(),
            }
        }
        async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
            let AuthStepInput::Callback(callback) = input else {
                return Err(ApiError::Other("stub flow expects a callback".into()));
            };
            Ok(AuthStepOutcome::Completed(AuthOutcome {
                credential_blob: format!("blob:{callback}"),
                credential_ref: "stub/acct".to_string(),
                account_label: "stub-user".to_string(),
                transport_instance: TransportId::new("stub/stub-user"),
                slot: daemon_host::CredentialSlotKind::Derived,
            }))
        }
    }

    struct StubFactory;
    #[async_trait]
    impl AuthFlowFactory for StubFactory {
        fn family(&self) -> &str {
            "stub"
        }
        fn provider_info(&self) -> AuthProviderInfo {
            AuthProviderInfo {
                family: "stub".into(),
                flow_kind: AuthFlowKind::OAuth2Pkce,
                display_name: "Stub IdP".into(),
                params_schema: vec![AuthParamField {
                    key: "homeserver".into(),
                    label: "Homeserver".into(),
                    required: true,
                    ..Default::default()
                }],
            }
        }
        async fn begin(
            &self,
            params: &BTreeMap<String, String>,
            redirect_uri: &str,
        ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
            let hs = params.get("homeserver").cloned().unwrap_or_default();
            Ok(Box::new(StubFlow {
                url: format!("{hs}/authorize?redirect_uri={redirect_uri}"),
            }))
        }
    }

    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create alpha");
    let creds = Arc::new(MemCredentialStore::new());

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
        profiles: Some(profiles.clone()),
        provider_resolver: None,
        credential_store: Some(creds),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![Arc::new(StubFactory)],
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

    // (1) discovery: the stub family is listed.
    let providers_list = node.auth_providers().await;
    assert_eq!(providers_list.len(), 1);
    assert_eq!(providers_list[0].family, "stub");
    assert_eq!(providers_list[0].flow_kind, AuthFlowKind::OAuth2Pkce);

    // (2) begin: parks a flow, mints the URL against our redirect, with a bind to `alpha`.
    let mut params = BTreeMap::new();
    params.insert("homeserver".to_string(), "https://idp.example".to_string());
    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "stub".into(),
            params,
            redirect_uri: "http://127.0.0.1:7777/cb".into(),
            bind: Some(AuthBindRequest {
                profile: ProfileRef::new("alpha"),
                transport_instance: None,
                credential_ref: None,
            }),
        })
        .await
        .expect("auth_begin");
    let auth_url = match &begun.challenge {
        AuthChallenge::Redirect { authorization_url } => authorization_url.clone(),
        other => panic!("expected a redirect challenge, got {other:?}"),
    };
    assert!(
        auth_url.contains("https://idp.example/authorize"),
        "authorization url from the family: {auth_url}"
    );
    assert!(
        auth_url.contains("redirect_uri=http://127.0.0.1:7777/cb"),
        "authorization url carries our redirect: {auth_url}"
    );

    // (3) complete: stores the blob, binds the account, returns the identity.
    let done = node
        .auth_complete(AuthCompleteRequest {
            flow_id: begun.flow_id.clone(),
            callback: "http://127.0.0.1:7777/cb?code=abc&state=xyz".into(),
        })
        .await
        .expect("auth_complete");
    assert_eq!(done.credential_ref, "stub/acct");
    assert_eq!(done.account_label, "stub-user");
    assert_eq!(done.transport_instance.as_str(), "stub/stub-user");
    assert_eq!(
        done.bound_profile.as_ref().map(|p| p.as_str()),
        Some("alpha")
    );

    let listed = node.credential_list().await;
    assert!(
        listed.iter().any(|c| c.profile == "stub/acct" && c.present),
        "the stored credential is listed (redacted): {listed:?}"
    );

    let alpha = profiles.get("alpha").unwrap().unwrap();
    assert!(
        alpha
            .bound_accounts
            .iter()
            .any(|a| a.transport_instance == "stub/stub-user" && a.credential_ref == "stub/acct"),
        "alpha gained the bound account: {:?}",
        alpha.bound_accounts
    );

    // (4a) a consumed flow_id cannot be completed twice.
    let reuse = node
        .auth_complete(AuthCompleteRequest {
            flow_id: begun.flow_id.clone(),
            callback: "http://127.0.0.1:7777/cb?code=abc".into(),
        })
        .await;
    assert!(
        reuse.is_err(),
        "a consumed flow_id cannot be completed twice"
    );

    // (4b) a cancelled flow cannot complete.
    let begun2 = node
        .auth_begin(AuthBeginRequest {
            family: "stub".into(),
            params: BTreeMap::new(),
            redirect_uri: "http://127.0.0.1:7777/cb".into(),
            bind: None,
        })
        .await
        .expect("auth_begin 2");
    node.auth_cancel(begun2.flow_id.clone())
        .await
        .expect("cancel is idempotent-ok");
    let after_cancel = node
        .auth_complete(AuthCompleteRequest {
            flow_id: begun2.flow_id,
            callback: "x".into(),
        })
        .await;
    assert!(after_cancel.is_err(), "a cancelled flow cannot complete");

    handle.shutdown().await;
}

/// CON-15 node half (the provider-bound OAuth family registration): the curated OpenRouter
/// descriptor registers as its own auth family whose id is EXACTLY `"provider/openrouter"` (the
/// string the sibling wire stream's `ProviderDescriptor.sign_in` advertisement points at) with an
/// EMPTY `params_schema` — the node owns every parameter, so the client calls
/// `auth_begin { family: "provider/openrouter", params: {} }`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_openrouter_family_registered_with_empty_schema() {
    use daemon_api::{AuthApi, AuthFlowKind};
    use daemon_oauth::{openrouter, DescriptorFlowFactory, OPENROUTER_FAMILY};

    let factory = Arc::new(
        DescriptorFlowFactory::new(openrouter()).expect("build the openrouter descriptor factory"),
    );
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
        profiles: None,
        provider_resolver: None,
        credential_store: Some(Arc::new(daemon_host::MemCredentialStore::new())),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![factory],
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

    let providers = node.auth_providers().await;
    let openrouter_info = providers
        .iter()
        .find(|p| p.family == OPENROUTER_FAMILY)
        .expect("the openrouter family is registered");
    assert_eq!(openrouter_info.family, "provider/openrouter");
    assert_eq!(openrouter_info.flow_kind, AuthFlowKind::OAuth2Pkce);
    assert!(
        openrouter_info.params_schema.is_empty(),
        "the provider-bound family owns every parameter (empty schema), got {:?}",
        openrouter_info.params_schema
    );

    handle.shutdown().await;
}

/// CON-15 node half (the provider-key slot mapping): a provider-bound OAuth family mints a MODEL
/// API key that must ride the BOUND PROFILE's credential slot — the id the model broker reads — so
/// it flows downstream exactly like a pasted key, and NO `BoundAccount` is attached (a provider key
/// is not a transport account). A `ProviderKeyForProfile` outcome with no bind is rejected (the key
/// would be stranded where no broker reads it). Driven through the node `AuthApi` end to end: a
/// provider-key descriptor whose JSON key-mint endpoint is a wiremock server returning `{"key":…}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_key_slots_under_bound_profile_and_requires_bind() {
    use daemon_api::{
        AuthApi, AuthBeginRequest, AuthBindRequest, AuthCompleteRequest, CredentialApi,
        ProfileSpec, ProviderSelector,
    };
    use daemon_host::{MemCredentialStore, MemProfileStore, ProfileStore};
    use daemon_oauth::{
        CallbackParam, CredentialShape, DescriptorFlowFactory, ExchangeStyle, OAuthFlowDescriptor,
        Source,
    };
    use std::collections::BTreeMap;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/keys"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "key": "sk-or-minted-abcd",
        })))
        .mount(&server)
        .await;

    // OpenRouter's provider-key shape, with the JSON key-mint endpoint pointed at wiremock (a
    // param) so the exchange runs against the mock; PKCE-only (no state), CallbackUrl.
    let descriptor = OAuthFlowDescriptor {
        family: "provider/openrouter",
        display_name: "OpenRouter (test)",
        authorization_endpoint: Source::Fixed("https://openrouter.ai/auth"),
        token_endpoint: Source::Param("token_endpoint"),
        client_id: None,
        client_secret_param: None,
        scopes: None,
        callback_param: CallbackParam::CallbackUrl,
        use_state: false,
        exchange: ExchangeStyle::JsonPost { key_field: "key" },
        credential: CredentialShape::ProviderKey {
            account_label: "openrouter",
        },
        params_schema: Vec::new(),
    };
    let factory = Arc::new(DescriptorFlowFactory::new(descriptor).expect("build factory"));

    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create alpha");
    let creds = Arc::new(MemCredentialStore::new());

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
        profiles: Some(profiles.clone()),
        provider_resolver: None,
        credential_store: Some(creds),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![factory],
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

    let params = || {
        let mut p = BTreeMap::new();
        p.insert(
            "token_endpoint".to_string(),
            format!("{}/keys", server.uri()),
        );
        p
    };

    // (1) WITH a bind: the minted bare key lands under the bound profile id (the broker's slot),
    // the response reports that ref, and NO BoundAccount is attached.
    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "provider/openrouter".into(),
            params: params(),
            redirect_uri: "http://127.0.0.1:7777/cb".into(),
            bind: Some(AuthBindRequest {
                profile: ProfileRef::new("alpha"),
                transport_instance: None,
                credential_ref: None,
            }),
        })
        .await
        .expect("auth_begin");
    let done = node
        .auth_complete(AuthCompleteRequest {
            flow_id: begun.flow_id,
            callback: "http://127.0.0.1:7777/cb?code=or-code".into(),
        })
        .await
        .expect("auth_complete");
    assert_eq!(
        done.credential_ref, "alpha",
        "the node slots the provider key under the BOUND PROFILE id, not a client-named ref"
    );
    assert_eq!(
        done.bound_profile.as_ref().map(|p| p.as_str()),
        Some("alpha")
    );

    let listed = node.credential_list().await;
    assert!(
        listed.iter().any(|c| c.profile == "alpha" && c.present),
        "the minted key is stored under the profile slot (redacted): {listed:?}"
    );
    let alpha = profiles.get("alpha").unwrap().unwrap();
    assert!(
        alpha.bound_accounts.is_empty(),
        "a provider key is NOT a transport account — no BoundAccount attach, got {:?}",
        alpha.bound_accounts
    );

    // (2) WITHOUT a bind: a provider-key mint is rejected (nowhere to slot the key).
    let begun2 = node
        .auth_begin(AuthBeginRequest {
            family: "provider/openrouter".into(),
            params: params(),
            redirect_uri: "http://127.0.0.1:7777/cb".into(),
            bind: None,
        })
        .await
        .expect("auth_begin 2");
    let err = node
        .auth_complete(AuthCompleteRequest {
            flow_id: begun2.flow_id,
            callback: "http://127.0.0.1:7777/cb?code=or-code".into(),
        })
        .await;
    assert!(
        err.is_err(),
        "a provider-key mint with no bind target must be rejected"
    );

    handle.shutdown().await;
}

/// FOUNDATION (account provisioning, daemon-event-io-spec §5.9.4 — the M2 bring-up seam): the
/// host exposes an in-process [`AccountProvisioning`] surface so a chat-transport adapter can
/// (a) enumerate the accounts it owns across every profile, by transport *family*; (b) resolve
/// each account's full credential blob in-process (the secret that never crosses the wire); and
/// (c) write back a refreshed blob (the token-refresh seam). Proves, with no chat transport:
/// (1) `bound_accounts("matrix")` returns exactly the two `matrix/...` accounts (right
/// profile/instance/credential_ref) and excludes the `slack/...` one (family-prefix matching);
/// (2) `account_credential(ref)` returns the opaque blob while the wire `credential_list` still
/// lists it redacted (enumeration vs. secret are least-privilege separate); (3)
/// `store_account_credential(ref, refreshed)` updates the store and `account_credential` reflects
/// the refresh.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn account_provisioning_enumerates_resolves_and_refreshes() {
    use daemon_api::{BoundAccount, CredentialApi, ProfileSpec, ProviderSelector};
    use daemon_host::{AccountProvisioning, MemCredentialStore, MemProfileStore, ProfileStore};
    use daemon_protocol::TransportId;

    // alpha owns one matrix account; beta owns a second matrix account AND a slack account. The
    // credential_ref of each names where its opaque session blob lives in the CredentialStore.
    let store = Arc::new(MemProfileStore::new());
    store
        .create(
            ProfileSpec::new("alpha", ProviderSelector::GenAi, "model-a")
                .with_bound_accounts(vec![BoundAccount::new("matrix/@a:hs", "matrix/alpha/a")]),
        )
        .expect("create alpha");
    store
        .create(
            ProfileSpec::new("beta", ProviderSelector::GenAi, "model-b").with_bound_accounts(vec![
                BoundAccount::new("matrix/@b:hs", "matrix/beta/b"),
                BoundAccount::new("slack/T0/@bot", "slack/beta/bot"),
            ]),
        )
        .expect("create beta");
    store.set_active("alpha").expect("set active");

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
        provider_resolver: None,
        credential_store: Some(Arc::new(MemCredentialStore::new())),
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
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });

    // 1. Enumerate by family: exactly the two matrix accounts, excluding slack.
    let mut matrix = node.bound_accounts("matrix");
    matrix.sort_by(|a, b| {
        a.transport_instance
            .as_str()
            .cmp(b.transport_instance.as_str())
    });
    assert_eq!(
        matrix.len(),
        2,
        "two matrix accounts, slack excluded: {matrix:?}"
    );
    assert_eq!(matrix[0].profile, ProfileRef::new("alpha"));
    assert_eq!(
        matrix[0].transport_instance,
        TransportId::new("matrix/@a:hs")
    );
    assert_eq!(matrix[0].credential_ref, "matrix/alpha/a");
    assert_eq!(matrix[1].profile, ProfileRef::new("beta"));
    assert_eq!(
        matrix[1].transport_instance,
        TransportId::new("matrix/@b:hs")
    );
    assert_eq!(matrix[1].credential_ref, "matrix/beta/b");
    assert_eq!(
        node.bound_accounts("slack").len(),
        1,
        "the slack family enumerates only its own account"
    );

    // 2. Resolve a blob in-process; the wire credential_list still hides it.
    node.credential_set("matrix/alpha/a".into(), "mxsession-blob-7f3c".into())
        .await
        .expect("store the opaque account session blob");
    assert_eq!(
        node.account_credential("matrix/alpha/a").as_deref(),
        Some("mxsession-blob-7f3c"),
        "the in-process seam resolves the full blob"
    );
    assert!(
        node.account_credential("matrix/does-not-exist").is_none(),
        "an unknown credential_ref resolves to None"
    );
    let listed = node.credential_list().await;
    let acct = listed
        .iter()
        .find(|c| c.profile == "matrix/alpha/a")
        .expect("the blob is listed under its credential ref");
    assert!(acct.present);
    assert_eq!(
        acct.hint, "…7f3c",
        "the wire surface stays redacted — the secret never crosses it"
    );

    // 3. Write-back: a refreshed blob updates the store and is reflected on the next resolve.
    node.store_account_credential("matrix/alpha/a", "mxsession-blob-REFRESHED")
        .expect("write back the refreshed credential");
    assert_eq!(
        node.account_credential("matrix/alpha/a").as_deref(),
        Some("mxsession-blob-REFRESHED"),
        "account_credential reflects the token-refresh write-back"
    );

    handle.shutdown().await;
}

/// USERPASSWORD INTERACTIVE-AUTH (wire v38 — the exchange-at-sign-in pattern): a reference
/// username/password family proves the enriched-field + `AuthFlowKind::UserPassword` surface end to
/// end through the node's `PendingAuthFlows` plumbing:
/// (1) `auth_providers` advertises the `UserPassword` flow with a MASKED (`AuthFieldKind::Password`)
///     password field beside a plain-text username;
/// (2) `auth_begin` parks a `Form` challenge carrying that same masked field;
/// (3) `auth_step(Fields)` with the RIGHT password validates + "exchanges" the (transient) password
///     for an opaque session-token blob and completes — the exchanged TOKEN (never the password) is
///     persisted through the `CredentialStore` (visible, redacted, via `credential_list`) and the
///     optional profile bind is honored;
/// (4) `auth_step(Fields)` with the WRONG password is rejected (the flow stays parked to retry).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interactive_auth_userpassword_form_exchange_completes() {
    use async_trait::async_trait;
    use daemon_api::{
        ApiError, AuthApi, AuthBeginRequest, AuthBindRequest, AuthChallenge, AuthFieldKind,
        AuthFlowKind, AuthParamField, AuthProviderInfo, AuthStepInput, AuthStepRequest,
        AuthStepResult, CredentialApi, ProfileSpec, ProviderSelector,
    };
    use daemon_host::{
        AuthFlowFactory, AuthOutcome, AuthStepOutcome, MemCredentialStore, MemProfileStore,
        PendingAuthFlow, ProfileStore,
    };
    use daemon_protocol::TransportId;
    use std::collections::BTreeMap;

    // The only accepted password (a real family exchanges against an IdP; this proves the seam).
    const GOOD_PASSWORD: &str = "correct-horse";

    // The masked `username`+`password` form this family collects (shared by discovery + the initial
    // challenge so the client renders identical fields either way).
    fn userpass_fields() -> Vec<AuthParamField> {
        vec![
            AuthParamField {
                key: "username".into(),
                label: "Username".into(),
                required: true,
                kind: AuthFieldKind::Text,
                default: None,
                placeholder: Some("you@example.org".into()),
                choices: Vec::new(),
            },
            AuthParamField {
                key: "password".into(),
                label: "Password".into(),
                required: true,
                kind: AuthFieldKind::Password,
                default: None,
                placeholder: None,
                choices: Vec::new(),
            },
        ]
    }

    struct UserPassFlow;
    #[async_trait]
    impl PendingAuthFlow for UserPassFlow {
        fn initial_challenge(&self) -> AuthChallenge {
            AuthChallenge::Form {
                title: "Sign in".into(),
                fields: userpass_fields(),
            }
        }
        async fn step(&self, input: AuthStepInput) -> Result<AuthStepOutcome, ApiError> {
            let AuthStepInput::Fields(fields) = input else {
                return Err(ApiError::Other("userpass expects form fields".into()));
            };
            let username = fields
                .get("username")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ApiError::Other("username is required".into()))?;
            let password = fields
                .get("password")
                .filter(|s| !s.is_empty())
                .ok_or_else(|| ApiError::Other("password is required".into()))?;
            if password != GOOD_PASSWORD {
                return Err(ApiError::Other("invalid username or password".into()));
            }
            // Exchange the (transient) password for an opaque session token; ONLY the token is
            // persisted — the password never enters the blob.
            Ok(AuthStepOutcome::Completed(AuthOutcome {
                credential_blob: format!("session-token:{username}"),
                credential_ref: format!("userpass/{username}"),
                account_label: username.clone(),
                transport_instance: TransportId::new(format!("userpass/{username}")),
                slot: daemon_host::CredentialSlotKind::Derived,
            }))
        }
    }

    struct UserPassFactory;
    #[async_trait]
    impl AuthFlowFactory for UserPassFactory {
        fn family(&self) -> &str {
            "userpass"
        }
        fn provider_info(&self) -> AuthProviderInfo {
            AuthProviderInfo {
                family: "userpass".into(),
                flow_kind: AuthFlowKind::UserPassword,
                display_name: "Username & password".into(),
                params_schema: userpass_fields(),
            }
        }
        async fn begin(
            &self,
            _params: &BTreeMap<String, String>,
            _redirect_uri: &str,
        ) -> Result<Box<dyn PendingAuthFlow>, ApiError> {
            Ok(Box::new(UserPassFlow))
        }
    }

    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .create(ProfileSpec::new(
            "alpha",
            ProviderSelector::GenAi,
            "model-a",
        ))
        .expect("create alpha");
    let creds = Arc::new(MemCredentialStore::new());

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
        profiles: Some(profiles.clone()),
        provider_resolver: None,
        credential_store: Some(creds),
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![Arc::new(UserPassFactory)],
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

    // (1) discovery: the userpass family advertises the UserPassword flow + a masked password field.
    let providers = node.auth_providers().await;
    let info = providers
        .iter()
        .find(|p| p.family == "userpass")
        .expect("the userpass family is listed");
    assert_eq!(info.flow_kind, AuthFlowKind::UserPassword);
    let password_field = info
        .params_schema
        .iter()
        .find(|f| f.key == "password")
        .expect("a password field in the schema");
    assert_eq!(
        password_field.kind,
        AuthFieldKind::Password,
        "the password field is advertised masked"
    );

    // (2) begin: parks a Form challenge carrying the masked password field, with a bind to alpha.
    let begun = node
        .auth_begin(AuthBeginRequest {
            family: "userpass".into(),
            params: BTreeMap::new(),
            redirect_uri: String::new(),
            bind: Some(AuthBindRequest {
                profile: ProfileRef::new("alpha"),
                transport_instance: None,
                credential_ref: None,
            }),
        })
        .await
        .expect("auth_begin");
    match &begun.challenge {
        AuthChallenge::Form { fields, .. } => assert!(
            fields
                .iter()
                .any(|f| f.key == "password" && f.kind == AuthFieldKind::Password && f.required),
            "the initial challenge carries the masked, required password field: {fields:?}"
        ),
        other => panic!("expected a Form challenge, got {other:?}"),
    }

    // (3) a WRONG password is rejected; the flow stays parked for a retry.
    let wrong = node
        .auth_step(AuthStepRequest {
            flow_id: begun.flow_id.clone(),
            input: AuthStepInput::Fields(BTreeMap::from([
                ("username".to_string(), "alice".to_string()),
                ("password".to_string(), "hunter2".to_string()),
            ])),
        })
        .await;
    assert!(wrong.is_err(), "a wrong password is rejected");

    // (4) the RIGHT password validates + exchanges for a session token and completes.
    let done = node
        .auth_step(AuthStepRequest {
            flow_id: begun.flow_id.clone(),
            input: AuthStepInput::Fields(BTreeMap::from([
                ("username".to_string(), "alice".to_string()),
                ("password".to_string(), GOOD_PASSWORD.to_string()),
            ])),
        })
        .await
        .expect("auth_step");
    let completed = match done {
        AuthStepResult::Completed(resp) => resp,
        AuthStepResult::Challenge(c) => panic!("expected completion, got a challenge {c:?}"),
    };
    assert_eq!(completed.credential_ref, "userpass/alice");
    assert_eq!(completed.account_label, "alice");
    assert_eq!(
        completed.bound_profile.as_ref().map(|p| p.as_str()),
        Some("alpha")
    );

    // The exchanged token — not the transient password — is stored (redacted) under the derived ref.
    let listed = node.credential_list().await;
    assert!(
        listed
            .iter()
            .any(|c| c.profile == "userpass/alice" && c.present),
        "the exchanged session token is persisted (redacted): {listed:?}"
    );
    let alpha = profiles.get("alpha").unwrap().unwrap();
    assert!(
        alpha
            .bound_accounts
            .iter()
            .any(|a| a.transport_instance == "userpass/alice"
                && a.credential_ref == "userpass/alice"),
        "alpha gained the bound account: {:?}",
        alpha.bound_accounts
    );

    handle.shutdown().await;
}
