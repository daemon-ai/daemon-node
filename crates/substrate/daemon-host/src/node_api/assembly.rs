// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Construction + wiring of [`NodeApiImpl`]: the [`NodeApiParts`] constructor and every
//! `with_*` / `set_*` builder seam the assembling binary uses to bind optional sub-surfaces.

use super::*;

/// The constructor inputs for [`NodeApiImpl::new`], grouped so node assembly passes one value
/// instead of six positional arguments.
pub struct NodeApiParts {
    /// The [`SupervisorObserver`] from `host.start().observer()`.
    pub supervisor: SupervisorObserver,
    /// The durable session store.
    pub store: Arc<dyn SessionStore>,
    /// The activation manager.
    pub manager: ActivationManager,
    /// This node's partition id.
    pub partition: PartitionId,
    /// Builds a fresh engine for each interactive (session sub-surface) session.
    pub engine_builder: SessionEngineBuilder,
    /// The optional control-surface fleet projection (`None` => empty fleet report).
    pub fleet: Option<Arc<dyn FleetControl>>,
}

impl NodeApiImpl {
    /// Assemble the node surface over the running substrate from its [`NodeApiParts`].
    pub fn new(parts: NodeApiParts) -> Self {
        let NodeApiParts {
            supervisor,
            store,
            manager,
            partition,
            engine_builder,
            fleet,
        } = parts;
        let session_modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>> =
            Arc::new(DashMap::new());
        let live = Arc::new(LiveSessions::new(
            engine_builder,
            session_modes.clone(),
            store.clone(),
        ));
        Self {
            supervisor,
            store,
            manager,
            fleet,
            partition,
            live,
            owners: Arc::new(DashMap::new()),
            verifier: None,
            models: None,
            default_local_profile: "default".to_string(),
            profiles: None,
            credentials: None,
            metrics: None,
            cloud_catalog: None,
            model_factory: None,
            session_models: Arc::new(DashMap::new()),
            session_modes,
            revisions: None,
            skills: None,
            routing: Arc::new(ArcSwap::from_pointee(RoutingRegistry::new())),
            routing_base: Arc::new(ArcSwap::from_pointee(RoutingRegistry::new())),
            chat_pins: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            routing_builder: None,
            adapters: Arc::new(ArcSwap::from_pointee(
                crate::adapters::AdapterRegistry::new(),
            )),
            mgmt_journal: Arc::new(std::sync::Mutex::new(None)),
            agents: None,
            last_agents: Arc::new(std::sync::RwLock::new(Vec::new())),
            checkpoints: None,
            auth_flows: None,
            fleet_events: None,
            node_events: None,
            workspace: None,
            blobs: None,
            cron: None,
            commands: Arc::new(ArcSwapOption::empty()),
            tools_inventory: Arc::new(ArcSwapOption::empty()),
            caps: daemon_api::CapsReport::default(),
            auth_store: None,
            auth_audit: None,
            revocations: None,
            credential_revoker: None,
        }
    }

    /// Bind the identity store backing the admin access-control sub-surface
    /// ([`daemon_api::AccessControlApi`]): `user_create`/`user_list`/`user_disable`/`user_set_roles`/
    /// `user_set_password`/`session_revoke`. Absent, those ops resolve to [`ApiError::Unsupported`]
    /// (`who_am_i` + `role_list` need no store and stay available).
    pub fn with_auth_store(mut self, auth_store: Arc<daemon_auth::AuthStore>) -> Self {
        self.auth_store = Some(auth_store);
        self
    }

    /// Bind the shared auth-audit sink so admin access-control mutations are recorded onto the
    /// verifiable `node-auth` journal stream. Pass the **same** [`AuthAudit`](crate::auth_audit::AuthAudit)
    /// to the transport's [`Authenticator`](crate::authn::Authenticator) so login/denial events chain
    /// together with the admin events. Absent, admin-op audit is a no-op.
    pub fn with_auth_audit(mut self, auth_audit: Arc<crate::auth_audit::AuthAudit>) -> Self {
        self.auth_audit = Some(auth_audit);
        self
    }

    /// Bind the shared per-principal revocation registry (Cluster F, Part A). Pass the **same**
    /// [`SessionRevocations`](crate::revocation::SessionRevocations) to the transport's
    /// [`Authenticator`](crate::authn::Authenticator) so an admin `session_revoke` (etc.) bump tears
    /// down the live connections it elevated. Absent, live connections are not torn down (the store
    /// mutation still invalidates the reconnect fast-path).
    pub fn with_revocations(
        mut self,
        revocations: Arc<crate::revocation::SessionRevocations>,
    ) -> Self {
        self.revocations = Some(revocations);
        self
    }

    /// Bind the credential-authority revoker (Cluster F, Part B) so `credential_remove`/
    /// `credential_set` invalidate the profile's outstanding leases. Pass the credential broker
    /// ([`MultiProfileStoreBroker`](crate::credentials::MultiProfileStoreBroker)), the same instance
    /// the engine acquires leases through. Absent, only the credential store is mutated.
    pub fn with_credential_revoker(
        mut self,
        revoker: Arc<dyn crate::revocation::CredentialRevoker>,
    ) -> Self {
        self.credential_revoker = Some(revoker);
        self
    }

    /// Bind the filesystem / workspace surface (`fs_*`), backed by the shared
    /// [`WorkspaceRoots`](crate::workspace_fs::WorkspaceRoots) the engine exec builder roots at, so
    /// operator (`fs_*`) and agent (`fs`/`shell` tools) see one filesystem. Absent, the `fs_*` ops
    /// resolve to [`ApiError::Unsupported`].
    pub fn with_workspace(mut self, workspace: Arc<crate::workspace_fs::WorkspaceFs>) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Bind the content store (blob CAS) backing the `blob_*` ops + `fs_write_from_blob`. Absent,
    /// those ops resolve to [`ApiError::Unsupported`].
    pub fn with_blobs(mut self, blobs: Arc<dyn crate::blob_store::BlobStore>) -> Self {
        self.blobs = Some(blobs);
        self
    }

    /// Bind the cron operations surface (I15) backing the `cron_*` control ops + suggestions. The
    /// same [`CronOps`](crate::cron::CronOps) is shared with the agent `cron` tool so both create
    /// jobs through one validation path. Absent, the cron ops keep their defaulted behavior.
    pub fn with_cron(mut self, cron: Arc<crate::cron::CronOps>) -> Self {
        self.cron = Some(cron);
        self
    }

    /// Bind the daemon-authoritative command catalog backing `command_list`/`command_invoke` at
    /// construction time. The assembling layer builds it from
    /// [`CommandRegistry::with_builtins`](crate::commands::CommandRegistry::with_builtins) plus the
    /// engine profile's command providers. Absent, the command surface stays empty / unsupported.
    pub fn with_commands(self, commands: Arc<crate::commands::CommandRegistry>) -> Self {
        self.commands.store(Some(commands));
        self
    }

    /// Bind (or replace) the command catalog *after* the node is wrapped in an `Arc` — the seam the
    /// assembling binary uses, since the registry's provider handles (`/lcm`, `/memory`) are resolved
    /// from node-owned bank caches the node construction does not itself hold.
    pub fn set_commands(&self, commands: Arc<crate::commands::CommandRegistry>) {
        self.commands.store(Some(commands));
    }

    /// Install the node-wide tool inventory backing [`daemon_api::ControlApi::tool_list`] (wire
    /// v29). Late-bound by the assembling binary, which owns the tool build gates and therefore
    /// knows both what registered and why a disabled optional surface did not.
    pub fn set_tool_inventory(&self, tools: Vec<daemon_api::ToolInfo>) {
        self.tools_inventory.store(Some(Arc::new(tools)));
    }

    /// Install the read-only delegation guardrail caps backing
    /// [`daemon_api::ControlApi::caps`] (wire v29) — the EFFECTIVE `orchestrate` ceilings the
    /// assembly composed (config policy min'd with the recursion budget).
    pub fn with_caps(mut self, caps: daemon_api::CapsReport) -> Self {
        self.caps = caps;
        self
    }

    /// Install the host routing registry consulted by [`SessionApi::submit_routed`] (the §5.9
    /// inbound-routing capability). Call during assembly; absent, routed submits fall back to
    /// `PerThread` naming with the node's active default profile.
    pub fn with_routing(mut self, routing: RoutingRegistry) -> Self {
        self.routing_base = Arc::new(ArcSwap::from_pointee(routing.clone()));
        self.routing = Arc::new(ArcSwap::from_pointee(routing));
        self
    }

    /// Install the transport-adapter registry (daemon-transport-adapter-spec.md §3.4): the node's
    /// self-describing events-IO adapters, enumerated read-only by `transport_adapters`. Call during
    /// assembly; absent, the node reports no adapters (the inert default). Lifecycle (`serve`) is not
    /// yet driven from here — that is deferred (spec §7 P1).
    pub fn with_adapters(mut self, adapters: crate::adapters::AdapterRegistry) -> Self {
        self.adapters = Arc::new(ArcSwap::from_pointee(adapters));
        self
    }

    /// Install (or replace) the transport-adapter registry **after** the node `Arc` exists — the
    /// runtime-injection counterpart of [`with_adapters`]. Required for adapters that must hold the
    /// assembled node as a seam (e.g. the Matrix adapter's `AccountProvisioning = node`), which cannot
    /// be built before the node and so cannot ride the consuming builder.
    pub fn set_adapters(&self, adapters: crate::adapters::AdapterRegistry) {
        self.adapters.store(Arc::new(adapters));
    }

    /// Drive every registered adapter's [`serve`](daemon_api::TransportAdapter::serve) loop with this
    /// node as their `api`, returning the spawned task handles (the binary aborts them on shutdown).
    /// Registry-driven lifecycle (daemon-messaging-adapter-spec.md §12.1). Adapters do not hold an
    /// `Arc<dyn NodeApi>` themselves, so handing `self.clone()` here introduces no reference cycle.
    ///
    /// Presence push (wire v29, B5): each serve loop is bracketed with
    /// [`NodeEvent::TransportChanged`](daemon_api::NodeEvent::TransportChanged) emits at the
    /// coarse REAL transitions — the instance's reported state at serve start, `Offline` at a
    /// clean teardown, `Error` when the loop crashes — so clients stop navigation-polling
    /// `TransportInstances`. Deliberately not a presence state machine: adapters that never
    /// transition simply never re-emit.
    pub fn spawn_adapters(self: &Arc<Self>) -> Vec<tokio::task::JoinHandle<()>> {
        use futures::FutureExt as _;
        let registry = self.adapters.load_full();
        registry
            .adapters()
            .iter()
            .map(|adapter| {
                let adapter = adapter.clone();
                let api: Arc<dyn daemon_api::NodeApi> = self.clone();
                let feed = self.node_events.clone();
                tokio::spawn(async move {
                    // Baseline push at serve start: each configured instance's reported state
                    // (a credentialed account reports Connected — the "serve start" transition).
                    if let Some(feed) = &feed {
                        for i in adapter.clone().instances().await {
                            feed.emit(daemon_api::NodeEvent::TransportChanged {
                                transport: i.transport,
                                connection: i.connection,
                                presence: i.presence,
                            });
                        }
                    }
                    let crashed = std::panic::AssertUnwindSafe(adapter.clone().serve(api))
                        .catch_unwind()
                        .await
                        .is_err();
                    // Teardown push: a clean serve exit is Offline; a crashed loop is Error.
                    if let Some(feed) = &feed {
                        let connection = if crashed {
                            daemon_api::ConnectionState::Error
                        } else {
                            daemon_api::ConnectionState::Offline
                        };
                        for i in adapter.clone().instances().await {
                            feed.emit(daemon_api::NodeEvent::TransportChanged {
                                transport: i.transport,
                                connection,
                                presence: daemon_api::PresenceState::Offline,
                            });
                        }
                    }
                })
            })
            .collect()
    }

    /// Install the routing *rebuild hook* (the §5.9 hot-reload seam): a closure that rebuilds the
    /// routing table from current node state. When set, it is run immediately to seed routing and
    /// re-run on every `profile_update` / `auth_complete`, so a profile/account change takes effect
    /// without a restart. The binary owns this closure because it owns the profile source.
    pub fn with_routing_builder(mut self, builder: RoutingBuilder) -> Self {
        self.routing_builder = Some(builder);
        self.rebuild_routing();
        self
    }

    /// Attach the §12 tool-checkpoint store so the `Checkpoint{List,Rewind}` ops can list rewind
    /// points and restore the workspace. Call during assembly with the same store wired into the
    /// engines (so a checkpoint recorded by a turn is visible + rewindable here).
    pub fn with_checkpoints(mut self, checkpoints: Arc<dyn daemon_core::CheckpointStore>) -> Self {
        self.checkpoints = Some(checkpoints.clone());
        // Share it with the live-session layer too, so a `RewindTo` rolls the workspace back to the
        // sealed-off range's earliest pre-mutation checkpoint (conversation-rewind spec §6).
        self.live.set_checkpoints(checkpoints);
        self
    }

    /// Attach the live model-provider factory so [`SessionApi::set_session_model`] can rebuild a
    /// running session's provider for a new model id. Call during assembly (needs the profile store
    /// + provider resolver to derive the provider from the session's profile bundle).
    pub fn with_model_factory(mut self, factory: ModelProviderFactory) -> Self {
        self.model_factory = Some(factory);
        self
    }

    /// Attach the resident telemetry aggregator so the `telemetry` control op surfaces the node's
    /// folded usage + event count + health (the same `Metrics` the host's metrics service dumps).
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach the live networked-model discovery hook (the binary's `genai`-backed catalog) so
    /// `models()` lists cloud models for adapters that have a resolvable key. Call during assembly.
    pub fn with_cloud_catalog(mut self, cloud_catalog: Arc<dyn CloudCatalog>) -> Self {
        self.cloud_catalog = Some(cloud_catalog);
        self
    }

    /// Attach the foreign-agent discovery hook (I7) so `agent_discover` probes the curated
    /// direct-binary recipe table (ACP entries via the `initialize` handshake). Injected by the
    /// binary (which owns `daemon-acp`).
    pub fn with_agent_discovery(mut self, agents: Arc<dyn AgentDiscovery>) -> Self {
        self.agents = Some(agents);
        self
    }

    /// Attach the host-owned fleet event bus (I4/I8) so [`ControlApi::tree_subscribe`] forwards live
    /// topology deltas. The same sender is handed to the orchestration producers (the
    /// `FleetJobWorker` delegation seam + the in-memory `FleetRuntime`) during assembly, so a real
    /// topology change pushes promptly instead of waiting for the next poll interval.
    pub fn with_fleet_events(mut self, tx: broadcast::Sender<daemon_api::TreeEvent>) -> Self {
        self.fleet_events = Some(tx);
        self
    }

    /// Wire the node-wide event feed (L3 `EventsSince`) so `events_*` serve live notifications and
    /// the §5 emit hooks (here + on the live-session actor) reach a real ring.
    pub fn with_node_events(mut self, feed: Arc<NodeEventFeed>) -> Self {
        self.live.set_node_events(feed.clone());
        self.node_events = Some(feed);
        self
    }

    /// Attach the §4.3 background-spawn materializer so a live session's `Effect::Spawn` raises an
    /// attached, non-joining review child (skill/memory review) without parking. Call during assembly.
    pub fn with_background(self, background: Arc<crate::background::BackgroundSpawner>) -> Self {
        self.live.set_background(background);
        self
    }

    /// Attach the auxiliary provider for background session-title generation: after a live
    /// session's first exchange completes, one best-effort `task = "title_generation"` call
    /// replaces the truncation-seeded roster title (hermes `title_generator` parity). Absent,
    /// sessions keep their seeded titles. Call during assembly.
    pub fn with_title_aux(self, aux: Arc<dyn daemon_core::Provider>) -> Self {
        self.live.set_title_aux(aux);
        self
    }

    /// Attach the model-management facade backing the `ModelApi` sub-surface, with the default
    /// profile a `model_activate` (no explicit profile) applies to. Call during assembly.
    pub fn with_models(
        mut self,
        models: Arc<ModelManager>,
        default_local_profile: impl Into<String>,
    ) -> Self {
        self.models = Some(models);
        self.default_local_profile = default_local_profile.into();
        self
    }

    /// Attach the durable profile store backing the `ProfileApi` sub-surface. Call during assembly.
    pub fn with_profiles(mut self, profiles: Arc<dyn ProfileStore>) -> Self {
        self.profiles = Some(profiles);
        self
    }

    /// Attach the persisted credential store backing the `CredentialApi` sub-surface. Call during
    /// assembly (the same store the node's credential authority provisions from).
    pub fn with_credential_store(mut self, credentials: Arc<dyn CredentialStore>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Register the interactive-auth factories backing the `AuthApi` sub-surface (the client-driven
    /// SSO/OAuth2 login seam). Each [`AuthFlowFactory`](crate::auth::AuthFlowFactory) serves one
    /// transport/provider family; absent (or empty), `auth_begin`/`auth_complete` resolve to
    /// [`ApiError::Unsupported`] and `auth_providers` is empty. The credential write + optional profile
    /// bind on completion go through the same credential/profile stores wired above. Call during assembly.
    pub fn with_auth_factories(
        mut self,
        factories: Vec<Arc<dyn crate::auth::AuthFlowFactory>>,
    ) -> Self {
        self.auth_flows = if factories.is_empty() {
            None
        } else {
            Some(Arc::new(PendingAuthFlows::new(factories)))
        };
        self
    }

    /// Attach the append-only revision log backing profile + skill versioning. Call during assembly
    /// (the same log the skills store records through, so operator and agent edits share one history).
    pub fn with_revisions(mut self, revisions: Arc<dyn daemon_common::RevisionLog>) -> Self {
        self.revisions = Some(revisions);
        self
    }

    /// Attach the per-profile skills provider backing skill versioning, distribution, and curation.
    /// Call during assembly (the same provider the engine path resolves per-session stores through).
    pub fn with_skills(mut self, skills: Arc<daemon_skills::SkillsProvider>) -> Self {
        self.skills = Some(skills);
        self
    }

    /// Durably journal live interactive sessions: each session's transcript (finished blocks +
    /// lifecycle) is sealed per turn into the unified verifiable journal keyed by its `SessionId`.
    /// Also records the node's `signer` so history reads verify sealed segments. Call during
    /// assembly, before any session is opened.
    pub fn with_journal(mut self, store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Self {
        self.verifier = Some(signer.clone());
        self.live.set_journal(JournalConfig { store, signer });
        self
    }
}
