//! [`NodeApiImpl`] — the node's [`daemon_api`] surface implemented over the running host.
//!
//! This is the one place the abstract interface ([`daemon_api::NodeApi`]) is bound to concrete
//! substrate machinery. Every transport (in-process, the Unix socket, the C FFI pump) ultimately
//! reaches *this* object; they differ only in how bytes arrive.
//!
//! - The **control sub-surface** ([`daemon_api::ControlApi`]) projects the durable node: the
//!   resident-service health ([`SupervisorObserver`]), durable queue/session stats and the session
//!   roster ([`SessionStore`]), session assignment (`ActivationManager::wake`, create-if-absent),
//!   and the orchestration fleet (via the injected [`crate::FleetView`]).
//! - The **session sub-surface** ([`daemon_api::SessionApi`]) drives live interactive engine
//!   sessions through the §17 actor ([`spawn_agent_session`]). Each session owns a drain buffer fed
//!   by the actor's event broadcast and a parked-request table so a poll-based embedder (the FFI)
//!   sees events *and* blocking host requests on one queue and answers them with `respond`.

use crate::engine_incarnation::JournalConfig;
use crate::journal::{JournalFeeder, JournalSink};
use crate::supervisor::{HealthStatus, SupervisorObserver};
use crate::FleetControl;
use async_trait::async_trait;
use daemon_activation::ActivationManager;
use crate::auth::PendingAuthFlows;
use crate::credstore::CredentialStore;
use crate::profiles::ProfileStore;
use crate::routing::RoutingRegistry;
use daemon_api::{
    from_cbor, to_cbor, AcpAgentEntry, AcpSource, ApiError, ApprovalInfo, ApprovalMode, AuthApi,
    AuthBeginRequest, AuthBeginResponse, AuthCompleteRequest, AuthCompleteResponse, AuthProviderInfo,
    BlobRef, BlobStat, BoundAccount, ByteRange, ChatRoute, ControlApi, CredentialApi, CredentialInfo,
    DeliverySink, Distribution, FleetReport, FsContent, FsEntry, FsRevision, FsRoot, FsRootId,
    FsRootKind, FsSearchPage, FsSearchQuery, FsWatchPageView, HealthReport, JournalPageView,
    JournalRecord, JournalRecordPayload, Lifecycle as ApiLifecycle,
    LogPageView, LogStream, ManageEventView, ModelApi, ModelDescriptor, Outbound, ProfileApi,
    ProfileInfo, ProfileSpec, ProviderSelector, RoomInfo, ServiceHealth, SessionApi, SessionDetail,
    SessionInfo, SessionMetaPatch, SessionOverlay, SessionPage, SessionQuery, SessionRole,
    SessionScope, SessionSearchHit, SessionState, StatsReport, TelemetryDump, TreeReport, UnitNode,
};
use daemon_common::{
    ContentHash, DownloadId, DownloadStatus, GgufInfo, InstalledModel, JobId, JournalStreamId,
    ModelEngine, ModelFile, ModelId, ModelRef, PartitionId, ProfileRef, QuantRecommendation,
    QuantizeId, QuantizeStatus, ReqId, SearchPage, SearchQuery, SessionId, UnitId, UsageDelta,
};
use daemon_core::{
    is_sensitive_path, spawn_agent_session, AgentHandle, ApprovalPolicy, Engine, LocalEnvironment,
    Provider, Snapshot,
};
use daemon_models::{ModelError, ModelManager};
use daemon_protocol::{
    AgentCommand, AgentEvent, DeliveryTarget, Direction, Disposition, HostRequest,
    HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, IsolationPolicy, Origin,
    OriginScope, SessionLogEntry, SessionPayload, SinkKind, TranscriptBlock, TransportId,
};
use daemon_store::{SessionMeta, SessionRole as StoreRole, SessionStatus, SessionStore};
use daemon_telemetry::{
    decode_entry, verify_segment, JournalPayload, Metrics, SegmentInput, TraceSigner, VerifyingKey,
    GENESIS_ROOT,
};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::BroadcastStream;

/// The session's own attribution for engine-emitted (outbound) merged-log entries.
fn engine_origin() -> Origin {
    Origin {
        transport: TransportId::new("engine"),
        scope: OriginScope::Internal,
    }
}

/// The attribution stamped on inbound items entering through the node api surface. The api `submit`
/// op carries no per-event origin yet (the surface-aware transports thread real origins in a later
/// phase), so node-api inbound is tagged with this generic local-api origin.
fn api_origin() -> Origin {
    Origin {
        transport: TransportId::new("api"),
        scope: OriginScope::Internal,
    }
}

/// Builds a fresh live [`Engine`] for an interactive session id (the session sub-surface's engine
/// seam — the binary supplies the provider/tools/system). The optional [`ProfileRef`] selects which
/// profile bundle the engine is materialized from (host routing's agent-selection degree of freedom);
/// `None` resolves the node's active default. The [`SessionOverlay`] is the session's persisted
/// per-session override (model/provider/tools/approval), applied on top of the bound profile at
/// build time, so a live override is **restored** when the actor is (re)spawned.
pub type SessionEngineBuilder =
    Arc<dyn Fn(SessionId, Option<ProfileRef>, &SessionOverlay) -> Engine + Send + Sync>;

/// Resolve a session's effective [`EngineProfile`] from its bound profile ref + persisted overlay —
/// the durable-path counterpart of [`SessionEngineBuilder`], injected into [`CoreEngineFactory`] by
/// the node (which owns the profile store + resolution rules). Returns `None` when no profile store
/// is configured or the bound profile is absent, so the durable path falls back to the factory's
/// default (orchestrator) profile. This is the seam that makes durable rehydration re-resolve from
/// the profile store + overlay instead of pinning the factory's fixed profile.
pub type DurableProfileResolver = Arc<
    dyn Fn(Option<ProfileRef>, &SessionOverlay) -> Option<daemon_core::EngineProfile> + Send + Sync,
>;

/// Encode a [`SessionOverlay`] to the opaque CBOR blob the store persists (host-level metadata).
pub fn encode_overlay(overlay: &SessionOverlay) -> Vec<u8> {
    let mut buf = Vec::new();
    // A SessionOverlay is always serializable; a failure here is a bug, not a runtime condition.
    ciborium::into_writer(overlay, &mut buf).expect("encode SessionOverlay");
    buf
}

/// Decode a [`SessionOverlay`] from its persisted blob; an empty/malformed blob is the empty
/// (all-inherit) overlay, so a session with no recorded override resolves straight from its profile.
pub fn decode_overlay(bytes: &[u8]) -> SessionOverlay {
    if bytes.is_empty() {
        return SessionOverlay::default();
    }
    ciborium::from_reader(bytes).unwrap_or_default()
}

/// Builds a fresh model [`Provider`] from a (model-overridden) [`ProfileSpec`] — the seam a live
/// [`SessionApi::set_session_model`](daemon_api::SessionApi::set_session_model) uses to rebuild a
/// running session's provider without `daemon-host` linking the provider crate.
pub type ModelProviderFactory = Arc<dyn Fn(&ProfileSpec) -> Arc<dyn Provider> + Send + Sync>;

/// The routing rebuild hook (the §5.9 hot-reload seam): produces a fresh [`RoutingRegistry`] from
/// current node state (profiles + bound accounts). Re-run on `profile_update` / `auth_complete` so
/// routing stays current without a restart. The assembling binary owns the closure (it owns the
/// profile source); the host never links the routing-from-profiles policy directly.
pub type RoutingBuilder = Arc<dyn Fn() -> RoutingRegistry + Send + Sync>;

/// Which lifecycle owns a `SessionId`. The durable and live lifecycles are intentionally distinct
/// (one runs an engine dormant-between-turns through the activation seam, the other keeps it
/// resident in an actor), and a single id must not exist as two divergent engine instances. This is
/// the guard-rail's ownership tag: a session is claimed by the first surface that touches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    /// Durable, control-surface managed (`assign` -> `ActivationManager`).
    Durable,
    /// Live, interactive session-surface managed (`submit` -> the §17 actor).
    Live,
}

/// The live networked-model discovery seam for the `ModelApi`'s `models()` listing.
///
/// `daemon-host` is provider-agnostic (it never links `genai`), so live cloud-model enumeration is
/// injected by the binary that *does* own the provider client. The implementation asks `genai`
/// (`Client::all_model_names`) for every adapter whose API key resolves, namespaces the ids so the
/// adapter round-trips through inference, and overlays local pricing/context. When no hook is wired
/// (tests, a remote-only node) `models()` falls back to the static [`ModelDescriptor`] catalog.
#[async_trait]
pub trait CloudCatalog: Send + Sync {
    /// The networked models a GUI can pick right now: the static catalog unioned with any live
    /// `genai` listing for adapters that have a resolvable key. Ids are namespaced (`groq::…`).
    async fn list(&self) -> Vec<ModelDescriptor>;
}

/// The ACP-discovery hook (I7). `daemon-host` does not link the ACP runtime (`daemon-acp` depends on
/// *it*, not the reverse), so the actual `initialize`-handshake probing — the curated direct-binary
/// recipe table + PATH probe — is injected by the assembling binary (which owns the ACP crate). When
/// no hook is wired, `acp_discover` returns empty and only manual registrations are catalogued.
#[async_trait]
pub trait AcpDiscovery: Send + Sync {
    /// Probe PATH + the curated direct-binary recipe table, confirming each candidate via the ACP
    /// `initialize` handshake; return verified catalog entries (`source = Builtin`).
    async fn discover(&self) -> Vec<daemon_api::AcpAgentEntry>;
    /// Verify/enrich a single (manual) recipe by attempting the `initialize` handshake — fills in
    /// `installed` / `version` / `capabilities`. Returns the entry unchanged on a failed probe.
    async fn probe(&self, entry: daemon_api::AcpAgentEntry) -> daemon_api::AcpAgentEntry;
}

/// The node interface implemented over a running [`crate::Host`].
#[derive(Clone)]
pub struct NodeApiImpl {
    supervisor: SupervisorObserver,
    store: Arc<dyn SessionStore>,
    manager: ActivationManager,
    fleet: Option<Arc<dyn FleetControl>>,
    partition: PartitionId,
    live: Arc<LiveSessions>,
    /// One-lifecycle-owner guard-rail: which lifecycle (if any) has claimed each session id.
    owners: Arc<DashMap<SessionId, Lifecycle>>,
    /// The node's journal signer, when journaling is enabled. Held here so a history read can verify
    /// each sealed segment (recompute root + check signature) before reporting it as `verified`.
    verifier: Option<Arc<TraceSigner>>,
    /// The model-management facade backing the `ModelApi` sub-surface. `None` on a node built
    /// without local-inference model management (every `ModelApi` call then resolves to
    /// [`ApiError::Unsupported`]).
    models: Option<Arc<ModelManager>>,
    /// The default profile a `model_activate` with no explicit profile applies to.
    default_local_profile: String,
    /// The durable profile store backing the `ProfileApi` sub-surface. `None` on a node built
    /// without profile management (every `ProfileApi` call then resolves to [`ApiError::Unsupported`]).
    profiles: Option<Arc<dyn ProfileStore>>,
    /// The persisted credential store backing the `CredentialApi` sub-surface. `None` on a node
    /// built without credential management (every `CredentialApi` call then resolves to
    /// [`ApiError::Unsupported`]).
    credentials: Option<Arc<dyn CredentialStore>>,
    /// The resident telemetry aggregator (the same handle the host's `Metrics/health` service
    /// dumps), surfaced through the `telemetry` control op. `None` => the op falls back to the
    /// store-projected default with a zero event counter.
    metrics: Option<Metrics>,
    /// The live networked-model discovery hook injected by the binary (the host never links
    /// `genai`). `None` => `models()` lists only the static cloud catalog + local models.
    cloud_catalog: Option<Arc<dyn CloudCatalog>>,
    /// The live model-provider factory backing `set_session_model`. `None` => per-session model
    /// switching resolves to [`ApiError::Unsupported`] (needs the profile store + provider resolver).
    model_factory: Option<ModelProviderFactory>,
    /// The per-session live model override set by `set_session_model` (transient; not persisted to
    /// the profile). Read by `model_current` when a session is being inspected.
    session_models: Arc<DashMap<SessionId, String>>,
    /// The per-session live edit-approval policy set by `set_session_mode` (transient). Read by the
    /// live [`ParkingHandler`] to decide auto-allow vs park, in lockstep with the engine's snapshot
    /// policy (both updated by the same op).
    session_modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
    /// The append-only revision history backing profile + skill versioning. `None` => the versioning
    /// ops (`profile_history`/`revert`, `skill_history`/`revert`) resolve to [`ApiError::Unsupported`].
    revisions: Option<Arc<dyn daemon_common::RevisionLog>>,
    /// The per-profile skills provider backing skill versioning, distribution, and curation. Resolves
    /// an `Arc<SkillStore>` per profile id (rooted at that agent's home), so skill ops act on the
    /// right agent's library. `None` => skill/curator ops + the skill payload of a distribution are
    /// unavailable.
    skills: Option<Arc<daemon_skills::SkillsProvider>>,
    /// The host routing registry (daemon-event-io-spec §5.9) consulted by [`SessionApi::submit_routed`]
    /// to resolve an inbound `Origin` to (session, profile, delivery). Empty by default — a pure
    /// passthrough: `PerThread` naming, node active-default profile, origin-seeded delivery.
    ///
    /// Held behind an `RwLock<Arc<…>>` so it is *hot-swappable*: a profile/auth change can rebuild
    /// the routing table live (via [`NodeApiImpl::rebuild_routing`]) without restarting the node.
    /// `submit_routed` clones the inner `Arc` under a brief read lock, so an in-flight resolve never
    /// blocks a swap.
    routing: Arc<std::sync::RwLock<Arc<RoutingRegistry>>>,
    /// The pin-free *base* routing registry (the static [`NodeApiImpl::with_routing`] table, or empty
    /// for the passthrough/builder cases). The live `routing` above is this base with the durable
    /// chat→session pins (`chat_pins`) layered on by [`NodeApiImpl::rebuild_routing`]; keeping the
    /// base separate lets a pin reload re-layer pins without losing the operator's binding table.
    routing_base: Arc<std::sync::RwLock<Arc<RoutingRegistry>>>,
    /// The resolve-first chat→session pins (§5.9, I5) loaded from the durable `chat_routes` store,
    /// keyed by canonical origin key. Re-layered onto a freshly-built registry on every rebuild;
    /// refreshed from the store by [`NodeApiImpl::load_routing_pins`] at boot and after a `routing_*`
    /// mutation.
    chat_pins: Arc<std::sync::RwLock<std::collections::HashMap<String, crate::routing::ChatPin>>>,
    /// The optional rebuild hook that produces a fresh [`RoutingRegistry`] from current node state
    /// (profiles + bound accounts). Installed by the assembling binary (which owns the profile
    /// source); when set, it is re-run on `profile_update` / `auth_complete` to keep routing current.
    /// `None` => routing is static (an explicit [`NodeApiImpl::with_routing`] table or the empty
    /// passthrough).
    routing_builder: Option<RoutingBuilder>,
    /// The ACP-discovery hook (I7), injected by the binary (which owns the ACP runtime). `None` =>
    /// `acp_discover` yields nothing and the catalog is just the durable manual registrations.
    acp: Option<Arc<dyn AcpDiscovery>>,
    /// The last ACP discovery scan's results, cached in-memory so `acp_catalog` can surface them
    /// alongside the durable manual entries without re-probing every read (discovery is the
    /// operator-triggered, subprocess-spawning scan; manual entries are the persisted half).
    last_acp: Arc<std::sync::RwLock<Vec<daemon_api::AcpAgentEntry>>>,
    /// The §12 tool-checkpoint store backing the `Checkpoint{List,Rewind}` ops. `None` => those ops
    /// resolve to an empty list / [`ApiError::Unsupported`] (a node with no checkpoint store).
    checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
    /// The interactive-auth registry backing the `AuthApi` sub-surface (the client-driven SSO/OAuth2
    /// login seam). `None` (or an empty registry) => every `AuthApi` call resolves to
    /// [`ApiError::Unsupported`] / an empty provider list.
    auth_flows: Option<Arc<PendingAuthFlows>>,
    /// The host-owned fleet event bus (I4/I8): the broadcast sender producers (the `FleetJobWorker`
    /// delegation seam, the in-memory `FleetRuntime`, and the `session_update_meta` op) ping on a
    /// real topology change, and [`ControlApi::tree_subscribe`] subscribes to so it can push live
    /// deltas instead of re-projecting `tree()` on a fixed poll interval. `None` => `tree_subscribe`
    /// falls back to the snapshot-only foundation with no live push source.
    fleet_events: Option<broadcast::Sender<daemon_api::TreeEvent>>,
    /// The filesystem / workspace surface (daemon-fs-surface-spec.md): resolves `FsRootId`s to
    /// directories (shared with the engine exec builder) and serves list/stat/read/write/search/
    /// watch. `None` => the `fs_*` ops resolve to [`ApiError::Unsupported`] (a node with no
    /// configured workspace).
    workspace: Option<Arc<crate::workspace_fs::WorkspaceFs>>,
    /// The content store (content-addressed blob CAS, daemon-content-transfer-spec.md): backs the
    /// `blob_*` ops and `fs_write_from_blob`. `None` => those ops resolve to
    /// [`ApiError::Unsupported`] (a node with no configured blob store).
    blobs: Option<Arc<dyn crate::blob_store::BlobStore>>,
    /// The cron operations surface (I15) backing the `cron_*` control ops + suggestions. `None` =>
    /// every cron op resolves to its defaulted [`ApiError::Unsupported`] / empty list (a node built
    /// without the cron backing). Shared with the agent `cron` tool, so both create through one path.
    cron: Option<Arc<crate::cron::CronOps>>,
}

impl NodeApiImpl {
    /// Assemble the node surface over the running substrate.
    ///
    /// - `supervisor` is the [`SupervisorObserver`] from `host.start().observer()`.
    /// - `engine_builder` builds a fresh engine for each interactive (session sub-surface) session.
    /// - `fleet` is the optional control-surface fleet projection (`None` => empty fleet report).
    pub fn new(
        supervisor: SupervisorObserver,
        store: Arc<dyn SessionStore>,
        manager: ActivationManager,
        partition: PartitionId,
        engine_builder: SessionEngineBuilder,
        fleet: Option<Arc<dyn FleetControl>>,
    ) -> Self {
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
            routing: Arc::new(std::sync::RwLock::new(Arc::new(RoutingRegistry::new()))),
            routing_base: Arc::new(std::sync::RwLock::new(Arc::new(RoutingRegistry::new()))),
            chat_pins: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            routing_builder: None,
            acp: None,
            last_acp: Arc::new(std::sync::RwLock::new(Vec::new())),
            checkpoints: None,
            auth_flows: None,
            fleet_events: None,
            workspace: None,
            blobs: None,
            cron: None,
        }
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

    /// The gated workspace write shared by `fs_write` and `fs_write_from_blob`: `Workspace`/`Session`
    /// roots only, sensitive-path + per-session `Deny` gate (overridable by `force`), a pre-mutation
    /// checkpoint for session roots, and the `Conflict`-on-stale-`base_revision` guard inside
    /// `WorkspaceFs::write`.
    async fn write_gated(
        &self,
        root: FsRootId,
        path: String,
        bytes: Vec<u8>,
        base_revision: Option<FsRevision>,
        force: bool,
    ) -> Result<FsRevision, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_write".into()))?;
        // Host browse roots are read-only.
        if !ws.writable(&root)? {
            return Err(ApiError::Unsupported(
                "host browse roots are read-only".into(),
            ));
        }
        // Sensitive-path gate (the same `.git`/`.ssh`/dotenv/keys rule the agent fs tool uses);
        // `force` overrides. The operator *is* the human, so this never routes through a host ask.
        if !force && is_sensitive_path(&path) {
            return Err(ApiError::Other(format!(
                "sensitive path {path:?} blocked; set force to override"
            )));
        }
        if let FsRootId::Session(sid) = &root {
            // A `Deny`-mode session blocks operator writes too, unless forced.
            if !force {
                if let Some(policy) = self.session_modes.get(sid) {
                    if *policy == ApprovalPolicy::Deny {
                        return Err(ApiError::Other(format!(
                            "session {} is in deny mode; set force to override",
                            sid.as_str()
                        )));
                    }
                }
            }
            // Capture a checkpoint before mutating, so an operator edit is rewindable like an agent
            // edit (best-effort; a capture failure never blocks the write).
            if let Some(store) = &self.checkpoints {
                let env = LocalEnvironment::new(ws.roots().session_root(sid.as_str()));
                let call_id = format!("operator-fs-write:{path}");
                let _ = store
                    .capture(sid.as_str(), &call_id, "operator_fs_write", &env)
                    .await;
            }
        }
        ws.write(&root, &path, &bytes, base_revision).await
    }

    /// Install the host routing registry consulted by [`SessionApi::submit_routed`] (the §5.9
    /// inbound-routing capability). Call during assembly; absent, routed submits fall back to
    /// `PerThread` naming with the node's active default profile.
    pub fn with_routing(mut self, routing: RoutingRegistry) -> Self {
        self.routing_base = Arc::new(std::sync::RwLock::new(Arc::new(routing.clone())));
        self.routing = Arc::new(std::sync::RwLock::new(Arc::new(routing)));
        self
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

    /// Hot-swap the *base* routing table (live). Used by the rebuild hook and available to the binary
    /// for an explicit refresh; an in-flight `submit_routed` resolve (which clones the inner `Arc`) is
    /// unaffected. The swap re-layers the current chat→session pins on top of the new base.
    pub fn swap_routing(&self, routing: RoutingRegistry) {
        *self.routing_base.write().unwrap() = Arc::new(routing);
        self.rebuild_routing();
    }

    /// Rebuild the live routing table: take the base (the rebuild hook's output when installed, else
    /// the static `routing_base`), layer the durable chat→session pins on top, and swap it in. A
    /// no-op-ish refresh when no builder is set, but always re-applies pins. Called after profile/auth
    /// mutations and after a pin reload so routing stays current without a restart.
    fn rebuild_routing(&self) {
        let mut reg = match &self.routing_builder {
            Some(builder) => builder(),
            None => (**self.routing_base.read().unwrap()).clone(),
        };
        reg.set_pins(self.chat_pins.read().unwrap().clone());
        *self.routing.write().unwrap() = Arc::new(reg);
    }

    /// Reload the durable chat→session routing pins (§5.9, I5) from the store into the in-memory pin
    /// cache and re-layer them onto the live registry. Called at boot (by the assembling binary) and
    /// after every `routing_*` mutation, riding the same hot-reload seam as profile/auth changes.
    pub async fn load_routing_pins(&self) {
        let routes = self.store.routing_list().await;
        let mut map = std::collections::HashMap::with_capacity(routes.len());
        for r in routes {
            map.insert(
                r.key.clone(),
                crate::routing::ChatPin {
                    session: r.session_id.clone(),
                    profile: r.profile.clone(),
                },
            );
        }
        *self.chat_pins.write().unwrap() = map;
        self.rebuild_routing();
    }

    /// A snapshot of the current routing registry (clones the inner `Arc` under a brief read lock).
    fn routing(&self) -> Arc<RoutingRegistry> {
        self.routing.read().unwrap().clone()
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

    /// Attach the ACP-discovery hook (I7) so `acp_discover` probes the curated direct-binary recipe
    /// table via the ACP `initialize` handshake. Injected by the binary (which owns `daemon-acp`).
    pub fn with_acp_discovery(mut self, acp: Arc<dyn AcpDiscovery>) -> Self {
        self.acp = Some(acp);
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

    /// Ping the fleet bus that the roster/tree changed (a rename/pin/archive that no producer models
    /// as a subagent transition). Projects a fresh `tree()` snapshot onto the bus off-thread so live
    /// `tree_subscribe` subscribers refresh promptly; a no-op when no bus is wired or there are no
    /// subscribers (so the projection cost is only paid when someone is watching).
    pub fn emit_tree_changed(&self) {
        if let Some(tx) = &self.fleet_events {
            if tx.receiver_count() == 0 {
                return;
            }
            let this = self.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                let report = this.tree().await;
                let _ = tx.send(daemon_api::TreeEvent::Snapshot(report));
            });
        }
    }

    /// Forward a concrete subagent/delegation lifecycle delta onto the fleet bus. A no-op when no bus
    /// is wired or there are no subscribers.
    pub fn emit_subagent(&self, view: daemon_protocol::ManageEventView) {
        if let Some(tx) = &self.fleet_events {
            let _ = tx.send(daemon_api::TreeEvent::Subagent(view));
        }
    }

    /// Attach the §4.3 background-spawn materializer so a live session's `Effect::Spawn` raises an
    /// attached, non-joining review child (skill/memory review) without parking. Call during assembly.
    pub fn with_background(self, background: Arc<crate::background::BackgroundSpawner>) -> Self {
        self.live.set_background(background);
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

    /// Fold the durable per-session usage totals across every known session — the node-wide
    /// accounting line (tokens, cache, reasoning, estimated `cost_micros`) reported on `stats`.
    async fn folded_usage(&self) -> UsageDelta {
        let mut total = UsageDelta::default();
        for (session, _status) in self.store.list_sessions().await {
            total.add(&self.store.usage_of(&session).await);
        }
        total
    }

    /// The unified, unscoped roster: every durable `session_record` row plus every live-interactive
    /// session, each enriched with its host meta (profile/title/last_activity/role/parent). The
    /// durable status wins when an id exists in both. The scope filter, sort, and pagination are
    /// applied by [`ControlApi::sessions_query`] on top of this.
    async fn roster(&self) -> Vec<SessionInfo> {
        let mut seen: std::collections::HashSet<SessionId> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (session, status) in self.store.list_sessions().await {
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            out.push(session_info_from(
                &session,
                Some(status),
                &meta,
                ApiLifecycle::Durable,
            ));
            seen.insert(session);
        }
        for session in self.live.live_ids() {
            if seen.contains(&session) {
                continue;
            }
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            out.push(session_info_from(&session, None, &meta, ApiLifecycle::Live));
        }
        out
    }

    /// Record activity on `session` from an inbound `command`: stamp `last_activity_ms` to now
    /// (roster sort key), seed a title from the first user turn when none is set, and index the
    /// turn's user text into the durable FTS surface (`session_search`). Read-modify-writes the host
    /// meta so the overlay/profile/role stay intact. Best-effort: a store error is swallowed (a
    /// missed stamp/index must never fail a submit).
    async fn note_activity(&self, session: &SessionId, command: &AgentCommand) {
        let turn_text = match command {
            AgentCommand::StartTurn { input, .. } => Some(input.text.clone()),
            AgentCommand::Steer { text, .. } => Some(text.clone()),
            _ => None,
        };
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        meta.last_activity_ms = Some(now_ms());
        // Seed a roster title from the opening user turn (truncated) when the session has none yet;
        // a real generated title can replace it later (the field is the foundation).
        if meta.title.is_none() {
            if let Some(text) = &turn_text {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    meta.title = Some(title_from_text(trimmed));
                }
            }
        }
        let title = meta.title.clone();
        let _ = self.store.set_session_meta(session, meta).await;
        if let Some(text) = turn_text {
            if !text.trim().is_empty() {
                self.store.index_session_text(session, title, &text).await;
            }
        }
        // Materialize any inbound message attachments into the session workspace `inbox/` before the
        // turn runs (daemon-content-transfer-spec.md Phase 2b), node-mediated: the client first
        // `blob_put`s the bytes, then submits the refs; the engine then sees the on-disk files.
        if let AgentCommand::StartTurn { input, .. } = command {
            if !input.attachments.is_empty() {
                self.materialize_inbound(session, &input.attachments).await;
            }
        }
    }

    /// Materialize inbound message attachment blobs into the session workspace `inbox/`. Best-effort:
    /// a fetch/write failure is skipped, never failing the submit. No-op when no workspace/blob store
    /// is bound.
    async fn materialize_inbound(&self, session: &SessionId, attachments: &[BlobRef]) {
        let (Some(ws), Some(blobs)) = (&self.workspace, &self.blobs) else {
            return;
        };
        let inbox = ws.roots().session_root(session.as_str()).join("inbox");
        if tokio::fs::create_dir_all(&inbox).await.is_err() {
            return;
        }
        for att in attachments {
            let Ok(bytes) = blobs.get(&att.hash, None).await else {
                continue;
            };
            let name = att
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.bin", att.hash.to_hex()));
            // Guard against a malicious name escaping inbox/ (use only the basename).
            let base = std::path::Path::new(&name)
                .file_name()
                .map(|n| n.to_owned())
                .unwrap_or_else(|| std::ffi::OsStr::new("attachment").to_owned());
            let _ = tokio::fs::write(inbox.join(base), bytes).await;
        }
    }

    /// The profile store, or [`ApiError::Unsupported`] when this node hosts no profile management.
    fn profile_store(&self) -> Result<&Arc<dyn ProfileStore>, ApiError> {
        self.profiles
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("profile management not available".into()))
    }

    /// Resolve the spec for an explicit id, or the active default when `profile` is `None`.
    fn resolve_profile(&self, profile: Option<String>) -> Result<Option<ProfileSpec>, ApiError> {
        let store = self.profile_store()?;
        let id = match profile {
            Some(id) => Some(id),
            None => store.active().map_err(profile_err)?,
        };
        match id {
            Some(id) => store.get(&id).map_err(profile_err),
            None => Ok(None),
        }
    }

    /// Resolve the [`ProfileSpec`] a session resolves its engine from: the session's persisted
    /// bound profile, falling back to the node's active default. The base for a live override apply.
    async fn session_spec(&self, session: &SessionId) -> Result<Option<ProfileSpec>, ApiError> {
        let bound = self
            .store
            .session_meta(session)
            .await
            .and_then(|m| m.bound_profile);
        match bound {
            Some(r) => self.resolve_profile(Some(r.as_str().to_string())),
            None => self.resolve_profile(None),
        }
    }

    /// Read-modify-write a session's persisted [`SessionOverlay`] (preserving its bound profile),
    /// returning the updated overlay. This is the single persistence path for every per-session
    /// override (model/provider/tools/approval), so an override is restored on rehydration.
    async fn update_overlay<F: FnOnce(&mut SessionOverlay)>(
        &self,
        session: &SessionId,
        f: F,
    ) -> SessionOverlay {
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        let mut overlay = decode_overlay(&meta.overlay);
        f(&mut overlay);
        meta.overlay = encode_overlay(&overlay);
        let _ = self.store.set_session_meta(session, meta).await;
        overlay
    }

    /// Apply a session's overlay to a live (resident) actor in place: rebuild the provider for a
    /// model/provider override and switch the edit-approval policy for a mode override. A
    /// non-resident (durable) session is a no-op here — it picks the overlay up at its next
    /// (re)hydration. Tool-allowlist overrides are *not* hot-applied (the live registry is fixed for
    /// the actor's lifetime); they take effect on the next (re)hydration.
    async fn apply_overlay_live(
        &self,
        session: &SessionId,
        overlay: &SessionOverlay,
    ) -> Result<(), ApiError> {
        let Some(handle) = self.live.handle_if_live(session) else {
            return Ok(());
        };
        if overlay.model.is_some() || overlay.provider.is_some() {
            let factory = self.model_factory.as_ref().ok_or_else(|| {
                ApiError::Unsupported("per-session model switch is not available".into())
            })?;
            let mut spec = self.session_spec(session).await?.ok_or_else(|| {
                ApiError::Unsupported("no profile to derive a provider from".into())
            })?;
            overlay.apply_to(&mut spec);
            handle.set_provider((factory)(&spec)).await;
        }
        if let Some(mode) = overlay.approval_mode {
            let policy = approval_mode_to_policy(mode);
            handle.set_approval_policy(policy).await;
            self.session_modes.insert(session.clone(), policy);
        }
        Ok(())
    }

    /// The revision log, or [`ApiError::Unsupported`] when this node hosts no versioning.
    fn revision_log(&self) -> Result<&Arc<dyn daemon_common::RevisionLog>, ApiError> {
        self.revisions
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("versioning not available".into()))
    }

    /// The skills provider, or [`ApiError::Unsupported`] when this node hosts no skills.
    fn skills_provider(&self) -> Result<&Arc<daemon_skills::SkillsProvider>, ApiError> {
        self.skills
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("skills not available".into()))
    }

    /// Resolve the [`SkillStore`](daemon_skills::SkillStore) for an explicit profile `id`.
    fn skills_store_for(&self, id: &str) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        Ok(self.skills_provider()?.for_profile(id))
    }

    /// Resolve the skills store for the profile a skill *versioning* op targets: the node's active
    /// default profile (falling back to the node's `default_local_profile` when no profile store /
    /// active default is set). The skill revision history is keyed by bare skill name, so this picks
    /// the library a name-keyed `skill_revert` writes back into.
    fn active_skills_store(&self) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        let id = self
            .profiles
            .as_ref()
            .and_then(|p| p.active().ok().flatten())
            .unwrap_or_else(|| self.default_local_profile.clone());
        self.skills_store_for(&id)
    }

    /// Resolve the skills store a curator op targets: an explicit `profile`, else the active default.
    fn curator_store(
        &self,
        profile: Option<String>,
    ) -> Result<Arc<daemon_skills::SkillStore>, ApiError> {
        match profile {
            Some(id) => self.skills_store_for(&id),
            None => self.active_skills_store(),
        }
    }

    /// The per-profile usage sidecar for a curator op, or [`ApiError::Unsupported`] when usage
    /// tracking is off (no `.usage.json` factory wired).
    fn curator_usage(
        &self,
        store: &Arc<daemon_skills::SkillStore>,
    ) -> Result<Arc<dyn daemon_common::SkillUsageLog>, ApiError> {
        store
            .usage()
            .cloned()
            .ok_or_else(|| ApiError::Unsupported("skill usage tracking not available".into()))
    }

    /// Record a profile revision of `id`'s current on-disk spec under `author`/`reason`. Best-effort:
    /// only when both a profile store and a revision log are wired, and a revision-log hiccup never
    /// fails the underlying profile mutation.
    fn record_profile(&self, id: &str, author: daemon_common::Author, reason: &str) {
        let (Some(store), Some(log)) = (self.profiles.as_ref(), self.revisions.as_ref()) else {
            return;
        };
        let Ok(Some(spec)) = store.get(id) else {
            return;
        };
        let mut blob = Vec::new();
        if ciborium::into_writer(&spec, &mut blob).is_ok() {
            let _ = log.append(daemon_common::RevisionKind::Profile, id, &blob, author, reason);
        }
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

    /// Read a stream's durable verifiable history: cursor-page the store, decode each entry to its
    /// typed view, decode block bodies into `TranscriptBlock`s, and stamp each entry with the
    /// verification result of its sealed segment. Non-destructive (the live drains are separate).
    async fn read_history(
        &self,
        stream: JournalStreamId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPageView {
        let page = self.store.load_journal(&stream, after_cursor, max).await;
        let key = self.verifier.as_ref().map(|s| s.verifying_key());

        // Verify each distinct sealed segment the page touches exactly once.
        let mut seg_verified: HashMap<u64, bool> = HashMap::new();
        for je in &page.entries {
            if let std::collections::hash_map::Entry::Vacant(slot) = seg_verified.entry(je.segment)
            {
                let ok = match &key {
                    Some(k) => self.verify_segment_in_store(&stream, je.segment, k).await,
                    None => false,
                };
                slot.insert(ok);
            }
        }

        let entries = page
            .entries
            .into_iter()
            .filter_map(|je| {
                let view = decode_entry(&je.entry.bytes).ok()?;
                let payload = match view.payload {
                    JournalPayload::Management { detail } => {
                        JournalRecordPayload::Management { detail }
                    }
                    JournalPayload::Block { body } => {
                        let block: TranscriptBlock = ciborium::from_reader(&body[..]).ok()?;
                        JournalRecordPayload::Block { block }
                    }
                };
                Some(JournalRecord {
                    cursor: je.cursor,
                    segment: je.segment,
                    seq: view.seq,
                    epoch: view.epoch,
                    trace: view.trace,
                    kind: view.kind,
                    timestamp_ms: view.timestamp_ms,
                    verified: seg_verified.get(&je.segment).copied().unwrap_or(false),
                    payload,
                })
            })
            .collect();

        let sealed_after = self
            .store
            .active_journal_seal(&stream)
            .await
            .map(|seal| seal.seal_cursor);

        JournalPageView {
            entries,
            next_cursor: page.next_cursor,
            head_cursor: page.head_cursor,
            sealed_after,
        }
    }

    /// Verify one sealed `(stream, segment)` against the node's verifying key: load the full
    /// segment, fold its entries onto the prior segment's sealed root, and check the signature. An
    /// open (unsealed) segment — or a broken prior link — reports `false`.
    async fn verify_segment_in_store(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        key: &VerifyingKey,
    ) -> bool {
        let Some(seg) = self.store.load_trace_segment(stream, segment).await else {
            return false;
        };
        let Some(committed) = seg.committed else {
            return false;
        };
        let prior = if segment == 0 {
            GENESIS_ROOT
        } else {
            match self
                .store
                .load_trace_segment(stream, segment - 1)
                .await
                .and_then(|s| s.committed)
            {
                Some(c) => c.root,
                None => return false,
            }
        };
        let entries: Vec<(u64, Vec<u8>, ContentHash)> = seg
            .entries
            .into_iter()
            .map(|e| (e.seq, e.bytes, e.content_hash))
            .collect();
        let input = SegmentInput {
            stream,
            segment,
            prior,
            entries: &entries,
        };
        verify_segment(&input, &committed.root, &committed.signature, key).is_ok()
    }

    /// Claim `session` for `want`, enforcing the one-lifecycle-owner invariant: the first surface to
    /// touch a session id owns it; the other surface is rejected with [`ApiError::Conflict`] until
    /// the session is released (via `cancel`). Re-claiming the same lifecycle is idempotent.
    fn claim(&self, session: &SessionId, want: Lifecycle) -> Result<(), ApiError> {
        use dashmap::mapref::entry::Entry;
        match self.owners.entry(session.clone()) {
            Entry::Occupied(e) if *e.get() != want => Err(ApiError::Conflict(format!(
                "session {session} is owned by the {:?} lifecycle; cannot use it as {:?}",
                e.get(),
                want
            ))),
            Entry::Occupied(_) => Ok(()),
            Entry::Vacant(v) => {
                v.insert(want);
                Ok(())
            }
        }
    }
}

fn map_state(status: SessionStatus) -> SessionState {
    match status {
        SessionStatus::Active => SessionState::Active,
        SessionStatus::Suspended { job_id } => SessionState::Suspended {
            job_id: job_id.to_string(),
        },
        SessionStatus::Ready => SessionState::Ready,
        SessionStatus::Completed => SessionState::Completed,
    }
}

/// Map a store-level [`StoreRole`] to the wire [`SessionRole`] (the two are distinct types so the
/// store stays protocol-free); a `None` role on a legacy meta row is a top-level `Primary`.
fn map_role(role: Option<StoreRole>) -> SessionRole {
    match role {
        Some(StoreRole::Primary) | None => SessionRole::Primary,
        Some(StoreRole::ManagedChild) => SessionRole::ManagedChild,
        Some(StoreRole::EphemeralSubagent) => SessionRole::EphemeralSubagent,
    }
}

/// Encode a wire [`ChatRoute`] into the protocol-free store row (§5.9, I5): the canonical origin key
/// + typed `session`/`profile` columns, with the full wire descriptor (origin + isolation) carried
/// as the opaque CBOR `descriptor` blob for faithful round-trip.
fn store_route_from_wire(route: &ChatRoute) -> daemon_store::ChatRoute {
    daemon_store::ChatRoute {
        key: crate::routing::origin_pin_key(&route.origin),
        session_id: route.session.clone(),
        profile: route.profile.clone(),
        descriptor: to_cbor(route),
    }
}

/// Decode a store row back to the wire [`ChatRoute`] from its opaque descriptor blob (`None` if the
/// blob fails to decode — a forward-compat/corruption guard).
fn wire_route_from_store(route: &daemon_store::ChatRoute) -> Option<ChatRoute> {
    from_cbor(&route.descriptor).ok()
}

/// Whether a stored route's transport instance belongs to the requested transport: an exact instance
/// match, or a family match (the requested id is the `family` segment before the first `/`). Lets
/// `transport_rooms("matrix")` enumerate rooms across every `matrix/@account` instance.
fn transport_family_matches(have: &TransportId, want: &TransportId) -> bool {
    have == want || have.as_str().split('/').next() == Some(want.as_str())
}

/// A human room/chat label for [`RoomInfo`], derived from an origin scope.
fn room_label(scope: &OriginScope) -> String {
    match scope {
        OriginScope::Dm { user } => user.clone(),
        OriginScope::Group { chat, .. } => chat.clone(),
        OriginScope::Api { key } => key.clone(),
        OriginScope::Internal => "internal".to_string(),
        other => format!("{other:?}"),
    }
}

/// Unix-millis now (roster `last_activity_ms` stamp).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The default roster page size when [`SessionQuery::limit`] is `0`.
const DEFAULT_ROSTER_PAGE: usize = 50;

/// A roster title seeded from the first user turn: the first line, trimmed to ~60 chars on a word
/// boundary with an ellipsis. A placeholder until a real generated title replaces it.
fn title_from_text(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or(text).trim();
    const MAX: usize = 60;
    if first_line.chars().count() <= MAX {
        return first_line.to_string();
    }
    let truncated: String = first_line.chars().take(MAX).collect();
    let cut = truncated.rsplit_once(' ').map(|(h, _)| h).unwrap_or(&truncated);
    format!("{}…", cut.trim_end())
}

/// Build a wire [`SessionInfo`] from a session id + its (optional) durable status + host meta +
/// lifecycle. The single place the enriched roster line is assembled, so the durable, live, and
/// detail paths stay consistent. A live session with no durable row reports `Active`.
fn session_info_from(
    session: &SessionId,
    status: Option<SessionStatus>,
    meta: &SessionMeta,
    lifecycle: ApiLifecycle,
) -> SessionInfo {
    SessionInfo {
        session: session.clone(),
        state: status.map(map_state).unwrap_or(SessionState::Active),
        // Daemon-core-backed engines own their conversation state and can truncate it, so durable
        // and live sessions are both rewindable; foreign ACP units are surfaced via the fleet API.
        rewindable: true,
        bound_profile: meta.bound_profile.clone(),
        title: meta.title.clone(),
        last_activity_ms: meta.last_activity_ms,
        lifecycle,
        role: map_role(meta.role),
        parent: meta.parent.clone(),
        pinned: meta.pinned,
        archived: meta.archived,
    }
}

/// Project a fresh tree snapshot, applying the subscriber's ephemeral filter — the re-projection a
/// coalescing `tree_subscribe` collapses a burst into, and the re-sync after a broadcast lag.
async fn filtered_tree(this: &NodeApiImpl, filter: &daemon_api::TreeSubFilter) -> TreeReport {
    let mut report = this.tree().await;
    if !filter.include_ephemeral {
        report
            .nodes
            .retain(|n| n.role != Some(SessionRole::EphemeralSubagent));
    }
    report
}

/// Apply the `TreeSubFilter` to one live bus event for the no-coalesce (forward-every-delta) path.
/// Returns `None` for events a stable-topology-only subscriber filters out (ephemeral subagent
/// deltas); a `Snapshot` is re-filtered to drop ephemeral nodes.
fn forward_event(
    event: daemon_api::TreeEvent,
    filter: &daemon_api::TreeSubFilter,
) -> Option<daemon_api::TreeEvent> {
    match event {
        daemon_api::TreeEvent::Snapshot(mut report) => {
            if !filter.include_ephemeral {
                report
                    .nodes
                    .retain(|n| n.role != Some(SessionRole::EphemeralSubagent));
            }
            Some(daemon_api::TreeEvent::Snapshot(report))
        }
        daemon_api::TreeEvent::Subagent(view) => {
            if !filter.include_ephemeral {
                if let daemon_protocol::ManageEventView::Subagent { role, .. } = &view {
                    if *role == SessionRole::EphemeralSubagent {
                        return None;
                    }
                }
            }
            Some(daemon_api::TreeEvent::Subagent(view))
        }
    }
}

#[async_trait]
impl ControlApi for NodeApiImpl {
    async fn health(&self) -> HealthReport {
        let services = self
            .supervisor
            .service_names()
            .into_iter()
            .map(|name| {
                let restarts = self.supervisor.restarts(&name).unwrap_or(0);
                let (ok, detail) = match self.supervisor.health(&name) {
                    Some(HealthStatus::Ok) => (true, None),
                    Some(HealthStatus::Degraded { reason })
                    | Some(HealthStatus::Unhealthy { reason }) => (false, Some(reason)),
                    None => (false, Some("unknown service".to_string())),
                };
                ServiceHealth {
                    name,
                    ok,
                    restarts,
                    detail,
                }
            })
            .collect();
        HealthReport {
            all_ok: self.supervisor.all_ok(),
            services,
        }
    }

    async fn stats(&self) -> StatsReport {
        let s = self.store.stats().await;
        StatsReport {
            pending_jobs: s.pending_jobs as u64,
            pending_wakes: s.pending_wakes as u64,
            sessions: s.sessions as u64,
            active: self.manager.active_count() as u64,
            usage: self.folded_usage().await,
        }
    }

    async fn telemetry(&self) -> TelemetryDump {
        let s = self.store.stats().await;
        // Prefer the resident aggregator's folded usage + event count when present; otherwise fall
        // back to the durable per-session fold (with no event counter).
        let (usage, events) = match &self.metrics {
            Some(m) => (m.usage(), m.events()),
            None => (self.folded_usage().await, 0),
        };
        TelemetryDump {
            usage,
            events,
            healthy: self.supervisor.all_ok(),
            pending_jobs: s.pending_jobs as u64,
            pending_wakes: s.pending_wakes as u64,
            sessions: s.sessions as u64,
            active: self.manager.active_count() as u64,
        }
    }

    async fn sessions(&self) -> Vec<SessionInfo> {
        self.sessions_query(SessionQuery::default()).await.sessions
    }

    async fn sessions_query(&self, query: SessionQuery) -> SessionPage {
        let mut roster = self.roster().await;
        // Scope filter. `TopLevel` (the inbox) shows only `Primary`; children are reached by walking
        // the tree. The by-profile/by-transport scopes back the per-agent/per-transport views.
        match &query.scope {
            SessionScope::TopLevel => {
                roster.retain(|i| i.role == SessionRole::Primary && !i.archived)
            }
            SessionScope::ByProfile(p) => {
                roster.retain(|i| i.bound_profile.as_ref() == Some(p) && !i.archived)
            }
            SessionScope::ByTransport(t) => {
                let owned: std::collections::HashSet<SessionId> =
                    self.live.delivery_sessions(t).into_iter().collect();
                roster.retain(|i| owned.contains(&i.session) && !i.archived);
            }
            SessionScope::Archived => {
                roster.retain(|i| i.role == SessionRole::Primary && i.archived)
            }
            SessionScope::All => {}
        }
        // Stable order: pinned conversations first, then most-recently-active, then id as the final
        // tie-break (so the cursor stays total across pages).
        roster.sort_by(|a, b| {
            b.pinned
                .cmp(&a.pinned)
                .then_with(|| b.last_activity_ms.cmp(&a.last_activity_ms))
                .then_with(|| a.session.as_str().cmp(b.session.as_str()))
        });
        // Cursor pagination: `after` is the last id of the previous page; skip through it.
        if let Some(after) = &query.after {
            if let Some(pos) = roster.iter().position(|i| &i.session == after) {
                roster.drain(..=pos);
            }
        }
        let limit = if query.limit == 0 {
            DEFAULT_ROSTER_PAGE
        } else {
            query.limit as usize
        };
        let next_cursor = if roster.len() > limit {
            roster.truncate(limit);
            roster.last().map(|i| i.session.clone())
        } else {
            None
        };
        SessionPage {
            sessions: roster,
            next_cursor,
        }
    }

    async fn session_get(&self, session: SessionId) -> Option<SessionDetail> {
        let status = self.store.status(&session).await;
        let is_live = self.live.live_ids().iter().any(|s| s == &session);
        if status.is_none() && !is_live {
            return None;
        }
        let meta = self.store.session_meta(&session).await.unwrap_or_default();
        let lifecycle = if status.is_some() {
            ApiLifecycle::Durable
        } else {
            ApiLifecycle::Live
        };
        let info = session_info_from(&session, status, &meta, lifecycle);
        let overlay = (!meta.overlay.is_empty()).then(|| decode_overlay(&meta.overlay));
        let model = self.session_models.get(&session).map(|m| m.clone());
        let delivery_targets = self.live.delivery_targets(&session);
        let children = self.store.children_of(&session).await;
        let checkpoints = match &self.checkpoints {
            Some(store) => store.list(Some(session.as_str())).await.len() as u32,
            None => 0,
        };
        Some(SessionDetail {
            info,
            overlay,
            model,
            delivery_targets,
            children,
            checkpoints,
        })
    }

    async fn sessions_by_profile(&self) -> Vec<(ProfileRef, Vec<SessionInfo>)> {
        let mut roster = self.roster().await;
        roster.retain(|i| i.role == SessionRole::Primary && i.bound_profile.is_some());
        let mut grouped: std::collections::BTreeMap<String, (ProfileRef, Vec<SessionInfo>)> =
            std::collections::BTreeMap::new();
        for info in roster {
            let profile = info.bound_profile.clone().expect("retained Some above");
            grouped
                .entry(profile.as_str().to_string())
                .or_insert_with(|| (profile, Vec::new()))
                .1
                .push(info);
        }
        grouped.into_values().collect()
    }

    async fn session_search(&self, query: String, limit: u32) -> Vec<SessionSearchHit> {
        self.store
            .search_sessions(&query, limit)
            .await
            .into_iter()
            .map(|hit| SessionSearchHit {
                session: hit.session_id,
                title: hit.title,
                snippet: hit.snippet,
            })
            .collect()
    }

    async fn session_update_meta(
        &self,
        session: SessionId,
        patch: SessionMetaPatch,
    ) -> Result<(), ApiError> {
        // Read-modify-write of the durable `SessionMeta`, preserving the fields the patch does not
        // touch (overlay/role/parent/bound profile/last activity). Each `None` patch field is a
        // leave-unchanged; `title: Some(None)` clears the title (rename-to-empty).
        let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
        if let Some(title) = patch.title {
            meta.title = title;
        }
        if let Some(pinned) = patch.pinned {
            meta.pinned = pinned;
        }
        if let Some(archived) = patch.archived {
            meta.archived = archived;
        }
        self.store
            .set_session_meta(&session, meta)
            .await
            .map_err(|e| ApiError::Other(e.to_string()))?;
        // Nudge live roster/tree subscribers so the rename/pin/archive shows up without a poll.
        self.emit_tree_changed();
        Ok(())
    }

    async fn approvals_pending(&self, session: Option<SessionId>) -> Vec<ApprovalInfo> {
        self.store
            .pending_approvals_of(session.as_ref())
            .await
            .into_iter()
            .map(|p| ApprovalInfo {
                session: p.session_id,
                request_id: p.job_id.as_str().to_string(),
                prompt: p.prompt,
                path: p.path,
            })
            .collect()
    }

    async fn approval_decide(
        &self,
        session: SessionId,
        request_id: String,
        allow: bool,
    ) -> Result<(), ApiError> {
        // Record the decision + enqueue the wake durably (one transaction in the store), then nudge
        // the activation manager so the dormant session rehydrates promptly and resolves the gated
        // tool call (allow -> runs it; deny -> injects a tool error). Idempotent in the store.
        let answered = self
            .store
            .answer_approval(&session, &JobId::new(request_id.clone()), allow)
            .await
            .map_err(|e| ApiError::Other(format!("answer approval: {e}")))?;
        if !answered {
            return Err(ApiError::Other(format!(
                "no pending approval {request_id} on session {session}"
            )));
        }
        self.manager
            .wake(session)
            .await
            .map_err(|e| ApiError::Other(format!("wake: {e}")))
    }

    async fn assign(&self, session: SessionId) -> Result<(), ApiError> {
        // Guard-rail: a session driven through the durable control surface must not also be a live
        // interactive session (two divergent engine instances for one id).
        self.claim(&session, Lifecycle::Durable)?;
        // Create-if-absent: a fresh durable session row with the engine's initial snapshot.
        if self.store.status(&session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            self.store
                .create_session(session.clone(), self.partition, blob)
                .await
                .map_err(|e| ApiError::Other(format!("create session: {e}")))?;
        }
        // Wake it: the activation manager runs (or resumes) the engine; the resident services then
        // carry the durable delegate -> suspend -> resume -> complete cycle forward.
        self.manager
            .wake(session)
            .await
            .map_err(|e| ApiError::Other(format!("wake: {e}")))
    }

    async fn cancel(&self, session: SessionId) -> Result<(), ApiError> {
        // Best-effort: cancel a matching fleet child and interrupt a matching live session.
        if let Some(fleet) = &self.fleet {
            fleet.cancel(&UnitId::new(session.as_str())).await;
        }
        self.live.interrupt(&session).await;
        // Release the lifecycle claim so the id can be reused by either surface.
        self.owners.remove(&session);
        Ok(())
    }

    async fn fleet(&self) -> FleetReport {
        match &self.fleet {
            Some(fleet) => fleet.report().await,
            None => FleetReport::default(),
        }
    }

    async fn tree(&self) -> TreeReport {
        match &self.fleet {
            Some(fleet) => fleet.tree().await,
            None => TreeReport::default(),
        }
    }

    async fn unit(&self, id: UnitId) -> Option<UnitNode> {
        match &self.fleet {
            Some(fleet) => fleet.unit(&id).await,
            None => None,
        }
    }

    async fn tree_subscribe(
        &self,
        filter: daemon_api::TreeSubFilter,
    ) -> Result<daemon_api::TreeStream, ApiError> {
        // Real event-driven merge (I4/I8): subscribe to the host fleet bus *first* (so no delta is
        // lost between the initial snapshot and the live tail), emit the current snapshot, then
        // forward live topology deltas. The `TreeSubFilter` is applied on the way out:
        //   - `include_ephemeral=false` drops `EphemeralSubagent` nodes from snapshots and drops
        //     `Subagent` deltas whose role is ephemeral (stable-topology-only subscribers).
        //   - `coalesce_ms` debounces a burst of deltas into one fresh `tree()` snapshot; `None`
        //     forwards every delta as it arrives.
        let this = self.clone();
        let rx = self.fleet_events.as_ref().map(|tx| tx.subscribe());

        // The initial snapshot is always emitted, bus or not.
        let initial = {
            let mut report = this.tree().await;
            if !filter.include_ephemeral {
                report
                    .nodes
                    .retain(|n| n.role != Some(SessionRole::EphemeralSubagent));
            }
            daemon_api::TreeEvent::Snapshot(report)
        };

        // No bus wired: fall back to the snapshot-only foundation (a single initial snapshot).
        let Some(rx) = rx else {
            return Ok(stream::once(async move { initial }).boxed());
        };

        let live = stream::unfold(
            (this, rx, filter),
            move |(this, mut rx, filter)| async move {
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            if let Some(window) = filter.coalesce_ms {
                                // Debounce: collapse this burst into one fresh re-projection.
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    window.max(1),
                                ))
                                .await;
                                while rx.try_recv().is_ok() {}
                                let report = filtered_tree(&this, &filter).await;
                                return Some((
                                    daemon_api::TreeEvent::Snapshot(report),
                                    (this, rx, filter),
                                ));
                            }
                            // No coalescing: forward the delta, applying the ephemeral filter.
                            match forward_event(event, &filter) {
                                Some(out) => return Some((out, (this, rx, filter))),
                                None => continue,
                            }
                        }
                        // We fell behind the bus: re-sync with a fresh authoritative snapshot.
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let report = filtered_tree(&this, &filter).await;
                            return Some((
                                daemon_api::TreeEvent::Snapshot(report),
                                (this, rx, filter),
                            ));
                        }
                        Err(broadcast::error::RecvError::Closed) => return None,
                    }
                }
            },
        );

        let stream = stream::once(async move { initial }).chain(live);
        Ok(stream.boxed())
    }

    async fn routing_list_chats(&self) -> Vec<ChatRoute> {
        self.store
            .routing_list()
            .await
            .iter()
            .filter_map(wire_route_from_store)
            .collect()
    }

    async fn routing_get(&self, origin: Origin) -> Option<ChatRoute> {
        let key = crate::routing::origin_pin_key(&origin);
        self.store
            .routing_get(&key)
            .await
            .as_ref()
            .and_then(wire_route_from_store)
    }

    async fn routing_set(&self, route: ChatRoute) -> Result<(), ApiError> {
        self.store
            .routing_set(store_route_from_wire(&route))
            .await
            .map_err(|e| ApiError::Other(format!("routing set: {e}")))?;
        // Ride the §5.9 hot-reload seam: reload pins into the live registry so the new pin resolves
        // immediately (resolve-first), without a restart.
        self.load_routing_pins().await;
        Ok(())
    }

    async fn routing_bind_chat(
        &self,
        origin: Origin,
        session: SessionId,
        profile: Option<ProfileRef>,
    ) -> Result<(), ApiError> {
        // The convenience form: a pin with the registry's default (`PerThread`) naming — the pinned
        // session id is authoritative, so the recorded isolation is informational.
        self.routing_set(ChatRoute {
            origin,
            session,
            profile,
            isolation: IsolationPolicy::PerThread,
        })
        .await
    }

    async fn routing_unbind_chat(&self, origin: Origin) -> Result<(), ApiError> {
        let key = crate::routing::origin_pin_key(&origin);
        self.store
            .routing_remove(&key)
            .await
            .map_err(|e| ApiError::Other(format!("routing unbind: {e}")))?;
        self.load_routing_pins().await;
        Ok(())
    }

    async fn transport_rooms(&self, transport: TransportId) -> Vec<RoomInfo> {
        // Read-only enumeration backed by the durable routing pins: the rooms this transport instance
        // (or family) has a pin for, each carrying its pinned session. A live adapter-backed room
        // listing (e.g. Matrix joined rooms) can layer on later behind the same shape.
        self.store
            .routing_list()
            .await
            .iter()
            .filter_map(wire_route_from_store)
            .filter(|r| transport_family_matches(&r.origin.transport, &transport))
            .map(|r| RoomInfo {
                transport: r.origin.transport.clone(),
                room: room_label(&r.origin.scope),
                name: None,
                session: Some(r.session.clone()),
            })
            .collect()
    }

    async fn acp_discover(&self) -> Vec<AcpAgentEntry> {
        // Probe the curated direct-binary recipe table via the injected ACP hook (the binary owns the
        // ACP runtime). Cache the results so `acp_catalog` surfaces them without re-probing, then
        // return the merged catalog (discovery results + durable manual registrations).
        if let Some(acp) = &self.acp {
            let discovered = acp.discover().await;
            *self.last_acp.write().unwrap() = discovered;
        }
        self.acp_catalog().await
    }

    async fn acp_catalog(&self) -> Vec<AcpAgentEntry> {
        // The durable manual registrations (source = Manual) take precedence over a builtin of the
        // same name; the in-memory last-discovery results fill in the auto-detected builtins.
        let mut by_name: std::collections::BTreeMap<String, AcpAgentEntry> =
            std::collections::BTreeMap::new();
        for entry in self.last_acp.read().unwrap().iter() {
            by_name.insert(entry.name.clone(), entry.clone());
        }
        for stored in self.store.acp_list().await {
            if let Ok(entry) = from_cbor::<AcpAgentEntry>(&stored.entry) {
                by_name.insert(entry.name.clone(), entry);
            }
        }
        by_name.into_values().collect()
    }

    async fn acp_register(&self, mut entry: AcpAgentEntry) -> Result<(), ApiError> {
        // A manual registration: force `source = Manual`, then verify/enrich it via the ACP
        // `initialize` handshake when a discovery hook is wired (fills installed/version/caps).
        entry.source = AcpSource::Manual;
        if let Some(acp) = &self.acp {
            entry = acp.probe(entry).await;
            entry.source = AcpSource::Manual;
        }
        self.store
            .acp_set(daemon_store::AcpEntry {
                name: entry.name.clone(),
                entry: to_cbor(&entry),
            })
            .await
            .map_err(|e| ApiError::Other(format!("acp register: {e}")))
    }

    async fn acp_remove(&self, name: String) -> Result<(), ApiError> {
        self.last_acp.write().unwrap().retain(|e| e.name != name);
        self.store
            .acp_remove(&name)
            .await
            .map_err(|e| ApiError::Other(format!("acp remove: {e}")))
    }

    // -- Cron (I15): every op delegates to the shared `CronOps`; absent it, the trait defaults
    //    (empty list / `Unsupported`) stand. --

    async fn cron_list(&self) -> Vec<daemon_api::CronJob> {
        match &self.cron {
            Some(cron) => cron.list().await,
            None => Vec::new(),
        }
    }

    async fn cron_create(&self, spec: daemon_api::CronSpec) -> Result<String, ApiError> {
        match &self.cron {
            Some(cron) => cron.create(spec).await,
            None => Err(ApiError::Unsupported("cron_create".into())),
        }
    }

    async fn cron_update(&self, id: String, spec: daemon_api::CronSpec) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.update(id, spec).await,
            None => Err(ApiError::Unsupported("cron_update".into())),
        }
    }

    async fn cron_delete(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.delete(id).await,
            None => Err(ApiError::Unsupported("cron_delete".into())),
        }
    }

    async fn cron_trigger(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.trigger(id).await,
            None => Err(ApiError::Unsupported("cron_trigger".into())),
        }
    }

    async fn cron_runs(&self, id: String) -> Vec<daemon_api::CronRun> {
        match &self.cron {
            Some(cron) => cron.runs(id).await,
            None => Vec::new(),
        }
    }

    async fn cron_pause(&self, id: String, paused: bool) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.pause(id, paused).await,
            None => Err(ApiError::Unsupported("cron_pause".into())),
        }
    }

    async fn cron_suggestions(&self) -> Vec<daemon_api::CronSuggestion> {
        match &self.cron {
            Some(cron) => cron.suggestions().await,
            None => Vec::new(),
        }
    }

    async fn cron_accept_suggestion(&self, id: String) -> Result<String, ApiError> {
        match &self.cron {
            Some(cron) => cron.accept_suggestion(id).await,
            None => Err(ApiError::Unsupported("cron_accept_suggestion".into())),
        }
    }

    async fn cron_dismiss_suggestion(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.dismiss_suggestion(id).await,
            None => Err(ApiError::Unsupported("cron_dismiss_suggestion".into())),
        }
    }

    async fn unit_events(&self, id: UnitId, max: u32) -> Vec<ManageEventView> {
        match &self.fleet {
            Some(fleet) => fleet.unit_events(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_outbound(&self, id: UnitId, max: u32) -> Vec<Outbound> {
        match &self.fleet {
            Some(fleet) => fleet.unit_outbound(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_history(&self, id: UnitId, after_cursor: u64, max: u32) -> JournalPageView {
        self.read_history(JournalStreamId::unit(&id), after_cursor, max)
            .await
    }

    async fn pause(&self, id: UnitId) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.pause(&id).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("pause {id}"))),
        }
    }

    async fn resume(&self, id: UnitId) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.resume(&id).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("resume {id}"))),
        }
    }

    async fn scale(&self, id: UnitId, n: u32) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.scale(&id, n).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("scale {id}"))),
        }
    }

    async fn verifying_key(&self) -> Option<String> {
        self.verifier.as_ref().map(|s| s.verifying_key().to_hex())
    }

    async fn checkpoints(&self, session: Option<SessionId>) -> Vec<daemon_api::CheckpointInfo> {
        let Some(store) = &self.checkpoints else {
            return Vec::new();
        };
        let filter = session.as_ref().map(|s| s.to_string());
        store
            .list(filter.as_deref())
            .await
            .into_iter()
            .map(|r| daemon_api::CheckpointInfo {
                id: r.id,
                session: SessionId::new(r.session),
                tool: r.tool,
                created_unix: r.created_unix,
                // The turn/cursor correlation is not yet recorded on the checkpoint ledger; the wire
                // fields exist (rewind-unify foundation) and fill in when the ledger carries them.
                turn_ordinal: None,
                cursor: None,
            })
            .collect()
    }

    async fn checkpoint_rewind(
        &self,
        _session: SessionId,
        checkpoint_id: String,
    ) -> Result<(), ApiError> {
        let store = self
            .checkpoints
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("checkpoint_rewind".into()))?;
        let record = store
            .get(&checkpoint_id)
            .await
            .ok_or_else(|| ApiError::Other(format!("unknown checkpoint: {checkpoint_id}")))?;
        store
            .restore(&record)
            .await
            .map_err(|e| ApiError::Other(format!("rewind failed: {e}")))
    }

    // ----- filesystem / workspace surface (daemon-fs-surface-spec.md) -----

    async fn fs_roots(&self) -> Vec<FsRoot> {
        let Some(ws) = &self.workspace else {
            return Vec::new();
        };
        let mut roots = Vec::new();
        // Host browse roots (home + operator allowlist) — discovery before binding.
        for (id, _dir) in ws.roots().browse_roots() {
            roots.push(FsRoot {
                id: FsRootId::Host(id.clone()),
                label: id.clone(),
                kind: FsRootKind::Host,
                session: None,
            });
        }
        // The node workspace root.
        roots.push(FsRoot {
            id: FsRootId::Workspace,
            label: "workspace".to_string(),
            kind: FsRootKind::Workspace,
            session: None,
        });
        // Opened (live) session sandboxes.
        for sid in self.live.live_ids() {
            roots.push(FsRoot {
                id: FsRootId::Session(sid.clone()),
                label: sid.as_str().to_string(),
                kind: FsRootKind::Session,
                session: Some(sid),
            });
        }
        roots
    }

    async fn fs_list(
        &self,
        root: FsRootId,
        dir: String,
        show_ignored: bool,
    ) -> Result<Vec<FsEntry>, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_list".into()))?;
        ws.list(&root, &dir, show_ignored).await
    }

    async fn fs_stat(&self, root: FsRootId, path: String) -> Result<FsEntry, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_stat".into()))?;
        ws.stat(&root, &path).await
    }

    async fn fs_read(
        &self,
        root: FsRootId,
        path: String,
        max_bytes: u64,
    ) -> Result<FsContent, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_read".into()))?;
        let mut content = ws.read(&root, &path, max_bytes).await?;
        // When a content store is bound and the whole file was returned, attach a content-addressed
        // ref so a client can hand the same bytes to an agent without re-uploading.
        if !content.truncated {
            if let Some(blobs) = &self.blobs {
                if let Ok(blob_ref) = blobs.put(&content.bytes).await {
                    content.blob_ref = Some(blob_ref);
                }
            }
        }
        Ok(content)
    }

    async fn fs_write(
        &self,
        root: FsRootId,
        path: String,
        bytes: Vec<u8>,
        base_revision: Option<FsRevision>,
        force: bool,
    ) -> Result<FsRevision, ApiError> {
        self.write_gated(root, path, bytes, base_revision, force).await
    }

    async fn fs_write_from_blob(
        &self,
        root: FsRootId,
        path: String,
        hash: ContentHash,
        base_revision: Option<FsRevision>,
        force: bool,
    ) -> Result<FsRevision, ApiError> {
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_write_from_blob".into()))?;
        let bytes = blobs
            .get(&hash, None)
            .await
            .map_err(|e| ApiError::Other(format!("blob fetch: {e}")))?;
        self.write_gated(root, path, bytes, base_revision, force).await
    }

    async fn blob_put(&self, bytes: Vec<u8>) -> Result<BlobRef, ApiError> {
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("blob_put".into()))?;
        blobs
            .put(&bytes)
            .await
            .map_err(|e| ApiError::Other(format!("blob put: {e}")))
    }

    async fn blob_get(
        &self,
        hash: ContentHash,
        range: Option<ByteRange>,
    ) -> Result<Vec<u8>, ApiError> {
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("blob_get".into()))?;
        blobs
            .get(&hash, range)
            .await
            .map_err(|e| ApiError::Other(format!("blob get: {e}")))
    }

    async fn blob_stat(&self, hash: ContentHash) -> BlobStat {
        match &self.blobs {
            Some(blobs) => match blobs.stat(&hash).await {
                Some(size) => BlobStat {
                    size,
                    present: true,
                },
                None => BlobStat {
                    size: 0,
                    present: false,
                },
            },
            None => BlobStat {
                size: 0,
                present: false,
            },
        }
    }

    async fn fs_search(
        &self,
        root: FsRootId,
        query: FsSearchQuery,
    ) -> Result<FsSearchPage, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_search".into()))?;
        ws.search(&root, &query).await
    }

    async fn fs_watch_after(
        &self,
        root: FsRootId,
        dir: String,
        after_seq: u64,
        max: u32,
    ) -> Result<FsWatchPageView, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_watch_after".into()))?;
        ws.watch_after(&root, &dir, after_seq, max).await
    }

    async fn rewind(
        &self,
        session: SessionId,
        point: daemon_api::RewindPoint,
    ) -> Result<(), ApiError> {
        // The unified rewind (conversation-rewind spec): truncate the transcript at `point.anchor`
        // and, when `point.restore_workspace`, roll the workspace back to the matching checkpoint —
        // sealing the journal on the way out. A resident session rewinds its in-process engine
        // directly through the shared seal+rollback seam that the live `RewindTo` command and the
        // managed/fleet engine path also call (so all three stay consistent).
        if self.live.handle_if_live(&session).is_some() {
            return self
                .live
                .rewind_resident(&session, point.anchor, point.restore_workspace)
                .await;
        }
        // A durable (non-resident) session has no live engine to truncate: its transcript is the
        // sealed journal, and rewinding it means re-incarnating the engine to truncate-and-reseal.
        // That activation-driven path is deferred (the checkpoint-ledger extension it needs is out of
        // scope this phase); surface it explicitly rather than silently no-op.
        Err(ApiError::Unsupported(
            "rewind of a non-resident durable session (re-incarnation path deferred)".into(),
        ))
    }
}

#[async_trait]
impl SessionApi for NodeApiImpl {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // Guard-rail: claim the session for the live lifecycle (rejects an id already durable-managed).
        self.claim(&session, Lifecycle::Live)?;
        self.note_activity(&session, &command).await;
        self.live.submit(session, command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        self.claim(&session, Lifecycle::Live)?;
        self.note_activity(&session, &command).await;
        self.live.submit_from(session, origin, command).await
    }

    async fn submit_as(
        &self,
        session: SessionId,
        origin: Option<Origin>,
        command: AgentCommand,
        profile: Option<ProfileRef>,
    ) -> Result<(), ApiError> {
        self.claim(&session, Lifecycle::Live)?;
        // Bind the explicit profile sticky-on-first-open (the same `ensure` seam `submit_routed`
        // uses), so a GUI can "open this chat as agent X" before the first turn submits.
        if profile.is_some() {
            self.live.ensure(&session, profile).await;
        }
        self.note_activity(&session, &command).await;
        match origin {
            Some(origin) => self.live.submit_from(session, origin, command).await,
            None => self.live.submit(session, command).await,
        }
    }

    async fn submit_routed(
        &self,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<SessionId, ApiError> {
        // Resolve the origin through the §5.9 routing registry: session name, the profile that runs
        // it (agent selection), and where its replies post.
        let resolved = self.routing().resolve(&origin);
        self.claim(&resolved.session, Lifecycle::Live)?;
        // For session-opening commands, bind the resolved profile (sticky on first `ensure`) and seed
        // the resolved `Primary` before submitting, so routing owns agent-selection + delivery. Other
        // commands act on an already-open session whose profile/Primary were bound when it opened.
        if matches!(
            command,
            AgentCommand::StartTurn { .. }
                | AgentCommand::Steer { .. }
                | AgentCommand::Observe { .. }
        ) {
            self.live
                .ensure(&resolved.session, resolved.profile.clone())
                .await;
            self.live
                .seed_primary_target(&resolved.session, resolved.delivery.clone());
        }
        self.note_activity(&resolved.session, &command).await;
        self.live
            .submit_from(resolved.session.clone(), origin, command)
            .await?;
        Ok(resolved.session)
    }

    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        self.live.poll(&session, max)
    }

    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError> {
        self.live.respond(&session, response)
    }

    async fn session_history(
        &self,
        session: SessionId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPageView {
        self.read_history(JournalStreamId::session(&session), after_cursor, max)
            .await
    }

    async fn log_after(
        &self,
        session: SessionId,
        after_seq: u64,
        max: u32,
    ) -> Result<LogPageView, ApiError> {
        Ok(self.live.log_after(&session, after_seq, max))
    }

    async fn subscribe(&self, session: SessionId, after_seq: u64) -> Result<LogStream, ApiError> {
        Ok(self.live.subscribe(&session, after_seq))
    }

    async fn delivery_targets(&self, session: SessionId) -> Vec<DeliveryTarget> {
        self.live.delivery_targets(&session)
    }

    async fn delivery_sessions(&self, transport: TransportId) -> Vec<SessionId> {
        self.live.delivery_sessions(&transport)
    }

    async fn handover(&self, session: SessionId, target: DeliveryTarget) -> Result<(), ApiError> {
        self.live.handover(&session, target)
    }

    async fn record_meta(
        &self,
        session: SessionId,
        origin: Origin,
        kind: String,
        body: Vec<u8>,
    ) -> Result<(), ApiError> {
        self.live.record_meta(&session, origin, kind, body)
    }

    async fn set_session_model(
        &self,
        session: SessionId,
        model: String,
        provider: Option<ProviderSelector>,
    ) -> Result<(), ApiError> {
        // Persist the model/provider override on the session overlay (durable host-level metadata),
        // then apply it to the live actor in place when resident. A non-resident session picks it up
        // at its next (re)hydration via the overlay — so a switch is no longer lost on restart.
        let overlay = self
            .update_overlay(&session, |o| {
                o.model = Some(model.clone());
                if let Some(p) = provider {
                    o.provider = Some(p);
                }
            })
            .await;
        self.apply_overlay_live(&session, &overlay).await?;
        self.session_models.insert(session, model);
        Ok(())
    }

    async fn set_session_mode(
        &self,
        session: SessionId,
        mode: ApprovalMode,
    ) -> Result<(), ApiError> {
        // Persist the edit-approval override on the overlay, then switch the live actor's policy in
        // place when resident (the live ParkingHandler reads `session_modes` to auto-allow vs park).
        let overlay = self.update_overlay(&session, |o| o.approval_mode = Some(mode)).await;
        self.apply_overlay_live(&session, &overlay).await?;
        // Keep the live mode cache populated even when not resident, so a freshly-resident actor's
        // ParkingHandler sees the persisted policy until `apply_overlay_live` refreshes it.
        self.session_modes
            .insert(session, approval_mode_to_policy(mode));
        Ok(())
    }

    async fn set_session_overlay(
        &self,
        session: SessionId,
        overlay: SessionOverlay,
    ) -> Result<(), ApiError> {
        // The unified per-session override write: persist the whole overlay, then apply what can be
        // hot-applied to a resident actor (model/provider/approval). A tool-allowlist change takes
        // effect on the next (re)hydration (the live registry is fixed for an actor's lifetime).
        let persisted = self
            .update_overlay(&session, |o| *o = overlay.clone())
            .await;
        self.apply_overlay_live(&session, &persisted).await?;
        if let Some(model) = &persisted.model {
            self.session_models.insert(session, model.clone());
        }
        Ok(())
    }
}

/// The **in-process** outbound push-registration surface (daemon-event-io-spec §5.9.3): an embedder
/// that holds the live [`NodeApiImpl`] hands the host a [`DeliverySink`] keyed by transport instance
/// so the per-session pump pushes outbound entries straight to it. This is deliberately *not* part
/// of the wire [`daemon_api::NodeApi`] surface — a sink is a live trait object that cannot cross a
/// process boundary, so cross-process transports use the pull path (`delivery_sessions` +
/// `subscribe`) instead. Registration is a live handle, not a wire op.
pub trait DeliveryHost: Send + Sync {
    /// Register (or replace) the push sink for `transport`; takes effect on the next pumped event.
    fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>);
    /// Drop the push sink for `transport` (its sessions revert to pull-only delivery).
    fn unregister_delivery_sink(&self, transport: &TransportId);
}

impl DeliveryHost for NodeApiImpl {
    fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>) {
        self.live.register_delivery_sink(transport, sink);
    }

    fn unregister_delivery_sink(&self, transport: &TransportId) {
        self.live.unregister_delivery_sink(transport);
    }
}

impl NodeApiImpl {
    /// Resolve a cron `deliver` directive to its concrete [`DeliveryTarget`]s, reusing the same
    /// origin/routing surface a live submit uses: `"origin"` is the job's captured origin's
    /// `primary_target()` (empty when no origin was captured — store-only fallback); `"all"` is every
    /// live session's `Primary` target (broadcast to active conversations); anything else is parsed as
    /// an explicit `"<transport>:<chat>"` direct target (split on the first `:`).
    fn resolve_delivery(&self, deliver: &str, origin: Option<&Origin>) -> Vec<DeliveryTarget> {
        match deliver.trim() {
            "origin" => origin.map(|o| vec![o.primary_target()]).unwrap_or_default(),
            "all" => self.live.all_primary_targets(),
            spec => match spec.split_once(':') {
                Some((transport, route)) if !transport.is_empty() && !route.is_empty() => {
                    vec![DeliveryTarget::new(transport, route, SinkKind::Primary)]
                }
                _ => Vec::new(),
            },
        }
    }
}

#[async_trait]
impl crate::CronDelivery for NodeApiImpl {
    async fn deliver(&self, deliver: &str, origin: Option<&Origin>, text: &str) {
        for target in self.resolve_delivery(deliver, origin) {
            // Attribute the delivered entry to the job's creating origin when known, else to a
            // host-internal origin on the target transport (a scheduled, principal-less push).
            let entry_origin = origin
                .cloned()
                .unwrap_or_else(|| Origin::internal(target.transport.clone()));
            // Carry the run's result as a single assistant text delta — the same outbound shape a
            // live reply takes, so a registered sink projects it to a message unchanged.
            let entry = SessionLogEntry::new(
                0,
                entry_origin,
                SessionPayload::Event(AgentEvent::TextDelta {
                    seq: 0,
                    text: text.to_string(),
                }),
            );
            self.live.push_to_target(target, entry).await;
        }
    }
}

/// One transport-instance account a profile is bound to (daemon-event-io-spec §5.9.4) — the read
/// side of [`ProfileSpec::bound_accounts`](daemon_api::ProfileSpec). It names *which* profile owns
/// the account, the instance-qualified [`TransportId`] (`matrix/@bot:hs.org`), and the
/// `credential_ref` of its opaque session blob in the `CredentialStore`. The secret itself is *not*
/// carried here — resolving it is a separate, in-process-only call ([`AccountProvisioning::account_credential`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvisionedAccount {
    /// The profile that declared (and runs) this account.
    pub profile: ProfileRef,
    /// The instance-qualified transport id (`matrix/@bot:hs.org`).
    pub transport_instance: TransportId,
    /// The `CredentialStore` key naming this account's opaque session blob.
    pub credential_ref: String,
}

/// The **in-process** account bring-up seam (daemon-event-io-spec §5.9.4): a chat-transport adapter
/// that holds the live [`NodeApiImpl`] uses this to discover the accounts it owns (across every
/// profile, by transport *family*), resolve each account's credential material, and write back a
/// refreshed blob after a token refresh. It is the read side of the account→profile binding the host
/// already derives routing from.
///
/// This is deliberately **not** part of the wire [`daemon_api::NodeApi`] surface — like
/// [`DeliveryHost`], it is a live in-process handle. [`Self::account_credential`] returns the *full*
/// secret blob, which never crosses the wire (the wire `CredentialApi` only lists redacted metadata);
/// enumeration ([`Self::bound_accounts`]) is kept separate from secret resolution so an adapter (or a
/// status view) can list accounts without touching secrets (least-privilege).
pub trait AccountProvisioning: Send + Sync {
    /// Every bound account whose `transport_instance` is in `transport_family` (the segment before the
    /// first `/`, e.g. `"matrix"` matches `matrix/@a:hs` and `matrix/@b:hs` but not `slack/…`), across
    /// all profiles. Empty if no profile store is wired or no account matches.
    fn bound_accounts(&self, transport_family: &str) -> Vec<ProvisionedAccount>;

    /// Resolve an account's full credential blob by its `credential_ref` (in-process only; the secret
    /// never crosses the wire). `None` if no credential store is wired or the ref is unknown.
    fn account_credential(&self, credential_ref: &str) -> Option<String>;

    /// Persist a refreshed credential `blob` under `credential_ref` — the token-refresh write-back
    /// seam (the `CredentialStore` is the system of record; e.g. driven from a `set_session_callback`).
    fn store_account_credential(&self, credential_ref: &str, blob: &str) -> Result<(), ApiError>;
}

impl AccountProvisioning for NodeApiImpl {
    fn bound_accounts(&self, transport_family: &str) -> Vec<ProvisionedAccount> {
        let Some(profiles) = self.profiles.as_ref() else {
            return Vec::new();
        };
        let Ok(specs) = profiles.list() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for spec in specs {
            for account in &spec.bound_accounts {
                // Family = the segment before the first `/` (the instance-qualified TransportId
                // convention, matching `routing::TransportPattern::Family`).
                if account.transport_instance.split('/').next() == Some(transport_family) {
                    out.push(ProvisionedAccount {
                        profile: ProfileRef::new(&spec.id),
                        transport_instance: TransportId::new(account.transport_instance.clone()),
                        credential_ref: account.credential_ref.clone(),
                    });
                }
            }
        }
        out
    }

    fn account_credential(&self, credential_ref: &str) -> Option<String> {
        self.credentials.as_ref()?.get(credential_ref)
    }

    fn store_account_credential(&self, credential_ref: &str, blob: &str) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .set(credential_ref, blob)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))
    }
}

/// Translate a wire-level [`daemon_api::ApprovalMode`] into the engine's
/// [`daemon_core::ApprovalPolicy`].
fn approval_mode_to_policy(mode: daemon_api::ApprovalMode) -> daemon_core::ApprovalPolicy {
    match mode {
        daemon_api::ApprovalMode::Ask => daemon_core::ApprovalPolicy::Ask,
        daemon_api::ApprovalMode::AcceptEdits => daemon_core::ApprovalPolicy::AcceptEdits,
        daemon_api::ApprovalMode::AutoAllow => daemon_core::ApprovalPolicy::AutoAllow,
        daemon_api::ApprovalMode::Deny => daemon_core::ApprovalPolicy::Deny,
    }
}

/// Map a `daemon-models` error onto the transport-stable [`ApiError`].
fn map_model_err(e: ModelError) -> ApiError {
    match e {
        ModelError::NotFound(m) => ApiError::Other(format!("not found: {m}")),
        ModelError::AccessDenied(m) => ApiError::Other(format!("access denied: {m}")),
        ModelError::Invalid(m) => ApiError::Unsupported(m),
        ModelError::Unknown(m) => ApiError::Other(format!("unknown id: {m}")),
        other => ApiError::Other(other.to_string()),
    }
}

#[async_trait]
impl ModelApi for NodeApiImpl {
    async fn model_search(&self, query: SearchQuery) -> Result<SearchPage, ApiError> {
        let m = self.require_models()?;
        m.search(query).await.map_err(map_model_err)
    }

    async fn model_files(
        &self,
        repo: String,
        revision: Option<String>,
        engine: ModelEngine,
    ) -> Result<Vec<ModelFile>, ApiError> {
        let m = self.require_models()?;
        m.model_files(&repo, revision.as_deref(), engine)
            .await
            .map_err(map_model_err)
    }

    async fn model_download(&self, model: ModelRef) -> Result<DownloadId, ApiError> {
        let m = self.require_models()?;
        m.download(model).await.map_err(map_model_err)
    }

    async fn model_downloads(&self) -> Vec<DownloadStatus> {
        match &self.models {
            Some(m) => m.downloads().await,
            None => Vec::new(),
        }
    }

    async fn model_cancel(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.cancel(id).await.map_err(map_model_err)
    }

    async fn model_pause(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.pause(id).await.map_err(map_model_err)
    }

    async fn model_resume(&self, id: DownloadId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.resume(id).await.map_err(map_model_err)
    }

    async fn model_catalog(&self) -> Vec<InstalledModel> {
        match &self.models {
            Some(m) => m.catalog().await,
            None => Vec::new(),
        }
    }

    async fn model_delete(&self, id: ModelId) -> Result<(), ApiError> {
        let m = self.require_models()?;
        m.delete(&id).await.map_err(map_model_err)
    }

    async fn model_activate(&self, id: ModelId, profile: Option<String>) -> Result<(), ApiError> {
        let m = self.require_models()?;
        let profile = profile.unwrap_or_else(|| self.default_local_profile.clone());
        m.activate(&id, &profile)
            .await
            .map(|_| ())
            .map_err(map_model_err)
    }

    async fn model_recommend(
        &self,
        repo: String,
        revision: Option<String>,
        engine: ModelEngine,
        budget_bytes: Option<u64>,
    ) -> Result<QuantRecommendation, ApiError> {
        let m = self.require_models()?;
        m.recommend(&repo, revision.as_deref(), engine, budget_bytes)
            .await
            .map_err(map_model_err)
    }

    async fn model_quantize(
        &self,
        repo: String,
        revision: Option<String>,
        target_quant: String,
        source_file: Option<String>,
    ) -> Result<QuantizeId, ApiError> {
        let m = self.require_models()?;
        m.quantize(&repo, revision.as_deref(), &target_quant, source_file)
            .await
            .map_err(map_model_err)
    }

    async fn model_quantizes(&self) -> Vec<QuantizeStatus> {
        match &self.models {
            Some(m) => m.quantizes().await,
            None => Vec::new(),
        }
    }

    async fn model_inspect(&self, id: ModelId) -> Result<GgufInfo, ApiError> {
        let m = self.require_models()?;
        m.inspect(&id).await.map_err(map_model_err)
    }

    async fn models(&self) -> Vec<ModelDescriptor> {
        // Networked models: a live `genai` listing (per adapter with a resolvable key, namespaced,
        // pricing/context overlaid) when the discovery hook is wired, else the static catalog
        // (incl. claude-opus-4-8). Then merge any locally-installed (GGUF) models.
        let mut out = match &self.cloud_catalog {
            Some(catalog) => catalog.list().await,
            None => ModelDescriptor::builtin_cloud_catalog(),
        };
        if let Some(m) = &self.models {
            for im in m.catalog().await {
                let provider = match im.model.engine {
                    ModelEngine::MistralRs => ProviderSelector::MistralRs,
                    ModelEngine::Llama => ProviderSelector::LlamaCpp,
                };
                out.push(ModelDescriptor {
                    id: im.id.as_str().to_string(),
                    provider,
                    context_length: im.context_length,
                    input_price_micros_per_mtok: None,
                    output_price_micros_per_mtok: None,
                    local: true,
                });
            }
        }
        out
    }

    async fn model_current(
        &self,
        profile: Option<String>,
    ) -> Result<Option<ModelDescriptor>, ApiError> {
        let spec = if self.profiles.is_some() {
            self.resolve_profile(profile)?
        } else {
            None
        };
        let Some(spec) = spec else { return Ok(None) };
        // Prefer a catalog entry (carries context/pricing); else synthesize from the profile spec.
        if let Some(found) = self.models().await.into_iter().find(|m| m.id == spec.model) {
            return Ok(Some(found));
        }
        Ok(Some(ModelDescriptor {
            id: spec.model.clone(),
            provider: spec.provider,
            context_length: ModelDescriptor::known_context_length(&spec.model),
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: matches!(
                spec.provider,
                ProviderSelector::LlamaCpp | ProviderSelector::MistralRs
            ),
        }))
    }
}

impl NodeApiImpl {
    /// The model-management facade, or [`ApiError::Unsupported`] when this node has none.
    fn require_models(&self) -> Result<&Arc<ModelManager>, ApiError> {
        self.models
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("model management is not enabled".into()))
    }
}

/// Map a profile-store error onto the wire [`ApiError`].
fn profile_err(e: crate::profiles::ProfileError) -> ApiError {
    use crate::profiles::ProfileError;
    match e {
        ProfileError::NotFound(id) => ApiError::UnknownSession(id),
        ProfileError::Exists(id) => ApiError::Conflict(format!("profile exists: {id}")),
        other => ApiError::Other(other.to_string()),
    }
}

/// Map a revision-log error onto the wire [`ApiError`].
fn revision_err(e: daemon_common::RevisionError) -> ApiError {
    use daemon_common::RevisionError;
    match e {
        RevisionError::NotFound { kind, id, seq } => {
            ApiError::UnknownSession(format!("{kind}/{id}@{seq}"))
        }
        other => ApiError::Other(other.to_string()),
    }
}

fn skill_err(e: daemon_skills::SkillError) -> ApiError {
    use daemon_skills::SkillError;
    match e {
        SkillError::NotFound(id) => ApiError::UnknownSession(format!("skill/{id}")),
        SkillError::Exists(id) => ApiError::Conflict(format!("skill exists: {id}")),
        other => ApiError::Other(other.to_string()),
    }
}

#[async_trait]
impl ProfileApi for NodeApiImpl {
    async fn profile_list(&self) -> Vec<ProfileInfo> {
        let Ok(store) = self.profile_store() else {
            return Vec::new();
        };
        let active = store.active().ok().flatten();
        match store.list() {
            Ok(specs) => {
                let mut out: Vec<ProfileInfo> = specs
                    .iter()
                    .map(|s| ProfileInfo::from_spec(s, active.as_deref() == Some(s.id.as_str())))
                    .collect();
                out.sort_by(|a, b| a.id.cmp(&b.id));
                out
            }
            Err(_) => Vec::new(),
        }
    }

    async fn profile_get(&self, id: String) -> Result<Option<ProfileSpec>, ApiError> {
        self.profile_store()?.get(&id).map_err(profile_err)
    }

    async fn profile_create(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        let id = spec.id.clone();
        self.profile_store()?.create(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "create");
        Ok(())
    }

    async fn profile_update(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        let id = spec.id.clone();
        self.profile_store()?.update(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "update");
        // A profile change can alter routing (agent selection / transport patterns): rebuild the
        // live table so routed submits pick up the change without a restart (§5.9 hot-reload).
        self.rebuild_routing();
        Ok(())
    }

    async fn profile_delete(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.delete(&id).map_err(profile_err)
    }

    async fn profile_select(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.set_active(&id).map_err(profile_err)
    }

    async fn profile_clone(&self, source: String, new_id: String) -> Result<(), ApiError> {
        let store = self.profile_store()?;
        let mut spec = store
            .get(&source)
            .map_err(profile_err)?
            .ok_or_else(|| ApiError::UnknownSession(source.clone()))?;
        spec.id = new_id.clone();
        store.create(spec).map_err(profile_err)?;
        self.record_profile(
            &new_id,
            daemon_common::Author::Operator,
            &format!("clone of {source}"),
        );
        Ok(())
    }

    async fn profile_export(&self, id: String) -> Result<Distribution, ApiError> {
        let spec = self
            .profile_store()?
            .get(&id)
            .map_err(profile_err)?
            .ok_or_else(|| ApiError::UnknownSession(id.clone()))?;
        // A profile distribution carries *that profile's* local (non-bundled) skills, resolved from
        // its own per-profile store; otherwise just the spec. credential_ref is kept (a name).
        let skills = match self.skills.as_ref() {
            Some(provider) => provider
                .for_profile(&id)
                .export_local()
                .map_err(|e| ApiError::Other(format!("skill export: {e}")))?,
            None => Vec::new(),
        };
        let head_seq = self
            .revisions
            .as_ref()
            .and_then(|log| log.head(daemon_common::RevisionKind::Profile, &id).ok().flatten())
            .map(|r| r.seq);
        Ok(Distribution {
            wire_version: daemon_common::WireVersion::CURRENT,
            profile: spec,
            skills,
            head_seq,
            source: None,
        })
    }

    async fn profile_import(
        &self,
        dist: Distribution,
        new_id: Option<String>,
    ) -> Result<String, ApiError> {
        if dist.wire_version != daemon_common::WireVersion::CURRENT {
            return Err(ApiError::Other(format!(
                "incompatible distribution wire version {} (node is {})",
                dist.wire_version.0,
                daemon_common::WireVersion::CURRENT.0
            )));
        }
        let store = self.profile_store()?;
        let mut spec = dist.profile;
        if let Some(id) = new_id {
            spec.id = id;
        }
        let id = spec.id.clone();
        store.create(spec).map_err(profile_err)?;
        self.record_profile(&id, daemon_common::Author::Operator, "import");
        // Materialize the distribution's local skills into the *imported profile's own* skills dir
        // (so a session that resolves this agent actually sees them), attributed to the operator. A
        // skill that already exists is left as-is rather than clobbered.
        if let Some(provider) = self.skills.as_ref() {
            let skill_store = provider.for_profile(&id);
            for bundle in &dist.skills {
                if skill_store.is_bundled(&bundle.name) {
                    continue;
                }
                skill_store
                    .import_bundle(bundle, daemon_common::Author::Operator, &format!("import via {id}"))
                    .map_err(|e| ApiError::Other(format!("skill import: {e}")))?;
            }
        }
        Ok(id)
    }

    async fn profile_history(&self, id: String) -> Result<Vec<daemon_common::Revision>, ApiError> {
        self.revision_log()?
            .history(daemon_common::RevisionKind::Profile, &id)
            .map_err(revision_err)
    }

    async fn profile_at(&self, id: String, seq: u64) -> Result<ProfileSpec, ApiError> {
        let blob = self
            .revision_log()?
            .get_at(daemon_common::RevisionKind::Profile, &id, seq)
            .map_err(revision_err)?;
        ciborium::from_reader(blob.as_slice())
            .map_err(|e| ApiError::Other(format!("decode profile revision: {e}")))
    }

    async fn profile_revert(&self, id: String, seq: u64) -> Result<(), ApiError> {
        let spec = self.profile_at(id.clone(), seq).await?;
        self.profile_store()?.update(spec).map_err(profile_err)?;
        self.record_profile(
            &id,
            daemon_common::Author::Operator,
            &format!("revert to {seq}"),
        );
        Ok(())
    }

    async fn skill_history(&self, name: String) -> Result<Vec<daemon_common::Revision>, ApiError> {
        self.revision_log()?
            .history(daemon_common::RevisionKind::Skill, &name)
            .map_err(revision_err)
    }

    async fn skill_at(
        &self,
        name: String,
        seq: u64,
    ) -> Result<daemon_common::SkillBundle, ApiError> {
        let blob = self
            .revision_log()?
            .get_at(daemon_common::RevisionKind::Skill, &name, seq)
            .map_err(revision_err)?;
        ciborium::from_reader(blob.as_slice())
            .map_err(|e| ApiError::Other(format!("decode skill revision: {e}")))
    }

    async fn skill_revert(&self, name: String, seq: u64) -> Result<(), ApiError> {
        let skills = self.active_skills_store()?;
        if skills.is_bundled(&name) {
            return Err(ApiError::Conflict(format!(
                "skill `{name}` is binary-bundled and cannot be reverted"
            )));
        }
        let bundle = self.skill_at(name.clone(), seq).await?;
        skills
            .import_bundle(
                &bundle,
                daemon_common::Author::Operator,
                &format!("revert to {seq}"),
            )
            .map_err(|e| ApiError::Other(format!("skill revert: {e}")))
    }

    async fn skill_get(&self, name: String) -> Result<daemon_common::SkillBundle, ApiError> {
        self.active_skills_store()?
            .export_bundle(&name)
            .map_err(|e| ApiError::Other(format!("skill get: {e}")))
    }

    async fn skill_put(&self, bundle: daemon_common::SkillBundle) -> Result<(), ApiError> {
        let skills = self.active_skills_store()?;
        if skills.is_bundled(&bundle.name) {
            return Err(ApiError::Conflict(format!(
                "skill `{}` is binary-bundled and cannot be edited",
                bundle.name
            )));
        }
        skills
            .import_bundle(&bundle, daemon_common::Author::Operator, "skill_put")
            .map_err(|e| ApiError::Other(format!("skill put: {e}")))
    }

    async fn curator_list(
        &self,
        profile: Option<String>,
    ) -> Result<Vec<daemon_api::CuratorEntry>, ApiError> {
        let store = self.curator_store(profile)?;
        let usage = store.usage();
        let mut entries = Vec::new();
        // Live (discovered) skills, with their usage record (defaulting when untracked).
        for item in store.list() {
            let record = usage
                .and_then(|u| u.get(&item.name))
                .unwrap_or_default();
            entries.push(daemon_api::CuratorEntry {
                name: item.name.clone(),
                category: item.category,
                is_bundled: store.is_bundled(&item.name),
                usage: record,
            });
        }
        // Archived skills (out of discovery): surfaced with their archived-state record so an
        // operator can see + restore them.
        for name in store.archived() {
            let mut record = usage.and_then(|u| u.get(&name)).unwrap_or_default();
            record.state = daemon_common::SkillState::Archived;
            entries.push(daemon_api::CuratorEntry {
                name: name.clone(),
                category: None,
                is_bundled: store.is_bundled(&name),
                usage: record,
            });
        }
        Ok(entries)
    }

    async fn curator_pin(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        let store = self.curator_store(profile)?;
        self.curator_usage(&store)?.set_pinned(&name, true);
        Ok(())
    }

    async fn curator_unpin(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        let store = self.curator_store(profile)?;
        self.curator_usage(&store)?.set_pinned(&name, false);
        Ok(())
    }

    async fn curator_archive(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        self.curator_store(profile)?.archive(&name).map_err(skill_err)
    }

    async fn curator_restore(&self, profile: Option<String>, name: String) -> Result<(), ApiError> {
        self.curator_store(profile)?.restore(&name).map_err(skill_err)
    }

    async fn curator_run(
        &self,
        profile: Option<String>,
    ) -> Result<Vec<daemon_api::CuratorChange>, ApiError> {
        let store = self.curator_store(profile)?;
        let usage = self.curator_usage(&store)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let transitions = daemon_skills::apply_automatic_transitions(
            &usage.all(),
            now,
            daemon_skills::CuratorConfig::default(),
        );
        let mut changes = Vec::new();
        for t in transitions {
            match t.to {
                // Physically archive (also flips usage state to Archived). A skill already gone from
                // discovery (race) yields a not-found we tolerate.
                daemon_common::SkillState::Archived => {
                    if store.archive(&t.name).is_err() {
                        continue;
                    }
                }
                // Stale / reactivation are soft (the body stays discoverable): just flip the record.
                state => usage.set_state(&t.name, state),
            }
            changes.push(daemon_api::CuratorChange {
                name: t.name,
                from: t.from,
                to: t.to,
            });
        }
        Ok(changes)
    }
}

#[async_trait]
impl CredentialApi for NodeApiImpl {
    async fn credential_set(&self, profile: String, secret: String) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .set(&profile, &secret)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))
    }

    async fn credential_list(&self) -> Vec<CredentialInfo> {
        match &self.credentials {
            Some(store) => store.list_redacted(),
            None => Vec::new(),
        }
    }

    async fn credential_remove(&self, profile: String) -> Result<(), ApiError> {
        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .remove(&profile)
            .map_err(|e| ApiError::Other(format!("credential remove: {e}")))
    }
}

#[async_trait]
impl AuthApi for NodeApiImpl {
    async fn auth_begin(&self, req: AuthBeginRequest) -> Result<AuthBeginResponse, ApiError> {
        let flows = self
            .auth_flows
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("interactive auth not available".into()))?;
        flows.begin(req).await
    }

    async fn auth_complete(
        &self,
        req: AuthCompleteRequest,
    ) -> Result<AuthCompleteResponse, ApiError> {
        let flows = self
            .auth_flows
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("interactive auth not available".into()))?;
        // Pull the parked flow out (and its bind request) *before* awaiting the family completion, so
        // the registry lock is never held across the network round-trip the family performs.
        let (flow, bind) = flows.take(&req.flow_id)?;
        let outcome = flow.complete(&req.callback).await?;

        // The credential ref: a bind-supplied override wins over the family-derived default, so an
        // operator can pin where the blob lands; otherwise the family names it (e.g. by resolved user).
        let credential_ref = bind
            .as_ref()
            .and_then(|b| b.credential_ref.clone())
            .unwrap_or_else(|| outcome.credential_ref.clone());

        let store = self
            .credentials
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("credential management not available".into()))?;
        store
            .set(&credential_ref, &outcome.credential_blob)
            .map_err(|e| ApiError::Other(format!("credential set: {e}")))?;

        // Optional account→profile bind: attach (or replace) the BoundAccount on the target profile so
        // the transport's account bring-up (`AccountProvisioning::bound_accounts`) discovers it.
        let mut bound_profile = None;
        if let Some(bind) = bind {
            let profiles = self.profile_store()?;
            let mut spec = profiles
                .get(bind.profile.as_str())
                .map_err(profile_err)?
                .ok_or_else(|| ApiError::Other(format!("unknown profile: {}", bind.profile)))?;
            let transport_instance = bind
                .transport_instance
                .clone()
                .unwrap_or_else(|| outcome.transport_instance.clone());
            spec.bound_accounts
                .retain(|a| a.transport_instance != transport_instance.as_str());
            spec.bound_accounts
                .push(BoundAccount::new(transport_instance.as_str(), &credential_ref));
            profiles.update(spec).map_err(profile_err)?;
            bound_profile = Some(bind.profile.clone());
        }

        // A completed account bind can change which transport instances route to which profile:
        // rebuild routing so the freshly-authenticated account is reachable without a restart.
        if bound_profile.is_some() {
            self.rebuild_routing();
        }

        Ok(AuthCompleteResponse {
            credential_ref,
            account_label: outcome.account_label,
            transport_instance: outcome.transport_instance,
            bound_profile,
        })
    }

    async fn auth_cancel(&self, flow_id: String) -> Result<(), ApiError> {
        if let Some(flows) = self.auth_flows.as_ref() {
            flows.cancel(&flow_id);
        }
        Ok(())
    }

    async fn auth_providers(&self) -> Vec<AuthProviderInfo> {
        self.auth_flows
            .as_ref()
            .map(|f| f.providers())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Live interactive sessions (the §17 actor, exposed via the poll/drain model)
// ---------------------------------------------------------------------------

type Drain = Arc<Mutex<VecDeque<Outbound>>>;
type Pending = Arc<Mutex<HashMap<ReqId, oneshot::Sender<HostResponse>>>>;
type Merged = Arc<Mutex<MergedLog>>;
/// A live session's outbound delivery targets (where its replies post). Seeded from the opening
/// origin; re-pointed by `handover`. The actual posting to a Primary is a chat transport's job (P5);
/// here it is the authoritative session-owned routing state.
type Delivery = Arc<Mutex<Vec<DeliveryTarget>>>;

/// The authoritative, **non-destructive** merged session event log for one live session: one
/// `seq`-stamped timeline across both directions (inbound commands/responses, outbound events +
/// raised host requests). Unlike the destructive `drain` (single-consumer `poll`), this is the
/// multi-surface observability surface — N consumers each page from their own cursor (`log_after`)
/// or hold a live push subscription (`subscribe`), and never steal each other's events.
struct MergedLog {
    /// The next `seq` to assign (one counter across both directions).
    next_seq: u64,
    /// The full ordered history (retained so a late joiner can backfill from any cursor).
    entries: Vec<SessionLogEntry>,
    /// The live fan-out to push subscribers.
    tx: broadcast::Sender<SessionLogEntry>,
}

impl MergedLog {
    fn new() -> Self {
        let (tx, _rx) = broadcast::channel(256);
        // Seq starts at 1 so the `after_seq` cursor convention (exclusive lower bound; 0 = "from the
        // start") can address the very first entry.
        Self {
            next_seq: 1,
            entries: Vec::new(),
            tx,
        }
    }

    /// Stamp the next `seq`, record the entry, fan it out to live subscribers, and return the
    /// stamped entry (so an in-process pusher delivers exactly what subscribers see).
    fn append(
        &mut self,
        direction: Direction,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) -> SessionLogEntry {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = SessionLogEntry {
            seq,
            direction,
            origin,
            disposition,
            payload,
        };
        self.entries.push(entry.clone());
        // A send error only means there are no live subscribers; the history retains the entry.
        let _ = self.tx.send(entry.clone());
        entry
    }

    /// A non-destructive page of entries with `seq > after_seq` (up to `max`, 0 = all).
    fn page(&self, after_seq: u64, max: u32) -> LogPageView {
        let head_seq = self.next_seq.saturating_sub(1);
        let mut entries = Vec::new();
        for e in self.entries.iter().filter(|e| e.seq > after_seq) {
            if max != 0 && entries.len() >= max as usize {
                break;
            }
            entries.push(e.clone());
        }
        let next_seq = entries.last().map(|e| e.seq).unwrap_or(after_seq);
        LogPageView {
            entries,
            next_seq,
            head_seq,
        }
    }

    /// A push stream that backfills `seq > after_seq` from history, then continues live. The caller
    /// holds the log mutex while calling this, so the backlog snapshot and the live subscription are
    /// taken atomically (no entry can slip between them).
    fn subscribe(&self, after_seq: u64) -> LogStream {
        let backlog: Vec<SessionLogEntry> = self
            .entries
            .iter()
            .filter(|e| e.seq > after_seq)
            .cloned()
            .collect();
        let rx = self.tx.subscribe();
        let live = BroadcastStream::new(rx).filter_map(|r| async move { r.ok() });
        stream::iter(backlog).chain(live).boxed()
    }
}

struct LiveSession {
    handle: AgentHandle,
    drain: Drain,
    pending: Pending,
    /// The non-destructive merged event log (multi-surface observability).
    log: Merged,
    /// Where this session's outbound replies post (the `Primary`) + passive `Spectator`s.
    delivery: Delivery,
    /// The event pump task; aborted when the session is dropped.
    pump: JoinHandle<()>,
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.pump.abort();
    }
}

struct LiveSessions {
    sessions: DashMap<SessionId, LiveSession>,
    builder: SessionEngineBuilder,
    /// The durable session store: read at `ensure` to restore a session's persisted overlay (so a
    /// live model/tools/mode override survives an actor respawn) and to record its bound profile.
    store: Arc<dyn SessionStore>,
    /// The verifiable-journal store + signer, when journaling is enabled for live sessions.
    journal: Mutex<Option<JournalConfig>>,
    /// The §12 workspace-checkpoint store, when wired: a `RewindTo` rolls the filesystem back to the
    /// sealed-off range's earliest pre-mutation checkpoint (conversation-rewind spec §6).
    checkpoints: Mutex<Option<Arc<dyn daemon_core::CheckpointStore>>>,
    /// The §4.3 background-spawn materializer, when configured: lets a live session's `Effect::Spawn`
    /// materialize an attached non-joining review child without parking (fire-and-forget).
    background: Mutex<Option<Arc<crate::background::BackgroundSpawner>>>,
    /// The per-session live edit-approval policy (shared with `NodeApiImpl::session_modes`), read by
    /// each session's [`ParkingHandler`] to auto-allow / deny without parking a human.
    modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
    /// In-process outbound push sinks keyed by transport instance (daemon-event-io-spec §5.9.3): a
    /// registered sink receives every outbound entry of every session whose `Primary` it owns,
    /// resolved live by the per-session pump (so handover demotion stops/starts delivery for free).
    /// Shared with each pump task; a missing instance simply means no in-process push (pull-only).
    sinks: Arc<DashMap<TransportId, Arc<dyn DeliverySink>>>,
}

impl LiveSessions {
    fn new(
        builder: SessionEngineBuilder,
        modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
        store: Arc<dyn SessionStore>,
    ) -> Self {
        Self {
            sessions: DashMap::new(),
            builder,
            store,
            journal: Mutex::new(None),
            checkpoints: Mutex::new(None),
            background: Mutex::new(None),
            modes,
            sinks: Arc::new(DashMap::new()),
        }
    }

    fn set_journal(&self, cfg: JournalConfig) {
        *self.journal.lock().unwrap() = Some(cfg);
    }

    fn set_checkpoints(&self, checkpoints: Arc<dyn daemon_core::CheckpointStore>) {
        *self.checkpoints.lock().unwrap() = Some(checkpoints);
    }

    fn set_background(&self, background: Arc<crate::background::BackgroundSpawner>) {
        *self.background.lock().unwrap() = Some(background);
    }

    /// The handle for `session` only if it is already resident (does not spawn a new actor).
    fn handle_if_live(&self, session: &SessionId) -> Option<AgentHandle> {
        self.sessions.get(session).map(|s| s.handle.clone())
    }

    /// Spawn (or reuse) the actor for `session`, returning its handle. The `profile` selects which
    /// profile bundle a *new* session's engine is built from (the routing agent-selection seam); a
    /// resident session ignores it (the first `ensure` binds the profile — bindings are sticky).
    ///
    /// The session's persisted [`SessionOverlay`] is read from the store and applied on top of the
    /// bound profile at build time, so a live model/tools/approval override is **restored** when the
    /// actor is (re)spawned (e.g. after a host restart). The first `ensure` also records the bound
    /// profile in the store metadata, so the durable path can later re-resolve the same profile.
    async fn ensure(&self, session: &SessionId, profile: Option<ProfileRef>) -> AgentHandle {
        if let Some(s) = self.sessions.get(session) {
            return s.handle.clone();
        }
        // Read (and, for a new session, establish) the host-level session metadata: the bound
        // profile + persisted overlay. A read-modify-write keeps the overlay intact when we are only
        // stamping the bound profile for the first time.
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        if meta.bound_profile.is_none() && profile.is_some() {
            meta.bound_profile = profile.clone();
            let _ = self.store.set_session_meta(session, meta.clone()).await;
        }
        let overlay = decode_overlay(&meta.overlay);
        let engine = (self.builder)(session.clone(), profile, &overlay);
        let drain: Drain = Arc::new(Mutex::new(VecDeque::new()));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let log: Merged = Arc::new(Mutex::new(MergedLog::new()));
        let delivery: Delivery = Arc::new(Mutex::new(Vec::new()));
        // A per-session journal feeder (keyed by SessionId), shared by the event pump and the
        // request handler so the live transcript is sealed per turn into the unified journal.
        let feeder: Option<Arc<JournalFeeder>> = self.journal.lock().unwrap().as_ref().map(|cfg| {
            let sink = JournalSink::new(
                cfg.store.clone(),
                cfg.signer.clone(),
                JournalStreamId::session(session),
            );
            Arc::new(JournalFeeder::new(Arc::new(sink)))
        });
        let host = Arc::new(ParkingHandler {
            drain: drain.clone(),
            pending: pending.clone(),
            log: log.clone(),
            journal: feeder.clone(),
            session: session.clone(),
            background: self.background.lock().unwrap().clone(),
            modes: self.modes.clone(),
        });
        let handle = spawn_agent_session(engine, host);

        // Pump §17 events from the actor broadcast into the destructive drain queue (lossless until
        // polled), record them on the non-destructive merged log (outbound / Context), and feed the
        // verifiable journal (coalesced finished blocks, sealed per turn) when enabled.
        let mut rx = handle.subscribe();
        let pump_drain = drain.clone();
        let pump_log = log.clone();
        let pump_journal = feeder.clone();
        // Clones for the in-process push path (§5.9.3): the pump re-reads the session's *current*
        // delivery targets per event and pushes the just-recorded entry to any registered sink owning
        // a target, so handover (a demoted `Primary`) silently stops one sink and starts the next.
        let pump_delivery = delivery.clone();
        let pump_sinks = self.sinks.clone();
        let pump = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        // Stamp + record on the merged log, capturing the freshly-stamped entry so the
                        // push path delivers exactly what subscribers see (one seq, one shape).
                        let entry = pump_log.lock().unwrap().append(
                            Direction::Outbound,
                            engine_origin(),
                            Disposition::Context,
                            SessionPayload::Event(ev.clone()),
                        );
                        let frame = Outbound::Event(ev);
                        pump_drain.lock().unwrap().push_back(frame.clone());
                        if let Some(feeder) = &pump_journal {
                            feeder.feed(&frame).await;
                        }
                        // In-process push: replies post to where the *current* `Primary` points, so
                        // snapshot the live targets (dropping the lock before any await) and push the
                        // just-recorded entry to the registered sink owning each `Primary`. Re-reading
                        // the targets every event is what makes handover free: a demoted matrix
                        // `Primary` falls to `Spectator` (stops receiving) and the new GUI `Primary`
                        // starts, with no work here. Passive `Spectator`s observe via the pull path
                        // (`subscribe`); pull subscribers are unaffected by this additive push.
                        let primaries: Vec<DeliveryTarget> = pump_delivery
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|t| t.kind == SinkKind::Primary)
                            .cloned()
                            .collect();
                        for target in primaries {
                            if let Some(sink) = pump_sinks.get(&target.transport) {
                                let sink = sink.clone();
                                sink.deliver(target, entry.clone()).await;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        self.sessions.insert(
            session.clone(),
            LiveSession {
                handle: handle.clone(),
                drain,
                pending,
                log,
                delivery,
                pump,
            },
        );
        handle
    }

    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // No external attribution supplied: default to the generic `api` origin.
        self.submit_from(session, api_origin(), command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        match command {
            AgentCommand::StartTurn { input, request_id } => {
                // Opening command: spawn-if-absent, then run the turn in the background so events
                // (including the terminal `TurnFinished`) flow to the drain queue for `poll`.
                let handle = self.ensure(&session, None).await;
                // Seed the session's Primary reply sink from the opening origin (where replies post by
                // default), unless one is already in force. Handover re-points it later.
                self.seed_primary(&session, &origin);
                // Record the inbound command on the merged log first, so an observer sees what was
                // submitted ahead of the engine's replies (StartTurn enters the conversation),
                // attributed to the submitting surface's `origin`.
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Context,
                    SessionPayload::Command(AgentCommand::StartTurn {
                        input: input.clone(),
                        request_id,
                    }),
                );
                tokio::spawn(async move {
                    let _ = handle.start_turn(input).await;
                });
                Ok(())
            }
            AgentCommand::Interrupt { reason } => {
                let handle = self.existing(&session)?;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Interrupt {
                        reason: reason.clone(),
                    }),
                );
                handle.interrupt(reason).await;
                Ok(())
            }
            AgentCommand::Shutdown => {
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Shutdown),
                );
                if let Some((_, s)) = self.sessions.remove(&session) {
                    s.handle.shutdown().await;
                }
                Ok(())
            }
            AgentCommand::Steer { text, request_id } => {
                // Steer-when-idle opens a fresh turn; mid-turn it is drained at a phase boundary.
                // Either way the ack + any turn events flow to the drain queue via the pump.
                let handle = self.ensure(&session, None).await;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Context,
                    SessionPayload::Command(AgentCommand::Steer {
                        text: text.clone(),
                        request_id,
                    }),
                );
                handle.steer(request_id, text).await;
                Ok(())
            }
            AgentCommand::Observe { input, request_id } => {
                // Context-only append (no turn): spawn-if-absent so the chatter has a conversation to
                // land in, record it as context, then hand it to the actor — which folds it in when
                // idle or queues it for the following turn when busy (event-io §5.9). No turn starts.
                let handle = self.ensure(&session, None).await;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Context,
                    SessionPayload::Command(AgentCommand::Observe {
                        input: input.clone(),
                        request_id,
                    }),
                );
                handle.observe(request_id, input).await;
                Ok(())
            }
            AgentCommand::Snapshot { request_id } => {
                let handle = self.existing(&session)?;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Snapshot { request_id }),
                );
                handle.snapshot(request_id).await;
                Ok(())
            }
            AgentCommand::RewindTo { anchor, request_id } => {
                // Conversation rewind (spec §4): the engine interrupts any live turn, truncates +
                // reconstructs + bumps epoch + emits `Rewound`; the host then seals the durable
                // journal and rolls the workspace back to the sealed-off range's earliest checkpoint.
                let handle = self.existing(&session)?;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::RewindTo {
                        anchor: anchor.clone(),
                        request_id,
                    }),
                );
                let outcome = handle
                    .rewind_to(request_id, anchor)
                    .await
                    .map_err(|e| ApiError::Other(e.to_string()))?;
                // A bare `RewindTo` command rewinds the conversation *and* rolls the workspace back —
                // the historical behavior. The finer conversation-only rewind is reachable via the
                // unified `ControlApi::rewind` op with `restore_workspace = false`.
                self.seal_and_rollback_after_rewind(&session, &outcome, true)
                    .await;
                Ok(())
            }
            _ => Err(ApiError::Unsupported("unknown agent command".into())),
        }
    }

    /// Apply the durable side-effects of a conversation rewind for this live session: seal the
    /// journal (when journaled) and roll the workspace back to the dropped range's earliest
    /// checkpoint. Delegates to the shared [`apply_rewind_side_effects`] helper so the live path and
    /// the managed/fleet path ([`crate::unit::LiveAgentSession`]) stay byte-for-byte consistent.
    async fn seal_and_rollback_after_rewind(
        &self,
        session: &SessionId,
        outcome: &daemon_core::RewindOutcome,
        restore_workspace: bool,
    ) {
        let journaled = self.journal.lock().unwrap().is_some();
        let checkpoints = self.checkpoints.lock().unwrap().clone();
        apply_rewind_side_effects(
            &self.store,
            checkpoints.as_ref(),
            journaled,
            session,
            outcome,
            restore_workspace,
        )
        .await;
    }

    /// Rewind a *resident* session's transcript at `anchor` (in-process engine truncate + epoch bump),
    /// then apply the shared durable side-effects honoring `restore_workspace`. The host-spec unified
    /// rewind seam for the live path; backs [`NodeApiImpl::rewind`] for a live session.
    async fn rewind_resident(
        &self,
        session: &SessionId,
        anchor: daemon_protocol::RewindAnchor,
        restore_workspace: bool,
    ) -> Result<(), ApiError> {
        let handle = self
            .handle_if_live(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let outcome = handle
            .rewind_to(daemon_common::ReqId(0), anchor)
            .await
            .map_err(|e| ApiError::Other(e.to_string()))?;
        self.seal_and_rollback_after_rewind(session, &outcome, restore_workspace)
            .await;
        Ok(())
    }

    /// Append an inbound entry to a live session's merged log (no-op if the session is gone),
    /// attributed to `origin` so per-event provenance is preserved on the authoritative log.
    fn record_inbound(
        &self,
        session: &SessionId,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) {
        if let Some(s) = self.sessions.get(session) {
            s.log
                .lock()
                .unwrap()
                .append(Direction::Inbound, origin, disposition, payload);
        }
    }

    /// Record an observability-only transport/meta event (`Disposition::Transport`) on the merged log
    /// — the "GUI attached" / presence / receipt channel. It lands on the live log + broadcast only
    /// (never the engine, never the journal), so it is cache-safe by construction.
    fn record_meta(
        &self,
        session: &SessionId,
        origin: Origin,
        kind: String,
        body: Vec<u8>,
    ) -> Result<(), ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        s.log.lock().unwrap().append(
            Direction::Inbound,
            origin,
            Disposition::Transport,
            SessionPayload::Meta { kind, body },
        );
        Ok(())
    }

    /// Seed the session's `Primary` reply sink from the opening origin if none is set yet.
    fn seed_primary(&self, session: &SessionId, origin: &Origin) {
        if let Some(s) = self.sessions.get(session) {
            let mut targets = s.delivery.lock().unwrap();
            if !targets.iter().any(|t| t.kind == SinkKind::Primary) {
                targets.push(origin.primary_target());
            }
        }
    }

    /// Seed the session's `Primary` reply sink to an already-resolved `target` if none is set yet —
    /// the routed-submit counterpart of [`Self::seed_primary`], honoring a binding's pinned delivery.
    fn seed_primary_target(&self, session: &SessionId, target: DeliveryTarget) {
        if let Some(s) = self.sessions.get(session) {
            let mut targets = s.delivery.lock().unwrap();
            if !targets.iter().any(|t| t.kind == SinkKind::Primary) {
                targets.push(target);
            }
        }
    }

    /// The session's current delivery targets (empty if the session is gone).
    fn delivery_targets(&self, session: &SessionId) -> Vec<DeliveryTarget> {
        match self.sessions.get(session) {
            Some(s) => s.delivery.lock().unwrap().clone(),
            None => Vec::new(),
        }
    }

    /// Every distinct `Primary` delivery target across all live sessions — the resolution of a cron
    /// job's `deliver = "all"` (broadcast a run result to every active conversation's reply sink).
    /// Deduplicated by `(transport, route)` so two sessions posting to the same chat deliver once.
    fn all_primary_targets(&self) -> Vec<DeliveryTarget> {
        let mut out: Vec<DeliveryTarget> = Vec::new();
        for s in self.sessions.iter() {
            for t in s.delivery.lock().unwrap().iter() {
                if t.kind == SinkKind::Primary
                    && !out
                        .iter()
                        .any(|e| e.transport == t.transport && e.route == t.route)
                {
                    out.push(t.clone());
                }
            }
        }
        out
    }

    /// Push a synthesized outbound `entry` to the registered sink owning `target`'s transport
    /// (post-settle cron delivery). A no-op when no sink is registered (pull-only transport).
    async fn push_to_target(&self, target: DeliveryTarget, entry: SessionLogEntry) {
        if let Some(sink) = self.sinks.get(&target.transport).map(|s| s.clone()) {
            sink.deliver(target, entry).await;
        }
    }

    /// The live sessions a transport instance owns for delivery (daemon-event-io-spec §5.9.3): every
    /// resident session whose `Primary` [`DeliveryTarget`] names `transport`. An on-demand scan of
    /// the live table (called on (re)connect, not per-event), so O(live sessions) is acceptable.
    fn delivery_sessions(&self, transport: &TransportId) -> Vec<SessionId> {
        self.sessions
            .iter()
            .filter(|s| {
                s.delivery
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|t| t.kind == SinkKind::Primary && &t.transport == transport)
            })
            .map(|s| s.key().clone())
            .collect()
    }

    /// Every live (in-memory, submit/poll) session id — the visibility half of the unified roster
    /// (these never appear in the durable `list_sessions` until `assign`). An on-demand snapshot scan.
    fn live_ids(&self) -> Vec<SessionId> {
        self.sessions.iter().map(|s| s.key().clone()).collect()
    }

    /// Register an in-process push [`DeliverySink`] for `transport` (a live handle, replacing any
    /// prior sink for the instance). The per-session pump picks it up on the next event.
    fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>) {
        self.sinks.insert(transport, sink);
    }

    /// Drop the in-process push sink for `transport` (delivery for that instance reverts to pull).
    fn unregister_delivery_sink(&self, transport: &TransportId) {
        self.sinks.remove(transport);
    }

    /// Re-point the session's `Primary` to `target`: any prior `Primary` is demoted to `Spectator`,
    /// any existing entry for the same transport+route is replaced, and `target` is installed as the
    /// new `Primary`.
    fn handover(&self, session: &SessionId, target: DeliveryTarget) -> Result<(), ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut targets = s.delivery.lock().unwrap();
        for t in targets.iter_mut() {
            if t.kind == SinkKind::Primary {
                t.kind = SinkKind::Spectator;
            }
        }
        targets.retain(|t| !(t.transport == target.transport && t.route == target.route));
        targets.push(DeliveryTarget::new(
            target.transport,
            target.route.0,
            SinkKind::Primary,
        ));
        Ok(())
    }

    /// Non-destructive cursor page of a live session's merged log (empty if the session is gone).
    fn log_after(&self, session: &SessionId, after_seq: u64, max: u32) -> LogPageView {
        match self.sessions.get(session) {
            Some(s) => s.log.lock().unwrap().page(after_seq, max),
            None => LogPageView::default(),
        }
    }

    /// A live push subscription to a session's merged log (empty stream if the session is gone).
    fn subscribe(&self, session: &SessionId, after_seq: u64) -> LogStream {
        match self.sessions.get(session) {
            Some(s) => s.log.lock().unwrap().subscribe(after_seq),
            None => stream::empty().boxed(),
        }
    }

    fn existing(&self, session: &SessionId) -> Result<AgentHandle, ApiError> {
        self.sessions
            .get(session)
            .map(|s| s.handle.clone())
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))
    }

    fn poll(&self, session: &SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut q = s.drain.lock().unwrap();
        let take = if max == 0 {
            q.len()
        } else {
            (max as usize).min(q.len())
        };
        Ok(q.drain(..take).collect())
    }

    fn respond(&self, session: &SessionId, response: HostResponse) -> Result<(), ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let tx = s.pending.lock().unwrap().remove(&response.request_id);
        match tx {
            Some(tx) => {
                // The answer to a raised host request enters the conversation (inbound / Context).
                s.log.lock().unwrap().append(
                    Direction::Inbound,
                    api_origin(),
                    Disposition::Context,
                    SessionPayload::Response(response.clone()),
                );
                let _ = tx.send(response);
                Ok(())
            }
            None => Err(ApiError::Other(format!(
                "no parked request {:?} on session {}",
                response.request_id, session
            ))),
        }
    }

    async fn interrupt(&self, session: &SessionId) -> bool {
        match self.sessions.get(session) {
            Some(s) => {
                s.handle.interrupt(Some("control cancel".into())).await;
                true
            }
            None => false,
        }
    }
}

/// The durable side-effects of a conversation rewind (conversation-rewind spec §6), factored out so
/// the live path ([`LiveSessions::seal_and_rollback_after_rewind`]) and the managed/fleet path
/// ([`crate::unit::LiveAgentSession`]) apply *exactly* the same seal + rollback. Previously only the
/// live path sealed/rolled-back, so a rewind on a managed engine silently skipped both — this is the
/// shared helper both now call.
///
/// - **Seal** (when `journaled`): append a `JournalSeal` at the journal head so the dropped tail is
///   marked `sealed_after` while the audit log stays complete.
/// - **Rollback** (when `restore_workspace` and there are dropped tool calls): restore the earliest
///   pre-mutation checkpoint among the dropped calls, undoing every later mutation in the sealed
///   range. A read-only rewound range (no checkpoints) leaves the filesystem untouched.
pub(crate) async fn apply_rewind_side_effects(
    store: &Arc<dyn SessionStore>,
    checkpoints: Option<&Arc<dyn daemon_core::CheckpointStore>>,
    journaled: bool,
    session: &SessionId,
    outcome: &daemon_core::RewindOutcome,
    restore_workspace: bool,
) {
    if journaled {
        let stream = JournalStreamId::session(session);
        let head = store.load_journal(&stream, u64::MAX, 1).await.head_cursor;
        let recorded_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = store
            .record_journal_seal(
                &stream,
                daemon_store::JournalSeal {
                    seal_cursor: head,
                    retained_turns: outcome.retained_turns as u64,
                    epoch: outcome.epoch.0,
                    recorded_unix,
                },
            )
            .await;
    }

    if restore_workspace && !outcome.dropped_call_ids.is_empty() {
        if let Some(store) = checkpoints {
            let dropped: std::collections::HashSet<&str> =
                outcome.dropped_call_ids.iter().map(|s| s.as_str()).collect();
            let mut matching: Vec<_> = store
                .list(Some(session.as_str()))
                .await
                .into_iter()
                .filter(|r| dropped.contains(r.call_id.as_str()))
                .collect();
            matching.sort_by_key(|r| r.created_unix);
            if let Some(earliest) = matching.first() {
                let _ = store.restore(earliest).await;
            }
        }
    }
}

/// The session sub-surface's host handler: park each blocking §17 request into the drain queue and
/// a pending table, await its `respond`. Events and parked requests thus ride one ordered queue
/// (daemon-ffi-spec §3.3).
struct ParkingHandler {
    drain: Drain,
    pending: Pending,
    /// The session's non-destructive merged log, so a raised request is observable to every surface.
    log: Merged,
    /// The per-session journal feeder, so a raised request graduates into a durable request block.
    journal: Option<Arc<JournalFeeder>>,
    /// This session's id (the parent of any background spawn it raises).
    session: SessionId,
    /// The §4.3 background-spawn materializer, when configured.
    background: Option<Arc<crate::background::BackgroundSpawner>>,
    /// The shared per-session live edit-approval policy, consulted on an `Approval` request to
    /// auto-allow / deny without parking a human (in lockstep with the engine's snapshot policy).
    modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
}

#[async_trait]
impl HostRequestHandler for ParkingHandler {
    async fn request(&self, req: HostRequest) -> HostResponse {
        // §4.3 fire-and-forget spawn: materialize the attached non-joining child immediately and
        // return — never park (parking would block the parent turn, defeating fire-and-forget).
        if let HostRequestKind::Spawn { spec } = &req.kind {
            let child = match &self.background {
                Some(bg) => bg
                    .spawn(&self.session, daemon_common::Epoch::ZERO, spec, None)
                    .await
                    .unwrap_or_else(|| self.session.clone()),
                None => self.session.clone(),
            };
            return HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Spawned(child),
            };
        }
        // Live edit-approval gate: an `Approval` reaching the host has already cleared the engine's
        // policy gate as `Ask`, but consult the live session policy as the host-side authority so a
        // GUI auto-allow / deny mode answers inline without parking a human (mirrors hermes' ACP
        // adapter resolving the mode in-process). `Ask`/`AcceptEdits` fall through to parking.
        if let HostRequestKind::Approval { .. } = &req.kind {
            match self.modes.get(&self.session).map(|p| *p) {
                Some(daemon_core::ApprovalPolicy::AutoAllow) => {
                    return HostResponse {
                        request_id: req.request_id,
                        body: HostResponseBody::Approved(true),
                    };
                }
                Some(daemon_core::ApprovalPolicy::Deny) => {
                    return HostResponse {
                        request_id: req.request_id,
                        body: HostResponseBody::Approved(false),
                    };
                }
                _ => {}
            }
        }
        let (tx, rx) = oneshot::channel();
        let request_id = req.request_id;
        self.pending.lock().unwrap().insert(request_id, tx);
        // Record the raised request on the merged log (outbound / Context) under the unified seq, so
        // it shares one ordered timeline with events and the eventual inbound response.
        self.log.lock().unwrap().append(
            Direction::Outbound,
            engine_origin(),
            Disposition::Context,
            SessionPayload::Request(req.clone()),
        );
        let frame = Outbound::Request(req);
        if let Some(feeder) = &self.journal {
            feeder.feed(&frame).await;
        }
        self.drain.lock().unwrap().push_back(frame);
        match rx.await {
            Ok(resp) => resp,
            // The session was dropped before an answer arrived: decline safely.
            Err(_) => HostResponse {
                request_id,
                body: HostResponseBody::Approved(false),
            },
        }
    }
}
