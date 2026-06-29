// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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

use crate::auth::PendingAuthFlows;
use crate::credstore::CredentialStore;
use crate::engine_incarnation::JournalConfig;
use crate::journal::{JournalFeeder, JournalSink};
use crate::profiles::ProfileStore;
use crate::routing::RoutingRegistry;
use crate::supervisor::{HealthStatus, SupervisorObserver};
use crate::FleetControl;
use arc_swap::{ArcSwap, ArcSwapOption};
use async_trait::async_trait;
use daemon_activation::ActivationManager;
use daemon_api::{
    from_cbor,
    to_cbor,
    AcpAgentEntry,
    AcpSource,
    ActionMenu,
    AdapterInfo,
    ApiError,
    ApprovalInfo,
    ApprovalMode,
    AuthApi,
    AuthBeginRequest,
    AuthBeginResponse,
    AuthCompleteRequest,
    AuthCompleteResponse,
    AuthProviderInfo,
    BlobRef,
    BlobStat,
    BoundAccount,
    ByteRange,
    ChannelJoinDetails,
    ChatRoute,
    CommandInvocation,
    CommandOutput,
    CommandScope,
    CommandSpec,
    ContactInfo,
    ControlApi,
    // C1 parameter structs (multi-arg interface methods).
    ConvHistoryArgs,
    ConvSendArgs,
    ConversationInfo,
    CreateConversationDetails,
    CredentialApi,
    CredentialInfo,
    DeliverySink,
    Distribution,
    EventsPage,
    FleetReport,
    FsContent,
    FsEntry,
    FsRevision,
    FsRoot,
    FsRootId,
    FsRootKind,
    FsSearchPage,
    FsSearchQuery,
    FsWatchAfterArgs,
    FsWatchPageView,
    FsWriteArgs,
    FsWriteFromBlobArgs,
    HealthReport,
    JournalPageView,
    JournalRecord,
    JournalRecordPayload,
    Lifecycle as ApiLifecycle,
    LogPageView,
    LogStream,
    LogStreamItem,
    ManageEventView,
    MemberBanArgs,
    MemberInviteArgs,
    MemberRemoveArgs,
    MemberSetRoleArgs,
    ModelApi,
    ModelDescriptor,
    ModelQuantizeArgs,
    ModelRecommendArgs,
    NodeEvent,
    NodeEventStream,
    Outbound,
    Participant,
    ProfileApi,
    ProfileInfo,
    ProfileSpec,
    ProviderSelector,
    RecordMetaArgs,
    RoomInfo,
    ServiceHealth,
    SessionApi,
    SessionDetail,
    SessionInfo,
    SessionMetaPatch,
    SessionOverlay,
    SessionPage,
    SessionQuery,
    SessionRole,
    SessionScope,
    SessionSearchHit,
    SessionState,
    StatsReport,
    SubmitAsArgs,
    SupportsContacts,
    SupportsConversations,
    SupportsDirectory,
    SupportsMembership,
    TelemetryDump,
    TransportInstanceInfo,
    TreeReport,
    UnitNode,
};
use daemon_common::cursored::CursoredRing;
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
    current_trace, decode_entry, verify_segment, JournalPayload, Metrics, SegmentInput,
    TraceSigner, VerifyingKey, GENESIS_ROOT,
};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;
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
    /// Held behind an [`ArcSwap`] so it is *hot-swappable*: a profile/auth change can rebuild the
    /// routing table live (via [`NodeApiImpl::rebuild_routing`]) without restarting the node. An
    /// in-flight `submit_routed` resolves against one immutable snapshot while a swap publishes the
    /// next snapshot without taking a read lock.
    routing: Arc<ArcSwap<RoutingRegistry>>,
    /// The pin-free *base* routing registry (the static [`NodeApiImpl::with_routing`] table, or empty
    /// for the passthrough/builder cases). The live `routing` above is this base with the durable
    /// chat→session pins (`chat_pins`) layered on by [`NodeApiImpl::rebuild_routing`]; keeping the
    /// base separate lets a pin reload re-layer pins without losing the operator's binding table.
    routing_base: Arc<ArcSwap<RoutingRegistry>>,
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
    /// The transport-adapter registry (daemon-transport-adapter-spec.md §3.4): the node's
    /// self-describing events-IO adapters, enumerated read-only by `transport_adapters`. Empty by
    /// default (skeleton: lifecycle still lives in `bins/daemon`; this only feeds the descriptor
    /// enumeration). Installed by the assembling binary via [`NodeApiImpl::with_adapters`].
    adapters: Arc<ArcSwap<crate::adapters::AdapterRegistry>>,
    /// The lazily-opened verifiable-journal writer for the `node-management` stream: management
    /// mutations (`conv_*`/`member_*`) are recorded + sealed onto it so the audit chains per op.
    /// `None` until the first mutation (and stays `None` when journaling is disabled).
    mgmt_journal: Arc<std::sync::Mutex<Option<Arc<JournalSink>>>>,
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
    /// The node-wide event feed (L3 `EventsSince`): a retained, cursored ring of payload-free
    /// notifications (roster/meta/approval/session-advanced/download/resync) that lets a client learn
    /// what changed out of focus without polling and re-baseline after a gap. `None` => `events_*`
    /// serve empty (a node assembled without the feed).
    node_events: Option<Arc<NodeEventFeed>>,
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
    /// The daemon-authoritative command catalog backing `command_list`/`command_invoke`: built-in
    /// node-op commands unified with the engine profile's [`CommandProvider`](daemon_core::CommandProvider)
    /// contributions (`/lcm`, `/memory`, …). Empty => the command surface resolves to its defaulted
    /// empty catalog / [`ApiError::Unsupported`]. Held behind an [`ArcSwapOption`] so the assembling
    /// binary can bind it *after* the node is wrapped in an `Arc` (see [`NodeApiImpl::set_commands`]),
    /// since the registry needs node-resolved provider handles the node construction does not own.
    commands: Arc<ArcSwapOption<crate::commands::CommandRegistry>>,
}

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
            acp: None,
            last_acp: Arc::new(std::sync::RwLock::new(Vec::new())),
            checkpoints: None,
            auth_flows: None,
            fleet_events: None,
            node_events: None,
            workspace: None,
            blobs: None,
            cron: None,
            commands: Arc::new(ArcSwapOption::empty()),
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

    /// The gated workspace write shared by `fs_write` and `fs_write_from_blob`: `Workspace`/`Session`
    /// roots only, sensitive-path + per-session `Deny` gate (overridable by `force`), a pre-mutation
    /// checkpoint for session roots, and the `Conflict`-on-stale-`base_revision` guard inside
    /// `WorkspaceFs::write`.
    async fn write_gated(&self, args: FsWriteArgs) -> Result<FsRevision, ApiError> {
        let FsWriteArgs {
            root,
            path,
            bytes,
            base_revision,
            force,
        } = args;
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
    pub fn spawn_adapters(self: &Arc<Self>) -> Vec<tokio::task::JoinHandle<()>> {
        self.adapters.load_full().spawn_all(self.clone())
    }

    /// Resolve the conversation-management feature for `transport` through the adapter registry
    /// (`adapter_for_transport -> messaging -> conversations`).
    fn conversations_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsConversations>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.conversations())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no conversation support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the membership-administration feature for `transport`.
    fn membership_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsMembership>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.membership())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no membership support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the remote-contacts feature for `transport`.
    fn contacts_for(&self, transport: &TransportId) -> Result<Arc<dyn SupportsContacts>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.contacts())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no contacts support",
                    transport.as_str()
                ))
            })
    }

    /// Resolve the contact/user-directory feature for `transport`.
    fn directory_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn SupportsDirectory>, ApiError> {
        self.adapters
            .load_full()
            .adapter_for_transport(transport)
            .and_then(|a| a.messaging())
            .and_then(|m| m.directory())
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "transport {} has no directory support",
                    transport.as_str()
                ))
            })
    }

    /// Journal + seal one management mutation onto the verifiable `node-management` stream (a sealed
    /// dCBOR entry per mutating `conv_*`/`member_*` op). No-op when journaling is disabled.
    async fn audit_management(&self, kind: &str, detail: String) {
        // Reuse one long-lived sink so the chain links per op (each `seal` advances to the next
        // segment); only build it on the first mutation, and only when journaling is enabled.
        let sink = {
            let mut guard = self.mgmt_journal.lock().unwrap();
            if guard.is_none() {
                let Some(signer) = self.verifier.clone() else {
                    return;
                };
                *guard = Some(Arc::new(JournalSink::new(
                    self.store.clone(),
                    signer,
                    JournalStreamId::unit(&UnitId::new("node-management")),
                )));
            }
            guard.as_ref().unwrap().clone()
        };
        if let Err(e) = sink.record_management(kind.to_string(), detail).await {
            tracing::warn!(error = %e, kind, "management audit: record failed");
            return;
        }
        if let Err(e) = sink.seal().await {
            tracing::warn!(error = %e, kind, "management audit: seal failed");
        }
    }

    /// Run a management operation, then record one audit entry (op-then-audit, audit only on
    /// success). Centralizes the `op.await?; audit_management(..); Ok(..)` shape shared by the
    /// conv/member/contact mutating wrappers.
    async fn audited<T>(
        &self,
        kind: &str,
        detail: String,
        op: impl std::future::Future<Output = Result<T, ApiError>>,
    ) -> Result<T, ApiError> {
        let out = op.await?;
        self.audit_management(kind, detail).await;
        Ok(out)
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
        self.routing_base.store(Arc::new(routing));
        self.rebuild_routing();
    }

    /// Rebuild the live routing table: take the base (the rebuild hook's output when installed, else
    /// the static `routing_base`), layer the durable chat→session pins on top, and swap it in. A
    /// no-op-ish refresh when no builder is set, but always re-applies pins. Called after profile/auth
    /// mutations and after a pin reload so routing stays current without a restart.
    fn rebuild_routing(&self) {
        let mut reg = match &self.routing_builder {
            Some(builder) => builder(),
            None => (*self.routing_base.load_full()).clone(),
        };
        reg.set_pins(self.chat_pins.read().unwrap().clone());
        self.routing.store(Arc::new(reg));
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

    /// Wire the node-wide event feed (L3 `EventsSince`) so `events_*` serve live notifications and
    /// the §5 emit hooks (here + on the live-session actor) reach a real ring.
    pub fn with_node_events(mut self, feed: Arc<NodeEventFeed>) -> Self {
        self.live.set_node_events(feed.clone());
        self.node_events = Some(feed);
        self
    }

    /// The node-wide event feed, when wired (cloned out for an emit / `bump_rev` in the §5 hooks that
    /// hang off `NodeApiImpl` directly — roster/meta changes).
    fn node_feed(&self) -> Option<Arc<NodeEventFeed>> {
        self.node_events.clone()
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
        // L3: a turn touched this session (recency + maybe a seeded title changed), so its roster row
        // is stale. Turn-level granularity (not per-delta — `SessionAdvanced` covers token growth).
        if let Some(feed) = self.node_feed() {
            let rev = feed.note_roster_change(session);
            feed.emit(NodeEvent::SessionMetaChanged {
                session: session.clone(),
                rev,
            });
        }
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
            let _ = log.append(
                daemon_common::RevisionKind::Profile,
                id,
                &blob,
                author,
                reason,
            );
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
/// plus typed `session`/`profile` columns, with the full wire descriptor (origin + isolation)
/// carried as the opaque CBOR `descriptor` blob for faithful round-trip.
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

/// A human label for a [`Participant`] (the management-audit detail; never a secret payload).
fn participant_label(who: &Participant) -> String {
    match who {
        Participant::Contact(c) => c.id.clone(),
        Participant::Agent { member, profile } => {
            format!("{member} (profile {})", profile.as_str())
        }
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

/// Whether a roster entry matches a queried [`SessionScope`]. `owned` is the resolved owned-session
/// set, used only by `ByTransport` (empty for other scopes).
fn session_in_scope(
    i: &SessionInfo,
    scope: &SessionScope,
    owned: &std::collections::HashSet<SessionId>,
) -> bool {
    match scope {
        SessionScope::TopLevel => i.role == SessionRole::Primary && !i.archived,
        SessionScope::ByProfile(p) => i.bound_profile.as_ref() == Some(p) && !i.archived,
        SessionScope::ByTransport(_) => owned.contains(&i.session) && !i.archived,
        SessionScope::Archived => i.role == SessionRole::Primary && i.archived,
        SessionScope::All => true,
    }
}

/// Apply cursor pagination in place: skip through the `after` id (exclusive), cap to the effective
/// limit, and return the next cursor (the last retained id) when the page was truncated.
fn paginate_roster(
    roster: &mut Vec<SessionInfo>,
    after: Option<&SessionId>,
    limit: u32,
) -> Option<SessionId> {
    if let Some(after) = after {
        if let Some(pos) = roster.iter().position(|i| &i.session == after) {
            roster.drain(..=pos);
        }
    }
    let limit = if limit == 0 {
        DEFAULT_ROSTER_PAGE
    } else {
        limit as usize
    };
    if roster.len() > limit {
        roster.truncate(limit);
        roster.last().map(|i| i.session.clone())
    } else {
        None
    }
}

/// A roster title seeded from the first user turn: the first line, trimmed to ~60 chars on a word
/// boundary with an ellipsis. A placeholder until a real generated title replaces it.
fn title_from_text(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or(text).trim();
    const MAX: usize = 60;
    if first_line.chars().count() <= MAX {
        return first_line.to_string();
    }
    let truncated: String = first_line.chars().take(MAX).collect();
    let cut = truncated
        .rsplit_once(' ')
        .map(|(h, _)| h)
        .unwrap_or(&truncated);
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

/// Map a [`daemon_core::CommandError`] to the wire [`ApiError`] at the command boundary.
fn command_err_to_api(err: daemon_core::CommandError) -> ApiError {
    use daemon_core::CommandError::*;
    match err {
        Unknown(name) => ApiError::Other(format!("unknown command: {name}")),
        BadArgs(msg) => ApiError::Other(format!("invalid arguments: {msg}")),
        MissingSession => ApiError::Other("command requires an active session".into()),
        Failed(msg) => ApiError::Other(msg),
    }
}

impl NodeApiImpl {
    /// Dispatch a resolved built-in command over the node's existing typed ops — a thin adapter, not
    /// a re-implementation (the logic for cancel/model/mode/approve lives once in the ops it calls).
    /// `command_invoke` has already gated access and verified a session is present for session-scoped
    /// commands.
    async fn run_builtin(
        &self,
        builtin: crate::commands::Builtin,
        invocation: &CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        use crate::commands::Builtin;
        let args = invocation.args.trim();
        match builtin {
            Builtin::Help => {
                let specs = self
                    .commands
                    .load()
                    .as_ref()
                    .map(|r| r.specs())
                    .unwrap_or_default();
                Ok(CommandOutput {
                    text: crate::commands::render_help(&specs),
                    ephemeral: true,
                })
            }
            Builtin::Whoami => Ok(CommandOutput {
                text: format!(
                    "profile: {}\npartition: {:?}",
                    self.default_local_profile, self.partition
                ),
                ephemeral: true,
            }),
            Builtin::Version => Ok(CommandOutput {
                text: format!("daemon {}", env!("CARGO_PKG_VERSION")),
                ephemeral: true,
            }),
            Builtin::Status => {
                let t = self.telemetry().await;
                Ok(CommandOutput {
                    text: format!(
                        "healthy: {}\nsessions: {} ({} active)\npending jobs: {}, wakes: {}\nevents: {}",
                        t.healthy, t.sessions, t.active, t.pending_jobs, t.pending_wakes, t.events
                    ),
                    ephemeral: true,
                })
            }
            Builtin::Usage => {
                let t = self.telemetry().await;
                let u = &t.usage;
                Ok(CommandOutput {
                    text: format!(
                        "tokens in/out: {}/{}\ncache read/write: {}/{}\nreasoning: {}\nest. cost: ${:.4}",
                        u.input_tokens,
                        u.output_tokens,
                        u.cache_read_tokens,
                        u.cache_write_tokens,
                        u.reasoning_tokens,
                        u.cost_micros as f64 / 1_000_000.0
                    ),
                    ephemeral: true,
                })
            }
            Builtin::Sessions => {
                let roster = self.sessions().await;
                let mut text = format!("{} session(s):\n", roster.len());
                for s in &roster {
                    let title = s.title.as_deref().unwrap_or("(untitled)");
                    text.push_str(&format!("  {} — {}\n", s.session.as_str(), title));
                }
                Ok(CommandOutput {
                    text,
                    ephemeral: true,
                })
            }
            Builtin::Stop => {
                let session = invocation
                    .session
                    .clone()
                    .ok_or_else(|| ApiError::Other("stop requires a session".into()))?;
                self.cancel(session).await?;
                Ok(CommandOutput {
                    text: "cancelled in-flight work".into(),
                    ephemeral: true,
                })
            }
            Builtin::Model => {
                if args.is_empty() {
                    return Err(ApiError::Other("usage: /model <model-id>".into()));
                }
                let session = invocation
                    .session
                    .clone()
                    .ok_or_else(|| ApiError::Other("model requires a session".into()))?;
                self.set_session_model(session, args.to_string(), None)
                    .await?;
                Ok(CommandOutput {
                    text: format!("session model set to {args}"),
                    ephemeral: true,
                })
            }
            Builtin::Mode => {
                let session = invocation
                    .session
                    .clone()
                    .ok_or_else(|| ApiError::Other("mode requires a session".into()))?;
                let mode = resolve_approval_mode(&invocation.name, args)?;
                self.set_session_mode(session, mode).await?;
                Ok(CommandOutput {
                    text: format!("approval mode set to {mode:?}"),
                    ephemeral: true,
                })
            }
            Builtin::Title => {
                if args.is_empty() {
                    return Err(ApiError::Other("usage: /title <new title>".into()));
                }
                let session = invocation
                    .session
                    .clone()
                    .ok_or_else(|| ApiError::Other("title requires a session".into()))?;
                let patch = SessionMetaPatch {
                    title: Some(Some(args.to_string())),
                    ..SessionMetaPatch::default()
                };
                self.session_update_meta(session, patch).await?;
                Ok(CommandOutput {
                    text: format!("title set to {args:?}"),
                    ephemeral: true,
                })
            }
            Builtin::Approve | Builtin::Deny => {
                let allow = matches!(builtin, Builtin::Approve);
                if args.is_empty() {
                    return Err(ApiError::Other(format!(
                        "usage: /{} <request-id>",
                        if allow { "approve" } else { "deny" }
                    )));
                }
                let session = invocation
                    .session
                    .clone()
                    .ok_or_else(|| ApiError::Other("approval requires a session".into()))?;
                self.approval_decide(session, args.to_string(), allow)
                    .await?;
                Ok(CommandOutput {
                    text: format!(
                        "request {args} {}",
                        if allow { "approved" } else { "denied" }
                    ),
                    ephemeral: true,
                })
            }
        }
    }
}

/// Resolve the requested [`ApprovalMode`] from a `/mode` invocation: the `yolo`/`fast` aliases map
/// directly, otherwise the first argument is parsed (`yolo`/`auto`, `fast`/`accept`, `ask`, `deny`).
fn resolve_approval_mode(name: &str, args: &str) -> Result<ApprovalMode, ApiError> {
    let key = match name
        .trim()
        .trim_start_matches('/')
        .to_ascii_lowercase()
        .as_str()
    {
        "yolo" => "yolo".to_string(),
        "fast" => "fast".to_string(),
        _ => args
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase(),
    };
    match key.as_str() {
        "yolo" | "auto" | "autoallow" | "auto-allow" => Ok(ApprovalMode::AutoAllow),
        "fast" | "accept" | "acceptedits" | "accept-edits" => Ok(ApprovalMode::AcceptEdits),
        "ask" | "default" => Ok(ApprovalMode::Ask),
        "deny" | "reject" => Ok(ApprovalMode::Deny),
        "" => Err(ApiError::Other("usage: /mode <yolo|fast|ask|deny>".into())),
        other => Err(ApiError::Other(format!("unknown approval mode: {other}"))),
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

mod control;
mod cred_auth;
mod model;
mod profile;
mod session;

mod internals;
pub use internals::NodeEventFeed;
pub(crate) use internals::{apply_rewind_side_effects, LiveSessions, RewindSideEffects};
