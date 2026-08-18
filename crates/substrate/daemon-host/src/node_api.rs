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
//!
//! This module file is the **thin spine**: it holds the [`NodeApiImpl`] struct + its injected seam
//! types, the one-lifecycle-owner invariant ([`NodeApiImpl::claim`]), and the assembly of the
//! cohesive sub-modules below. The behavior lives in those sub-modules:
//! [`assembly`] (construction/wiring), [`control`]/[`session`]/[`model`]/[`profile`]/[`cred_auth`]
//! (the `*Api` trait impls), and the helper concerns [`roster`], [`overlay`], [`messaging`],
//! [`journal_audit`], [`routing`], [`delivery`], [`provisioning`], [`builtins`], [`internals`].

use crate::auth::PendingAuthFlows;
use crate::credstore::CredentialStore;
use crate::engine_incarnation::JournalConfig;
use crate::journal::{JournalFeeder, JournalSink};
use crate::profiles::ProfileStore;
use crate::request_context::{current_principal, with_request_context, RequestContext};
use crate::routing::RoutingRegistry;
use crate::supervisor::{HealthStatus, SupervisorObserver};
use crate::FleetControl;
use arc_swap::{ArcSwap, ArcSwapOption};
use async_trait::async_trait;
use daemon_activation::ActivationManager;
use daemon_api::{
    from_cbor,
    to_cbor,
    AccountSettingsValues,
    ActionMenu,
    AdapterInfo,
    AgentEntry,
    AgentSource,
    ApiError,
    ApprovalInfo,
    ApprovalMode,
    AuthApi,
    AuthBeginRequest,
    AuthBeginResponse,
    AuthBindRequest,
    AuthCompleteResponse,
    AuthProviderInfo,
    AuthStepRequest,
    AuthStepResult,
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
    CustomProvider,
    DeliverySink,
    Distribution,
    EventsPage,
    FeedbackAck,
    FeedbackKind,
    FeedbackRating,
    FeedbackSubmitArgs,
    FleetReport,
    FsContent,
    FsEntry,
    FsListPage,
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
    GatewayStatus,
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
    ProviderDescriptor,
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
    SupportsRoster,
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
    AgentCommand, AgentEvent, ConvView, DeliveryTarget, Direction, Disposition, HostRequest,
    HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, IsolationPolicy, Origin,
    OriginScope, SessionLogEntry, SessionPayload, SinkKind, TranscriptBlock, TranscriptRole,
    TransportId, UserMsg,
};
use daemon_store::{
    FeedbackRecord, NewSplice, SessionMeta, SessionRole as StoreRole, SessionStatus, SessionStore,
    SpliceKind,
};
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

/// Builds a fresh live session backend for an interactive session id (the session sub-surface's
/// engine seam — the binary supplies the provider/tools/system). The optional [`ProfileRef`] selects
/// which profile bundle the backend is materialized from (host routing's agent-selection degree of
/// freedom); `None` resolves the node's active default. The [`SessionOverlay`] is the session's
/// persisted per-session override (model/provider/tools/approval), applied on top of the bound
/// profile at build time, so a live override is **restored** when the actor is (re)spawned.
pub type SessionEngineBuilder =
    Arc<dyn Fn(SessionId, Option<ProfileRef>, &SessionOverlay) -> SessionBackend + Send + Sync>;

/// Constructs a foreign live session (e.g. an ACP agent) once the host hands it the session's
/// [`HostRequestHandler`] (the parking handler that answers the agent's blocking §17 requests —
/// permission prompts park exactly like a native engine's). Deferred + async because resolving the
/// profile's catalog NAME to a launch recipe reads the durable ACP registrations, and fallible so
/// a vanished/uninstalled agent fails the spawn with a clear [`ApiError`] instead of a dead actor.
/// Injected by the assembling binary — `daemon-host` never links the foreign runtime (`daemon-acp`
/// depends on *it*), mirroring the [`AgentDiscovery`] injection.
pub type ForeignSessionFactory = Box<
    dyn FnOnce(
            Arc<dyn HostRequestHandler>,
        ) -> futures::future::BoxFuture<
            'static,
            Result<Arc<dyn crate::AgentSession>, ApiError>,
        > + Send,
>;

/// How a live interactive session's backend is constructed by the [`SessionEngineBuilder`]: the
/// in-process `daemon-core` [`Engine`] (the native default), or a foreign engine supplied as a
/// deferred [`ForeignSessionFactory`] (a profile whose `engine = Foreign{agent}` resolved through
/// the node's agent catalog). Both present identically on the live surface — one merged log, one drain,
/// one journal feeder — only the backend construction differs.
// Built once per session open and consumed immediately by `ensure` — the variant size delta is
// irrelevant, and boxing the Engine would leak into the builder closures for no benefit (mirrors
// the fleet spawner's AgentBackend).
#[allow(clippy::large_enum_variant)]
pub enum SessionBackend {
    /// The native in-process `daemon-core` engine (run on the §17 actor).
    Core(Engine),
    /// A foreign engine, materialized by the injected factory at `ensure` time. Carries the
    /// catalog agent NAME the factory resolves so the live registry can key the residency by
    /// agent — the seam that lets a completed `agent/<name>` auth flow evict (and thereby
    /// respawn-with-fresh-credentials) exactly that agent's resident sessions.
    Foreign {
        agent: String,
        factory: ForeignSessionFactory,
    },
}

/// Resolve a session's effective [`EngineProfile`] from its bound profile ref + persisted overlay —
/// the durable-path counterpart of [`SessionEngineBuilder`], injected into [`CoreEngineFactory`] by
/// the node (which owns the profile store + resolution rules). Returns `None` when no profile store
/// is configured or the bound profile is absent, so the durable path falls back to the factory's
/// default (orchestrator) profile. This is the seam that makes durable rehydration re-resolve from
/// the profile store + overlay instead of pinning the factory's fixed profile.
/// The `inline` argument carries the opaque CBOR of an inline sub-agent's host `ProfileSpec` (Phase
/// 1), read from `SessionMeta.inline_profile`; empty for every non-inline session. When non-empty
/// and the decoded engine is `Core`, the resolver builds the sub-agent's engine from it directly
/// (`bound_profile` is `None` for an inline child); a `Foreign` inline is handled by the dispatching
/// factory's foreign incarnation, so the resolver returns `None` for it. The [`SessionId`] is the
/// rehydrating session's own id — the transient engine identity an inline spec resolves under
/// (its credential/context/memory scope key).
pub type DurableProfileResolver = Arc<
    dyn Fn(
            &SessionId,
            Option<ProfileRef>,
            &[u8],
            &SessionOverlay,
        ) -> Option<daemon_core::EngineProfile>
        + Send
        + Sync,
>;

/// Builds a fresh model [`Provider`] from a (model-overridden) [`ProfileSpec`] — the seam a live
/// [`SessionApi::set_session_model`](daemon_api::SessionApi::set_session_model) uses to rebuild a
/// running session's provider without `daemon-host` linking the provider crate.
pub type ModelProviderFactory = Arc<dyn Fn(&ProfileSpec) -> Arc<dyn Provider> + Send + Sync>;

/// The routing rebuild hook (the §5.9 hot-reload seam): produces a fresh [`RoutingRegistry`] from
/// current node state (profiles + bound accounts). Re-run on `profile_update` / `auth_complete` so
/// routing stays current without a restart. The assembling binary owns the closure (it owns the
/// profile source); the host never links the routing-from-profiles policy directly.
pub type RoutingBuilder = Arc<dyn Fn() -> RoutingRegistry + Send + Sync>;

/// The stage-5 backend probe (session-unification §8): whether a bound profile resolves to a
/// Foreign engine, which keeps its explicit live actor rail under the cutover. Injected by the
/// assembly, which owns the profile store + resolution rules.
pub type ForeignProbe = Arc<dyn Fn(&ProfileRef) -> bool + Send + Sync>;

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

    /// The discoverable provider catalog for the setup picker: local engines + every genai cloud
    /// vendor + Daemon Cloud. Static metadata (no network); independent of the launch default, so an
    /// unconfigured node still lists providers.
    async fn providers(&self) -> Vec<ProviderDescriptor>;

    /// One provider's discoverable models, keyed by [`ProviderDescriptor::id`]. Credential-aware for
    /// genai vendors (the resolved `key` authenticates the LIST call); Daemon Cloud lists keyless.
    /// Local engines are served by the host from the `ModelManager` catalog, not here.
    /// Structured outcome (wire v48): a listing failure returns a classified
    /// [`daemon_api::ProviderListError`] instead of masquerading as an empty catalog.
    async fn provider_models(
        &self,
        provider_id: &str,
        key: Option<String>,
    ) -> Result<Vec<ModelDescriptor>, daemon_api::ProviderListError>;

    /// List an arbitrary OpenAI-compatible endpoint's models via `GET {base_url}/models`,
    /// credential-aware (`key` is sent as a bearer when present, keyless otherwise). Backs custom
    /// providers: the host resolves the stored `base_url` + credential and calls this, so the host
    /// never links `genai`/egress. Default: unsupported (a catalog with no OpenAI-compatible probe
    /// wired).
    async fn openai_compat_models(
        &self,
        _base_url: &str,
        _key: Option<String>,
    ) -> Result<Vec<ModelDescriptor>, daemon_api::ProviderListError> {
        Err(daemon_api::ProviderListError {
            kind: daemon_api::ProviderListErrorKind::Unsupported,
            message: "this node has no OpenAI-compatible discovery probe wired".into(),
        })
    }
}

/// The foreign-agent discovery hook (I7). `daemon-host` does not link the ACP runtime (`daemon-acp`
/// depends on *it*, not the reverse), so the actual probing — the curated direct-binary recipe
/// table + PATH probe, plus the ACP `initialize` handshake for ACP entries — is injected by the
/// assembling binary (which owns the ACP crate). Stream-json entries are probed installed-on-PATH
/// only (no handshake). When no hook is wired, `agent_discover` returns empty and only manual
/// registrations are catalogued.
#[async_trait]
pub trait AgentDiscovery: Send + Sync {
    /// Probe PATH + the curated direct-binary recipe table, confirming each ACP candidate via the
    /// `initialize` handshake; return verified catalog entries (`source = Builtin`).
    async fn discover(&self) -> Vec<daemon_api::AgentEntry>;
    /// The cheap presence half of [`discover`](Self::discover): the curated table with only the
    /// PATH `installed` check — **no** `initialize` handshakes, so it answers in microseconds and
    /// installed rows surface `Unverified`. Backs the fast `agent_discover` reply (wire v46);
    /// the host runs the slow verified scan in the background afterwards. Default: empty (a
    /// minimal discoverer that only implements the full scan).
    fn presence(&self) -> Vec<daemon_api::AgentEntry> {
        Vec::new()
    }
    /// Verify/enrich a single (manual) recipe: a PATH-presence `installed` check, plus the ACP
    /// `initialize` handshake for `protocol = Acp` entries — fills in `installed` / `version` /
    /// `capabilities`. Returns the entry unchanged on a failed probe.
    async fn probe(&self, entry: daemon_api::AgentEntry) -> daemon_api::AgentEntry;
    /// Resolve a curated builtin recipe by `name` WITHOUT the `initialize` probe: the recipe plus a
    /// cheap PATH-presence `installed` check only. Backs the fast-path lookups that must not spawn
    /// candidate processes (profile-engine validation, foreign-engine spawn resolution) when the
    /// name is not among the durable manual registrations. `None` when the name is not curated.
    fn builtin(&self, name: &str) -> Option<daemon_api::AgentEntry> {
        let _ = name;
        None
    }
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
    /// Live serve-loop handles per adapter family (wire v30, item 1): `spawn_adapters` records each
    /// adapter's supervised serve task here so `transport_disconnect`/`transport_remove` can stop a
    /// single instance's adapter. Keyed by adapter family (the coarsest per-instance granularity the
    /// single-serve-loop-per-adapter architecture supports).
    adapter_handles:
        Arc<std::sync::Mutex<std::collections::HashMap<String, tokio::task::AbortHandle>>>,
    /// Per-transport fatal-disconnect flags (wire v30, item 2): the [`daemon_api::LifecycleSink`]
    /// sets one when an adapter reports a fatal cause (auth/settings/cert); the reconnect supervisor
    /// in `spawn_adapters` reads it to short-circuit the backoff loop (stop, offer re-auth) instead
    /// of respawning a serve loop that will only fail again.
    disconnect_fatal: Arc<dashmap::DashMap<TransportId, bool>>,
    /// A weak self-handle captured by `spawn_adapters` (wire v35): `transport_connect` needs an
    /// owned `Arc<Self>` to (re)spawn a single family's supervised serve loop, but it is a `&self`
    /// `ControlApi` method. `OnceLock`-set at the first `spawn_adapters` (boot); absent on a node
    /// whose adapters were never spawned, where `transport_connect` has nothing to resume anyway
    /// (and returns `Unsupported`).
    self_weak: std::sync::OnceLock<std::sync::Weak<NodeApiImpl>>,
    /// The lazily-opened verifiable-journal writer for the `node-management` stream: management
    /// mutations (`conv_*`/`member_*`) are recorded + sealed onto it so the audit chains per op.
    /// `None` until the first mutation (and stays `None` when journaling is disabled).
    mgmt_journal: Arc<std::sync::Mutex<Option<Arc<JournalSink>>>>,
    /// The lazily-opened per-conversation chat-journal writers (wire v38), one per
    /// `conv:<transport>:<conv>` stream: the [`daemon_api::LifecycleSink::chat_message`] seam
    /// records every adapter-reported send/delivery through the stream's one long-lived sink so
    /// the chain links per message. Empty until the first message (and stays empty when
    /// journaling is disabled).
    chat_journals:
        Arc<std::sync::Mutex<std::collections::HashMap<JournalStreamId, Arc<JournalSink>>>>,
    /// The foreign-agent discovery hook (I7), injected by the binary (which owns the ACP runtime).
    /// `None` => `agent_discover` yields nothing and the catalog is just the durable manual
    /// registrations.
    agents: Option<Arc<dyn AgentDiscovery>>,
    /// The last discovery scan's results, cached in-memory so `agent_catalog` can surface them
    /// alongside the durable manual entries without re-probing every read (discovery is the
    /// operator-triggered, subprocess-spawning scan; manual entries are the persisted half).
    last_agents: Arc<std::sync::RwLock<Vec<daemon_api::AgentEntry>>>,
    /// Whether a background verification sweep (the `initialize`-handshake half of
    /// `agent_discover`, wire v46) is in flight — a reconnect storm of discover calls must not
    /// stack subprocess sweeps; late callers get the fast presence pass only and the running
    /// sweep's `AgentsChanged` when it lands.
    agents_scan_running: Arc<std::sync::atomic::AtomicBool>,
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
    /// The saved-presence manager (wire v37) backing the `presence_*` control ops. `None`
    /// => every presence op resolves to its defaulted empty list / [`ApiError::Unsupported`] (a node
    /// built without saved-presence management).
    presences: Option<Arc<crate::presence::PresenceManager>>,
    /// The daemon-authoritative command catalog backing `command_list`/`command_invoke`: built-in
    /// node-op commands unified with the engine profile's [`CommandProvider`](daemon_core::CommandProvider)
    /// contributions (`/lcm`, `/memory`, …). Empty => the command surface resolves to its defaulted
    /// empty catalog / [`ApiError::Unsupported`]. Held behind an [`ArcSwapOption`] so the assembling
    /// binary can bind it *after* the node is wrapped in an `Arc` (see [`NodeApiImpl::set_commands`]),
    /// since the registry needs node-resolved provider handles the node construction does not own.
    commands: Arc<ArcSwapOption<crate::commands::CommandRegistry>>,
    /// The node-wide tool inventory backing [`ControlApi::tool_list`] (wire v29): one row per
    /// registered tool plus one per disabled config-gated surface (with `requires`). Late-bound by
    /// the assembling binary (which owns the tool build gates); `None` => `tool_list` returns empty.
    tools_inventory: Arc<ArcSwapOption<Vec<daemon_api::ToolInfo>>>,
    /// The read-only delegation guardrail caps backing [`ControlApi::caps`] (wire v29): the
    /// EFFECTIVE `orchestrate` ceilings, set at assembly (which owns the policy/budget
    /// composition). Zeros until wired.
    caps: daemon_api::CapsReport,
    /// The identity store backing the admin access-control sub-surface ([`daemon_api::AccessControlApi`]).
    /// `None` => every admin op resolves to [`ApiError::Unsupported`] (a node assembled without an
    /// identity store — the FFI / conformance harness). `who_am_i` needs no store (it reads the
    /// request principal); `role_list` is store-free (the built-in role→capability matrix).
    auth_store: Option<Arc<daemon_auth::AuthStore>>,
    /// The shared auth-audit sink (the `node-auth` verifiable journal chain). `None` => admin-op
    /// audit is a no-op (no journaling). The same handle is given to the transport's
    /// [`Authenticator`](crate::authn::Authenticator) so login/denial events ride the same chain.
    auth_audit: Option<Arc<crate::auth_audit::AuthAudit>>,
    /// The shared per-principal revocation registry (Cluster F, Part A). The admin ops that revoke a
    /// principal (`session_revoke`/`user_disable`/`user_set_roles`/`user_set_password`) bump the
    /// user's epoch here *after* the store mutation, so a live mux connection holding the old epoch
    /// is torn down. Pass the **same** [`SessionRevocations`](crate::revocation::SessionRevocations)
    /// to the transport's [`Authenticator`](crate::authn::Authenticator). `None` => live-connection
    /// revocation is not enforced (the store mutation still invalidates the reconnect fast-path).
    revocations: Option<Arc<crate::revocation::SessionRevocations>>,
    /// The credential-authority revoker (Cluster F, Part B). `credential_remove`/`credential_set`
    /// call [`revoke_profile`](crate::revocation::CredentialRevoker::revoke_profile) so the profile's
    /// cached [`CredentialAuthority`](daemon_credentials::CredentialAuthority) bumps its lease epoch
    /// (invalidating outstanding leases at `use_capability`) and drops retained proxied keys. `None`
    /// => only the credential *store* is mutated (a fresh acquire no longer sees the removed key,
    /// but an already-minted lease is not invalidated).
    credential_revoker: Option<Arc<dyn crate::revocation::CredentialRevoker>>,
    /// The user-feedback outbox drain seam (N1 → N2): the wired OTLP exporter the `FeedbackSubmit`
    /// enqueue + node startup drain each queued [`daemon_store::FeedbackRecord`] through, mapped to a
    /// [`daemon_telemetry::feedback::FeedbackEvent`] and shipped to `telemetry.feedback_endpoint`.
    /// `None` => export is inert (no endpoint configured, or the `otel` feature is off) and records
    /// simply stay queued. Bound via [`NodeApiImpl::with_feedback_endpoint`] at assembly.
    feedback_drain: Option<Arc<feedback::FeedbackDrain>>,
    /// The node-managed backend resources (the gateway + local inference) surfaced in
    /// [`ControlApi::health`] alongside the resident-service supervisor's children. Registered
    /// post-`Arc` by the assembling binary via [`NodeApiImpl::register_managed`]; empty on a node
    /// with no managed backends wired.
    managed: Arc<Mutex<Vec<Arc<dyn crate::managed::ManagedResource>>>>,
    /// The typed gateway control seam backing `gateway_get`/`gateway_set`. `None` => those ops
    /// resolve to [`ApiError::Unsupported`]. Bound post-`Arc` (the gateway backend needs the
    /// assembled node) via [`NodeApiImpl::set_gateway`], which also registers it into `managed` so
    /// the gateway reports its health like any other managed backend.
    gateway: Arc<Mutex<Option<Arc<dyn crate::managed::GatewayControl>>>>,
    /// The shared profile create/validate/persist/version surface (I15-style) backing the operator
    /// `profile_create`/`profile_update` ops AND the agent `profile_manage` tool, so both author
    /// profiles through one validation + persistence + revision path. Its validator is late-bound to
    /// this node (`ProfileValidator = NodeApiImpl`). `None` => the operator path falls back to the
    /// inline validate+persist+record (a minimal node with no shared facade / no agent tool).
    profile_ops: Option<Arc<crate::profile_ops::ProfileOps>>,
    /// The persona (SOUL.md) backend behind the wire `SoulGet`/`SoulSet` ops (wire v36). `None` on
    /// a node built without persona management (both ops then resolve to
    /// [`ApiError::Unsupported`]). The real `PersonaStore` adapter is bound at node assembly via
    /// [`with_persona_ops`](Self::with_persona_ops).
    persona_ops: Option<Arc<dyn crate::persona_ops::PersonaOps>>,
    /// The node's live notification manager (wire v37), ported from libpurple's
    /// `PurpleNotificationManager`. Backs [`ControlApi::notification_list`](daemon_api::ControlApi::notification_list);
    /// mutations emit [`NodeEvent::NotificationsChanged`](daemon_api::NodeEvent) via
    /// [`emit_notifications_changed`](Self::emit_notifications_changed) so clients re-list.
    notifications: Arc<std::sync::Mutex<crate::notifications::NotificationManager>>,
    /// The node's person/metacontact registry (wire v37), ported from the person half of
    /// libpurple's `PurpleContactManager`. Backs
    /// [`ControlApi::person_list`](daemon_api::ControlApi::person_list); mutations emit
    /// [`NodeEvent::PersonsChanged`](daemon_api::NodeEvent) via
    /// [`emit_persons_changed`](Self::emit_persons_changed) so clients re-list.
    persons: Arc<std::sync::Mutex<crate::person::PersonManager>>,
    /// The stage-5 cutover arm (session-unification §8): when set, a NON-resident, Core-backed
    /// session's wire commands route onto the durable rail (typed splice + wake / the session's
    /// [`AttachmentHub`]) and the observation ops (`poll`/`log_after`/`subscribe`/`respond`) are
    /// served from the hub — the live actor stops being the interactive authority. The SAME
    /// registry must be wired into the durable [`CoreEngineFactory`] (the incarnation side).
    /// `None` => every route keeps the legacy live-first behavior (the pre-cutover node).
    attachments: Option<Arc<attachments::AttachmentHubs>>,
    /// The stage-5 backend probe: whether a bound profile resolves to a Foreign engine (which
    /// keeps its explicit live actor rail — the cutover moves only Core-backed sessions). Injected
    /// by the assembly (which owns the profile store + resolution rules); `None` (with the cutover
    /// armed) treats every bound profile as Core.
    foreign_probe: Option<ForeignProbe>,
    /// The vhc-training service backing the [`daemon_api::VhcApi`] sub-surface (spec §10.4).
    /// `None` on a node built without vhc training (`[vhc] enabled = false`, the default): every
    /// `VhcApi` call then resolves to [`ApiError::Unsupported`] / an empty stream. Bound at
    /// assembly via [`with_vhc`](Self::with_vhc) OR **post-`Arc`** via
    /// [`set_vhc`](Self::set_vhc) (B3 — the service is built after the node exists, like the
    /// gateway/managed backends), only when the service is enabled — the node never spawns a training
    /// worker unless a vhc service is present. A write-once cell (like a managed backend seam).
    vhc: std::sync::OnceLock<Arc<dyn daemon_api::VhcApi>>,
}

impl NodeApiImpl {
    /// Inject host-originated input (a background process-exit notification, a watch-pattern match,
    /// a message to a managed child) into `session`'s conversation, driving a reactive turn — the
    /// one seam that works across **both** session lifecycles:
    ///
    /// - a **live** (actor-resident) session takes a real [`AgentCommand::StartTurn`] through the
    ///   normal submit path (Observe-while-idle only folds context and drives no turn, so a
    ///   notification would otherwise sit unseen until the user next speaks);
    /// - a **durable** (non-resident) session gets a durable inbox splice
    ///   ([`SessionStore::append_splice`]) plus a wake; the incarnation folds it into the
    ///   conversation at hydrate and the woken turn runs with it.
    ///
    /// Routing is by residency + durable evidence (the retired in-memory owner map is not
    /// consulted): a live-resident session (Foreign) takes the submit path; a session with a
    /// durable row takes the store seam (never spawning a divergent live engine over durable
    /// state); anything else opens via `submit`, exactly like an inbound message would. A
    /// `Completed` durable session drops the input (its owner is gone).
    pub async fn inject_session_input(
        &self,
        session: &SessionId,
        text: String,
    ) -> Result<(), ApiError> {
        self.inject_session_msg(session, UserMsg::new(text)).await
    }

    /// [`Self::inject_session_input`] with a structured [`UserMsg`] (wire v29): the
    /// completion-notice worker passes the provenance-tagged message
    /// (`UserMsg::with_notice`) so the injected turn's `StartTurn` carries the chip-link fields
    /// through both the live submit and the durable pending-input rail.
    pub async fn inject_session_msg(
        &self,
        session: &SessionId,
        msg: UserMsg,
    ) -> Result<(), ApiError> {
        let durable = !self.live.is_resident(session) && self.store.status(session).await.is_some();
        if durable {
            match self.store.status(session).await {
                Some(SessionStatus::Completed) | None => {
                    tracing::debug!(
                        session = %session,
                        "dropping injected input for a settled durable session"
                    );
                    return Ok(());
                }
                Some(_) => {}
            }
            self.enqueue_durable_input(session, SpliceKind::StartTurn, &msg, "host-inject")
                .await?;
            return Ok(());
        }
        // `self.submit` is the `SessionApi` trait method (Auth 4 ownership-gated). This seam is
        // driven by background workers (the process notifier, the delegation notice worker) that
        // carry no request context, so bind the trusted in-process `internal` principal — otherwise
        // the ownership check would see `None` (now deny) and drop the injection.
        with_request_context(
            RequestContext::internal(),
            self.submit(
                session.clone(),
                AgentCommand::StartTurn {
                    input: msg,
                    request_id: daemon_common::ReqId(0),
                },
            ),
        )
        .await
    }

    /// The shared durable inbox rail (session-unification §4, stage 2): encode `msg`, append it
    /// as a typed splice ([`SessionStore::append_splice`] — durable, deduped, `Idle → Ready` in
    /// the same transaction) + a wake. The woken incarnation folds the claimed splice into the
    /// conversation at hydrate. Used by both the host-originated injection seam
    /// ([`Self::inject_session_msg`]) and the F4 durable-resume submit gate
    /// ([`Self::durable_resume_input`]).
    ///
    /// The dedupe `origin_op` is MINTED here (`op-<32 hex>`): the wire `ReqId` is a
    /// per-connection counter (colliding across reconnects on one session) and the internal
    /// producers carry no client op-id at all — so within this rail a retry-able end-to-end id
    /// does not exist yet. Threading the rung-3 client `op_id` through to the splice is the
    /// stage-5 cutover's routing work; until then the API-layer `command_dedup` table still
    /// absorbs wire retries in front of this seam.
    async fn enqueue_durable_input(
        &self,
        session: &SessionId,
        kind: SpliceKind,
        msg: &UserMsg,
        origin: &str,
    ) -> Result<(), ApiError> {
        let mut payload = Vec::new();
        ciborium::into_writer(msg, &mut payload)
            .map_err(|e| ApiError::Other(format!("encode injected input: {e}")))?;
        self.store
            .append_splice(NewSplice {
                session_id: session.clone(),
                kind,
                payload,
                origin_op: mint_op_id(),
                origin: origin.into(),
            })
            .await
            .map_err(|e| ApiError::Other(format!("append durable input: {e}")))?;
        self.store.enqueue_wake(session.clone()).await;
        Ok(())
    }

    /// The stage-5 cutover routing gate (session-unification §8): whether wire commands addressed
    /// at `session` route onto the durable rail. True iff the cutover is armed
    /// ([`Self::attachments`]), the session is NOT currently live-resident (a resident actor —
    /// Foreign, or a legacy live Core session — keeps its rail until it winds down), and the
    /// session is not Foreign-bound (the Foreign actor rail is untouched by the cutover).
    async fn cutover_routes(&self, session: &SessionId) -> bool {
        if self.attachments.is_none() {
            return false;
        }
        if self.live.is_resident(session) {
            return false;
        }
        if let Some(probe) = &self.foreign_probe {
            let bound = self
                .store
                .session_meta(session)
                .await
                .and_then(|m| m.bound_profile);
            if let Some(profile) = &bound {
                if probe(profile) {
                    return false;
                }
            }
        }
        true
    }

    /// Get-or-create the session's [`attachments::AttachmentHub`] (cutover-armed paths only).
    /// Creation stamps the merged log with the durable activation generation and BUMPS the stored
    /// generation (the same L2-resync rule the live `ensure()` applies): a fresh hub after a
    /// restart carries a strictly greater `log_epoch`, so a client detects the generation change
    /// and re-baselines its cursor.
    async fn attach_hub(&self, session: &SessionId) -> Arc<attachments::AttachmentHub> {
        let hubs = self
            .attachments
            .as_ref()
            .expect("attach_hub only routes when the cutover is armed");
        if let Some(hub) = hubs.get(session) {
            return hub;
        }
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        let epoch = meta.activation_epoch;
        meta.activation_epoch = epoch + 1;
        let _ = self.store.set_session_meta(session, meta).await;
        hubs.attach(session, epoch)
    }

    /// The stage-5 durable command router (§8's matrix): every wire command a live actor used to
    /// serve, homed on the durable lifecycle + the session's [`attachments::AttachmentHub`].
    ///
    /// - `StartTurn`/`Steer`/`Observe` → typed splice + wake (the ack rides the splice commit);
    ///   a `Steer` that finds the hub's turn slot OCCUPIED is delivered INTO the resident turn
    ///   instead (§7's timing contract — steering never waits for the next activation).
    /// - `Interrupt` → the resident turn's `TurnControl` via the hub (idle = benign no-op).
    /// - `Snapshot` → the resident turn serves it at its next phase boundary; an idle session's
    ///   reply is projected from the durable snapshot (`conv_view_of`) — current live semantics.
    /// - `Shutdown` → detach the hub (observers' streams end; nothing to tear down durably).
    /// - `RewindTo` → NOT yet served durably (the fenced re-incarnation rewind lands before the
    ///   production arm); explicit `Unsupported`, mirroring `ControlApi::rewind`'s durable path.
    ///
    /// The caller has already enforced Auth 4 and `note_activity`. Inbound commands are recorded
    /// on the hub's merged log so subscribers replay one seq-ordered conversation (live parity).
    async fn submit_attached(
        &self,
        session: &SessionId,
        command: AgentCommand,
        origin: Option<&daemon_protocol::Origin>,
    ) -> Result<(), ApiError> {
        let hubs = self
            .attachments
            .as_ref()
            .expect("submit_attached only routes when the cutover is armed");
        // A brand-new id submitted without a `session_create`: create the durable Idle row first
        // (the same atomic create-if-absent; a concurrent duplicate is benign).
        if self.store.status(session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            match self
                .store
                .create_idle(session.clone(), self.partition, blob)
                .await
            {
                Ok(()) | Err(daemon_store::StoreError::AlreadyExists(_)) => {}
                Err(e) => return Err(ApiError::Other(format!("create session: {e}"))),
            }
            if let Some(feed) = self.node_feed() {
                let rev = feed.note_roster_change(session);
                feed.emit(NodeEvent::RosterChanged { rev });
            }
        }
        let hub = self.attach_hub(session).await;
        // The splice provenance: the submitting origin's transport id when the surface passed one
        // (armed one-shot at the fold so per-turn surface hints key on THIS submit — live
        // `start_turn_from` parity), else the generic wire rail label (an unknown transport
        // family, composing no hint).
        let provenance = origin
            .map(|o| o.transport.as_str().to_string())
            .unwrap_or_else(|| "wire-submit".into());
        match command {
            AgentCommand::StartTurn { input, request_id } => {
                hub.record_inbound(AgentCommand::StartTurn {
                    input: input.clone(),
                    request_id,
                });
                self.enqueue_wire_input(session, SpliceKind::StartTurn, &input, provenance)
                    .await
            }
            AgentCommand::Steer { text, request_id } => {
                // Occupied slot: claimed into the resident turn (the hub records it inbound).
                if hub.deliver_steer(request_id, text.clone()) {
                    return Ok(());
                }
                hub.record_inbound(AgentCommand::Steer {
                    text: text.clone(),
                    request_id,
                });
                self.enqueue_wire_input(
                    session,
                    SpliceKind::Steer,
                    &UserMsg::new(text),
                    provenance,
                )
                .await?;
                // Splice-before-ack (§4.2): the append IS acceptance — the fold at the next wake
                // cannot correlate (the splice payload is the bare `UserMsg`), so the routing
                // layer acks here, the durable analogue of the live actor's idle-steer ack.
                hub.publish_event(daemon_protocol::AgentEvent::Steered {
                    seq: 0,
                    request_id,
                    accepted: true,
                });
                Ok(())
            }
            AgentCommand::Observe { input, request_id } => {
                hub.record_inbound(AgentCommand::Observe {
                    input: input.clone(),
                    request_id,
                });
                // Fold-only: the woken incarnation commits the folded context without opening a
                // model turn (`CoreIncarnation::fold_only`).
                self.enqueue_wire_input(session, SpliceKind::Observe, &input, provenance)
                    .await
            }
            AgentCommand::Interrupt { reason } => {
                hub.record_inbound(AgentCommand::Interrupt { reason });
                // Idle = benign no-op, exactly like a live interrupt with no turn in flight.
                hub.interrupt();
                Ok(())
            }
            AgentCommand::Snapshot { request_id } => {
                if hub.request_snapshot(request_id) {
                    return Ok(());
                }
                // Commit-boundary consistency: the engine publishes `TurnFinished` and frees the
                // hub slot BEFORE the activation manager persists the checkpoint (`commit_turn`),
                // so an idle-path read raced against that window would serve the pre-turn
                // snapshot. `Active` marks exactly that in-flight cycle, and `Ready` marks queued
                // work (e.g. a boundary-drained Observe splice) whose commit is imminent — wait
                // both out (bounded) before peeking, mirroring the live actor's read-your-turn
                // snapshot semantics.
                let settle = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while matches!(
                    self.store.status(session).await,
                    Some(SessionStatus::Active | SessionStatus::Ready)
                ) && std::time::Instant::now() < settle
                {
                    // A turn may have started between `request_snapshot` and here: queue on it.
                    if hub.request_snapshot(request_id) {
                        return Ok(());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                let blob = self.store.peek_snapshot(session).await.ok_or_else(|| {
                    ApiError::Other(format!("session {session} has no durable snapshot"))
                })?;
                let snap = Snapshot::decode(&blob)
                    .map_err(|e| ApiError::Other(format!("decode durable snapshot: {e}")))?;
                hub.publish_snapshot(request_id, daemon_core::conv_view_of(&snap));
                Ok(())
            }
            AgentCommand::Shutdown => {
                hubs.detach(session);
                Ok(())
            }
            AgentCommand::RewindTo { anchor, request_id } => {
                // A bare `RewindTo` command rewinds the conversation *and* rolls the workspace
                // back — the live command's historical behavior. Conversation-only rewind is
                // reachable via `ControlApi::rewind` with `restore_workspace = false`.
                self.rewind_durable(session, &anchor, request_id, true)
                    .await
            }
            // `#[non_exhaustive]`: a future command must be routed deliberately, not dropped.
            other => Err(ApiError::Unsupported(format!(
                "durable routing for {other:?}"
            ))),
        }
    }

    /// Sticky-on-first-open profile binding for the durable rail (§8): record `explicit`, else the
    /// node's ACTIVE default, onto the durable meta — the same resolution the live builder performs
    /// at open (`store.active()`), persisted so every later hydrate resolves the identical profile
    /// through the `DurableProfileResolver`. A recorded binding is never overwritten.
    async fn bind_profile_on_first_open(&self, session: &SessionId, explicit: Option<ProfileRef>) {
        let bind = match explicit {
            Some(p) => Some(p),
            None => self
                .profile_store()
                .ok()
                .and_then(|s| s.active().ok().flatten())
                .map(ProfileRef::new),
        };
        let Some(profile) = bind else { return };
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        if meta.bound_profile.is_none() {
            meta.bound_profile = Some(profile);
            let _ = self.store.set_session_meta(session, meta).await;
        }
    }

    /// The wire-addressed durable input rail (§8): splice, then — Completed only — REOPEN the
    /// settled session (`Completed → Ready`), then wake. An explicit client submit addressed at a
    /// settled one-shot resurrects it over its retained durable transcript (the pre-cutover live
    /// rail opened a BLANK fresh engine over the id; the durable resume is strictly better).
    /// Splice-before-reopen so a raced scanner never sees a `Ready` row with nothing to run.
    /// Host-originated injections keep dropping input at settled sessions (`inject_session_msg`).
    async fn enqueue_wire_input(
        &self,
        session: &SessionId,
        kind: SpliceKind,
        msg: &UserMsg,
        provenance: String,
    ) -> Result<(), ApiError> {
        let mut payload = Vec::new();
        ciborium::into_writer(msg, &mut payload)
            .map_err(|e| ApiError::Other(format!("encode wire input: {e}")))?;
        self.store
            .append_splice(NewSplice {
                session_id: session.clone(),
                kind,
                payload,
                origin_op: mint_op_id(),
                origin: provenance,
            })
            .await
            .map_err(|e| ApiError::Other(format!("append wire input: {e}")))?;
        let _ = self.store.reopen_if_settled(session).await;
        self.store.enqueue_wake(session.clone()).await;
        Ok(())
    }

    /// Rewind a DURABLE (non-resident) session (§8 + conversation-rewind spec): interrupt-first
    /// when a turn occupies the session's hub slot (waiting for its commit to land), then a
    /// dormant-only compare-and-swap of the persisted snapshot through the shared snapshot
    /// surgery ([`daemon_core::rewind_snapshot`] — the same truncate + reset + epoch bump the
    /// resident engine performs), then the shared durable side-effects (journal seal + optional
    /// workspace rollback). The `Rewound` event is published on the hub so subscribers see the
    /// same reply a live rewind streams.
    async fn rewind_durable(
        &self,
        session: &SessionId,
        anchor: &daemon_protocol::RewindAnchor,
        request_id: daemon_common::ReqId,
        restore_workspace: bool,
    ) -> Result<(), ApiError> {
        // Interrupt-first: cooperatively cancel a resident turn and wait (bounded) for it to
        // commit and vacate the slot, so the CAS below sees a dormant session.
        if let Some(hubs) = &self.attachments {
            if let Some(hub) = hubs.get(session) {
                if hub.occupied() {
                    hub.interrupt();
                    let mut occupancy = hub.occupancy();
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_secs(15),
                        occupancy.wait_for(|occupied| !occupied),
                    )
                    .await;
                }
            }
        }
        // Dormant-only CAS: the swap refuses while an activation is `Active` or when the blob
        // changed under us; bounded retries cover the interrupt's commit landing just after the
        // occupancy flip.
        for attempt in 0..5u32 {
            let blob = self
                .store
                .peek_snapshot(session)
                .await
                .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
            let mut snap = Snapshot::decode(&blob)
                .map_err(|e| ApiError::Other(format!("decode durable snapshot: {e}")))?;
            let outcome = daemon_core::rewind_snapshot(&mut snap, anchor)
                .map_err(|e| ApiError::Other(e.to_string()))?;
            let new = snap
                .encode()
                .map_err(|e| ApiError::Other(format!("encode rewound snapshot: {e}")))?;
            let swapped = self
                .store
                .swap_snapshot_if_dormant(session, &blob, new)
                .await
                .map_err(|e| ApiError::Other(format!("rewind swap: {e}")))?;
            if swapped {
                self.live
                    .seal_and_rollback_after_rewind(session, &outcome, restore_workspace)
                    .await;
                if self.attachments.is_some() {
                    self.attach_hub(session).await.publish_event(
                        daemon_protocol::AgentEvent::Rewound {
                            seq: 0,
                            request_id,
                            to_cursor: outcome.retained_turns as u64,
                            epoch: outcome.epoch.0,
                        },
                    );
                }
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(50 << attempt)).await;
        }
        Err(ApiError::Conflict(format!(
            "session {session} stayed active; rewind could not claim a dormant snapshot"
        )))
    }

    /// The F4 durable-resume gate: whether a wire `Submit { StartTurn | Steer }` addressed at
    /// `session` must ride the durable inbox rail instead of opening a fresh live incarnation.
    /// Returns `Some((kind, msg))` — the typed splice to fold into the durable transcript —
    /// only for a **parked-durable** session: NOT live-resident (a resident actor — Foreign —
    /// keeps its rail) AND a durable activation row evidences it live-but-dormant
    /// (`Active | Suspended | Ready`, never `Completed`/absent). A `Completed` durable session
    /// keeps today's fresh-incarnation behavior (its durable owner is gone), and any non-
    /// `StartTurn`/`Steer` command falls through to the live path (`None`). The caller enforces
    /// ownership (Auth 4) before enqueuing.
    async fn durable_resume_input(
        &self,
        session: &SessionId,
        command: &AgentCommand,
    ) -> Option<(SpliceKind, UserMsg)> {
        let (kind, msg) = match command {
            AgentCommand::StartTurn { input, .. } => (SpliceKind::StartTurn, input.clone()),
            AgentCommand::Steer { text, .. } => (SpliceKind::Steer, UserMsg::new(text.clone())),
            _ => return None,
        };
        if self.live.is_resident(session) {
            return None;
        }
        match self.store.status(session).await {
            Some(
                SessionStatus::Active | SessionStatus::Suspended { .. } | SessionStatus::Ready,
            ) => Some((kind, msg)),
            // Completed / absent: not parked-durable — fall through to the live path.
            _ => None,
        }
    }
}

/// Mint a fresh splice op-id: `op-<32 hex>` from 16 random bytes (mirrors `mint_session_id`). A
/// getrandom failure is astronomically unlikely; fall back to a time-seeded id rather than
/// panicking.
fn mint_op_id() -> String {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_err() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        bytes.copy_from_slice(&nanos.to_le_bytes());
    }
    let mut hex = String::with_capacity(3 + bytes.len() * 2);
    hex.push_str("op-");
    for b in bytes {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

mod access;
mod assembly;
pub mod attachments;
mod authorized;
mod builtins;
mod control;
mod cred_auth;
mod delivery;
mod feedback;
mod journal_audit;
mod membership;
mod messaging;
mod model;
mod overlay;
mod profile;
mod provisioning;
mod roster;
mod routing;
mod session;
mod vhc;

mod internals;

// Public re-exports (the stable `node_api::*` surface lib.rs re-exports for daemon-node / daemon-ffi
// / daemon-conformance).
pub use assembly::NodeApiParts;
pub use attachments::{AttachmentHub, AttachmentHubs};
pub use delivery::DeliveryHost;
pub use internals::NodeEventFeed;
pub use overlay::{decode_overlay, encode_overlay};
pub use provisioning::{AccountProvisioning, ProvisionedAccount};

// Crate-internal re-exports so the sibling sub-modules (each `use super::*;`) resolve the helpers
// that live in another concern module.
pub(crate) use authorized::{AuthorizedFor, Session};
pub(crate) use builtins::command_err_to_api;
pub(crate) use internals::{
    apply_rewind_side_effects, index_and_title_session, LiveSessions, RewindSideEffects,
};
pub(crate) use messaging::participant_label;
pub(crate) use overlay::approval_mode_to_policy;
pub(crate) use profile::profile_err;
pub(crate) use roster::{
    filtered_tree, forward_event, owner_visible, paginate_roster, seed_title, session_in_scope,
    session_info_from,
};
pub(crate) use routing::{
    room_label, store_route_from_wire, transport_family_matches, wire_route_from_store,
};
