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
    ProviderKindWire,
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
    AgentCommand, AgentEvent, ConvView, DeliveryTarget, Direction, Disposition, HostRequest,
    HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, IsolationPolicy, Origin,
    OriginScope, SessionLogEntry, SessionPayload, SinkKind, TranscriptBlock, TransportId, UserMsg,
};
use daemon_store::{
    FeedbackRecord, SessionMeta, SessionRole as StoreRole, SessionStatus, SessionStore,
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
    /// A foreign engine, materialized by the injected factory at `ensure` time.
    Foreign(ForeignSessionFactory),
}

/// Resolve a session's effective [`EngineProfile`] from its bound profile ref + persisted overlay —
/// the durable-path counterpart of [`SessionEngineBuilder`], injected into [`CoreEngineFactory`] by
/// the node (which owns the profile store + resolution rules). Returns `None` when no profile store
/// is configured or the bound profile is absent, so the durable path falls back to the factory's
/// default (orchestrator) profile. This is the seam that makes durable rehydration re-resolve from
/// the profile store + overlay instead of pinning the factory's fixed profile.
pub type DurableProfileResolver = Arc<
    dyn Fn(Option<ProfileRef>, &SessionOverlay) -> Option<daemon_core::EngineProfile> + Send + Sync,
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

    /// The discoverable provider catalog for the setup picker: local engines + every genai cloud
    /// vendor + Daemon Cloud. Static metadata (no network); independent of the launch default, so an
    /// unconfigured node still lists providers.
    async fn providers(&self) -> Vec<ProviderDescriptor>;

    /// One provider's discoverable models, keyed by [`ProviderDescriptor::id`]. Credential-aware for
    /// genai vendors (the resolved `key` authenticates the LIST call); Daemon Cloud lists keyless.
    /// Local engines are served by the host from the `ModelManager` catalog, not here.
    async fn provider_models(&self, provider_id: &str, key: Option<String>)
        -> Vec<ModelDescriptor>;
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
    /// The lazily-opened verifiable-journal writer for the `node-management` stream: management
    /// mutations (`conv_*`/`member_*`) are recorded + sealed onto it so the audit chains per op.
    /// `None` until the first mutation (and stays `None` when journaling is disabled).
    mgmt_journal: Arc<std::sync::Mutex<Option<Arc<JournalSink>>>>,
    /// The foreign-agent discovery hook (I7), injected by the binary (which owns the ACP runtime).
    /// `None` => `agent_discover` yields nothing and the catalog is just the durable manual
    /// registrations.
    agents: Option<Arc<dyn AgentDiscovery>>,
    /// The last discovery scan's results, cached in-memory so `agent_catalog` can surface them
    /// alongside the durable manual entries without re-probing every read (discovery is the
    /// operator-triggered, subprocess-spawning scan; manual entries are the persisted half).
    last_agents: Arc<std::sync::RwLock<Vec<daemon_api::AgentEntry>>>,
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
}

impl NodeApiImpl {
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

    /// Inject host-originated input (a background process-exit notification, a watch-pattern match,
    /// a message to a managed child) into `session`'s conversation, driving a reactive turn — the
    /// one seam that works across **both** session lifecycles:
    ///
    /// - a **live** (actor-resident) session takes a real [`AgentCommand::StartTurn`] through the
    ///   normal submit path (Observe-while-idle only folds context and drives no turn, so a
    ///   notification would otherwise sit unseen until the user next speaks);
    /// - a **durable** (activation-lifecycle) session — which `submit` must reject under the
    ///   one-lifecycle-owner guard-rail — gets a durable pending input
    ///   ([`SessionStore::enqueue_session_input`]) plus a wake; the incarnation drains it into the
    ///   conversation at hydrate and the woken turn runs with it.
    ///
    /// An **unclaimed** id (the in-memory owner map is empty after a restart) routes by durable
    /// evidence: a session with a durable activation row takes the store seam (never spawning a
    /// divergent live engine over durable state); anything else opens the live path, exactly like
    /// an inbound message would. A `Completed` durable session drops the input (its owner is gone).
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
        let owner = self.owners.get(session).map(|o| *o.value());
        let durable = match owner {
            Some(Lifecycle::Live) => false,
            Some(Lifecycle::Durable) => true,
            None => self.store.status(session).await.is_some(),
        };
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
            self.enqueue_durable_input(session, &msg).await?;
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

    /// The shared durable pending-input rail: encode `msg`, enqueue it on the durable session's
    /// FIFO pending-input queue ([`SessionStore::enqueue_session_input`]) + a wake. The woken
    /// incarnation drains it into the conversation at hydrate. Used by both the host-originated
    /// injection seam ([`Self::inject_session_msg`]) and the F4 durable-resume submit gate
    /// ([`Self::durable_resume_input`]).
    async fn enqueue_durable_input(
        &self,
        session: &SessionId,
        msg: &UserMsg,
    ) -> Result<(), ApiError> {
        let mut payload = Vec::new();
        ciborium::into_writer(msg, &mut payload)
            .map_err(|e| ApiError::Other(format!("encode injected input: {e}")))?;
        self.store.enqueue_session_input(session, payload).await;
        self.store.enqueue_wake(session.clone()).await;
        Ok(())
    }

    /// The F4 durable-resume gate: whether a wire `Submit { StartTurn | Steer }` addressed at
    /// `session` must ride the durable pending-input rail instead of opening a fresh live
    /// incarnation. Returns `Some(msg)` — the [`UserMsg`] to fold into the durable transcript —
    /// only for a **parked-durable** session: the durable lifecycle owns it (or, when unclaimed
    /// after a restart, a durable activation row evidences it) AND it is live-but-dormant
    /// (`Active | Suspended | Ready`, never `Completed`/absent). A `Completed` durable session
    /// keeps today's fresh-incarnation behavior (its durable owner is gone), and any non-
    /// `StartTurn`/`Steer` command falls through to the live path (`None`). The caller enforces
    /// ownership (Auth 4) before enqueuing.
    async fn durable_resume_input(
        &self,
        session: &SessionId,
        command: &AgentCommand,
    ) -> Option<UserMsg> {
        let msg = match command {
            AgentCommand::StartTurn { input, .. } => input.clone(),
            AgentCommand::Steer { text, .. } => UserMsg::new(text.clone()),
            _ => return None,
        };
        let durable = match self.owners.get(session).map(|o| *o.value()) {
            Some(Lifecycle::Live) => false,
            Some(Lifecycle::Durable) => true,
            None => self.store.status(session).await.is_some(),
        };
        if !durable {
            return None;
        }
        match self.store.status(session).await {
            Some(
                SessionStatus::Active | SessionStatus::Suspended { .. } | SessionStatus::Ready,
            ) => Some(msg),
            // Completed / absent: not parked-durable — fall through to the live path.
            _ => None,
        }
    }
}

mod access;
mod assembly;
mod authorized;
mod builtins;
mod control;
mod cred_auth;
mod delivery;
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

mod internals;

// Public re-exports (the stable `node_api::*` surface lib.rs re-exports for daemon-node / daemon-ffi
// / daemon-conformance).
pub use assembly::NodeApiParts;
pub use delivery::DeliveryHost;
pub use internals::NodeEventFeed;
pub use overlay::{decode_overlay, encode_overlay};
pub use provisioning::{AccountProvisioning, ProvisionedAccount};

// Crate-internal re-exports so the sibling sub-modules (each `use super::*;`) resolve the helpers
// that live in another concern module.
pub(crate) use authorized::{AuthorizedFor, Session};
pub(crate) use builtins::command_err_to_api;
pub(crate) use internals::{apply_rewind_side_effects, LiveSessions, RewindSideEffects};
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
