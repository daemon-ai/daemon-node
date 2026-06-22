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
use crate::credstore::CredentialStore;
use crate::profiles::ProfileStore;
use crate::routing::RoutingRegistry;
use daemon_api::{
    ApiError, ApprovalInfo, ConfigPatch, ConfigSchema, ControlApi, CredentialApi, CredentialInfo,
    FleetReport, HealthReport, JournalPageView, JournalRecord, JournalRecordPayload, LogPageView,
    LogStream, ManageEventView, ModelApi, ModelDescriptor, Outbound, ProfileApi, ProfileInfo,
    ProfileSpec, ProviderSelector, ServiceHealth, SessionApi, SessionInfo, SessionState,
    StatsReport, TelemetryDump, TreeReport, UnitNode,
};
use daemon_common::{
    ContentHash, DownloadId, DownloadStatus, GgufInfo, InstalledModel, JobId, JournalStreamId,
    ModelEngine, ModelFile, ModelId, ModelRef, PartitionId, ProfileRef, QuantRecommendation,
    QuantizeId, QuantizeStatus, ReqId, SearchPage, SearchQuery, SessionId, UnitId, UsageDelta,
};
use daemon_core::{spawn_agent_session, AgentHandle, Engine, Provider, Snapshot};
use daemon_models::{ModelError, ModelManager};
use daemon_protocol::{
    AgentCommand, DeliveryTarget, Direction, Disposition, HostRequest, HostRequestHandler,
    HostRequestKind, HostResponse, HostResponseBody, Origin, OriginScope, SessionLogEntry,
    SessionPayload, SinkKind, TranscriptBlock, TransportId,
};
use daemon_store::{SessionStatus, SessionStore};
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
/// `None` resolves the node's active default, preserving the legacy single-profile behavior.
pub type SessionEngineBuilder = Arc<dyn Fn(SessionId, Option<ProfileRef>) -> Engine + Send + Sync>;

/// Builds a fresh model [`Provider`] from a (model-overridden) [`ProfileSpec`] — the seam a live
/// [`SessionApi::set_session_model`](daemon_api::SessionApi::set_session_model) uses to rebuild a
/// running session's provider without `daemon-host` linking the provider crate.
pub type ModelProviderFactory = Arc<dyn Fn(&ProfileSpec) -> Arc<dyn Provider> + Send + Sync>;

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
    /// The host routing registry (daemon-event-io-spec §5.9) consulted by [`SessionApi::submit_routed`]
    /// to resolve an inbound `Origin` to (session, profile, delivery). Empty by default — a pure
    /// passthrough: `PerThread` naming, node active-default profile, origin-seeded delivery.
    routing: Arc<RoutingRegistry>,
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
        Self {
            supervisor,
            store,
            manager,
            fleet,
            partition,
            live: Arc::new(LiveSessions::new(engine_builder, session_modes.clone())),
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
            routing: Arc::new(RoutingRegistry::new()),
        }
    }

    /// Install the host routing registry consulted by [`SessionApi::submit_routed`] (the §5.9
    /// inbound-routing capability). Call during assembly; absent, routed submits fall back to
    /// `PerThread` naming with the node's active default profile.
    pub fn with_routing(mut self, routing: RoutingRegistry) -> Self {
        self.routing = Arc::new(routing);
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

    /// Fold the durable per-session usage totals across every known session — the node-wide
    /// accounting line (tokens, cache, reasoning, estimated `cost_micros`) reported on `stats`.
    async fn folded_usage(&self) -> UsageDelta {
        let mut total = UsageDelta::default();
        for (session, _status) in self.store.list_sessions().await {
            total.add(&self.store.usage_of(&session).await);
        }
        total
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

        JournalPageView {
            entries,
            next_cursor: page.next_cursor,
            head_cursor: page.head_cursor,
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
        self.store
            .list_sessions()
            .await
            .into_iter()
            .map(|(session, status)| SessionInfo {
                session,
                state: map_state(status),
            })
            .collect()
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
}

#[async_trait]
impl SessionApi for NodeApiImpl {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // Guard-rail: claim the session for the live lifecycle (rejects an id already durable-managed).
        self.claim(&session, Lifecycle::Live)?;
        self.live.submit(session, command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        self.claim(&session, Lifecycle::Live)?;
        self.live.submit_from(session, origin, command).await
    }

    async fn submit_routed(
        &self,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<SessionId, ApiError> {
        // Resolve the origin through the §5.9 routing registry: session name, the profile that runs
        // it (agent selection), and where its replies post.
        let resolved = self.routing.resolve(&origin);
        self.claim(&resolved.session, Lifecycle::Live)?;
        // For session-opening commands, bind the resolved profile (sticky on first `ensure`) and seed
        // the resolved `Primary` before submitting, so routing owns agent-selection + delivery. Other
        // commands act on an already-open session whose profile/Primary were bound when it opened.
        if matches!(
            command,
            AgentCommand::StartTurn { .. } | AgentCommand::Steer { .. }
        ) {
            self.live.ensure(&resolved.session, resolved.profile.clone());
            self.live
                .seed_primary_target(&resolved.session, resolved.delivery.clone());
        }
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
        let factory = self.model_factory.as_ref().ok_or_else(|| {
            ApiError::Unsupported("per-session model switch is not available".into())
        })?;
        // Only a resident (live-actor) session can have its provider swapped in place; a durable
        // session re-resolves its model from the profile on rehydration.
        let handle = self
            .live
            .handle_if_live(&session)
            .ok_or_else(|| ApiError::UnknownSession(session.as_str().to_string()))?;
        // Derive the new provider from the session's bound profile (the active default), overriding
        // the model (and optionally the provider). The profile itself is left unmodified.
        let mut spec = self.resolve_profile(None)?.ok_or_else(|| {
            ApiError::Unsupported("no active profile to derive a provider from".into())
        })?;
        spec.model = model.clone();
        if let Some(p) = provider {
            spec.provider = p;
        }
        let new_provider = (factory)(&spec);
        handle.set_provider(new_provider).await;
        self.session_models.insert(session, model);
        Ok(())
    }

    async fn set_session_mode(
        &self,
        session: SessionId,
        mode: daemon_api::ApprovalMode,
    ) -> Result<(), ApiError> {
        let policy = approval_mode_to_policy(mode);
        // Only a resident (live-actor) session can have its mode switched in place; a durable
        // session re-resolves its policy from the engine config / snapshot on rehydration.
        let handle = self
            .live
            .handle_if_live(&session)
            .ok_or_else(|| ApiError::UnknownSession(session.as_str().to_string()))?;
        handle.set_approval_policy(policy).await;
        // Record the live mode so the live ParkingHandler can consult it (auto-allow vs park).
        self.session_modes.insert(session, policy);
        Ok(())
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
        self.profile_store()?.create(spec).map_err(profile_err)
    }

    async fn profile_update(&self, spec: ProfileSpec) -> Result<(), ApiError> {
        self.profile_store()?.update(spec).map_err(profile_err)
    }

    async fn profile_delete(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.delete(&id).map_err(profile_err)
    }

    async fn profile_select(&self, id: String) -> Result<(), ApiError> {
        self.profile_store()?.set_active(&id).map_err(profile_err)
    }

    async fn config_get(&self, profile: Option<String>) -> Result<Option<ProfileSpec>, ApiError> {
        self.resolve_profile(profile)
    }

    async fn config_set(
        &self,
        profile: Option<String>,
        patch: ConfigPatch,
    ) -> Result<(), ApiError> {
        let store = self.profile_store()?;
        let id = match profile {
            Some(id) => id,
            None => store
                .active()
                .map_err(profile_err)?
                .ok_or_else(|| ApiError::Other("no active profile to patch".into()))?,
        };
        let mut spec = store
            .get(&id)
            .map_err(profile_err)?
            .ok_or_else(|| ApiError::UnknownSession(id.clone()))?;
        patch.apply(&mut spec);
        store.update(spec).map_err(profile_err)
    }

    async fn config_schema(&self) -> ConfigSchema {
        ConfigSchema::builtin()
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

    /// Stamp the next `seq`, record the entry, and fan it out to live subscribers.
    fn append(
        &mut self,
        direction: Direction,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) {
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
        let _ = self.tx.send(entry);
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
    /// The verifiable-journal store + signer, when journaling is enabled for live sessions.
    journal: Mutex<Option<JournalConfig>>,
    /// The §4.3 background-spawn materializer, when configured: lets a live session's `Effect::Spawn`
    /// materialize an attached non-joining review child without parking (fire-and-forget).
    background: Mutex<Option<Arc<crate::background::BackgroundSpawner>>>,
    /// The per-session live edit-approval policy (shared with `NodeApiImpl::session_modes`), read by
    /// each session's [`ParkingHandler`] to auto-allow / deny without parking a human.
    modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
}

impl LiveSessions {
    fn new(
        builder: SessionEngineBuilder,
        modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
    ) -> Self {
        Self {
            sessions: DashMap::new(),
            builder,
            journal: Mutex::new(None),
            background: Mutex::new(None),
            modes,
        }
    }

    fn set_journal(&self, cfg: JournalConfig) {
        *self.journal.lock().unwrap() = Some(cfg);
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
    fn ensure(&self, session: &SessionId, profile: Option<ProfileRef>) -> AgentHandle {
        if let Some(s) = self.sessions.get(session) {
            return s.handle.clone();
        }
        let engine = (self.builder)(session.clone(), profile);
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
        let pump = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        pump_log.lock().unwrap().append(
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
                let handle = self.ensure(&session, None);
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
                let handle = self.ensure(&session, None);
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
            _ => Err(ApiError::Unsupported("unknown agent command".into())),
        }
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
