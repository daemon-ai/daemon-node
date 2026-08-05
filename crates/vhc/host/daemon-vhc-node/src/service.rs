// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`VhcService`] — the resident node-side vhc-training service (spec §10.3/§10.4).
//!
//! It owns a worker-control seam ([`WorkerControl`], implemented for `daemon-vhc-supervisor`'s
//! `TrainSupervisor`), the durable [`VhcStore`] (`vhc.db`), and a broadcast of [`VhcEvent`]s.
//! It:
//!
//! - Translates worker [`protocol::Event`]s into [`VhcEvent`]s, persists them to the windowed log, folds contribution counters, broadcasts to `vhc_subscribe`, and emits a payload-free [`NodeEvent::VhcChanged`] pointer onto the node feed (§10.4).
//! - Drives **durable-intent re-convergence** on [`start`](VhcService::start): re-issues `JoinRun` for every persisted active join-intent so a restart rejoins without app involvement (§10.3).
//! - Is **OFF by default** (`[vhc] enabled = false`): a disabled service never touches the worker, so no training worker is ever spawned unless vhc is enabled.
//! - Implements [`VhcApi`], mapping requests → worker commands + store reads (eligibility is node-computed from the worker probe/assess and mirrored, ADR-003 — the app never re-derives it).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    ApiError, NodeEvent, VhcApi, VhcCapabilities, VhcEligibility, VhcEvent, VhcEventStream,
    VhcHardwareReport, VhcLeaveMode, VhcPolicy, VhcPolicyMode, VhcRunDetail, VhcRunSummary,
};
use daemon_vhc_abi::{AbiRefusalCode, RESERVATION_DEVICE_BYTES_KEY, RESERVATION_HOST_BYTES_KEY};
use daemon_vhc_session::config::VhcConfig;
use daemon_vhc_session::protocol::{
    self, Eligibility, Hardware, JoinPolicy, LeaveMode, PolicyMode,
};
use daemon_vhc_supervisor::{SwitchOutcome, TrainSupervisor};
use futures::StreamExt;
use std::collections::BTreeMap;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::arbiter::{
    AdmitRefusal, ClaimTiers, InstanceCharge, OwnerArbiter, OwnerBudget, RoleInstanceId, TierBytes,
};
use crate::discovery::RunDiscovery;
use crate::store::{
    effective_state, DesiredState, PersistedRun, RunState, StoreError, VhcStore, EVENT_WINDOW,
};

/// A node-feed sink: the node passes a closure over `NodeEventFeed::emit` so live vhc updates ride
/// the existing `events_subscribe` channel as `VhcChanged` pointers (no new transport).
pub type NodeFeed = Arc<dyn Fn(NodeEvent) + Send + Sync>;

/// A per-role-instance worker factory (Phase E multi-instance supervision, decisions D1/D6):
/// when configured, every admitted join gets its **own** supervised child (one sandbox = one
/// role-instance) instead of sharing the single default worker. In production this spawns a
/// fresh `TrainSupervisor`; tests hand out recording fakes.
pub type WorkerFactory = Arc<dyn Fn() -> Arc<dyn WorkerControl> + Send + Sync>;

/// A vhc-service error.
#[derive(Debug, thiserror::Error)]
pub enum VhcError {
    /// A `vhc.db` error.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// A worker-control failure (mapped from the supervisor).
    #[error("worker: {0}")]
    Worker(String),
    /// A run-discovery / envelope-fetch failure (registry unreachable, run unknown, envelope hash
    /// mismatch — the §6.1/§6.5 join-time discovery seam).
    #[error("discovery: {0}")]
    Discovery(String),
    /// The registry refused this node's roster publish as STALE and returned the slot's stored
    /// record — surfaced TYPED (never folded into a string) so the join transaction can judge
    /// it: a record that verifies to this node's OWN base identity for the same
    /// `(run, role, endpoint)` scope is own-floor evidence (the transaction restarts once above
    /// it, [`VhcStore::mint_incarnation_above`](crate::store::VhcStore::mint_incarnation_above));
    /// anything else fails closed. The registry's decision itself is structural retry SIGNAL,
    /// never trusted state — only the record's own verified signature repairs a counter.
    #[error("roster publish refused stale (stored incarnation {stored_incarnation})")]
    RosterStale {
        /// The registry-reported stored incarnation (diagnostic; the VERIFIED record governs).
        stored_incarnation: u64,
        /// The slot's stored record as returned (UNVERIFIED until judged).
        stored: Option<Box<daemon_vhc_proto::RosterRecord>>,
    },
    /// The join transaction observed ITS OWN fresher roster record — verified own-base
    /// evidence: the stored record's signature verifies, its certificate chains to THIS node's
    /// base identity, and its `(run, role, endpoint)` scope is this instance's — and repaired
    /// the local execution counter strictly above it. The transaction restarts authorship from
    /// the top ONCE with a freshly minted incarnation (a changed incarnation changes the
    /// reservation key, the admitted tuple, the certificate, and the credentials — it can never
    /// be an internal publish retry).
    #[error("own roster floor {floor} observed and repaired; the join transaction restarts")]
    OwnFloorRepaired {
        /// The verified floor incarnation the counter was raised above.
        floor: u64,
    },
    /// The role's freshest restore checkpoint is too far behind the run's live head for the
    /// retained record horizon
    /// ([`daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS`]): replay-forward could never
    /// bridge the gap, so the join is refused TYPED at join time — before rehydration — instead
    /// of wedging into the module's `GapRefused`/`StaleRestore` after it (the recovery-honesty
    /// check; the in-module refusal remains the authoritative backstop).
    #[error(
        "checkpoint too stale for the retained horizon: restored fence {restored} vs live head \
         {head} (horizon {horizon} rounds)"
    )]
    CheckpointStale {
        /// The restore fence (the freshest pointer round for the joining role).
        restored: u64,
        /// The live-head estimate (the freshest checkpoint round visible for the run).
        head: u64,
        /// The retained record horizon (rounds).
        horizon: u64,
    },
    /// The owner arbiter refused the join: an aggregate resource ledger is exhausted (decisions
    /// D6 — the admission funnel's last, supreme stage; the owner can always refuse).
    #[error("owner arbitration refused the join: {0}")]
    Resources(#[from] AdmitRefusal),
    /// The vhc service is disabled (`[vhc] enabled = false`).
    #[error("vhc is disabled")]
    Disabled,
    /// The discovered coordinator endpoint is outside `[vhc].coordinator_allowlist` (spec §11.1):
    /// a typed refusal — the node never dials, assesses against, or authors credentials for an
    /// endpoint the owner has not allowlisted.
    #[error("coordinator endpoint refused by the allowlist: {0}")]
    AllowlistRefused(String),
    /// An internal invariant failure (e.g. canonical-CBOR encode of the admitted tuple).
    #[error("internal: {0}")]
    Internal(String),
}

impl VhcError {
    fn worker(e: impl std::fmt::Display) -> Self {
        Self::Worker(e.to_string())
    }

    fn to_api(&self) -> ApiError {
        match self {
            VhcError::Disabled => ApiError::Unsupported("vhc is disabled".into()),
            other => ApiError::Other(other.to_string()),
        }
    }
}

/// The worker-supervision seam the service drives (join/leave/probe/assess/throttle). Implemented for
/// `daemon-vhc-supervisor`'s `TrainSupervisor` (real worker); a fake impl in tests exercises the
/// service without a subprocess.
#[async_trait]
pub trait WorkerControl: Send + Sync {
    /// Probe hardware + capability vocabulary (§10.2).
    async fn probe(&self) -> Result<Hardware, VhcError>;
    /// Assess a run envelope against effective resources (§6.5) — the eligibility source.
    /// `role` names the envelope role to assess for (node-directed selection — the seat-claim
    /// path directs the coordinator role); `None` = the single-trainer default.
    async fn assess(
        &self,
        envelope: Vec<u8>,
        role: Option<String>,
    ) -> Result<Eligibility, VhcError>;
    /// Assess a live-upgrade TARGET (ABI §10.3 pre-switch assessment): the worker — which alone
    /// touches module bytes — resolves the hash-pinned target, re-derives the grants document
    /// against the committed record's grants anchor, runs the claim admission funnel, and
    /// answers with a post-switch admitted tuple (claim hash computed worker-side). Default:
    /// unsupported (fakes / non-upgrading workers); `TrainSupervisor` overrides it.
    async fn assess_switch(
        &self,
        envelope: Vec<u8>,
        role: Option<String>,
        target: daemon_vhc_session::protocol::SwitchTarget,
    ) -> Result<Eligibility, VhcError> {
        let _ = (envelope, role, target);
        Err(VhcError::Worker(
            "assess_switch unsupported by this worker".into(),
        ))
    }
    /// Join a run.
    async fn join(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError>;
    /// Join a run and return the **continuous** worker event stream (the continuous event pump). The default
    /// delegates to [`join`](Self::join) and returns an already-closed receiver, so test fakes and
    /// non-streaming workers keep the drain-and-drop behavior; `TrainSupervisor` overrides it with the real
    /// per-round stream. `admitted_tuple` carries the node-minted incarnation the worker runs as.
    async fn join_streaming(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<protocol::Event>, VhcError> {
        self.join(run_id, coordinator, credentials, policy, admitted_tuple)
            .await?;
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        Ok(rx)
    }
    /// Leave a run.
    async fn leave(&self, run_id: String, mode: LeaveMode) -> Result<(), VhcError>;
    /// Push a GPU-governor throttle lever (§10.5).
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), VhcError>;
    /// Initiate a live module upgrade for a running instance (ABI §10.3; architecture §5.4) — the
    /// node command surface for `SwitchModule`. The run-level upgrade record has already committed
    /// to the transition chain (deliverable 1); this drives the LOCAL half of the transaction on
    /// the worker (quiesce → snapshot → owner-law re-admission → migrate → validate → activate).
    /// Default: unsupported (fakes / non-upgrading workers); `TrainSupervisor` overrides it with
    /// the real command path.
    #[allow(clippy::too_many_arguments)]
    async fn switch_module(
        &self,
        run_id: String,
        epoch: u64,
        role: String,
        new_module: [u8; 32],
        grants_hash: [u8; 32],
        deadline_ms: u64,
        admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<SwitchOutcome, VhcError> {
        let _ = (
            run_id,
            epoch,
            role,
            new_module,
            grants_hash,
            deadline_ms,
            admitted_tuple,
        );
        Err(VhcError::Worker(
            "switch_module unsupported by this worker".into(),
        ))
    }
    /// Tear the worker down (a factory child that was refused admission or whose run left —
    /// Phase E multi-instance supervision). Default no-op for fakes/shared workers.
    async fn shutdown(&self) {}
}

#[async_trait]
impl WorkerControl for TrainSupervisor {
    async fn probe(&self) -> Result<Hardware, VhcError> {
        TrainSupervisor::probe(self).await.map_err(VhcError::worker)
    }
    async fn assess(
        &self,
        envelope: Vec<u8>,
        role: Option<String>,
    ) -> Result<Eligibility, VhcError> {
        TrainSupervisor::assess(self, envelope, role)
            .await
            .map_err(VhcError::worker)
    }
    async fn assess_switch(
        &self,
        envelope: Vec<u8>,
        role: Option<String>,
        target: daemon_vhc_session::protocol::SwitchTarget,
    ) -> Result<Eligibility, VhcError> {
        TrainSupervisor::assess_switch(self, envelope, role, target)
            .await
            .map_err(VhcError::worker)
    }
    async fn join(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        TrainSupervisor::join(
            self,
            run_id,
            coordinator,
            credentials,
            policy,
            admitted_tuple,
        )
        .await
        .map_err(VhcError::worker)
    }
    async fn join_streaming(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<protocol::Event>, VhcError> {
        TrainSupervisor::join_streaming(
            self,
            run_id,
            coordinator,
            credentials,
            policy,
            admitted_tuple,
        )
        .await
        .map_err(VhcError::worker)
    }
    async fn leave(&self, run_id: String, mode: LeaveMode) -> Result<(), VhcError> {
        TrainSupervisor::leave(self, run_id, mode)
            .await
            .map_err(VhcError::worker)
    }
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), VhcError> {
        TrainSupervisor::throttle(self, vram_cap_mb, duty_cycle_pct, paused)
            .await
            .map_err(VhcError::worker)
    }
    #[allow(clippy::too_many_arguments)]
    async fn switch_module(
        &self,
        run_id: String,
        epoch: u64,
        role: String,
        new_module: [u8; 32],
        grants_hash: [u8; 32],
        deadline_ms: u64,
        admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<SwitchOutcome, VhcError> {
        TrainSupervisor::switch_module(
            self,
            run_id,
            epoch,
            role,
            new_module,
            grants_hash,
            deadline_ms,
            admitted_tuple,
        )
        .await
        .map_err(VhcError::worker)
    }
    async fn shutdown(&self) {
        TrainSupervisor::shutdown(self).await;
    }
}

/// Construction parts for a [`VhcService`].
pub struct VhcServiceParts {
    /// The `[vhc]` config (spec §10.6); `enabled` gates all worker activity.
    pub config: VhcConfig,
    /// The durable `vhc.db` store.
    pub store: VhcStore,
    /// The default worker-control seam (a real `TrainSupervisor` in production) — the probe
    /// surface, and the shared child when no [`VhcServiceParts::worker_factory`] is set.
    pub worker: Arc<dyn WorkerControl>,
    /// The node-feed sink for `VhcChanged` pointers (`None` on a headless / test build).
    pub feed: Option<NodeFeed>,
    /// The run-discovery seam. When present, `vhc_join` discovers the run + fetches the frozen
    /// envelope + runs the worker's real §6.5 `AssessRun` before `JoinRun`. `None` keeps the
    /// probe-based eligibility path (no coordinator configured), so the service stays usable offline.
    pub discovery: Option<Arc<dyn RunDiscovery>>,
    /// The owner's aggregate resource grants (decisions D6). `None` = permissive
    /// ([`OwnerBudget::unbounded`]) — arbitration still keys/tracks every instance, it just
    /// never runs out.
    pub budget: Option<OwnerBudget>,
    /// The per-role-instance worker factory (Phase E N-sandbox supervision). `None` = every run
    /// shares the single default worker (the pre-E single-child behavior).
    pub worker_factory: Option<WorkerFactory>,
    /// The vhc identity keystore directory (the node's base identity + per-run key/cert/credential
    /// records). `Some` ⇒ the node AUTHORS per-run identity + credentials at join (D-P8: it mints
    /// the key, issues the certificate under the base identity, and delivers the minted incarnation
    /// in the admitted tuple). `None` (tests / headless) keeps the pre-authorship path (no
    /// credentials, no node-minted identity) — the worker's mandatory-tuple check then refuses a
    /// live join, which is exactly right off the production boot path.
    pub identity_dir: Option<std::path::PathBuf>,
    /// The registry seat-slot directory (architecture §6.3; D-P9). `Some` + `[vhc] seat_claim` +
    /// an identity store ⇒ the resident seat keeper covers every joined run whose admitted role
    /// is the configured seat role (claim on boot, heartbeat at the lease cadence, fenced release
    /// on pause/leave/shutdown). `None` = this node never claims (the trainer default).
    pub seat_directory: Option<Arc<dyn crate::seat_keeper::SeatDirectory>>,
}

/// The authored-join delivery: the tuple (incarnation stamped), the wire credentials bytes, and
/// the `credentials_ref` to persist.
type AuthoredDelivery = (Option<protocol::AdmittedTuple>, Vec<u8>, Option<String>);

/// The role-instance that authors + binds this node's SINGLE iroh endpoint for a run
/// (architecture [CI-10]: the iroh endpoint id is the NODE's transport identity and each admitted
/// node publishes ONE roster record per run). Identified by `(role, incarnation)` — the pair that
/// names a role-instance in the instance maps.
#[derive(Clone, Debug, PartialEq, Eq)]
struct IrohEndpointOwner {
    /// The owning role-instance's role.
    role: String,
    /// The owning role-instance's incarnation (== its generation).
    incarnation: u64,
}

/// One live supervised role-instance: its ledger identity and its worker child.
struct InstanceEntry {
    /// The identity ADMITTED against the owner ledgers — the arbiter's reservation key. It
    /// stays fixed for the entry's whole life (release must present the admitted key), even
    /// across a live module switch.
    id: RoleInstanceId,
    /// The instance's CURRENT generation (== incarnation): `id.instance` at join, advanced by
    /// a live module switch (the switch mints a new never-reused incarnation without a new
    /// sandbox). The stale-generation event guard compares against THIS.
    generation: u64,
    worker: Arc<dyn WorkerControl>,
}

/// The node-side vhc-training service.
pub struct VhcService {
    config: VhcConfig,
    store: Arc<VhcStore>,
    worker: Arc<dyn WorkerControl>,
    discovery: Option<Arc<dyn RunDiscovery>>,
    /// The D6 owner arbiter: every join is admitted against the aggregate typed ledgers before
    /// any child is touched, and released only on observed teardown.
    arbiter: OwnerArbiter,
    /// Per-role-instance children (Phase E N-sandbox supervision), keyed by `RunLabel`. Holds the
    /// node's PRIMARY role-instance for a run (the coordinator seat instance on a seat-claiming
    /// node, else the trainer).
    instances: Mutex<BTreeMap<String, InstanceEntry>>,
    /// The CO-LOCATED trainer role-instance a seat-holding node ALSO runs when
    /// `[vhc].coordinator_trains` is set (a trainer+coordinator box — the single-peer /
    /// self-coordinated fix, defect D). Keyed by `RunLabel`, SEPARATE from `instances` (which holds
    /// the coordinator seat instance) so the primary join/leave/reconcile bookkeeping is unchanged;
    /// its own worker child, incarnation, and ledger reservation. Empty unless the flag is set, so
    /// every existing gate is byte-identical.
    co_trainers: Mutex<BTreeMap<String, InstanceEntry>>,
    worker_factory: Option<WorkerFactory>,
    /// The identity keystore directory (D-P8 credential + per-run cert authorship); `None`
    /// disables node-side authorship (tests / headless).
    identity_dir: Option<std::path::PathBuf>,
    /// The resident coordinator seat keeper (present only when the owner enabled coordinator
    /// duty AND a seat directory + identity store are wired).
    seat: Option<crate::seat_keeper::SeatKeeper>,
    /// The registry seat READ seam, retained for EVERY node with a registry (not only claimers):
    /// a trainer reads the coordinator seat at join to bootstrap the incumbent's certificate into
    /// its session credentials (the seat lease is the out-of-band trust the on-plane §12.3
    /// distribution cannot supply to a late subscriber — the coordinator's one-shot announcement
    /// predates the trainer's connection).
    seat_read: Option<Arc<dyn crate::seat_keeper::SeatDirectory>>,
    events_tx: broadcast::Sender<VhcEvent>,
    feed: Option<NodeFeed>,
    /// The run the worker is currently on (from the last `RunPhase`), used to attribute events that
    /// don't carry a run id (`RoundProgress`/`RoundOutcome`/…).
    current_run: Mutex<Option<String>>,
    /// The coalescing vhc-feed revision stamped on each `VhcChanged` pointer.
    rev: AtomicU64,
    /// The pinned iroh bind port per run label (chosen once per node lifetime, so the published
    /// roster addresses and every re-authored credentials body agree on the socket).
    iroh_ports: Mutex<BTreeMap<String, u16>>,
    /// Which role-instance owns this node's single iroh endpoint per run label ([CI-10]) — see
    /// [`VhcService::claim_node_iroh_endpoint`]. Co-located siblings of the owner attach WS-only
    /// instead of binding a second socket on the node's one pinned port.
    iroh_endpoint: Mutex<BTreeMap<String, IrohEndpointOwner>>,
    /// The service's own `Arc` handle (the event-pump wiring), bound post-construction via [`bind_self`](Self::bind_self)
    /// so `vhc_join`/`start` can spawn a detached event-pump task that outlives the `&self` call.
    /// Unbound (test builds) → the non-streaming `join` path, drained-and-dropped.
    me: std::sync::OnceLock<std::sync::Weak<VhcService>>,
}

impl VhcService {
    /// Build a service. The worker is never touched until [`start`](Self::start) / an API call, and
    /// only when `config.enabled`.
    pub fn new(parts: VhcServiceParts) -> Self {
        let (events_tx, _) = broadcast::channel(1024);
        // The durable store is SHARED with the seat keeper (execution-incarnation counter +
        // persisted verified leadership-term floors live in one `vhc.db`).
        let store = Arc::new(parts.store);
        // Coordinator duty is opt-in AND fully wired or absent: the keeper exists only when the
        // owner enabled it and both the seat directory + the identity store are present.
        let seat = match (
            parts.config.seat_claim,
            &parts.seat_directory,
            &parts.identity_dir,
        ) {
            (true, Some(directory), Some(identity_dir)) => {
                Some(crate::seat_keeper::SeatKeeper::new(
                    directory.clone(),
                    identity_dir.clone(),
                    store.clone(),
                ))
            }
            _ => None,
        };
        let seat_read = parts.seat_directory.clone();
        Self {
            config: parts.config,
            store,
            worker: parts.worker,
            discovery: parts.discovery,
            arbiter: OwnerArbiter::new(parts.budget.unwrap_or_else(OwnerBudget::unbounded)),
            instances: Mutex::new(BTreeMap::new()),
            co_trainers: Mutex::new(BTreeMap::new()),
            worker_factory: parts.worker_factory,
            identity_dir: parts.identity_dir,
            seat,
            seat_read,
            events_tx,
            feed: parts.feed,
            current_run: Mutex::new(None),
            rev: AtomicU64::new(0),
            iroh_ports: Mutex::new(BTreeMap::new()),
            iroh_endpoint: Mutex::new(BTreeMap::new()),
            me: std::sync::OnceLock::new(),
        }
    }

    /// The owner arbiter (observability / tests): remaining ledgers + live instance count.
    pub fn arbiter(&self) -> &OwnerArbiter {
        &self.arbiter
    }

    /// Bind the service's own `Arc` handle (the continuous event pump), mirroring the node's `set_vhc`
    /// post-`Arc` binder. After this, `vhc_join` / `start` drive `join_streaming` + a detached pump
    /// task feeding the worker's continuous event stream into [`handle_worker_event`](Self::handle_worker_event)
    /// → `NodeEvent::VhcChanged`, so `vhc.db` reflects live round progression (§10.3/§10.4).
    /// Idempotent; never bound → the non-streaming join path (unchanged, for tests).
    pub fn bind_self(self: &Arc<Self>) {
        let _ = self.me.set(Arc::downgrade(self));
    }

    /// Join a run and pump its continuous worker event stream into the service. The public entry
    /// the boot site / e2e use to drive a **live-attach** join with authored `JoinCredentials`; the
    /// pump feeds each event through `handle_worker_event`. Requires [`bind_self`](Self::bind_self).
    pub async fn join_and_pump(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        let generation = admitted_tuple.as_ref().map_or(0, |t| t.incarnation);
        let rx = self
            .worker
            .join_streaming(
                run_id.clone(),
                coordinator,
                credentials,
                policy,
                admitted_tuple,
            )
            .await?;
        self.spawn_pump(Some((run_id, generation)), rx);
        Ok(())
    }

    /// Spawn the detached event-pump task that drains a worker event stream into
    /// [`handle_worker_event`](Self::handle_worker_event). When the service is unbound (tests) it
    /// drains-and-drops so a streaming worker never backs up.
    ///
    /// `instance` names the `(run, generation)` this stream belongs to. When it is known and the
    /// stream ends WITHOUT that instance's terminal event, the closure itself is the observation
    /// of teardown (worker crash / transport loss — the supervisor closes the pump sink when the
    /// child's stream ends), and the instance transitions `failed_retryable` so the durable join
    /// intent reconverges under the retry budget.
    fn spawn_pump(
        &self,
        instance: Option<(String, u64)>,
        mut rx: tokio::sync::mpsc::UnboundedReceiver<protocol::Event>,
    ) {
        match self.me.get().and_then(std::sync::Weak::upgrade) {
            Some(me) => {
                tokio::spawn(async move {
                    let mut terminated = false;
                    // Whether THIS instance's session has spoken (any generation-matching
                    // RunPhase). Before that, the child holds no session: a typed worker error
                    // (e.g. an assess/join refusal) is the join transaction FAILING, and the
                    // session's own terminal event will never come — the pump synthesizes it.
                    // After it, errors are informational; termination arrives as RunTerminated.
                    let mut sessioned = false;
                    while let Some(ev) = rx.recv().await {
                        if let Some((run, generation)) = &instance {
                            match &ev {
                                protocol::Event::RunPhase {
                                    run_id,
                                    generation: gen,
                                    ..
                                } if run_id == run && gen == generation => {
                                    sessioned = true;
                                }
                                protocol::Event::RunTerminated {
                                    run_id,
                                    generation: gen,
                                    ..
                                } if run_id == run && gen == generation => {
                                    terminated = true;
                                }
                                protocol::Event::Error { detail, .. } if !sessioned => {
                                    let _ = me.handle_worker_event(&ev);
                                    let _ = me.handle_pre_session_refusal(run, *generation, detail);
                                    terminated = true;
                                    break;
                                }
                                // The worker's tuple-rederivation refusal is a PRE-SESSION join
                                // refusal like any other typed error — without this arm it was
                                // translated to a Warning only, leaving the `Starting` row and
                                // its live instance entry in place, which the reconciler then
                                // skips forever (observed live: the M4 hard-kill drill wedged in
                                // `Starting` after macOS's working-set figure drifted). The
                                // synthesized retryable terminal releases the reservation and
                                // schedules the retry, whose reconvergence re-assesses and
                                // adopts the drifted tuple.
                                protocol::Event::AdmittedTupleMismatch {
                                    run_id,
                                    field,
                                    generation: gen,
                                } if run_id == run && gen == generation && !sessioned => {
                                    let _ = me.handle_worker_event(&ev);
                                    let _ = me.handle_pre_session_refusal(
                                        run,
                                        *generation,
                                        &format!(
                                            "admitted tuple field `{field}` drifted at join-time \
                                             rederivation"
                                        ),
                                    );
                                    terminated = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        // Best-effort fan-out: a persist error never stalls the pump (mirrors the
                        // existing "a broadcast send error only means no live subscribers" posture;
                        // the durable log + a VhcChanged pointer let a client re-baseline).
                        let _ = me.handle_worker_event(&ev);
                    }
                    if let (Some((run, generation)), false) = (instance, terminated) {
                        let _ = me.handle_stream_closed(&run, generation);
                    }
                });
            }
            None => {
                tokio::spawn(async move { while rx.recv().await.is_some() {} });
            }
        }
    }

    /// Whether vhc training is enabled.
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// The durable store (test/observability access).
    pub fn store(&self) -> &VhcStore {
        &self.store
    }

    /// Start the service: **no-op when disabled** (the worker is never spawned). When enabled,
    /// re-converge every persisted active join-intent through the ONE join transaction
    /// ([`Self::reconverge`]) — durable-intent re-convergence, so a restart rejoins without app
    /// involvement (§10.3). Returns the number of runs whose transaction was re-dispatched.
    ///
    /// A restart is always the REPLACE entry mode: the children died with the node (stdio cut),
    /// so no execution has sequence continuity to resume — each intent re-runs assess on a fresh
    /// child and mints a fresh never-reused incarnation, exactly like a mid-run reconvergence.
    /// (The retained-incarnation restart was the ghost-instance defect: `JoinRun` without a
    /// prior `AssessRun` is a typed worker refusal the old path never observed, while it marked
    /// the row running regardless.) Re-convergence is also the arbiter's restart reconciliation:
    /// the fresh ledger is re-charged exactly by re-admitting each intent (decisions D6 point 7).
    /// A persisted intent that no longer converges (owner budget shrunk, authorship failed) is
    /// surfaced LOUD as a persisted `VhcEvent::Error` and skipped — one refused run never blocks
    /// the rest.
    pub async fn start(&self) -> Result<usize, VhcError> {
        if !self.config.enabled {
            return Ok(0);
        }
        // Crash-window repair FIRST (the startup reconciliation pass): any release whose marker
        // was persisted but whose terminal commit never landed is finished now — every child died
        // with the node, so worker teardown is definitionally observed. Only then is the
        // reconvergence set read (a repaired `completed` never rejoins).
        self.store.repair_pending_releases()?;
        let intents = self.store.active_intents()?;
        let mut rejoined = 0;
        for run in &intents {
            match self.reconverge(run).await {
                Ok(()) => rejoined += 1,
                Err(e) => {
                    // An owner-arbitration refusal keeps its typed class (the budget shrank
                    // under a persisted intent — an owner-visible condition, not a fault).
                    let class = match &e {
                        VhcError::Resources(_) => "owner_arbitration",
                        _ => "reconvergence",
                    };
                    let mut emitted = Vec::new();
                    let _ = self.emit(
                        VhcEvent::Error {
                            run_id: run.run_id.clone(),
                            class: class.to_string(),
                            detail: format!("restart re-convergence failed: {e}"),
                        },
                        &mut emitted,
                    );
                }
            }
        }
        if rejoined > 0 {
            self.emit_changed(None);
        }
        Ok(rejoined)
    }

    /// Spawn the resident coordinator seat keeper loop (requires [`bind_self`](Self::bind_self)):
    /// one keeper pass per lease heartbeat interval — claim-on-boot for uncovered slots, renew for
    /// held leases, fenced-out drops surfaced as events. Returns `None` when coordinator duty is
    /// not fully wired (`[vhc] seat_claim` off, or no seat directory / identity store).
    pub fn spawn_seat_keeper(self: &Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        self.seat.as_ref()?;
        let weak = Arc::downgrade(self);
        let tick =
            std::time::Duration::from_millis(daemon_vhc_proto::DEFAULT_SEAT_HEARTBEAT_MS.max(100));
        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let Some(me) = weak.upgrade() else { return };
                let _ = me.seat_tick().await;
            }
        }))
    }

    /// One seat-keeper pass (the resident loop's body; callable directly for deterministic
    /// tests): cover every joined run whose admitted role is the configured seat role. Fenced
    /// drops and step failures surface as persisted warnings. Returns the notes observed.
    pub async fn seat_tick(&self) -> Result<Vec<crate::seat_keeper::SeatNote>, VhcError> {
        let Some(keeper) = &self.seat else {
            return Ok(Vec::new());
        };
        let candidates = self.seat_candidates()?;
        let notes = keeper.tick(&candidates, now_ms() as u64).await;
        for note in &notes {
            use crate::seat_keeper::SeatNote;
            let (run_id, class, detail) = match note {
                SeatNote::Claimed {
                    run_label,
                    incarnation,
                    leadership_term,
                } => {
                    // A won seat is a state change the app should see (not a warning).
                    self.emit_changed(Some(run_label.clone()));
                    let _ = (run_label, incarnation, leadership_term);
                    continue;
                }
                SeatNote::Renewed { .. } => continue,
                SeatNote::Fenced { run_label, detail } => (
                    run_label.clone(),
                    "seat_fenced",
                    format!("coordinator seat moved (this claimant is fenced): {detail}"),
                ),
                SeatNote::Error { run_label, detail } => {
                    (run_label.clone(), "seat_keeper", detail.clone())
                }
            };
            let mut emitted = Vec::new();
            let _ = self.emit(
                VhcEvent::Warning {
                    run_id,
                    class: class.to_string(),
                    detail,
                },
                &mut emitted,
            );
        }
        Ok(notes)
    }

    /// The seat keeper's coverage: every non-terminal joined intent whose admitted role matches
    /// the configured seat role, resolved to its lease scope (identity from the persisted
    /// admitted tuple; the endpoint peers dial is the run's coordinator endpoint).
    fn seat_candidates(&self) -> Result<Vec<crate::seat_keeper::SeatCandidate>, VhcError> {
        let seat_role = &self.config.seat_role;
        let mut out = Vec::new();
        for run in self.store.active_intents()? {
            if &run.role != seat_role {
                continue;
            }
            let Some(tuple) = run
                .admitted_tuple
                .as_deref()
                .and_then(|b| protocol::decode::<protocol::AdmittedTuple>(b).ok())
            else {
                continue; // no assessed identity to scope a lease under
            };
            out.push(crate::seat_keeper::SeatCandidate {
                run_label: run.run_id.clone(),
                genesis_hash: tuple.genesis_hash,
                role: run.role.clone(),
                epoch: run.epoch,
                module_hash: tuple.module_hash,
                endpoint: daemon_vhc_proto::ControlEndpoint {
                    ws: (!run.coordinator.is_empty()).then(|| run.coordinator.clone()),
                    iroh_ticket: None,
                },
            });
        }
        Ok(out)
    }

    /// Release the held coordinator seat for `run_id` (owner pause/leave): the fenced release, so
    /// a successor takes over at floor + 1 without waiting out the TTL. Best-effort — a failed
    /// release surfaces as a warning (the lease TTL remains the safety net).
    async fn release_seat_for(&self, run_id: &str) {
        let Some(keeper) = &self.seat else { return };
        if let Err(e) = keeper.release_run(run_id).await {
            let mut emitted = Vec::new();
            let _ = self.emit(
                VhcEvent::Warning {
                    run_id: run_id.to_string(),
                    class: "seat_release".to_string(),
                    detail: format!("seat release failed (the lease TTL will expire it): {e}"),
                },
                &mut emitted,
            );
        }
    }

    /// Release every held coordinator seat (node shutdown) — the fenced release path.
    pub async fn release_seats(&self) {
        if let Some(keeper) = &self.seat {
            let _ = keeper.release_all().await;
        }
    }

    /// Spawn the resident reconciliation tick (requires [`bind_self`](Self::bind_self)): the
    /// periodic pass that repairs crash windows, applies the uptime-based retry reset, and fires
    /// due reconvergences. The task ends when the service drops (the `Weak` no longer upgrades).
    pub fn spawn_reconciler(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(self);
        let tick = std::time::Duration::from_millis(self.config.retry.reconcile_tick_ms.max(100));
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tick);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let Some(me) = weak.upgrade() else { return };
                let _ = me.reconcile_tick().await;
            }
        })
    }

    /// One reconciliation pass (the resident tick's body; callable directly for deterministic
    /// tests): finish any half-committed release, reset the retry budget of instances that have
    /// been stably running past the minimum uptime, and reconverge every recoverable intent whose
    /// backoff has elapsed. Returns the number of reconverged runs.
    pub async fn reconcile_tick(&self) -> Result<usize, VhcError> {
        if !self.config.enabled {
            return Ok(0);
        }
        let retry = &self.config.retry;
        // Crash-window safety net: a release begun by a path that died mid-transition commits.
        self.store.repair_pending_releases()?;
        // The uptime reset: a stably-running instance has genuinely recovered.
        self.store
            .reset_recovered_retries(now_ms(), retry.min_uptime_ms as i64)?;
        let due = self.store.runs_awaiting_retry(now_ms())?;
        let mut reconverged = 0;
        for run in due {
            // A live instance for this run means a competing path already reconverged it.
            if self.instances.lock().unwrap().contains_key(&run.run_id) {
                continue;
            }
            // The interim storage gate: a run whose last terminal was HostStorageExhausted is
            // redispatched only once the node-state filesystem clears the reserve floor. A held
            // gate defers WITHOUT consuming budget (the disk being full is not a failed attempt)
            // and never blocks the rest of the pass. The Phase 6 custodian replaces this check
            // as the resume authority.
            if run.storage_gated {
                if self.storage_gate_open() {
                    self.store.set_storage_gated(&run.run_id, false)?;
                } else {
                    let due_at = now_ms() + retry.max_backoff_ms as i64;
                    let _ = self.store.defer_retry(&run.run_id, due_at);
                    let mut emitted = Vec::new();
                    let _ = self.emit(
                        VhcEvent::Warning {
                            run_id: run.run_id.clone(),
                            class: "storage_gate".to_string(),
                            detail: format!(
                                "redispatch held: free space below the {} MiB reserve floor \
                                 (reclaim disk space to resume)",
                                self.config.storage.reserve_mb
                            ),
                        },
                        &mut emitted,
                    );
                    continue;
                }
            }
            match self.reconverge(&run).await {
                Ok(()) => reconverged += 1,
                Err(e) => {
                    // A failed reconvergence attempt consumes budget like any recoverable
                    // failure: escalate on exhaustion, else reschedule with backoff. Loud
                    // either way — one refused run never blocks the rest of the pass.
                    let consumed = run.retry_count;
                    if consumed >= retry.max_retries {
                        self.store.begin_release(
                            &run.run_id,
                            RunState::FailedTerminal,
                            Some(&format!(
                                "retry budget exhausted ({consumed} of {} attempts \
                                 consumed): reconvergence failed: {e}",
                                retry.max_retries
                            )),
                        )?;
                        self.store.commit_release(&run.run_id)?;
                    } else {
                        let due_at = now_ms() + retry_backoff_ms(retry, consumed) as i64;
                        let _ = self.store.bump_retry(&run.run_id, due_at);
                    }
                    let mut emitted = Vec::new();
                    let _ = self.emit(
                        VhcEvent::Error {
                            run_id: run.run_id.clone(),
                            class: "reconvergence".to_string(),
                            detail: format!("reconvergence attempt failed: {e}"),
                        },
                        &mut emitted,
                    );
                    self.emit_changed(Some(run.run_id.clone()));
                }
            }
        }
        Ok(reconverged)
    }

    /// The interim storage gate's free-space check: open when the node-state filesystem (probed
    /// at the identity dir — the durable state root) clears the configured reserve floor.
    /// `reserve_mb = 0` disables the gate; a node without an identity dir has no probeable state
    /// root, so the gate cannot hold. The probe's 0-on-failure maps to CLOSED — a state root
    /// that cannot answer a space query is not one to redispatch onto.
    fn storage_gate_open(&self) -> bool {
        let reserve_mb = self.config.storage.reserve_mb;
        if reserve_mb == 0 {
            return true;
        }
        let Some(dir) = &self.identity_dir else {
            return true;
        };
        daemon_vhc_session::host_disk_free_mb(dir) >= reserve_mb
    }

    /// The ONE join transaction for every non-owner-initiated entry (restart, retry, resume
    /// recovery): `assess (fresh child) → reserve → author identity/planes → JoinRun →
    /// supervised Starting → observed readiness promotes Running`. Reconvergence is always the
    /// REPLACE entry mode — it never retains the failed incarnation: the predecessor may still
    /// be surrendering devices, and a fresh never-reused incarnation guarantees the generation
    /// strictly advances (its stale events stay gated) and its ledger key cannot collide.
    /// Credentials and the per-run certificate are re-authored fresh for the new incarnation
    /// (D-P8 — never replayed). The row is marked `starting`, never `running` — readiness is an
    /// observation the event pump makes from the session's own `RunPhase "running"`.
    ///
    /// A verified own-roster-floor repair ([`VhcError::OwnFloorRepaired`], judged inside the
    /// authorship step) restarts the whole transaction ONCE: the raised counter makes the next
    /// mint strictly above the observed floor, so the restarted attempt re-reserves, re-authors,
    /// and re-publishes under a fresh superseding identity. A second stale refusal surfaces
    /// typed — never a loop.
    async fn reconverge(&self, run: &PersistedRun) -> Result<(), VhcError> {
        match self.reconverge_attempt(run).await {
            Err(VhcError::OwnFloorRepaired { floor }) => {
                tracing::info!(
                    run_id = run.run_id,
                    floor,
                    "reconverge: restarting the join transaction above the repaired floor"
                );
                self.reconverge_attempt(run).await
            }
            other => other,
        }
    }

    /// One attempt of the [`reconverge`](Self::reconverge) transaction body.
    async fn reconverge_attempt(&self, run: &PersistedRun) -> Result<(), VhcError> {
        let worker = self.instance_worker();
        let id = RoleInstanceId {
            run_id: run
                .run_id_hash
                .unwrap_or_else(|| *blake3::hash(run.run_id.as_bytes()).as_bytes()),
            epoch: run.epoch,
            role: if run.role.is_empty() {
                "trainer".to_string()
            } else {
                run.role.clone()
            },
            instance: self.store.mint_incarnation()?,
        };
        let charge = self.derive_charge(&run.eligibility, &run.policy, &id.role)?;
        let priority = self.store.run_priority(&run.run_id)?;
        if let Err(refusal) = self.admit_placed(&id, charge, priority) {
            if self.worker_factory.is_some() {
                worker.shutdown().await;
            }
            return Err(VhcError::Resources(refusal));
        }
        self.store
            .set_execution_identity(&run.run_id, id.epoch, &id.role, id.instance)?;
        // The fresh child holds NO resolved run: a JoinRun without a prior AssessRun is a typed
        // worker refusal ("JoinRun before AssessRun"). Re-run the assess on THIS child with the
        // run's envelope and the persisted directed role — exactly what the original join did —
        // so the child resolves the genesis/module before the join lands.
        let mut fresh_tuple: Option<protocol::AdmittedTuple> = None;
        if let Some(discovery) = &self.discovery {
            let assess = async {
                let envelope = discovery.fetch_envelope(&run.run_id).await?;
                let elig = worker.assess(envelope, Some(id.role.clone())).await?;
                if !elig.eligible {
                    return Err(VhcError::Internal(format!(
                        "reconverge re-assess ineligible: {}",
                        elig.reasons.join("; ")
                    )));
                }
                Ok(elig.admitted_tuple)
            };
            match assess.await {
                Ok(t) => fresh_tuple = t,
                Err(e) => {
                    tracing::warn!(run_id = run.run_id, error = %e, "reconverge: re-assess on the fresh child failed");
                    self.arbiter.release(&id);
                    if self.worker_factory.is_some() {
                        worker.shutdown().await;
                    }
                    return Err(e);
                }
            }
        }
        // The tuple the join delivers: the persisted one while it still describes this box, the
        // fresh re-assessment when any join-recomputed field drifted. Reconvergence is the
        // REPLACE entry mode — a fresh incarnation whose certificate is re-authored either way —
        // so adopting the fresh measurement is re-admission, not identity mutation. Without the
        // adoption a drifted field wedges the intent permanently: the worker rederives at
        // `JoinRun` and refuses the stale tuple on every retry (observed live on macOS, whose
        // capability report digests `recommendedMaxWorkingSetSize` — a figure that moves with
        // ambient memory state, e.g. after the killed predecessor's memory was reclaimed).
        let persisted_tuple = run
            .admitted_tuple
            .as_deref()
            .and_then(|b| protocol::decode::<protocol::AdmittedTuple>(b).ok());
        let persisted_tuple = match (persisted_tuple, fresh_tuple) {
            (Some(p), Some(mut f)) => {
                // Equalize the incarnation before comparing — the fresh tuple predates this
                // attempt's minting (the worker's own rederivation does the same through
                // `TupleIdentity`), and `author_join` re-stamps it either way.
                f.incarnation = p.incarnation;
                match p.first_artifact_mismatch(&f) {
                    Some(field) => {
                        tracing::info!(
                            run_id = run.run_id,
                            field,
                            "reconverge: admitted tuple drifted since the last admission; \
                             adopting the fresh assessment (REPLACE-mode re-admission)"
                        );
                        Some(f)
                    }
                    None => Some(p),
                }
            }
            // No persisted tuple (a pre-admission row): the fresh assessment IS the admission.
            (None, f @ Some(_)) => f,
            // No discovery/no fresh tuple: the persisted one is all there is (offline path).
            (p, None) => p,
        };
        let restore = match self.resolve_restore(&run.run_id, &run.role).await {
            Ok(r) => r,
            Err(e) => {
                self.arbiter.release(&id);
                if self.worker_factory.is_some() {
                    worker.shutdown().await;
                }
                return Err(e);
            }
        };
        let seat = self.resolve_seat_bootstrap(&run.run_id).await;
        let (delivery_tuple, credentials, _credentials_ref) = match self
            .author_join(
                &run.run_id,
                &run.coordinator,
                &id,
                persisted_tuple,
                restore,
                seat,
            )
            .await
        {
            Ok(v) => v,
            Err(e) => {
                self.arbiter.release(&id);
                if self.worker_factory.is_some() {
                    worker.shutdown().await;
                }
                return Err(e);
            }
        };
        if let Some(tuple) = &delivery_tuple {
            if let Ok(bytes) = protocol::encode(tuple) {
                let _ = self.store.set_admitted_tuple(&run.run_id, &bytes);
            }
        }
        let rx = match worker
            .join_streaming(
                run.run_id.clone(),
                run.coordinator.clone(),
                credentials,
                to_join_policy(&run.policy),
                delivery_tuple,
            )
            .await
        {
            Ok(rx) => rx,
            Err(e) => {
                // No child came up — surrender the fresh reservation.
                self.arbiter.release(&id);
                if self.worker_factory.is_some() {
                    worker.shutdown().await;
                }
                return Err(e);
            }
        };
        let generation = id.instance;
        self.instances.lock().unwrap().insert(
            run.run_id.clone(),
            InstanceEntry {
                generation: id.instance,
                id,
                worker,
            },
        );
        self.spawn_pump(Some((run.run_id.clone(), generation)), rx);
        // Supervised Starting — never Running: the readiness promotion is event-driven
        // (`handle_worker_event`'s `RunPhase "running"` arm), so a worker that refuses the join
        // pre-session (typed error, no phase) leaves no false-running ghost and keeps its retry
        // schedule.
        self.store.mark_starting(&run.run_id)?;
        // A seat-holding trainer+coordinator intent re-converges its co-located trainer sibling
        // too (defect D): a reconverged seat without its trainer never meets its own membership
        // floor. Best-effort, same as the original join.
        if self.config.coordinator_trains
            && !run.role.is_empty()
            && run.role == self.config.seat_role
        {
            self.spawn_co_located_trainer(&run.run_id, &run.policy)
                .await;
        }
        self.emit_changed(Some(run.run_id.clone()));
        Ok(())
    }

    /// The worker child for a new role-instance: a fresh factory child when configured (one
    /// sandbox = one role-instance, decisions D1), else the shared default worker.
    fn instance_worker(&self) -> Arc<dyn WorkerControl> {
        match &self.worker_factory {
            Some(f) => f(),
            None => self.worker.clone(),
        }
    }

    /// Bring up the CO-LOCATED trainer role-instance for a seat-holding node (defect D — the
    /// single-peer / trainer+coordinator fix). It runs in a SEPARATE worker child (one sandbox =
    /// one role-instance) that assesses + joins the run's **trainer** role and attaches to the
    /// coordinator THIS node just seated, exactly like a remote trainer — so a self-coordinated run
    /// meets its own membership floor instead of parking at `peers=0` below `min_peers`.
    ///
    /// Requires the per-role-instance worker factory (the coordinator seat instance already holds
    /// the shared/primary worker; a second concurrent role-instance needs its own child). It is
    /// best-effort: any failure is surfaced as a `Warning` event and warned, but never fails the
    /// (already-succeeded) coordinator join — the absence of round progress is the operator signal.
    async fn spawn_co_located_trainer(&self, run_id: &str, policy: &VhcPolicy) {
        let mut emitted = Vec::new();
        macro_rules! warn_co {
            ($detail:expr) => {{
                let _ = self.emit(
                    VhcEvent::Warning {
                        run_id: run_id.to_string(),
                        class: "co_trainer".to_string(),
                        detail: $detail,
                    },
                    &mut emitted,
                );
            }};
        }
        if self.worker_factory.is_none() {
            warn_co!(
                "coordinator_trains is set but no per-role-instance worker factory is configured; \
                 the co-located trainer cannot start"
                    .to_string()
            );
            return;
        }
        // Idempotent: a repeated seat-join re-converges on the existing co-trainer.
        if self.co_trainers.lock().unwrap().contains_key(run_id) {
            return;
        }
        let worker = self.instance_worker();
        // Assess + resolve the coordinator endpoint for the TRAINER role (node-directed).
        let (coordinator, eligibility, assessed_tuple) = match self
            .resolve_join(&worker, run_id, Some("trainer".to_string()))
            .await
        {
            Ok(v) => v,
            Err(e) => {
                warn_co!(format!("co-located trainer assess failed: {e}"));
                worker.shutdown().await;
                return;
            }
        };
        if !eligibility.eligible {
            warn_co!(format!(
                "co-located trainer ineligible: {}",
                eligibility.reasons.join("; ")
            ));
            worker.shutdown().await;
            return;
        }
        let run_hash = self
            .instances
            .lock()
            .unwrap()
            .get(run_id)
            .map(|e| e.id.run_id)
            .unwrap_or_else(|| *blake3::hash(run_id.as_bytes()).as_bytes());
        let instance = match self.store.mint_incarnation() {
            Ok(i) => i,
            Err(e) => {
                warn_co!(format!("co-located trainer incarnation mint failed: {e}"));
                worker.shutdown().await;
                return;
            }
        };
        let id = RoleInstanceId {
            run_id: run_hash,
            epoch: 0,
            role: "trainer".to_string(),
            instance,
        };
        let charge = match self.derive_charge(&eligibility, policy, &id.role) {
            Ok(charge) => charge,
            Err(e) => {
                // Nothing to reserve means nothing to admit. Surrender the worker rather than
                // charging the owner's ceiling for a figure that was supposed to be derived.
                warn_co!(format!(
                    "co-located trainer has no derivable reservation: {e}"
                ));
                worker.shutdown().await;
                return;
            }
        };
        let priority = self.store.run_priority(run_id).unwrap_or(0);
        if let Err(refusal) = self.admit_placed(&id, charge, priority) {
            warn_co!(format!(
                "co-located trainer refused by owner arbitration: {refusal}"
            ));
            worker.shutdown().await;
            return;
        }
        let restore = match self.resolve_restore(run_id, "trainer").await {
            Ok(r) => r,
            Err(e) => {
                self.arbiter.release(&id);
                warn_co!(format!("co-located trainer restore refused: {e}"));
                worker.shutdown().await;
                return;
            }
        };
        let seat = self.resolve_seat_bootstrap(run_id).await;
        let (delivery_tuple, credentials, _credentials_ref) = match self
            .author_join(run_id, &coordinator, &id, assessed_tuple, restore, seat)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                self.arbiter.release(&id);
                warn_co!(format!(
                    "co-located trainer credential authorship failed: {e}"
                ));
                worker.shutdown().await;
                return;
            }
        };
        let rx = match worker
            .join_streaming(
                run_id.to_string(),
                coordinator,
                credentials,
                to_join_policy(policy),
                delivery_tuple,
            )
            .await
        {
            Ok(rx) => rx,
            Err(e) => {
                self.arbiter.release(&id);
                warn_co!(format!("co-located trainer join failed: {e}"));
                worker.shutdown().await;
                return;
            }
        };
        let generation = id.instance;
        self.co_trainers.lock().unwrap().insert(
            run_id.to_string(),
            InstanceEntry {
                generation,
                id,
                worker,
            },
        );
        self.spawn_pump(Some((run_id.to_string(), generation)), rx);
        self.emit_changed(Some(run_id.to_string()));
    }

    /// Derive a role-instance's ledger charge (decisions D6 point 3) from the node-computed
    /// eligibility. Both live eligibility sources reach this through **one** claim-shaped input
    /// contract (decisions D-10):
    ///
    /// - **v2 assess** (`eligibility_from_assess`, the discovery path): the claim funnel's verdict,
    ///   whose headroom carries `claim_device_bytes` / `admitted_host_bytes` (bytes) — the disjoint
    ///   tier sums the worker's `admit` computed. Charged verbatim onto the device + host tiers,
    ///   so an admitted instance's reservation equals the assess claim totals.
    /// - **probe fallback** (`eligibility_from_hardware`, the no-registry default path): headroom
    ///   carries `claim_device_bytes` from the probed dedicated VRAM (bytes) — a conservative
    ///   device-tier charge. This read is load-bearing: dropping it would collapse the default-path
    ///   device charge to the (zero) cap default and admit everything (the audit hazard), so the
    ///   fallback NEVER falls through to a zero charge while a device is probed.
    ///
    /// `policy.vram_cap_mb` (the owner's standing VRAM cap, MiB) stands as the device estimate only
    /// when no device claim is present (the tightening overlay, never an inflation); `duty_cycle_pct`
    /// is the duty charge. Net/disk tiers are left to the assess claim (not separately reported
    /// here yet). The device id is placed by [`Self::admit_placed`] (first-fit) — a placeholder here.
    ///
    /// **Duty is the ACCELERATOR's duty-cycle ledger** (`OwnerBudget.duty_pct`, 100 = one full
    /// accelerator-duty). A **coordinator/consensus** role-instance (`role == seat_role`) runs
    /// ONLY the consensus wasm — it performs no training compute on the accelerator — so it claims
    /// **zero** accelerator duty by design (decisions D6; the seat role's lane is host-side
    /// consensus, not the training lane). This is what lets a single-peer trainer+coordinator box
    /// admit BOTH role-instances under the default 100% duty ledger: the coordinator seat instance
    /// claims 0% and its co-located trainer claims the policy duty, so the seat no longer starves
    /// its own trainer (the M4 fleet-smoke duty-arbitration defect — a full-duty coordinator
    /// exhausted the ledger and refused the co-located trainer). A non-seat (trainer) role keeps
    /// the policy duty. The device/host tiers are the assess claim verbatim regardless of role —
    /// the consensus module's own (near-zero) device footprint stands, never a silent zero
    /// fall-through (D-10).
    fn derive_charge(
        &self,
        eligibility: &VhcEligibility,
        policy: &VhcPolicy,
        role: &str,
    ) -> Result<InstanceCharge, VhcError> {
        // **Present-and-zero is a figure; absent is not.** A role that performs no accelerator
        // computation — the consensus seat — has a genuinely derived device figure of zero, and
        // charging it anything is wrong. The previous lookup discarded zero as though it were
        // missing, which is exactly what sent a zero-footprint role down the owner-cap fallback and
        // had it reserve the whole standing budget. Only a key that is not there at all means "no
        // figure was derived".
        let figure = |key: &str| {
            eligibility
                .headroom
                .get(key)
                .copied()
                .map(|v| u64::try_from(v).unwrap_or(0))
        };

        // The MEMORY reservation derives from the composed estimate, and from nothing else.
        //
        // The certification path reports the reservation the governor derived: already
        // scope-correct, with process- and device-scoped terms charged once at their scope rather
        // than once per role instance, and with the per-allocation constraint excluded because it is
        // a maximum to validate and not occupancy to hold. A lower-minor module has no composed
        // claim and reports its legacy declared tiers instead; that path is unchanged.
        //
        // **The owner-cap fallback is gone.** It used to substitute the owner's own VRAM cap when no
        // device figure was present. With the guest's device tiers retired that fallback becomes the
        // ONLY path for a certification-minor module, and the ledger would charge the owner's
        // ceiling instead of the workload — reserving the whole budget for every role, refusing the
        // second one, and calling it a resource decision. An absent figure is now a typed refusal.
        let reserved_device = figure(RESERVATION_DEVICE_BYTES_KEY);
        let reserved_host = figure(RESERVATION_HOST_BYTES_KEY);
        let (device_bytes, host_bytes) = match (reserved_device, reserved_host) {
            (Some(device), host) => (device, host.unwrap_or(0)),
            (None, _) => {
                let legacy_device = figure("claim_device_bytes").ok_or_else(|| {
                    VhcError::Internal(format!(
                        "{}: no composed reservation and no declared device claim for role \
                         `{role}`, so there is nothing to reserve. The owner's cap is not a \
                         substitute for a figure that was supposed to be derived",
                        AbiRefusalCode::EstimateNotComposable.slug()
                    ))
                })?;
                (legacy_device, figure("admitted_host_bytes").unwrap_or(0))
            }
        };

        // The NON-memory ledgers keep their established sources: duty from the run's join policy
        // under the owner's standing budget, and zero for a role that performs no accelerator
        // computation so it cannot starve a co-resident sibling that does. They are deliberately not
        // swept into claim derivation — but they share the prohibition above: an absent input is a
        // typed refusal or an explicit policy default, never the owner's own ceiling.
        let duty_pct = if role == self.config.seat_role {
            0
        } else {
            policy.duty_cycle_pct.min(100) as u8
        };
        Ok(InstanceCharge {
            device: String::new(),
            // The reservation is carried in ONE tier on purpose. `device_total()` sums all three,
            // which is correct for a per-role term and a double-count for any shared-scope term
            // across co-resident roles; the reservation has already composed every term at its
            // declared scope, so summing it again here would undo that.
            tiers: ClaimTiers {
                hard_accountable: TierBytes {
                    device: device_bytes,
                    host: host_bytes,
                },
                ..ClaimTiers::default()
            },
            disk_bytes: 0,
            net_up_bps: 0,
            net_down_bps: 0,
            duty_pct,
        })
    }

    /// Node-side placement (decisions D6 point 2 — one accelerator per role-instance): try the
    /// owner's device ledgers first-fit in deterministic (id) order; the reservation is committed
    /// atomically on the first device that admits. Returns the FIRST device's refusal when none
    /// fits (deterministic, names a concrete ledger).
    fn admit_placed(
        &self,
        id: &RoleInstanceId,
        charge: InstanceCharge,
        priority: u8,
    ) -> Result<(), AdmitRefusal> {
        let devices = self.arbiter.devices();
        let mut first_refusal = None;
        for device in devices {
            let mut placed = charge.clone();
            placed.device = device;
            match self.arbiter.admit(id.clone(), placed, priority) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if first_refusal.is_none() {
                        first_refusal = Some(e);
                    }
                }
            }
        }
        Err(first_refusal.unwrap_or(AdmitRefusal::MaxInstances { max: 0 }))
    }

    /// Translate + persist + fan out a worker event (spec §10.3 "all are persisted / fanned out by
    /// the node"). Returns the [`VhcEvent`]s emitted (0..2 per worker event). The boot wiring routes the live
    /// worker event stream into this; unit tests drive it directly.
    pub fn handle_worker_event(&self, ev: &protocol::Event) -> Result<Vec<VhcEvent>, VhcError> {
        // Stale-generation discard (the idempotency/generation invariant): every run-scoped
        // pump/session event carries the emitting instance's generation (== incarnation); an
        // event stamped with a generation other than the run's CURRENT one is from a reaped
        // predecessor and is dropped whole — it can never fold contribution, transition state,
        // release a ledger, or touch key custody for the replacement. Generation 0 = un-stamped
        // (pre-counter frames / request-reply events) passes through.
        if let Some(generation) = event_generation(ev) {
            if generation != 0 {
                if let Some(run_id) = self.event_run_id(ev) {
                    // Accept an event whose generation matches the run's CURRENT primary instance
                    // OR its co-located trainer sibling (defect D — a seat-holding
                    // trainer+coordinator node runs BOTH, each with its own incarnation; the
                    // trainer's per-round digest events carry the trainer generation and must not
                    // be dropped by the coordinator-generation guard). An event passes ungated only
                    // when NEITHER a primary nor a co-trainer generation is known for the run.
                    let expected = self.expected_generation(&run_id)?;
                    let co_gen = self
                        .co_trainers
                        .lock()
                        .unwrap()
                        .get(&run_id)
                        .map(|e| e.generation);
                    let known = expected.is_some() || co_gen.is_some();
                    if known && expected != Some(generation) && co_gen != Some(generation) {
                        return Ok(Vec::new());
                    }
                }
            }
        }
        // The terminal transition (the run-instance state machine's worker-driven edges).
        if let protocol::Event::RunTerminated {
            run_id,
            generation,
            outcome,
        } = ev
        {
            return self.handle_run_terminated(run_id, *generation, outcome);
        }
        // Track the current run + persist phase from a RunPhase.
        if let protocol::Event::RunPhase {
            run_id,
            phase,
            round,
            generation,
            ..
        } = ev
        {
            *self.current_run.lock().unwrap() = Some(run_id.clone());
            self.store.set_phase(run_id, phase, *round)?;
            // The readiness promotion (Starting → Running): the session's OWN "running" phase is
            // the observation — a dispatched `JoinRun` proves nothing (the ghost-instance
            // defect). Only the PRIMARY instance's readiness promotes the row: the co-located
            // trainer sibling shares the run id but not the row's lifecycle.
            if phase == "running" {
                let primary = {
                    let instances = self.instances.lock().unwrap();
                    instances.get(run_id).map(|e| e.generation)
                };
                if primary == Some(*generation) {
                    self.store.mark_running(run_id)?;
                    self.emit_changed(Some(run_id.clone()));
                }
            }
        }
        let run_id = self.event_run_id(ev);
        let Some(run_id) = run_id else {
            return Ok(Vec::new()); // Unattributable (e.g. a Probed before any RunPhase).
        };

        // Fold contribution counters from the raw event.
        match ev {
            protocol::Event::RoundProgress {
                up_bytes,
                down_bytes,
                ..
            } => self
                .store
                .bump_contribution(&run_id, 0, 0, *up_bytes, *down_bytes, 0, 0)?,
            protocol::Event::RoundOutcome { stalled, .. } => {
                self.store
                    .bump_contribution(&run_id, u64::from(!*stalled), 0, 0, 0, 0, 0)?
            }
            protocol::Event::CheckpointPublished { .. } => {
                self.store.bump_contribution(&run_id, 0, 0, 0, 0, 0, 1)?
            }
            _ => {}
        }

        // Checkpoint-pointer publication (spec §9): the checkpoint DOCUMENT is already on
        // the payload plane (the session put it there); record the round → content-address
        // pointer at the registry under the run's `(role, kind)` slot so a late joiner restores
        // role-scoped (a coordinator pointer never shadows a trainer restore source). Best-effort
        // + detached (a pointer is advisory; the joiner hash-verifies regardless, so an unknown
        // size is 0).
        if let protocol::Event::CheckpointPublished {
            round,
            hash,
            kind,
            generation,
            ..
        } = ev
        {
            if let Some(discovery) = &self.discovery {
                // The pointer's role is the EMITTING INSTANCE's role, not the run row's. On a
                // seat-holding node the run row reads "coordinator" while the co-located trainer
                // sibling publishes the (much larger) trainer checkpoints; attributing those to
                // the coordinator slot poisons every coordinator restore — observed live as
                // `MigrateIncompatible` on every coordinator rejoin (the quorum guest correctly
                // refusing a TinyLlama trainer document). The event's generation names the
                // emitter: the co-trainer's generation ⇒ "trainer", else the run row's role.
                let co_gen = self
                    .co_trainers
                    .lock()
                    .unwrap()
                    .get(&run_id)
                    .map(|e| e.generation);
                let role = if co_gen == Some(*generation) {
                    "trainer".to_string()
                } else {
                    self.store
                        .get_run(&run_id)?
                        .map(|r| r.role)
                        .filter(|r| !r.is_empty())
                        .unwrap_or_else(|| "trainer".to_string())
                };
                let discovery = discovery.clone();
                let (run, round, hash, kind) = (run_id.clone(), *round, hash.clone(), kind.clone());
                tokio::spawn(async move {
                    match discovery
                        .publish_checkpoint(&run, &role, &kind, round, &hash, 0)
                        .await
                    {
                        Ok(()) => {
                            tracing::debug!(
                                run,
                                role,
                                kind,
                                round,
                                hash,
                                "checkpoint pointer published"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(run, role, kind, round, error = %e, "checkpoint pointer publication failed");
                        }
                    }
                });
            } else {
                tracing::debug!(
                    run_id,
                    "checkpoint published but no discovery is wired (pointer not recorded)"
                );
            }
        }

        let mut emitted = Vec::new();
        if let Some(sev) = translate(ev, &run_id) {
            self.emit(sev, &mut emitted)?;
        }
        // A checkpoint is a contribution delta — surface the fresh totals as a Contribution event.
        if matches!(ev, protocol::Event::CheckpointPublished { .. }) {
            let contribution = self.store.get_contribution(&run_id)?;
            self.emit(
                VhcEvent::Contribution {
                    run_id: run_id.clone(),
                    contribution,
                },
                &mut emitted,
            )?;
        }
        if !emitted.is_empty() {
            self.emit_changed(Some(run_id));
        }
        Ok(emitted)
    }

    /// Drive one worker-observed terminal edge of the run-instance state machine, in the
    /// crash-repairable order (teardown-observed-before-ledger-release):
    ///
    /// 1. the durable RELEASE MARKER commits first — teardown is observed, the terminal target
    ///    recorded (`begin_release`); a node crash after this point is finished by the startup
    ///    reconciliation pass, never leaked;
    /// 2. the live instance leaves the map and its ledger reservation releases — a replacement
    ///    can only be admitted after this, never while the predecessor may hold devices;
    /// 3. the terminal state commits (`commit_release`);
    /// 4. retry bookkeeping: a recoverable failure consumes one attempt of the bounded budget
    ///    (backoff-scheduled); exhaustion escalates to `failed_terminal` with a typed reason;
    /// 5. per-run identity custody ([CI-7]): a terminal that ends the run identity deletes its
    ///    key material / certificates / credentials record;
    /// 6. the phase mirror + app event surface the transition.
    ///
    /// IDEMPOTENT: a duplicate terminal for an already-transitioned instance is a no-op — it can
    /// never double-release the ledger or re-transition the row.
    fn handle_run_terminated(
        &self,
        run_id: &str,
        generation: u64,
        outcome: &protocol::TerminalOutcome,
    ) -> Result<Vec<VhcEvent>, VhcError> {
        // Defect D: a terminal stamped with a CO-LOCATED trainer's generation tears down ONLY that
        // sibling role-instance (its ledger reservation) — never the shared run row or the
        // coordinator seat instance, whose state machine is driven by the PRIMARY instance's
        // terminal. Idempotent: `vhc_leave` may have already removed it (then this is a no-op).
        {
            let co = {
                let mut co = self.co_trainers.lock().unwrap();
                match co.get(run_id) {
                    Some(e) if e.generation == generation => co.remove(run_id),
                    _ => None,
                }
            };
            if let Some(e) = co {
                self.arbiter.release(&e.id);
                return Ok(Vec::new());
            }
        }
        let row = self.store.get_run(run_id)?;
        let entry_live = self.instances.lock().unwrap().contains_key(run_id);
        if !entry_live {
            match &row {
                // Duplicate delivery: the instance already transitioned (no live entry, the
                // observed state is settled) — nothing left to release or record. `Starting` is
                // NOT settled: a terminal for an in-flight transaction still transitions it.
                Some(r)
                    if r.run_state != RunState::Running
                        && r.run_state != RunState::Starting
                        && r.pending_run_state.is_none() =>
                {
                    return Ok(Vec::new());
                }
                Some(_) => {}
                // A terminal for a run this node never recorded: nothing to transition.
                None => return Ok(Vec::new()),
            }
        }
        let (target, reason) = match outcome {
            protocol::TerminalOutcome::Completed { outcome } => (
                RunState::Completed,
                format!("module signaled run end (outcome {outcome})"),
            ),
            protocol::TerminalOutcome::Left { checkpoint } => (
                RunState::Left,
                match checkpoint {
                    Some(hash) => format!("left (drain snapshot {hash})"),
                    None => "left".to_string(),
                },
            ),
            protocol::TerminalOutcome::FailedRetryable { reason } => {
                (RunState::FailedRetryable, reason.clone())
            }
            protocol::TerminalOutcome::FailedTerminal { reason } => {
                (RunState::FailedTerminal, reason.clone())
            }
            // The typed storage taxonomy: exhaustion (ENOSPC/quota) is recoverable, but its
            // redispatch is STORAGE-GATED — the reconcile loop holds it behind the free-space
            // check, and the wait neither consumes budget nor escalates (a full disk is a
            // capacity condition, not a crash loop).
            protocol::TerminalOutcome::FailedStorage { reason } => {
                (RunState::FailedRetryable, reason.clone())
            }
        };
        let storage_gated = matches!(outcome, protocol::TerminalOutcome::FailedStorage { .. });
        // The bounded retry budget: a recoverable failure past the budget escalates to terminal
        // with a typed reason; within it, the next reconvergence is backoff-scheduled. A
        // storage-gated failure bypasses the budget entirely — the gate (not the budget) is
        // what bounds it, so it can never launder a crash loop: the moment the disk has
        // headroom the run redispatches and any non-storage failure consumes budget normally.
        let retry = &self.config.retry;
        let consumed = row.as_ref().map_or(0, |r| r.retry_count);
        let (target, reason, next_retry) = if storage_gated {
            let due = now_ms() + retry.max_backoff_ms as i64;
            (RunState::FailedRetryable, reason, Some(due))
        } else if target == RunState::FailedRetryable {
            if consumed >= retry.max_retries {
                (
                    RunState::FailedTerminal,
                    format!(
                        "retry budget exhausted ({consumed} of {} attempts consumed): {reason}",
                        retry.max_retries
                    ),
                    None,
                )
            } else {
                let due = now_ms() + retry_backoff_ms(retry, consumed) as i64;
                (RunState::FailedRetryable, reason, Some(due))
            }
        } else {
            (target, reason, None)
        };

        // 1) The durable marker: teardown observed, terminal in flight (the crash window closes).
        self.store.begin_release(run_id, target, Some(&reason))?;
        // 2) Instance-map removal + ledger release — only AFTER the observation is durable.
        let entry = self.instances.lock().unwrap().remove(run_id);
        if let Some(e) = &entry {
            self.arbiter.release(&e.id);
        }
        // 3) The terminal commits.
        self.store.commit_release(run_id)?;
        // 4) Retry bookkeeping. A storage-gated terminal defers (schedules the next free-space
        // check) without consuming budget, and durably marks the gate.
        if let Some(due) = next_retry {
            if storage_gated {
                let _ = self.store.defer_retry(run_id, due);
                let _ = self.store.set_storage_gated(run_id, true);
            } else {
                let _ = self.store.bump_retry(run_id, due);
            }
        }
        // 5) [CI-7] identity custody: a terminal that ends the run identity (completed / left /
        // failed_terminal) deletes its per-run key material, certificates, and credentials
        // record — no run identity outlives the run it was minted for. `failed_retryable` is NOT
        // terminal for the identity (reconvergence supersedes it under the retry budget), so its
        // material survives. Idempotent (remove_run tolerates absence).
        let mut emitted = Vec::new();
        if target != RunState::FailedRetryable {
            if let Some(dir) = &self.identity_dir {
                if let Ok(keystore) = daemon_vhc_session::keystore::VhcKeystore::open(dir) {
                    if let Err(e) = keystore.remove_run(run_id) {
                        let _ = self.emit(
                            VhcEvent::Warning {
                                run_id: run_id.to_string(),
                                class: "identity_cleanup".to_string(),
                                detail: format!("per-run key/credential cleanup failed: {e}"),
                            },
                            &mut emitted,
                        );
                    }
                }
            }
        }
        // 6) Surface the transition: the phase mirror + one app-facing event.
        let (epoch, round) = row.as_ref().map_or((0, 0), |r| (r.epoch, r.last_round));
        self.store.set_phase(run_id, target.as_str(), round)?;
        self.emit(
            VhcEvent::Phase {
                run_id: run_id.to_string(),
                phase: target.as_str().to_string(),
                epoch,
                round,
            },
            &mut emitted,
        )?;
        let _ = generation; // gated by the caller; carried for symmetry with the wire event
        self.emit_changed(Some(run_id.to_string()));
        Ok(emitted)
    }

    /// The pump-stream-closure observation: the worker's event stream for `(run, generation)`
    /// ended without that instance's terminal event — the child crashed or the transport was
    /// lost. If that instance is still the run's CURRENT one, it transitions `failed_retryable`
    /// (process absence is the teardown observation); a reaped/replaced instance is left alone.
    pub fn handle_stream_closed(&self, run_id: &str, generation: u64) -> Result<(), VhcError> {
        if !self.instance_current(run_id, generation) {
            return Ok(());
        }
        self.handle_run_terminated(
            run_id,
            generation,
            &protocol::TerminalOutcome::FailedRetryable {
                reason: "worker event stream closed without a terminal event".to_string(),
            },
        )?;
        Ok(())
    }

    /// The pre-session refusal observation: the worker child answered the join transaction with
    /// a typed error BEFORE its session ever spoke (no generation-matching `RunPhase`), so the
    /// session's own terminal event will never arrive — the transaction failed. Synthesizes the
    /// retryable terminal for the (still `Starting`) instance: reservation released, entry
    /// removed, retry scheduled — no ghost survives a refused `JoinRun`.
    fn handle_pre_session_refusal(
        &self,
        run_id: &str,
        generation: u64,
        detail: &str,
    ) -> Result<(), VhcError> {
        if !self.instance_current(run_id, generation) {
            return Ok(());
        }
        self.handle_run_terminated(
            run_id,
            generation,
            &protocol::TerminalOutcome::FailedRetryable {
                reason: format!("pre-session worker refusal: {detail}"),
            },
        )?;
        Ok(())
    }

    /// Whether `generation` is still a CURRENT supervised instance for `run_id` — the primary
    /// entry or the co-located trainer sibling (whose teardown `handle_run_terminated` routes to
    /// its own reservation). A reaped/replaced generation is settled: nothing to transition.
    fn instance_current(&self, run_id: &str, generation: u64) -> bool {
        let primary = {
            let instances = self.instances.lock().unwrap();
            instances
                .get(run_id)
                .is_some_and(|e| e.id.instance == generation)
        };
        if primary {
            return true;
        }
        self.co_trainers
            .lock()
            .unwrap()
            .get(run_id)
            .is_some_and(|e| e.generation == generation)
    }

    /// The generation the node currently expects for `run_id`'s events: the live instance's
    /// incarnation when one is supervised, else the persisted incarnation (`None` for an unknown
    /// run or a pre-incarnation row — such events pass ungated).
    fn expected_generation(&self, run_id: &str) -> Result<Option<u64>, VhcError> {
        let live = {
            let instances = self.instances.lock().unwrap();
            // The CURRENT generation (advanced by a live module switch), not the admitted
            // ledger key — post-switch events carry the new incarnation.
            instances.get(run_id).map(|e| e.generation)
        };
        if live.is_some() {
            return Ok(live);
        }
        Ok(self
            .store
            .get_run(run_id)?
            .map(|r| r.instance)
            .filter(|i| *i > 0))
    }

    fn emit(&self, sev: VhcEvent, out: &mut Vec<VhcEvent>) -> Result<(), VhcError> {
        self.store.append_event(&sev)?;
        // A send error only means "no live subscribers"; the durable log already has it.
        let _ = self.events_tx.send(sev.clone());
        out.push(sev);
        Ok(())
    }

    fn emit_changed(&self, run_id: Option<String>) {
        if let Some(feed) = &self.feed {
            let rev = self.rev.fetch_add(1, Ordering::SeqCst) + 1;
            feed(NodeEvent::VhcChanged { run_id, rev });
        }
    }

    fn event_run_id(&self, ev: &protocol::Event) -> Option<String> {
        match ev {
            protocol::Event::RunPhase { run_id, .. }
            | protocol::Event::AdmittedTupleMismatch { run_id, .. }
            | protocol::Event::RunTerminated { run_id, .. } => Some(run_id.clone()),
            protocol::Event::RoundProgress { .. }
            | protocol::Event::RoundOutcome { .. }
            | protocol::Event::Metric { .. }
            | protocol::Event::CheckpointPublished { .. }
            | protocol::Event::Warning { .. }
            | protocol::Event::Error { .. } => self.current_run.lock().unwrap().clone(),
            _ => None,
        }
    }

    fn require_enabled(&self) -> Result<(), VhcError> {
        if self.config.enabled {
            Ok(())
        } else {
            Err(VhcError::Disabled)
        }
    }

    /// The fallback coordinator endpoint (the first allowlisted endpoint, §11.1) used when no
    /// discovery seam is configured (offline / no-registry path).
    fn coordinator(&self) -> String {
        self.config
            .coordinator_allowlist
            .first()
            .cloned()
            .unwrap_or_default()
    }

    /// Resolve the `(coordinator, eligibility)` for a join, against the role-instance's own
    /// `worker` (per-instance children run their own §6.5 assess).
    ///
    /// With a discovery seam: `GET /runs/:id` → fetch + blake3-verify the frozen envelope →
    /// `worker.assess(envelope)` (real §6.5), taking the coordinator from the registry. Without one:
    /// the probe against the allowlisted coordinator. Eligibility is always node-computed.
    async fn resolve_join(
        &self,
        worker: &Arc<dyn WorkerControl>,
        run_id: &str,
        role: Option<String>,
    ) -> Result<(String, VhcEligibility, Option<protocol::AdmittedTuple>), VhcError> {
        if let Some(discovery) = &self.discovery {
            let run = discovery
                .get_run(run_id)
                .await?
                .ok_or_else(|| VhcError::Discovery(format!("run {run_id} not found")))?;
            // The allowlist gate (spec §11.1): a discovered coordinator outside the owner's
            // allowlist is a typed refusal BEFORE any envelope fetch reaches the worker — the
            // registry names an endpoint, the owner authorizes it.
            if !allowlisted(&self.config.coordinator_allowlist, &run.coordinator) {
                return Err(VhcError::AllowlistRefused(run.coordinator.clone()));
            }
            let envelope = discovery.fetch_envelope(run_id).await?;
            let verdict = worker.assess(envelope, role).await?;
            // Stamp the node-owned revisions into the immutable admitted tuple (architecture
            // §6.3); the incarnation stays 0 (unassigned) until the node mints it at join.
            let tuple = self.stamp_admitted_tuple(verdict.admitted_tuple.clone())?;
            Ok((run.coordinator, eligibility_from_assess(&verdict), tuple))
        } else {
            let coordinator = self.coordinator();
            let eligibility = match worker.probe().await {
                Ok(hw) => eligibility_from_hardware(&hw),
                Err(_) => VhcEligibility {
                    eligible: false,
                    reasons: vec!["worker probe failed".into()],
                    headroom: BTreeMap::new(),
                },
            };
            Ok((coordinator, eligibility, None))
        }
    }

    /// The coordinator-duty join attempt ([SEAT-1] end-to-end): assess the configured seat role,
    /// then claim (or re-adopt) the registry seat slot through the resident keeper — the join
    /// runs at the WON LEASE'S incarnation, so the fencing token IS the certified execution
    /// identity's incarnation. Any refusal (keeper unwired, ineligible seat-role assessment, a
    /// live foreign incumbent, a lost CAS race, a directory fault) returns `None`: the caller
    /// stands down to the trainer default — coordinator duty is opportunistic, never a join
    /// failure.
    async fn try_seat_join(
        &self,
        worker: &Arc<dyn WorkerControl>,
        run_id: &str,
    ) -> Option<(
        String,
        VhcEligibility,
        Option<protocol::AdmittedTuple>,
        Option<u64>,
    )> {
        let keeper = self.seat.as_ref()?;
        let seat_role = self.config.seat_role.clone();
        let (coordinator, eligibility, tuple) = self
            .resolve_join(worker, run_id, Some(seat_role.clone()))
            .await
            .ok()?;
        if !eligibility.eligible {
            return None;
        }
        let tuple_ref = tuple.as_ref()?;
        let candidate = crate::seat_keeper::SeatCandidate {
            run_label: run_id.to_string(),
            genesis_hash: tuple_ref.genesis_hash,
            role: seat_role,
            epoch: 0,
            module_hash: tuple_ref.module_hash,
            endpoint: daemon_vhc_proto::ControlEndpoint {
                ws: (!coordinator.is_empty()).then(|| coordinator.clone()),
                iroh_ticket: None,
            },
        };
        match keeper.claim_now(&candidate, now_ms() as u64).await {
            Ok(Some(incarnation)) => Some((coordinator, eligibility, tuple, Some(incarnation))),
            Ok(None) => None, // a live incumbent / lost race — stand down to the trainer default
            Err(e) => {
                let mut emitted = Vec::new();
                let _ = self.emit(
                    VhcEvent::Warning {
                        run_id: run_id.to_string(),
                        class: "seat_claim".to_string(),
                        detail: format!("seat claim failed (joining as trainer): {e}"),
                    },
                    &mut emitted,
                );
                None
            }
        }
    }

    /// Stamp the node-owned device-profile / owner-policy revisions into an assessed admitted
    /// tuple (architecture §6.3). `None` in ⇒ `None` out (an ineligible assessment admitted
    /// nothing). The incarnation field stays as assessed (0 = unassigned); the join mints it.
    fn stamp_admitted_tuple(
        &self,
        tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<Option<protocol::AdmittedTuple>, VhcError> {
        let Some(mut tuple) = tuple else {
            return Ok(None);
        };
        tuple.device_profile_rev = self.store.counter("device_profile_rev")?;
        tuple.owner_policy_rev = self.store.counter("owner_policy_rev")?;
        Ok(Some(tuple))
    }

    /// Finalize the admitted tuple for delivery: stamp the node-minted incarnation, and — when the
    /// node authors identity ([`VhcServiceParts::identity_dir`]) — mint the per-run key, issue its
    /// certificate under the base identity, and author the plane-selection credentials. Returns
    /// the delivery tuple, the wire credentials bytes, and the `credentials_ref` to persist.
    async fn author_join(
        &self,
        run_label: &str,
        coordinator: &str,
        id: &RoleInstanceId,
        tuple: Option<protocol::AdmittedTuple>,
        restore: Option<protocol::CheckpointRestore>,
        seat: crate::credentials::SeatBootstrap,
    ) -> Result<AuthoredDelivery, VhcError> {
        let tuple = tuple.map(|mut t| {
            t.incarnation = id.instance;
            t
        });
        let Some(dir) = &self.identity_dir else {
            // No node-side authorship (tests / headless): no credentials, tuple as stamped.
            return Ok((tuple, Vec::new(), None));
        };
        // Authorship needs the assessed tuple (its genesis + module hashes scope the certificate).
        let Some(tuple) = tuple else {
            return Ok((None, Vec::new(), None));
        };
        let keystore = daemon_vhc_session::keystore::VhcKeystore::open(dir)
            .map_err(|e| VhcError::Internal(format!("open identity keystore: {e}")))?;
        let identity = crate::credentials::RunInstanceIdentity {
            run_label,
            genesis_hash: tuple.genesis_hash,
            epoch: id.epoch,
            role: &id.role,
            incarnation: id.instance,
            module_hash: tuple.module_hash,
        };
        // The iroh plane (opt-in): publish this node's signed roster record, fetch + verify the
        // run's roster, and select the dual plane. Fail-closed when enabled — a roster that
        // cannot be published/verified refuses the join typed (retryable via reconciliation),
        // never a silently-degraded WS-only run.
        let iroh = self.resolve_iroh_plane(&keystore, &identity).await?;
        let mut seat = seat;
        self.extend_roster_certs(run_label, &mut seat.peer_certs)
            .await;
        let authored = crate::credentials::author_join(
            &keystore,
            &identity,
            coordinator,
            &self.config.registry,
            !self.config.payload_dir.is_empty(),
            crate::credentials::JoinBootstrap {
                restore,
                seat,
                iroh,
            },
        )?;
        Ok((Some(tuple), authored.wire, authored.credentials_ref))
    }

    /// Resolve the iroh half of the plane selection (`None` when `[vhc].iroh.enabled` is off —
    /// the WS-only default, byte-identical to pre-roster behavior — or when another role-instance
    /// on this node already owns the node's single iroh endpoint for the run, see
    /// [`Self::claim_node_iroh_endpoint`]):
    ///
    /// 1. pin this run's iroh bind port (once per node lifetime; config `bind_port` or a free
    ///    UDP port) and derive the advertised `ip:port` addresses;
    /// 2. author + publish this node's signed roster record (per-run key + certificate — the
    ///    registry stores it under the structural monotonic upsert);
    /// 3. fetch the run's roster and verify every entry node-side (signature, certificate chain
    ///    to a genesis-trusted base, freshness precedence) — the registry is never trusted;
    /// 4. hand the verified peers + pinned relays + bind address to the credentials body.
    async fn resolve_iroh_plane(
        &self,
        keystore: &daemon_vhc_session::keystore::VhcKeystore,
        identity: &crate::credentials::RunInstanceIdentity<'_>,
    ) -> Result<Option<daemon_vhc_session::protocol::IrohPlane>, VhcError> {
        if !self.config.iroh.enabled {
            return Ok(None);
        }
        let Some(discovery) = &self.discovery else {
            return Err(VhcError::Discovery(
                "iroh plane enabled but no registry discovery seam is wired".into(),
            ));
        };
        let run_label = identity.run_label;
        // The node's iroh endpoint is a NODE-level singleton ([CI-10]): a co-located sibling of
        // its owner attaches WS-only instead of binding a second socket on the same pinned port.
        if !self.claim_node_iroh_endpoint(run_label, identity.role, identity.incarnation) {
            return Ok(None);
        }

        // 1. The pinned bind socket + the advertised addresses (agree by construction).
        let port = self.iroh_bind_port(run_label)?;
        let direct_addrs: Vec<String> = self
            .config
            .iroh
            .advertise_ips
            .iter()
            .map(|ip| format!("{ip}:{port}"))
            .collect();
        let relay_urls = self.config.iroh.relay_urls();
        // Loopback-only advertisement binds loopback (the single-host topology); anything else
        // binds the wildcard so every advertised interface is served. An EMPTY advertise list is
        // the relay-only posture (a NAT'd node dialable through its home relay, no direct
        // addresses published) and MUST bind the wildcard too: `all()` on an empty list is
        // vacuously true, and a loopback-bound socket cannot transmit to any non-loopback peer
        // or relay (`EINVAL`/`ENETUNREACH` on every send) — observed live on the two-box WAN
        // rung as a gossip plane that came "up", then failed every outbound dial for the life
        // of the run.
        let advertised = &self.config.iroh.advertise_ips;
        let all_loopback =
            !advertised.is_empty() && advertised.iter().all(|ip| ip == "127.0.0.1" || ip == "::1");
        let bind_addr = if all_loopback {
            format!("127.0.0.1:{port}")
        } else {
            format!("0.0.0.0:{port}")
        };

        // 2. Author + publish our record (provisioning is idempotent within the incarnation).
        let record = crate::roster::author_roster_record(
            keystore,
            identity,
            direct_addrs,
            relay_urls.first().cloned(),
            u64::try_from(now_ms()).unwrap_or(1).max(1),
        )
        .map_err(|e| VhcError::Internal(format!("author roster record: {e}")))?;
        if let Err(e) = discovery.publish_roster(run_label, &record).await {
            // A stale refusal is judged HERE, where the identity material lives: verified
            // own-base evidence repairs the counter and surfaces `OwnFloorRepaired` (the join
            // transaction restarts once); anything else fails closed typed.
            return Err(match e {
                VhcError::RosterStale {
                    stored_incarnation,
                    stored,
                } => self.repair_own_roster_floor(
                    keystore,
                    identity,
                    stored_incarnation,
                    stored.as_deref(),
                ),
                other => other,
            });
        }

        // 3. Fetch + verify: trust is the genesis-named base set, never the registry.
        let trusted = self.genesis_trusted_bases(run_label).await?;
        let own = crate::roster::local_endpoint_id(keystore)
            .map_err(|e| VhcError::Internal(format!("local endpoint id: {e}")))?;
        let records = discovery.fetch_roster(run_label).await?;
        let roster = crate::roster::verified_iroh_roster(records, trusted.bases(), own);
        tracing::info!(
            run = run_label,
            peers = roster.len(),
            relays = relay_urls.len(),
            bind = %bind_addr,
            "iroh plane resolved from the verified registry roster"
        );

        Ok(Some(daemon_vhc_session::protocol::IrohPlane {
            relay_urls,
            roster,
            bind_addr: Some(bind_addr),
        }))
    }

    /// Judge a roster-stale refusal against the recovery invariant (floors advance only from
    /// verified own-base evidence) and return the error the join transaction acts on:
    ///
    /// - the stored record verifies (signature + certificate chain) **to this node's own base
    ///   identity**, for **this instance's scope** (run hash, role, our endpoint id) ⇒ it is a
    ///   fresher execution of OUR OWN ladder (a predecessor this restart lost track of):
    ///   bounds-checked [`mint_incarnation_above`](crate::store::VhcStore::mint_incarnation_above)
    ///   repairs the counter and [`VhcError::OwnFloorRepaired`] tells the transaction to restart
    ///   authorship from the top, once;
    /// - anything else — a foreign base in our slot, an unverifiable record, a scope mismatch,
    ///   an out-of-domain ordinal, an empty stored slot — **fails closed** as a typed terminal:
    ///   a collision (or a lying registry) is never a floor to adopt. The registry's structural
    ///   decision drove the retry logic only; no naked registry number ever touched a counter.
    fn repair_own_roster_floor(
        &self,
        keystore: &daemon_vhc_session::keystore::VhcKeystore,
        identity: &crate::credentials::RunInstanceIdentity<'_>,
        stored_incarnation: u64,
        stored: Option<&daemon_vhc_proto::RosterRecord>,
    ) -> VhcError {
        let fail_closed = |why: String| {
            VhcError::Discovery(format!(
                "roster publish refused stale (stored incarnation {stored_incarnation}) and the \
                 stored record is not verified own-base evidence — failing closed, no floor \
                 adopted: {why}"
            ))
        };
        let Some(record) = stored else {
            return fail_closed("the registry returned no stored record".into());
        };
        let own_base = match keystore.base_identity() {
            Ok(key) => daemon_vhc_proto::peer_id(&key),
            Err(e) => return fail_closed(format!("no base identity: {e}")),
        };
        // Full verification AGAINST OUR OWN BASE as the only trusted issuer: signature by the
        // record's sender, certificate chain + scope binding, issuer == our base.
        if let Err(e) = record.authorize(std::slice::from_ref(&own_base)) {
            return fail_closed(format!("stored record does not verify to our base: {e}"));
        }
        if record.body.run_id.0 != identity.genesis_hash {
            return fail_closed("stored record is scoped to a different run".into());
        }
        if record.body.role != identity.role {
            return fail_closed(format!(
                "stored record is scoped to role `{}`, ours is `{}`",
                record.body.role, identity.role
            ));
        }
        match crate::roster::local_endpoint_id(keystore) {
            Ok(own_endpoint) if record.body.endpoint_id == own_endpoint => {}
            Ok(_) => return fail_closed("stored record names a different endpoint id".into()),
            Err(e) => return fail_closed(format!("no local endpoint id: {e}")),
        }
        // Verified own-base evidence: bounds-checked counter repair (typed refusal on an
        // out-of-domain floor or exhaustion — never a wrap).
        let floor = record.body.incarnation;
        match self.store.mint_incarnation_above(floor) {
            Ok(minted) => {
                tracing::info!(
                    run = identity.run_label,
                    role = identity.role,
                    floor,
                    minted,
                    "own roster floor repaired: the join transaction restarts above it"
                );
                VhcError::OwnFloorRepaired { floor }
            }
            Err(e) => fail_closed(format!("counter repair refused: {e}")),
        }
    }

    /// The run's genesis-trusted certificate issuers: fetch the frozen envelope (blake3-verified
    /// by the discovery seam), reopen + verify the signed genesis, and read its `[identities]`
    /// section — never ambient config.
    async fn genesis_trusted_bases(
        &self,
        run_label: &str,
    ) -> Result<daemon_vhc_session::identity::TrustedBases, VhcError> {
        let Some(discovery) = &self.discovery else {
            return Err(VhcError::Discovery(
                "no discovery seam to resolve the genesis trust set".into(),
            ));
        };
        let bytes = discovery.fetch_envelope(run_label).await?;
        let wire: daemon_vhc_proto::SignedEnvelope = daemon_vhc_proto::from_canonical_slice(&bytes)
            .map_err(|e| VhcError::Discovery(format!("decode signed envelope: {e}")))?;
        let frozen = daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
            .map_err(|e| VhcError::Discovery(format!("verify genesis envelope: {e}")))?;
        let env = frozen
            .decode()
            .map_err(|e| VhcError::Discovery(format!("decode genesis: {e}")))?;
        Ok(daemon_vhc_session::identity::TrustedBases::from_genesis(
            &env,
        ))
    }

    /// Claim this node's SINGLE iroh endpoint for `run_label` on behalf of the `(role,
    /// incarnation)` role-instance about to author credentials. `true` ⇒ this instance authors +
    /// publishes the node's roster record and binds the pinned socket; `false` ⇒ another LIVE
    /// role-instance on this node already owns it, so this one attaches WS-only.
    ///
    /// **Why ownership is exclusive (architecture [CI-10] / [CI-11]).** The iroh endpoint id is
    /// the NODE's transport identity — one `identity/iroh.key`, read by both the node's roster
    /// authorship and every worker's endpoint (`keystore.iroh_secret()`, §7.2) — and each admitted
    /// node publishes ONE reachability record per run binding that endpoint id to its addresses.
    /// Two co-resident role-instances therefore cannot each bind their own socket: they would be
    /// two live iroh endpoints presenting the SAME endpoint id, so a record naming either socket
    /// is a false statement about the other, roster readers (who fold per node and drop entries
    /// matching their own endpoint id, [CI-11]) cannot tell them apart, and a peer dialing the node
    /// reaches whichever socket the fold happened to keep. Co-located role-instances share the
    /// node's one endpoint; the WS control plane (mandatory on every attach, and the plane every
    /// member of a run is connected to) carries the sibling's frames.
    ///
    /// Ownership is `(role, incarnation)`-keyed and **self-healing**: it is honored only while the
    /// recorded owner is still a live instance of that run, so a terminal / leave / reconvergence
    /// at a new incarnation lets the next authorship take the endpoint over. Re-authoring for the
    /// SAME role-instance (reconvergence within an incarnation) keeps it — the credentials body
    /// must keep naming the socket the published record already advertises.
    fn claim_node_iroh_endpoint(&self, run_label: &str, role: &str, incarnation: u64) -> bool {
        let mut owners = self.iroh_endpoint.lock().expect("iroh endpoint lock");
        if let Some(owner) = owners.get(run_label) {
            if owner.role == role && owner.incarnation == incarnation {
                return true; // Idempotent re-authorship by the owner itself.
            }
            if self.iroh_owner_is_live(run_label, owner) {
                tracing::info!(
                    run = run_label,
                    role,
                    incarnation,
                    owner_role = owner.role,
                    owner_incarnation = owner.incarnation,
                    "co-located role-instance shares this node's single iroh endpoint (owned by \
                     its sibling): attaching WS-only, no second bind on the node's pinned port"
                );
                return false;
            }
        }
        owners.insert(
            run_label.to_string(),
            IrohEndpointOwner {
                role: role.to_string(),
                incarnation,
            },
        );
        true
    }

    /// Whether `owner` still names a live role-instance of `run_label` (its primary instance or
    /// its co-located trainer sibling) — the liveness half of [`Self::claim_node_iroh_endpoint`].
    fn iroh_owner_is_live(&self, run_label: &str, owner: &IrohEndpointOwner) -> bool {
        let live = |map: &Mutex<BTreeMap<String, InstanceEntry>>| {
            map.lock()
                .expect("instances lock")
                .get(run_label)
                .is_some_and(|e| e.id.role == owner.role && e.generation == owner.incarnation)
        };
        live(&self.instances) || live(&self.co_trainers)
    }

    /// Forget which role-instance owned this node's iroh endpoint for `run_label` (the run is
    /// over — leave / terminal). Purely bookkeeping: a stale owner is already ignored by
    /// [`Self::claim_node_iroh_endpoint`]'s liveness check.
    fn forget_node_iroh_endpoint(&self, run_label: &str) {
        self.iroh_endpoint
            .lock()
            .expect("iroh endpoint lock")
            .remove(run_label);
    }

    /// The pinned iroh bind port for `run_label`: the configured `[vhc].iroh.bind_port`, or a
    /// free UDP port chosen once per node lifetime (cached so republishes and re-authored
    /// credentials always agree on the socket).
    fn iroh_bind_port(&self, run_label: &str) -> Result<u16, VhcError> {
        if self.config.iroh.bind_port != 0 {
            return Ok(self.config.iroh.bind_port);
        }
        let mut ports = self.iroh_ports.lock().expect("iroh ports lock");
        if let Some(port) = ports.get(run_label) {
            return Ok(*port);
        }
        let socket = std::net::UdpSocket::bind(("127.0.0.1", 0))
            .map_err(|e| VhcError::Internal(format!("pick iroh bind port: {e}")))?;
        let port = socket
            .local_addr()
            .map_err(|e| VhcError::Internal(format!("read iroh bind port: {e}")))?
            .port();
        ports.insert(run_label.to_string(), port);
        Ok(port)
    }

    /// Resolve the coordinator-seat bootstrap for a run: read the configured seat role's slot and,
    /// when it holds a lease, hand the incumbent's certificate + published endpoint to credential
    /// authorship. This is the out-of-band trust a trainer needs — the coordinator's on-plane
    /// §12.3 certificate announcement is a one-shot a later subscriber never sees, so the seat
    /// lease (untrusted storage, but carrying the cert) is how a joining trainer authenticates the
    /// incumbent's frames from the first round. `CertCheck` still gates trust by the genesis base.
    async fn resolve_seat_bootstrap(&self, run_label: &str) -> crate::credentials::SeatBootstrap {
        let Some(dir) = &self.seat_read else {
            return crate::credentials::SeatBootstrap::default();
        };
        let lease = match dir.read_seat(run_label, &self.config.seat_role).await {
            Ok(daemon_vhc_proto::SeatState::Leased(lease)) => lease,
            _ => return crate::credentials::SeatBootstrap::default(),
        };
        // The stored lease is registry METADATA until the full peer-side acceptance passes:
        // signature + certificate chain to a genesis-trusted base + expiry + the persisted
        // verified leadership-term floor ([SEAT-3] v2 — a registry serving a stale lease is
        // refused here, and nothing it serves ever advances a local counter). A refused lease
        // degrades to the empty bootstrap: the join proceeds without seat trust, and `CertCheck`
        // still gates every frame.
        let trusted = match self.genesis_trusted_bases(run_label).await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(run = run_label, error = %e, "seat bootstrap: no trust set");
                return crate::credentials::SeatBootstrap::default();
            }
        };
        let mut terms = daemon_vhc_proto::SeatTermLedger::new();
        if let Ok(Some((term, claimant))) = self.store.seat_term(run_label, &self.config.seat_role)
        {
            terms.restore_floor(
                lease.body.run_id,
                &self.config.seat_role,
                term,
                daemon_vhc_proto::PeerId(claimant),
            );
        }
        match crate::seat::authorize_incumbent(
            &lease,
            trusted.bases(),
            &daemon_vhc_proto::RevocationLedger::new(),
            &terms,
            now_ms() as u64,
            crate::seat::default_skew_ms(),
        ) {
            Ok(authorized) => {
                // Verified evidence — and only verified evidence — advances the persisted floor.
                if let Err(e) = self.store.observe_seat_term(
                    run_label,
                    &self.config.seat_role,
                    authorized.leadership_term,
                    &authorized.claimant.0,
                ) {
                    tracing::warn!(run = run_label, error = %e, "seat bootstrap: floor persist");
                }
                crate::credentials::SeatBootstrap {
                    peer_certs: vec![authorized.certificate],
                    ws_base: lease.body.endpoint.ws.clone(),
                    seat_grant: Some((*lease).clone()),
                }
            }
            Err(e) => {
                tracing::warn!(
                    run = run_label,
                    error = %e,
                    "seat bootstrap: stored lease refused (joining without seat trust)"
                );
                crate::credentials::SeatBootstrap::default()
            }
        }
    }

    /// The §12.3 pull half — seed a joining instance's judge with every VERIFIED roster
    /// certificate. The on-plane certificate announcement is a one-shot a later subscriber never
    /// sees, and the seat-lease bootstrap covers the coordinator's cert only: a trainer that
    /// joined BEFORE this instance attached has no push path left (its WS reannounce heals only a
    /// shared live WS hop, and gossip carries no re-announcements) — observed live on the two-box
    /// WAN rung as a coordinator refusing every peer frame `UncertifiedSender` forever. Every
    /// roster record already carries its publisher's certificate; each record is authorized here
    /// (signature + chain to a genesis-trusted base) before its certificate is handed to
    /// credential authorship — `CertCheck` still gates every frame, so this only ever ADDS
    /// verifiable bootstrap trust. Best-effort by design (the push half still runs): a roster
    /// fetch failure degrades to the pre-seeded certs, it never refuses the join.
    async fn extend_roster_certs(
        &self,
        run_label: &str,
        certs: &mut Vec<daemon_vhc_proto::RunKeyCertificate>,
    ) {
        let Some(discovery) = &self.discovery else {
            return;
        };
        let trusted = match self.genesis_trusted_bases(run_label).await {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(run = run_label, error = %e, "roster cert seeding: no trust set");
                return;
            }
        };
        let records = match discovery.fetch_roster(run_label).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(run = run_label, error = %e, "roster cert seeding: fetch failed");
                return;
            }
        };
        let mut seeded = 0usize;
        for record in records {
            if record.authorize(trusted.bases()).is_ok() && !certs.contains(&record.certificate) {
                certs.push(record.certificate.clone());
                seeded += 1;
            }
        }
        tracing::info!(
            run = run_label,
            seeded,
            total = certs.len(),
            "roster cert seeding: judge bootstrap trust extended from the verified roster"
        );
    }

    /// Consume one committed run-level module-upgrade record and drive the live switch through
    /// the worker-control surface (architecture §5.4; ABI §10.3) — the node-side record
    /// consumption seam behind [`VhcApi::vhc_switch_module`].
    ///
    /// The record is NEVER trusted as presented. Validation is total and fail-closed, against
    /// state the node derives itself:
    ///
    /// 1. decode the canonical-CBOR record; find the live role-instance;
    /// 2. fetch + verify the frozen genesis (discovery), cross-check the run identity;
    /// 3. rebuild the transition chain from genesis + the node's persisted record mirror, then
    ///    validate-and-append the presented record (domain, hash-link, strictly-monotone epoch,
    ///    stale-old-module, authority threshold — any failure refuses, the chain untouched);
    /// 4. worker pre-switch assessment of the target (the worker computes the post-switch
    ///    tuple's claim hash — the node never touches module bytes);
    /// 5. mint the post-switch incarnation (strictly above the running one) and provision its
    ///    key + certificate in the identity keystore (the re-issuance handshake);
    /// 6. drive `switch_module`; on activation persist the record mirror + the advanced
    ///    execution identity.
    ///
    /// Validation refusals return [`VhcSwitchOutcome::Refused`] in-band (operator-facing);
    /// infrastructure faults (store/discovery/worker transport) are `Err`.
    async fn consume_upgrade_record(
        &self,
        run_id: &str,
        record_bytes: &[u8],
    ) -> Result<daemon_api::VhcSwitchOutcome, VhcError> {
        use daemon_api::VhcSwitchOutcome as Outcome;
        use daemon_vhc_proto::{TransitionChain, UpgradeAuthority, UpgradeRecord};

        let refused = |reason: String| Ok(Outcome::Refused { reason });

        // 1. Decode the presented record (canonical CBOR, never trusted as presented).
        let record: UpgradeRecord = match daemon_vhc_proto::from_canonical_slice(record_bytes) {
            Ok(r) => r,
            Err(e) => return refused(format!("undecodable upgrade record: {e}")),
        };

        // The live role-instance this record targets.
        let instance = {
            let instances = self.instances.lock().unwrap();
            instances
                .get(run_id)
                .map(|e| (e.id.clone(), e.worker.clone()))
        };
        let Some((id, worker)) = instance else {
            return refused(format!("no live role-instance is held for run `{run_id}`"));
        };
        if record.body.role != id.role {
            return refused(format!(
                "the record upgrades role `{}` but this node's instance holds `{}`",
                record.body.role, id.role
            ));
        }

        // 2. The frozen genesis (the chain anchor), fetched + verified through discovery.
        let Some(discovery) = &self.discovery else {
            return refused(
                "no run discovery is configured (the record cannot be validated against a \
                 verified genesis)"
                    .into(),
            );
        };
        let envelope_bytes = discovery.fetch_envelope(run_id).await?;
        let wire: daemon_vhc_proto::SignedEnvelope =
            match daemon_vhc_proto::from_canonical_slice(&envelope_bytes) {
                Ok(w) => w,
                Err(e) => return refused(format!("run envelope is not a signed envelope: {e}")),
            };
        let frozen =
            match daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer) {
                Ok(f) => f,
                Err(e) => return refused(format!("genesis envelope verification: {e}")),
            };
        let genesis = match frozen.decode() {
            Ok(g) => g,
            Err(e) => return refused(format!("genesis decode: {e}")),
        };
        let run_id_hash = *frozen.run_id();
        if record.body.run_id != run_id_hash {
            return refused("the record's run_id is not this run's genesis hash".into());
        }
        // Cross-check the row's backfilled identity when present (a stale label / spoof guard).
        if let Some(row) = self.store.get_run(run_id)? {
            if let Some(existing) = row.run_id_hash {
                if existing != run_id_hash.0 {
                    return refused(
                        "the run row's backfilled genesis hash disagrees with the fetched \
                         envelope"
                            .into(),
                    );
                }
            }
        }

        // 3. Rebuild the chain from genesis + the persisted record mirror, then validate-append.
        let authority = match UpgradeAuthority::from_genesis(&genesis.identities) {
            Ok(a) => a,
            Err(e) => return refused(format!("upgrade authority: {e}")),
        };
        let mut chain = match TransitionChain::genesis(&genesis, run_id_hash) {
            Ok(c) => c,
            Err(e) => return refused(format!("chain anchor: {e}")),
        };
        for (epoch, bytes) in self.store.upgrade_records(run_id)? {
            let mirrored: UpgradeRecord =
                daemon_vhc_proto::from_canonical_slice(&bytes).map_err(|e| {
                    VhcError::Internal(format!(
                        "persisted upgrade-record mirror for epoch {epoch} is undecodable: {e}"
                    ))
                })?;
            chain.append(mirrored, &authority).map_err(|e| {
                VhcError::Internal(format!(
                    "persisted upgrade-record mirror for epoch {epoch} fails re-validation: {e}"
                ))
            })?;
        }
        let target = match chain.append(record.clone(), &authority) {
            Ok(descriptor) => descriptor,
            Err(e) => return refused(format!("upgrade record refused: {e}")),
        };
        let epoch = target.epoch;
        let new_module = record.body.new_module;
        let grants_hash = record.body.grants_hash;

        // 4. Worker pre-switch assessment of the committed target (claim hash computed where
        // the module bytes live).
        let verdict = worker
            .assess_switch(
                envelope_bytes,
                Some(id.role.clone()),
                daemon_vhc_session::protocol::SwitchTarget {
                    epoch,
                    new_module: new_module.0,
                    grants_hash: grants_hash.0,
                },
            )
            .await?;
        if !verdict.eligible {
            return refused(format!(
                "switch target assessment refused: {}",
                verdict.reasons.join("; ")
            ));
        }
        let Some(mut tuple) = self.stamp_admitted_tuple(verdict.admitted_tuple)? else {
            return refused("switch target assessment produced no admitted tuple".into());
        };

        // 5. Mint the post-switch incarnation (strictly above the running one — a seat-leased
        // incarnation can exceed the counter) and provision its key + re-issued certificate.
        let Some(dir) = &self.identity_dir else {
            return refused(
                "node identity authorship is not configured (no identity keystore to provision \
                 the post-switch certificate in)"
                    .into(),
            );
        };
        let new_instance = self.store.mint_incarnation_above(id.instance)?;
        tuple.incarnation = new_instance;
        let keystore = daemon_vhc_session::keystore::VhcKeystore::open(dir)
            .map_err(|e| VhcError::Internal(format!("open identity keystore: {e}")))?;
        daemon_vhc_session::provisioning::provision_run_identity(
            &keystore,
            &daemon_vhc_session::provisioning::ProvisionScope {
                run_label: run_id,
                genesis_hash: run_id_hash.0,
                epoch,
                role: &id.role,
                incarnation: new_instance,
                module_hash: new_module.0,
            },
        )
        .map_err(|e| VhcError::Internal(format!("provision post-switch identity: {e}")))?;

        // 6. Drive the local transaction (the node clamps the drain deadline to its ceiling).
        let deadline_ms = self.config.upgrade.quiesce_deadline_max_ms;
        let outcome = worker
            .switch_module(
                run_id.to_string(),
                epoch,
                id.role.clone(),
                new_module.0,
                grants_hash.0,
                deadline_ms,
                Some(tuple.clone()),
            )
            .await?;
        match outcome {
            SwitchOutcome::Activated {
                epoch,
                module,
                retries,
            } => {
                // The record consumed cleanly: persist the mirror + the advanced identity
                // (backfilling the row's cryptographic RunId — the consumption verified the
                // frozen genesis, so the label→hash binding is now known-good).
                self.store.put_upgrade_record(run_id, epoch, record_bytes)?;
                self.store.backfill_run_id(run_id, &run_id_hash.0)?;
                self.store
                    .set_execution_identity(run_id, epoch, &id.role, new_instance)?;
                if let Some(row) = self.store.get_run(run_id)? {
                    self.store.set_observability(
                        run_id,
                        row.envelope_schema_major,
                        row.module_abi_major,
                        row.selected_driver.as_deref(),
                        Some(&module),
                    )?;
                }
                if let Ok(bytes) = protocol::encode(&tuple) {
                    let _ = self.store.set_admitted_tuple(run_id, &bytes);
                }
                // Advance the live entry's CURRENT generation so post-switch pump events
                // (stamped with the new incarnation) pass the stale-generation guard. The
                // ledger id (`entry.id`) deliberately stays as admitted — it is the arbiter's
                // reservation key, and the reservation carries across the switch (same
                // sandbox, same charge) until the instance's terminal release.
                {
                    let mut instances = self.instances.lock().unwrap();
                    if let Some(entry) = instances.get_mut(run_id) {
                        entry.generation = new_instance;
                    }
                }
                let mut emitted = Vec::new();
                let row = self.store.get_run(run_id)?;
                self.emit(
                    VhcEvent::Phase {
                        run_id: run_id.to_string(),
                        phase: "module_switched".to_string(),
                        epoch,
                        round: row.map(|r| r.last_round).unwrap_or(0),
                    },
                    &mut emitted,
                )?;
                self.emit_changed(Some(run_id.to_string()));
                Ok(Outcome::Activated {
                    epoch,
                    module_hash: daemon_vhc_proto::Hash(module).to_hex(),
                    retries,
                })
            }
            SwitchOutcome::Refused { reason } => Ok(Outcome::Refused { reason }),
            SwitchOutcome::Left { reason } => {
                // Post-fence exit: the run-level record stays committed; this node's instance
                // left (its terminal RunTerminated drives the state machine via the pump).
                Ok(Outcome::Left { reason })
            }
        }
    }

    /// Resolve the late-join checkpoint restore for a run (spec §9): the registry's best
    /// pointer FOR THIS ROLE — the freshest live pointer, else the freshest drain snapshot;
    /// another role's pointer is never consulted for the restore itself — decoded to the wire
    /// restore form (`None` = fresh start / no discovery / nothing published for the role). A
    /// malformed pointer hash is dropped (fresh start), never a hard join failure.
    ///
    /// **Join-time cadence-vs-ring reachability** (the recovery-honesty check): a restorer
    /// replays forward across the coordinator's retained record ring
    /// ([`daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS`]), and with three seats the run
    /// progresses while one trainer is absent — no static cadence/ring relation guarantees the
    /// fence stays reachable. So the judgment is made HERE, from actual checkpoint metadata,
    /// before any rehydration: when the restore fence is more than the horizon behind the run's
    /// live head, the join refuses typed ([`VhcError::CheckpointStale`]) instead of wedging
    /// into the module's post-restore `GapRefused`. The head estimate is the freshest
    /// checkpoint round visible for the run (the coordinator's pointer) — a LOWER bound on the
    /// true head, so this check is conservative and the in-module refusal remains the
    /// authoritative backstop. No coordinator pointer = no judgment (nothing to compare
    /// against). Benign at min/max 2/2 (rounds pause while a peer is down); mandatory before
    /// C2's larger fleets.
    async fn resolve_restore(
        &self,
        run_id: &str,
        role: &str,
    ) -> Result<Option<protocol::CheckpointRestore>, VhcError> {
        let Some(discovery) = self.discovery.as_ref() else {
            return Ok(None);
        };
        let role = if role.is_empty() { "trainer" } else { role };
        let Some(pointer) = discovery
            .fetch_checkpoint(run_id, role)
            .await
            .ok()
            .flatten()
        else {
            return Ok(None);
        };
        let Some(hash) = hex32(&pointer.hash) else {
            return Ok(None);
        };
        if role != "coordinator" {
            if let Ok(Some(coord)) = discovery.fetch_checkpoint(run_id, "coordinator").await {
                let head = coord.round.max(pointer.round);
                let horizon = daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS;
                if head > pointer.round.saturating_add(horizon) {
                    return Err(VhcError::CheckpointStale {
                        restored: pointer.round,
                        head,
                        horizon,
                    });
                }
            }
        }
        // The node resolved this role's freshest restore pointer (spec §9); the joining worker
        // fetches its by-reference checkpoint document and streams each family's windows via
        // chunk-keyed rehydration ([SF-6]). This is the node-visible half of the restore path.
        tracing::info!(
            run = run_id,
            role,
            round = pointer.round,
            "resolved late-join checkpoint restore pointer (streaming by-ref rehydration follows)"
        );
        Ok(Some(protocol::CheckpointRestore {
            round: pointer.round,
            hash,
        }))
    }
}

/// Decode a 64-char lowercase-hex blake3 into a 32-byte array (`None` on any malformation).
fn hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[async_trait]
impl VhcApi for VhcService {
    async fn vhc_run_list(&self) -> Result<Vec<VhcRunSummary>, ApiError> {
        let runs = self
            .store
            .list_runs()
            .map_err(|e| VhcError::from(e).to_api())?;
        Ok(runs.into_iter().map(run_summary).collect())
    }

    async fn vhc_run_detail(&self, run_id: String) -> Result<Option<VhcRunDetail>, ApiError> {
        let map = |e: StoreError| VhcError::from(e).to_api();
        let Some(run) = self.store.get_run(&run_id).map_err(map)? else {
            return Ok(None);
        };
        let contribution = self.store.get_contribution(&run_id).map_err(map)?;
        let recent_events = self
            .store
            .recent_events(&run_id, EVENT_WINDOW)
            .map_err(map)?;
        // Surface the newest round's digest on the snapshot (wire v44): the highest-round
        // `RoundOutcome` in the window carries it, so a polling `--watch` client collects the
        // digest-agreement transcript without an event subscription. The node decides; the app
        // renders — this is a projection of events the node already owns, never re-derived.
        let last_round_digest = recent_events
            .iter()
            .filter_map(|e| match e {
                VhcEvent::RoundOutcome { round, digest, .. } => Some((*round, *digest)),
                _ => None,
            })
            .max_by_key(|(round, _)| *round)
            .map(|(_, digest)| digest);
        Ok(Some(VhcRunDetail {
            coordinator: run.coordinator.clone(),
            summary: run_summary(run),
            contribution,
            recent_events,
            last_round_digest,
        }))
    }

    async fn vhc_join(
        &self,
        run_id: String,
        policy: VhcPolicy,
        _op_id: String,
    ) -> Result<(), ApiError> {
        // Idempotency is enforced upstream by the dispatch op-id dedup guard; the store's
        // INSERT-OR-UPDATE keeps a repeated join convergent regardless.
        self.require_enabled().map_err(|e| e.to_api())?;

        // A repeated join of a LIVE role-instance re-converges on the existing child + its
        // standing reservation — it never double-charges the ledgers.
        let existing = {
            let instances = self.instances.lock().unwrap();
            instances
                .get(&run_id)
                .map(|e| (e.id.clone(), e.worker.clone()))
        };
        let worker = existing
            .as_ref()
            .map_or_else(|| self.instance_worker(), |(_, w)| w.clone());
        // A freshly-spawned factory child serves probe/assess (the assessment instance precedes
        // arbitration — the claim is arbitration's input); if the join is later refused, that
        // child is torn down, never left idling.
        let fresh_child = existing.is_none() && self.worker_factory.is_some();

        // Node-computed eligibility (ADR-003). When a discovery seam is configured, resolve the
        // run + fetch the frozen envelope + run the worker's real §6.5 `AssessRun` before `JoinRun`,
        // and take the coordinator endpoint from discovery. With no discovery configured, fall back
        // to the probe-based eligibility against the allowlisted coordinator (offline / no-registry
        // path). Either way the persisted eligibility is node-computed — the app never re-derives it.
        //
        // COORDINATOR DUTY FIRST (architecture §6.3; [SEAT-1] v2): when the owner enabled seat
        // claiming and this is a fresh instance, try the seat — assess the configured seat role,
        // claim (or re-adopt) the registry slot, and run the join AT THE WON LEASE'S
        // counter-minted execution incarnation (the leadership term rides the lease
        // independently). A live foreign incumbent, a lost CAS race, or an ineligible seat-role
        // assessment stands down to the trainer default.
        let seat_join = if existing.is_none() {
            self.try_seat_join(&worker, &run_id).await
        } else {
            None
        };
        let (coordinator, eligibility, assessed_tuple, seat_incarnation) = match seat_join {
            Some(won) => won,
            None => {
                let (coordinator, eligibility, assessed_tuple) = self
                    .resolve_join(&worker, &run_id, None)
                    .await
                    .map_err(|e| e.to_api())?;
                (coordinator, eligibility, assessed_tuple, None)
            }
        };

        // An ineligible assessment refuses HERE, in the funnel's own words. Letting it fall
        // through to charge derivation used to convert every assess refusal into the same
        // "nothing to reserve" internal error — the worker's typed reasons (a profile that did
        // not authenticate, an estimate that would not compose, a missing backend) were swallowed
        // by a message about ledger arithmetic that never got the chance to be the problem.
        if !eligibility.eligible {
            if fresh_child {
                worker.shutdown().await;
            }
            return Err(VhcError::Worker(format!(
                "assessment refused the join: {}",
                if eligibility.reasons.is_empty() {
                    "(no reason reported)".to_string()
                } else {
                    eligibility.reasons.join("; ")
                }
            ))
            .to_api());
        }

        // The admission funnel's LAST stage (decisions D6 point 5; architecture §3.5): the
        // aggregate owner arbitration — an atomic check-and-reserve against the remaining
        // per-device + host-wide ledgers, committed BEFORE the run instance is created. Owner
        // priority comes from the node-side policy store, never the envelope.
        let id = match &existing {
            Some((id, _)) => id.clone(),
            None => {
                let persisted = self
                    .store
                    .get_run(&run_id)
                    .map_err(|e| VhcError::from(e).to_api())?;
                // A resumable persisted row contributes its epoch/role/run-hash (the intent's
                // execution CONTEXT) but never its incarnation: with no live instance there is
                // no sequence continuity to resume, so a rejoin is always the REPLACE entry
                // mode — a fresh never-reused incarnation whose stale predecessor stays gated.
                // (Retention re-published the dead instance's incarnation and freshness pair,
                // which the registry's monotonic roster fold rightly refused — the run-g
                // poisoned-slot livelock.)
                let (epoch, role, run_hash) = persisted
                    .as_ref()
                    .filter(|r| r.desired_state == DesiredState::Joined && r.run_state.resumable())
                    .map_or((0, String::new(), None), |r| {
                        (r.epoch, r.role.clone(), r.run_id_hash)
                    });
                let id = RoleInstanceId {
                    // The cryptographic RunId when backfilled; a v1-era run keys its node-local
                    // ledger entry by blake3(RunLabel) until then (decisions D1 lazy backfill).
                    run_id: run_hash.unwrap_or_else(|| *blake3::hash(run_id.as_bytes()).as_bytes()),
                    epoch,
                    // The seat-won join runs the configured seat role; else the persisted role
                    // (the trainer default for a fresh row).
                    role: if seat_incarnation.is_some() {
                        self.config.seat_role.clone()
                    } else if role.is_empty() {
                        "trainer".to_string()
                    } else {
                        role
                    },
                    // The seat-won join runs at the LEASE'S execution incarnation — which the
                    // keeper COUNTER-MINTED at claim ([SEAT-1] v2: the counter mints every
                    // execution identity; the leadership term rides the lease independently).
                    // Every other join is likewise a fresh counter mint (decisions D1). No path
                    // re-publishes a dead instance's incarnation.
                    instance: if let Some(inc) = seat_incarnation {
                        inc
                    } else {
                        self.store
                            .mint_incarnation()
                            .map_err(|e| VhcError::from(e).to_api())?
                    },
                };
                let charge = self
                    .derive_charge(&eligibility, &policy, &id.role)
                    .map_err(|e| e.to_api())?;
                let priority = self
                    .store
                    .run_priority(&run_id)
                    .map_err(|e| VhcError::from(e).to_api())?;
                if let Err(refusal) = self.admit_placed(&id, charge, priority) {
                    if fresh_child {
                        worker.shutdown().await;
                    }
                    return Err(VhcError::from(refusal).to_api());
                }
                id
            }
        };

        // Author identity + credentials for THIS incarnation (D-P8): mint the per-run key + its
        // certificate under the base identity, stamp the minted incarnation into the tuple, and
        // author the secrets-free plane-selection credentials (the token, if any, lands only in
        // the keystore record `credentials_ref` points at — never on the wire).
        let restore = match self.resolve_restore(&run_id, &id.role).await {
            Ok(r) => r,
            Err(e) => {
                if existing.is_none() {
                    self.arbiter.release(&id);
                    if fresh_child {
                        worker.shutdown().await;
                    }
                }
                return Err(e.to_api());
            }
        };
        let seat = self.resolve_seat_bootstrap(&run_id).await;
        let mut id = id;
        let authored = match self
            .author_join(
                &run_id,
                &coordinator,
                &id,
                assessed_tuple.clone(),
                restore.clone(),
                seat.clone(),
            )
            .await
        {
            // A verified own-roster-floor repair (judged inside the authorship step): the raised
            // counter invalidates this provisional identity WHOLESALE — unwind the reservation,
            // mint the superseding incarnation, re-admit, re-author. Once; a second stale
            // refusal surfaces typed. A seat-won join is excluded: its incarnation binds the
            // won lease, so a poisoned roster there stands down (the keeper's next claim mints
            // fresh above the repaired counter anyway).
            Err(VhcError::OwnFloorRepaired { floor })
                if existing.is_none() && seat_incarnation.is_none() =>
            {
                self.arbiter.release(&id);
                tracing::info!(
                    run_id,
                    floor,
                    "join: restarting authorship above the repaired roster floor"
                );
                let restarted = async {
                    id.instance = self.store.mint_incarnation()?;
                    let charge = self.derive_charge(&eligibility, &policy, &id.role)?;
                    let priority = self.store.run_priority(&run_id)?;
                    self.admit_placed(&id, charge, priority)
                        .map_err(VhcError::Resources)?;
                    Ok(())
                }
                .await;
                match restarted {
                    Ok(()) => {
                        self.author_join(&run_id, &coordinator, &id, assessed_tuple, restore, seat)
                            .await
                    }
                    Err(e) => Err(e),
                }
            }
            other => other,
        };
        let (delivery_tuple, credentials, credentials_ref) = match authored {
            Ok(v) => v,
            Err(e) => {
                if existing.is_none() {
                    self.arbiter.release(&id);
                    if fresh_child {
                        worker.shutdown().await;
                    }
                }
                return Err(e.to_api());
            }
        };

        if let Err(e) = self.store.put_join_intent(
            &run_id,
            &coordinator,
            &policy,
            credentials_ref.as_deref(),
            &eligibility,
        ) {
            if existing.is_none() {
                self.arbiter.release(&id);
                if fresh_child {
                    worker.shutdown().await;
                }
            }
            return Err(VhcError::from(e).to_api());
        }
        let _ = self
            .store
            .set_execution_identity(&run_id, id.epoch, &id.role, id.instance);
        // Persist the immutable admitted tuple (now carrying the minted incarnation) beside the
        // durable join intent (architecture §6.3), so a restart re-converges against the exact
        // assessed identity + incarnation.
        if let Some(tuple) = &delivery_tuple {
            if let Ok(bytes) = protocol::encode(tuple) {
                let _ = self.store.set_admitted_tuple(&run_id, &bytes);
            }
        }

        // Join over the streaming path + pump the continuous worker event stream into
        // `handle_worker_event` so vhc.db reflects live round progression (§10.3/§10.4). The
        // opaque `JoinRun.credentials` the worker's live attach parses are
        // authored where the node identity + roster are known (the e2e / boot join_and_pump path);
        // an API-initiated join with no authored credentials keeps the worker's self-driven round
        // (WS-only baseline), still pumped.
        let rx = match worker
            .join_streaming(
                run_id.clone(),
                coordinator,
                credentials,
                to_join_policy(&policy),
                delivery_tuple,
            )
            .await
        {
            Ok(rx) => rx,
            Err(e) => {
                // The child never joined: the reservation is surrendered (a fresh admission
                // only — a live instance keeps its) and a fresh factory child is torn down.
                if existing.is_none() {
                    self.arbiter.release(&id);
                    if fresh_child {
                        worker.shutdown().await;
                    }
                }
                return Err(e.to_api());
            }
        };
        let generation = id.instance;
        self.instances.lock().unwrap().insert(
            run_id.clone(),
            InstanceEntry {
                generation: id.instance,
                id,
                worker,
            },
        );
        self.spawn_pump(Some((run_id.clone(), generation)), rx);
        // Defect D: a seat-holding trainer+coordinator node ALSO runs a trainer role-instance for
        // the same run (config-gated: `coordinator_trains`), so a self-coordinated run meets its
        // own membership floor. Only on a fresh seat-won join; best-effort (never fails the join).
        if seat_incarnation.is_some() && existing.is_none() && self.config.coordinator_trains {
            self.spawn_co_located_trainer(&run_id, &policy).await;
        }
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn vhc_leave(
        &self,
        run_id: String,
        mode: VhcLeaveMode,
        _op_id: String,
    ) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        let map = |e: StoreError| VhcError::from(e).to_api();
        self.store
            .set_desired_state(&run_id, DesiredState::Left)
            .map_err(map)?;
        // The owner took manual control: no reconvergence is pending or owed.
        self.store.clear_retry(&run_id).map_err(map)?;
        let (worker, held) = {
            let instances = self.instances.lock().unwrap();
            match instances.get(&run_id) {
                Some(e) => (e.worker.clone(), true),
                None => (self.worker.clone(), false),
            }
        };
        tracing::debug!(
            run_id,
            instance_held = held,
            "vhc leave: dispatching to the run's worker"
        );
        worker
            .leave(run_id.clone(), to_leave_mode(mode))
            .await
            .map_err(|e| e.to_api())?;
        // Observed teardown → release (decisions D6 point 6): the leave has been accepted by the
        // child's serial command loop (which drops the run's wasm instance + device allocations
        // before servicing anything else), so the ledger entry is surrendered only NOW — never
        // optimistically before the victim gave the memory back. The release rides the durable
        // marker protocol so a crash between the observation and the terminal commit is repaired
        // by the startup reconciliation pass.
        self.store
            .begin_release(&run_id, RunState::Left, None)
            .map_err(map)?;
        let entry = self.instances.lock().unwrap().remove(&run_id);
        if let Some(e) = &entry {
            self.arbiter.release(&e.id);
        }
        self.store.commit_release(&run_id).map_err(map)?;
        // Defect D: tear down the co-located trainer sibling too — drain it (graceful/hard per the
        // owner's mode), stop its child, and release its ledger reservation. Its pending terminal
        // (dispatched by generation) then finds no entry and is a no-op.
        let co = self.co_trainers.lock().unwrap().remove(&run_id);
        if let Some(e) = co {
            let _ = e.worker.leave(run_id.clone(), to_leave_mode(mode)).await;
            e.worker.shutdown().await;
            self.arbiter.release(&e.id);
        }
        // The run holds no role-instance here anymore: this node's iroh endpoint is unowned.
        self.forget_node_iroh_endpoint(&run_id);
        // A leaving coordinator surrenders its seat (fenced release; the floor persists).
        self.release_seat_for(&run_id).await;
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn vhc_pause(&self, run_id: String, _op_id: String) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        let map = |e: StoreError| VhcError::from(e).to_api();
        let Some(_) = self.store.get_run(&run_id).map_err(map)? else {
            return Err(ApiError::Other(format!("unknown run `{run_id}`")));
        };
        // Durable owner intent FIRST: pause survives a crash/restart from this point on (a
        // paused run is never reconverged until resumed), and the owner taking manual control
        // clears any pending reconvergence.
        self.store
            .set_desired_state(&run_id, DesiredState::Paused)
            .map_err(map)?;
        self.store.clear_retry(&run_id).map_err(map)?;
        // The HARD pause reaches the run's worker (its own factory child under multi-instance
        // supervision; the shared default worker otherwise — whose pause is host-wide by
        // construction). `paused` promises memory, not just time: the worker drops the wasm
        // instance + device allocations and keeps only CPU masters.
        let entry_worker = {
            let instances = self.instances.lock().unwrap();
            instances
                .get(&run_id)
                .map(|e| (e.id.clone(), e.worker.clone()))
        };
        if let Some((id, worker)) = entry_worker {
            worker
                .throttle(None, None, true)
                .await
                .map_err(|e| e.to_api())?;
            // Release-on-pause: the pause was accepted by the child's serial command loop (the
            // memory is surrendered), so the ledger reservation releases — accurate accounting
            // over a held-but-idle reservation; resume re-admits against the CURRENT ledgers.
            self.arbiter.release(&id);
        }
        // Release-on-pause covers the coordinator seat too: a paused coordinator surrenders its
        // lease (fenced release) so a standby can take the seat at floor + 1.
        self.release_seat_for(&run_id).await;
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn vhc_resume(&self, run_id: String, _op_id: String) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        let map = |e: StoreError| VhcError::from(e).to_api();
        let Some(run) = self.store.get_run(&run_id).map_err(map)? else {
            return Err(ApiError::Other(format!("unknown run `{run_id}`")));
        };
        if run.desired_state != DesiredState::Paused {
            // Idempotent for an already-joined run; anything else is a typed refusal.
            return if run.desired_state == DesiredState::Joined {
                Ok(())
            } else {
                Err(ApiError::Other(format!(
                    "run `{run_id}` is not paused (it was left); join it instead"
                )))
            };
        }
        // A live paused instance resumes in place: re-admit against the CURRENT ledgers (the
        // pause released the reservation), then lift the hard pause. A refusal is typed and
        // LOUD — the run stays paused, nothing is half-resumed.
        let entry_worker = {
            let instances = self.instances.lock().unwrap();
            instances
                .get(&run_id)
                .map(|e| (e.id.clone(), e.worker.clone()))
        };
        if let Some((id, worker)) = entry_worker {
            let charge = self
                .derive_charge(&run.eligibility, &run.policy, &id.role)
                .map_err(|e| e.to_api())?;
            let priority = self.store.run_priority(&run_id).map_err(map)?;
            if let Err(refusal) = self.admit_placed(&id, charge, priority) {
                return Err(VhcError::from(refusal).to_api());
            }
            if let Err(e) = worker.throttle(None, None, false).await {
                // The lever never landed: surrender the fresh reservation, stay paused.
                self.arbiter.release(&id);
                return Err(e.to_api());
            }
            self.store
                .set_desired_state(&run_id, DesiredState::Joined)
                .map_err(map)?;
            self.store.mark_running(&run_id).map_err(map)?;
        } else {
            // No live instance (paused across a restart / the instance died while paused):
            // reconverge fresh under the reinstated intent. Failure keeps the run paused.
            self.reconverge(&run).await.map_err(|e| e.to_api())?;
            self.store
                .set_desired_state(&run_id, DesiredState::Joined)
                .map_err(map)?;
        }
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn vhc_switch_module(
        &self,
        run_id: String,
        upgrade_record: Vec<u8>,
        _op_id: String,
    ) -> Result<daemon_api::VhcSwitchOutcome, ApiError> {
        // Idempotency is enforced upstream by the dispatch op-id dedup guard.
        self.require_enabled().map_err(|e| e.to_api())?;
        self.consume_upgrade_record(&run_id, &upgrade_record)
            .await
            .map_err(|e| e.to_api())
    }

    async fn vhc_set_policy(&self, policy: VhcPolicy) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        // Push the governor levers to the worker (§10.5) — and, under multi-instance
        // supervision, to every live role-instance child (the owner lever is host-wide). The
        // persisted default-policy slot for future joins is the config `[vhc].default_policy`;
        // a durable override lands with the policy store in a later wave.
        let vram = Some(policy.vram_cap_mb);
        let duty = Some(policy.duty_cycle_pct.min(100) as u8);
        self.worker
            .throttle(vram, duty, false)
            .await
            .map_err(|e| e.to_api())?;
        let children: Vec<Arc<dyn WorkerControl>> = {
            let instances = self.instances.lock().unwrap();
            instances.values().map(|e| e.worker.clone()).collect()
        };
        for child in children {
            // The shared worker was already throttled above; factory children are distinct.
            if !Arc::ptr_eq(&child, &self.worker) {
                child
                    .throttle(vram, duty, false)
                    .await
                    .map_err(|e| e.to_api())?;
            }
        }
        Ok(())
    }

    async fn vhc_hardware_report(&self) -> Result<VhcHardwareReport, ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        let hw = self.worker.probe().await.map_err(|e| e.to_api())?;
        Ok(hardware_report(hw))
    }

    async fn vhc_subscribe(&self, run_id: Option<String>) -> Result<VhcEventStream, ApiError> {
        let rx = self.events_tx.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(move |res| {
            let want = run_id.clone();
            async move {
                match res {
                    // Filter to one run when requested; drop `Lagged` gaps (the durable log + a
                    // VhcChanged pointer let a lagging client re-baseline via run_detail).
                    Ok(ev) => match &want {
                        Some(r) if ev.run_id() != r => None,
                        _ => Some(ev),
                    },
                    Err(_) => None,
                }
            }
        });
        Ok(stream.boxed())
    }
}

/// The generation stamp a run-scoped worker event carries (`None` for unstamped request/reply
/// events — `Ready`/`Probed`/`Assessed`/`Metric`/`Warning`/`Error`/`Pong`).
fn event_generation(ev: &protocol::Event) -> Option<u64> {
    match ev {
        protocol::Event::RunPhase { generation, .. }
        | protocol::Event::RoundProgress { generation, .. }
        | protocol::Event::RoundOutcome { generation, .. }
        | protocol::Event::CheckpointPublished { generation, .. }
        | protocol::Event::ModuleSwitched { generation, .. }
        | protocol::Event::AdmittedTupleMismatch { generation, .. }
        | protocol::Event::RunTerminated { generation, .. } => Some(*generation),
        _ => None,
    }
}

/// Exponential reconvergence backoff for the `attempt`-th consecutive recoverable failure
/// (0-based), capped at the configured ceiling.
fn retry_backoff_ms(cfg: &daemon_vhc_session::config::RetryConfig, attempt: u32) -> u64 {
    cfg.initial_backoff_ms
        .saturating_mul(2u64.saturating_pow(attempt.min(16)))
        .min(cfg.max_backoff_ms.max(cfg.initial_backoff_ms))
}

/// Wall-clock unix ms (the retry schedule is a coarse wall-clock quantity, like the store's).
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Wire<->worker mappings (the node is the single translation point)
// ---------------------------------------------------------------------------

fn run_summary(run: PersistedRun) -> VhcRunSummary {
    let joined = run.desired_state == DesiredState::Joined;
    let hex = |bytes: &[u8; 32]| {
        use std::fmt::Write as _;
        bytes.iter().fold(String::with_capacity(64), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    };
    // The D0 execution-identity trio travels only once the run is v2-identified (its RunId is
    // backfilled); a v1-only row keeps them absent — its identity is the RunLabel alone
    // (decisions D1 lazy backfill).
    let identified = run.run_id_hash.is_some();
    VhcRunSummary {
        run_id: run.run_id,
        phase: run.last_phase,
        joined,
        eligibility: run.eligibility,
        policy: if joined { Some(run.policy) } else { None },
        last_round: run.last_round,
        run_id_hash: run.run_id_hash.as_ref().map(hex),
        epoch: identified.then_some(run.epoch),
        role: identified.then(|| run.role.clone()),
        instance: identified.then_some(run.instance),
        envelope_schema_major: Some(run.envelope_schema_major),
        module_abi_major: run.module_abi_major,
        selected_driver: run.selected_driver,
        module_hash: run.module_hash.as_ref().map(hex),
        run_state: Some(effective_state(run.desired_state, run.run_state).to_string()),
        retry_count: Some(u64::from(run.retry_count)),
        terminal_reason: run.terminal_reason,
    }
}

fn to_policy_mode(mode: VhcPolicyMode) -> PolicyMode {
    match mode {
        VhcPolicyMode::Always => PolicyMode::Always,
        VhcPolicyMode::Idle => PolicyMode::Idle,
        VhcPolicyMode::Scheduled => PolicyMode::Scheduled,
        VhcPolicyMode::Manual => PolicyMode::Manual,
    }
}

fn to_join_policy(p: &VhcPolicy) -> JoinPolicy {
    JoinPolicy {
        mode: to_policy_mode(p.mode),
        vram_cap_mb: p.vram_cap_mb,
        duty_cycle_pct: p.duty_cycle_pct.min(100) as u8,
        schedule: p.schedule.clone(),
    }
}

fn to_leave_mode(mode: VhcLeaveMode) -> LeaveMode {
    match mode {
        VhcLeaveMode::Graceful => LeaveMode::Graceful,
        VhcLeaveMode::Immediate => LeaveMode::Immediate,
    }
}

fn hardware_report(hw: Hardware) -> VhcHardwareReport {
    VhcHardwareReport {
        gpus: hw.gpus,
        vram_mb: hw.vram_mb,
        // Wire v42: mirror the worker's unified-memory spillover (GTT) into the app-facing DTO
        // additively (a recorded follow-on), so the GUI's "what can my GPU do" panel
        // shows the true effective budget on integrated/UMA boxes.
        shared_mb: hw.shared_mb,
        ram_mb: hw.ram_mb,
        backend_lanes: hw.backend_lanes,
        capabilities: VhcCapabilities {
            abi_version: u32::from(hw.capabilities.abi_version),
            ops: hw.capabilities.ops,
            payload_stores: hw.capabilities.payload_stores,
        },
        up_kbps: hw.up_kbps,
        down_kbps: hw.down_kbps,
        disk_free_mb: hw.disk_free_mb,
        throughput_class: hw.throughput_class,
    }
}

/// Map the worker's real §6.5 `AssessRun` verdict onto the app-facing eligibility DTO. The
/// worker's `headroom` is an ordered `Vec<(String, i64)>`; the wire DTO is a `BTreeMap`. The app
/// renders this; it never re-derives eligibility (ADR-003).
fn eligibility_from_assess(e: &Eligibility) -> VhcEligibility {
    VhcEligibility {
        eligible: e.eligible,
        reasons: e.reasons.clone(),
        headroom: e.headroom.iter().cloned().collect(),
    }
}

/// A coarse node-computed eligibility from a hardware probe (the fallback when no discovery seam is
/// configured): eligible if the worker reports a usable GPU or backend lane. The app renders this;
/// it never re-derives eligibility (ADR-003).
///
/// D-10: the charge key is **claim-shaped** (`claim_device_bytes`, in BYTES), so this no-registry
/// default path presents `derive_charge` the SAME contract the v2 assess verdict does — one input
/// shape, no `derive_charge` branch. The conservative device-tier charge is the probed dedicated
/// VRAM (`Hardware.vram_mb`, MiB → bytes); `derive_charge` reads it directly, so the default path
/// charges probed VRAM and never falls through to a zero/default cap. Host RAM stays an
/// informational readout (`ram_mb`, MiB) for the app panel — it is not charged on this path (the
/// assess claim is the host-tier source).
fn eligibility_from_hardware(hw: &Hardware) -> VhcEligibility {
    let eligible = hw.gpus > 0 || !hw.backend_lanes.is_empty();
    let mut reasons = Vec::new();
    if !eligible {
        reasons.push("no usable GPU or backend lane".to_string());
    }
    let mut headroom = BTreeMap::new();
    headroom.insert(
        "claim_device_bytes".to_string(),
        hw.vram_mb.saturating_mul(1 << 20) as i64,
    );
    headroom.insert("ram_mb".to_string(), hw.ram_mb as i64);
    VhcEligibility {
        eligible,
        reasons,
        headroom,
    }
}

fn translate(ev: &protocol::Event, run_id: &str) -> Option<VhcEvent> {
    match ev {
        protocol::Event::RunPhase {
            phase,
            epoch,
            round,
            ..
        } => Some(VhcEvent::Phase {
            run_id: run_id.to_string(),
            phase: phase.clone(),
            epoch: *epoch,
            round: *round,
        }),
        protocol::Event::RoundProgress {
            inner_step,
            loss,
            tokens_per_s,
            peers,
            ..
        } => Some(VhcEvent::Progress {
            run_id: run_id.to_string(),
            inner_step: *inner_step,
            loss_micros: fixed(*loss, 1_000_000.0),
            tokens_per_s_milli: fixed(*tokens_per_s, 1_000.0),
            peers: *peers,
        }),
        protocol::Event::RoundOutcome {
            round,
            committed,
            ingested,
            stalled,
            digest,
            ..
        } => Some(VhcEvent::RoundOutcome {
            run_id: run_id.to_string(),
            round: *round,
            committed: *committed,
            ingested: *ingested,
            stalled: *stalled,
            // Surface the worker session's post-ingest det-state digest (§5.6) — the node used to
            // drop it here with a `..` pattern; it is the G-2 digest-agreement evidence (wire v44).
            digest: *digest,
        }),
        protocol::Event::Warning { class, detail } => Some(VhcEvent::Warning {
            run_id: run_id.to_string(),
            class: class.clone(),
            detail: detail.clone(),
        }),
        protocol::Event::Error { class, detail } => Some(VhcEvent::Error {
            run_id: run_id.to_string(),
            class: format!("{class:?}"),
            detail: detail.clone(),
        }),
        protocol::Event::AdmittedTupleMismatch { field, .. } => Some(VhcEvent::Warning {
            run_id: run_id.to_string(),
            class: "admitted_tuple_mismatch".to_string(),
            detail: format!("join aborted: admitted tuple field `{field}` changed; reassessing"),
        }),
        _ => None,
    }
}

/// Whether `endpoint` sits under one of the owner's allowlisted coordinator bases (spec §11.1):
/// a normalized (trailing-`/`-trimmed) PREFIX match, so one allowlisted base authorizes its
/// route tree (`{base}/runs/:id/ws`, presign, …) and nothing else. An empty allowlist authorizes
/// nothing — the discovery path refuses loud until the owner names a coordinator.
fn allowlisted(allowlist: &[String], endpoint: &str) -> bool {
    let endpoint = endpoint.trim_end_matches('/');
    allowlist.iter().any(|base| {
        let base = base.trim_end_matches('/');
        !base.is_empty() && (endpoint == base || endpoint.starts_with(&format!("{base}/")))
    })
}

/// Convert an `f32` telemetry value to a non-negative fixed-point integer (saturating).
fn fixed(v: f32, scale: f32) -> u64 {
    let scaled = (v.max(0.0) * scale).round();
    if scaled.is_finite() {
        scaled as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::allowlisted;

    #[test]
    fn allowlist_matches_on_path_boundaries_only() {
        let allow = vec!["https://coord.example/api/v1/vhc".to_string()];
        // The base itself and its route tree pass (trailing slashes normalized).
        assert!(allowlisted(&allow, "https://coord.example/api/v1/vhc"));
        assert!(allowlisted(&allow, "https://coord.example/api/v1/vhc/"));
        assert!(allowlisted(
            &allow,
            "https://coord.example/api/v1/vhc/runs/r1/ws"
        ));
        // A hostname EXTENDING the base is not a prefix match (the classic bypass).
        assert!(!allowlisted(
            &["https://coord.example".to_string()],
            "https://coord.example.evil.tld/api/v1/vhc"
        ));
        // A sibling path is refused; an empty allowlist authorizes nothing.
        assert!(!allowlisted(&allow, "https://coord.example/api/v1/other"));
        assert!(!allowlisted(&[], "https://coord.example/api/v1/vhc"));
        assert!(!allowlisted(
            &[String::new()],
            "https://coord.example/api/v1/vhc"
        ));
    }
}

/// Defect — the co-located trainer duty-arbitration fix (the M4 fleet-smoke blocker): a
/// seat-holding node with `coordinator_trains = true` must admit BOTH its coordinator seat
/// role-instance AND its co-located trainer role-instance under the node's DEFAULT (finite) owner
/// duty ledger — the coordinator/consensus role claiming zero accelerator duty is what makes room
/// for the trainer's full duty. These tests drive the SAME arbitration code path the live node
/// runs ([`VhcService::derive_charge`] → [`VhcService::admit_placed`] → [`OwnerArbiter::admit`])
/// against the exact `OwnerBudget::from_config` default the M4 used (`duty_pct = 100`), so the
/// pre-fix behavior (a full-duty coordinator exhausting the ledger and refusing its own trainer)
/// is caught here rather than only on real hardware.
#[cfg(test)]
mod arbitration_tests {
    use super::*;
    use daemon_api::{VhcEligibility, VhcPolicy, VhcPolicyMode};
    use daemon_vhc_session::config::OwnerBudgetConfig;

    /// A worker seam that never spawns a subprocess — these tests exercise only the node-side
    /// charge derivation + arbiter admission (no worker traffic).
    pub(super) struct NoopWorker;

    #[async_trait]
    impl WorkerControl for NoopWorker {
        async fn probe(&self) -> Result<Hardware, VhcError> {
            Ok(Hardware::default())
        }
        async fn assess(
            &self,
            _envelope: Vec<u8>,
            _role: Option<String>,
        ) -> Result<Eligibility, VhcError> {
            Ok(Eligibility::default())
        }
        async fn join(
            &self,
            _run_id: String,
            _coordinator: String,
            _credentials: Vec<u8>,
            _policy: JoinPolicy,
            _admitted_tuple: Option<protocol::AdmittedTuple>,
        ) -> Result<(), VhcError> {
            Ok(())
        }
        async fn leave(&self, _run_id: String, _mode: LeaveMode) -> Result<(), VhcError> {
            Ok(())
        }
        async fn throttle(
            &self,
            _vram_cap_mb: Option<u32>,
            _duty_cycle_pct: Option<u8>,
            _paused: bool,
        ) -> Result<(), VhcError> {
            Ok(())
        }
    }

    /// A claim-bearing eligibility (the discovery/assess input shape `derive_charge` reads):
    /// `device`/`host` bytes onto the claim-shaped headroom keys.
    fn eligibility(device: i64, host: i64) -> VhcEligibility {
        let mut headroom = BTreeMap::new();
        headroom.insert("claim_device_bytes".to_string(), device);
        headroom.insert("admitted_host_bytes".to_string(), host);
        VhcEligibility {
            eligible: true,
            reasons: Vec::new(),
            headroom,
        }
    }

    /// An eligibility reporting the composed reservation the governor derived — the certification
    /// path's input shape, already scope-correct.
    fn reserved_eligibility(device: i64, host: i64) -> VhcEligibility {
        let mut headroom = BTreeMap::new();
        headroom.insert(RESERVATION_DEVICE_BYTES_KEY.to_string(), device);
        headroom.insert(RESERVATION_HOST_BYTES_KEY.to_string(), host);
        VhcEligibility {
            eligible: true,
            reasons: Vec::new(),
            headroom,
        }
    }

    /// An eligibility with no memory figure at all: neither a composed reservation nor a declared
    /// legacy claim.
    fn figureless_eligibility() -> VhcEligibility {
        VhcEligibility {
            eligible: true,
            reasons: Vec::new(),
            headroom: BTreeMap::new(),
        }
    }

    /// The owner-cap fallback is GONE. It used to substitute the owner's own VRAM cap whenever no
    /// device figure was present, which with the guest's device tiers retired would have become the
    /// only path — reserving the whole budget for every role, refusing the second one, and calling
    /// that a resource decision. An absent figure is now a typed refusal naming exactly that.
    #[test]
    fn a_missing_memory_figure_is_a_typed_refusal_and_never_the_owners_cap() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // A generous standing cap is exactly what the old fallback would have charged.
        let mut generous = policy(100);
        generous.vram_cap_mb = 24_000;

        let err = svc
            .derive_charge(&figureless_eligibility(), &generous, "trainer")
            .expect_err("no figure means no reservation and no admission");
        let rendered = err.to_string();
        assert!(
            rendered.contains(AbiRefusalCode::EstimateNotComposable.slug()),
            "the refusal names the reason: {rendered}"
        );
        assert!(
            rendered.contains("not a substitute"),
            "and says why the cap is not used: {rendered}"
        );

        // A composed reservation is charged verbatim — not the cap, and not a re-sum of tiers.
        let reserved = svc
            .derive_charge(
                &reserved_eligibility(3 << 30, 1 << 30),
                &generous,
                "trainer",
            )
            .expect("a composed reservation admits");
        assert_eq!(reserved.tiers.hard_accountable.device, 3 << 30);
        assert_eq!(reserved.tiers.hard_accountable.host, 1 << 30);
        assert_eq!(
            reserved.tiers.device_total(),
            3 << 30,
            "the reservation composed every term at its declared scope; summing the tiers again \
             would double-count a shared-scope term across co-resident roles"
        );
        assert_eq!(reserved.tiers.host_total(), 1 << 30);

        // A lower-minor module still reports its declared tiers, and that path is unchanged.
        let legacy = svc
            .derive_charge(&eligibility(2 << 30, 512 << 20), &generous, "trainer")
            .expect("the legacy declared claim still admits");
        assert_eq!(legacy.tiers.hard_accountable.device, 2 << 30);

        // And the reservation takes precedence when both are reported.
        let mut both = reserved_eligibility(3 << 30, 1 << 30);
        both.headroom
            .insert("claim_device_bytes".to_string(), 9 << 30);
        assert_eq!(
            svc.derive_charge(&both, &generous, "trainer")
                .unwrap()
                .tiers
                .hard_accountable
                .device,
            3 << 30,
            "the composed reservation governs, not a legacy declaration beside it"
        );
    }

    fn policy(duty: u32) -> VhcPolicy {
        VhcPolicy {
            mode: VhcPolicyMode::Always,
            vram_cap_mb: 0,
            duty_cycle_pct: duty,
            schedule: None,
        }
    }

    /// A seat-claiming, coordinator-training node over `budget` — the single-peer trainer+coordinator
    /// shape (the ceremony's M4/Strix box in miniature).
    pub(super) fn coordinator_trainer_service(budget: OwnerBudget) -> VhcService {
        VhcService::new(VhcServiceParts {
            config: VhcConfig {
                enabled: true,
                seat_claim: true,
                coordinator_trains: true,
                ..VhcConfig::default()
            },
            store: VhcStore::open_in_memory().unwrap(),
            worker: Arc::new(NoopWorker),
            feed: None,
            discovery: None,
            budget: Some(budget),
            worker_factory: Some(Arc::new(|| Arc::new(NoopWorker) as Arc<dyn WorkerControl>)),
            identity_dir: None,
            seat_directory: None,
        })
    }

    pub(super) fn instance_id(role: &str, instance: u64) -> RoleInstanceId {
        RoleInstanceId {
            run_id: [0x5A; 32],
            epoch: 0,
            role: role.to_string(),
            instance,
        }
    }

    /// The DEFAULT finite owner budget the live node derives (`OwnerBudget::from_config` with the
    /// probe, no `[vhc.owner_budget]` configured) grants exactly ONE full accelerator-duty
    /// (`duty_pct = 100`) — the ledger the M4 self-coordinated run ran under.
    #[test]
    fn default_owner_budget_grants_one_full_accelerator_duty() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        assert_eq!(
            budget.duty_pct, 100,
            "the default duty ledger is one full accelerator-duty"
        );
    }

    /// The seat-holding node admits BOTH role-instances under the default 100% duty ledger: the
    /// coordinator/consensus seat instance claims ZERO accelerator duty (it runs only the consensus
    /// wasm), so its co-located trainer's full-duty claim still fits. Pre-fix the coordinator also
    /// claimed the policy's 100% duty and exhausted the ledger, refusing its own trainer
    /// (`duty cycle exhausted: requested 100%, remaining 0%`) — the exact M4 fleet-smoke STOP.
    #[test]
    fn seat_holder_admits_coordinator_and_co_located_trainer_under_default_duty_ledger() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // The coordinator seat role-instance (role == seat_role): consensus wasm, no device
        // footprint, ZERO accelerator duty by design.
        let coord_charge = svc
            .derive_charge(&eligibility(0, 64 << 20), &policy(100), "coordinator")
            .unwrap();
        assert_eq!(
            coord_charge.duty_pct, 0,
            "the consensus coordinator claims zero accelerator duty"
        );
        svc.admit_placed(&instance_id("coordinator", 1), coord_charge, 100)
            .expect("the coordinator seat instance admits");

        // The co-located trainer role-instance: the policy duty (full), a real device+host claim.
        let trainer_charge = svc
            .derive_charge(&eligibility(5 << 30, 8 << 30), &policy(100), "trainer")
            .unwrap();
        assert_eq!(
            trainer_charge.duty_pct, 100,
            "the co-located trainer claims the policy duty"
        );
        svc.admit_placed(&instance_id("trainer", 2), trainer_charge, 100)
            .expect(
                "the co-located trainer admits under the SAME finite duty ledger the live node runs",
            );

        // Both live; the duty ledger is exactly spent (coordinator 0% + trainer 100% == 100%).
        let snap = svc.arbiter().remaining();
        assert_eq!(snap.instances, 2, "both role-instances are reserved");
        assert_eq!(
            snap.duty_pct, 0,
            "the trainer consumed the whole duty ledger"
        );
    }

    /// An admitted role-instance whose LIVE ATTACH was refused must give its duty back. The worker
    /// reports such a refusal as that generation's terminal (`FailedRetryable` — the fixed worker
    /// emits one when a plane cannot be brought up), and the node's terminal edge releases the
    /// ledger. Pre-fix a refused attach produced only a classified `Error` event: no terminal, no
    /// release, no retry — the admitted-instance record lived on holding 100% duty, which is how
    /// the Windows box's superseded runs rehydrated on boot and refused every new admission until
    /// the operator wiped its run state.
    #[test]
    fn a_refused_live_attach_releases_the_instances_duty() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // The co-located trainer is admitted (full duty) and live.
        let id = instance_id("trainer", 8);
        let charge = svc
            .derive_charge(&eligibility(2 << 30, 2 << 30), &policy(100), "trainer")
            .unwrap();
        svc.admit_placed(&id, charge, 100).expect("trainer admits");
        svc.co_trainers.lock().unwrap().insert(
            "run-a".to_string(),
            InstanceEntry {
                id,
                generation: 8,
                worker: Arc::new(NoopWorker),
            },
        );
        assert_eq!(svc.arbiter.remaining().duty_pct, 0, "the ledger is spent");

        // Its worker could not bring up its transport: the typed, generation-stamped terminal.
        svc.handle_worker_event(&protocol::Event::RunTerminated {
            run_id: "run-a".to_string(),
            generation: 8,
            outcome: protocol::TerminalOutcome::FailedRetryable {
                reason: "iroh plane: iroh gossip plane did not come up within 60s".to_string(),
            },
        })
        .expect("terminal handled");

        assert_eq!(
            svc.arbiter.remaining().duty_pct,
            100,
            "a refused attach must return its duty to the ledger, not hold it until a state wipe"
        );
        assert_eq!(svc.arbiter.remaining().instances, 0);
    }

    /// The negative: over-budget refusal still works — a SECOND full-duty (trainer) role-instance
    /// on the same box exceeds the 100% duty ledger and is refused TYPED (`DutyExhausted`), so the
    /// fix widens admission for the legitimate coordinator+trainer pair without disabling the
    /// ledger's protection.
    #[test]
    fn over_budget_duty_is_still_refused_typed() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // Coordinator (0%) + one full-duty trainer (100%) fill the ledger.
        let coord = svc
            .derive_charge(&eligibility(0, 64 << 20), &policy(100), "coordinator")
            .unwrap();
        svc.admit_placed(&instance_id("coordinator", 1), coord, 100)
            .expect("coordinator admits");
        let trainer = svc
            .derive_charge(&eligibility(2 << 30, 2 << 30), &policy(100), "trainer")
            .unwrap();
        svc.admit_placed(&instance_id("trainer", 2), trainer, 100)
            .expect("first trainer admits");

        // A second full-duty trainer has no duty left — a typed refusal, never a silent admit.
        let extra = svc
            .derive_charge(&eligibility(2 << 30, 2 << 30), &policy(100), "trainer")
            .unwrap();
        let refusal = svc
            .admit_placed(&instance_id("trainer", 3), extra, 100)
            .expect_err("a second full-duty trainer exceeds the 100% duty ledger");
        assert!(
            matches!(
                refusal,
                AdmitRefusal::DutyExhausted {
                    requested: 100,
                    remaining: 0
                }
            ),
            "expected a typed duty-exhausted refusal, got {refusal:?}"
        );
    }
}

/// Defect — the co-located role-instance iroh **bind collision** (the M4 + Windows fleet-smoke
/// blocker): with `coordinator_trains = true` both role-instances of one node authored their own
/// iroh endpoint on the node's single pinned `[vhc.iroh] bind_port`, so the second one either
/// failed closed (`endpoint bind failed: Failed to bind sockets`, macOS) or hung before seed-init
/// (Windows). The node's iroh endpoint is a NODE-level singleton ([CI-10]: one `identity/iroh.key`,
/// one reachability record per run), so exactly one role-instance per `(node, run)` may own it and
/// its co-located siblings attach WS-only. These tests pin that ownership contract.
#[cfg(test)]
mod iroh_endpoint_tests {
    use super::arbitration_tests::{coordinator_trainer_service, instance_id, NoopWorker};
    use super::*;

    /// Register `role`/`incarnation` as the run's live PRIMARY role-instance (what a completed
    /// join does) so the ownership liveness check has something to observe.
    fn live_primary(svc: &VhcService, run: &str, role: &str, incarnation: u64) {
        svc.instances.lock().unwrap().insert(
            run.to_string(),
            InstanceEntry {
                id: instance_id(role, incarnation),
                generation: incarnation,
                worker: Arc::new(NoopWorker),
            },
        );
    }

    /// The ceremony's Strix/M4 shape: the seat instance authors first and owns the node's endpoint
    /// (hence the pinned `bind_port`); its co-located trainer sibling does NOT get a second
    /// endpoint while the owner is live, so the two never contend for the same socket.
    #[test]
    fn a_co_located_sibling_never_authors_a_second_iroh_endpoint() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        assert!(
            svc.claim_node_iroh_endpoint("run-a", "coordinator", 7),
            "the first (seat) instance takes the node's endpoint"
        );
        assert!(
            svc.claim_node_iroh_endpoint("run-a", "coordinator", 7),
            "re-authorship within the incarnation keeps it (the published record's socket stands)"
        );
        live_primary(&svc, "run-a", "coordinator", 7);
        assert!(
            !svc.claim_node_iroh_endpoint("run-a", "trainer", 8),
            "the co-located trainer must attach WS-only, never bind the node's port twice"
        );
        // Ownership is per RUN: the same node joining another run authors that run's endpoint.
        assert!(svc.claim_node_iroh_endpoint("run-b", "trainer", 8));
    }

    /// Ownership is self-healing: once the owning instance is gone (terminal / leave / a
    /// reconvergence that minted a new incarnation), the next authorship takes the endpoint over —
    /// a dead owner can never strand the node's reachability.
    #[test]
    fn a_dead_owner_hands_the_endpoint_to_the_next_authorship() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        assert!(svc.claim_node_iroh_endpoint("run-a", "coordinator", 7));
        live_primary(&svc, "run-a", "coordinator", 7);
        assert!(!svc.claim_node_iroh_endpoint("run-a", "trainer", 8));

        // The owner's instance is reaped (the terminal edge removed it).
        svc.instances.lock().unwrap().remove("run-a");
        assert!(
            svc.claim_node_iroh_endpoint("run-a", "trainer", 8),
            "an owner that is no longer live releases the node's endpoint"
        );
        live_primary(&svc, "run-a", "trainer", 8);

        // The coordinator rejoining at a NEW incarnation finds the endpoint held by a live owner.
        assert!(!svc.claim_node_iroh_endpoint("run-a", "coordinator", 9));

        // Leaving the run drops the bookkeeping outright.
        svc.forget_node_iroh_endpoint("run-a");
        assert!(svc.claim_node_iroh_endpoint("run-a", "coordinator", 9));
    }

    /// A co-located sibling is recognised as a live owner too (the endpoint may be owned by the
    /// trainer half of the pair — whichever authored first), so the coordinator does not then bind
    /// a second socket behind its back.
    #[test]
    fn a_co_trainer_owner_is_honored_as_live() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        assert!(svc.claim_node_iroh_endpoint("run-a", "trainer", 4));
        svc.co_trainers.lock().unwrap().insert(
            "run-a".to_string(),
            InstanceEntry {
                id: instance_id("trainer", 4),
                generation: 4,
                worker: Arc::new(NoopWorker),
            },
        );
        assert!(!svc.claim_node_iroh_endpoint("run-a", "coordinator", 5));
    }
}

#[cfg(test)]
mod storage_gate_tests {
    use super::arbitration_tests::{coordinator_trainer_service, instance_id, NoopWorker};
    use super::*;
    use daemon_api::{VhcEligibility, VhcPolicy, VhcPolicyMode};

    fn seed_joined_run(svc: &VhcService, run: &str, generation: u64) {
        svc.store
            .put_join_intent(
                run,
                "https://coord.local/vhc",
                &VhcPolicy {
                    mode: VhcPolicyMode::Idle,
                    vram_cap_mb: 8_000,
                    duty_cycle_pct: 90,
                    schedule: None,
                },
                None,
                &VhcEligibility::default(),
            )
            .expect("seed the joined intent");
        svc.instances.lock().unwrap().insert(
            run.to_string(),
            InstanceEntry {
                id: instance_id("trainer", generation),
                generation,
                worker: Arc::new(NoopWorker),
            },
        );
    }

    /// A `FailedStorage` terminal (host storage exhausted — the typed taxonomy) transitions the
    /// row `failed_retryable` WITH the storage gate set, schedules the next free-space check,
    /// and consumes NO retry budget: a full disk is a capacity condition to wait out, never a
    /// crash loop to escalate.
    #[test]
    fn a_storage_terminal_gates_the_row_and_spends_no_budget() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-s", 3);

        svc.handle_run_terminated(
            "run-s",
            3,
            &protocol::TerminalOutcome::FailedStorage {
                reason: "host storage exhausted: journal append".into(),
            },
        )
        .expect("the terminal transitions");

        let row = svc.store.get_run("run-s").expect("read").expect("row");
        assert_eq!(row.run_state, RunState::FailedRetryable);
        assert!(row.storage_gated, "the gate is durably marked");
        assert_eq!(
            row.retry_count, 0,
            "a storage wait never consumes the retry budget"
        );
        assert!(
            row.next_retry_ms.is_some(),
            "the next free-space check is scheduled"
        );
        // The identity survives (failed_retryable is not identity-terminal) and repeated
        // storage terminals can never escalate: run it past the ordinary budget.
        for _ in 0..svc.config.retry.max_retries + 2 {
            seed_joined_run(&svc, "run-s", 3);
            svc.store.set_storage_gated("run-s", true).unwrap();
            svc.handle_run_terminated(
                "run-s",
                3,
                &protocol::TerminalOutcome::FailedStorage {
                    reason: "still full".into(),
                },
            )
            .expect("transitions again");
        }
        let row = svc.store.get_run("run-s").expect("read").expect("row");
        assert_eq!(
            row.run_state,
            RunState::FailedRetryable,
            "storage exhaustion never escalates to failed_terminal"
        );
    }

    /// An ordinary retryable terminal is unchanged by the taxonomy: budget consumed, no gate.
    #[test]
    fn an_ordinary_retryable_terminal_is_ungated_and_budgeted() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-r", 5);

        svc.handle_run_terminated(
            "run-r",
            5,
            &protocol::TerminalOutcome::FailedRetryable {
                reason: "transport loss".into(),
            },
        )
        .expect("the terminal transitions");

        let row = svc.store.get_run("run-r").expect("read").expect("row");
        assert_eq!(row.run_state, RunState::FailedRetryable);
        assert!(!row.storage_gated);
        assert_eq!(row.retry_count, 1, "the ordinary path consumes budget");
    }

    /// The gate's free-space check: disabled (`reserve_mb = 0`) and unprobeable (no identity
    /// dir) both read OPEN — the interim gate is best-effort and must never wedge a node it
    /// cannot measure; an unmeetable floor over a real path reads CLOSED.
    #[test]
    fn the_free_space_check_opens_and_closes_on_the_reserve_floor() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        assert!(svc.storage_gate_open(), "no identity dir: nothing to probe");

        let dir = tempfile::tempdir().expect("tempdir");
        svc.identity_dir = Some(dir.path().to_path_buf());
        assert!(
            svc.storage_gate_open(),
            "the default floor is satisfiable on a working filesystem"
        );

        svc.config.storage.reserve_mb = u64::MAX;
        assert!(
            !svc.storage_gate_open(),
            "an unmeetable floor holds the gate"
        );

        svc.config.storage.reserve_mb = 0;
        assert!(svc.storage_gate_open(), "reserve 0 disables the gate");
    }
}
