// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`SwarmService`] — the resident node-side swarm-training service (spec §10.3/§10.4).
//!
//! It owns a worker-control seam ([`WorkerControl`], implemented for `daemon-vhc-supervisor`'s
//! `TrainSupervisor`), the durable [`SwarmStore`] (`swarm.db`), and a broadcast of [`SwarmEvent`]s.
//! It:
//!
//! - Translates worker [`protocol::Event`]s into [`SwarmEvent`]s, persists them to the windowed log, folds contribution counters, broadcasts to `swarm_subscribe`, and emits a payload-free [`NodeEvent::SwarmChanged`] pointer onto the node feed (§10.4).
//! - Drives **durable-intent re-convergence** on [`start`](SwarmService::start): re-issues `JoinRun` for every persisted active join-intent so a restart rejoins without app involvement (§10.3).
//! - Is **OFF by default** (`[swarm] enabled = false`): a disabled service never touches the worker, so no training worker is ever spawned unless swarm is enabled.
//! - Implements [`SwarmApi`], mapping requests → worker commands + store reads (eligibility is node-computed from the worker probe/assess and mirrored, ADR-003 — the app never re-derives it).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    ApiError, NodeEvent, SwarmApi, SwarmCapabilities, SwarmEligibility, SwarmEvent,
    SwarmEventStream, SwarmHardwareReport, SwarmLeaveMode, SwarmPolicy, SwarmPolicyMode,
    SwarmRunDetail, SwarmRunSummary,
};
use daemon_vhc_session::config::SwarmConfig;
use daemon_vhc_session::protocol::{
    self, Eligibility, Hardware, JoinPolicy, LeaveMode, PolicyMode,
};
use daemon_vhc_supervisor::TrainSupervisor;
use futures::StreamExt;
use std::collections::BTreeMap;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::arbiter::{
    AdmitRefusal, ClaimTiers, InstanceCharge, OwnerArbiter, OwnerBudget, RoleInstanceId, TierBytes,
};
use crate::discovery::RunDiscovery;
use crate::store::{DesiredState, PersistedRun, StoreError, SwarmStore, EVENT_WINDOW};

/// A node-feed sink: the node passes a closure over `NodeEventFeed::emit` so live swarm updates ride
/// the existing `events_subscribe` channel as `SwarmChanged` pointers (no new transport).
pub type NodeFeed = Arc<dyn Fn(NodeEvent) + Send + Sync>;

/// A per-role-instance worker factory (Phase E multi-instance supervision, decisions D1/D6):
/// when configured, every admitted join gets its **own** supervised child (one sandbox = one
/// role-instance) instead of sharing the single default worker. In production this spawns a
/// fresh `TrainSupervisor`; tests hand out recording fakes.
pub type WorkerFactory = Arc<dyn Fn() -> Arc<dyn WorkerControl> + Send + Sync>;

/// A swarm-service error.
#[derive(Debug, thiserror::Error)]
pub enum SwarmError {
    /// A `swarm.db` error.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// A worker-control failure (mapped from the supervisor).
    #[error("worker: {0}")]
    Worker(String),
    /// A run-discovery / envelope-fetch failure (registry unreachable, run unknown, envelope hash
    /// mismatch — the §6.1/§6.5 join-time discovery seam).
    #[error("discovery: {0}")]
    Discovery(String),
    /// The owner arbiter refused the join: an aggregate resource ledger is exhausted (decisions
    /// D6 — the admission funnel's last, supreme stage; the owner can always refuse).
    #[error("owner arbitration refused the join: {0}")]
    Resources(#[from] AdmitRefusal),
    /// The swarm service is disabled (`[swarm] enabled = false`).
    #[error("swarm is disabled")]
    Disabled,
}

impl SwarmError {
    fn worker(e: impl std::fmt::Display) -> Self {
        Self::Worker(e.to_string())
    }

    fn to_api(&self) -> ApiError {
        match self {
            SwarmError::Disabled => ApiError::Unsupported("swarm is disabled".into()),
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
    async fn probe(&self) -> Result<Hardware, SwarmError>;
    /// Assess a run envelope against effective resources (§6.5) — the eligibility source.
    async fn assess(&self, envelope: Vec<u8>) -> Result<Eligibility, SwarmError>;
    /// Join a run.
    async fn join(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<(), SwarmError>;
    /// Join a run and return the **continuous** worker event stream (A3 event pump). The default
    /// delegates to [`join`](Self::join) and returns an already-closed receiver, so test fakes and
    /// non-streaming workers keep the pre-A3 behavior; `TrainSupervisor` overrides it with the real
    /// per-round stream.
    async fn join_streaming(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<protocol::Event>, SwarmError> {
        self.join(run_id, coordinator, credentials, policy).await?;
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        Ok(rx)
    }
    /// Leave a run.
    async fn leave(&self, run_id: String, mode: LeaveMode) -> Result<(), SwarmError>;
    /// Push a GPU-governor throttle lever (§10.5).
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), SwarmError>;
    /// Tear the worker down (a factory child that was refused admission or whose run left —
    /// Phase E multi-instance supervision). Default no-op for fakes/shared workers.
    async fn shutdown(&self) {}
}

#[async_trait]
impl WorkerControl for TrainSupervisor {
    async fn probe(&self) -> Result<Hardware, SwarmError> {
        TrainSupervisor::probe(self)
            .await
            .map_err(SwarmError::worker)
    }
    async fn assess(&self, envelope: Vec<u8>) -> Result<Eligibility, SwarmError> {
        TrainSupervisor::assess(self, envelope)
            .await
            .map_err(SwarmError::worker)
    }
    async fn join(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<(), SwarmError> {
        TrainSupervisor::join(self, run_id, coordinator, credentials, policy)
            .await
            .map_err(SwarmError::worker)
    }
    async fn join_streaming(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<protocol::Event>, SwarmError> {
        TrainSupervisor::join_streaming(self, run_id, coordinator, credentials, policy)
            .await
            .map_err(SwarmError::worker)
    }
    async fn leave(&self, run_id: String, mode: LeaveMode) -> Result<(), SwarmError> {
        TrainSupervisor::leave(self, run_id, mode)
            .await
            .map_err(SwarmError::worker)
    }
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), SwarmError> {
        TrainSupervisor::throttle(self, vram_cap_mb, duty_cycle_pct, paused)
            .await
            .map_err(SwarmError::worker)
    }
    async fn shutdown(&self) {
        TrainSupervisor::shutdown(self).await;
    }
}

/// Construction parts for a [`SwarmService`].
pub struct SwarmServiceParts {
    /// The `[swarm]` config (spec §10.6); `enabled` gates all worker activity.
    pub config: SwarmConfig,
    /// The durable `swarm.db` store.
    pub store: SwarmStore,
    /// The default worker-control seam (a real `TrainSupervisor` in production) — the probe
    /// surface, and the shared child when no [`SwarmServiceParts::worker_factory`] is set.
    pub worker: Arc<dyn WorkerControl>,
    /// The node-feed sink for `SwarmChanged` pointers (`None` on a headless / test build).
    pub feed: Option<NodeFeed>,
    /// The run-discovery seam (A1). When present, `swarm_join` discovers the run + fetches the frozen
    /// envelope + runs the worker's real §6.5 `AssessRun` before `JoinRun`. `None` keeps the W1
    /// probe-based eligibility path (no coordinator configured), so the service stays usable offline.
    pub discovery: Option<Arc<dyn RunDiscovery>>,
    /// The owner's aggregate resource grants (decisions D6). `None` = permissive
    /// ([`OwnerBudget::unbounded`]) — arbitration still keys/tracks every instance, it just
    /// never runs out.
    pub budget: Option<OwnerBudget>,
    /// The per-role-instance worker factory (Phase E N-sandbox supervision). `None` = every run
    /// shares the single default worker (the pre-E single-child behavior).
    pub worker_factory: Option<WorkerFactory>,
}

/// One live supervised role-instance: its ledger identity and its worker child.
struct InstanceEntry {
    id: RoleInstanceId,
    worker: Arc<dyn WorkerControl>,
}

/// The node-side swarm-training service.
pub struct SwarmService {
    config: SwarmConfig,
    store: SwarmStore,
    worker: Arc<dyn WorkerControl>,
    discovery: Option<Arc<dyn RunDiscovery>>,
    /// The D6 owner arbiter: every join is admitted against the aggregate typed ledgers before
    /// any child is touched, and released only on observed teardown.
    arbiter: OwnerArbiter,
    /// Per-role-instance children (Phase E N-sandbox supervision), keyed by `RunLabel`.
    instances: Mutex<BTreeMap<String, InstanceEntry>>,
    worker_factory: Option<WorkerFactory>,
    events_tx: broadcast::Sender<SwarmEvent>,
    feed: Option<NodeFeed>,
    /// The run the worker is currently on (from the last `RunPhase`), used to attribute events that
    /// don't carry a run id (`RoundProgress`/`RoundOutcome`/…).
    current_run: Mutex<Option<String>>,
    /// The coalescing swarm-feed revision stamped on each `SwarmChanged` pointer.
    rev: AtomicU64,
    /// The service's own `Arc` handle (A3), bound post-construction via [`bind_self`](Self::bind_self)
    /// so `swarm_join`/`start` can spawn a detached event-pump task that outlives the `&self` call.
    /// Unbound (test builds) → the non-streaming `join` path, drained-and-dropped.
    me: std::sync::OnceLock<std::sync::Weak<SwarmService>>,
}

impl SwarmService {
    /// Build a service. The worker is never touched until [`start`](Self::start) / an API call, and
    /// only when `config.enabled`.
    pub fn new(parts: SwarmServiceParts) -> Self {
        let (events_tx, _) = broadcast::channel(1024);
        Self {
            config: parts.config,
            store: parts.store,
            worker: parts.worker,
            discovery: parts.discovery,
            arbiter: OwnerArbiter::new(parts.budget.unwrap_or_else(OwnerBudget::unbounded)),
            instances: Mutex::new(BTreeMap::new()),
            worker_factory: parts.worker_factory,
            events_tx,
            feed: parts.feed,
            current_run: Mutex::new(None),
            rev: AtomicU64::new(0),
            me: std::sync::OnceLock::new(),
        }
    }

    /// The owner arbiter (observability / tests): remaining ledgers + live instance count.
    pub fn arbiter(&self) -> &OwnerArbiter {
        &self.arbiter
    }

    /// Bind the service's own `Arc` handle (A3 event pump), mirroring the node's `set_swarm`
    /// post-`Arc` binder. After this, `swarm_join` / `start` drive `join_streaming` + a detached pump
    /// task feeding the worker's continuous event stream into [`handle_worker_event`](Self::handle_worker_event)
    /// → `NodeEvent::SwarmChanged`, so `swarm.db` reflects live round progression (§10.3/§10.4).
    /// Idempotent; never bound → the non-streaming join path (unchanged, for tests).
    pub fn bind_self(self: &Arc<Self>) {
        let _ = self.me.set(Arc::downgrade(self));
    }

    /// Join a run and pump its continuous worker event stream into the service (A3). The public entry
    /// the boot site / e2e use to drive a **live-attach** join with authored `JoinCredentials`; the
    /// pump feeds each event through `handle_worker_event`. Requires [`bind_self`](Self::bind_self).
    pub async fn join_and_pump(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<(), SwarmError> {
        let rx = self
            .worker
            .join_streaming(run_id, coordinator, credentials, policy)
            .await?;
        self.spawn_pump(rx);
        Ok(())
    }

    /// Spawn the detached event-pump task that drains a worker event stream into
    /// [`handle_worker_event`](Self::handle_worker_event). When the service is unbound (tests) it
    /// drains-and-drops so a streaming worker never backs up.
    fn spawn_pump(&self, mut rx: tokio::sync::mpsc::UnboundedReceiver<protocol::Event>) {
        match self.me.get().and_then(std::sync::Weak::upgrade) {
            Some(me) => {
                tokio::spawn(async move {
                    while let Some(ev) = rx.recv().await {
                        // Best-effort fan-out: a persist error never stalls the pump (mirrors the
                        // existing "a broadcast send error only means no live subscribers" posture;
                        // the durable log + a SwarmChanged pointer let a client re-baseline).
                        let _ = me.handle_worker_event(&ev);
                    }
                });
            }
            None => {
                tokio::spawn(async move { while rx.recv().await.is_some() {} });
            }
        }
    }

    /// Whether swarm training is enabled.
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// The durable store (test/observability access).
    pub fn store(&self) -> &SwarmStore {
        &self.store
    }

    /// Start the service: **no-op when disabled** (the worker is never spawned). When enabled,
    /// re-issue `JoinRun` for every persisted active join-intent — durable-intent re-convergence, so
    /// a restart rejoins without app involvement (§10.3). Returns the number of runs re-joined.
    ///
    /// Re-convergence is also the arbiter's restart reconciliation for this supervision model:
    /// the children died with the node (stdio cut), so the fresh ledger is re-charged exactly by
    /// re-admitting each persisted intent (decisions D6 point 7 — the ledger converges to the
    /// genuinely-running set). A persisted intent the (possibly shrunk) owner budget no longer
    /// fits is surfaced LOUD as a persisted `SwarmEvent::Error` and skipped — one refused run
    /// never blocks the rest of re-convergence. The persisted incarnation is retained (a process
    /// restart retains the logical instance id, decisions D1); only a genuinely new
    /// role-instance mints a new one.
    pub async fn start(&self) -> Result<usize, SwarmError> {
        if !self.config.enabled {
            return Ok(0);
        }
        let intents = self.store.active_intents()?;
        let mut rejoined = 0;
        for run in &intents {
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
                instance: if run.instance > 0 {
                    run.instance
                } else {
                    self.store.mint_incarnation()?
                },
            };
            let charge = self.derive_charge(&run.eligibility, &run.policy);
            let priority = self.store.run_priority(&run.run_id)?;
            if let Err(refusal) = self.admit_placed(&id, charge, priority) {
                if self.worker_factory.is_some() {
                    worker.shutdown().await;
                }
                let mut emitted = Vec::new();
                let _ = self.emit(
                    SwarmEvent::Error {
                        run_id: run.run_id.clone(),
                        class: "owner_arbitration".to_string(),
                        detail: format!("re-convergence refused: {refusal}"),
                    },
                    &mut emitted,
                );
                continue;
            }
            if run.instance == 0 {
                self.store
                    .set_execution_identity(&run.run_id, id.epoch, &id.role, id.instance)?;
            }
            // A3: re-issue via the streaming path + pump so a re-converged run resumes reporting
            // live round progression into swarm.db (durable-intent re-convergence, §10.3).
            let rx = match worker
                .join_streaming(
                    run.run_id.clone(),
                    run.coordinator.clone(),
                    Vec::new(),
                    to_join_policy(&run.policy),
                )
                .await
            {
                Ok(rx) => rx,
                Err(e) => {
                    // No child came up — surrender the reservation (nothing to observe tear down).
                    self.arbiter.release(&id);
                    return Err(e);
                }
            };
            self.instances
                .lock()
                .unwrap()
                .insert(run.run_id.clone(), InstanceEntry { id, worker });
            self.spawn_pump(rx);
            rejoined += 1;
        }
        if rejoined > 0 {
            self.emit_changed(None);
        }
        Ok(rejoined)
    }

    /// The worker child for a new role-instance: a fresh factory child when configured (one
    /// sandbox = one role-instance, decisions D1), else the shared default worker.
    fn instance_worker(&self) -> Arc<dyn WorkerControl> {
        match &self.worker_factory {
            Some(f) => f(),
            None => self.worker.clone(),
        }
    }

    /// Derive a role-instance's ledger charge (decisions D6 point 3) from the node-computed
    /// eligibility. Both live eligibility sources reach this through **one** claim-shaped input
    /// contract (decisions D-10):
    ///
    /// - **v2 assess** (`eligibility_from_assess`, the discovery path): the claim funnel's verdict,
    ///   whose headroom carries `claim_device_bytes` / `claim_host_bytes` (bytes) — the disjoint
    ///   tier sums the worker's `admit_v2` computed. Charged verbatim onto the device + host tiers,
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
    fn derive_charge(
        &self,
        eligibility: &SwarmEligibility,
        policy: &SwarmPolicy,
    ) -> InstanceCharge {
        const MIB: u64 = 1 << 20;
        let claim = |key: &str| {
            eligibility
                .headroom
                .get(key)
                .copied()
                .filter(|v| *v > 0)
                .map(|v| v as u64)
        };
        let claim_host = claim("claim_host_bytes").unwrap_or(0);
        // The device claim (bytes) when present, else the owner's standing VRAM cap as the
        // estimate — NEVER a silent zero fall-through (D-10).
        let cap_bytes = u64::from(policy.vram_cap_mb).saturating_mul(MIB);
        let device_bytes = claim("claim_device_bytes").unwrap_or(cap_bytes);
        InstanceCharge {
            device: String::new(),
            tiers: ClaimTiers {
                hard_accountable: TierBytes {
                    device: device_bytes,
                    host: claim_host,
                },
                ..ClaimTiers::default()
            },
            disk_bytes: 0,
            net_up_bps: 0,
            net_down_bps: 0,
            duty_pct: policy.duty_cycle_pct.min(100) as u8,
        }
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
    /// the node"). Returns the [`SwarmEvent`]s emitted (0..2 per worker event). B3 wires the live
    /// worker event stream into this; W1 tests drive it directly.
    pub fn handle_worker_event(&self, ev: &protocol::Event) -> Result<Vec<SwarmEvent>, SwarmError> {
        // Track the current run + persist phase from a RunPhase.
        if let protocol::Event::RunPhase {
            run_id,
            phase,
            round,
            ..
        } = ev
        {
            *self.current_run.lock().unwrap() = Some(run_id.clone());
            self.store.set_phase(run_id, phase, *round)?;
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

        let mut emitted = Vec::new();
        if let Some(sev) = translate(ev, &run_id) {
            self.emit(sev, &mut emitted)?;
        }
        // A checkpoint is a contribution delta — surface the fresh totals as a Contribution event.
        if matches!(ev, protocol::Event::CheckpointPublished { .. }) {
            let contribution = self.store.get_contribution(&run_id)?;
            self.emit(
                SwarmEvent::Contribution {
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

    fn emit(&self, sev: SwarmEvent, out: &mut Vec<SwarmEvent>) -> Result<(), SwarmError> {
        self.store.append_event(&sev)?;
        // A send error only means "no live subscribers"; the durable log already has it.
        let _ = self.events_tx.send(sev.clone());
        out.push(sev);
        Ok(())
    }

    fn emit_changed(&self, run_id: Option<String>) {
        if let Some(feed) = &self.feed {
            let rev = self.rev.fetch_add(1, Ordering::SeqCst) + 1;
            feed(NodeEvent::SwarmChanged { run_id, rev });
        }
    }

    fn event_run_id(&self, ev: &protocol::Event) -> Option<String> {
        match ev {
            protocol::Event::RunPhase { run_id, .. } => Some(run_id.clone()),
            protocol::Event::RoundProgress { .. }
            | protocol::Event::RoundOutcome { .. }
            | protocol::Event::Metric { .. }
            | protocol::Event::CheckpointPublished { .. }
            | protocol::Event::Warning { .. }
            | protocol::Event::Error { .. } => self.current_run.lock().unwrap().clone(),
            _ => None,
        }
    }

    fn require_enabled(&self) -> Result<(), SwarmError> {
        if self.config.enabled {
            Ok(())
        } else {
            Err(SwarmError::Disabled)
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

    /// Resolve the `(coordinator, eligibility)` for a join (A1), against the role-instance's own
    /// `worker` (per-instance children run their own §6.5 assess).
    ///
    /// With a discovery seam: `GET /runs/:id` → fetch + blake3-verify the frozen envelope →
    /// `worker.assess(envelope)` (real §6.5), taking the coordinator from the registry. Without one:
    /// the W1 probe against the allowlisted coordinator. Eligibility is always node-computed.
    async fn resolve_join(
        &self,
        worker: &Arc<dyn WorkerControl>,
        run_id: &str,
    ) -> Result<(String, SwarmEligibility), SwarmError> {
        if let Some(discovery) = &self.discovery {
            let run = discovery
                .get_run(run_id)
                .await?
                .ok_or_else(|| SwarmError::Discovery(format!("run {run_id} not found")))?;
            let envelope = discovery.fetch_envelope(run_id).await?;
            let verdict = worker.assess(envelope).await?;
            Ok((run.coordinator, eligibility_from_assess(&verdict)))
        } else {
            let coordinator = self.coordinator();
            let eligibility = match worker.probe().await {
                Ok(hw) => eligibility_from_hardware(&hw),
                Err(_) => SwarmEligibility {
                    eligible: false,
                    reasons: vec!["worker probe failed".into()],
                    headroom: BTreeMap::new(),
                },
            };
            Ok((coordinator, eligibility))
        }
    }
}

#[async_trait]
impl SwarmApi for SwarmService {
    async fn swarm_run_list(&self) -> Result<Vec<SwarmRunSummary>, ApiError> {
        let runs = self
            .store
            .list_runs()
            .map_err(|e| SwarmError::from(e).to_api())?;
        Ok(runs.into_iter().map(run_summary).collect())
    }

    async fn swarm_run_detail(&self, run_id: String) -> Result<Option<SwarmRunDetail>, ApiError> {
        let map = |e: StoreError| SwarmError::from(e).to_api();
        let Some(run) = self.store.get_run(&run_id).map_err(map)? else {
            return Ok(None);
        };
        let contribution = self.store.get_contribution(&run_id).map_err(map)?;
        let recent_events = self
            .store
            .recent_events(&run_id, EVENT_WINDOW)
            .map_err(map)?;
        Ok(Some(SwarmRunDetail {
            coordinator: run.coordinator.clone(),
            summary: run_summary(run),
            contribution,
            recent_events,
        }))
    }

    async fn swarm_join(
        &self,
        run_id: String,
        policy: SwarmPolicy,
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

        // Node-computed eligibility (ADR-003). A1: when a discovery seam is configured, resolve the
        // run + fetch the frozen envelope + run the worker's real §6.5 `AssessRun` before `JoinRun`,
        // and take the coordinator endpoint from discovery. With no discovery configured, fall back
        // to the W1 probe-based eligibility against the allowlisted coordinator (offline / no-registry
        // path). Either way the persisted eligibility is node-computed — the app never re-derives it.
        let (coordinator, eligibility) = self
            .resolve_join(&worker, &run_id)
            .await
            .map_err(|e| e.to_api())?;

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
                    .map_err(|e| SwarmError::from(e).to_api())?;
                let (epoch, role, persisted_instance, run_hash) = persisted
                    .as_ref()
                    .filter(|r| r.desired_state == DesiredState::Joined)
                    .map_or((0, String::new(), 0, None), |r| {
                        (r.epoch, r.role.clone(), r.instance, r.run_id_hash)
                    });
                let id = RoleInstanceId {
                    // The cryptographic RunId when backfilled; a v1-era run keys its node-local
                    // ledger entry by blake3(RunLabel) until then (decisions D1 lazy backfill).
                    run_id: run_hash.unwrap_or_else(|| *blake3::hash(run_id.as_bytes()).as_bytes()),
                    epoch,
                    role: if role.is_empty() {
                        "trainer".to_string()
                    } else {
                        role
                    },
                    // A restart-re-join retains the logical incarnation; a genuinely new
                    // role-instance mints a never-reused one (decisions D1).
                    instance: if persisted_instance > 0 {
                        persisted_instance
                    } else {
                        self.store
                            .mint_incarnation()
                            .map_err(|e| SwarmError::from(e).to_api())?
                    },
                };
                let charge = self.derive_charge(&eligibility, &policy);
                let priority = self
                    .store
                    .run_priority(&run_id)
                    .map_err(|e| SwarmError::from(e).to_api())?;
                if let Err(refusal) = self.admit_placed(&id, charge, priority) {
                    if fresh_child {
                        worker.shutdown().await;
                    }
                    return Err(SwarmError::from(refusal).to_api());
                }
                id
            }
        };

        if let Err(e) =
            self.store
                .put_join_intent(&run_id, &coordinator, &policy, None, &eligibility)
        {
            if existing.is_none() {
                self.arbiter.release(&id);
                if fresh_child {
                    worker.shutdown().await;
                }
            }
            return Err(SwarmError::from(e).to_api());
        }
        let _ = self
            .store
            .set_execution_identity(&run_id, id.epoch, &id.role, id.instance);

        // A3: join over the streaming path + pump the continuous worker event stream into
        // `handle_worker_event` so swarm.db reflects live round progression (§10.3/§10.4). The
        // opaque `JoinRun.credentials` the worker's live attach parses (§2 of the A3 ledger) are
        // authored where the node identity + roster are known (the e2e / boot join_and_pump path);
        // an API-initiated join with no authored credentials keeps the worker's self-driven round
        // (WS-only baseline), still pumped.
        let rx = match worker
            .join_streaming(
                run_id.clone(),
                coordinator,
                Vec::new(),
                to_join_policy(&policy),
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
        self.instances
            .lock()
            .unwrap()
            .insert(run_id.clone(), InstanceEntry { id, worker });
        self.spawn_pump(rx);
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn swarm_leave(
        &self,
        run_id: String,
        mode: SwarmLeaveMode,
        _op_id: String,
    ) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        self.store
            .set_desired_state(&run_id, DesiredState::Left)
            .map_err(|e| SwarmError::from(e).to_api())?;
        let entry = self.instances.lock().unwrap().remove(&run_id);
        let worker = entry
            .as_ref()
            .map_or_else(|| self.worker.clone(), |e| e.worker.clone());
        worker
            .leave(run_id.clone(), to_leave_mode(mode))
            .await
            .map_err(|e| e.to_api())?;
        // Observed teardown → release (decisions D6 point 6): the leave has been accepted by the
        // child's serial command loop (which drops the run's wasm instance + device allocations
        // before servicing anything else), so the ledger entry is surrendered only NOW — never
        // optimistically before the victim gave the memory back.
        if let Some(e) = &entry {
            self.arbiter.release(&e.id);
        }
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn swarm_set_policy(&self, policy: SwarmPolicy) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        // W1: push the governor levers to the worker (§10.5) — and, under multi-instance
        // supervision, to every live role-instance child (the owner lever is host-wide). The
        // persisted default-policy slot for future joins is the config `[swarm].default_policy`;
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

    async fn swarm_hardware_report(&self) -> Result<SwarmHardwareReport, ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        let hw = self.worker.probe().await.map_err(|e| e.to_api())?;
        Ok(hardware_report(hw))
    }

    async fn swarm_subscribe(&self, run_id: Option<String>) -> Result<SwarmEventStream, ApiError> {
        let rx = self.events_tx.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(move |res| {
            let want = run_id.clone();
            async move {
                match res {
                    // Filter to one run when requested; drop `Lagged` gaps (the durable log + a
                    // SwarmChanged pointer let a lagging client re-baseline via run_detail).
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

// ---------------------------------------------------------------------------
// Wire<->worker mappings (the node is the single translation point)
// ---------------------------------------------------------------------------

fn run_summary(run: PersistedRun) -> SwarmRunSummary {
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
    let v2_identified = run.run_id_hash.is_some();
    SwarmRunSummary {
        run_id: run.run_id,
        phase: run.last_phase,
        joined,
        eligibility: run.eligibility,
        policy: if joined { Some(run.policy) } else { None },
        last_round: run.last_round,
        run_id_hash: run.run_id_hash.as_ref().map(hex),
        epoch: v2_identified.then_some(run.epoch),
        role: v2_identified.then(|| run.role.clone()),
        instance: v2_identified.then_some(run.instance),
        envelope_schema_major: Some(run.envelope_schema_major),
        module_abi_major: run.module_abi_major,
        selected_driver: run.selected_driver,
        module_hash: run.module_hash.as_ref().map(hex),
    }
}

fn to_policy_mode(mode: SwarmPolicyMode) -> PolicyMode {
    match mode {
        SwarmPolicyMode::Always => PolicyMode::Always,
        SwarmPolicyMode::Idle => PolicyMode::Idle,
        SwarmPolicyMode::Scheduled => PolicyMode::Scheduled,
        SwarmPolicyMode::Manual => PolicyMode::Manual,
    }
}

fn to_join_policy(p: &SwarmPolicy) -> JoinPolicy {
    JoinPolicy {
        mode: to_policy_mode(p.mode),
        vram_cap_mb: p.vram_cap_mb,
        duty_cycle_pct: p.duty_cycle_pct.min(100) as u8,
        schedule: p.schedule.clone(),
    }
}

fn to_leave_mode(mode: SwarmLeaveMode) -> LeaveMode {
    match mode {
        SwarmLeaveMode::Graceful => LeaveMode::Graceful,
        SwarmLeaveMode::Immediate => LeaveMode::Immediate,
    }
}

fn hardware_report(hw: Hardware) -> SwarmHardwareReport {
    SwarmHardwareReport {
        gpus: hw.gpus,
        vram_mb: hw.vram_mb,
        // A1 / wire v42: mirror the worker's unified-memory spillover (GTT) into the app-facing DTO
        // additively (the P1 Merge-2 recorded follow-on), so the GUI's "what can my GPU do" panel
        // shows the true effective budget on integrated/UMA boxes.
        shared_mb: hw.shared_mb,
        ram_mb: hw.ram_mb,
        backend_lanes: hw.backend_lanes,
        capabilities: SwarmCapabilities {
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

/// Map the worker's real §6.5 `AssessRun` verdict onto the app-facing eligibility DTO (A1). The
/// worker's `headroom` is an ordered `Vec<(String, i64)>`; the wire DTO is a `BTreeMap`. The app
/// renders this; it never re-derives eligibility (ADR-003).
fn eligibility_from_assess(e: &Eligibility) -> SwarmEligibility {
    SwarmEligibility {
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
fn eligibility_from_hardware(hw: &Hardware) -> SwarmEligibility {
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
    SwarmEligibility {
        eligible,
        reasons,
        headroom,
    }
}

fn translate(ev: &protocol::Event, run_id: &str) -> Option<SwarmEvent> {
    match ev {
        protocol::Event::RunPhase {
            phase,
            epoch,
            round,
            ..
        } => Some(SwarmEvent::Phase {
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
        } => Some(SwarmEvent::Progress {
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
            ..
        } => Some(SwarmEvent::RoundOutcome {
            run_id: run_id.to_string(),
            round: *round,
            committed: *committed,
            ingested: *ingested,
            stalled: *stalled,
        }),
        protocol::Event::Warning { class, detail } => Some(SwarmEvent::Warning {
            run_id: run_id.to_string(),
            class: class.clone(),
            detail: detail.clone(),
        }),
        protocol::Event::Error { class, detail } => Some(SwarmEvent::Error {
            run_id: run_id.to_string(),
            class: format!("{class:?}"),
            detail: detail.clone(),
        }),
        _ => None,
    }
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
