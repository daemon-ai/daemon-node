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
use crate::store::{DesiredState, PersistedRun, StoreError, VhcStore, EVENT_WINDOW};

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
    /// Join a run.
    async fn join(
        &self,
        run_id: String,
        coordinator: String,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError>;
    /// Join a run and return the **continuous** worker event stream (A3 event pump). The default
    /// delegates to [`join`](Self::join) and returns an already-closed receiver, so test fakes and
    /// non-streaming workers keep the pre-A3 behavior; `TrainSupervisor` overrides it with the real
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
    /// The run-discovery seam (A1). When present, `vhc_join` discovers the run + fetches the frozen
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
}

/// The authored-join delivery: the tuple (incarnation stamped), the wire credentials bytes, and
/// the `credentials_ref` to persist.
type AuthoredDelivery = (Option<protocol::AdmittedTuple>, Vec<u8>, Option<String>);

/// One live supervised role-instance: its ledger identity and its worker child.
struct InstanceEntry {
    id: RoleInstanceId,
    worker: Arc<dyn WorkerControl>,
}

/// The node-side vhc-training service.
pub struct VhcService {
    config: VhcConfig,
    store: VhcStore,
    worker: Arc<dyn WorkerControl>,
    discovery: Option<Arc<dyn RunDiscovery>>,
    /// The D6 owner arbiter: every join is admitted against the aggregate typed ledgers before
    /// any child is touched, and released only on observed teardown.
    arbiter: OwnerArbiter,
    /// Per-role-instance children (Phase E N-sandbox supervision), keyed by `RunLabel`.
    instances: Mutex<BTreeMap<String, InstanceEntry>>,
    worker_factory: Option<WorkerFactory>,
    /// The identity keystore directory (D-P8 credential + per-run cert authorship); `None`
    /// disables node-side authorship (tests / headless).
    identity_dir: Option<std::path::PathBuf>,
    events_tx: broadcast::Sender<VhcEvent>,
    feed: Option<NodeFeed>,
    /// The run the worker is currently on (from the last `RunPhase`), used to attribute events that
    /// don't carry a run id (`RoundProgress`/`RoundOutcome`/…).
    current_run: Mutex<Option<String>>,
    /// The coalescing vhc-feed revision stamped on each `VhcChanged` pointer.
    rev: AtomicU64,
    /// The service's own `Arc` handle (A3), bound post-construction via [`bind_self`](Self::bind_self)
    /// so `vhc_join`/`start` can spawn a detached event-pump task that outlives the `&self` call.
    /// Unbound (test builds) → the non-streaming `join` path, drained-and-dropped.
    me: std::sync::OnceLock<std::sync::Weak<VhcService>>,
}

impl VhcService {
    /// Build a service. The worker is never touched until [`start`](Self::start) / an API call, and
    /// only when `config.enabled`.
    pub fn new(parts: VhcServiceParts) -> Self {
        let (events_tx, _) = broadcast::channel(1024);
        Self {
            config: parts.config,
            store: parts.store,
            worker: parts.worker,
            discovery: parts.discovery,
            arbiter: OwnerArbiter::new(parts.budget.unwrap_or_else(OwnerBudget::unbounded)),
            instances: Mutex::new(BTreeMap::new()),
            worker_factory: parts.worker_factory,
            identity_dir: parts.identity_dir,
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

    /// Bind the service's own `Arc` handle (A3 event pump), mirroring the node's `set_vhc`
    /// post-`Arc` binder. After this, `vhc_join` / `start` drive `join_streaming` + a detached pump
    /// task feeding the worker's continuous event stream into [`handle_worker_event`](Self::handle_worker_event)
    /// → `NodeEvent::VhcChanged`, so `vhc.db` reflects live round progression (§10.3/§10.4).
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
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), VhcError> {
        let rx = self
            .worker
            .join_streaming(run_id, coordinator, credentials, policy, admitted_tuple)
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
                        // the durable log + a VhcChanged pointer let a client re-baseline).
                        let _ = me.handle_worker_event(&ev);
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
    /// re-issue `JoinRun` for every persisted active join-intent — durable-intent re-convergence, so
    /// a restart rejoins without app involvement (§10.3). Returns the number of runs re-joined.
    ///
    /// Re-convergence is also the arbiter's restart reconciliation for this supervision model:
    /// the children died with the node (stdio cut), so the fresh ledger is re-charged exactly by
    /// re-admitting each persisted intent (decisions D6 point 7 — the ledger converges to the
    /// genuinely-running set). A persisted intent the (possibly shrunk) owner budget no longer
    /// fits is surfaced LOUD as a persisted `VhcEvent::Error` and skipped — one refused run
    /// never blocks the rest of re-convergence. The persisted incarnation is retained (a process
    /// restart retains the logical instance id, decisions D1); only a genuinely new
    /// role-instance mints a new one.
    pub async fn start(&self) -> Result<usize, VhcError> {
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
                    VhcEvent::Error {
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
            // Re-author identity + credentials for the retained incarnation (D-P8): the node
            // re-resolves + REFRESHES the per-run cert + credentials on every reconvergence (the
            // keystore recovers the same key within the incarnation; the certificate + credential
            // record are re-issued). The persisted admitted tuple carries the incarnation to run.
            let persisted_tuple = run
                .admitted_tuple
                .as_deref()
                .and_then(|b| protocol::decode::<protocol::AdmittedTuple>(b).ok());
            let restore = self.resolve_restore(&run.run_id).await;
            let (delivery_tuple, credentials, credentials_ref) = match self.author_join(
                &run.run_id,
                &run.coordinator,
                &id,
                persisted_tuple,
                restore,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.arbiter.release(&id);
                    return Err(e);
                }
            };
            // The credentials-record reference is deterministic per `(role, incarnation)`, so a
            // refresh reuses the persisted ref; only the tuple bytes (incarnation stamped) are
            // re-persisted here.
            let _ = &credentials_ref;
            if let Some(tuple) = &delivery_tuple {
                if let Ok(bytes) = protocol::encode(tuple) {
                    let _ = self.store.set_admitted_tuple(&run.run_id, &bytes);
                }
            }
            // A3: re-issue via the streaming path + pump so a re-converged run resumes reporting
            // live round progression into vhc.db (durable-intent re-convergence, §10.3).
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
    fn derive_charge(&self, eligibility: &VhcEligibility, policy: &VhcPolicy) -> InstanceCharge {
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
    /// the node"). Returns the [`VhcEvent`]s emitted (0..2 per worker event). B3 wires the live
    /// worker event stream into this; unit tests drive it directly.
    pub fn handle_worker_event(&self, ev: &protocol::Event) -> Result<Vec<VhcEvent>, VhcError> {
        // [CI-7] terminal cleanup: when a run instance reaches a terminal state that ends the run
        // identity (Completed / Left / FailedTerminal), delete its per-run key material,
        // certificates, and credentials record — no run identity outlives the run it was minted
        // for. FailedRetryable is NOT terminal for the identity (the reconvergence/rejoin path
        // reuses or supersedes it under the retry budget), so its material survives.
        // Instance-map removal + arbiter release beyond this are run-lifecycle management
        // concerns; this hook is scoped to secret custody. Idempotent (remove_run tolerates
        // absence).
        if let protocol::Event::RunTerminated {
            run_id, outcome, ..
        } = ev
        {
            let terminal = matches!(
                outcome,
                protocol::TerminalOutcome::Completed { .. }
                    | protocol::TerminalOutcome::Left { .. }
                    | protocol::TerminalOutcome::FailedTerminal { .. }
            );
            if terminal {
                if let Some(dir) = &self.identity_dir {
                    if let Ok(keystore) = daemon_vhc_session::keystore::VhcKeystore::open(dir) {
                        if let Err(e) = keystore.remove_run(run_id) {
                            let mut emitted = Vec::new();
                            let _ = self.emit(
                                VhcEvent::Warning {
                                    run_id: run_id.clone(),
                                    class: "identity_cleanup".to_string(),
                                    detail: format!("per-run key/credential cleanup failed: {e}"),
                                },
                                &mut emitted,
                            );
                        }
                    }
                }
            }
        }
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

        // Checkpoint-pointer publication (spec §9; lane R): the checkpoint DOCUMENT is already on
        // the payload plane (the session put it there); record the round → content-address
        // pointer at the registry so a late joiner can restore. Best-effort + detached (a pointer
        // is advisory; the joiner hash-verifies regardless, so an unknown size is 0).
        if let protocol::Event::CheckpointPublished { round, hash, .. } = ev {
            if let Some(discovery) = &self.discovery {
                let discovery = discovery.clone();
                let (run, round, hash) = (run_id.clone(), *round, hash.clone());
                tokio::spawn(async move {
                    let _ = discovery.publish_checkpoint(&run, round, &hash, 0).await;
                });
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
            | protocol::Event::AdmittedTupleMismatch { run_id, .. } => Some(run_id.clone()),
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

    /// Resolve the `(coordinator, eligibility)` for a join (A1), against the role-instance's own
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
    fn author_join(
        &self,
        run_label: &str,
        coordinator: &str,
        id: &RoleInstanceId,
        tuple: Option<protocol::AdmittedTuple>,
        restore: Option<protocol::CheckpointRestore>,
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
        let authored = crate::credentials::author_join(
            &keystore,
            &crate::credentials::RunInstanceIdentity {
                run_label,
                genesis_hash: tuple.genesis_hash,
                epoch: id.epoch,
                role: &id.role,
                incarnation: id.instance,
                module_hash: tuple.module_hash,
            },
            coordinator,
            &self.config.registry,
            restore,
        )?;
        Ok((Some(tuple), authored.wire, authored.credentials_ref))
    }

    /// Resolve the late-join checkpoint restore for a run (spec §9): the registry's latest
    /// checkpoint pointer, decoded to the wire restore form (`None` = fresh start / no discovery
    /// / no checkpoint published). A malformed pointer hash is dropped (fresh start), never a
    /// hard join failure.
    async fn resolve_restore(&self, run_id: &str) -> Option<protocol::CheckpointRestore> {
        let discovery = self.discovery.as_ref()?;
        let pointer = discovery.fetch_checkpoint(run_id).await.ok()??;
        let hash = hex32(&pointer.hash)?;
        Some(protocol::CheckpointRestore {
            round: pointer.round,
            hash,
        })
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
        Ok(Some(VhcRunDetail {
            coordinator: run.coordinator.clone(),
            summary: run_summary(run),
            contribution,
            recent_events,
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

        // Node-computed eligibility (ADR-003). A1: when a discovery seam is configured, resolve the
        // run + fetch the frozen envelope + run the worker's real §6.5 `AssessRun` before `JoinRun`,
        // and take the coordinator endpoint from discovery. With no discovery configured, fall back
        // to the probe-based eligibility against the allowlisted coordinator (offline / no-registry
        // path). Either way the persisted eligibility is node-computed — the app never re-derives it.
        let (coordinator, eligibility, assessed_tuple) = self
            .resolve_join(&worker, &run_id, None)
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
                    .map_err(|e| VhcError::from(e).to_api())?;
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
                            .map_err(|e| VhcError::from(e).to_api())?
                    },
                };
                let charge = self.derive_charge(&eligibility, &policy);
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
        let restore = self.resolve_restore(&run_id).await;
        let (delivery_tuple, credentials, credentials_ref) =
            match self.author_join(&run_id, &coordinator, &id, assessed_tuple, restore) {
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

        // A3: join over the streaming path + pump the continuous worker event stream into
        // `handle_worker_event` so vhc.db reflects live round progression (§10.3/§10.4). The
        // opaque `JoinRun.credentials` the worker's live attach parses (§2 of the A3 ledger) are
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
        self.instances
            .lock()
            .unwrap()
            .insert(run_id.clone(), InstanceEntry { id, worker });
        self.spawn_pump(rx);
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
        self.store
            .set_desired_state(&run_id, DesiredState::Left)
            .map_err(|e| VhcError::from(e).to_api())?;
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
        // A1 / wire v42: mirror the worker's unified-memory spillover (GTT) into the app-facing DTO
        // additively (the P1 Merge-2 recorded follow-on), so the GUI's "what can my GPU do" panel
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

/// Map the worker's real §6.5 `AssessRun` verdict onto the app-facing eligibility DTO (A1). The
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
            ..
        } => Some(VhcEvent::RoundOutcome {
            run_id: run_id.to_string(),
            round: *round,
            committed: *committed,
            ingested: *ingested,
            stalled: *stalled,
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
