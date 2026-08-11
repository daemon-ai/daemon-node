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
    ApiError, NodeEvent, VhcApi, VhcCapabilities, VhcDiskScope, VhcDiskUsage, VhcDiskWipeOutcome,
    VhcEligibility, VhcEvent, VhcEventStream, VhcHardwareReport, VhcLeaveMode, VhcPolicy,
    VhcPolicyMode, VhcRunDetail, VhcRunSummary,
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
    /// The run-state root (`<data_dir>/vhc/runs` — the same path the node exports to workers as
    /// `DAEMON_VHC_RUN_DIR`/`DAEMON_VHC_CUSTODY_ROOT`). `Some` ⇒ the node opens the root's disk
    /// custodian for resume authorization (the storage gate) + usage reporting. `None` (tests /
    /// headless) leaves storage-gated runs to the bare free-space fallback.
    pub run_dir: Option<std::path::PathBuf>,
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
/// The durable-checkpoint lag alarm threshold (Gate B'): warn when a trainer generation's
/// freshest DURABLE checkpoint (its restore fence) trails the live head by three quarters of
/// the retained-record horizon. Sizing: the tolerated steady-state fence lag is one entirely
/// missed publisher cycle plus the in-flight upload (`2 × cadence + upload slack` — c15h
/// measured the by-ref family walk at 14–25 min ≈ 1–2 rounds), which the
/// `2 × cadence ≤ horizon` authoring rule keeps at or under half the horizon; three quarters
/// is therefore past every healthy shape while still leaving a quarter of the horizon to act
/// before a rejoin needs archive catch-up (defect 14's wedge was exactly this drift, unvoiced).
const CHECKPOINT_LAG_WARN_ROUNDS: u64 = daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS / 4 * 3;

/// The REL-5 stall-warning floor (reliability spec §6): the minimum no-committed-progress age
/// before a joined, alive run is announced `run_stalled`. The per-run threshold is
/// `max(floor, 2 × the slowest inter-commit gap observed on THIS run)` — adaptive per run
/// because round walls span seconds to many minutes across authored runs, and a healthy live
/// checkpoint publication (measured 14–25 min, c15h) stretches one inter-round gap that the
/// adaptive half then covers. Ten minutes absorbs the slow tail seen in C2 while still
/// beating the multi-hour operator-response delays the warning exists to end (RQ-4: the
/// derivation is validated against a checkpoint-heavy run before these defaults freeze; an
/// AUTHORED round wall in the run document remains future vocabulary).
const RUN_STALL_WARN_FLOOR_MS: i64 = 600_000;

/// REL-5 run-progress track (reliability spec §6): the committed-progress watermark the stall
/// warning keys on, the local-activity watermark its detail reports, and the observed-cadence
/// input to the per-run threshold. Keyed by run id; created when the reconcile pass first sees
/// the row joined + running (session readiness, within one tick), dropped when the row leaves
/// that state.
struct ProgressTrack {
    /// When committed progress (a `RoundOutcome`) was last observed — initialized at track
    /// creation so a run that never commits round 1 is detected.
    committed_at_ms: i64,
    /// The last committed round, `None` until the first `RoundOutcome`.
    last_round: Option<u64>,
    /// The slowest inter-commit gap observed on this run (ms) — the adaptive threshold input.
    max_commit_gap_ms: i64,
    /// When ANY local activity (`RoundProgress` / `CheckpointPublished`) was last observed —
    /// reported in the warning detail to distinguish "alive but not committing" from "silent".
    local_at_ms: i64,
    /// Whether `run_stalled` has been voiced for the CURRENT episode — one stateful transition
    /// each way, never a recurring alarm.
    stalled: bool,
}

impl ProgressTrack {
    fn fresh(now: i64) -> Self {
        Self {
            committed_at_ms: now,
            last_round: None,
            max_commit_gap_ms: 0,
            local_at_ms: now,
            stalled: false,
        }
    }

    /// The per-run stall threshold (ms): the floor, stretched by observed cadence.
    fn stall_threshold_ms(&self) -> i64 {
        RUN_STALL_WARN_FLOOR_MS.max(self.max_commit_gap_ms.saturating_mul(2))
    }
}

/// Durable-checkpoint lag alarm state for one role-instance generation (Gate B').
#[derive(Default)]
struct CkptLagTrack {
    /// The freshest round with a DURABLE checkpoint document (chunks + doc on the content
    /// plane — the `CheckpointPublished` edge), `None` until the generation first publishes.
    fence: Option<u64>,
    /// The largest lag already warned about — the alarm re-voices only when the drift GROWS,
    /// so a run sitting steadily past the threshold is one warning per widening round, never
    /// a per-event flood.
    warned_lag: u64,
}

/// The RAII half of the defect-17 bring-up serialization ([`VhcService::begin_bring_up`]):
/// holds the run's slot in `bring_up` for the life of one join/reconverge transaction and
/// releases it on drop — every early return of those long async bodies releases correctly.
struct BringUpGuard<'a> {
    set: &'a Mutex<std::collections::BTreeSet<String>>,
    run_id: String,
}

impl Drop for BringUpGuard<'_> {
    fn drop(&mut self) {
        self.set.lock().unwrap().remove(&self.run_id);
    }
}

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

/// The custodian sizing derived from `[vhc.storage]` — the SAME derivation the worker applies
/// to the exported environment (`CustodyConfig::from_env`), so the node's resume-authorization
/// custodian and the workers' write-path custodians judge one policy.
fn custody_config(
    storage: &daemon_vhc_session::config::StorageConfig,
) -> daemon_vhc_custody::CustodyConfig {
    const MIB: u64 = 1024 * 1024;
    daemon_vhc_custody::CustodyConfig {
        quota_bytes: (storage.quota_mb > 0).then(|| storage.quota_mb.saturating_mul(MIB)),
        scope_quota_bytes: (storage.run_quota_mb > 0)
            .then(|| storage.run_quota_mb.saturating_mul(MIB)),
        reserve_bytes: storage.reserve_mb.saturating_mul(MIB),
        emergency_bytes: storage.emergency_mb.saturating_mul(MIB),
    }
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
    /// Run bring-up transactions currently in flight (defect 17): an explicit `vhc join` and
    /// the auto-resume reconvergence SERIALIZE here — the second entrant refuses/skips instead
    /// of minting a competing incarnation. The c15k zombie: a restart's auto-resume raced the
    /// operator's explicit join, three coordinator incarnations were minted in seconds, and
    /// the survivor sat permanently superseded — WS-attached, certified outbound, refusing
    /// every inbound record, with no liveness signal anywhere.
    bring_up: Mutex<std::collections::BTreeSet<String>>,
    /// The co-located trainer's paced-respawn lane (defect 15), `run id → (attempts, due ms)`.
    /// A retryable-class sibling terminal arms it; the reconcile tick's repair pass fires it —
    /// the sibling has no run-level retry lane of its own because the PRIMARY seat instance
    /// stays live (the run row never leaves `Running`), so without this a retryable sibling
    /// terminal silently halved the run's membership until the operator intervened (c15k:
    /// `OUTCOME_STALE_RESTORE` after coordinator reconstruction stalled the run for good).
    co_retry: Mutex<BTreeMap<String, (u32, i64)>>,
    /// REL-7(d), reliability spec §9: the co-located trainer's CYCLE budget ledger,
    /// `run id → (short-lived flap cycles, last spawn unix ms)`. `co_retry` above counts only
    /// consecutive spawn REFUSALS (cleared on every successful spawn), so a flap-die-respawn
    /// loop ran unbounded — C2 recorded 461 `attempt 0` cycles at flat 1 s pace. This ledger
    /// survives successful spawns and adopts the primary keeper's discipline: a sibling
    /// terminal counts against `retry.max_retries` unless the session survived
    /// `retry.min_uptime_ms` (which resets the count); exhaustion parks the respawn lane
    /// loudly instead of silently cycling a dead seat forever.
    co_cycles: Mutex<BTreeMap<String, (u32, i64)>>,
    /// REL-8(d), reliability spec §10: the run ids whose storage-pressure episode is currently
    /// announced (a `storage_pressure` warning voiced once when a run scope crosses the
    /// threshold of its quota, cleared when it drops back) — pressure is announced before it
    /// kills, so quota deaths stop being surprises.
    storage_pressured: Mutex<std::collections::BTreeSet<String>>,
    worker_factory: Option<WorkerFactory>,
    /// The identity keystore directory (D-P8 credential + per-run cert authorship); `None`
    /// disables node-side authorship (tests / headless).
    identity_dir: Option<std::path::PathBuf>,
    /// The run-state root ([`VhcServiceParts::run_dir`]) — the disk custodian's governed root.
    run_dir: Option<std::path::PathBuf>,
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
    /// Durable-checkpoint lag alarm state, `run id → generation → track` (Gate B'): fed by
    /// `CheckpointPublished` (the fence) and `RoundOutcome` (the head), cleared on leave/wipe.
    ckpt_lag: Mutex<BTreeMap<String, BTreeMap<u64, CkptLagTrack>>>,
    /// REL-5 run-progress watermarks, `run id → track`: fed by `RoundOutcome` (committed) and
    /// `RoundProgress`/`CheckpointPublished` (local activity); the reconcile tick's stall
    /// observer reads them. Cleared on leave/wipe and when the row leaves joined + running.
    progress: Mutex<BTreeMap<String, ProgressTrack>>,
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
            bring_up: Mutex::new(std::collections::BTreeSet::new()),
            co_retry: Mutex::new(BTreeMap::new()),
            co_cycles: Mutex::new(BTreeMap::new()),
            storage_pressured: Mutex::new(std::collections::BTreeSet::new()),
            worker_factory: parts.worker_factory,
            identity_dir: parts.identity_dir,
            run_dir: parts.run_dir,
            seat,
            seat_read,
            events_tx,
            feed: parts.feed,
            current_run: Mutex::new(None),
            ckpt_lag: Mutex::new(BTreeMap::new()),
            progress: Mutex::new(BTreeMap::new()),
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
        // Manifest-driven orphan reconciliation (Phase 6): reclaim superseded incarnation dirs
        // the custody ledgers prove archived — BEFORE any join mints a fresh incarnation (the
        // newest dir per role is the reconstruction input and is never touched). Best-effort:
        // reconciliation trouble is loud but never blocks the boot.
        self.reconcile_run_state_dirs();
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

    /// The startup orphan-reconciliation pass ([`crate::reclaim`]): superseded incarnation
    /// directories whose custody ledgers prove full archival are reclaimed; everything retained
    /// is logged with its reason. No-op without a wired runs root.
    fn reconcile_run_state_dirs(&self) {
        let Some(root) = &self.run_dir else {
            return;
        };
        let known: std::collections::BTreeSet<String> = match self.store.list_runs() {
            Ok(runs) => runs
                .iter()
                .map(|r| blake3::hash(r.run_id.as_bytes()).to_hex().to_string())
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "orphan reconciliation skipped: run rows unreadable");
                return;
            }
        };
        let custodian =
            daemon_vhc_custody::DiskCustodian::for_root(root, custody_config(&self.config.storage))
                .ok();
        match crate::reclaim::reconcile_orphans(root, &known, custodian.as_ref()) {
            Ok(report) => {
                if report.incarnations > 0 || report.spills_only > 0 {
                    tracing::info!(
                        incarnations = report.incarnations,
                        spills_only = report.spills_only,
                        bytes = report.bytes,
                        "orphan reconciliation reclaimed superseded run state"
                    );
                }
                for (dir, reason) in &report.retained {
                    tracing::debug!(dir, reason, "orphan reconciliation retained");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "orphan reconciliation pass failed");
            }
        }
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
                if self.storage_gate_open_reclaiming() {
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
                // REL-8(b), reliability spec §10: a re-assess refusal on the free-disk/ram
                // lane floor is an ENVIRONMENT condition, not a failed attempt — it joins the
                // storage-gate lane (park → reclaim → re-check → resume) exactly like
                // `FailedStorage` does, instead of burning retry budget. A floor breach that
                // survives reclamation still escalates loudly: the gated redispatch above
                // refuses each pass and says why. Non-disk refusals keep the budgeted lane.
                Err(e) if e.to_string().contains("below lane floor: ram/disk") => {
                    self.store.set_storage_gated(&run.run_id, true)?;
                    let due_at = now_ms() + retry.max_backoff_ms as i64;
                    let _ = self.store.defer_retry(&run.run_id, due_at);
                    let mut emitted = Vec::new();
                    let _ = self.emit(
                        VhcEvent::Warning {
                            run_id: run.run_id.clone(),
                            class: "storage_gate".to_string(),
                            detail: format!(
                                "reconverge re-assess refused on the ram/disk lane floor — \
                                 parked storage-gated (retry budget untouched; reclaim runs \
                                 before the gate re-check): {e}"
                            ),
                        },
                        &mut emitted,
                    );
                    self.emit_changed(Some(run.run_id.clone()));
                }
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
        // Defect 15 — the co-located trainer repair pass: a seat-holding run whose sibling died
        // on a retryable-class terminal re-converges the sibling here, paced by the lane armed
        // in `handle_run_terminated`. The lane fires only while the run is genuinely live on
        // this node (row `Running`, primary instance present, seat role); anything else clears
        // it — a leave/terminal of the primary owns the whole teardown.
        if self.config.coordinator_trains {
            let now = now_ms();
            let due_siblings: Vec<String> = {
                let lane = self.co_retry.lock().unwrap();
                lane.iter()
                    .filter(|(_, (_, due))| *due <= now)
                    .map(|(run_id, _)| run_id.clone())
                    .collect()
            };
            for run_id in due_siblings {
                if self.co_trainers.lock().unwrap().contains_key(&run_id) {
                    // A competing path (reconvergence of the primary) already respawned it.
                    self.co_retry.lock().unwrap().remove(&run_id);
                    continue;
                }
                let row = self.store.get_run(&run_id)?;
                // Defect 19: only a GENUINE teardown clears the lane — the row gone, a
                // deliberate end (completed / left / failed_terminal / intent withdrawn), or a
                // non-seat role. A TRANSIENT window (row mid-churn in `failed_retryable` /
                // `starting`, or the primary mid-replacement) used to clear it too, which
                // orphaned the sibling forever once the seat came back (c15m: the lane died
                // during the seat-replacement churn and no trainer respawn ever fired again).
                let torn_down = row.as_ref().is_none_or(|r| {
                    matches!(
                        r.run_state,
                        RunState::Completed | RunState::Left | RunState::FailedTerminal
                    ) || r.desired_state != DesiredState::Joined
                        || r.role.is_empty()
                        || r.role != self.config.seat_role
                });
                if torn_down {
                    self.co_retry.lock().unwrap().remove(&run_id);
                    self.co_cycles.lock().unwrap().remove(&run_id);
                    continue;
                }
                let primary_live = self.instances.lock().unwrap().contains_key(&run_id);
                let ready = row
                    .as_ref()
                    .is_some_and(|r| r.run_state == RunState::Running)
                    && primary_live;
                if !ready {
                    // Transient: keep the lane armed with grown backoff and let a later tick
                    // (or the seat's own reconvergence) pick it up.
                    let mut lane = self.co_retry.lock().unwrap();
                    let attempts = lane.get(&run_id).map_or(1, |(n, _)| *n);
                    lane.insert(
                        run_id.clone(),
                        (
                            attempts.saturating_add(1),
                            now_ms() + retry_backoff_ms(retry, attempts) as i64,
                        ),
                    );
                    continue;
                }
                let policy = row.expect("eligible row present").policy;
                self.spawn_co_located_trainer(&run_id, &policy).await;
                if self.co_trainers.lock().unwrap().contains_key(&run_id) {
                    self.co_retry.lock().unwrap().remove(&run_id);
                    reconverged += 1;
                } else {
                    // The spawn refused (warned inside): re-arm with grown backoff so the
                    // repair keeps trying without a hot loop.
                    let mut lane = self.co_retry.lock().unwrap();
                    let attempts = lane.get(&run_id).map_or(1, |(n, _)| *n);
                    lane.insert(
                        run_id.clone(),
                        (
                            attempts.saturating_add(1),
                            now_ms() + retry_backoff_ms(retry, attempts) as i64,
                        ),
                    );
                }
            }
        }
        // REL-8(d) (reliability spec §10): pressure is announced before it kills. A run scope
        // crossing the announce threshold of its per-run quota voices ONE `storage_pressure`
        // warning (the episode clears silently when usage drops back, re-arming the announce)
        // — so the two C2 quota-death surfaces (journal-sink `HostStorageExhausted` mid-run,
        // lane-floor refusals against stacked dead incarnations) stop being surprises.
        if self.config.storage.run_quota_mb > 0 {
            if let Some(root) = &self.run_dir {
                if let Ok(custodian) = daemon_vhc_custody::DiskCustodian::for_root(
                    root,
                    custody_config(&self.config.storage),
                ) {
                    let usage = custodian.usage();
                    let scope_used: BTreeMap<&str, u64> =
                        usage.scopes.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                    for run in self.store.list_runs()? {
                        let scope = blake3::hash(run.run_id.as_bytes()).to_hex().to_string();
                        let used = scope_used.get(scope.as_str()).copied().unwrap_or(0);
                        let pressured =
                            Self::run_scope_pressured_at(used, self.config.storage.run_quota_mb);
                        let mut announced = self.storage_pressured.lock().unwrap();
                        if pressured && !announced.contains(&run.run_id) {
                            announced.insert(run.run_id.clone());
                            drop(announced);
                            let mut emitted = Vec::new();
                            let _ = self.emit(
                                VhcEvent::Warning {
                                    run_id: run.run_id.clone(),
                                    class: "storage_pressure".to_string(),
                                    detail: format!(
                                        "run scope at {} of its {} MiB quota — reclaim or \
                                         completion must land before the quota refuses writes",
                                        format_args!("{} MiB", used / (1024 * 1024)),
                                        self.config.storage.run_quota_mb
                                    ),
                                },
                                &mut emitted,
                            );
                        } else if !pressured {
                            announced.remove(&run.run_id);
                        }
                    }
                }
            }
        }
        // REL-5 stall observer (reliability spec §6): a joined, alive run whose committed
        // progress has aged past its per-run threshold is announced ONCE (`run_stalled`);
        // `run_progress_resumed` closes the episode from the event pump. Detection only —
        // recovery stays with the keeper (REL-9). Tracks are created here at the first tick
        // that observes the row joined + running (session readiness, within one tick) and
        // dropped when the row leaves that state, so a fresh incarnation restarts its watermark.
        {
            let now = now_ms();
            let live: Vec<String> = self
                .store
                .list_runs()?
                .into_iter()
                .filter(|r| {
                    r.desired_state == DesiredState::Joined && r.run_state == RunState::Running
                })
                .map(|r| r.run_id)
                .collect();
            let fire: Vec<(String, String)> = {
                let mut progress = self.progress.lock().unwrap();
                progress.retain(|id, _| live.contains(id));
                let mut fire = Vec::new();
                for run_id in &live {
                    let t = progress
                        .entry(run_id.clone())
                        .or_insert_with(|| ProgressTrack::fresh(now));
                    let age = now.saturating_sub(t.committed_at_ms);
                    let threshold = t.stall_threshold_ms();
                    if !t.stalled && age > threshold {
                        t.stalled = true;
                        let round = t
                            .last_round
                            .map_or("no round committed yet".to_string(), |r| {
                                format!("last committed round {r}")
                            });
                        fire.push((
                            run_id.clone(),
                            format!(
                                "no committed round for {}s ({round}; last local activity {}s \
                                 ago; threshold {}s = max({}s floor, 2× the slowest observed \
                                 inter-round gap)) — the session is alive and joined but the \
                                 run head is not advancing",
                                age / 1000,
                                now.saturating_sub(t.local_at_ms) / 1000,
                                threshold / 1000,
                                RUN_STALL_WARN_FLOOR_MS / 1000,
                            ),
                        ));
                    }
                }
                fire
            };
            for (run_id, detail) in fire {
                tracing::warn!(run = run_id, "{detail}");
                let mut emitted = Vec::new();
                let _ = self.emit(
                    VhcEvent::Warning {
                        run_id: run_id.clone(),
                        class: "run_stalled".to_string(),
                        detail,
                    },
                    &mut emitted,
                );
                self.emit_changed(Some(run_id));
            }
        }
        Ok(reconverged)
    }

    /// REL-8(d): whether a run scope's usage has crossed the announce threshold (80%) of the
    /// per-run quota — the point where reclaim/completion still has margin to land before the
    /// quota starts refusing writes.
    fn run_scope_pressured_at(used_bytes: u64, run_quota_mb: u64) -> bool {
        run_quota_mb > 0 && used_bytes >= run_quota_mb.saturating_mul(1024 * 1024) / 5 * 4
    }

    /// The storage gate with RECLAIM-BEFORE-REFUSE: a held gate first runs the manifest-driven
    /// orphan reconciliation (the same startup pass — superseded incarnations' spills are
    /// reclaimable unconditionally, proven-archived journals go entirely; the reclaimed bytes
    /// discharge the shared per-root custody ledger) and re-checks. A box sitting on gigabytes
    /// of its OWN superseded attempts must never wait on an operator to clear space the
    /// manifests prove reclaimable — that is exactly how a quota-refused run once accumulated a
    /// dozen orphaned incarnations and died terminal on a healthy disk.
    fn storage_gate_open_reclaiming(&self) -> bool {
        self.storage_gate_open() || {
            self.reconcile_run_state_dirs();
            self.storage_gate_open()
        }
    }

    /// The storage gate's resume-authorization check (Phase 6): open when the run-state root's
    /// disk CUSTODIAN confirms capacity — the floor clears AND the quotas admit new work
    /// (`Pressure::RefuseNew` holds the gate; the pre-custodian bare free-space probe held only
    /// on the floor). `reserve_mb = 0` with no quota disables the gate. A node without a wired
    /// run-state root falls back to the interim identity-dir free-space probe; a root that
    /// cannot answer a capacity query maps to CLOSED — not one to redispatch onto.
    fn storage_gate_open(&self) -> bool {
        let storage = &self.config.storage;
        if storage.reserve_mb == 0 && storage.quota_mb == 0 && storage.run_quota_mb == 0 {
            return true;
        }
        if let Some(root) = &self.run_dir {
            return match daemon_vhc_custody::DiskCustodian::for_root(root, custody_config(storage))
            {
                Ok(custodian) => custodian.pressure() != daemon_vhc_custody::Pressure::RefuseNew,
                Err(_) => false,
            };
        }
        // Interim fallback (headless / test wiring without a runs root): the bare floor probe.
        if storage.reserve_mb == 0 {
            return true;
        }
        let Some(dir) = &self.identity_dir else {
            return true;
        };
        daemon_vhc_session::host_disk_free_mb(dir) >= storage.reserve_mb
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
    /// Enter the run's bring-up transaction (defect 17): `None` when another bring-up (an
    /// explicit join or an auto-resume reconvergence) already holds it. The guard releases on
    /// drop, so every early return of the long async transaction bodies releases correctly.
    fn begin_bring_up(&self, run_id: &str) -> Option<BringUpGuard<'_>> {
        let mut set = self.bring_up.lock().unwrap();
        if !set.insert(run_id.to_string()) {
            return None;
        }
        Some(BringUpGuard {
            set: &self.bring_up,
            run_id: run_id.to_string(),
        })
    }

    async fn reconverge(&self, run: &PersistedRun) -> Result<(), VhcError> {
        // Defect 17: the auto-resume half of the serialization — an explicit join (or another
        // reconvergence) already bringing this run up owns the transaction; this pass simply
        // yields (the retry lane re-fires if the standing transaction fails to land).
        let Some(_guard) = self.begin_bring_up(&run.run_id) else {
            tracing::info!(
                run_id = run.run_id,
                "reconverge: a bring-up for this run is already in flight; yielding to it"
            );
            return Ok(());
        };
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
        // REL-8(a), reliability spec §10: run the existing orphan reclamation BEFORE the fresh
        // child's assess. Every reconverge recycles an incarnation, and the superseded
        // predecessors' spills are exactly the bytes the re-assess's disk probe is about to
        // judge — C2 accumulated ~59.8 GiB of dead recoverable state this way until the lane
        // floor refused and the budget burned out against a full disk. Same judgment as the
        // startup/gate-open passes: proven-superseded incarnations only, the newest is never
        // reclaimed, the payload/archive evidence planes are never touched.
        self.reconcile_run_state_dirs();
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
        // The recovery half of the join transaction (§8.8): the verified archive lineage —
        // the reconstruction directive for a seat-role reconvergence (a crash restart is
        // exactly the case with published history behind it) and the verified head estimate
        // for the restore staleness judgment.
        let recovery = match self.resolve_recovery(&run.run_id, &run.role).await {
            Ok(r) => r,
            Err(e) => {
                self.arbiter.release(&id);
                if self.worker_factory.is_some() {
                    worker.shutdown().await;
                }
                return Err(e);
            }
        };
        let (restore, catch_up) = match self
            .resolve_restore(&run.run_id, &run.role, &recovery)
            .await
        {
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
                JoinResume {
                    restore,
                    reconstruct: recovery.reconstruct,
                    catch_up,
                },
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
        // Defect 19: a registered sibling here is STALE by construction — every caller gates on
        // a fresh seat bring-up (`existing.is_none()` / reconverge's fresh mint) or on an empty
        // lane (the repair pass), so an entry can only belong to a REPLACED owner. It is bound
        // to the superseded owner's endpoint and will only die asynchronously; left registered
        // it blocked every respawn (the old idempotence early-return) while its eventual
        // stream-closure cleanup raced the next bring-up (c15m: trainer-0 never respawned for
        // 2+ hours across five seat replacements). The transaction that replaces the seat owns
        // the sibling's replacement: reap deterministically, then spawn fresh.
        let stale = self.co_trainers.lock().unwrap().remove(run_id);
        if let Some(e) = stale {
            self.arbiter.release(&e.id);
            e.worker.shutdown().await;
            warn_co!(format!(
                "reaped the superseded owner's co-trainer sibling (generation {}) before \
                 respawning under the fresh seat",
                e.generation
            ));
        }
        let worker = self.instance_worker();
        // Assess + resolve the coordinator endpoint for the TRAINER role. Undirected: against a
        // SEATED genesis the node selects the trainer seat authored for its own identity
        // (defect 6); against the pre-seat form the worker's first-non-coordinator default is
        // the `trainer` role as before.
        let (coordinator, eligibility, assessed_tuple, directed) =
            match self.resolve_join(&worker, run_id, None).await {
                Ok(v) => v,
                Err(e) => {
                    warn_co!(format!("co-located trainer assess failed: {e}"));
                    worker.shutdown().await;
                    return;
                }
            };
        let trainer_role = directed.unwrap_or_else(|| "trainer".to_string());
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
            role: trainer_role.clone(),
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
        // The verified head estimate for the staleness judgment (a trainer never carries the
        // reconstruction directive — the seat lineage is not its history to rebuild).
        let recovery = match self.resolve_recovery(run_id, &trainer_role).await {
            Ok(r) => r,
            Err(e) => {
                self.arbiter.release(&id);
                warn_co!(format!(
                    "co-located trainer recovery resolution refused: {e}"
                ));
                worker.shutdown().await;
                return;
            }
        };
        let (restore, catch_up) = match self.resolve_restore(run_id, &trainer_role, &recovery).await
        {
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
            .author_join(
                run_id,
                &coordinator,
                &id,
                assessed_tuple,
                JoinResume {
                    restore,
                    reconstruct: recovery.reconstruct,
                    catch_up,
                },
                seat,
            )
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
        // Stamp the spawn for the cycle-budget uptime judgment (REL-7(d)): a sibling that
        // survives `min_uptime_ms` from here resets the flap-cycle count on its next terminal.
        self.co_cycles
            .lock()
            .unwrap()
            .entry(run_id.to_string())
            .or_insert((0, 0))
            .1 = now_ms();
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
            // The run's disk footprint is bounded by the per-run custody quota (Phase 6), so
            // that is the honest figure the owner's disk ledger reserves. An unbounded run
            // quota (0) charges nothing — the ledger cannot price a reservation the custodian
            // does not bound (the free-space floor still protects the host).
            disk_bytes: self.config.storage.run_quota_mb.saturating_mul(1024 * 1024),
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
            protocol::Event::RoundOutcome { round, stalled, .. } => {
                // The durable round head: `RunPhase` writes the row only at lifecycle edges,
                // so without this `vhc detail` reads round=0 for the whole run (c15 defect).
                self.store.advance_round(&run_id, *round)?;
                self.store
                    .bump_contribution(&run_id, u64::from(!*stalled), 0, 0, 0, 0, 0)?
            }
            protocol::Event::CheckpointPublished { .. } => {
                self.store.bump_contribution(&run_id, 0, 0, 0, 0, 0, 1)?
            }
            _ => {}
        }

        let mut emitted = Vec::new();

        // REL-5 progress watermarks (reliability spec §6): committed progress (`RoundOutcome`)
        // advances the watermark the stall observer keys on and closes an announced stall; local
        // activity (`RoundProgress` / `CheckpointPublished`) advances the watermark the warning
        // detail reports. The inter-commit gap feeds the per-run adaptive threshold.
        {
            let resumed = {
                let mut progress = self.progress.lock().unwrap();
                match ev {
                    protocol::Event::RoundOutcome { round, .. } => {
                        progress.get_mut(&run_id).and_then(|t| {
                            let now = now_ms();
                            let gap = now.saturating_sub(t.committed_at_ms);
                            t.max_commit_gap_ms = t.max_commit_gap_ms.max(gap);
                            t.committed_at_ms = now;
                            t.local_at_ms = now;
                            t.last_round = Some(*round);
                            std::mem::take(&mut t.stalled).then_some((*round, gap))
                        })
                    }
                    protocol::Event::RoundProgress { .. }
                    | protocol::Event::CheckpointPublished { .. } => {
                        if let Some(t) = progress.get_mut(&run_id) {
                            t.local_at_ms = now_ms();
                        }
                        None
                    }
                    _ => None,
                }
            };
            if let Some((round, gap_ms)) = resumed {
                let detail = format!(
                    "committed progress resumed at round {round} after {}s without a commit",
                    gap_ms / 1000
                );
                tracing::info!(run = run_id, "{detail}");
                self.emit(
                    VhcEvent::Warning {
                        run_id: run_id.clone(),
                        class: "run_progress_resumed".to_string(),
                        detail,
                    },
                    &mut emitted,
                )?;
            }
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
                // emitter: the co-trainer's generation ⇒ ITS role (a seat-directed join runs an
                // authored per-seat label like `trainer-0`), else the run row's role.
                let co_role = {
                    let co = self.co_trainers.lock().unwrap();
                    co.get(&run_id)
                        .filter(|e| e.generation == *generation)
                        .map(|e| e.id.role.clone())
                };
                let role = if let Some(role) = co_role {
                    role
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
                    // Bounded in-task retry (Gate B'): the pointer is the fence every restore
                    // resolves against, minted here AFTER the 14–25-minute family upload walk
                    // already paid its full durability cost — dropping it on one failed POST
                    // silently discards that entire slot (the seat believes it published; the
                    // registry never learned). A registry blip is far shorter than a slot, so
                    // a few paced attempts close the gap; a still-failing pointer stays what it
                    // always was (advisory, best-effort) and is voiced as a warning.
                    let mut last = None;
                    for (attempt, pause_s) in [0u64, 2, 8, 30].into_iter().enumerate() {
                        if pause_s > 0 {
                            tokio::time::sleep(std::time::Duration::from_secs(pause_s)).await;
                        }
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
                                    attempt,
                                    "checkpoint pointer published"
                                );
                                return;
                            }
                            Err(e) => last = Some(e),
                        }
                    }
                    let error = last.expect("at least one attempt ran");
                    tracing::warn!(run, role, kind, round, %error, "checkpoint pointer publication failed (all attempts)");
                });
            } else {
                tracing::debug!(
                    run_id,
                    "checkpoint published but no discovery is wired (pointer not recorded)"
                );
            }
        }

        // The durable-checkpoint lag alarm (Gate B'; defect 14's unvoiced drift): the FENCE a
        // rejoin would restore from is the freshest DURABLE checkpoint (`CheckpointPublished` —
        // doc + chunks on the content plane), and it trails the head by checkpoint assembly +
        // the by-ref upload walk (measured 14–25 min per slot at ceremony scale). Past the
        // retained-record horizon that drift wedges re-admission (`CheckpointStale`), so it is
        // voiced as a persisted warning while there is still margin to act. Trainer generations
        // only: a coordinator recovers through archive reconstruction, never a live pointer.
        match ev {
            protocol::Event::CheckpointPublished {
                round, generation, ..
            } => {
                let mut lag_map = self.ckpt_lag.lock().unwrap();
                let track = lag_map
                    .entry(run_id.clone())
                    .or_default()
                    .entry(*generation)
                    .or_default();
                track.fence = Some(track.fence.map_or(*round, |f| f.max(*round)));
                // A fresh fence closes the voiced drift; a NEW drift warns anew.
                track.warned_lag = 0;
            }
            protocol::Event::RoundOutcome {
                round, generation, ..
            } => {
                let role = {
                    let co = self.co_trainers.lock().unwrap();
                    co.get(&run_id)
                        .filter(|e| e.generation == *generation)
                        .map(|e| e.id.role.clone())
                }
                .or_else(|| {
                    self.store
                        .get_run(&run_id)
                        .ok()
                        .flatten()
                        .map(|r| r.role)
                        .filter(|r| !r.is_empty())
                });
                if role.as_deref().is_some_and(|r| r != "coordinator") {
                    let warn = {
                        let mut lag_map = self.ckpt_lag.lock().unwrap();
                        let track = lag_map
                            .entry(run_id.clone())
                            .or_default()
                            .entry(*generation)
                            .or_default();
                        // No fence at all → the exposure is the whole span since genesis.
                        let lag = round.saturating_sub(track.fence.unwrap_or(0));
                        (lag >= CHECKPOINT_LAG_WARN_ROUNDS && lag > track.warned_lag).then(|| {
                            track.warned_lag = lag;
                            match track.fence {
                                Some(f) => format!(
                                    "{}'s freshest durable checkpoint (round {f}) trails the \
                                     live head (round {round}) by {lag} rounds — the retained \
                                     horizon is {}; past it a rejoin needs archive catch-up \
                                     (is checkpoint publication stalled?)",
                                    role.as_deref().unwrap_or("trainer"),
                                    daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS,
                                ),
                                None => format!(
                                    "{} has published NO durable checkpoint by round {round} — \
                                     the retained horizon is {}; a crash before the first \
                                     durable checkpoint restarts this seat from scratch",
                                    role.as_deref().unwrap_or("trainer"),
                                    daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS,
                                ),
                            }
                        })
                    };
                    if let Some(detail) = warn {
                        tracing::warn!(run = run_id, generation, "{detail}");
                        self.emit(
                            VhcEvent::Warning {
                                run_id: run_id.clone(),
                                class: "checkpoint_lag".to_string(),
                                detail,
                            },
                            &mut emitted,
                        )?;
                    }
                }
            }
            _ => {}
        }
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
                // Defect 15: a retryable-class sibling terminal arms the PACED respawn lane —
                // the primary seat instance is still live, so the run-level retry lane never
                // fires for the sibling; without this the run silently loses half its local
                // membership (c15k: the trainer's `OUTCOME_STALE_RESTORE` after coordinator
                // reconstruction, dropped here, wedged the run below its floor). Pacing
                // mirrors the primary lane: transport = jittered budget-free, storage = the
                // gate ceiling, retryable = growing backoff; deliberate ends (completed /
                // left / failed_terminal) never respawn.
                let retry = &self.config.retry;
                let (respawn, reason) = match outcome {
                    protocol::TerminalOutcome::FailedRetryable { reason } => {
                        (true, reason.as_str())
                    }
                    protocol::TerminalOutcome::FailedStorage { reason } => (true, reason.as_str()),
                    protocol::TerminalOutcome::FailedTransport { reason } => {
                        (true, reason.as_str())
                    }
                    _ => (false, ""),
                };
                if respawn {
                    // REL-7(d), reliability spec §9: the CYCLE budget — the `co_retry`
                    // attempts counter below is cleared on every successful spawn (it counts
                    // consecutive spawn refusals), so it never bounded a flap-die-respawn
                    // loop (C2: 461 `attempt 0` cycles, ≥ 2.7 h of churn). The primary
                    // keeper's discipline applies instead: this terminal counts a cycle
                    // UNLESS the sibling survived `min_uptime_ms` (which resets the count);
                    // an exhausted budget parks the lane LOUDLY — a dead network must
                    // surface, never hide behind an eternally-cycling seat.
                    let cycles = {
                        let mut cyc = self.co_cycles.lock().unwrap();
                        let entry = cyc.entry(run_id.to_string()).or_insert((0, 0));
                        let uptime_ms = now_ms().saturating_sub(entry.1);
                        if entry.1 > 0 && uptime_ms >= retry.min_uptime_ms as i64 {
                            entry.0 = 0;
                        }
                        entry.0 = entry.0.saturating_add(1);
                        entry.0
                    };
                    if cycles > retry.max_retries {
                        self.co_retry.lock().unwrap().remove(run_id);
                        let mut emitted = Vec::new();
                        let _ = self.emit(
                            VhcEvent::Warning {
                                run_id: run_id.to_string(),
                                class: "co_trainer".to_string(),
                                detail: format!(
                                    "co-located trainer cycle budget exhausted (after {} paced \
                                     respawns, none surviving {} ms): {reason} — respawn lane \
                                     parked; recovery escalates to the run-level lane (a \
                                     primary reconvergence re-mints the sibling)",
                                    cycles - 1,
                                    retry.min_uptime_ms
                                ),
                            },
                            &mut emitted,
                        );
                        return Ok(emitted);
                    }
                    let (attempts, pace_ms) = {
                        let mut lane = self.co_retry.lock().unwrap();
                        let attempts = lane.get(run_id).map_or(0, |(n, _)| *n);
                        let pace_ms = match outcome {
                            protocol::TerminalOutcome::FailedTransport { .. } => {
                                transport_backoff_jittered_ms(retry, run_id)
                            }
                            protocol::TerminalOutcome::FailedStorage { .. } => retry.max_backoff_ms,
                            _ => retry_backoff_ms(retry, attempts),
                        };
                        lane.insert(
                            run_id.to_string(),
                            (attempts.saturating_add(1), now_ms() + pace_ms as i64),
                        );
                        (attempts, pace_ms)
                    };
                    let mut emitted = Vec::new();
                    let _ = self.emit(
                        VhcEvent::Warning {
                            run_id: run_id.to_string(),
                            class: "co_trainer".to_string(),
                            detail: format!(
                                "co-located trainer terminated (attempt {attempts}, cycle \
                                 {cycles}/{}): {reason} — paced respawn in {pace_ms} ms",
                                retry.max_retries
                            ),
                        },
                        &mut emitted,
                    );
                    return Ok(emitted);
                }
                // A deliberate end (completed / left / failed_terminal) never respawns — and
                // closes the cycle ledger with it.
                self.co_cycles.lock().unwrap().remove(run_id);
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
            // The typed transport taxonomy (Gate C, defect 10): a transient network fault
            // (connect/timeout/reset/5xx — e.g. the content plane unreachable during
            // reconstruction) is environmental, never a failed attempt. The deferral below is
            // BUDGET-FREE and the reason is surfaced prefixed so `vhc detail` shows the run
            // is deliberately waiting out an outage rather than crash-looping.
            protocol::TerminalOutcome::FailedTransport { reason } => (
                RunState::FailedRetryable,
                format!("transport fault (deferred budget-free): {reason}"),
            ),
        };
        let storage_gated = matches!(outcome, protocol::TerminalOutcome::FailedStorage { .. });
        let transport_deferred =
            matches!(outcome, protocol::TerminalOutcome::FailedTransport { .. });
        // The bounded retry budget: a recoverable failure past the budget escalates to terminal
        // with a typed reason; within it, the next reconvergence is backoff-scheduled. A
        // storage-gated failure bypasses the budget entirely — the gate (not the budget) is
        // what bounds it, so it can never launder a crash loop: the moment the disk has
        // headroom the run redispatches and any non-storage failure consumes budget normally.
        // A transport deferral equally bypasses the budget: the retry is paced with JITTER
        // (half the ceiling plus a spread, so a fleet knocked over by one outage never
        // thunders back in lockstep), indefinitely by design — `vhc leave` cancels the intent,
        // and any non-transport failure after redispatch consumes budget normally.
        let retry = &self.config.retry;
        let consumed = row.as_ref().map_or(0, |r| r.retry_count);
        let (target, reason, next_retry) = if storage_gated {
            let due = now_ms() + retry.max_backoff_ms as i64;
            (RunState::FailedRetryable, reason, Some(due))
        } else if transport_deferred {
            let due = now_ms() + transport_backoff_jittered_ms(retry, run_id) as i64;
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
        // check) without consuming budget, and durably marks the gate. A transport deferral
        // equally defers without consuming budget (the jittered due time was computed above)
        // and surfaces itself as a typed warning — the operator-visible evidence that the run
        // is waiting out a network outage by design.
        let mut emitted = Vec::new();
        if let Some(due) = next_retry {
            if storage_gated {
                let _ = self.store.defer_retry(run_id, due);
                let _ = self.store.set_storage_gated(run_id, true);
            } else if transport_deferred {
                let _ = self.store.defer_retry(run_id, due);
                let _ = self.emit(
                    VhcEvent::Warning {
                        run_id: run_id.to_string(),
                        class: "transport_deferred".to_string(),
                        detail: format!(
                            "transient transport fault: retry deferred budget-free until \
                             +{} ms (retry budget unchanged; `vhc leave` cancels)",
                            due.saturating_sub(now_ms())
                        ),
                    },
                    &mut emitted,
                );
            } else {
                let _ = self.store.bump_retry(run_id, due);
            }
        }
        // 5) [CI-7] identity custody: a terminal that ends the run identity (completed / left /
        // failed_terminal) deletes its per-run key material, certificates, and credentials
        // record — no run identity outlives the run it was minted for. `failed_retryable` is NOT
        // terminal for the identity (reconvergence supersedes it under the retry budget), so its
        // material survives. Idempotent (remove_run tolerates absence).
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
    ) -> Result<
        (
            String,
            VhcEligibility,
            Option<protocol::AdmittedTuple>,
            Option<String>,
        ),
        VhcError,
    > {
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
            // An undirected join against a SEATED genesis selects the seat authored for this
            // node's identity (defect 6): the worker's first-non-coordinator default would
            // decode another seat's plan identity.
            let role = match role {
                Some(r) => Some(r),
                None => self.authored_seat(&envelope)?,
            };
            let verdict = worker.assess(envelope, role.clone()).await?;
            // Stamp the node-owned revisions into the immutable admitted tuple (architecture
            // §6.3); the incarnation stays 0 (unassigned) until the node mints it at join.
            let tuple = self.stamp_admitted_tuple(verdict.admitted_tuple.clone())?;
            Ok((
                run.coordinator,
                eligibility_from_assess(&verdict),
                tuple,
                role,
            ))
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
            Ok((coordinator, eligibility, None, role))
        }
    }

    /// The authored-SEAT selection for an undirected join (defect 6 of the c15 drills): the
    /// label of the non-coordinator role whose [`daemon_vhc_proto::RoleEntry::identity`] binds
    /// this node's base identity. The role's opaque config carries a per-participant plan
    /// identity the host may never decode (the seam rule); the identity binding is the
    /// host-visible half that lets the node pick ITS seat.
    ///
    /// - `Ok(None)` — the role set carries no identity-bound worker roles (the pre-seat form):
    ///   the worker's undirected default (first non-coordinator role) applies unchanged.
    /// - `Ok(Some(label))` — the seat authored for this node.
    /// - `Err` — the role set IS seated and this node holds no seat (or has no base identity to
    ///   select with). Joining undirected would silently run someone else's plan identity —
    ///   every box training the same window slice, the checkpoint slots elected to the
    ///   un-impersonated seats published by nobody — so the join refuses typed instead.
    fn authored_seat(&self, envelope_wire: &[u8]) -> Result<Option<String>, VhcError> {
        let Ok(wire) = daemon_vhc_proto::from_canonical_slice::<daemon_vhc_proto::SignedEnvelope>(
            envelope_wire,
        ) else {
            return Ok(None); // not the signed wire form — the worker's typed refusal names it
        };
        let Ok(frozen) =
            daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
        else {
            return Ok(None); // not a verifiable genesis — likewise the worker's refusal to give
        };
        let Ok(env) = frozen.decode() else {
            return Ok(None);
        };
        let seats: Vec<(&String, daemon_vhc_proto::PeerId)> = env
            .roles
            .iter()
            .filter(|(_, r)| r.lane != "coordinator")
            .filter_map(|(name, r)| r.identity.map(|id| (name, id)))
            .collect();
        if seats.is_empty() {
            return Ok(None);
        }
        let Some(dir) = &self.identity_dir else {
            return Err(VhcError::Worker(
                "the genesis authors identity-bound seats but this node has no identity \
                 keystore configured to select one"
                    .into(),
            ));
        };
        let keystore = daemon_vhc_session::keystore::VhcKeystore::open(dir)
            .map_err(|e| VhcError::Internal(format!("open identity keystore: {e}")))?;
        let own = keystore
            .base_identity()
            .map(|k| daemon_vhc_proto::peer_id(&k))
            .map_err(|e| VhcError::Internal(format!("no base identity to select a seat: {e}")))?;
        match seats.iter().find(|(_, id)| *id == own) {
            Some((label, _)) => Ok(Some((*label).clone())),
            None => Err(VhcError::Worker(format!(
                "the genesis authors {} identity-bound seat(s) and none binds this node's base \
                 identity {} — join from a box whose identity holds a seat, or re-author the \
                 genesis with this box in the roster",
                seats.len(),
                own.to_hex()
            ))),
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
        let (coordinator, eligibility, tuple, _directed) = self
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
        resume: JoinResume,
        seat: crate::credentials::SeatBootstrap,
    ) -> Result<AuthoredDelivery, VhcError> {
        let JoinResume {
            restore,
            reconstruct,
            catch_up,
        } = resume;
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
                reconstruct,
                catch_up,
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

    /// Resolve the late-join checkpoint restore for a run (spec §9, Gate D' order): the seat's
    /// OWN pointer first (correct replica-local semantics), falling back to a sibling seat in
    /// the same role family only when the seat has published nothing
    /// ([`crate::discovery::best_restore_pointer`]); a pointer outside the family is never
    /// consulted for the restore itself — decoded to the wire restore form (`None` = fresh
    /// start / no discovery / nothing published for the family). A malformed pointer hash is
    /// dropped (fresh start), never a hard join failure.
    ///
    /// A SIBLING adoption is a recorded posture, never a silent equivalence: the sibling's doc
    /// carries its class-1 replica-local sections (optimizer moments, error feedback — per-seat
    /// trajectories), so the adoption emits a persisted `sibling_restore_adopted` warning
    /// (windowed into `vhc detail`'s recent events) naming both seats and the adopted round.
    /// Consensus safety is unaffected (round digests cover exactly the class-0 consensus
    /// sections; c15g proved live agreement).
    ///
    /// **Join-time fence reachability** (the recovery-honesty check, re-scoped by Gate B'): a
    /// restorer replays forward across the coordinator's retained record ring
    /// ([`daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS`]), and with three seats the run
    /// progresses while one trainer is absent — no static cadence/ring relation guarantees the
    /// fence stays reachable. The judgment is made HERE, from actual run evidence, before any
    /// rehydration — but a fence past the ring is no longer an automatic refusal: when the
    /// VERIFIED archive lineage covers the gap (its latest round claim reaches within a ring of
    /// the head), the join instead carries a [`protocol::TrainerCatchUp`] directive — the worker
    /// folds the archived records up to the archive tip, and the ring replay covers the
    /// unarchived tail (architecture §5.3). [`VhcError::CheckpointStale`] remains only for the
    /// gap the planes genuinely cannot bridge (no lineage / an archive tip itself past ring
    /// reach of the head) — typed, naming the missing closure.
    ///
    /// The head estimate prefers `verified_head` — the latest committed-round claim across the
    /// seat lineage's SIGNED archive heads ([`Self::resolve_recovery`]); certificate-chained
    /// evidence, never registry metadata — and keeps the registry's coordinator pointer as a
    /// fallback lower bound (heads published before the round claim existed carry none). Every
    /// source is a LOWER bound on the true head, so the check stays conservative and the
    /// in-module refusal remains the authoritative backstop. No evidence = no judgment (nothing
    /// to compare against). Benign at min/max 2/2 (rounds pause while a peer is down);
    /// mandatory before C2's larger fleets.
    async fn resolve_restore(
        &self,
        run_id: &str,
        role: &str,
        recovery: &RecoveryResolution,
    ) -> Result<
        (
            Option<protocol::CheckpointRestore>,
            Option<protocol::TrainerCatchUp>,
        ),
        VhcError,
    > {
        let Some(discovery) = self.discovery.as_ref() else {
            return Ok((None, None));
        };
        let role = if role.is_empty() { "trainer" } else { role };
        let Some(pointer) = discovery
            .fetch_checkpoint(run_id, role)
            .await
            .ok()
            .flatten()
        else {
            return Ok((None, None));
        };
        let Some(hash) = hex32(&pointer.hash) else {
            return Ok((None, None));
        };
        let mut catch_up = None;
        if role != "coordinator" {
            let pointer_head = discovery
                .fetch_checkpoint(run_id, "coordinator")
                .await
                .ok()
                .flatten()
                .map(|coord| coord.round);
            let verified_head = recovery.verified_head;
            if let Some(evidence) = verified_head.into_iter().chain(pointer_head).max() {
                let head = evidence.max(pointer.round);
                let horizon = daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS;
                // ANY fence gap stages archive catch-up when the verified lineage usefully
                // reaches it (defect 18, c15m live): the nominal ring horizon is NOT a replay
                // guarantee — a reconstructed coordinator's ring starts at its boot round, so
                // a within-horizon gap can still be unreplayable and the trainer loops
                // OUTCOME_STALE_RESTORE through the paced-respawn lane (fence 0, head 3,
                // horizon 16 — the c15m shape). Catch-up overlap with the live ring replay is
                // absorbed by the dedup window by design (Gate B'). "Usefully reaches": the
                // tip covers the fence AND stands within a ring of the head. Without that,
                // a within-ring gap proceeds bare (a young run's live ring is all there is)
                // and a past-ring gap refuses typed — the genuine CheckpointStale.
                if head > pointer.round {
                    let ring_nominal = head <= pointer.round.saturating_add(horizon);
                    let archive_reaches = !recovery.lineage_heads.is_empty()
                        && verified_head.is_some_and(|tip| {
                            tip >= pointer.round && tip.saturating_add(horizon) >= head
                        });
                    if archive_reaches {
                        catch_up = Some(protocol::TrainerCatchUp {
                            heads: recovery.lineage_heads.clone(),
                            from_round: pointer.round,
                        });
                        let detail = format!(
                            "{role}'s restore fence (round {}) trails the live head \
                             (round {head}); the verified archive lineage (tip round {}) \
                             covers the gap — staging archive catch-up before live attach \
                             (ring replay + dedup absorb any overlap)",
                            pointer.round,
                            verified_head.unwrap_or(0),
                        );
                        tracing::info!(run = run_id, "{detail}");
                        let mut emitted = Vec::new();
                        let _ = self.emit(
                            VhcEvent::Warning {
                                run_id: run_id.to_string(),
                                class: "archive_catch_up".to_string(),
                                detail,
                            },
                            &mut emitted,
                        );
                    } else if !ring_nominal {
                        return Err(VhcError::CheckpointStale {
                            restored: pointer.round,
                            head,
                            horizon,
                        });
                    }
                }
            }
        }
        // The node resolved this role's restore pointer (spec §9); the joining worker fetches
        // its by-reference checkpoint document and streams each family's windows via
        // chunk-keyed rehydration ([SF-6]). This is the node-visible half of the restore path.
        if pointer.role != role {
            let mut emitted = Vec::new();
            let _ = self.emit(
                VhcEvent::Warning {
                    run_id: run_id.to_string(),
                    class: "sibling_restore_adopted".to_string(),
                    detail: format!(
                        "{role} has no published checkpoint; restoring from sibling seat \
                         {}'s round-{} doc — adopting its replica-local (class-1) sections \
                         (consensus class-0 state is digest-covered and identical)",
                        pointer.role, pointer.round
                    ),
                },
                &mut emitted,
            );
            tracing::warn!(
                run = run_id,
                role,
                sibling = pointer.role,
                round = pointer.round,
                "sibling-seat checkpoint adopted for restore (no own-seat doc published)"
            );
        }
        tracing::info!(
            run = run_id,
            role,
            round = pointer.round,
            "resolved late-join checkpoint restore pointer (streaming by-ref rehydration follows)"
        );
        Ok((
            Some(protocol::CheckpointRestore {
                round: pointer.round,
                hash,
            }),
            catch_up,
        ))
    }

    /// Resolve the run's published archive lineage for the join (§8.8; the recovery half of the
    /// join transaction): fetch every stored archive-head record, VERIFY each one against the
    /// genesis-trusted bases (the registry stores, it never vouches —
    /// [`daemon_vhc_proto::verify_chains`]), order the seat role's chains founding-first by
    /// their succession links, and derive
    ///
    /// - the **coordinator reconstruction directive**: a SEAT-ROLE join with published history
    ///   must rebuild the coordinator's consensus state (retained ring + delivery cursors) from
    ///   its durable journal before reporting ready — the verified lineage's head records ride
    ///   the credentials ([`protocol::CoordinatorRecovery`]; carriage, not trust — the worker
    ///   re-verifies against the same genesis); and
    /// - the **verified head estimate**: the latest committed-round claim across the lineage's
    ///   signed heads — the freshness evidence [`Self::resolve_restore`]'s staleness judgment
    ///   compares against.
    ///
    /// No discovery seam / no published seat history resolves EMPTY (a genuinely fresh seat).
    /// An unverifiable snapshot refuses TYPED and fails the join closed: booting a fresh
    /// coordinator past history that exists but does not authenticate would fork the run — the
    /// node's retry schedule keeps it live, never a silent fresh boot.
    async fn resolve_recovery(
        &self,
        run_id: &str,
        role: &str,
    ) -> Result<RecoveryResolution, VhcError> {
        let Some(discovery) = self.discovery.as_ref() else {
            return Ok(RecoveryResolution::default());
        };
        let heads = discovery.fetch_archive_heads(run_id).await?;
        if heads.is_empty() {
            return Ok(RecoveryResolution::default());
        }
        // The run's cryptographic identity + trust set, from the frozen genesis (never registry
        // metadata, never ambient config).
        let bytes = discovery.fetch_envelope(run_id).await?;
        let wire: daemon_vhc_proto::SignedEnvelope = daemon_vhc_proto::from_canonical_slice(&bytes)
            .map_err(|e| VhcError::Discovery(format!("decode signed envelope: {e}")))?;
        let frozen = daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
            .map_err(|e| VhcError::Discovery(format!("verify genesis envelope: {e}")))?;
        let run_hash = *frozen.run_id();
        let env = frozen
            .decode()
            .map_err(|e| VhcError::Discovery(format!("decode genesis: {e}")))?;
        let trusted = daemon_vhc_proto::envelope_trusted_bases(&env);
        let chains = daemon_vhc_proto::verify_chains(&run_hash, &trusted, heads).map_err(|e| {
            VhcError::Discovery(format!("published archive heads do not verify: {e}"))
        })?;
        let seat_role = &self.config.seat_role;
        if !chains.iter().any(|c| &c.role == seat_role) {
            // Trainer chains may exist without a seat lineage (their custody is their own);
            // with no seat history there is no reconstruction and no freshness claim.
            return Ok(RecoveryResolution::default());
        }
        let lineage = daemon_vhc_proto::coordinator_lineage(&chains, seat_role)
            .map_err(|e| VhcError::Discovery(format!("archive lineage cannot be ordered: {e}")))?;
        let verified_head = daemon_vhc_proto::latest_round_claim(&lineage);
        let lineage_heads: Vec<daemon_vhc_proto::ArchiveHeadRecord> = lineage
            .iter()
            .flat_map(|chain| chain.heads.iter().cloned())
            .collect();
        let reconstruct = (role == seat_role).then(|| protocol::CoordinatorRecovery {
            heads: lineage_heads.clone(),
        });
        if let Some(directive) = &reconstruct {
            tracing::info!(
                run = run_id,
                chains = lineage.len(),
                heads = directive.heads.len(),
                round = verified_head,
                "resolved coordinator reconstruction directive from the verified archive lineage"
            );
        }
        Ok(RecoveryResolution {
            reconstruct,
            verified_head,
            lineage_heads,
        })
    }
}

/// What [`VhcService::resolve_recovery`] resolved from the run's published archive heads: the
/// seat-role reconstruction directive (a coordinator join only) and the latest VERIFIED
/// committed-round claim across the seat lineage (every join's staleness evidence). Both empty =
/// a fresh seat / no published history.
#[derive(Default)]
struct RecoveryResolution {
    /// The reconstruction directive to ride the credentials (`None` = fresh seat, or not the
    /// seat role).
    reconstruct: Option<protocol::CoordinatorRecovery>,
    /// The latest verified committed-round claim across the seat lineage's signed heads —
    /// simultaneously the freshness evidence for the restore staleness judgment AND the
    /// archive-coverage tip the Gate B' catch-up decision bridges with.
    verified_head: Option<u64>,
    /// The seat lineage's verified head records, founding order (the carriage for BOTH the
    /// coordinator reconstruction directive and a trainer catch-up directive). Empty = no
    /// published seat history.
    lineage_heads: Vec<daemon_vhc_proto::ArchiveHeadRecord>,
}

/// The join's node-resolved RESUME inputs, authored into the credentials as one unit: the role's
/// checkpoint restore pointer ([`VhcService::resolve_restore`]) and the §8.8 coordinator
/// reconstruction directive ([`VhcService::resolve_recovery`]).
struct JoinResume {
    restore: Option<protocol::CheckpointRestore>,
    reconstruct: Option<protocol::CoordinatorRecovery>,
    catch_up: Option<protocol::TrainerCatchUp>,
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

        // Defect 17 + defect 19a: the bring-up guard is acquired BEFORE the live-instance check
        // so the coalesce-vs-mint decision is made INSIDE the serialized section. The old order
        // (check, then guard) left a window where a join racing a COMPLETING bring-up read the
        // instance map before that transaction's insert, found nothing, then acquired the freed
        // guard and minted a superseding seat over a healthy one — cascading supersession churn
        // (c15m). Every completed bring-up inserts its instance before releasing the guard, so
        // a holder of the guard sees a settled map.
        let bring_up = self.begin_bring_up(&run_id);
        // A repeated join of a LIVE role-instance re-converges on the existing child + its
        // standing reservation — it never double-charges the ledgers (and needs no guard;
        // nothing is minted).
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

        // Defect 17: a FRESH bring-up serializes against the auto-resume reconvergence (and any
        // concurrent explicit join). Two racing bring-ups mint competing incarnations whose
        // supersession leaves the survivor a zombie — the second entrant refuses TYPED instead.
        let _bring_up = if existing.is_none() {
            match bring_up {
                Some(guard) => Some(guard),
                None => {
                    if fresh_child {
                        worker.shutdown().await;
                    }
                    return Err(VhcError::Internal(format!(
                        "a bring-up for `{run_id}` is already in flight (auto-resume or a \
                         concurrent join); the standing transaction owns this run — retry \
                         after it settles"
                    ))
                    .to_api());
                }
            }
        } else {
            drop(bring_up);
            None
        };

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
        let (coordinator, eligibility, assessed_tuple, seat_incarnation, directed_role) =
            match seat_join {
                Some((coordinator, eligibility, assessed_tuple, incarnation)) => {
                    (coordinator, eligibility, assessed_tuple, incarnation, None)
                }
                None => {
                    // Undirected: against a SEATED genesis this selects the trainer seat
                    // authored for this node's identity (defect 6).
                    let (coordinator, eligibility, assessed_tuple, directed) = self
                        .resolve_join(&worker, &run_id, None)
                        .await
                        .map_err(|e| e.to_api())?;
                    (coordinator, eligibility, assessed_tuple, None, directed)
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
                    // The seat-won join runs the configured seat role; else the persisted role;
                    // else the seat the genesis authored for this identity (defect 6), with the
                    // pre-seat `trainer` default last.
                    role: if seat_incarnation.is_some() {
                        self.config.seat_role.clone()
                    } else if !role.is_empty() {
                        role
                    } else {
                        directed_role.unwrap_or_else(|| "trainer".to_string())
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
        // The recovery half of the join transaction (§8.8): the verified archive lineage — the
        // reconstruction directive for a seat-role join with published history, and the
        // verified head estimate the restore staleness judgment consumes.
        let recovery = match self.resolve_recovery(&run_id, &id.role).await {
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
        let (restore, catch_up) = match self.resolve_restore(&run_id, &id.role, &recovery).await {
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
                JoinResume {
                    restore: restore.clone(),
                    reconstruct: recovery.reconstruct.clone(),
                    catch_up: catch_up.clone(),
                },
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
                        self.author_join(
                            &run_id,
                            &coordinator,
                            &id,
                            assessed_tuple,
                            JoinResume {
                                restore,
                                reconstruct: recovery.reconstruct,
                                catch_up,
                            },
                            seat,
                        )
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
        // The owner's leave cancels any pending sibling respawn (defect 15's paced lane).
        self.co_retry.lock().unwrap().remove(&run_id);
        // The run holds no role-instance here anymore: this node's iroh endpoint is unowned.
        self.forget_node_iroh_endpoint(&run_id);
        self.ckpt_lag.lock().unwrap().remove(&run_id);
        self.progress.lock().unwrap().remove(&run_id);
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

    async fn vhc_disk_usage(&self) -> Result<VhcDiskUsage, ApiError> {
        const MIB: u64 = 1024 * 1024;
        let Some(root) = &self.run_dir else {
            return Err(ApiError::Unsupported("vhc_disk_usage: no runs root".into()));
        };
        let custodian =
            daemon_vhc_custody::DiskCustodian::for_root(root, custody_config(&self.config.storage))
                .map_err(|e| ApiError::Other(format!("disk custodian: {e}")))?;
        let usage = custodian.usage();
        // Map on-disk scopes back to run labels + liveness; walk the DISK for the rows (the
        // ledger is per-process; the disk is the operator's truth).
        let runs = self
            .store
            .list_runs()
            .map_err(|e| VhcError::from(e).to_api())?;
        let label_by_scope: BTreeMap<String, String> = runs
            .iter()
            .map(|r| {
                (
                    blake3::hash(r.run_id.as_bytes()).to_hex().to_string(),
                    r.run_id.clone(),
                )
            })
            .collect();
        let live_scopes: std::collections::BTreeSet<String> = {
            let instances = self.instances.lock().unwrap();
            instances
                .keys()
                .map(|label| blake3::hash(label.as_bytes()).to_hex().to_string())
                .collect()
        };
        let mut scopes: Vec<VhcDiskScope> = crate::reclaim::scope_rows(root)
            .into_iter()
            .map(|(scope, split)| VhcDiskScope {
                run_id: label_by_scope.get(&scope).cloned(),
                active: live_scopes.contains(&scope),
                scope,
                recoverable_mb: split.recoverable.div_ceil(MIB),
                evidence_mb: split.evidence.div_ceil(MIB),
            })
            .collect();
        scopes.sort_by(|a, b| {
            (b.recoverable_mb + b.evidence_mb).cmp(&(a.recoverable_mb + a.evidence_mb))
        });
        Ok(VhcDiskUsage {
            root: usage.root,
            free_mb: usage.free_bytes / MIB,
            used_mb: usage.used_bytes.div_ceil(MIB),
            quota_mb: usage.quota_bytes / MIB,
            reserve_mb: usage.reserve_bytes / MIB,
            emergency_mb: usage.emergency_bytes / MIB,
            pressure: match usage.pressure {
                daemon_vhc_custody::Pressure::Nominal => "nominal".into(),
                daemon_vhc_custody::Pressure::Warn => "warn".into(),
                daemon_vhc_custody::Pressure::RefuseNew => "refuse_new".into(),
            },
            scopes,
        })
    }

    async fn vhc_disk_wipe(
        &self,
        run_id: String,
        include_evidence: bool,
    ) -> Result<VhcDiskWipeOutcome, ApiError> {
        const MIB: u64 = 1024 * 1024;
        let Some(root) = &self.run_dir else {
            return Err(ApiError::Unsupported("vhc_disk_wipe: no runs root".into()));
        };
        // Refuse while the run is live or the node still intends to run it: a wipe under a
        // standing joined intent would race the very reconstruction the journal feeds.
        if self.instances.lock().unwrap().contains_key(&run_id) {
            return Err(ApiError::Conflict(format!(
                "run `{run_id}` is live; leave or pause it before wiping"
            )));
        }
        let intents = self
            .store
            .active_intents()
            .map_err(|e| VhcError::from(e).to_api())?;
        if intents.iter().any(|r| r.run_id == run_id) {
            return Err(ApiError::Conflict(format!(
                "run `{run_id}` holds a standing joined intent; leave it before wiping"
            )));
        }
        let scope = blake3::hash(run_id.as_bytes()).to_hex().to_string();
        let outcome = crate::reclaim::wipe_scope(&root.join(&scope), include_evidence)
            .map_err(|e| ApiError::Other(format!("wipe `{run_id}`: {e}")))?;
        // Keep the custodian's ledger honest about what left the disk.
        if let Ok(custodian) =
            daemon_vhc_custody::DiskCustodian::for_root(root, custody_config(&self.config.storage))
        {
            if include_evidence {
                custodian.forget_scope(&scope);
            } else {
                custodian.discharge(&scope, outcome.bytes);
            }
        }
        tracing::info!(
            run = run_id,
            bytes = outcome.bytes,
            evidence = outcome.wiped_evidence,
            "safe wipe reclaimed run state"
        );
        let mut preserved = vec!["identity keystore (base.key + run keys)".to_string()];
        if !include_evidence {
            preserved.push("archive planes (payload + heads)".to_string());
        }
        Ok(VhcDiskWipeOutcome {
            run_id,
            reclaimed_mb: outcome.bytes.div_ceil(MIB),
            wiped_evidence: outcome.wiped_evidence,
            preserved,
        })
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

/// The transport-deferral pacing (Gate C): half the backoff ceiling plus a per-(run, moment)
/// jitter across the other half — so a fleet knocked over by one shared outage never retries in
/// lockstep, and each individual deferral still lands within `[max/2, max]`. Deterministic
/// inputs (run id + wall clock) through the std hasher — pacing needs spread, not secrecy, so
/// no rand dependency is warranted.
fn transport_backoff_jittered_ms(
    cfg: &daemon_vhc_session::config::RetryConfig,
    run_id: &str,
) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let ceiling = cfg.max_backoff_ms.max(cfg.initial_backoff_ms).max(2);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    run_id.hash(&mut hasher);
    now_ms().hash(&mut hasher);
    let jitter = hasher.finish() % (ceiling / 2);
    ceiling / 2 + jitter
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
            run_dir: None,
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

    /// **Defect 15 (c15k): a retryable-class co-located trainer terminal arms the paced respawn
    /// lane; a deliberate end never does.** Pre-fix the co arm of `handle_run_terminated` removed
    /// the sibling entry and returned — the primary seat instance stays live, so the run-level
    /// retry lane never fires for the sibling, and the run silently lost half its local
    /// membership (the c15k trainer's `OUTCOME_STALE_RESTORE` after coordinator reconstruction
    /// wedged the run below its floor for good).
    #[test]
    fn a_retryable_co_trainer_terminal_arms_the_paced_respawn_lane() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // The retryable sibling terminal: entry torn down, duty released, lane ARMED.
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
        svc.handle_worker_event(&protocol::Event::RunTerminated {
            run_id: "run-a".to_string(),
            generation: 8,
            outcome: protocol::TerminalOutcome::FailedRetryable {
                reason: "guest outcome 3 (stale restore)".to_string(),
            },
        })
        .expect("terminal handled");
        assert_eq!(svc.arbiter.remaining().duty_pct, 100, "duty released");
        {
            let lane = svc.co_retry.lock().unwrap();
            let (attempts, due) = lane.get("run-a").expect("the respawn lane is armed");
            assert_eq!(*attempts, 1);
            assert!(*due > now_ms(), "the respawn is PACED, not immediate");
        }

        // A deliberate end (completed / left): torn down, and NO respawn is ever scheduled.
        let id = instance_id("trainer", 9);
        let charge = svc
            .derive_charge(&eligibility(2 << 30, 2 << 30), &policy(100), "trainer")
            .unwrap();
        svc.admit_placed(&id, charge, 100).expect("trainer admits");
        svc.co_trainers.lock().unwrap().insert(
            "run-b".to_string(),
            InstanceEntry {
                id,
                generation: 9,
                worker: Arc::new(NoopWorker),
            },
        );
        svc.handle_worker_event(&protocol::Event::RunTerminated {
            run_id: "run-b".to_string(),
            generation: 9,
            outcome: protocol::TerminalOutcome::Completed { outcome: 0 },
        })
        .expect("terminal handled");
        assert!(
            !svc.co_retry.lock().unwrap().contains_key("run-b"),
            "a deliberate sibling end never respawns"
        );
    }

    /// **Defect 15, the repair side:** the reconcile tick's co-trainer pass fires a due lane
    /// entry — a live seat-holding run gets a respawn ATTEMPT (re-armed with grown backoff when
    /// the attempt refuses, so the repair survives transient refusals without a hot loop), and
    /// a lane entry whose run is gone is cleared, never retried forever.
    #[tokio::test]
    async fn the_reconcile_repair_pass_fires_due_co_trainer_lanes_and_clears_dead_ones() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // A LIVE seat-holding run: durable row (running, seat role) + live primary instance.
        svc.store
            .put_join_intent(
                "run-a",
                "wss://example/run-a",
                &policy(100),
                None,
                &eligibility(2 << 30, 2 << 30),
            )
            .expect("row");
        svc.store.mark_running("run-a").expect("running");
        svc.store
            .set_execution_identity("run-a", 0, "coordinator", 1)
            .expect("identity");
        svc.instances.lock().unwrap().insert(
            "run-a".to_string(),
            InstanceEntry {
                id: instance_id("coordinator", 1),
                generation: 1,
                worker: Arc::new(NoopWorker),
            },
        );

        // Both lanes due in the past: one live run, one the store never heard of.
        svc.co_retry
            .lock()
            .unwrap()
            .insert("run-a".to_string(), (1, now_ms() - 1));
        svc.co_retry
            .lock()
            .unwrap()
            .insert("ghost".to_string(), (3, now_ms() - 1));

        svc.reconcile_tick().await.expect("tick");

        {
            let lane = svc.co_retry.lock().unwrap();
            assert!(
                !lane.contains_key("ghost"),
                "a lane whose run no longer exists is cleared, not retried forever"
            );
            // The NoopWorker cannot complete a real join, so the attempt refused — the lane
            // re-arms with GROWN backoff instead of dropping the intent (the repair stays
            // alive) and never leaves a sibling entry behind.
            let (attempts, due) = lane
                .get("run-a")
                .expect("the live run's lane survives a refused attempt");
            assert!(*attempts >= 2, "the attempt count grew, got {attempts}");
            assert!(*due > now_ms(), "re-armed paced, not hot-looped");
        }
        assert!(
            !svc.co_trainers.lock().unwrap().contains_key("run-a"),
            "no sibling entry appears from a refused attempt"
        );
    }

    /// **Defect 19 (c15m), the lane-survival half:** a due co-trainer lane whose run is in a
    /// TRANSIENT window (the primary seat mid-replacement — row standing, instance absent)
    /// re-arms with grown backoff instead of being cleared. The old pass cleared it, so once
    /// the seat came back no trainer respawn ever fired again (trainer-0 stayed dead for 2+
    /// hours across five seat replacements). A GENUINE teardown (intent withdrawn) still
    /// clears.
    #[tokio::test]
    async fn a_due_co_trainer_lane_survives_a_transient_seat_replacement_window() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // The transient shape: a seat-holding Running row whose primary instance is ABSENT
        // (mid-replacement — the next bring-up will re-insert it).
        svc.store
            .put_join_intent(
                "run-a",
                "wss://example/run-a",
                &policy(100),
                None,
                &eligibility(2 << 30, 2 << 30),
            )
            .expect("row");
        svc.store.mark_running("run-a").expect("running");
        svc.store
            .set_execution_identity("run-a", 0, "coordinator", 1)
            .expect("identity");
        svc.co_retry
            .lock()
            .unwrap()
            .insert("run-a".to_string(), (1, now_ms() - 1));

        svc.reconcile_tick().await.expect("tick");
        {
            let lane = svc.co_retry.lock().unwrap();
            let (attempts, due) = lane
                .get("run-a")
                .expect("the lane SURVIVES the transient window");
            assert!(
                *attempts >= 2,
                "re-armed with grown backoff, got {attempts}"
            );
            assert!(*due > now_ms(), "paced, not hot-looped");
        }

        // The genuine teardown: the owner withdrew the intent — the lane clears.
        svc.store
            .set_desired_state("run-a", DesiredState::Left)
            .expect("intent withdrawn");
        svc.co_retry
            .lock()
            .unwrap()
            .insert("run-a".to_string(), (1, now_ms() - 1));
        svc.reconcile_tick().await.expect("tick");
        assert!(
            !svc.co_retry.lock().unwrap().contains_key("run-a"),
            "a withdrawn intent clears the lane"
        );
    }

    /// **Defect 19 (c15m), the stale-sibling half:** a fresh seat bring-up that finds a
    /// REGISTERED co-trainer entry reaps it (reservation released, entry removed) before
    /// spawning — the entry can only belong to a replaced owner (every caller gates on a fresh
    /// seat mint or an empty lane). The old idempotence early-return kept the corpse
    /// registered, which blocked every respawn while its stream-closure cleanup raced the next
    /// bring-up.
    #[tokio::test]
    async fn a_fresh_seat_bring_up_reaps_the_superseded_owners_co_trainer_sibling() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // The superseded owner's sibling: registered, holding a duty reservation.
        let id = instance_id("trainer", 7);
        let charge = svc
            .derive_charge(&eligibility(2 << 30, 2 << 30), &policy(100), "trainer")
            .unwrap();
        svc.admit_placed(&id, charge, 100).expect("trainer admits");
        assert!(svc.arbiter.remaining().duty_pct < 100, "duty held");
        svc.co_trainers.lock().unwrap().insert(
            "run-a".to_string(),
            InstanceEntry {
                id,
                generation: 7,
                worker: Arc::new(NoopWorker),
            },
        );

        // The fresh bring-up's sibling spawn: the NoopWorker cannot pass assess, so no NEW
        // sibling lands — but the stale one must be reaped and its reservation released
        // regardless.
        svc.spawn_co_located_trainer("run-a", &policy(100)).await;
        assert!(
            !svc.co_trainers.lock().unwrap().contains_key("run-a"),
            "the stale sibling entry was reaped, not kept by the old idempotence early-return"
        );
        assert_eq!(
            svc.arbiter.remaining().duty_pct,
            100,
            "the superseded sibling's duty reservation was released"
        );
    }

    /// **Defect 17 (c15k): an explicit join and the auto-resume reconvergence SERIALIZE.**
    /// Pre-fix both bring-up transactions ran concurrently through assess → mint → join,
    /// minting competing incarnations (105/107/110 live) whose supersession left the survivor
    /// a zombie — certified outbound, refused inbound, no liveness signal. The guard makes the
    /// second entrant refuse typed (explicit join) or yield (reconvergence), and releases on
    /// every exit path.
    #[tokio::test]
    async fn a_join_during_an_in_flight_bring_up_refuses_typed_and_reconverge_yields() {
        let probe = Hardware {
            gpus: 1,
            vram_mb: 25_559,
            ram_mb: 32_000,
            ..Hardware::default()
        };
        let budget = OwnerBudget::from_config(&OwnerBudgetConfig::default(), Some(&probe), 50);
        let svc = coordinator_trainer_service(budget);

        // An auto-resume bring-up holds the run's transaction.
        let guard = svc.begin_bring_up("run-a").expect("first entrant enters");
        assert!(
            svc.begin_bring_up("run-a").is_none(),
            "a second entrant never enters the same run's bring-up"
        );

        // The explicit join refuses TYPED without touching the worker or minting anything.
        let err = svc
            .vhc_join("run-a".into(), policy(100), "op".into())
            .await
            .expect_err("a join during an in-flight bring-up refuses");
        assert!(
            err.to_string().contains("already in flight"),
            "the refusal names the standing transaction: {err}"
        );
        assert!(
            svc.instances.lock().unwrap().is_empty(),
            "no competing instance was created"
        );

        // The reconvergence half yields Ok without minting a competing incarnation.
        svc.store
            .put_join_intent(
                "run-a",
                "wss://example/run-a",
                &policy(100),
                None,
                &eligibility(2 << 30, 2 << 30),
            )
            .expect("row");
        let run = svc.store.get_run("run-a").expect("read").expect("row");
        svc.reconverge(&run).await.expect("reconverge yields");
        assert!(
            svc.instances.lock().unwrap().is_empty(),
            "the yielding reconvergence created nothing"
        );

        // Release: the next entrant proceeds (and the guard is re-entrant-safe by run id).
        drop(guard);
        let reacquired = svc.begin_bring_up("run-a");
        assert!(
            reacquired.is_some(),
            "the transaction slot releases with the guard"
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

    /// A `FailedTransport` terminal (Gate C, defect 10 — a transient network fault, e.g. the
    /// content plane unreachable during coordinator reconstruction) defers BUDGET-FREE: the
    /// row stays `failed_retryable` with the join intent retained, `retry_count` is untouched,
    /// the next attempt is scheduled with jittered pacing inside `[ceiling/2, ceiling]`, the
    /// deferral is surfaced typed (`transport_deferred` warning + prefixed reason — the
    /// `vhc detail` evidence), and repeated transport terminals can NEVER escalate to
    /// terminal — an outage is waited out, not crash-looped.
    #[test]
    fn a_transport_terminal_defers_budget_free_with_jittered_pacing() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-t", 7);

        let before = now_ms();
        let emitted = svc
            .handle_run_terminated(
                "run-t",
                7,
                &protocol::TerminalOutcome::FailedTransport {
                    reason: "coordinator reconstruction: transient transport fault (connect): \
                             egress: request failed"
                        .into(),
                },
            )
            .expect("the terminal transitions");

        let row = svc.store.get_run("run-t").expect("read").expect("row");
        assert_eq!(row.run_state, RunState::FailedRetryable);
        assert!(!row.storage_gated, "transport is not the storage gate");
        assert_eq!(
            row.retry_count, 0,
            "a transport deferral never consumes the retry budget"
        );
        let due = row.next_retry_ms.expect("a paced retry is scheduled");
        let ceiling = svc.config.retry.max_backoff_ms as i64;
        assert!(
            due >= before + ceiling / 2 && due <= now_ms() + ceiling,
            "the jittered due time lands within [ceiling/2, ceiling]: due={due} before={before}"
        );
        assert!(
            row.terminal_reason
                .as_deref()
                .is_some_and(|r| r.starts_with("transport fault (deferred budget-free):")),
            "the reason is visibly prefixed for vhc detail: {:?}",
            row.terminal_reason
        );
        assert!(
            emitted.iter().any(|e| matches!(
                e,
                VhcEvent::Warning { class, .. } if class == "transport_deferred"
            )),
            "the deferral surfaces as a typed warning event"
        );

        // Indefinite by design: run it past the ordinary budget — no escalation, no budget.
        for _ in 0..svc.config.retry.max_retries + 2 {
            seed_joined_run(&svc, "run-t", 7);
            svc.handle_run_terminated(
                "run-t",
                7,
                &protocol::TerminalOutcome::FailedTransport {
                    reason: "still unreachable".into(),
                },
            )
            .expect("transitions again");
        }
        let row = svc.store.get_run("run-t").expect("read").expect("row");
        assert_eq!(
            row.run_state,
            RunState::FailedRetryable,
            "a transport outage never escalates to failed_terminal"
        );
        assert_eq!(row.retry_count, 0, "…and never consumes budget");
    }

    /// Gate D' restore honesty: adopting a SIBLING seat's checkpoint doc (the fallback when the
    /// seat itself published nothing) is a recorded posture — a persisted
    /// `sibling_restore_adopted` warning naming both seats and the adopted round — because the
    /// doc carries the sibling's class-1 replica-local sections. An own-seat restore records
    /// nothing (nothing foreign was adopted).
    #[tokio::test]
    async fn a_sibling_restore_adoption_is_recorded_as_a_persisted_warning() {
        struct SiblingOnly;
        #[async_trait]
        impl crate::discovery::RunDiscovery for SiblingOnly {
            async fn list_runs(&self) -> Result<Vec<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(Vec::new())
            }
            async fn get_run(
                &self,
                _run_id: &str,
            ) -> Result<Option<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(None)
            }
            async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
                Ok(Vec::new())
            }
            async fn fetch_checkpoint(
                &self,
                _run_id: &str,
                role: &str,
            ) -> Result<Option<crate::discovery::CheckpointPointer>, VhcError> {
                // Seat 1 published nothing; seat 0's doc stands (the Gate D' fallback shape).
                // The coordinator has no pointer, so no staleness evidence exists.
                if role == "coordinator" {
                    return Ok(None);
                }
                Ok(Some(crate::discovery::CheckpointPointer {
                    role: "trainer-0".into(),
                    kind: "live".into(),
                    round: 8,
                    hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
                    size: 2048,
                }))
            }
        }

        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        svc.discovery = Some(Arc::new(SiblingOnly));

        let (restore, catch_up) = svc
            .resolve_restore("run-d", "trainer-1", &RecoveryResolution::default())
            .await
            .expect("the sibling fallback resolves");
        let restore = restore.expect("a restore pointer stands");
        assert_eq!(restore.round, 8);
        assert!(
            catch_up.is_none(),
            "no head evidence, no catch-up directive"
        );

        let events = svc.store.recent_events("run-d", 16).expect("events read");
        let warning = events
            .iter()
            .find_map(|e| match e {
                VhcEvent::Warning { class, detail, .. } if class == "sibling_restore_adopted" => {
                    Some(detail.clone())
                }
                _ => None,
            })
            .expect("the adoption is a persisted warning");
        assert!(
            warning.contains("trainer-1") && warning.contains("trainer-0"),
            "the warning names both seats: {warning}"
        );
    }

    /// Gate B' catch-up decision (defect 14 closed properly): a restore fence past the ring
    /// horizon is no longer an unconditional `CheckpointStale` — when the verified archive
    /// lineage reaches within a ring of the head, the join carries a staged catch-up directive
    /// (the lineage heads + the fence) and a persisted `archive_catch_up` warning; the typed
    /// refusal remains ONLY for the gap the planes genuinely cannot bridge (archive tip itself
    /// past ring reach, or no lineage at all).
    #[tokio::test]
    async fn a_stale_fence_bridged_by_the_archive_lineage_rides_a_catch_up_directive() {
        struct StaleShape;
        #[async_trait]
        impl crate::discovery::RunDiscovery for StaleShape {
            async fn list_runs(&self) -> Result<Vec<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(Vec::new())
            }
            async fn get_run(
                &self,
                _run_id: &str,
            ) -> Result<Option<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(None)
            }
            async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
                Ok(Vec::new())
            }
            async fn fetch_checkpoint(
                &self,
                _run_id: &str,
                role: &str,
            ) -> Result<Option<crate::discovery::CheckpointPointer>, VhcError> {
                // The c15h shape scaled to the production horizon: the trainer's freshest
                // fence (round 4) trails the coordinator's live head (round 30) past the
                // 16-round ring.
                let round = if role == "coordinator" { 30 } else { 4 };
                Ok(Some(crate::discovery::CheckpointPointer {
                    role: role.into(),
                    kind: "live".into(),
                    round,
                    hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
                    size: 2048,
                }))
            }
        }

        // One signed head stands in for the verified seat lineage (authenticity was already
        // resolve_recovery's job; this decision consumes its outputs).
        let base = daemon_vhc_proto::SigningKey::from_bytes(&[0xB0; 32]);
        let run_key = daemon_vhc_proto::SigningKey::from_bytes(&[0x4A; 32]);
        let cert = daemon_vhc_proto::RunKeyCertificate::issue(
            &base,
            daemon_vhc_proto::CertScope {
                run_id: daemon_vhc_proto::Hash([0x1D; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: daemon_vhc_proto::Hash([0x2A; 32]),
            },
            daemon_vhc_proto::peer_id(&run_key),
        )
        .expect("cert issues");
        let head = daemon_vhc_proto::ArchiveHeadRecord::publish(
            &run_key,
            cert,
            daemon_vhc_proto::ArchiveHeadBody {
                domain: daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN.into(),
                run_id: daemon_vhc_proto::Hash([0x1D; 32]),
                role: "coordinator".into(),
                chain_instance: 1,
                segment: 0,
                segment_hash: daemon_vhc_proto::Hash([0xAA; 32]),
                prev_hash: daemon_vhc_proto::Hash([0; 32]),
                records: 1,
                instance: 1,
                epoch: 0,
                module: daemon_vhc_proto::Hash([0x2A; 32]),
                predecessor: None,
                round: Some(28),
            },
        )
        .expect("head publishes");

        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        svc.discovery = Some(Arc::new(StaleShape));

        // Bridgeable: the archive tip (28) reaches within a ring (16) of the head (30) —
        // the join proceeds, carrying the catch-up directive fenced at the restore round.
        let bridged = RecoveryResolution {
            reconstruct: None,
            verified_head: Some(28),
            lineage_heads: vec![head.clone()],
        };
        let (restore, catch_up) = svc
            .resolve_restore("run-b", "trainer-1", &bridged)
            .await
            .expect("a bridgeable gap is not a refusal");
        assert_eq!(restore.expect("the fence still restores").round, 4);
        let directive = catch_up.expect("the catch-up directive rides the join");
        assert_eq!(directive.from_round, 4, "fenced at the restore round");
        assert_eq!(directive.heads.len(), 1, "the verified lineage rides along");
        let events = svc.store.recent_events("run-b", 16).expect("events read");
        assert!(
            events.iter().any(|e| matches!(
                e,
                VhcEvent::Warning { class, .. } if class == "archive_catch_up"
            )),
            "the staged catch-up is a persisted warning"
        );

        // Unbridgeable: the archive tip (10) is itself past ring reach of the head — the
        // genuine CheckpointStale, typed with the real shape.
        let gapped = RecoveryResolution {
            reconstruct: None,
            verified_head: Some(10),
            lineage_heads: vec![head],
        };
        let err = svc
            .resolve_restore("run-b", "trainer-1", &gapped)
            .await
            .expect_err("an archive gap refuses");
        assert!(
            matches!(
                err,
                VhcError::CheckpointStale {
                    restored: 4,
                    head: 30,
                    horizon: daemon_vhc_proto::RETAINED_RECORD_HORIZON_ROUNDS,
                }
            ),
            "typed with the genuine shape, got {err:?}"
        );

        // No lineage at all (head evidence only from the registry pointer): equally refused.
        let err = svc
            .resolve_restore("run-b", "trainer-1", &RecoveryResolution::default())
            .await
            .expect_err("no lineage cannot bridge");
        assert!(matches!(err, VhcError::CheckpointStale { .. }));
    }

    /// Defect 18 (c15m live): a WITHIN-horizon fence gap must still stage archive catch-up
    /// when the verified lineage covers it — the nominal ring horizon is not a replay
    /// guarantee. A reconstructed coordinator's ring starts at its boot round, so the live
    /// shape (fence 0, head 3, horizon 16) replayed nothing before round 3: the respawned
    /// trainer looped `OUTCOME_STALE_RESTORE` through the paced-respawn lane every ~15 min
    /// with no catch-up ever staged, because the old trigger required the gap to EXCEED the
    /// horizon. Also pins the young-run posture: the same small gap with NO published lineage
    /// proceeds bare (no directive, no refusal) — the live ring is all there is.
    #[tokio::test]
    async fn a_within_horizon_gap_after_reconstruction_still_stages_archive_catch_up() {
        struct ReconstructedShape;
        #[async_trait]
        impl crate::discovery::RunDiscovery for ReconstructedShape {
            async fn list_runs(&self) -> Result<Vec<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(Vec::new())
            }
            async fn get_run(
                &self,
                _run_id: &str,
            ) -> Result<Option<crate::discovery::DiscoveredRun>, VhcError> {
                Ok(None)
            }
            async fn fetch_envelope(&self, _run_id: &str) -> Result<Vec<u8>, VhcError> {
                Ok(Vec::new())
            }
            async fn fetch_checkpoint(
                &self,
                _run_id: &str,
                role: &str,
            ) -> Result<Option<crate::discovery::CheckpointPointer>, VhcError> {
                // The c15m shape verbatim: the co-trainer's only checkpoint is the round-0
                // slot; the coordinator was killed with round 3 committed.
                let round = if role == "coordinator" { 3 } else { 0 };
                Ok(Some(crate::discovery::CheckpointPointer {
                    role: role.into(),
                    kind: "live".into(),
                    round,
                    hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
                    size: 2048,
                }))
            }
        }

        let base = daemon_vhc_proto::SigningKey::from_bytes(&[0xB1; 32]);
        let run_key = daemon_vhc_proto::SigningKey::from_bytes(&[0x4B; 32]);
        let cert = daemon_vhc_proto::RunKeyCertificate::issue(
            &base,
            daemon_vhc_proto::CertScope {
                run_id: daemon_vhc_proto::Hash([0x1E; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: daemon_vhc_proto::Hash([0x2B; 32]),
            },
            daemon_vhc_proto::peer_id(&run_key),
        )
        .expect("cert issues");
        let head = daemon_vhc_proto::ArchiveHeadRecord::publish(
            &run_key,
            cert,
            daemon_vhc_proto::ArchiveHeadBody {
                domain: daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN.into(),
                run_id: daemon_vhc_proto::Hash([0x1E; 32]),
                role: "coordinator".into(),
                chain_instance: 1,
                segment: 0,
                segment_hash: daemon_vhc_proto::Hash([0xAB; 32]),
                prev_hash: daemon_vhc_proto::Hash([0; 32]),
                records: 1,
                instance: 1,
                epoch: 0,
                module: daemon_vhc_proto::Hash([0x2B; 32]),
                predecessor: None,
                round: Some(2),
            },
        )
        .expect("head publishes");

        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        svc.discovery = Some(Arc::new(ReconstructedShape));

        // Lineage present (tip 2 covers fence 0, within a ring of head 3): the within-horizon
        // gap stages catch-up — the leg the old past-horizon-only trigger dropped.
        let recovered = RecoveryResolution {
            reconstruct: None,
            verified_head: Some(2),
            lineage_heads: vec![head],
        };
        let (restore, catch_up) = svc
            .resolve_restore("run-e", "trainer-0", &recovered)
            .await
            .expect("a covered gap is never a refusal");
        assert_eq!(restore.expect("the fence still restores").round, 0);
        let directive = catch_up.expect("defect 18: the within-horizon gap stages catch-up");
        assert_eq!(directive.from_round, 0, "fenced at the restore round");
        assert_eq!(directive.heads.len(), 1, "the verified lineage rides along");
        let events = svc.store.recent_events("run-e", 16).expect("events read");
        assert!(
            events.iter().any(|e| matches!(
                e,
                VhcEvent::Warning { class, .. } if class == "archive_catch_up"
            )),
            "the staged catch-up is a persisted warning"
        );

        // Young run (same gap, NO published lineage): bare restore — no directive, no refusal;
        // the live ring is the only replay plane that exists.
        let (restore, catch_up) = svc
            .resolve_restore("run-f", "trainer-0", &RecoveryResolution::default())
            .await
            .expect("a within-ring gap with no archive proceeds");
        assert_eq!(restore.expect("the fence still restores").round, 0);
        assert!(
            catch_up.is_none(),
            "nothing to extract from an unpublished lineage"
        );
    }

    /// Gate B' durable-checkpoint lag alarm (defect 14's unvoiced drift): a trainer whose
    /// freshest DURABLE checkpoint trails the live head by the warn margin gets a persisted
    /// `checkpoint_lag` warning — voiced when the drift GROWS (never a per-round flood), closed
    /// by a fresh durable fence, and re-armed for a new drift.
    #[tokio::test]
    async fn checkpoint_lag_past_the_horizon_margin_is_a_persisted_warning() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-lag", 7);
        svc.store
            .set_execution_identity("run-lag", 0, "trainer-0", 7)
            .expect("the row carries the trainer role");
        svc.handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-lag".into(),
            phase: "running".into(),
            epoch: 0,
            round: 0,
            generation: 7,
        })
        .expect("phase attributes the run");

        let outcome = |round| protocol::Event::RoundOutcome {
            round,
            committed: 2,
            ingested: 2,
            stalled: false,
            digest: [0u8; 16],
            generation: 7,
        };
        let lag_warning = |outs: &[VhcEvent]| {
            outs.iter().find_map(|e| match e {
                VhcEvent::Warning { class, detail, .. } if class == "checkpoint_lag" => {
                    Some(detail.clone())
                }
                _ => None,
            })
        };

        // Under the margin nothing is voiced.
        let outs = svc
            .handle_worker_event(&outcome(CHECKPOINT_LAG_WARN_ROUNDS - 1))
            .expect("outcome handled");
        assert!(lag_warning(&outs).is_none(), "no warning under the margin");

        // Crossing it with NO durable checkpoint at all names the from-scratch exposure.
        let outs = svc
            .handle_worker_event(&outcome(CHECKPOINT_LAG_WARN_ROUNDS))
            .expect("outcome handled");
        let warning = lag_warning(&outs).expect("the crossing is voiced");
        assert!(
            warning.contains("NO durable checkpoint") && warning.contains("trainer-0"),
            "the no-fence exposure is named: {warning}"
        );

        // An unchanged lag does not re-voice.
        let outs = svc
            .handle_worker_event(&outcome(CHECKPOINT_LAG_WARN_ROUNDS))
            .expect("outcome handled");
        assert!(lag_warning(&outs).is_none(), "an unchanged lag stays quiet");

        // A fresh durable fence closes the drift…
        svc.handle_worker_event(&protocol::Event::CheckpointPublished {
            round: CHECKPOINT_LAG_WARN_ROUNDS,
            hash: "abc".into(),
            location: "payload/abc".into(),
            generation: 7,
            kind: "live".into(),
        })
        .expect("checkpoint handled");
        let outs = svc
            .handle_worker_event(&outcome(CHECKPOINT_LAG_WARN_ROUNDS + 1))
            .expect("outcome handled");
        assert!(
            lag_warning(&outs).is_none(),
            "a fresh fence closes the drift"
        );

        // …and a NEW drift past the margin re-warns, naming the stale fence.
        let outs = svc
            .handle_worker_event(&outcome(2 * CHECKPOINT_LAG_WARN_ROUNDS))
            .expect("outcome handled");
        let warning = lag_warning(&outs).expect("a new drift is voiced");
        assert!(
            warning.contains(&format!("round {CHECKPOINT_LAG_WARN_ROUNDS}")),
            "the stale fence is named: {warning}"
        );

        // The warning is persisted (the `vhc detail` recent-events window), not only broadcast.
        let events = svc.store.recent_events("run-lag", 64).expect("events read");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VhcEvent::Warning { class, .. } if class == "checkpoint_lag")),
            "the lag warning is a persisted event"
        );
    }

    /// REL-5 stall observer (reliability spec §6): a joined, alive run whose committed progress
    /// ages past the per-run threshold is announced `run_stalled` exactly ONCE per episode, and
    /// a committed round closes the episode with `run_progress_resumed` — one stateful
    /// transition each way, never a recurring alarm.
    #[tokio::test]
    async fn a_stalled_run_is_announced_once_and_closed_by_committed_progress() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-stall", 4);
        svc.store.mark_running("run-stall").expect("running");
        let stall_count = |svc: &VhcService| {
            svc.store
                .recent_events("run-stall", 64)
                .expect("events read")
                .into_iter()
                .filter(|e| matches!(e, VhcEvent::Warning { class, .. } if class == "run_stalled"))
                .count()
        };

        // The first tick creates the watermark at readiness; a fresh run never warns.
        svc.reconcile_tick().await.expect("tick");
        assert_eq!(stall_count(&svc), 0, "a fresh watermark does not warn");

        // Age the committed watermark past the floor: ONE warning, then silence.
        {
            let mut progress = svc.progress.lock().unwrap();
            let t = progress
                .get_mut("run-stall")
                .expect("track created at readiness");
            t.committed_at_ms -= RUN_STALL_WARN_FLOOR_MS + 60_000;
        }
        svc.reconcile_tick().await.expect("tick");
        assert_eq!(stall_count(&svc), 1, "the stall is voiced");
        svc.reconcile_tick().await.expect("tick");
        assert_eq!(stall_count(&svc), 1, "an announced episode never re-voices");
        let detail = svc
            .store
            .recent_events("run-stall", 64)
            .expect("events read")
            .into_iter()
            .find_map(|e| match e {
                VhcEvent::Warning { class, detail, .. } if class == "run_stalled" => Some(detail),
                _ => None,
            })
            .expect("the stall detail persists");
        assert!(
            detail.contains("no committed round for") && detail.contains("threshold"),
            "the detail carries the ages and the threshold derivation: {detail}"
        );

        // A committed round closes the episode LOUDLY (the resumed half of the pair)…
        svc.handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-stall".into(),
            phase: "running".into(),
            epoch: 0,
            round: 0,
            generation: 4,
        })
        .expect("phase attributes the run");
        let outs = svc
            .handle_worker_event(&protocol::Event::RoundOutcome {
                round: 3,
                committed: 2,
                ingested: 2,
                stalled: false,
                digest: [0u8; 16],
                generation: 4,
            })
            .expect("outcome handled");
        assert!(
            outs.iter().any(
                |e| matches!(e, VhcEvent::Warning { class, .. } if class == "run_progress_resumed")
            ),
            "committed progress closes the episode"
        );

        // …and a NEW stall episode re-arms. The threshold is now adaptive: the aged gap above
        // stretched max_commit_gap_ms, so the re-aging must clear 2× that observed gap.
        {
            let mut progress = svc.progress.lock().unwrap();
            let t = progress.get_mut("run-stall").expect("track survives");
            assert!(!t.stalled, "the resumed transition cleared the episode");
            t.committed_at_ms -= t.stall_threshold_ms() + 60_000;
        }
        svc.reconcile_tick().await.expect("tick");
        assert_eq!(stall_count(&svc), 2, "a new episode is voiced anew");

        // A run that leaves joined+running drops its track (no warning for a parked row).
        svc.store
            .begin_release("run-stall", RunState::Left, None)
            .expect("release");
        svc.store.commit_release("run-stall").expect("commit");
        svc.reconcile_tick().await.expect("tick");
        assert!(
            !svc.progress.lock().unwrap().contains_key("run-stall"),
            "the track is dropped with the row's liveness"
        );
    }

    /// The lag alarm is trainer-scoped: a coordinator generation never warns (it recovers
    /// through archive reconstruction, never a live pointer).
    #[tokio::test]
    async fn checkpoint_lag_never_warns_for_a_coordinator_generation() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-coord", 3);
        svc.store
            .set_execution_identity("run-coord", 0, "coordinator", 3)
            .expect("the row carries the coordinator role");
        svc.handle_worker_event(&protocol::Event::RunPhase {
            run_id: "run-coord".into(),
            phase: "running".into(),
            epoch: 0,
            round: 0,
            generation: 3,
        })
        .expect("phase attributes the run");

        let outs = svc
            .handle_worker_event(&protocol::Event::RoundOutcome {
                round: 4 * CHECKPOINT_LAG_WARN_ROUNDS,
                committed: 2,
                ingested: 2,
                stalled: false,
                digest: [0u8; 16],
                generation: 3,
            })
            .expect("outcome handled");
        assert!(
            !outs
                .iter()
                .any(|e| matches!(e, VhcEvent::Warning { class, .. } if class == "checkpoint_lag")),
            "a coordinator generation never trips the trainer fence alarm"
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

    /// A worker whose join refuses on the admission funnel's free-disk/ram lane floor — the
    /// REL-8(b) reconverge shape (the re-assess probing a disk filled by dead incarnations).
    struct FloorRefusingWorker;

    #[async_trait]
    impl WorkerControl for FloorRefusingWorker {
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
            Err(VhcError::Internal(
                "assess refused: below lane floor: ram/disk".into(),
            ))
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

    /// REL-8(b): a reconverge refusal on the ram/disk lane floor parks the run STORAGE-GATED
    /// (reclaim → re-check → resume) without burning retry budget — the C2 mechanism where a
    /// disk full of dead incarnations converted a recoverable environment condition into
    /// terminal escalation. Non-floor refusals keep the budgeted lane (pinned elsewhere).
    #[tokio::test]
    async fn a_lane_floor_reconverge_refusal_parks_storage_gated_without_burning_budget() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        svc.worker_factory = Some(Arc::new(|| {
            Arc::new(FloorRefusingWorker) as Arc<dyn WorkerControl>
        }));
        // A claim-bearing intent (derive_charge needs a memory figure) with no live instance:
        // the run awaits retry (the terminal already released it).
        let mut headroom = BTreeMap::new();
        headroom.insert("claim_device_bytes".to_string(), 1_000_000i64);
        headroom.insert("admitted_host_bytes".to_string(), 1_000_000i64);
        svc.store
            .put_join_intent(
                "run-floor",
                "https://coord.local/vhc",
                &VhcPolicy {
                    mode: VhcPolicyMode::Idle,
                    vram_cap_mb: 8_000,
                    duty_cycle_pct: 90,
                    schedule: None,
                },
                None,
                &VhcEligibility {
                    eligible: true,
                    reasons: Vec::new(),
                    headroom,
                },
            )
            .expect("seed the joined intent");
        svc.store
            .begin_release("run-floor", RunState::FailedRetryable, Some("worker died"))
            .expect("release");
        svc.store.commit_release("run-floor").expect("commit");
        svc.store
            .defer_retry("run-floor", now_ms() - 1_000)
            .expect("the retry is due");

        svc.reconcile_tick().await.expect("tick");

        let row = svc.store.get_run("run-floor").expect("read").expect("row");
        assert!(row.storage_gated, "the floor refusal joins the gate lane");
        assert_eq!(row.retry_count, 0, "the retry budget is untouched");
        let warned = svc
            .store
            .recent_events("run-floor", 32)
            .expect("events")
            .into_iter()
            .any(|e| {
                matches!(
                    e,
                    VhcEvent::Warning { class, detail, .. }
                        if class == "storage_gate" && detail.contains("lane floor")
                )
            });
        assert!(warned, "the parking is voiced with the floor reason");
    }

    /// REL-8(d): a run scope crossing 80% of its per-run quota voices ONE `storage_pressure`
    /// warning (announced before the quota kills), and an announced episode never re-voices.
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test fixture seeds raw bytes under a temp scope
    async fn storage_pressure_is_announced_once_when_a_run_scope_nears_its_quota() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        let root = tempfile::tempdir().expect("tempdir");
        // The run scope directory (keyed by the run id's blake3 hex) sits at 90% of a 1 MiB
        // per-run quota before the custodian's seeding walk.
        let scope = blake3::hash("run-press".as_bytes()).to_hex().to_string();
        std::fs::create_dir_all(root.path().join(&scope)).expect("scope dir");
        std::fs::write(
            root.path().join(&scope).join("state.bin"),
            vec![0u8; 943_718],
        )
        .expect("scope bytes");
        svc.run_dir = Some(root.path().to_path_buf());
        svc.config.storage.run_quota_mb = 1;
        seed_joined_run(&svc, "run-press", 2);
        svc.store.mark_running("run-press").expect("running");

        svc.reconcile_tick().await.expect("tick");
        svc.reconcile_tick().await.expect("tick");

        let pressure_events = svc
            .store
            .recent_events("run-press", 32)
            .expect("events")
            .into_iter()
            .filter(|e| matches!(e, VhcEvent::Warning { class, .. } if class == "storage_pressure"))
            .count();
        assert_eq!(pressure_events, 1, "announced once, never re-voiced");
    }

    /// REL-7(d): the co-trainer respawn lane is CYCLE-bounded — a flap-die-respawn loop whose
    /// sibling never survives `min_uptime_ms` parks the lane loudly after `max_retries`
    /// cycles (C2 recorded 461 unbounded `attempt 0` cycles), while a sibling that survived
    /// its uptime threshold resets the count (the primary keeper's discipline).
    #[test]
    fn the_co_trainer_cycle_budget_parks_the_lane_after_sustained_flap() {
        let svc = coordinator_trainer_service(OwnerBudget::unbounded());
        seed_joined_run(&svc, "run-flap", 1);
        svc.store.mark_running("run-flap").expect("running row");
        let sibling_dies = |generation: u64, spawned_ms: i64| -> Vec<VhcEvent> {
            svc.co_trainers.lock().unwrap().insert(
                "run-flap".to_string(),
                InstanceEntry {
                    id: instance_id("trainer", generation),
                    generation,
                    worker: Arc::new(NoopWorker),
                },
            );
            svc.co_cycles
                .lock()
                .unwrap()
                .entry("run-flap".to_string())
                .or_insert((0, 0))
                .1 = spawned_ms;
            svc.handle_run_terminated(
                "run-flap",
                generation,
                &protocol::TerminalOutcome::FailedRetryable {
                    reason: "inbound sequence gap unrecoverable".into(),
                },
            )
            .expect("the sibling terminal is handled")
        };
        let max = svc.config.retry.max_retries;

        // Short-lived cycles (spawned "just now") re-arm the paced lane up to the budget…
        for cycle in 1..=max {
            let outs = sibling_dies(u64::from(cycle), now_ms());
            assert!(
                outs.iter().any(|e| matches!(
                    e,
                    VhcEvent::Warning { class, detail, .. }
                        if class == "co_trainer" && detail.contains("paced respawn")
                )),
                "cycle {cycle} re-arms the paced lane"
            );
            assert!(
                svc.co_retry.lock().unwrap().contains_key("run-flap"),
                "cycle {cycle} leaves the lane armed"
            );
        }

        // …and the cycle past the budget parks the lane LOUDLY instead of cycling forever.
        let outs = sibling_dies(u64::from(max) + 1, now_ms());
        assert!(
            outs.iter().any(|e| matches!(
                e,
                VhcEvent::Warning { class, detail, .. }
                    if class == "co_trainer" && detail.contains("cycle budget exhausted")
            )),
            "exhaustion is voiced, never silent"
        );
        assert!(
            !svc.co_retry.lock().unwrap().contains_key("run-flap"),
            "the parked lane never re-arms itself"
        );

        // A sibling that SURVIVED its uptime threshold resets the count: the next terminal is
        // cycle 1 again, not a continuation of the exhausted ledger.
        let long_ago = now_ms() - i64::try_from(svc.config.retry.min_uptime_ms).unwrap() - 1_000;
        let outs = sibling_dies(u64::from(max) + 2, long_ago);
        assert!(
            outs.iter().any(|e| matches!(
                e,
                VhcEvent::Warning { class, detail, .. }
                    if class == "co_trainer" && detail.contains("cycle 1/")
            )),
            "a healthy uptime resets the cycle budget: {outs:?}"
        );
        assert!(
            svc.co_retry.lock().unwrap().contains_key("run-flap"),
            "the reset lane re-arms"
        );
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

    /// Phase 6: with a wired run-state root the gate asks the CUSTODIAN, not the bare probe —
    /// a global quota already consumed by on-disk run state reads `RefuseNew` and holds the
    /// gate (the pre-custodian floor check would have opened it: the disk has plenty of free
    /// space); raising the quota opens it. This is the resume-authorization contract: a
    /// storage-gated run redispatches only when the custodian confirms capacity.
    #[test]
    // Test-only fixture writes inside the test's own temp root — not a production fs path.
    #[allow(clippy::disallowed_methods)]
    fn the_storage_gate_asks_the_custodian_when_a_runs_root_is_wired() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        let root = tempfile::tempdir().expect("tempdir");
        // Pre-existing run state the open-time walk seeds the ledger with.
        std::fs::create_dir_all(root.path().join("scope-a/trainer-1/journal")).unwrap();
        std::fs::write(
            root.path().join("scope-a/trainer-1/journal/segment"),
            vec![0u8; 3 * 1024 * 1024],
        )
        .unwrap();
        svc.run_dir = Some(root.path().to_path_buf());
        svc.config.storage.reserve_mb = 0;
        svc.config.storage.emergency_mb = 0;
        svc.config.storage.quota_mb = 2; // 2 MiB quota, 3 MiB already used → RefuseNew
        assert!(
            !svc.storage_gate_open(),
            "an exhausted quota holds the gate even with free disk space"
        );

        // NOTE: the per-root custodian is a process singleton whose FIRST opener's config
        // wins, so the relaxed policy needs a fresh root (exactly what a reconfigured node
        // restart does).
        let roomy = tempfile::tempdir().expect("tempdir");
        svc.run_dir = Some(roomy.path().to_path_buf());
        svc.config.storage.quota_mb = 64;
        assert!(
            svc.storage_gate_open(),
            "a clear quota + floor opens the gate"
        );
    }

    /// Reclaim-before-refuse (the c15c M4 regression): a storage gate held by a quota that is
    /// consumed by the run's OWN superseded incarnations must run the manifest-driven orphan
    /// reconciliation and re-check — never park the run behind an operator when the manifests
    /// prove the space reclaimable. The newest incarnation (the reconstruction input) survives.
    #[test]
    // Test-only fixture writes inside the test's own temp root — not a production fs path.
    #[allow(clippy::disallowed_methods)]
    fn a_held_gate_reclaims_proven_orphans_before_refusing() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        let root = tempfile::tempdir().expect("tempdir");
        svc.run_dir = Some(root.path().to_path_buf());
        svc.config.storage.reserve_mb = 0;
        svc.config.storage.emergency_mb = 0;
        svc.config.storage.quota_mb = 2; // 2 MiB quota

        // A persisted run row makes the scope KNOWN (reconciliation never touches unknowns).
        seed_joined_run(&svc, "run-g", 2);
        let scope = blake3::hash(b"run-g").to_hex().to_string();
        let scope_path = root.path().join(&scope);
        // trainer-1: superseded, journal absent (vacuously archived) — a 3 MiB dead spill.
        std::fs::create_dir_all(scope_path.join("trainer-1/state")).unwrap();
        std::fs::write(
            scope_path.join("trainer-1/state/chunk"),
            vec![0u8; 3 * 1024 * 1024],
        )
        .unwrap();
        // trainer-2: the newest incarnation — the reconstruction input, never reclaimed.
        std::fs::create_dir_all(scope_path.join("trainer-2/journal")).unwrap();
        std::fs::write(scope_path.join("trainer-2/journal/keep"), vec![1u8; 64]).unwrap();

        assert!(
            !svc.storage_gate_open(),
            "the superseded spill holds the quota-exhausted gate"
        );
        assert!(
            svc.storage_gate_open_reclaiming(),
            "reconciliation reclaims the proven orphan and the gate opens"
        );
        assert!(
            !scope_path.join("trainer-1").exists(),
            "the superseded incarnation was reclaimed"
        );
        assert!(
            scope_path.join("trainer-2/journal/keep").exists(),
            "the newest incarnation survives"
        );
    }

    /// The wire-v45 disk surface: the usage report maps scopes back to run labels and splits
    /// bytes by reclaim class; the safe wipe refuses a live run, refuses a standing joined
    /// intent, then reclaims recoverable state while the archive planes and everything outside
    /// the runs root (the identity keystore) survive — evidence goes only on explicit request.
    #[tokio::test]
    // Test-only fixture writes inside the test's own temp root — not a production fs path.
    #[allow(clippy::disallowed_methods)]
    async fn the_disk_surface_reports_by_reclaim_class_and_wipes_identity_preservingly() {
        let mut svc = coordinator_trainer_service(OwnerBudget::unbounded());
        let root = tempfile::tempdir().expect("tempdir");
        svc.run_dir = Some(root.path().to_path_buf());
        svc.config.storage.reserve_mb = 0;
        svc.config.storage.emergency_mb = 0;

        seed_joined_run(&svc, "run-w", 1);
        let scope = blake3::hash(b"run-w").to_hex().to_string();
        let scope_path = root.path().join(&scope);
        std::fs::create_dir_all(scope_path.join("trainer-1/journal")).unwrap();
        std::fs::write(
            scope_path.join("trainer-1/journal/segment"),
            vec![1u8; 2 * 1024 * 1024],
        )
        .unwrap();
        std::fs::create_dir_all(scope_path.join("payload")).unwrap();
        std::fs::write(scope_path.join("payload/blob"), vec![2u8; 1024 * 1024]).unwrap();

        let usage = svc.vhc_disk_usage().await.expect("usage report");
        let row = usage
            .scopes
            .iter()
            .find(|s| s.scope == scope)
            .expect("the run's scope row");
        assert_eq!(row.run_id.as_deref(), Some("run-w"), "scope maps to label");
        assert!(row.active, "the live instance marks the row active");
        assert_eq!(row.recoverable_mb, 2);
        assert_eq!(row.evidence_mb, 1);

        // Live ⇒ refused; intent standing ⇒ refused.
        let live = svc.vhc_disk_wipe("run-w".into(), false).await;
        assert!(matches!(live, Err(ApiError::Conflict(_))), "{live:?}");
        svc.instances.lock().unwrap().remove("run-w");
        let intent = svc.vhc_disk_wipe("run-w".into(), false).await;
        assert!(matches!(intent, Err(ApiError::Conflict(_))), "{intent:?}");

        // Left ⇒ the default wipe takes recoverable state, spares the evidence planes.
        svc.store
            .set_desired_state("run-w", DesiredState::Left)
            .unwrap();
        let outcome = svc.vhc_disk_wipe("run-w".into(), false).await.unwrap();
        assert_eq!(outcome.reclaimed_mb, 2);
        assert!(!outcome.wiped_evidence);
        assert!(!scope_path.join("trainer-1").exists());
        assert!(scope_path.join("payload/blob").exists(), "evidence spared");
        assert!(
            outcome.preserved.iter().any(|p| p.contains("base.key")),
            "the identity guarantee is named: {:?}",
            outcome.preserved
        );

        // The explicit evidence wipe finishes the job; idempotent thereafter.
        let evidence = svc.vhc_disk_wipe("run-w".into(), true).await.unwrap();
        assert_eq!(evidence.reclaimed_mb, 1);
        assert!(evidence.wiped_evidence);
        assert!(!scope_path.exists());
        let rerun = svc.vhc_disk_wipe("run-w".into(), true).await.unwrap();
        assert_eq!(rerun.reclaimed_mb, 0);
    }
}
