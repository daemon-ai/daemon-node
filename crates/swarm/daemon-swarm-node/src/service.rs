// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`SwarmService`] — the resident node-side swarm-training service (spec §10.3/§10.4).
//!
//! It owns a worker-control seam ([`WorkerControl`], implemented for `daemon-train-client`'s
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
use daemon_swarm_run::config::SwarmConfig;
use daemon_swarm_run::protocol::{self, Eligibility, Hardware, JoinPolicy, LeaveMode, PolicyMode};
use daemon_train_client::TrainSupervisor;
use futures::StreamExt;
use std::collections::BTreeMap;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::discovery::RunDiscovery;
use crate::store::{DesiredState, PersistedRun, StoreError, SwarmStore, EVENT_WINDOW};

/// A node-feed sink: the node passes a closure over `NodeEventFeed::emit` so live swarm updates ride
/// the existing `events_subscribe` channel as `SwarmChanged` pointers (no new transport).
pub type NodeFeed = Arc<dyn Fn(NodeEvent) + Send + Sync>;

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
/// `daemon-train-client`'s `TrainSupervisor` (real worker); a fake impl in tests exercises the
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
    /// Leave a run.
    async fn leave(&self, run_id: String, mode: LeaveMode) -> Result<(), SwarmError>;
    /// Push a GPU-governor throttle lever (§10.5).
    async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), SwarmError>;
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
}

/// Construction parts for a [`SwarmService`].
pub struct SwarmServiceParts {
    /// The `[swarm]` config (spec §10.6); `enabled` gates all worker activity.
    pub config: SwarmConfig,
    /// The durable `swarm.db` store.
    pub store: SwarmStore,
    /// The worker-control seam (a real `TrainSupervisor` in production).
    pub worker: Arc<dyn WorkerControl>,
    /// The node-feed sink for `SwarmChanged` pointers (`None` on a headless / test build).
    pub feed: Option<NodeFeed>,
    /// The run-discovery seam (A1). When present, `swarm_join` discovers the run + fetches the frozen
    /// envelope + runs the worker's real §6.5 `AssessRun` before `JoinRun`. `None` keeps the W1
    /// probe-based eligibility path (no coordinator configured), so the service stays usable offline.
    pub discovery: Option<Arc<dyn RunDiscovery>>,
}

/// The node-side swarm-training service.
pub struct SwarmService {
    config: SwarmConfig,
    store: SwarmStore,
    worker: Arc<dyn WorkerControl>,
    discovery: Option<Arc<dyn RunDiscovery>>,
    events_tx: broadcast::Sender<SwarmEvent>,
    feed: Option<NodeFeed>,
    /// The run the worker is currently on (from the last `RunPhase`), used to attribute events that
    /// don't carry a run id (`RoundProgress`/`RoundOutcome`/…).
    current_run: Mutex<Option<String>>,
    /// The coalescing swarm-feed revision stamped on each `SwarmChanged` pointer.
    rev: AtomicU64,
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
            events_tx,
            feed: parts.feed,
            current_run: Mutex::new(None),
            rev: AtomicU64::new(0),
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
    pub async fn start(&self) -> Result<usize, SwarmError> {
        if !self.config.enabled {
            return Ok(0);
        }
        let intents = self.store.active_intents()?;
        let mut rejoined = 0;
        for run in &intents {
            self.worker
                .join(
                    run.run_id.clone(),
                    run.coordinator.clone(),
                    Vec::new(),
                    to_join_policy(&run.policy),
                )
                .await?;
            rejoined += 1;
        }
        if rejoined > 0 {
            self.emit_changed(None);
        }
        Ok(rejoined)
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

    /// Resolve the `(coordinator, eligibility)` for a join (A1).
    ///
    /// With a discovery seam: `GET /runs/:id` → fetch + blake3-verify the frozen envelope →
    /// `worker.assess(envelope)` (real §6.5), taking the coordinator from the registry. Without one:
    /// the W1 probe against the allowlisted coordinator. Eligibility is always node-computed.
    async fn resolve_join(&self, run_id: &str) -> Result<(String, SwarmEligibility), SwarmError> {
        if let Some(discovery) = &self.discovery {
            let run = discovery
                .get_run(run_id)
                .await?
                .ok_or_else(|| SwarmError::Discovery(format!("run {run_id} not found")))?;
            let envelope = discovery.fetch_envelope(run_id).await?;
            let verdict = self.worker.assess(envelope).await?;
            Ok((run.coordinator, eligibility_from_assess(&verdict)))
        } else {
            let coordinator = self.coordinator();
            let eligibility = match self.worker.probe().await {
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
        // Node-computed eligibility (ADR-003). A1: when a discovery seam is configured, resolve the
        // run + fetch the frozen envelope + run the worker's real §6.5 `AssessRun` before `JoinRun`,
        // and take the coordinator endpoint from discovery. With no discovery configured, fall back
        // to the W1 probe-based eligibility against the allowlisted coordinator (offline / no-registry
        // path). Either way the persisted eligibility is node-computed — the app never re-derives it.
        let (coordinator, eligibility) =
            self.resolve_join(&run_id).await.map_err(|e| e.to_api())?;
        self.store
            .put_join_intent(&run_id, &coordinator, &policy, None, &eligibility)
            .map_err(|e| SwarmError::from(e).to_api())?;
        self.worker
            .join(
                run_id.clone(),
                coordinator,
                Vec::new(),
                to_join_policy(&policy),
            )
            .await
            .map_err(|e| e.to_api())?;
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
        self.worker
            .leave(run_id.clone(), to_leave_mode(mode))
            .await
            .map_err(|e| e.to_api())?;
        self.emit_changed(Some(run_id));
        Ok(())
    }

    async fn swarm_set_policy(&self, policy: SwarmPolicy) -> Result<(), ApiError> {
        self.require_enabled().map_err(|e| e.to_api())?;
        // W1: push the governor levers to the worker (§10.5). The persisted default-policy slot for
        // future joins is the config `[swarm].default_policy`; a durable override lands with the
        // policy store in a later wave.
        self.worker
            .throttle(
                Some(policy.vram_cap_mb),
                Some(policy.duty_cycle_pct.min(100) as u8),
                false,
            )
            .await
            .map_err(|e| e.to_api())?;
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
    SwarmRunSummary {
        run_id: run.run_id,
        phase: run.last_phase,
        joined,
        eligibility: run.eligibility,
        policy: if joined { Some(run.policy) } else { None },
        last_round: run.last_round,
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
/// configured): eligible if the worker reports a usable GPU or backend lane, with VRAM/RAM
/// headroom. The app renders this; it never re-derives eligibility (ADR-003).
fn eligibility_from_hardware(hw: &Hardware) -> SwarmEligibility {
    let eligible = hw.gpus > 0 || !hw.backend_lanes.is_empty();
    let mut reasons = Vec::new();
    if !eligible {
        reasons.push("no usable GPU or backend lane".to_string());
    }
    let mut headroom = BTreeMap::new();
    headroom.insert("vram_mb".to_string(), hw.vram_mb as i64);
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
