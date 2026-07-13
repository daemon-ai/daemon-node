// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The swarm-training sub-surface ([`SwarmApi`]) + its wire DTOs (swarm-training-spec.md §10.4).
//!
//! The node is the single authority for swarm participation state; the app is a thin mirror
//! (ADR-003): every run row carries the **node-computed** [`SwarmEligibility`] ("joinable or why
//! not"), which the app renders and never re-derives (§6.5). The DTOs keep experiment-opaque fields
//! opaque (the seam rule): they carry participation state — phase, policy, eligibility, contribution
//! counters — and never any experiment config or module bytes.
//!
//! Like [`ModelApi`](crate::ModelApi), every method defaults to [`ApiError::Unsupported`] / empty so
//! a transport that hosts no swarm service (the session-only FFI, test stubs) inherits the surface;
//! the node's [`NodeApi`](crate::NodeApi) binds the real implementation (backed by the node
//! `SwarmService` over a `daemon-train` worker).

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::ApiError;

/// A live, push-based stream of [`SwarmEvent`]s — the delivery shape [`SwarmApi::swarm_subscribe`]
/// returns for the in-process transport and the node `SwarmService`'s own broadcast. Over the socket
/// mux, live swarm updates ride the **existing** node-event feed as payload-free
/// [`NodeEvent::SwarmChanged`](crate::NodeEvent::SwarmChanged) pointers (the client refetches
/// [`SwarmRunDetail`], whose `recent_events` carries the windowed events, §10.3) — no new transport.
pub type SwarmEventStream = BoxStream<'static, SwarmEvent>;

/// The peer's availability posture for a run (spec §10.5). Wire mirror of the worker's
/// `PolicyMode`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmPolicyMode {
    /// Participate whenever admitted.
    Always,
    /// Participate only when no inference activity + user-idle heuristics hold.
    #[default]
    Idle,
    /// Participate on a cron schedule (`schedule`).
    Scheduled,
    /// Participate only on explicit manual start.
    Manual,
}

/// A participation policy (spec §10.4/§10.5): the GPU-governor caps + availability mode a peer joins
/// a run under. Caps also define the peer's *effective* resources for eligibility (§6.5).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmPolicy {
    /// The availability mode.
    pub mode: SwarmPolicyMode,
    /// A VRAM cap in MiB (`0` = uncapped).
    pub vram_cap_mb: u32,
    /// A duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u32,
    /// An optional cron schedule (for [`SwarmPolicyMode::Scheduled`]); absent on the wire when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
}

impl Default for SwarmPolicy {
    fn default() -> Self {
        // Spec §10.6 default_policy: `{ mode = "idle", vram_cap_mb = 0, duty_cycle_pct = 100 }`.
        Self {
            mode: SwarmPolicyMode::Idle,
            vram_cap_mb: 0,
            duty_cycle_pct: 100,
            schedule: None,
        }
    }
}

/// The node-computed self-assessment for a run (§6.5): the app renders "joinable, or why not" from
/// this and NEVER re-derives it (ADR-003 mirror). `headroom` is per-dimension slack (e.g.
/// `"vram_mb" => 4096`); a negative value is a deficit.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmEligibility {
    /// Whether this peer can join.
    pub eligible: bool,
    /// Human-readable reasons (why-not / caveats).
    pub reasons: Vec<String>,
    /// Per-dimension headroom (positive = slack, negative = deficit).
    pub headroom: BTreeMap<String, i64>,
}

/// The worker's capability vocabulary as mirrored to the app (wire mirror of `WorkerCapabilities`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmCapabilities {
    /// The tensor-ABI major version the worker implements.
    pub abi_version: u32,
    /// The host-vocabulary ops the worker implements (`name@version`).
    pub ops: Vec<String>,
    /// The payload stores the worker can speak (`r2`, `iroh-blobs`, …).
    pub payload_stores: Vec<String>,
}

/// This node's training capability (spec §10.4 `SwarmHardwareReport`): the probe results + active
/// lanes the GUI's "what can my GPU do" panel renders. Wire mirror of the worker's `Hardware`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmHardwareReport {
    /// The number of usable GPUs.
    pub gpus: u32,
    /// Total VRAM in MiB (across GPUs).
    pub vram_mb: u64,
    /// Installed host RAM in MiB.
    pub ram_mb: u64,
    /// The backend lanes the worker was built with (`cpu`, `cuda`, `rocm`, `vulkan`).
    pub backend_lanes: Vec<String>,
    /// The capability vocabulary.
    pub capabilities: SwarmCapabilities,
    /// Measured uplink in kbit/s.
    pub up_kbps: u64,
    /// Measured downlink in kbit/s.
    pub down_kbps: u64,
    /// Free disk for the data/checkpoint cache in MiB.
    pub disk_free_mb: u64,
    /// The measured throughput class (`c1`..`c4`).
    pub throughput_class: String,
}

/// The per-run contribution ledger (spec §10.3 `swarm_contrib`): what this node's GPU did for a run.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmContribution {
    /// Rounds participated in.
    pub rounds: u64,
    /// Tokens processed.
    pub tokens: u64,
    /// Bytes uploaded (update objects + checkpoints).
    pub bytes_up: u64,
    /// Bytes downloaded (peer updates + artifacts).
    pub bytes_down: u64,
    /// Times this node acted as a witness.
    pub witness_count: u64,
    /// Checkpoints this node published (checkpointer credits).
    pub checkpoint_credits: u64,
}

/// One row of the run list (spec §10.4): a discovered/joined run annotated with node-computed
/// eligibility. Experiment-opaque (the seam rule): no experiment config or module bytes.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmRunSummary {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// The node's last-known phase string for the run (display-only; opaque).
    pub phase: String,
    /// Whether this node holds a durable join-intent for the run.
    pub joined: bool,
    /// The node-computed eligibility (§6.5); the app renders it, never re-derives it.
    pub eligibility: SwarmEligibility,
    /// The policy this node joined the run under (present only when `joined`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<SwarmPolicy>,
    /// The last-known round the node observed for the run.
    pub last_round: u64,
}

/// The full detail view for one run (spec §10.4): the summary + coordinator endpoint + contribution
/// ledger + the windowed recent events (§10.3 `swarm_events`, ADR-007).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmRunDetail {
    /// The list-row summary (carries eligibility).
    pub summary: SwarmRunSummary,
    /// The coordinator endpoint this run is served from.
    pub coordinator: String,
    /// The per-run contribution ledger.
    pub contribution: SwarmContribution,
    /// The windowed recent events for the run (newest last).
    pub recent_events: Vec<SwarmEvent>,
}

/// How a peer leaves a run (spec §10.2/§10.4). Wire mirror of the worker's `LeaveMode`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmLeaveMode {
    /// Finish the current round, then leave.
    #[default]
    Graceful,
    /// Leave immediately (abort any in-flight work).
    Immediate,
}

/// A swarm run event (spec §10.4): phase transitions, per-round progress, outcomes, contribution
/// deltas, and warnings/errors. Numeric telemetry is fixed-point integer (no floats on the wire —
/// keeps the vendored C codec + the `arbitrary` conformance proptest simple): `loss_micros` is the
/// loss × 1e6, `tokens_per_s_milli` is tokens/s × 1e3.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmEvent {
    /// A run/phase transition.
    Phase {
        /// The run id.
        run_id: String,
        /// The phase string.
        phase: String,
        /// The training epoch.
        epoch: u64,
        /// The round at the transition.
        round: u64,
    },
    /// Per-round training progress (loss + throughput sparkline inputs).
    Progress {
        /// The run id.
        run_id: String,
        /// The inner optimizer step within the round.
        inner_step: u32,
        /// Loss × 1e6 (fixed-point).
        loss_micros: u64,
        /// Tokens/s × 1e3 (fixed-point).
        tokens_per_s_milli: u64,
        /// Peers observed this round.
        peers: u32,
    },
    /// A round's finalization outcome.
    RoundOutcome {
        /// The run id.
        run_id: String,
        /// The round.
        round: u64,
        /// Committed peers.
        committed: u32,
        /// Ingested payloads.
        ingested: u32,
        /// Whether this node stalled the round.
        stalled: bool,
    },
    /// A contribution-ledger delta (the running totals after the update).
    Contribution {
        /// The run id.
        run_id: String,
        /// The updated running totals.
        contribution: SwarmContribution,
    },
    /// A non-fatal warning (typed class + detail).
    Warning {
        /// The run id.
        run_id: String,
        /// The warning class.
        class: String,
        /// Human-readable detail.
        detail: String,
    },
    /// A classified error (the run may drop this peer per §13).
    Error {
        /// The run id.
        run_id: String,
        /// The error class.
        class: String,
        /// Human-readable detail.
        detail: String,
    },
}

impl SwarmEvent {
    /// The run id this event pertains to (every variant carries one).
    pub fn run_id(&self) -> &str {
        match self {
            SwarmEvent::Phase { run_id, .. }
            | SwarmEvent::Progress { run_id, .. }
            | SwarmEvent::RoundOutcome { run_id, .. }
            | SwarmEvent::Contribution { run_id, .. }
            | SwarmEvent::Warning { run_id, .. }
            | SwarmEvent::Error { run_id, .. } => run_id,
        }
    }

    /// The stable wire tag for this event (the `swarm_events.kind` column + display discriminator).
    pub fn kind(&self) -> &'static str {
        match self {
            SwarmEvent::Phase { .. } => "phase",
            SwarmEvent::Progress { .. } => "progress",
            SwarmEvent::RoundOutcome { .. } => "round_outcome",
            SwarmEvent::Contribution { .. } => "contribution",
            SwarmEvent::Warning { .. } => "warning",
            SwarmEvent::Error { .. } => "error",
        }
    }
}

/// The swarm-training sub-surface (spec §10.4): discover/join/leave runs, set the participation
/// policy, report training hardware, and subscribe to run events. Every method defaults to
/// [`ApiError::Unsupported`] / empty so a transport with no swarm service inherits the surface; the
/// node's [`NodeApi`](crate::NodeApi) binds the real implementation.
#[async_trait]
pub trait SwarmApi: Send + Sync {
    /// Discovered + joined runs, each annotated with the node-computed [`SwarmEligibility`] (§6.5).
    async fn swarm_run_list(&self) -> Result<Vec<SwarmRunSummary>, ApiError> {
        Err(ApiError::Unsupported("swarm_run_list".into()))
    }

    /// One run's full detail (`None` if unknown to this node).
    async fn swarm_run_detail(&self, _run_id: String) -> Result<Option<SwarmRunDetail>, ApiError> {
        Err(ApiError::Unsupported("swarm_run_detail".into()))
    }

    /// Join a run under `policy` (durable intent; idempotent via `op_id`, ADR-006). The node persists
    /// the desired-state flag so a restart re-converges (rejoins) without app involvement (§10.3).
    async fn swarm_join(
        &self,
        _run_id: String,
        _policy: SwarmPolicy,
        _op_id: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("swarm_join".into()))
    }

    /// Leave a run (durable intent; idempotent via `op_id`).
    async fn swarm_leave(
        &self,
        _run_id: String,
        _mode: SwarmLeaveMode,
        _op_id: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("swarm_leave".into()))
    }

    /// Set the default participation policy for newly-joined runs (§10.5).
    async fn swarm_set_policy(&self, _policy: SwarmPolicy) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("swarm_set_policy".into()))
    }

    /// This node's training-capability report (probe results + active lanes).
    async fn swarm_hardware_report(&self) -> Result<SwarmHardwareReport, ApiError> {
        Err(ApiError::Unsupported("swarm_hardware_report".into()))
    }

    /// Subscribe to run events (all runs when `run_id` is `None`, else one run). Rides the existing
    /// feed machinery: the default is an empty stream; the node returns a live [`SwarmEventStream`].
    async fn swarm_subscribe(&self, _run_id: Option<String>) -> Result<SwarmEventStream, ApiError> {
        Ok(stream::empty().boxed())
    }
}
