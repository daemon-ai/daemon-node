// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The vhc-training sub-surface ([`VhcApi`]) + its wire DTOs (swarm-training-spec.md §10.4).
//!
//! The node is the single authority for vhc participation state; the app is a thin mirror
//! (ADR-003): every run row carries the **node-computed** [`VhcEligibility`] ("joinable or why
//! not"), which the app renders and never re-derives (§6.5). The DTOs keep experiment-opaque fields
//! opaque (the seam rule): they carry participation state — phase, policy, eligibility, contribution
//! counters — and never any experiment config or module bytes.
//!
//! Like [`ModelApi`](crate::ModelApi), every method defaults to [`ApiError::Unsupported`] / empty so
//! a transport that hosts no vhc service (the session-only FFI, test stubs) inherits the surface;
//! the node's [`NodeApi`](crate::NodeApi) binds the real implementation (backed by the node
//! `VhcService` over a `daemon-vhc-host` worker).

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::ApiError;

/// A live, push-based stream of [`VhcEvent`]s — the delivery shape [`VhcApi::vhc_subscribe`]
/// returns for the in-process transport and the node `VhcService`'s own broadcast. Over the socket
/// mux, live vhc updates ride the **existing** node-event feed as payload-free Vhc
/// [`NodeEvent::ProjectionChanged`](crate::NodeEvent::ProjectionChanged) pointers (the client refetches
/// [`VhcRunDetail`], whose `recent_events` carries the windowed events, §10.3) — no new transport.
pub type VhcEventStream = BoxStream<'static, VhcEvent>;

/// The peer's availability posture for a run (spec §10.5). Wire mirror of the worker's
/// `PolicyMode`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VhcPolicyMode {
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
pub struct VhcPolicy {
    /// The availability mode.
    pub mode: VhcPolicyMode,
    /// A VRAM cap in MiB (`0` = uncapped).
    pub vram_cap_mb: u32,
    /// A duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u32,
    /// An optional cron schedule (for [`VhcPolicyMode::Scheduled`]); absent on the wire when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
}

impl Default for VhcPolicy {
    fn default() -> Self {
        // Spec §10.6 default_policy: `{ mode = "idle", vram_cap_mb = 0, duty_cycle_pct = 100 }`.
        Self {
            mode: VhcPolicyMode::Idle,
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
pub struct VhcEligibility {
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
pub struct VhcCapabilities {
    /// The tensor-ABI major version the worker implements.
    pub abi_version: u32,
    /// The host-vocabulary ops the worker implements (`name@version`).
    pub ops: Vec<String>,
    /// The payload stores the worker can speak (`r2`, `iroh-blobs`, …).
    pub payload_stores: Vec<String>,
}

/// This node's training capability (spec §10.4 `VhcHardwareReport`): the probe results + active
/// lanes the GUI's "what can my GPU do" panel renders. Wire mirror of the worker's `Hardware`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcHardwareReport {
    /// The number of usable GPUs.
    pub gpus: u32,
    /// Total VRAM in MiB (across GPUs).
    pub vram_mb: u64,
    /// Shared / unified spillover memory in MiB (GTT on an integrated/UMA GPU): the host DRAM the
    /// GPU can page tensors into beyond [`Self::vram_mb`]; `0` on a classic discrete GPU. The
    /// effective device budget is `vram_mb + 90%·shared_mb` (§10.5). **Additive (wire v42):**
    /// `#[serde(default)]` keeps a pre-v42 report decodable (fills `0`) and mirrors the worker
    /// `Hardware.shared_mb` the node already probes (P1 Merge-2 follow-on).
    #[serde(default)]
    pub shared_mb: u64,
    /// Installed host RAM in MiB.
    pub ram_mb: u64,
    /// The backend lanes the worker was built with (`cpu`, `cuda`, `rocm`, `vulkan`).
    pub backend_lanes: Vec<String>,
    /// The capability vocabulary.
    pub capabilities: VhcCapabilities,
    /// Measured uplink in kbit/s.
    pub up_kbps: u64,
    /// Measured downlink in kbit/s.
    pub down_kbps: u64,
    /// Free disk for the data/checkpoint cache in MiB.
    pub disk_free_mb: u64,
    /// The measured throughput class (`c1`..`c4`).
    pub throughput_class: String,
}

/// One run scope's disk-custody row (wire v45): the bytes the disk custodian ledgers under the
/// scope's run-state directory, split by reclaim class so an operator sees exactly what a wipe
/// would touch — recoverable state (journal segments + spill, rebuildable from the archive) vs
/// archived evidence (payload + archive planes, the authenticated product).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcDiskScope {
    /// The run label when the node can map the scope back to a known run row; absent for an
    /// orphaned scope (state on disk with no surviving run row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// The on-disk scope directory name (the hashed run-state key).
    pub scope: String,
    /// Recoverable-state bytes in MiB (journal + spill): safe to wipe, rebuildable from archive.
    pub recoverable_mb: u64,
    /// Archived-evidence bytes in MiB (payload + archive planes): wiped only on explicit request.
    pub evidence_mb: u64,
    /// Whether the run is live right now (a wipe refuses while live).
    pub active: bool,
}

/// The node's disk-custody report (wire v45, `vhc_disk_usage`): the custodian's ledger for the
/// VHC runs root — probed free space, committed usage, the configured envelope (quota / OS-floor
/// reserve / emergency sealing margin), the derived pressure state, and the per-run breakdown.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcDiskUsage {
    /// The custodied runs root.
    pub root: String,
    /// Probed free bytes on the root's filesystem, in MiB.
    pub free_mb: u64,
    /// Committed bytes in the custodian's ledger, in MiB.
    pub used_mb: u64,
    /// The configured global quota in MiB (`0` = unbounded).
    pub quota_mb: u64,
    /// The configured OS free-space floor in MiB.
    pub reserve_mb: u64,
    /// The configured emergency sealing margin in MiB.
    pub emergency_mb: u64,
    /// The derived pressure state (`nominal` | `warn` | `refuse_new`).
    pub pressure: String,
    /// Per-run-scope custody rows (largest first).
    pub scopes: Vec<VhcDiskScope>,
}

/// One safe wipe's outcome (wire v45, `vhc_disk_wipe`). The wipe NEVER touches the identity
/// keystore (`base.key` and the per-run signing keys live outside the runs root), and it wipes
/// archived evidence only when explicitly asked.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcDiskWipeOutcome {
    /// The wiped run.
    pub run_id: String,
    /// Bytes reclaimed, in MiB.
    pub reclaimed_mb: u64,
    /// Whether archived evidence (payload + archive planes) was wiped too.
    pub wiped_evidence: bool,
    /// What the wipe deliberately preserved (operator-facing, e.g. `identity/base.key`,
    /// `archive plane`).
    pub preserved: Vec<String>,
}

/// The per-run contribution ledger (spec §10.3 `vhc_contrib`): what this node's GPU did for a run.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcContribution {
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
pub struct VhcRunSummary {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// The node's last-known phase string for the run (display-only; opaque).
    pub phase: String,
    /// Whether this node holds a durable join-intent for the run.
    pub joined: bool,
    /// The node-computed eligibility (§6.5); the app renders it, never re-derives it.
    pub eligibility: VhcEligibility,
    /// The policy this node joined the run under (present only when `joined`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<VhcPolicy>,
    /// The last-known round the node observed for the run.
    pub last_round: u64,
    /// The run instance's effective lifecycle state (`running | completed | paused |
    /// failed_retryable | failed_terminal | left`) — the node's two-axis state machine (owner
    /// intent × observed instance lifecycle) projected for display. Additive: absent from a
    /// pre-lifecycle node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_state: Option<String>,
    /// Reconvergence attempts consumed since the last stable interval (the bounded retry
    /// budget). Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_count: Option<u64>,
    /// The typed reason recorded with a terminal transition (operator-facing detail). Additive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<String>,
    // -- D0 additive run-identity + sunset-observability fields (envelope v2; decisions D1/D5).
    // The node decides, the app renders (never re-derives): these mirror the vhc.db M2
    // columns. All optional — absent on pre-D0 nodes and for fields a v1 run never acquires.
    /// The cryptographic `RunId` — lowercase hex of the 32-byte genesis-envelope hash — once
    /// known (v2 runs; absent for a v1-only run, whose identity is the `run_id` label alone).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id_hash: Option<String>,
    /// The transition-chain epoch of the run's execution identity (present with `run_id_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u64>,
    /// The envelope-level role label this node serves (present with `run_id_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The never-reused u64 role-instance incarnation id (present with `run_id_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<u64>,
    /// Sunset observability (decisions D5): the run's envelope schema major (1 = v1, 2 = v2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope_schema_major: Option<u32>,
    /// Sunset observability: the admitted worker module's `da_abi` major (1 or 2), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_abi_major: Option<u32>,
    /// Sunset observability: the selected driver (`"v1"` / `"v2"`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_driver: Option<String>,
    /// Sunset observability: lowercase hex of the current pinned module blake3, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_hash: Option<String>,
}

/// The full detail view for one run (spec §10.4): the summary + coordinator endpoint + contribution
/// ledger + the windowed recent events (§10.3 `vhc_events`, ADR-007).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VhcRunDetail {
    /// The list-row summary (carries eligibility).
    pub summary: VhcRunSummary,
    /// The coordinator endpoint this run is served from.
    pub coordinator: String,
    /// The per-run contribution ledger.
    pub contribution: VhcContribution,
    /// The windowed recent events for the run (newest last).
    pub recent_events: Vec<VhcEvent>,
    /// The 16-byte post-ingest det-state digest of the newest round this node has observed an
    /// outcome for (the digest carried by the highest-round [`VhcEvent::RoundOutcome`]), or `None`
    /// before any round completes. **Additive (wire v44):** surfaces the per-round digest on the
    /// snapshot so a polling client (`daemon-cli vhc detail --watch`) collects the digest-agreement
    /// transcript without an event subscription. The node decides, the app renders — never
    /// re-derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_round_digest: Option<[u8; 16]>,
}

/// The outcome of a consumed run-level module-upgrade record (wire v43): the node validated the
/// record against the run's transition chain fail-closed and drove the local switch transaction
/// through its worker. The three arms mirror the switch's terminal facts — activation, a
/// pre-fence refusal with the old module untouched, or a post-fence exit that left the run.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VhcSwitchOutcome {
    /// The new module activated locally under the target epoch — no process restart.
    Activated {
        /// The epoch now running locally (the record's committed target epoch).
        epoch: u64,
        /// Lowercase hex of the new module's blake3 now bound to the role-instance.
        module_hash: String,
        /// Rollback-and-retry cycles used before activation (`0` on a clean first migration).
        retries: u32,
    },
    /// The switch was refused before the transaction touched the running instance (record
    /// validation, target resolution, identity provisioning, or pre-fence admission). The old
    /// module keeps running.
    Refused {
        /// Why (operator-facing detail, never branched on).
        reason: String,
    },
    /// The local transaction failed closed / exhausted its retries after the fence and left the
    /// run. The run-level record stays committed; only this node left.
    Left {
        /// Why (operator-facing detail, never branched on).
        reason: String,
    },
}

/// How a peer leaves a run (spec §10.2/§10.4). Wire mirror of the worker's `LeaveMode`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VhcLeaveMode {
    /// Finish the current round, then leave.
    #[default]
    Graceful,
    /// Leave immediately (abort any in-flight work).
    Immediate,
}

/// A vhc run event (spec §10.4): phase transitions, per-round progress, outcomes, contribution
/// deltas, and warnings/errors. Numeric telemetry is fixed-point integer (no floats on the wire —
/// keeps the vendored C codec + the `arbitrary` conformance proptest simple): `loss_micros` is the
/// loss × 1e6, `tokens_per_s_milli` is tokens/s × 1e3.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VhcEvent {
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
        /// The 16-byte post-ingest det-state digest for the round (§5.6): the worker session
        /// reports it and every peer that completes the round must produce a byte-identical value
        /// — the digest-agreement (G-2) evidence a polling client collects. **Additive (wire
        /// v44):** the node previously dropped this at the API boundary; `#[serde(default)]` keeps
        /// a pre-v44 stored event decodable (fills all-zero).
        #[serde(default)]
        digest: [u8; 16],
    },
    /// A contribution-ledger delta (the running totals after the update).
    Contribution {
        /// The run id.
        run_id: String,
        /// The updated running totals.
        contribution: VhcContribution,
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

impl VhcEvent {
    /// The run id this event pertains to (every variant carries one).
    pub fn run_id(&self) -> &str {
        match self {
            VhcEvent::Phase { run_id, .. }
            | VhcEvent::Progress { run_id, .. }
            | VhcEvent::RoundOutcome { run_id, .. }
            | VhcEvent::Contribution { run_id, .. }
            | VhcEvent::Warning { run_id, .. }
            | VhcEvent::Error { run_id, .. } => run_id,
        }
    }

    /// The stable wire tag for this event (the `vhc_events.kind` column + display discriminator).
    pub fn kind(&self) -> &'static str {
        match self {
            VhcEvent::Phase { .. } => "phase",
            VhcEvent::Progress { .. } => "progress",
            VhcEvent::RoundOutcome { .. } => "round_outcome",
            VhcEvent::Contribution { .. } => "contribution",
            VhcEvent::Warning { .. } => "warning",
            VhcEvent::Error { .. } => "error",
        }
    }
}

/// The vhc-training sub-surface (spec §10.4): discover/join/leave runs, set the participation
/// policy, report training hardware, and subscribe to run events. Every method defaults to
/// [`ApiError::Unsupported`] / empty so a transport with no vhc service inherits the surface; the
/// node's [`NodeApi`](crate::NodeApi) binds the real implementation.
#[async_trait]
pub trait VhcApi: Send + Sync {
    /// Discovered + joined runs, each annotated with the node-computed [`VhcEligibility`] (§6.5).
    async fn vhc_run_list(&self) -> Result<Vec<VhcRunSummary>, ApiError> {
        Err(ApiError::Unsupported("vhc_run_list".into()))
    }

    /// One run's full detail (`None` if unknown to this node).
    async fn vhc_run_detail(&self, _run_id: String) -> Result<Option<VhcRunDetail>, ApiError> {
        Err(ApiError::Unsupported("vhc_run_detail".into()))
    }

    /// Join a run under `policy` (durable intent; idempotent via `op_id`, ADR-006). The node persists
    /// the desired-state flag so a restart re-converges (rejoins) without app involvement (§10.3).
    async fn vhc_join(
        &self,
        _run_id: String,
        _policy: VhcPolicy,
        _op_id: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("vhc_join".into()))
    }

    /// Leave a run (durable intent; idempotent via `op_id`).
    async fn vhc_leave(
        &self,
        _run_id: String,
        _mode: VhcLeaveMode,
        _op_id: String,
    ) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("vhc_leave".into()))
    }

    /// Pause a run (durable owner intent; idempotent via `op_id`): training hard-stops, the
    /// run's resource reservations release, and the node never reconverges the run — across
    /// restarts — until the owner resumes it.
    async fn vhc_pause(&self, _run_id: String, _op_id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("vhc_pause".into()))
    }

    /// Resume a paused run (durable owner intent; idempotent via `op_id`): resources are
    /// re-admitted against the owner's current ledgers — a refusal is typed and LOUD, leaving
    /// the run paused — and participation reconverges.
    async fn vhc_resume(&self, _run_id: String, _op_id: String) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("vhc_resume".into()))
    }

    /// Consume a committed run-level module-upgrade record and drive the live module switch
    /// for the run's role-instance (idempotent via `op_id`). `upgrade_record` is the
    /// canonical-CBOR authorized record; the node validates it fail-closed against the run's
    /// rebuilt transition chain (domain, hash-link, monotone epoch, authority threshold),
    /// provisions the post-switch identity, and drives the local transaction through its
    /// worker. The record is never trusted as presented — validation happens node-side.
    async fn vhc_switch_module(
        &self,
        _run_id: String,
        _upgrade_record: Vec<u8>,
        _op_id: String,
    ) -> Result<VhcSwitchOutcome, ApiError> {
        Err(ApiError::Unsupported("vhc_switch_module".into()))
    }

    /// Set the default participation policy for newly-joined runs (§10.5).
    async fn vhc_set_policy(&self, _policy: VhcPolicy) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("vhc_set_policy".into()))
    }

    /// This node's training-capability report (probe results + active lanes).
    async fn vhc_hardware_report(&self) -> Result<VhcHardwareReport, ApiError> {
        Err(ApiError::Unsupported("vhc_hardware_report".into()))
    }

    /// The disk-custody report for the VHC runs root (wire v45): free space, ledgered usage,
    /// the configured envelope, pressure, and the per-run breakdown by reclaim class.
    async fn vhc_disk_usage(&self) -> Result<VhcDiskUsage, ApiError> {
        Err(ApiError::Unsupported("vhc_disk_usage".into()))
    }

    /// Safely wipe one run's local state (wire v45). Refuses while the run is live. Wipes
    /// recoverable state (journal + spill) always; wipes archived evidence (payload + archive
    /// planes) only when `include_evidence`. Never touches the identity keystore — `base.key`
    /// survives every wipe.
    async fn vhc_disk_wipe(
        &self,
        _run_id: String,
        _include_evidence: bool,
    ) -> Result<VhcDiskWipeOutcome, ApiError> {
        Err(ApiError::Unsupported("vhc_disk_wipe".into()))
    }

    /// Subscribe to run events (all runs when `run_id` is `None`, else one run). Rides the existing
    /// feed machinery: the default is an empty stream; the node returns a live [`VhcEventStream`].
    async fn vhc_subscribe(&self, _run_id: Option<String>) -> Result<VhcEventStream, ApiError> {
        Ok(stream::empty().boxed())
    }
}
