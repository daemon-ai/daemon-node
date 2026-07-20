// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The training-worker wire protocol (spec §10.2) — [`Command`] (down) / [`Event`] (up) + a CBOR
//! codec.
//!
//! The node-side supervisor (`daemon-vhc-supervisor`) and the `daemon-vhc-host` worker exchange these
//! frames over a length-framed stdio cut (`daemon_provision::CutChannel`, `Framing::Length`), same
//! supervision contract as `daemon-infer` (respawn with backoff, crash-loop meltdown). Each frame
//! body is CBOR; the `u32`-length prefix is handled by the channel, so this module owns only the
//! body [`encode`]/[`decode`] — the exact conventions of [`daemon_infer::protocol`].
//!
//! This is the **worker** protocol (node ↔ `daemon-vhc-host` child) — distinct from the **vhc**
//! control protocol (`daemon-vhc.cddl`, lane P). It lives in `daemon-vhc-session` (not the client)
//! so lane E's `daemon-vhc-host` worker implements the worker side against it later (§10.1:
//! `daemon-vhc-host` depends on `daemon-vhc-session`).
//!
//! **v1 retirement (WS4).** The §10.2 [`Event`] stream dropped three producer-less variants —
//! `MicroBatch`, `OomLadder`, `ResyncProgress` — that only the retired v1 live-attach path ever
//! emitted. Their behaviors are superseded (micro-batch/OOM adaptation → the claim()-based
//! admission funnel + a typed `BudgetMemory` trap plus node churn, no halving ladder;
//! resync-progress → typed-checkpoint / late-join / record-replay). This is a vocabulary change,
//! not a re-numbering: CBOR enum variants are name-keyed, so removing an unused variant leaves the
//! remaining frames encoding-identical, and a stale worker emitting one now decodes as an
//! unknown-variant error (logged, dropped) at the node pump — the same posture as any undecodable
//! frame.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::seam::RoundId;

/// How a peer participates on hardware primarily wanted for inference (§10.5). Mirrors the wire
/// `vhc-policy-mode` (§10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyMode {
    /// Always available for training.
    Always,
    /// Only when there is no inference activity + the user is idle.
    Idle,
    /// Within `daemon-schedule` cron windows.
    Scheduled,
    /// Manual start/stop only.
    Manual,
}

/// The participation policy for a joined run (§10.4/§10.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinPolicy {
    /// The availability mode.
    pub mode: PolicyMode,
    /// A VRAM cap in MiB (`0` = uncapped) — also tightens eligibility (§6.5).
    pub vram_cap_mb: u32,
    /// A duty-cycle percentage (`0..=100`).
    pub duty_cycle_pct: u8,
    /// An optional cron schedule (for [`PolicyMode::Scheduled`]).
    pub schedule: Option<String>,
}

/// How a peer leaves a run (§10.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LeaveMode {
    /// Finish the current round, then leave.
    Graceful,
    /// Leave immediately (abort any in-flight work).
    Immediate,
}

/// A classified worker failure (§10.2) — the vhc analogue of `daemon_infer`'s `ErrorClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// VRAM/host allocator OOM. Post-v1 semantics (decisions D-6): a memory breach is the module's
    /// typed `BudgetMemory` trap, and the node's recovery is **churn** — preempt/replace the
    /// role-instance (preemption-as-churn, [`Command::Throttle`]`{ paused }`) — never a dynamic
    /// micro-batch halving/re-probe ladder. Micro-batching is guest/module policy fixed at admission
    /// from the run schedule; it is not renegotiated at runtime.
    OutOfMemory,
    /// A transient network/transport fault — retry in place.
    Transient,
    /// State divergence — the resync path (§9).
    Desync,
    /// An experiment-module trap / sandbox-budget violation — leave the run, worker unharmed (§13).
    Module,
    /// Unrecoverable (crash-loop meltdown, internal bug).
    Fatal,
    /// Cancelled cooperatively.
    Cancelled,
}

/// The worker's capability vocabulary, reported by [`Command::Probe`] (§6.5, §10.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// The tensor-ABI major version the worker implements.
    pub abi_version: u16,
    /// The host-vocabulary ops the worker implements (`name@version`, §5.2).
    pub ops: Vec<String>,
    /// The payload stores the worker can speak (`r2`, `iroh-blobs`, …).
    pub payload_stores: Vec<String>,
}

/// One execution backend this worker build can actually run, as probed on this host — the
/// per-backend half of the capability advertisement (architecture §9: the measured half of
/// fleet heterogeneity). The worker advertises one record per COMPILED lane whose runtime
/// probe found a device (plus the always-present CPU record); the measured selection ladder
/// consumes exactly these records, so advertisement and selection cannot diverge.
///
/// Wire discipline: a new additive record type carried by [`Hardware::backends`]
/// (`#[serde(default)]` — pre-extension frames decode to an empty list). All-fields-default
/// construction is meaningful (an unknown/legacy record).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapability {
    /// The engine lane slug (`"cpu"` / `"wgpu"` / `"cuda"` — `BackendKind` vocabulary).
    pub backend: String,
    /// The device backend CLASS the run pre-screen matches (`device_min.backend_class`
    /// vocabulary): `"cuda"`, `"vulkan"`, `"metal"`, `"dx12"`, or `"cpu"`.
    pub class: String,
    /// The adapter/device name as probed (operator-facing; never branched on).
    pub adapter: String,
    /// The device index this record describes (device placement vocabulary; single-device
    /// probes report `0`).
    pub device_index: u32,
    /// Dedicated device memory in MiB (the platform-correct budget source; `0` = none/unknown).
    pub vram_mb: u64,
    /// The largest single device allocation in MiB (the per-buffer ceiling — the number the
    /// fleet preflight checks the pinned model's largest tensor against; `0` = unbounded or
    /// unknown).
    pub max_alloc_mb: u64,
    /// Shared/spillover memory in MiB (GTT / unified pool; `0` = none).
    pub shared_mb: u64,
    /// Whether the device shares host DRAM (unified-memory budget math applies).
    pub unified: bool,
    /// Whether the lane can serve RIGHT NOW (device probed AND its runtime staged — for CUDA
    /// the two-leg NVRTC gate). A present-but-not-ready record advertises the hardware while
    /// keeping the lane unselectable.
    pub ready: bool,
}

/// A hardware + capability probe result (§10.2 — extends the daemon-models `HardwareProbe`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    /// The number of usable GPUs.
    pub gpus: u32,
    /// Dedicated VRAM in MiB (across GPUs). On a unified/integrated GPU this is the small
    /// dedicated carve-out (sysfs `mem_info_vram_total`), NOT the usable budget — that spills into
    /// [`Self::shared_mb`].
    pub vram_mb: u64,
    /// Shared / unified spillover memory in MiB (GTT — sysfs `mem_info_gtt_total`): the host DRAM
    /// an integrated GPU can page tensors into beyond [`Self::vram_mb`]. `0` = none (a classic
    /// discrete GPU). **Additive (Merge 2):** `#[serde(default)]` keeps pre-Merge-2 `Hardware`
    /// payloads (which lack this field) decodable, and a `shared_mb == 0` value serializes
    /// compatibly. This is the worker↔node protocol type; it does NOT cross the VhcApi wire (the
    /// app-facing DTO is `daemon_api::VhcHardwareReport`, mapped in the node service), so no CDDL
    /// / wire-version change is implied.
    #[serde(default)]
    pub shared_mb: u64,
    /// Installed host RAM in MiB (§5.1 host-RAM planning).
    pub ram_mb: u64,
    /// The backend lanes the worker was built with (`cpu`, `cuda`, `rocm`, `vulkan`).
    pub backend_lanes: Vec<String>,
    /// The structured per-backend capability records (one per compiled lane with a probed
    /// device, plus the CPU record) — what the measured backend-selection ladder consumes and
    /// what the fleet preflight sizes models against. **Additive:** `#[serde(default)]` keeps
    /// pre-extension `Hardware` frames decodable (empty list = no structured advertisement).
    #[serde(default)]
    pub backends: Vec<BackendCapability>,
    /// The capability vocabulary (ABI version, ops, payload stores).
    pub capabilities: WorkerCapabilities,
    /// Measured uplink in kbit/s.
    pub up_kbps: u64,
    /// Measured downlink in kbit/s.
    pub down_kbps: u64,
    /// Free disk for the data/checkpoint cache in MiB.
    pub disk_free_mb: u64,
    /// The measured throughput class (§6.3: `c1`..`c4`).
    pub throughput_class: String,
}

/// The immutable **admitted tuple** an assessment produces (architecture §6.3): the exact
/// assessed identity of what will run, carried to `JoinRun` and re-verified there. Join rederives
/// every artifact-addressed field from the artifacts it is about to run and compares field by
/// field; any mismatch aborts the join and reruns assessment (a stale or swapped artifact can
/// never be silently joined). The two revisions are node-owned monotonic counters (the device
/// profile revision and the owner policy revision) stamped at assess and compared by the node.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedTuple {
    /// Content address of the module actually admitted.
    pub module_hash: [u8; 32],
    /// Canonical-CBOR hash of the run config.
    pub config_hash: [u8; 32],
    /// Canonical-CBOR hash of the FULL grants document (ABI §2.6).
    pub grants_hash: [u8; 32],
    /// The module's `claim()` output as admitted (hash of the claim bytes).
    pub claim_hash: [u8; 32],
    /// The run identity — the genesis-envelope hash.
    pub genesis_hash: [u8; 32],
    /// The envelope-level role label admitted.
    pub role: String,
    /// The role-instance incarnation admitted.
    pub incarnation: u64,
    /// The hardware-probe revision consulted for eligibility (node-owned counter).
    #[serde(default)]
    pub device_profile_rev: u64,
    /// The owner-policy revision consulted at admission (node-owned counter).
    #[serde(default)]
    pub owner_policy_rev: u64,
    /// The execution backend the assessment SELECTED on the measured ladder (`BackendKind`
    /// slug: `"cpu"` / `"burn-ndarray"` / `"wgpu"` / `"cuda"`). Join reruns the same measured
    /// selection and compares — a device that disappeared or shrank between assess and join
    /// rederives differently and refuses typed (the claim-revalidation flow; the node
    /// reassesses). **Additive:** `#[serde(default)]` — a pre-extension tuple carries `""`,
    /// which compares equal only to another pre-extension rederivation.
    #[serde(default)]
    pub backend: String,
    /// The device placement the assessment selected (`EngineConfig.gpu_index` vocabulary;
    /// single-device hosts place on `0`). Compared at join like [`Self::backend`]. Additive.
    #[serde(default)]
    pub gpu_index: u32,
}

impl AdmittedTuple {
    /// The first field that differs from `rederived` — the fields join recomputes: the
    /// artifact-addressed hashes/identity AND the measured backend placement (`backend` /
    /// `gpu_index`, whose join-time rederivation IS the device-claim revalidation: a device
    /// that vanished between assess and join rederives a different selection and refuses here,
    /// never OOMs later). `None` when they match. The node-owned revisions are compared
    /// separately by the node, not here.
    #[must_use]
    pub fn first_artifact_mismatch(&self, rederived: &Self) -> Option<&'static str> {
        if self.module_hash != rederived.module_hash {
            Some("module_hash")
        } else if self.config_hash != rederived.config_hash {
            Some("config_hash")
        } else if self.grants_hash != rederived.grants_hash {
            Some("grants_hash")
        } else if self.claim_hash != rederived.claim_hash {
            Some("claim_hash")
        } else if self.genesis_hash != rederived.genesis_hash {
            Some("genesis_hash")
        } else if self.role != rederived.role {
            Some("role")
        } else if self.incarnation != rederived.incarnation {
            Some("incarnation")
        } else if self.backend != rederived.backend {
            Some("backend")
        } else if self.gpu_index != rederived.gpu_index {
            Some("gpu_index")
        } else {
            None
        }
    }
}

/// The live-upgrade target a pre-switch assessment evaluates (ABI §10.3): the committed
/// transition-chain facts — the epoch the switch activates, the hash-pinned target module, and
/// the committed record's grants anchor. Carried inside [`Command::AssessRun`] so the worker
/// (which alone touches module bytes) can compute the post-switch admitted tuple's claim hash;
/// every field is re-verified fail-closed by the session's own pre-fence checks regardless.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwitchTarget {
    /// The committed transition-chain epoch the switch activates (§8.1).
    pub epoch: u64,
    /// The hash-pinned target module for the assessed role at `epoch`.
    pub new_module: [u8; 32],
    /// The committed upgrade record's grants anchor — the re-derived grants document must hash
    /// to it or the assessment refuses.
    pub grants_hash: [u8; 32],
}

/// The classified terminal outcome of one role-instance run — the typed exit the role session
/// produces and the worker reports verbatim ([`Event::RunTerminated`]). The node's run-instance
/// state machine consumes the class; the reason strings are operator-facing detail, never
/// branched on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalOutcome {
    /// The module signaled run end (`da_run` returned an outcome code; `0` is the clean end).
    Completed {
        /// The module's outcome code (ABI §4.5). `0` on a clean run end.
        outcome: u32,
    },
    /// Owner intent ended the run (a leave command — graceful or immediate).
    Left {
        /// The content hash (blake3 hex) of the drain snapshot persisted to the payload plane
        /// on a graceful leave; `None` when the leave was immediate or the module had nothing
        /// to snapshot.
        checkpoint: Option<String>,
    },
    /// A recoverable environment fault: transport loss, provider fault, resource-budget breach,
    /// an unrecoverable inbound sequence gap with no backfill. The node may re-converge (rejoin
    /// as a new incarnation) under its retry budget.
    FailedRetryable {
        /// Operator-facing detail (never branched on).
        reason: String,
    },
    /// A non-recoverable failure: a module trap, an admission identity mismatch, a certificate
    /// refusal, an init/migrate refusal. The node must not rejoin without owner action.
    FailedTerminal {
        /// Operator-facing detail (never branched on).
        reason: String,
    },
}

/// A self-assessment result for a run (§6.5, §10.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eligibility {
    /// Whether this peer can join.
    pub eligible: bool,
    /// Human-readable reasons (why-not).
    pub reasons: Vec<String>,
    /// Per-dimension headroom (e.g. `"vram_mb" => 4096`).
    pub headroom: Vec<(String, i64)>,
    /// The immutable admitted tuple this assessment produced (architecture §6.3), carried to the
    /// node and on into `JoinRun`. `None` on an ineligible verdict (nothing was admitted).
    /// Additive `#[serde(default)]` — pre-tuple frames decode to `None`.
    #[serde(default)]
    pub admitted_tuple: Option<AdmittedTuple>,
    /// The split typed admission-refusal code slug (ABI Draft 3 §1.5 — e.g.
    /// `"AbiUnsupportedMajor"`, `"AbiDeclarationMismatch"`, `"ModuleHashMismatch"`), set when
    /// ineligibility is a driver-selection/ABI admission refusal rather than a resource verdict.
    /// Additive `#[serde(default)]` field (the established wire discipline): pre-A0 frames decode
    /// to `None`, and refusals stay admission *outcomes* on the `Assessed` surface — never a
    /// runtime `Event::Error`, never a reused v1 `TrapCode::AbiMismatch` (decisions D2).
    #[serde(default)]
    pub refusal_code: Option<String>,
}

/// The default checkpoint-pointer kind for an unstamped `CheckpointPublished` frame: the
/// graceful-leave drain snapshot (the pre-cadence sole pointer source).
fn checkpoint_kind_drain() -> String {
    "drain".to_string()
}

/// A parent → worker command frame (§10.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Command {
    /// Report hardware + capability vocabulary (cached; refreshed on hardware/config change).
    Probe,
    /// Assess a run against this peer's effective resources — read-only, no GPU allocation.
    AssessRun {
        /// The run envelope bytes (opaque here; lane P owns the schema — MERGE-1).
        envelope: Vec<u8>,
        /// The envelope role LABEL to assess for (node-directed role selection — the seat-claim
        /// path assesses the coordinator role). `None` = the single-trainer default (the first
        /// non-coordinator role). Additive `#[serde(default)]`; a label absent from the genesis
        /// role set is a typed refusal.
        #[serde(default)]
        role: Option<String>,
        /// When present, assess the LIVE-UPGRADE TARGET instead of the genesis-pinned module
        /// (ABI §10.3 pre-switch assessment): the worker resolves the target bytes by their
        /// committed content hash, re-derives the grants document against the record's grants
        /// anchor, runs the same claim admission funnel, and answers [`Event::Assessed`] with a
        /// post-switch admitted tuple (its claim hash computed over the target's re-evaluated
        /// claim — the node never touches module bytes). Additive `#[serde(default)]`; boxed
        /// for variant-size parity.
        #[serde(default)]
        switch_target: Option<Box<SwitchTarget>>,
    },
    /// Join a run, then stream [`Event`]s.
    JoinRun {
        /// The run to join.
        run_id: String,
        /// The coordinator endpoint (WS/HTTP).
        coordinator: String,
        /// Opaque credentials (daemon-credentials reference / token bytes).
        credentials: Vec<u8>,
        /// The participation policy.
        policy: JoinPolicy,
        /// The immutable admitted tuple assessment produced (architecture §6.3), carried back so
        /// join can rederive-and-compare before running. Additive `#[serde(default)]` — a join
        /// without a tuple (the self-driven interim / a pre-tuple caller) authors and checks its
        /// own. Boxed for variant-size parity (serde encodes through the box — no wire change).
        #[serde(default)]
        admitted_tuple: Option<Box<AdmittedTuple>>,
    },
    /// GPU-governor lever (§10.5). `paused` promises memory, not just time: the worker aborts any
    /// in-flight guest call, drops the wasm instance + GPU allocations, and keeps only CPU masters.
    Throttle {
        /// A VRAM cap in MiB (`None` = unchanged).
        vram_cap_mb: Option<u32>,
        /// A duty-cycle percentage (`None` = unchanged).
        duty_cycle_pct: Option<u8>,
        /// Whether training is paused (preemption-as-churn).
        paused: bool,
    },
    /// Leave a run.
    Leave {
        /// The run to leave.
        run_id: String,
        /// How to leave.
        mode: LeaveMode,
    },
    /// Initiate a live module upgrade for a running instance (ABI §10.3; architecture §5.4).
    ///
    /// The run-level upgrade record has ALREADY committed to the transition chain (the global
    /// event — deliverable 1; it advanced the epoch). This command tells the worker to run the
    /// **local** half of the transaction against the already-committed target epoch: quiesce →
    /// snapshot → owner-law re-admission (grant-expanding **fails closed**) → migrate → validate →
    /// activate, or roll back and retry / leave. It is **hash-pinned and authority-bound**: the
    /// worker admits the target under owner law only if its bytes match `new_module`, and re-derives
    /// the grants document against `grants_hash` (the committed record's anchor). Success answers
    /// with [`Event::ModuleSwitched`]; a fail-closed / exhausted transaction leaves the run and
    /// answers [`Event::Error`]`{ class: Module, .. }` (the worker is unharmed; the node churns).
    SwitchModule {
        /// The run whose role-instance is upgrading.
        run_id: String,
        /// The committed transition-chain epoch this switch activates (§8.1). No host advances the
        /// chain — it advanced when the upgrade record committed.
        epoch: u64,
        /// The role whose module is switching. The target module is a pure function of
        /// `(run_id, epoch, role)` via the chain (ABI §8.1/§10.3).
        role: String,
        /// The hash-pinned target module for `role` at `epoch` — admission refuses any mismatch.
        new_module: [u8; 32],
        /// The committed upgrade record's grants anchor
        /// (`daemon_vhc_proto::UpgradeRecordBody::grants_hash`) the owner-law re-check verifies the
        /// re-derived grants document against.
        grants_hash: [u8; 32],
        /// The drain deadline in ms (§4.4). The node clamps it to the lane's
        /// `quiesce_deadline_max_ms` ceiling before sending (§9.6).
        deadline_ms: u64,
        /// The node-assessed admitted tuple for the POST-SWITCH identity (architecture §6.3):
        /// `module_hash = new_module`, `grants_hash` = the committed record's anchor, and —
        /// load-bearing — the node-minted, never-reused incarnation the migrated instance runs
        /// as (a live upgrade mints a new incarnation, ABI §8.1/§10.3). The node provisions the
        /// new incarnation's per-run key and its certificate — bound to
        /// `(run, epoch, role, new incarnation, new_module)` — in the identity keystore BEFORE
        /// sending this command; the worker resolves both read-only by reference and refuses
        /// typed when either is absent or mis-scoped (the certificate re-issuance handshake).
        /// Additive `#[serde(default)]`; a switch without a tuple is a typed refusal on the
        /// production path. Boxed for variant-size parity (serde encodes through the box — no
        /// wire change).
        #[serde(default)]
        admitted_tuple: Option<Box<AdmittedTuple>>,
    },
    /// Ask the worker to exit cleanly.
    Shutdown,
    /// Liveness probe (answered with [`Event::Pong`]).
    Ping,
}

/// A worker → parent event frame (§10.2). All are persisted / fanned out by the node (§10.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// The worker started and is ready for commands; reports its capability vocabulary.
    Ready {
        /// The worker's capabilities.
        capabilities: WorkerCapabilities,
    },
    /// A [`Command::Probe`] result.
    Probed(Hardware),
    /// An [`Command::AssessRun`] result.
    Assessed(Eligibility),
    /// The run's phase advanced.
    RunPhase {
        /// The run this phase belongs to.
        run_id: String,
        /// The phase name (`warmup`, `train`, `witness`, `cooldown`, …).
        phase: String,
        /// The current epoch.
        epoch: u64,
        /// The current round.
        round: RoundId,
        /// The emitting role-instance's generation counter (the never-reused incarnation id).
        /// The node discards events stamped with a stale generation — a reaped instance's late
        /// events can never mutate its replacement. Additive `#[serde(default)]`: pre-counter
        /// frames decode to `0` (un-stamped).
        #[serde(default)]
        generation: u64,
    },
    /// Progress within a round.
    RoundProgress {
        /// The inner step within the round.
        inner_step: u32,
        /// The current loss.
        loss: f32,
        /// Throughput in tokens/s.
        tokens_per_s: f32,
        /// Bytes uploaded this round so far.
        up_bytes: u64,
        /// Bytes downloaded this round so far.
        down_bytes: u64,
        /// Peers this round involves.
        peers: u32,
        /// The emitting role-instance's generation counter (see [`Event::RunPhase`]).
        #[serde(default)]
        generation: u64,
    },
    /// The §6.4 protocol as seen from this peer, at round end.
    RoundOutcome {
        /// The round that ended.
        round: RoundId,
        /// The number of payloads committed to the record.
        committed: u32,
        /// The number of payloads this peer ingested.
        ingested: u32,
        /// Whether this peer stalled (missed a committed payload at the barrier).
        stalled: bool,
        /// The post-ingest state digest (§5.6).
        digest: [u8; 16],
        /// The emitting role-instance's generation counter (see [`Event::RunPhase`]).
        #[serde(default)]
        generation: u64,
    },
    /// A named scalar metric readout.
    Metric {
        /// The metric name.
        name: String,
        /// The metric value.
        value: f64,
    },
    /// A checkpoint was published.
    CheckpointPublished {
        /// The round the checkpoint covers.
        round: RoundId,
        /// The checkpoint's content hash (blake3 hex).
        hash: String,
        /// A locator (store key / blob ticket).
        location: String,
        /// The emitting role-instance's generation counter (see [`Event::RunPhase`]).
        #[serde(default)]
        generation: u64,
        /// The pointer kind (spec §9): `"live"` — the periodic mid-run cadence — or `"drain"` —
        /// a graceful-leave drain snapshot. Additive `#[serde(default)]`; an unstamped frame
        /// reads as a drain snapshot (the pre-cadence sole source).
        #[serde(default = "checkpoint_kind_drain")]
        kind: String,
    },
    /// A non-fatal warning (desync-warning, straggling, quota).
    Warning {
        /// The warning class.
        class: String,
        /// A short human-readable detail.
        detail: String,
    },
    /// A classified failure.
    Error {
        /// The failure class (maps to the node's recovery loop).
        class: ErrorClass,
        /// A short human-readable detail.
        detail: String,
    },
    /// A live module upgrade activated (ABI §10.3 step 6; architecture §5.4): the old module
    /// quiesced at the fence, its state migrated, and the new module resumed under the target epoch
    /// — without a process restart, det digests continuous across the fence. The answer to a
    /// successful [`Command::SwitchModule`].
    ModuleSwitched {
        /// The run that upgraded.
        run_id: String,
        /// The epoch now running locally (the already-committed target epoch).
        epoch: u64,
        /// The new module hash now bound to the role-instance.
        module: [u8; 32],
        /// Rollback-and-retry cycles used before activation (`0` on a clean first migration).
        retries: u32,
        /// The emitting role-instance's generation counter (see [`Event::RunPhase`]).
        #[serde(default)]
        generation: u64,
    },
    /// A module switch was REFUSED before the upgrade transaction touched the running instance
    /// (ABI §10.3 pre-transaction refusals: no live instance, an unresolvable/mismatched target
    /// artifact, missing or mis-scoped re-issued identity, an admission refusal evaluated ahead
    /// of the fence). The old module keeps running untouched — distinct from a POST-quiesce
    /// failure, which leaves the run with a terminal [`Event::RunTerminated`].
    SwitchRefused {
        /// The run whose switch was refused.
        run_id: String,
        /// The (still-running) role-instance's generation counter.
        generation: u64,
        /// Why the switch was refused (operator-facing detail, never branched on).
        reason: String,
    },
    /// Join aborted: the admitted tuple carried from assessment does not match the tuple
    /// rederived from the artifacts at join (architecture §6.3) — a stale or swapped artifact.
    /// The node's recovery is to reassess, never to run on stale admission.
    AdmittedTupleMismatch {
        /// The run whose join aborted.
        run_id: String,
        /// The first tuple field that differed (e.g. `"module_hash"`).
        field: String,
        /// The emitting role-instance's generation counter (see [`Event::RunPhase`]).
        #[serde(default)]
        generation: u64,
    },
    /// The role-instance run reached its terminal state: exactly one per spawned role task, the
    /// last run-scoped event a generation emits. The worker reports the session's classified
    /// exit verbatim; the node's run-instance state machine transitions on the class
    /// (idempotently — duplicate delivery must not double-release resources).
    RunTerminated {
        /// The run that terminated.
        run_id: String,
        /// The terminated role-instance's generation counter (see [`Event::RunPhase`]).
        generation: u64,
        /// The classified terminal outcome.
        outcome: TerminalOutcome,
    },
    /// Liveness reply to [`Command::Ping`].
    Pong,
}

/// A CBOR codec error (mirrors `daemon_infer::protocol::CodecError`).
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoding a frame to CBOR failed.
    #[error("cbor encode: {0}")]
    Encode(String),
    /// Decoding a frame from CBOR failed.
    #[error("cbor decode: {0}")]
    Decode(String),
}

/// Encode a frame body to CBOR bytes (the `CutChannel` adds the length prefix).
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

/// Decode a CBOR frame body.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    ciborium::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

// ---------------------------------------------------------------------------
// JoinRun.credentials contract (A3, frozen at Merge 2)
// ---------------------------------------------------------------------------
//
// `Command::JoinRun.credentials` stays an OPAQUE `Vec<u8>` on the frozen worker wire (§10.2). A3
// defines the canonical-CBOR **schema** carried in it: the node's `VhcService` / run-authoring
// path AUTHORS a [`JoinCredentials`], `to_bytes()` it into `credentials`, and the worker's live
// attach `from_bytes()` it to construct the live plane + engine. It is a NEW additive type — no
// `Command`/`Event` shape change — so a worker built without the live-attach feature ignores the
// bytes exactly as before.

/// The WS coordinator auth for the live attach (the canonical-CBOR mirror of
/// `daemon_vhc_net::ws_client::WsAuth`, defined here so the dependency-light protocol crate owns
/// the credentials schema; the worker converts it under the `ws` feature — never hardcoded).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WsAuthSpec {
    /// No auth (bare `ws://` dev target).
    #[default]
    None,
    /// `Authorization: Bearer <token>` (the gateway `vhc:join` path).
    Bearer(String),
    /// The internal identity headers `x-daemon-org-id` / `x-daemon-actor` (direct-to-`apps/vhc`).
    Internal {
        /// The org id header value.
        org_id: String,
        /// The actor header value.
        actor: String,
    },
}

/// One iroh roster peer (the canonical-CBOR mirror of `daemon_vhc_net::IrohPeer` for the
/// credentials body: an `endpoint_id` + reachability). Direct addrs are `"ip:port"` strings so the
/// protocol crate takes no `std::net` serialization dependency; the worker parses them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohRosterPeer {
    /// The peer's iroh `EndpointId` (32 raw bytes).
    pub endpoint_id: [u8; 32],
    /// Direct socket addresses (`"ip:port"`); may be empty for relay-only reachability.
    #[serde(default)]
    pub direct_addrs: Vec<String>,
    /// Home relay URL (NAT-proof reachability); `None` for direct-only (LAN/loopback).
    #[serde(default)]
    pub relay_url: Option<String>,
}

/// The optional iroh half of the credentials. Present ⇒ the worker builds a
/// `DualPlane(WsControlPlane, IrohGossip)`; absent ⇒ WS-only (the T0 baseline).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohCredentials {
    /// The iroh secret key (32 bytes) — separate from the node ed25519 identity (§7.2).
    pub secret_key: [u8; 32],
    /// Envelope-pinned relay URLs (empty ⇒ direct-only / loopback).
    #[serde(default)]
    pub relay_urls: Vec<String>,
    /// The bootstrap roster (may be empty; roster updates arrive dynamically as coordinator frames
    /// and are wired to `IrohGossip::update_roster`).
    #[serde(default)]
    pub roster: Vec<IrohRosterPeer>,
}

/// The engine + corpus knobs the worker's `RoundEngine` needs (from the run's declared config /
/// frozen envelope). Deterministic across peers so the digest transcript agrees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineParams {
    /// Inner steps per round (§5.1 cadence).
    pub steps_per_round: u32,
    /// Micro-batch (sequences) within an inner step.
    pub micro_batch: u32,
    /// Fetch-recovery budget before a stalled peer leaves for the epoch (§6.4).
    pub stall_rounds_max: u32,
    /// Round-boundary checkpoint cadence (§9); `0` disables.
    pub checkpoint_every_rounds: u32,
    /// §9 resync-replay window: the payload-retention floor (rounds) a rejoining peer may replay
    /// forward from the latest checkpoint. A desync/rejoin gap wider than this waits for the next
    /// epoch checkpoint instead (`plan_resync`). Additive field: `#[serde(default)]` keeps
    /// credential buffers minted before it existed decodable; `0` ⇒ unbounded (replay whatever is retained).
    #[serde(default)]
    pub payload_retention_rounds: u64,
    /// §7.3 receive-side per-peer payload cap in bytes (`0` = uncapped) — the worker mirrors the DO
    /// shell's pre-filter (Merge-1 Decision 2).
    #[serde(default)]
    pub update_max_bytes: u64,
    /// Synthetic-corpus seed (deterministic training data — identical across peers → agreeing
    /// digests). Mirrors `daemon_vhc_session::data::Corpus::synthetic(seed, shards, tokens_per_shard,
    /// seq_len)`.
    pub corpus_seed: u64,
    /// Synthetic-corpus shard count.
    pub corpus_shards: u32,
    /// Synthetic-corpus tokens per shard.
    pub corpus_tokens_per_shard: u64,
    /// Synthetic-corpus sequence length (tokens).
    pub corpus_seq_len: u32,
    /// Clamp corpus token ids into the experiment's vocabulary (`token % clamp`; `0` = no clamp) —
    /// the deterministic per-token stand-in for tokenizing the corpus at the model's vocab (the B3
    /// live-e2e shim recipe, applied identically by every peer so digests agree).
    #[serde(default)]
    pub corpus_vocab_clamp: u32,
}

/// The canonical-CBOR body of [`Command::JoinRun`]'s `credentials` (A3, frozen at Merge 2). Authored
/// node-side, parsed by the worker's live attach.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoinCredentials {
    /// The peer's node ed25519 signing-key seed (32 bytes) — the `RoundEngine`'s `Join` signer
    /// identity (§7.2). Also the `PeerId` this peer contributes under.
    pub node_secret: [u8; 32],
    /// WS coordinator auth.
    #[serde(default)]
    pub ws_auth: WsAuthSpec,
    /// The epoch roster (node pubkeys, 32-byte) the engine folds each round.
    pub roster: Vec<[u8; 32]>,
    /// blake3 of the frozen envelope (§6.1) — the iroh topic-derivation input + admission binding.
    pub envelope_hash: [u8; 32],
    /// Optional iroh transport (dual-plane). Absent ⇒ WS-only (T0 baseline).
    #[serde(default)]
    pub iroh: Option<IrohCredentials>,
    /// Optional presign base for the `R2Store` payload plane (e.g.
    /// `http://127.0.0.1:8795/api/v1/vhc`). Absent ⇒ the content-addressed `FsContentStore`
    /// fallback under the node-delivered run dir (tests / LAN).
    #[serde(default)]
    pub presign_base: Option<String>,
    /// The engine + corpus knobs.
    pub engine: EngineParams,
}

impl JoinCredentials {
    /// Encode to the canonical-CBOR bytes carried in `JoinRun.credentials`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CodecError> {
        encode(self)
    }

    /// Decode from `JoinRun.credentials` bytes. An empty / non-`JoinCredentials` buffer decodes to
    /// an error, which the worker treats as "no live attach" (the self-driven fallback).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        decode(bytes)
    }
}

// ---------------------------------------------------------------------------
// The role-session plane selection (the live-transport credentials body)
// ---------------------------------------------------------------------------

/// The optional iroh half of a [`SessionCredentials`]: PUBLIC reachability only — relay URLs and
/// the bootstrap roster. The iroh transport SECRET never rides the wire; the worker resolves it
/// from the identity keystore by reference (architecture §7.2; [CI-9]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrohPlane {
    /// Envelope-pinned relay URLs (empty ⇒ direct-only / loopback).
    #[serde(default)]
    pub relay_urls: Vec<String>,
    /// The bootstrap roster (may be empty; later roster updates ride the control plane).
    #[serde(default)]
    pub roster: Vec<IrohRosterPeer>,
    /// The node-chosen iroh bind address (`"ip:port"`). The node pins the port BEFORE publishing
    /// this peer's roster record, so the published direct addresses and the socket the worker
    /// actually binds agree by construction. `None` ⇒ the worker binds ephemeral (relay-only
    /// reachability, or tests that wire rosters by hand). **Additive:** `#[serde(default)]`
    /// keeps pre-extension credential buffers decodable.
    #[serde(default)]
    pub bind_addr: Option<String>,
}

/// The canonical-CBOR body of `Command::JoinRun.credentials` for the ROLE-SESSION live attach:
/// the node-authored **plane selection** the worker builds its transport providers from.
/// Everything here is NON-SECRET by construction — signing identity, the iroh transport secret,
/// and (with the credential-authorship rework) token material are keystore references, never
/// command-payload bytes ([CI-9]; D-P8 redaction by construction).
///
/// Replaces the engine-era [`JoinCredentials`] on the role-session path (that shape carried a
/// raw signing seed, the iroh secret, and round vocabulary — all three retired from this
/// surface; [`JoinCredentials`] survives only until its remaining harness consumers retire).
/// A buffer that decodes as neither is "no live attach" (the in-process seat / typed refusal).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCredentials {
    /// The run's cryptographic identity (the genesis hash): the iroh topic-derivation input and
    /// the sanity anchor against the resolved run.
    pub genesis_hash: [u8; 32],
    /// WS control-plane base URL. `None` ⇒ the worker uses `JoinRun.coordinator` (the
    /// node-resolved, allowlist-checked endpoint).
    #[serde(default)]
    pub ws_base: Option<String>,
    /// WS coordinator auth. `None` for unauthenticated targets (local relay lanes); the
    /// secret-bearing modes move behind a keystore reference with the credential-authorship
    /// rework (this field then carries only non-secret forms).
    #[serde(default)]
    pub ws_auth: WsAuthSpec,
    /// Optional iroh half. Present ⇒ the worker composes a dual plane (WS + iroh gossip);
    /// absent ⇒ WS-only.
    #[serde(default)]
    pub iroh: Option<IrohPlane>,
    /// Optional presign base for the content-addressed R2 payload plane. Absent ⇒ the
    /// filesystem content store under the run's node-delivered state dir.
    #[serde(default)]
    pub presign_base: Option<String>,
    /// Bootstrap peer certificates known at join (PUBLIC §12.3 records — e.g. the seat holder's);
    /// later arrivals ride the control plane as distribution records.
    #[serde(default)]
    pub peer_certs: Vec<daemon_vhc_proto::RunKeyCertificate>,
    /// The keystore reference of the node-authored per-run CREDENTIALS RECORD (WS/presign auth
    /// material) — resolved by the worker against the identity store it already holds by path
    /// reference; token material never rides this body ([CI-9] custody; D-P8 redaction by
    /// construction). `None` ⇒ the body's `ws_auth` stands alone (unauthenticated local lanes).
    #[serde(default)]
    pub secret_ref: Option<String>,
    /// Advisory expiry (unix ms) of the referenced credentials record: past it the worker's
    /// planes re-resolve the record before dialing (the node refreshes by atomically rewriting
    /// the record — never by restarting the session). `0` = no expiry.
    #[serde(default)]
    pub expires_at_ms: u64,
    /// An optional LATE-JOIN checkpoint restore (§9/§10.2): the node-resolved registry checkpoint
    /// pointer this instance restores from before it runs. `None` = a fresh start from genesis.
    #[serde(default)]
    pub restore: Option<CheckpointRestore>,
}

/// A late-join checkpoint restore reference (the node-resolved registry pointer): the round the
/// checkpoint covers and the content address (blake3) of the checkpoint document on the payload
/// plane. The worker fetches the bytes, hash-verifies, and migrates the fresh instance from them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRestore {
    /// The round the checkpoint captures (post-ingest state).
    pub round: u64,
    /// blake3 of the checkpoint document (the payload-plane content key).
    pub hash: [u8; 32],
}

/// The node-authored per-run CREDENTIALS RECORD the [`SessionCredentials::secret_ref`] points at
/// (persisted in the identity keystore, 0600, atomically rewritten on refresh — NEVER on the
/// command wire): the WS/presign auth material and its advisory expiry. Lives here so the
/// dependency-light protocol crate owns the schema beside the plane selection that references it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialsRecord {
    /// The WS coordinator + presign auth (the secret-bearing modes live HERE, by reference).
    #[serde(default)]
    pub ws_auth: WsAuthSpec,
    /// Advisory expiry (unix ms; `0` = none) — mirrored onto the wire body so planes know when
    /// to re-resolve.
    #[serde(default)]
    pub expires_at_ms: u64,
}

impl SessionCredentials {
    /// Encode to the canonical-CBOR bytes carried in `JoinRun.credentials`.
    ///
    /// # Errors
    /// CBOR encode failure.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CodecError> {
        encode(self)
    }

    /// Decode from `JoinRun.credentials` bytes. A buffer that is not a `SessionCredentials`
    /// (empty, or an engine-era body) is the worker's "no live attach" signal.
    ///
    /// # Errors
    /// The bytes are not a `SessionCredentials`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        decode(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_command(cmd: Command) {
        let bytes = encode(&cmd).expect("encode command");
        let back: Command = decode(&bytes).expect("decode command");
        assert_eq!(cmd, back);
    }

    #[test]
    fn admitted_tuple_first_artifact_mismatch_reports_the_differing_field() {
        let base = AdmittedTuple {
            module_hash: [1; 32],
            config_hash: [2; 32],
            grants_hash: [3; 32],
            claim_hash: [4; 32],
            genesis_hash: [5; 32],
            role: "trainer".into(),
            incarnation: 1,
            device_profile_rev: 9,
            owner_policy_rev: 9,
            backend: "wgpu".into(),
            gpu_index: 0,
        };
        // Identical artifact fields: no mismatch (node-owned revs are compared separately).
        let mut other = base.clone();
        other.device_profile_rev = 0;
        other.owner_policy_rev = 0;
        assert_eq!(base.first_artifact_mismatch(&other), None);
        // A swapped module is caught first.
        let mut swapped = base.clone();
        swapped.module_hash = [0xFF; 32];
        assert_eq!(base.first_artifact_mismatch(&swapped), Some("module_hash"));
        // A changed grants document is caught.
        let mut regrant = base.clone();
        regrant.grants_hash = [0xAB; 32];
        assert_eq!(base.first_artifact_mismatch(&regrant), Some("grants_hash"));
        // The measured backend placement is rederived at join like the artifact hashes: a
        // device that disappeared (the rederivation lands on a different rung) or a moved
        // placement refuses typed — the claim-revalidation flow.
        let mut lost_device = base.clone();
        lost_device.backend = "cpu".into();
        assert_eq!(base.first_artifact_mismatch(&lost_device), Some("backend"));
        let mut moved = base.clone();
        moved.gpu_index = 1;
        assert_eq!(base.first_artifact_mismatch(&moved), Some("gpu_index"));
    }

    fn round_trip_event(ev: Event) {
        let bytes = encode(&ev).expect("encode event");
        let back: Event = decode(&bytes).expect("decode event");
        assert_eq!(ev, back);
    }

    #[test]
    fn commands_round_trip() {
        round_trip_command(Command::Probe);
        round_trip_command(Command::AssessRun {
            envelope: vec![1, 2, 3, 4],
            role: Some("coordinator".into()),
            switch_target: None,
        });
        round_trip_command(Command::AssessRun {
            envelope: vec![1, 2, 3, 4],
            role: Some("trainer".into()),
            switch_target: Some(Box::new(SwitchTarget {
                epoch: 3,
                new_module: [7u8; 32],
                grants_hash: [9u8; 32],
            })),
        });
        round_trip_command(Command::JoinRun {
            run_id: "run-42".into(),
            coordinator: "wss://coord.example/vhc".into(),
            credentials: vec![0xde, 0xad, 0xbe, 0xef],
            policy: JoinPolicy {
                mode: PolicyMode::Idle,
                vram_cap_mb: 12_000,
                duty_cycle_pct: 80,
                schedule: Some("0 2 * * *".into()),
            },
            admitted_tuple: Some(Box::new(AdmittedTuple {
                module_hash: [0x11; 32],
                config_hash: [0x22; 32],
                grants_hash: [0x33; 32],
                claim_hash: [0x44; 32],
                genesis_hash: [0x55; 32],
                role: "trainer".into(),
                incarnation: 3,
                device_profile_rev: 7,
                owner_policy_rev: 2,
                backend: "cuda".into(),
                gpu_index: 0,
            })),
        });
        round_trip_command(Command::Throttle {
            vram_cap_mb: Some(8_000),
            duty_cycle_pct: None,
            paused: true,
        });
        round_trip_command(Command::Leave {
            run_id: "run-42".into(),
            mode: LeaveMode::Graceful,
        });
        round_trip_command(Command::SwitchModule {
            run_id: "run-42".into(),
            epoch: 1,
            role: "worker".into(),
            new_module: [0x5A; 32],
            grants_hash: [0x6B; 32],
            deadline_ms: 5_000,
            admitted_tuple: Some(Box::new(AdmittedTuple {
                module_hash: [0x5A; 32],
                config_hash: [0x00; 32],
                grants_hash: [0x6B; 32],
                claim_hash: [0x7C; 32],
                genesis_hash: [0x8D; 32],
                role: "worker".into(),
                incarnation: 4,
                device_profile_rev: 1,
                owner_policy_rev: 1,
                backend: "cpu".into(),
                gpu_index: 0,
            })),
        });
        round_trip_command(Command::Shutdown);
        round_trip_command(Command::Ping);
    }

    /// The post-switch admitted tuple is additive on the switch command: a frame authored before
    /// the field existed (a CBOR map WITHOUT it) still decodes, with the tuple defaulting to
    /// `None` — and the production path then refuses the switch typed rather than guessing an
    /// identity.
    #[test]
    fn switch_tuple_is_additive_back_compatible() {
        #[derive(serde::Serialize)]
        enum LegacyCommand {
            SwitchModule {
                run_id: String,
                epoch: u64,
                role: String,
                new_module: [u8; 32],
                grants_hash: [u8; 32],
                deadline_ms: u64,
            },
        }
        let legacy = LegacyCommand::SwitchModule {
            run_id: "run-42".into(),
            epoch: 1,
            role: "worker".into(),
            new_module: [0x5A; 32],
            grants_hash: [0x6B; 32],
            deadline_ms: 5_000,
        };
        let decoded: Command = decode(&encode(&legacy).expect("encode legacy")).expect(
            "a pre-tuple switch command still decodes (the field is additive, never re-keying)",
        );
        assert!(matches!(
            decoded,
            Command::SwitchModule {
                admitted_tuple: None,
                ..
            }
        ));
    }

    #[test]
    fn events_round_trip() {
        round_trip_event(Event::Ready {
            capabilities: WorkerCapabilities {
                abi_version: 1,
                ops: vec!["matmul@1".into(), "flash_attn@1".into()],
                payload_stores: vec!["r2".into(), "iroh-blobs".into()],
            },
        });
        round_trip_event(Event::Probed(Hardware {
            gpus: 2,
            vram_mb: 24_000,
            shared_mb: 120_000,
            ram_mb: 64_000,
            backend_lanes: vec!["cuda".into()],
            backends: vec![
                BackendCapability {
                    backend: "cuda".into(),
                    class: "cuda".into(),
                    adapter: "discrete 24 GiB".into(),
                    device_index: 0,
                    vram_mb: 24_000,
                    max_alloc_mb: 24_000,
                    shared_mb: 0,
                    unified: false,
                    ready: true,
                },
                BackendCapability {
                    backend: "cpu".into(),
                    class: "cpu".into(),
                    adapter: "host".into(),
                    device_index: 0,
                    vram_mb: 0,
                    max_alloc_mb: 0,
                    shared_mb: 0,
                    unified: false,
                    ready: true,
                },
            ],
            capabilities: WorkerCapabilities {
                abi_version: 1,
                ops: vec!["adamw_step@1".into()],
                payload_stores: vec!["r2".into()],
            },
            up_kbps: 50_000,
            down_kbps: 200_000,
            disk_free_mb: 500_000,
            throughput_class: "c3".into(),
        }));
        round_trip_event(Event::Assessed(Eligibility {
            eligible: false,
            reasons: vec!["vram below floor".into()],
            headroom: vec![("vram_mb".into(), -2048), ("ram_mb".into(), 16_000)],
            refusal_code: None,
            admitted_tuple: None,
        }));
        round_trip_event(Event::Assessed(Eligibility {
            eligible: false,
            reasons: vec!["AbiUnsupportedMajor: module declares abi major 2".into()],
            headroom: Vec::new(),
            refusal_code: Some("AbiUnsupportedMajor".into()),
            admitted_tuple: None,
        }));
        round_trip_event(Event::Assessed(Eligibility {
            eligible: true,
            reasons: vec!["admitted".into()],
            headroom: Vec::new(),
            refusal_code: None,
            admitted_tuple: Some(AdmittedTuple {
                module_hash: [0x11; 32],
                config_hash: [0x22; 32],
                grants_hash: [0x33; 32],
                claim_hash: [0x44; 32],
                genesis_hash: [0x55; 32],
                role: "trainer".into(),
                incarnation: 1,
                device_profile_rev: 4,
                owner_policy_rev: 1,
                backend: "cpu".into(),
                gpu_index: 0,
            }),
        }));
        round_trip_event(Event::AdmittedTupleMismatch {
            run_id: "run-42".into(),
            field: "module_hash".into(),
            generation: 3,
        });
        round_trip_event(Event::RunPhase {
            run_id: "run-42".into(),
            phase: "train".into(),
            epoch: 3,
            round: 128,
            generation: 3,
        });
        round_trip_event(Event::RoundProgress {
            inner_step: 12,
            loss: 2.5,
            tokens_per_s: 4200.0,
            up_bytes: 1024,
            down_bytes: 8192,
            peers: 7,
            generation: 3,
        });
        round_trip_event(Event::RoundOutcome {
            round: 128,
            committed: 6,
            ingested: 6,
            stalled: false,
            digest: [0xAB; 16],
            generation: 3,
        });
        round_trip_event(Event::Metric {
            name: "grad_norm".into(),
            value: 0.75,
        });
        round_trip_event(Event::CheckpointPublished {
            round: 200,
            hash: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
            location: "r2://run-42/ckpt-200.safetensors".into(),
            generation: 3,
            kind: "live".into(),
        });
        round_trip_event(Event::Warning {
            class: "straggle".into(),
            detail: "late fetch".into(),
        });
        for class in [
            ErrorClass::OutOfMemory,
            ErrorClass::Transient,
            ErrorClass::Desync,
            ErrorClass::Module,
            ErrorClass::Fatal,
            ErrorClass::Cancelled,
        ] {
            round_trip_event(Event::Error {
                class,
                detail: "boom".into(),
            });
        }
        round_trip_event(Event::ModuleSwitched {
            run_id: "run-42".into(),
            epoch: 3,
            module: [0x5A; 32],
            retries: 1,
            generation: 3,
        });
        round_trip_event(Event::SwitchRefused {
            run_id: "run-42".into(),
            generation: 3,
            reason: "target artifact does not hash to the committed module".into(),
        });
        for outcome in [
            TerminalOutcome::Completed { outcome: 0 },
            TerminalOutcome::Left { checkpoint: None },
            TerminalOutcome::Left {
                checkpoint: Some(
                    "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262".into(),
                ),
            },
            TerminalOutcome::FailedRetryable {
                reason: "control plane lost".into(),
            },
            TerminalOutcome::FailedTerminal {
                reason: "module trapped: GrantViolation".into(),
            },
        ] {
            round_trip_event(Event::RunTerminated {
                run_id: "run-42".into(),
                generation: 3,
                outcome,
            });
        }
        round_trip_event(Event::Pong);
    }

    /// The generation counter is additive on every run-scoped event: a frame authored before the
    /// counter existed (a CBOR map WITHOUT the field) still decodes, with `generation`
    /// defaulting to `0` (un-stamped) — the node treats `0` as pre-counter, never as a stale
    /// generation.
    #[test]
    fn generation_counter_is_additive_back_compatible() {
        #[derive(serde::Serialize)]
        enum LegacyEvent {
            RunPhase {
                run_id: String,
                phase: String,
                epoch: u64,
                round: RoundId,
            },
            RoundOutcome {
                round: RoundId,
                committed: u32,
                ingested: u32,
                stalled: bool,
                digest: [u8; 16],
            },
        }
        let legacy_phase = LegacyEvent::RunPhase {
            run_id: "run-42".into(),
            phase: "train".into(),
            epoch: 3,
            round: 128,
        };
        let decoded: Event = decode(&encode(&legacy_phase).expect("encode legacy")).expect(
            "a pre-counter RunPhase still decodes (the counter is additive, never re-keying)",
        );
        assert_eq!(
            decoded,
            Event::RunPhase {
                run_id: "run-42".into(),
                phase: "train".into(),
                epoch: 3,
                round: 128,
                generation: 0,
            }
        );
        let legacy_outcome = LegacyEvent::RoundOutcome {
            round: 128,
            committed: 6,
            ingested: 6,
            stalled: false,
            digest: [0xAB; 16],
        };
        let decoded: Event =
            decode(&encode(&legacy_outcome).expect("encode legacy")).expect("decodes");
        assert!(
            matches!(decoded, Event::RoundOutcome { generation: 0, .. }),
            "missing counter defaults to 0"
        );
    }

    /// The backend-placement fields are additive on the admitted tuple, and the structured
    /// backend records are additive on `Hardware`: frames authored before the extension (CBOR
    /// maps WITHOUT the fields) still decode, with the backend slug defaulting to `""`, the
    /// placement to `0`, and the record list to empty.
    #[test]
    fn backend_placement_and_capability_fields_are_additive_back_compatible() {
        #[derive(serde::Serialize)]
        struct PreExtensionTuple {
            module_hash: [u8; 32],
            config_hash: [u8; 32],
            grants_hash: [u8; 32],
            claim_hash: [u8; 32],
            genesis_hash: [u8; 32],
            role: String,
            incarnation: u64,
        }
        let legacy = PreExtensionTuple {
            module_hash: [1; 32],
            config_hash: [2; 32],
            grants_hash: [3; 32],
            claim_hash: [4; 32],
            genesis_hash: [5; 32],
            role: "trainer".into(),
            incarnation: 2,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&legacy, &mut bytes).expect("encode pre-extension tuple");
        let decoded: AdmittedTuple =
            ciborium::from_reader(bytes.as_slice()).expect("pre-extension tuple decodes");
        assert_eq!(decoded.backend, "", "missing backend defaults empty");
        assert_eq!(decoded.gpu_index, 0, "missing placement defaults to 0");

        #[derive(serde::Serialize)]
        struct PreExtensionHardware {
            gpus: u32,
            vram_mb: u64,
            ram_mb: u64,
            backend_lanes: Vec<String>,
            capabilities: WorkerCapabilities,
            up_kbps: u64,
            down_kbps: u64,
            disk_free_mb: u64,
            throughput_class: String,
        }
        let legacy_hw = PreExtensionHardware {
            gpus: 1,
            vram_mb: 4096,
            ram_mb: 128_000,
            backend_lanes: vec!["vulkan".into(), "cpu".into()],
            capabilities: WorkerCapabilities::default(),
            up_kbps: 0,
            down_kbps: 0,
            disk_free_mb: 0,
            throughput_class: "c1".into(),
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&legacy_hw, &mut bytes).expect("encode pre-extension hardware");
        let decoded: Hardware =
            ciborium::from_reader(bytes.as_slice()).expect("pre-extension hardware decodes");
        assert!(
            decoded.backends.is_empty(),
            "missing records default to an empty advertisement"
        );
    }

    /// `join_credentials_round_trip`: the A3 `JoinCredentials` schema carried in
    /// `JoinRun.credentials` round-trips through the canonical-CBOR codec, incl. the optional iroh /
    /// presign halves and the WS-auth variants.
    #[test]
    fn join_credentials_round_trip() {
        let creds = JoinCredentials {
            node_secret: [0x11; 32],
            ws_auth: WsAuthSpec::Internal {
                org_id: "org_live".into(),
                actor: "key:live".into(),
            },
            roster: vec![[0x22; 32], [0x33; 32]],
            envelope_hash: [0xEE; 32],
            iroh: Some(IrohCredentials {
                secret_key: [0x44; 32],
                relay_urls: vec!["http://127.0.0.1:3340".into()],
                roster: vec![IrohRosterPeer {
                    endpoint_id: [0x55; 32],
                    direct_addrs: vec!["127.0.0.1:4550".into()],
                    relay_url: Some("http://127.0.0.1:3340".into()),
                }],
            }),
            presign_base: Some("http://127.0.0.1:8795/api/v1/vhc".into()),
            engine: EngineParams {
                steps_per_round: 2,
                micro_batch: 2,
                stall_rounds_max: 3,
                checkpoint_every_rounds: 0,
                update_max_bytes: 1 << 20,
                corpus_seed: 7,
                corpus_shards: 4,
                corpus_tokens_per_shard: 256,
                corpus_seq_len: 8,
                corpus_vocab_clamp: 64,
                payload_retention_rounds: 16,
            },
        };
        let bytes = creds.to_bytes().expect("encode credentials");
        let back = JoinCredentials::from_bytes(&bytes).expect("decode credentials");
        assert_eq!(creds, back);

        // The WS-only baseline (no iroh, no presign) also round-trips, and a non-credentials buffer
        // is a decode error (the worker's "no live attach → self-driven fallback" signal).
        let ws_only = JoinCredentials {
            iroh: None,
            presign_base: None,
            ws_auth: WsAuthSpec::None,
            ..creds
        };
        let back2 = JoinCredentials::from_bytes(&ws_only.to_bytes().unwrap()).unwrap();
        assert_eq!(ws_only, back2);
        assert!(JoinCredentials::from_bytes(&[]).is_err());
    }

    /// The role-session plane selection round-trips (WS-only baseline, and the full dual-plane +
    /// presign + bootstrap-cert form), and an engine-era `JoinCredentials` body is NOT a
    /// `SessionCredentials` — the structural "no live attach" signal (no secret seed can ever be
    /// mistaken for a plane selection).
    #[test]
    fn session_credentials_round_trip_and_reject_the_engine_era_body() {
        let ws_only = SessionCredentials {
            genesis_hash: [0xE1; 32],
            ws_base: None,
            ws_auth: WsAuthSpec::None,
            iroh: None,
            presign_base: None,
            peer_certs: Vec::new(),
            secret_ref: None,
            expires_at_ms: 0,
            restore: None,
        };
        let back = SessionCredentials::from_bytes(&ws_only.to_bytes().unwrap()).unwrap();
        assert_eq!(back, ws_only);

        // The additive secret_ref/expiry round-trip too, and a pre-secret_ref body still decodes.
        let referenced = SessionCredentials {
            secret_ref: Some("coordinator-3.creds".into()),
            expires_at_ms: 1_900_000_000_000,
            ..ws_only.clone()
        };
        assert_eq!(
            SessionCredentials::from_bytes(&referenced.to_bytes().unwrap()).unwrap(),
            referenced
        );

        let base = daemon_vhc_proto::SigningKey::from_bytes(&[7; 32]);
        let run_key =
            daemon_vhc_proto::peer_id(&daemon_vhc_proto::SigningKey::from_bytes(&[9; 32]));
        let cert = daemon_vhc_proto::RunKeyCertificate::issue(
            &base,
            daemon_vhc_proto::CertScope {
                run_id: daemon_vhc_proto::Hash([0xE1; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: daemon_vhc_proto::Hash([2; 32]),
            },
            run_key,
        )
        .unwrap();
        let full = SessionCredentials {
            genesis_hash: [0xE1; 32],
            ws_base: Some("http://127.0.0.1:8795/api/v1/vhc".into()),
            ws_auth: WsAuthSpec::Internal {
                org_id: "org_live".into(),
                actor: "key:live".into(),
            },
            iroh: Some(IrohPlane {
                relay_urls: vec!["http://127.0.0.1:3340".into()],
                roster: vec![IrohRosterPeer {
                    endpoint_id: [0x55; 32],
                    direct_addrs: vec!["127.0.0.1:4550".into()],
                    relay_url: None,
                }],
                bind_addr: Some("127.0.0.1:4551".into()),
            }),
            presign_base: Some("http://127.0.0.1:8795/api/v1/vhc".into()),
            peer_certs: vec![cert],
            secret_ref: Some("coordinator-1.creds".into()),
            expires_at_ms: 1_800_000_000_000,
            restore: Some(CheckpointRestore {
                round: 42,
                hash: [0x7C; 32],
            }),
        };
        let back = SessionCredentials::from_bytes(&full.to_bytes().unwrap()).unwrap();
        assert_eq!(back, full);

        // The engine-era body (raw seed + roster + engine knobs) is structurally NOT a plane
        // selection; and an empty buffer never is.
        let legacy = JoinCredentials {
            node_secret: [0x11; 32],
            ws_auth: WsAuthSpec::None,
            roster: vec![[0x22; 32]],
            envelope_hash: [0xEE; 32],
            iroh: None,
            presign_base: None,
            engine: EngineParams {
                steps_per_round: 2,
                micro_batch: 2,
                stall_rounds_max: 3,
                checkpoint_every_rounds: 0,
                update_max_bytes: 0,
                corpus_seed: 7,
                corpus_shards: 4,
                corpus_tokens_per_shard: 256,
                corpus_seq_len: 8,
                corpus_vocab_clamp: 64,
                payload_retention_rounds: 0,
            },
        };
        assert!(SessionCredentials::from_bytes(&legacy.to_bytes().unwrap()).is_err());
        assert!(SessionCredentials::from_bytes(&[]).is_err());
    }

    /// The node-pinned iroh bind address is additive on the plane selection: a credentials
    /// buffer authored before the field existed (a CBOR map WITHOUT it) still decodes, with
    /// `bind_addr` defaulting to `None` (the worker binds ephemeral — the pre-extension
    /// behavior, unchanged).
    #[test]
    fn iroh_plane_bind_addr_is_additive_back_compatible() {
        #[derive(serde::Serialize)]
        struct LegacyIrohPlane {
            relay_urls: Vec<String>,
            roster: Vec<IrohRosterPeer>,
        }
        let legacy = LegacyIrohPlane {
            relay_urls: vec!["http://127.0.0.1:3340".into()],
            roster: vec![IrohRosterPeer {
                endpoint_id: [0x55; 32],
                direct_addrs: vec!["127.0.0.1:4550".into()],
                relay_url: None,
            }],
        };
        let decoded: IrohPlane = decode(&encode(&legacy).expect("encode legacy")).expect(
            "a pre-extension iroh plane still decodes (the field is additive, never re-keying)",
        );
        assert_eq!(decoded.bind_addr, None, "missing bind addr defaults None");
        assert_eq!(decoded.roster.len(), 1);
    }

    /// `engine_params_payload_retention_is_additive_back_compatible`: the resync-work field
    /// `payload_retention_rounds` is additive. An older `EngineParams` (a CBOR map WITHOUT the
    /// field) still decodes, defaulting to `0` (unbounded) — keeping A3's back-compat contract
    /// (a non-decoding-in-full buffer never regresses existing joiners).
    #[test]
    fn engine_params_payload_retention_is_additive_back_compatible() {
        #[derive(serde::Serialize)]
        struct LegacyEngineParams {
            steps_per_round: u32,
            micro_batch: u32,
            stall_rounds_max: u32,
            checkpoint_every_rounds: u32,
            update_max_bytes: u64,
            corpus_seed: u64,
            corpus_shards: u32,
            corpus_tokens_per_shard: u64,
            corpus_seq_len: u32,
            corpus_vocab_clamp: u32,
        }
        let legacy = LegacyEngineParams {
            steps_per_round: 2,
            micro_batch: 2,
            stall_rounds_max: 3,
            checkpoint_every_rounds: 2,
            update_max_bytes: 0,
            corpus_seed: 7,
            corpus_shards: 4,
            corpus_tokens_per_shard: 256,
            corpus_seq_len: 8,
            corpus_vocab_clamp: 64,
        };
        let decoded: EngineParams =
            decode(&encode(&legacy).expect("encode legacy")).expect("decode");
        assert_eq!(
            decoded.payload_retention_rounds, 0,
            "missing field defaults to 0 (unbounded)"
        );
        assert_eq!(decoded.checkpoint_every_rounds, 2);
    }

    /// `hardware_shared_mb_is_additive_back_compatible`: the Merge-2 `shared_mb` field is additive.
    /// A pre-Merge-2 `Hardware` payload (a CBOR map WITHOUT `shared_mb`) still decodes, with
    /// `shared_mb` defaulting to 0; and a `shared_mb == 0` value is carried through a round-trip.
    #[test]
    fn hardware_shared_mb_is_additive_back_compatible() {
        // A pre-Merge-2 `Hardware` had no `shared_mb`. Model it with a mirror struct and decode the
        // legacy bytes into the current type: `#[serde(default)]` fills `shared_mb = 0`.
        #[derive(serde::Serialize)]
        struct LegacyHardware {
            gpus: u32,
            vram_mb: u64,
            ram_mb: u64,
            backend_lanes: Vec<String>,
            capabilities: WorkerCapabilities,
            up_kbps: u64,
            down_kbps: u64,
            disk_free_mb: u64,
            throughput_class: String,
        }
        let legacy = LegacyHardware {
            gpus: 1,
            vram_mb: 4096,
            ram_mb: 124_419,
            backend_lanes: vec!["vulkan".into(), "cpu".into()],
            capabilities: WorkerCapabilities::default(),
            up_kbps: 0,
            down_kbps: 0,
            disk_free_mb: 0,
            throughput_class: "c1".into(),
        };
        let bytes = encode(&legacy).expect("encode legacy");
        let decoded: Hardware = decode(&bytes).expect("legacy Hardware still decodes");
        assert_eq!(decoded.shared_mb, 0, "missing field defaults to 0");
        assert_eq!(decoded.vram_mb, 4096);

        // Full round-trip preserves a real GTT number.
        let hw = Hardware {
            gpus: 1,
            vram_mb: 4096,
            shared_mb: 120_000,
            ram_mb: 124_419,
            ..Hardware::default()
        };
        let back: Hardware = decode(&encode(&hw).expect("encode")).expect("decode");
        assert_eq!(back.shared_mb, 120_000);
    }

    /// `engine_params_without_corpus_ref_still_decode` (corpus contract): an `EngineParams` CBOR
    /// carrying the RETIRED `corpus` reference field (a map with an extra key) still decodes —
    /// the production data path is the genesis-pinned chunk-addressed corpus manifest, and the
    /// engine-era credentials schema simply no longer carries a corpus reference.
    #[test]
    fn engine_params_without_corpus_ref_still_decode() {
        // A retired-era EngineParams carried an extra `corpus` key; decode ignores it.
        #[derive(serde::Serialize)]
        struct RetiredEngineParams {
            steps_per_round: u32,
            micro_batch: u32,
            stall_rounds_max: u32,
            checkpoint_every_rounds: u32,
            corpus_seed: u64,
            corpus_shards: u32,
            corpus_tokens_per_shard: u64,
            corpus_seq_len: u32,
            corpus: Option<u32>,
        }
        let retired = RetiredEngineParams {
            steps_per_round: 2,
            micro_batch: 2,
            stall_rounds_max: 3,
            checkpoint_every_rounds: 0,
            corpus_seed: 7,
            corpus_shards: 4,
            corpus_tokens_per_shard: 256,
            corpus_seq_len: 8,
            corpus: None,
        };
        let decoded: EngineParams =
            decode(&encode(&retired).expect("encode")).expect("EngineParams decodes");
        assert_eq!(decoded.corpus_shards, 4);
    }
}
