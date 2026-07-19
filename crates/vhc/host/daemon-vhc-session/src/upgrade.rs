// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The host-enforced upgrade transaction (architecture §5.4; ABI §10.3; refactor §9) — the
//! **local** half of the two-key model.
//!
//! Live-run module replacement separates two authorities that must never be conflated
//! (architecture §5.4):
//!
//! - **Run-internal authorization is policy** — an [`UpgradeRecord`] committed to the transition
//!   chain ([`daemon_vhc_proto::TransitionChain`]) **once, globally, before any host acts**. That
//!   global commit is the run-level event; it advances the epoch. It is deliverable 1
//!   ([`daemon_vhc_proto::transition`]) and is NOT performed here — this module runs *after* the
//!   chain has advanced, against the already-committed [`EpochDescriptor`].
//! - **Machine-owner authorization is host law** — at every switch the host re-runs admission
//!   (fetch, verify hash, re-check the new manifest/claim against the owner's standing policy).
//!   **Grant-expanding upgrades fail closed**: the worker exits the run rather than silently
//!   granting more (architecture §5.4; refactor invariant 7).
//!
//! The switch itself is a **host-enforced transaction** (ABI §10.3), because `switch_module` plus
//! `migrate` alone underspecifies failure. This module is the transaction state machine:
//!
//! 1. **Quiesce** at the SDK-selected fence: drain the compute queue, freeze event delivery
//!    (authoritative frames spool, advisory events coalesce), the old module snapshots and returns
//!    `QuiesceReady`.
//! 2. **Snapshot** old-module state + the journal cursor — durable (§10.2 + §8.4 barrier). Captured
//!    here behind the [`SnapshotSeam`] (see its docs: the labelled seam E1's typed manifest swaps
//!    into).
//! 3. **Admit** the new module: full owner-law re-check + `claim()` re-evaluation. Grant-expanding
//!    fails closed.
//! 4. **Migrate** under budget: `da_migrate(descriptor)` on the new run instance.
//! 5. **Validate** readiness: `da_migrate` returns `Ready`.
//! 6. **Activate locally, atomically**: the instance binding swaps to the already-committed
//!    transition — **no host advances the global chain**; it advanced when the upgrade record was
//!    committed. Spooled frames drain into the new instance.
//! 7. **Roll back** to the snapshot on any local failure *before* activation, then retry or leave.
//!    **A failed local migration never rolls back the chain and never resumes the old epoch.**
//!
//! ## Cross-track seam (E1 owns the manifest format)
//!
//! E1 (checkpoint bridge + typed manifests) OWNS the checkpoint/state-manifest vocabulary and
//! merges first. Until then this transaction snapshots behind [`SnapshotSeam`] — an **opaque
//! snapshot blob + journal cursor** — so swapping to E1's typed manifest at merge is mechanical.
//! This module deliberately defines **no** manifest section vocabulary of its own.
//!
//! ## What this module is (and is not)
//!
//! This is the transaction's **orchestration and its invariants** — the ordering and the
//! fail-closed / rollback / never-roll-back-the-chain rules the spec fixes. The five side-effecting
//! steps are abstracted behind [`UpgradeSteps`] so the invariants are unit-tested deterministically;
//! the production adapter wires each step onto the real host primitives (the `PumpHandle` quiesce,
//! `daemon_vhc_host::run::admit`, and the migrate-instance `start_run` path). The wasm-level
//! drills that drive a real migratable guest through these steps live in the host testkit.

use daemon_vhc_proto::{AdmittedQuotas, EpochDescriptor, Hash};
use daemon_vhc_sdk_consensus::checkpoint::CheckpointManifest;

/// The **labelled snapshot seam** (architecture §5.3/§5.4; ABI §10.3 step 2): the upgrade
/// transaction's "snapshot state + journal cursor".
///
/// E1 (the checkpoint bridge + typed manifests track) OWNS the manifest format and merged first;
/// this seam's interim opaque blob was swapped — mechanically, as designed — for E1's
/// [`CheckpointManifest`] at that merge ("checkpointing and migration are one discipline", the
/// checkpoint module's own words). The orchestrator never inspects the manifest's sections; the
/// producing/consuming adapters do, so the seam stays a label, not a vocabulary fork.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotSeam {
    /// E1's typed checkpoint manifest describing the snapshot captured at the quiesce fence:
    /// content-addressed sections (module state et al.) + the consensus digest.
    pub manifest: CheckpointManifest,
    /// The journal cursor at the quiesce fence (§10.3 step 2): the ordinal the new instance's
    /// journal continues from, so replay of the pre-upgrade prefix is exact. (Also expressible
    /// as the manifest's `JournalPosition` section; carried explicitly so the orchestrator can
    /// log it without section decoding.)
    pub journal_cursor: u64,
}

/// Why the local upgrade transaction left the run (the worker exits; the chain is NOT rolled back
/// and the old epoch is NOT resumed — §10.3 step 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaveReason {
    /// The target epoch descriptor does not carry this role (a malformed transition — should never
    /// happen for a well-formed chain, but the transaction fails closed if it does).
    UnknownRole(String),
    /// The quiesce drain failed (deadline exceeded / forced interruption, ABI §4.4/§11.3).
    QuiesceFailed(String),
    /// Owner-law re-check refused the new module (owner policy / lane / claim, `admit`).
    OwnerRefused(String),
    /// The upgrade would expand grants beyond the previously-admitted set — **fail closed**
    /// (architecture §5.4; refactor invariant 7). Carries the offending bound.
    GrantExpansion(String),
    /// `migrate`/`validate` failed and the retry budget was exhausted; the worker leaves rather
    /// than resume the old epoch.
    MigrateExhausted(String),
}

impl core::fmt::Display for LeaveReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRole(r) => write!(f, "target epoch descriptor has no role `{r}`"),
            Self::QuiesceFailed(e) => write!(f, "quiesce failed: {e}"),
            Self::OwnerRefused(e) => write!(f, "owner-law re-check refused the upgrade: {e}"),
            Self::GrantExpansion(e) => {
                write!(f, "grant-expanding upgrade refused (fail closed): {e}")
            }
            Self::MigrateExhausted(e) => write!(f, "migration exhausted its retry budget: {e}"),
        }
    }
}

/// The outcome of a local upgrade transaction. In **both** arms the transition chain is untouched
/// and the old epoch is never resumed — the run has advanced regardless (§10.3 step 7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalUpgradeOutcome {
    /// The new epoch's instance activated locally (§10.3 step 6). `retries` counts how many
    /// rollback-and-retry cycles preceded success (`0` on a clean first migration).
    Activated {
        /// The epoch now running locally (the already-committed target epoch).
        epoch: u64,
        /// Rollback-and-retry cycles used before activation.
        retries: u32,
    },
    /// The worker left the run. The chain is NOT rolled back; the old epoch is NOT resumed.
    Left {
        /// The already-committed target epoch the local switch failed to activate (the run stays
        /// at this epoch globally; only this node left).
        epoch: u64,
        /// Why the worker left.
        reason: LeaveReason,
    },
}

/// A failed migrate/validate/activate step — carries whether it looked transient (e.g. a
/// mid-migration crash the retry can recover, refactor §9 crash drill) for logging; the transaction
/// retries either way up to its budget.
#[derive(Debug, Clone)]
pub struct StepFailure {
    /// A human-readable detail.
    pub detail: String,
}

impl StepFailure {
    /// Construct a step failure from any displayable error.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

/// The five side-effecting steps of the transaction (ABI §10.3), abstracted so the ordering and
/// fail-closed / rollback invariants are unit-testable. The production adapter implements each over
/// the real host primitives; the migrate-instance driving and the wasm drills live in the host
/// testkit.
pub trait UpgradeSteps {
    /// **Step 1–2.** Deliver `Quiesce{Upgrade, deadline_ms}`, wait for the old module to drain and
    /// `snapshot_state` durably, and capture the snapshot behind the [`SnapshotSeam`]. A drain
    /// deadline miss / forced interruption (ABI §4.4/§11.3) is an `Err`.
    ///
    /// # Errors
    /// A [`StepFailure`] describing the quiesce failure.
    fn quiesce(&mut self, deadline_ms: u64) -> Result<SnapshotSeam, StepFailure>;

    /// **Step 3.** Re-run owner-law admission for the new module (owner policy + lane + `claim()`,
    /// `daemon_vhc_host::run::admit`) and return the newly-admitted grant quotas. An owner/lane
    /// refusal is an `Err` (the worker leaves). Grant-expansion vs the previous grants is checked by
    /// the orchestrator against this result.
    ///
    /// # Errors
    /// A [`StepFailure`] carrying the owner/lane refusal.
    fn readmit(
        &mut self,
        new_module: Hash,
        new_grants_hash: Hash,
    ) -> Result<AdmittedQuotas, StepFailure>;

    /// **Step 4–5.** Instantiate the new run instance (tag-13 reason 2, before `da_init`), run
    /// `da_init`, stage the snapshot's restore sections, call `da_migrate(descriptor)` under budget,
    /// and validate it returned `Ready`. Any failure (init/`Incompatible`/`MigrateBudget`/crash) is
    /// an `Err`; the orchestrator rolls back and retries or leaves.
    ///
    /// # Errors
    /// A [`StepFailure`] describing the migrate/validate failure.
    fn migrate(&mut self, seam: &SnapshotSeam) -> Result<(), StepFailure>;

    /// **Step 6.** Activate locally, atomically: swap the instance binding to the already-committed
    /// transition and drain spooled frames into the new instance. No host advances the chain.
    ///
    /// # Errors
    /// A [`StepFailure`] if the atomic swap could not complete (treated as a local failure — roll
    /// back and retry/leave; the chain is never rolled back).
    fn activate(&mut self) -> Result<(), StepFailure>;

    /// **Step 7.** Roll back the *local* snapshot after a failed step before activation (restore the
    /// old-module instance from `seam`). Never touches the chain; never resumes the old epoch as the
    /// live one beyond this rollback-and-retry.
    fn rollback(&mut self, seam: &SnapshotSeam);

    /// Leave the run (the worker exits): the terminal action when the transaction fails closed or
    /// exhausts its retries. `reason` is for logging/telemetry.
    fn leave(&mut self, reason: &LeaveReason);
}

/// Whether `new` expands any grant beyond `old` (the fail-closed rule, architecture §5.4). Returns
/// `Some(reason)` naming the first expanded bound, or `None` if `new` is tighten-or-equal
/// everywhere.
///
/// Numeric bounds follow the ABI §2.3 convention where **`0` means "unbounded by this grant"** —
/// the loosest value. So tightening from unbounded (`0`) to any finite bound is fine; loosening a
/// finite bound to a larger one, or to unbounded (`0`), is expansion. The granted-artifact set must
/// be a subset of the old set.
#[must_use]
pub fn grant_expansion(old: &AdmittedQuotas, new: &AdmittedQuotas) -> Option<String> {
    // `0` = unbounded (loosest). `expands(old, new)`: new is looser than old.
    fn expands(old: u64, new: u64) -> bool {
        match (old, new) {
            (o, n) if o == n => false,
            (0, _) => false, // old already unbounded — nothing is looser
            (_, 0) => true,  // new unbounded, old finite — expansion
            (o, n) => n > o, // both finite — larger is looser
        }
    }
    let checks: [(&str, u64, u64); 11] = [
        ("max_frame_bytes", old.max_frame_bytes, new.max_frame_bytes),
        ("spool_frames", old.spool_frames, new.spool_frames),
        (
            "per_sender_quota",
            old.per_sender_quota,
            new.per_sender_quota,
        ),
        ("advisory_depth", old.advisory_depth, new.advisory_depth),
        ("payload_depth", old.payload_depth, new.payload_depth),
        ("gossip_depth", old.gossip_depth, new.gossip_depth),
        (
            "max_live_handles",
            old.max_live_handles,
            new.max_live_handles,
        ),
        ("max_live_bytes", old.max_live_bytes, new.max_live_bytes),
        (
            "max_readback_bytes",
            old.max_readback_bytes,
            new.max_readback_bytes,
        ),
        (
            "max_outstanding_ops",
            old.max_outstanding_ops,
            new.max_outstanding_ops,
        ),
        (
            "compute_queue_depth",
            old.compute_queue_depth,
            new.compute_queue_depth,
        ),
    ];
    for (name, o, n) in checks {
        if expands(o, n) {
            return Some(format!(
                "grant `{name}` expands from {o} to {n} (0 = unbounded)"
            ));
        }
    }
    // Artifacts: the new allow-list must be a subset of the old (no newly-reachable artifact).
    for h in &new.granted_artifacts {
        if !old.granted_artifacts.contains(h) {
            return Some(format!(
                "granted artifact {} is not in the previously-admitted set",
                h.to_hex()
            ));
        }
    }
    None
}

/// Run the **local** upgrade transaction against an already-committed target epoch (ABI §10.3).
///
/// `target` is the epoch descriptor the transition chain advanced to (the global commit already
/// happened — deliverable 1); `role` is the local role-instance's role; `new_grants_hash` is the
/// committed upgrade record's grants anchor ([`daemon_vhc_proto::UpgradeRecordBody::grants_hash`])
/// the owner-law re-check verifies the re-derived grants document against; `previous_grants` is the
/// grants the run instance is currently admitted under (the fail-closed comparison baseline);
/// `max_retries` bounds the rollback-and-retry cycles (refactor §9 crash drill).
///
/// This function performs **no** chain mutation: on every failure path the chain stays at `target`
/// and the old epoch is never resumed (§10.3 step 7).
pub fn run_local_upgrade(
    target: &EpochDescriptor,
    role: &str,
    new_grants_hash: Hash,
    previous_grants: &AdmittedQuotas,
    deadline_ms: u64,
    max_retries: u32,
    steps: &mut dyn UpgradeSteps,
) -> LocalUpgradeOutcome {
    let epoch = target.epoch;

    // The new module hash is a pure function of (run_id, epoch, role) via the chain (D1-EPOCH).
    let Some(new_module) = target.module_for(role) else {
        let reason = LeaveReason::UnknownRole(role.to_string());
        steps.leave(&reason);
        return LocalUpgradeOutcome::Left { epoch, reason };
    };

    // Step 1–2: quiesce + durable snapshot. A failed quiesce leaves the run; the chain has already
    // advanced, so the old epoch is not resumed.
    let seam = match steps.quiesce(deadline_ms) {
        Ok(seam) => seam,
        Err(e) => {
            let reason = LeaveReason::QuiesceFailed(e.detail);
            steps.leave(&reason);
            return LocalUpgradeOutcome::Left { epoch, reason };
        }
    };

    // Step 3: owner-law re-check. An owner/lane refusal leaves (rolling back the local snapshot).
    let new_grants = match steps.readmit(new_module, new_grants_hash) {
        Ok(q) => q,
        Err(e) => {
            steps.rollback(&seam);
            let reason = LeaveReason::OwnerRefused(e.detail);
            steps.leave(&reason);
            return LocalUpgradeOutcome::Left { epoch, reason };
        }
    };

    // Grant-expanding upgrades FAIL CLOSED — the worker exits rather than silently granting more.
    if let Some(why) = grant_expansion(previous_grants, &new_grants) {
        steps.rollback(&seam);
        let reason = LeaveReason::GrantExpansion(why);
        steps.leave(&reason);
        return LocalUpgradeOutcome::Left { epoch, reason };
    }

    // Steps 4–6 with rollback-and-retry (the mid-migration crash drill recovers here).
    let mut attempt: u32 = 0;
    loop {
        let migrated = steps.migrate(&seam).and_then(|()| steps.activate());
        match migrated {
            Ok(()) => {
                return LocalUpgradeOutcome::Activated {
                    epoch,
                    retries: attempt,
                }
            }
            Err(e) => {
                // A failed activation never resumes the old epoch and never rolls back the chain:
                // it rolls back the LOCAL snapshot and retries, or leaves.
                steps.rollback(&seam);
                if attempt >= max_retries {
                    let reason = LeaveReason::MigrateExhausted(e.detail);
                    steps.leave(&reason);
                    return LocalUpgradeOutcome::Left { epoch, reason };
                }
                attempt += 1;
            }
        }
    }
}

// ============================================================================================
// The production step adapter over the live host primitives (ABI §10.3).
// ============================================================================================

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    admit, start_run_migrating, DeviceProfile, EnvelopeRoleGrants, MemorySink, MigrationInput,
    OwnerPolicy, ParticipationLane, PumpHandle, Run, RunConfig, RunEnd, RunIdentity, SinkEntry,
    SnapshotCapture, SpooledFrame,
};
use daemon_vhc_host::Worker;
use daemon_vhc_proto::blake3_hash;
use daemon_vhc_sdk_consensus::checkpoint::SectionKind;

/// The **live activated instance** an upgrade produced (ABI §10.3 step 6): the running new-epoch
/// [`Run`], its [`PumpHandle`], and the **continued** journal (the old incarnation's records up
/// to the fence, followed by this incarnation's — one gapless log, see [`MemorySink::continuing`]).
pub struct ActivatedInstance {
    /// The running new-epoch instance (join it to finish the run).
    pub run: Run,
    /// The pump driving the new instance (spooled frames have already drained into it).
    pub pump: PumpHandle,
    /// The continued run journal: `old prefix ++ this incarnation's records`.
    pub journal: Arc<Mutex<MemorySink>>,
}

/// The inputs that start a live upgrade transaction: the live OLD instance plus the NEW module's
/// admission bundle (ABI §10.3). Handed to [`LiveUpgradeSteps::new`]; the orchestrator
/// ([`run_local_upgrade`]) then drives the five steps over these.
pub struct LiveUpgradeInputs<'w> {
    /// The host worker (engine) both instances run under.
    pub worker: &'w Worker,
    /// The role-instance's role label (§8.1).
    pub role: String,
    /// The live OLD instance (consumed by the quiesce drain).
    pub old_run: Run,
    /// The OLD instance's pump (quiesce / snapshot capture / spool drain).
    pub old_pump: PumpHandle,
    /// The OLD instance's journal (its length at the fence is the [`SnapshotSeam::journal_cursor`],
    /// and its records seed the new incarnation's continuation).
    pub old_sink: Arc<Mutex<MemorySink>>,
    /// The OLD module hash (recorded in the snapshot manifest's `module` field).
    pub old_module: Hash,
    /// The NEW module bytes (hash-pinned target; re-admitted under owner law).
    pub new_wasm: Vec<u8>,
    /// The NEW instance's execution identity (epoch = the committed target epoch, §8.1).
    pub new_identity: RunIdentity,
    /// The NEW instance's per-run signing key seed (a fresh sender — §12.1/§12.2).
    pub new_signing_seed: [u8; 32],
    /// The NEW module's admitted grants document (must hash to the committed record's
    /// `grants_hash`).
    pub new_grants_bytes: Vec<u8>,
    /// The participation lane the owner-law re-check evaluates against (§9.4).
    pub lane: ParticipationLane,
    /// The device profile for the re-admission funnel.
    pub device: DeviceProfile,
    /// The owner policy (participation + caps) the re-admission enforces (fail-closed at step 3).
    pub owner: OwnerPolicy,
    /// The envelope-derived role grants the re-admission derives quotas from.
    pub envelope_grants: EnvelopeRoleGrants,
    /// Per-attempt migrate fuel (§10.2 "explicit bounded budget"): entry `i` bounds attempt `i`
    /// (`None` = the engine default). Production passes the migration-grant fuel; the crash-recovery
    /// path (refactor §9) can starve an early attempt and grant a later one.
    pub migrate_fuel: Vec<Option<u64>>,
    /// A frame delivered **during** the drain window (§4.4): the network-reader seat keeps
    /// servicing while the guest drains, so a frame arriving mid-drain spools and drains into the
    /// new instance at activation (§10.3 step 6). Production leaves this `None` (real frames arrive
    /// through [`PumpHandle::deliver_frame`]); a drill sets it to pin the drain-time spool.
    pub drain_window_frame: Option<(u64, [u8; 32], Vec<u8>)>,
}

/// The production [`UpgradeSteps`] adapter: each step wired onto the real host primitive the spec
/// names — [`PumpHandle::quiesce`] + [`PumpHandle::snapshot_capture`] (steps 1–2), [`admit`]
/// (step 3), [`start_run_migrating`] (steps 4–5), spooled-frame drain (step 6). It is the concrete
/// side-effecting half of the transaction whose ordering/invariants [`run_local_upgrade`] fixes.
///
/// The new incarnation's journal is a **continuation** of the old one, seeded at the fence's
/// journal cursor ([`MemorySink::continuing`]) rather than a fresh sink — so the run journal is one
/// gapless log across the module switch, replayable end to end (no replay gap, no double-delivery).
pub struct LiveUpgradeSteps<'w> {
    worker: &'w Worker,
    role: String,
    old_run: Option<Run>,
    old_pump: PumpHandle,
    old_sink: Arc<Mutex<MemorySink>>,
    old_module: Hash,
    new_wasm: Vec<u8>,
    new_identity: RunIdentity,
    new_signing_seed: [u8; 32],
    new_grants_bytes: Vec<u8>,
    lane: ParticipationLane,
    device: DeviceProfile,
    owner: OwnerPolicy,
    envelope_grants: EnvelopeRoleGrants,
    migrate_fuel: Vec<Option<u64>>,
    drain_window_frame: Option<(u64, [u8; 32], Vec<u8>)>,
    attempt: usize,
    // captured / derived state
    journal_prefix: Vec<SinkEntry>,
    capture: Option<SnapshotCapture>,
    spooled: Vec<SpooledFrame>,
    pending: Option<ActivatedInstance>,
    activated: Option<ActivatedInstance>,
    migrate_failures: Vec<String>,
    left: Option<String>,
}

impl<'w> LiveUpgradeSteps<'w> {
    /// Build the adapter from its inputs (all captured state starts empty).
    #[must_use]
    pub fn new(inputs: LiveUpgradeInputs<'w>) -> Self {
        Self {
            worker: inputs.worker,
            role: inputs.role,
            old_run: Some(inputs.old_run),
            old_pump: inputs.old_pump,
            old_sink: inputs.old_sink,
            old_module: inputs.old_module,
            new_wasm: inputs.new_wasm,
            new_identity: inputs.new_identity,
            new_signing_seed: inputs.new_signing_seed,
            new_grants_bytes: inputs.new_grants_bytes,
            lane: inputs.lane,
            device: inputs.device,
            owner: inputs.owner,
            envelope_grants: inputs.envelope_grants,
            migrate_fuel: inputs.migrate_fuel,
            drain_window_frame: inputs.drain_window_frame,
            attempt: 0,
            journal_prefix: Vec::new(),
            capture: None,
            spooled: Vec::new(),
            pending: None,
            activated: None,
            migrate_failures: Vec::new(),
            left: None,
        }
    }

    /// The role label this transaction upgrades.
    #[must_use]
    pub fn role(&self) -> &str {
        &self.role
    }

    /// The live activated instance (ABI §10.3 step 6), once the transaction has activated one.
    #[must_use]
    pub fn activated(&self) -> Option<&ActivatedInstance> {
        self.activated.as_ref()
    }

    /// Take the activated instance (to finish/stop it).
    pub fn take_activated(&mut self) -> Option<ActivatedInstance> {
        self.activated.take()
    }

    /// Whether an instance is currently pending activation (a migrated-but-not-activated instance).
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// The leave reason, if the transaction failed closed / exhausted its retries.
    #[must_use]
    pub fn left(&self) -> Option<&str> {
        self.left.as_deref()
    }

    /// How many migrate attempts ran (0 before any migrate; the crash-recovery drill increments).
    #[must_use]
    pub fn attempts(&self) -> usize {
        self.attempt
    }

    /// The per-attempt migrate/validate failure details (for logging / crash-drill assertions).
    #[must_use]
    pub fn migrate_failures(&self) -> &[String] {
        &self.migrate_failures
    }

    /// The continuation sink for a new incarnation: the old prefix (up to the fence) + this
    /// incarnation's records, with the publish-seq high-water marks reset (a fresh sender, §12.2).
    fn continuation_sink(&self) -> Arc<Mutex<MemorySink>> {
        Arc::new(Mutex::new(MemorySink::continuing(
            self.journal_prefix.clone(),
        )))
    }
}

/// Poll until `cond` holds or `timeout` elapses; returns whether it held.
fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

impl UpgradeSteps for LiveUpgradeSteps<'_> {
    fn quiesce(&mut self, deadline_ms: u64) -> Result<SnapshotSeam, StepFailure> {
        self.old_pump
            .quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, deadline_ms)
            .map_err(|e| StepFailure::new(format!("quiesce delivery: {e}")))?;
        // A frame arriving DURING the drain freezes — it spools, never reaching the draining
        // instance (§4.4), and drains into the new instance at activation (§10.3 step 6).
        if let Some((seq, sender, payload)) = self.drain_window_frame.take() {
            self.old_pump
                .deliver_frame(
                    daemon_vhc_abi::DEFAULT_CHANNEL_CONTROL_ID,
                    seq,
                    sender,
                    payload,
                    b"drain-window-signed-frame".to_vec(),
                )
                .map_err(|e| StepFailure::new(format!("drain-window frame: {e}")))?;
        }
        let run = self
            .old_run
            .take()
            .ok_or_else(|| StepFailure::new("old instance already consumed"))?;
        match run.wait() {
            Ok(RunEnd::Outcome(code))
                if u64::from(code) == u64::from(daemon_vhc_abi::OUTCOME_QUIESCE_READY) => {}
            other => {
                return Err(StepFailure::new(format!(
                    "drain did not quiesce: {other:?}"
                )))
            }
        }
        let capture = self
            .old_pump
            .snapshot_capture()
            .ok_or_else(|| StepFailure::new("no accepted snapshot in the drain"))?;
        self.spooled = self.old_pump.take_spooled_frames();
        // The journal cursor (§10.3 step 2): where the old incarnation's records end. The new
        // incarnation continues the SAME journal from here — this prefix seeds the continuation.
        let old_entries = self.old_sink.lock().expect("old sink").entries.clone();
        let journal_cursor = old_entries.len() as u64;
        self.journal_prefix = old_entries;
        // The labelled seam, in the checkpoint typed-manifest format (E1): the captured
        // module-state section(s), content-addressed, plus the journal cursor —
        // "checkpointing and migration are one discipline".
        let mut builder = CheckpointManifest::builder(
            Hash(self.new_identity.run_id),
            0, // the snapshot captures the OLD epoch's state, produced by the OLD module
            0,
            self.old_module,
            daemon_vhc_proto::StateDigest([0u8; 16]),
        );
        for (name, bytes) in &capture.sections {
            builder = builder.section(name.clone(), SectionKind::Module, 1, bytes);
        }
        let manifest = builder
            .section(
                "journal-position",
                SectionKind::JournalPosition,
                1,
                &journal_cursor.to_le_bytes(),
            )
            .build()
            .map_err(|e| StepFailure::new(format!("seam manifest: {e:?}")))?;
        self.capture = Some(capture);
        Ok(SnapshotSeam {
            manifest,
            journal_cursor,
        })
    }

    fn readmit(
        &mut self,
        new_module: Hash,
        new_grants_hash: Hash,
    ) -> Result<AdmittedQuotas, StepFailure> {
        // The committed record's grants anchor: the re-derived grants document must hash to it.
        if blake3_hash(&self.new_grants_bytes) != new_grants_hash {
            return Err(StepFailure::new(
                "re-derived grants do not match the committed record's grants_hash",
            ));
        }
        // Owner-law re-check (§10.3 step 3): the full admission funnel over the NEW module.
        let admission = admit(
            self.worker,
            &self.new_wasm,
            Some(new_module.as_bytes()),
            &[],
            &self.new_grants_bytes,
            &self.lane,
            &self.device,
            &self.owner,
            None,
            Some(&self.envelope_grants),
        )
        .map_err(|refusal| StepFailure::new(refusal.to_string()))?;
        admission
            .quotas
            .ok_or_else(|| StepFailure::new("envelope-grants admission yields quotas"))
    }

    fn migrate(&mut self, seam: &SnapshotSeam) -> Result<(), StepFailure> {
        let capture = self
            .capture
            .clone()
            .ok_or_else(|| StepFailure::new("quiesce did not capture a snapshot"))?;
        // The typed seam content-addresses exactly the captured module state (E1's format).
        if let Some(module_section) = seam.manifest.section(SectionKind::Module) {
            if let Some((_, first)) = capture.sections.first() {
                if module_section.hash != blake3_hash(first) {
                    return Err(StepFailure::new(
                        "seam module section does not content-address the captured snapshot",
                    ));
                }
            }
        }
        let fuel = self.migrate_fuel.get(self.attempt).copied().flatten();
        self.attempt += 1;
        // The new incarnation continues the SAME journal (seeded at the fence cursor), not a fresh
        // sink — the run journal is one gapless log across the switch (§10.3 step 6).
        let sink = self.continuation_sink();
        let cfg = RunConfig::new(
            self.new_identity.clone(),
            self.new_signing_seed,
            Vec::new(),
            self.new_grants_bytes.clone(),
        );
        let run = start_run_migrating(
            self.worker,
            &self.new_wasm,
            cfg,
            Box::new(sink.clone()),
            Some(MigrationInput {
                capture,
                restore: true,
                migrate_fuel: fuel,
            }),
        )
        .map_err(|e| StepFailure::new(format!("start_run_migrating: {e}")))?;
        let pump = run.pump.clone();
        // The migrated module announces the restored state as its first publish; a failed migrate
        // tears the instance down before da_run (§10.3 step 5).
        let ok = wait_until(Duration::from_secs(30), || {
            !pump.published().is_empty() || run.is_finished()
        });
        if !ok {
            return Err(StepFailure::new(
                "migrating instance neither published nor tore down within the budget",
            ));
        }
        if pump.published().is_empty() {
            let end = run.wait().map_err(|e| StepFailure::new(e.to_string()))?;
            let detail = format!("migrate step failed: {end:?}");
            self.migrate_failures.push(detail.clone());
            return Err(StepFailure::new(detail));
        }
        self.pending = Some(ActivatedInstance {
            run,
            pump,
            journal: sink,
        });
        Ok(())
    }

    fn activate(&mut self) -> Result<(), StepFailure> {
        let instance = self
            .pending
            .take()
            .ok_or_else(|| StepFailure::new("activate without a migrated instance"))?;
        // Spooled frames drain into the new instance (§10.3 step 6).
        for f in self.spooled.drain(..) {
            instance
                .pump
                .deliver_frame(
                    f.channel,
                    f.seq,
                    f.sender,
                    f.payload,
                    f.original_signed_frame,
                )
                .map_err(|e| StepFailure::new(format!("spool drain: {e}")))?;
        }
        self.activated = Some(instance);
        Ok(())
    }

    fn rollback(&mut self, _seam: &SnapshotSeam) {
        // A failed migrate already tore the new instance down (guest-thread-owned teardown); the
        // snapshot capture IS the recovery point — nothing else to restore locally.
        if let Some(instance) = self.pending.take() {
            let _ = instance.pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE);
            let _ = instance.run.wait();
        }
    }

    fn leave(&mut self, reason: &LeaveReason) {
        self.left = Some(reason.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::{peer_id, SigningKey, TransitionChain, UpgradeAuthority, UpgradeRecord};
    use std::collections::BTreeSet;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn hash(n: u8) -> Hash {
        Hash([n; 32])
    }

    // A grants baseline: everything unbounded-ish, no artifacts (tighten-only from here is safe).
    fn base_grants() -> AdmittedQuotas {
        AdmittedQuotas {
            max_frame_bytes: 1000,
            spool_frames: 100,
            per_sender_quota: 50,
            advisory_depth: 64,
            payload_depth: 64,
            gossip_depth: 64,
            max_live_handles: 64,
            max_live_bytes: 1 << 20,
            max_readback_bytes: 1 << 20,
            max_outstanding_ops: 16,
            compute_queue_depth: 1024,
            data_read_budget_bytes: 0,
            granted_artifacts: BTreeSet::new(),
        }
    }

    // Build a real epoch-1 descriptor from a real transition chain (composing deliverable 1).
    fn target_at_epoch_1(new_worker_module: Hash) -> (EpochDescriptor, TransitionChain) {
        use daemon_vhc_proto::envelope::{Access, DeviceMinimums};
        use daemon_vhc_proto::genesis::{
            GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
            TransportSelection, GENESIS_SCHEMA_MAJOR,
        };
        use std::collections::BTreeMap;

        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "worker-mod".to_string(),
            SnapshotArtifact {
                url: "r2://w".into(),
                blake3: hash(1),
                size: None,
            },
        );
        artifacts.insert(
            "coord-mod".to_string(),
            SnapshotArtifact {
                url: "r2://c".into(),
                blake3: hash(2),
                size: None,
            },
        );
        let mut roles = BTreeMap::new();
        for (name, module) in [("worker", "worker-mod"), ("coordinator", "coord-mod")] {
            roles.insert(
                name.to_string(),
                RoleEntry {
                    lane: if name == "coordinator" {
                        "coordinator"
                    } else {
                        "trainer"
                    }
                    .into(),
                    module: module.into(),
                    abi: "vhc@2".into(),
                    config: ciborium::value::Value::Map(vec![]),
                    grants: RoleGrants::default(),
                    device_min: DeviceMinimums::default(),
                },
            );
        }
        let genesis = GenesisEnvelope {
            run: RunSection {
                schema: GENESIS_SCHEMA_MAJOR,
                run_label: "up".into(),
                min_peers: 1,
                max_peers: 8,
                access: Access::Org,
            },
            roles,
            artifacts,
            corpus_manifest: None,
            authority: ciborium::value::Value::Map(vec![]),
            transport: TransportSelection::default(),
            identities: Identities {
                upgrade_authority: vec![peer_id(&key(1))],
                ..Default::default()
            },
        };
        let frozen = genesis.freeze(&key(200)).unwrap();
        let run_id = *frozen.run_id();
        let mut chain = TransitionChain::genesis(&genesis, run_id).unwrap();
        let auth = UpgradeAuthority::from_genesis(&genesis.identities).unwrap();
        let rec = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            new_worker_module,
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        chain.append(rec, &auth).unwrap();
        (chain.descriptor(), chain)
    }

    /// A scripted steps double: each step returns a queued result, and every call is recorded so a
    /// test can assert the exact ordering the transaction drove.
    struct ScriptedSteps {
        quiesce: Result<SnapshotSeam, StepFailure>,
        readmit: Result<AdmittedQuotas, StepFailure>,
        // `migrate`/`activate` outcomes per attempt (popped front to back); missing → Ok.
        migrate_results: Vec<Result<(), StepFailure>>,
        activate_results: Vec<Result<(), StepFailure>>,
        log: Vec<String>,
    }

    impl ScriptedSteps {
        fn happy() -> Self {
            // A minimal E1 typed manifest (the swapped-in seam format): one module-state section.
            let manifest = CheckpointManifest::builder(
                hash(1),
                1,
                0,
                hash(42),
                daemon_vhc_proto::StateDigest([0u8; 16]),
            )
            .section(
                "module",
                daemon_vhc_sdk_consensus::checkpoint::SectionKind::Module,
                1,
                b"state",
            )
            .build()
            .expect("drill manifest");
            Self {
                quiesce: Ok(SnapshotSeam {
                    manifest,
                    journal_cursor: 42,
                }),
                readmit: Ok(base_grants()),
                migrate_results: Vec::new(),
                activate_results: Vec::new(),
                log: Vec::new(),
            }
        }
    }

    impl UpgradeSteps for ScriptedSteps {
        fn quiesce(&mut self, _deadline_ms: u64) -> Result<SnapshotSeam, StepFailure> {
            self.log.push("quiesce".into());
            self.quiesce.clone()
        }
        fn readmit(
            &mut self,
            _new_module: Hash,
            _new_grants_hash: Hash,
        ) -> Result<AdmittedQuotas, StepFailure> {
            self.log.push("readmit".into());
            self.readmit.clone()
        }
        fn migrate(&mut self, _seam: &SnapshotSeam) -> Result<(), StepFailure> {
            self.log.push("migrate".into());
            if self.migrate_results.is_empty() {
                Ok(())
            } else {
                self.migrate_results.remove(0)
            }
        }
        fn activate(&mut self) -> Result<(), StepFailure> {
            self.log.push("activate".into());
            if self.activate_results.is_empty() {
                Ok(())
            } else {
                self.activate_results.remove(0)
            }
        }
        fn rollback(&mut self, _seam: &SnapshotSeam) {
            self.log.push("rollback".into());
        }
        fn leave(&mut self, _reason: &LeaveReason) {
            self.log.push("leave".into());
        }
    }

    #[test]
    fn happy_path_activates_new_epoch_without_touching_the_chain() {
        let (target, chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        assert_eq!(
            out,
            LocalUpgradeOutcome::Activated {
                epoch: 1,
                retries: 0
            }
        );
        // The orchestrator never advances/rolls back the chain (it advanced globally already).
        assert_eq!(chain.epoch(), 1);
        assert_eq!(chain.module_for("worker"), Some(hash(42)));
        assert_eq!(steps.log, ["quiesce", "readmit", "migrate", "activate"]);
    }

    #[test]
    fn grant_expanding_upgrade_fails_closed_and_leaves() {
        let (target, _chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        // The new module is admitted with a LOOSER frame bound — expansion.
        let mut expanded = base_grants();
        expanded.max_frame_bytes = 2000;
        let mut steps = ScriptedSteps::happy();
        steps.readmit = Ok(expanded);
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        match out {
            LocalUpgradeOutcome::Left {
                epoch,
                reason: LeaveReason::GrantExpansion(_),
            } => assert_eq!(epoch, 1),
            other => panic!("expected grant-expansion Left, got {other:?}"),
        }
        // Snapshot rolled back, worker left; migrate/activate never ran.
        assert_eq!(steps.log, ["quiesce", "readmit", "rollback", "leave"]);
    }

    #[test]
    fn grant_expansion_to_unbounded_is_refused() {
        let old = base_grants();
        let mut new = base_grants();
        new.max_outstanding_ops = 0; // unbounded — looser than the finite old bound
        assert!(grant_expansion(&old, &new).is_some());
    }

    #[test]
    fn grant_tightening_and_equal_is_allowed() {
        let old = base_grants();
        let mut tighter = base_grants();
        tighter.max_frame_bytes = 500; // tighter
        tighter.spool_frames = 100; // equal
        assert!(grant_expansion(&old, &tighter).is_none());
        // Tightening from unbounded to finite is allowed.
        let mut old_unbounded = base_grants();
        old_unbounded.max_live_bytes = 0;
        let mut now_finite = base_grants();
        now_finite.max_live_bytes = 1 << 20;
        assert!(grant_expansion(&old_unbounded, &now_finite).is_none());
    }

    #[test]
    fn new_artifact_grant_is_expansion() {
        let old = base_grants();
        let mut new = base_grants();
        new.granted_artifacts.insert(hash(7));
        assert!(grant_expansion(&old, &new).is_some());
    }

    #[test]
    fn quiesce_failure_leaves_without_resuming_old_epoch() {
        let (target, chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        steps.quiesce = Err(StepFailure::new("QuiesceDeadlineExceeded"));
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        match out {
            LocalUpgradeOutcome::Left {
                epoch,
                reason: LeaveReason::QuiesceFailed(_),
            } => assert_eq!(
                epoch, 1,
                "run stays at the committed epoch; old epoch not resumed"
            ),
            other => panic!("expected quiesce Left, got {other:?}"),
        }
        assert_eq!(chain.epoch(), 1);
        assert_eq!(steps.log, ["quiesce", "leave"]);
    }

    #[test]
    fn owner_refusal_rolls_back_and_leaves() {
        let (target, _chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        steps.readmit = Err(StepFailure::new("ClaimExceedsPolicy"));
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        assert!(matches!(
            out,
            LocalUpgradeOutcome::Left {
                epoch: 1,
                reason: LeaveReason::OwnerRefused(_)
            }
        ));
        assert_eq!(steps.log, ["quiesce", "readmit", "rollback", "leave"]);
    }

    #[test]
    fn mid_migration_crash_recovers_by_local_rollback_and_retry() {
        let (target, chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        // Attempt 0 crashes mid-migration; attempt 1 succeeds (retry).
        steps.migrate_results = vec![Err(StepFailure::new("guest trapped mid-migration"))];
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        assert_eq!(
            out,
            LocalUpgradeOutcome::Activated {
                epoch: 1,
                retries: 1
            }
        );
        // The chain never rolled back; the second attempt activated the SAME (new) epoch.
        assert_eq!(chain.epoch(), 1);
        assert_eq!(
            steps.log,
            ["quiesce", "readmit", "migrate", "rollback", "migrate", "activate"]
        );
    }

    #[test]
    fn migrate_exhausts_retries_then_leaves_without_chain_rollback() {
        let (target, chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        // Every attempt fails; retry budget 1 → two attempts, then leave.
        steps.migrate_results = vec![
            Err(StepFailure::new("Incompatible")),
            Err(StepFailure::new("Incompatible")),
        ];
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 1, &mut steps);
        assert!(matches!(
            out,
            LocalUpgradeOutcome::Left {
                epoch: 1,
                reason: LeaveReason::MigrateExhausted(_)
            }
        ));
        assert_eq!(
            chain.epoch(),
            1,
            "a failed activation never rolls back the chain"
        );
        assert_eq!(
            steps.log,
            ["quiesce", "readmit", "migrate", "rollback", "migrate", "rollback", "leave"]
        );
    }

    #[test]
    fn failed_activation_rolls_back_and_retries() {
        let (target, _chain) = target_at_epoch_1(hash(42));
        let prev = base_grants();
        let mut steps = ScriptedSteps::happy();
        // migrate ok, activate fails once, then retry: migrate ok, activate ok.
        steps.activate_results = vec![Err(StepFailure::new("swap raced"))];
        let out = run_local_upgrade(&target, "worker", hash(51), &prev, 5000, 2, &mut steps);
        assert_eq!(
            out,
            LocalUpgradeOutcome::Activated {
                epoch: 1,
                retries: 1
            }
        );
        assert_eq!(
            steps.log,
            ["quiesce", "readmit", "migrate", "activate", "rollback", "migrate", "activate"]
        );
    }
}
