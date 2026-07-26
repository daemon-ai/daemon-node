// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Run configuration + typed outcomes: the frozen execution identity, the admitted per-run
//! bounds the driver enforces, the embedder-facing verdict/outcome enums, and the snapshot /
//! migration types the upgrade transaction carries (ABI §8.1, §4.7, §10.2–§10.3).

use crate::run::journal::SinkError;
use crate::trap::Trap;

/// The frozen execution-identity five-tuple (ABI §8.1) as the driver consumes it.
#[derive(Debug, Clone)]
pub struct RunIdentity {
    /// The 32-byte genesis/frozen-envelope hash.
    pub run_id: [u8; 32],
    /// The transition-chain position.
    pub epoch: u64,
    /// The envelope-level role label.
    pub role: String,
    /// The never-reused monotonic role-instance incarnation id.
    pub instance: u64,
    /// The pinned module blob hash.
    pub module: [u8; 32],
}

/// Configuration for one v2 run instance.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The execution identity (journal + signing scope, ABI §8.1/§12).
    pub identity: RunIdentity,
    /// The per-run software signing key seed (certified key chains arrive at D1; the §12.1
    /// envelope fields are final now).
    pub signing_seed: [u8; 32],
    /// The admitted config bytes (`da_init` receives byte-identical copies, §9.4 step 11).
    pub config: Vec<u8>,
    /// The admitted grants bytes.
    pub grants: Vec<u8>,
    /// Run-header fields the admission path pinned (verbatim canonical bytes; §8.3 tag 0). Empty
    /// until the A2 admission funnel wires them — recorded as such.
    pub manifest_bytes: Vec<u8>,
    /// Run-header claim bytes (see [`RunConfig::manifest_bytes`]). Empty at the certification minor,
    /// which records the plan, the composed claims and the grant instead.
    pub claim_bytes: Vec<u8>,
    /// The composed role Physical Claim's canonical bytes — a certification-minor run-header member.
    ///
    /// Empty below the certification minor, where [`RunConfig::claim_bytes`] carries the module's
    /// declared claim instead. The run header records one branch or the other, never both.
    pub physical_claim_bytes: Vec<u8>,
    /// The node/device aggregate claim's canonical bytes (see [`RunConfig::physical_claim_bytes`]).
    ///
    /// Distinct from the per-instance claim: a role colocated with another shares device resources, and
    /// the aggregate is what the node actually reserved.
    pub aggregate_claim_bytes: Vec<u8>,
    /// The negotiated major-2 ABI minor for this instance.
    ///
    /// It is the selector for the minor-dependent behaviour: which assessment export ran, which
    /// run-header variant the journal writes, and which terminal-context renderer applies. Defaults
    /// to the host's own implemented minor; admission overwrites it with the module's declared one.
    pub abi_minor: u32,
    /// The canonical Logical Resource Plan bytes (run-header, certification minor). Empty below it.
    pub resource_plan_bytes: Vec<u8>,
    /// The canonical Execution Grant bytes to apply before `da_init`.
    ///
    /// Empty means there is no grant to apply — a lower-minor module, which has no grant seam at
    /// all. A certification-minor run carries the exact bytes: copied verbatim from the signed role
    /// entry for a uniform run, or host-derived and bound to the role instance and incarnation for
    /// per-participant selection. The span the guest sees is **borrowed** and is reclaimed with the
    /// instance; neither side frees it.
    pub execution_grant: Vec<u8>,
    /// Run-header channel-table bytes (the Phase-A default table until D0).
    pub channels_bytes: Vec<u8>,
    /// Run-header device-profile bytes.
    pub device_bytes: Vec<u8>,
    /// Per-frame byte ceiling on `publish` (lane-profile-supplied at admission; a default here).
    pub max_frame_bytes: u32,
    /// Per-slice ceiling on bytes `read_back` may write into linear memory (§5.5).
    pub max_readback_bytes_per_slice: u64,
    /// Bounded advisory `Timer` queue depth (manifest-declared `event-caps` once the funnel
    /// wiring lands; a default here). Overflow is latest-wins on the queue: the oldest queued
    /// `Timer` drops, journaled (§4.7).
    pub advisory_depth: usize,
    /// Bounded advisory `PayloadReady` queue depth (§2.3 `event-caps`; §4.7 class rule
    /// dedup-by-hash). Overflow beyond distinct hashes drops the oldest announcement, journaled.
    pub payload_depth: usize,
    /// Bounded advisory gossip-class queue depth (§4.7 drop-oldest, journaled).
    pub gossip_depth: usize,
    /// Authoritative bounded spool: max undelivered authoritative frames (§4.7/§6.2
    /// `spool_frames`). Overflow BACK-PRESSURES the deliverer (never drops); hitting the bound
    /// journals the typed `SpoolExhausted` run condition (§6.7) once per exhaustion episode.
    pub spool_frames: usize,
    /// Authoritative per-sender outstanding quota (§4.7/§6.2 `per_sender_quota`): a single
    /// sender cannot use the reliable class as a memory-DoS vector. Overflow back-pressures
    /// that sender only.
    pub per_sender_quota: usize,
    /// The claim's **host-tier staging cap** in raw bytes (`0` = uncapped): the ceiling on
    /// guest-authored host-side bytes this instance may hold — staged sections (`stage_state`) and
    /// sealed buffers (`create_from`, `buffer_append`). Breach is the typed attributable
    /// `BudgetMemory` trap — the under-claim acceptance (refactor §5 A2).
    ///
    /// Sourced from the evaluated claim's `declared_peak.host` (ABI §9.1), not its hard tier: for a
    /// wasm instance the HARD tier is the guest's linear-memory ceiling — the resource the pooling
    /// allocator meters exactly, and what [`crate::EngineConfig::with_claimed_memory`] enforces —
    /// while a container the module stages host-side is precisely the "transient host scratch above
    /// the state floor" the peak tier accounts. Two exactly-metered resources, two tiers of one
    /// declaration.
    pub hard_accountable_host_bytes: u64,
    /// The **admitted artifact set** (track B2, architecture §3.2 `data@`): the blake3 hashes of
    /// the envelope's committed artifact map, intersected with the role's artifact grants ("which
    /// artifacts a module may touch is a grant"). A `data.fetch` naming a hash outside this set
    /// traps `GrantViolation` — and because the set descends from the envelope's edge-pinned
    /// snapshot descriptors (§5.1), an unpinned source is unreachable by construction. Empty =
    /// no artifacts granted (fail closed).
    pub granted_artifacts: std::collections::BTreeSet<[u8; 32]>,
    /// The cumulative `data@2` **read budget** in raw bytes (`0` = unbounded by this grant):
    /// the total artifact bytes the module may `data.fetch` across the run instance, charged
    /// per call from the requested range (deterministic — a pure function of guest call order,
    /// like every grant refusal). Breach completes `Err(GrantExhausted)`, never a trap and
    /// never silent truncation. Derived from the admitted grants (`AdmittedQuotas`).
    pub data_read_budget_bytes: u64,
    /// `buffer-req.max_live_handles` (ABI §2.3): the standing live-buffer handle ceiling
    /// (`0` = unbounded by this grant). Breach traps `BudgetHandles` (§7.3).
    pub max_live_buffer_handles: u64,
    /// `buffer-req.max_live_bytes` (ABI §2.3): the standing live-buffer byte ceiling — track B1's
    /// buffer quota (`0` = unbounded by this grant). Breach traps `BudgetMemory`.
    pub max_live_buffer_bytes: u64,
    /// `grant-bound.max_outstanding` (ABI §2.3): the concurrent-operation ceiling for the async
    /// completion protocol (`0` = unbounded by this grant). Breach traps `GrantViolation`.
    pub max_outstanding_ops: u64,
    // ---- compute@2 (track C1, ABI §15; architecture §3.3) — CLEARLY DELIMITED for the D0 union
    // merge: D0's envelope-derived AdmittedQuotas should tighten these exactly like the Phase-B
    // bound fields above (defaults here; grants derivation at admission). ---------------------
    /// The **queue-depth grant** (architecture §3.3 "a queue-depth grant bounds outstanding
    /// device work"): the maximum ops enqueued on the compute command queue since the last
    /// fence (`0` = unbounded by this grant). Breach traps `GrantViolation` — the guest must
    /// fence (and handle `Event::Fence`) to reclaim depth.
    pub compute_queue_depth: u64,
    // ---- the det-state plane (ABI §12.14 [SF-4]/[SF-7]) ---------------------------------------
    /// The run-pinned `state_chunk_size` from the genesis state contract (ABI §12.14 [SF-5]).
    /// `0` = no state contract ⇒ the state plane is not provisioned and `state_open` traps
    /// typed. Set by the session/worker from `GenesisEnvelope.run.state_contract`.
    pub state_chunk_size: u64,
    /// `state-streams-max` (ABI §12.14 [SF-7]): concurrent open state write streams
    /// (`0` = unbounded by this grant). Breach traps `GrantViolation`.
    pub state_streams_max: u64,
    /// `state-write-budget.max_bytes` ([SF-7]): the per-emit byte ceiling (`0` = unbounded).
    /// Breach traps `GrantViolation` (writes are guest-driven, so attributable).
    pub state_emit_max_bytes: u64,
    /// `state-write-budget.rate_per_min` ([SF-7]): the write token-bucket rate in raw bytes per
    /// minute (`0` = unbounded). Live-pump enforcement only (logical pump time) — replay is not
    /// the budget gate, the epoch-watchdog posture.
    pub state_write_rate_per_min: u64,
    /// `state-store-bytes` ([SF-7]): the live retained-byte ceiling across sealed families
    /// (`0` = unbounded). Enforced at `state_seal` after retention eviction; a seal that would
    /// still exceed it is refused typed and rolled back.
    pub state_store_bytes: u64,
    /// `state_retain_roots` ([SF-7], design §8.2): sealed roots retained per family beyond the
    /// pinned set (`0` = unbounded retention). Default
    /// [`daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT`] — the current round base and the
    /// freshly sealed round.
    pub state_retain_roots: u64,
    /// The per-instance **state-chunk spill directory** (design §8.1): when `Some`, the host
    /// state store spills canonical det-lane chunk BYTES here (`<dir>/<blake3-hex>`) and keeps
    /// only the index (lengths/refcounts/seal order) resident, so the retained roots live on disk
    /// instead of RAM — mandatory at the ceremony tier where the retained footprint would overrun
    /// the memory floor peer's unified budget. `None` ⇒ the resident (RAM) backing (the
    /// acceptance tier + tests). Set by the worker from the durable run-state home.
    pub state_dir: Option<std::path::PathBuf>,
    // ---- migration grant (Phase E, ABI §2.6 `migration-grant` / §10.2) ------------------------
    /// `migration-grant.max_sections`: the max sections one `snapshot_state` manifest may declare
    /// (`0` = unbounded by this grant). Exceeding it returns `SNAPSHOT_STATE_GRANT_EXCEEDED`.
    pub migration_max_sections: u64,
    /// `migration-grant.max_section_bytes`: the per-section byte ceiling (`0` = unbounded).
    /// Exceeding it returns `SNAPSHOT_STATE_GRANT_EXCEEDED`.
    pub migration_max_section_bytes: u64,
    /// **Deferred-device-fault injection (test/testkit seam — the simulated-providers pattern).**
    /// `Some(n)`: a synthetic device fault is latched after the `n`-th accepted `submit_op`
    /// (0-based: `n = 0` faults the first op), surfacing at the next fence (typed
    /// `ComputeFault` trap) or export (`COMP_ERR_DEVICE` completion) — the CUDA/wgpu
    /// deferred-error *timing* shape, exercised without a GPU (the ndarray tier is synchronous;
    /// real async faults ride the same `ComputeRunner` latch). `None` in production.
    pub compute_fault_after_ops: Option<u64>,
}

impl RunConfig {
    /// A config with Phase-A defaults for the bound fields.
    #[must_use]
    pub fn new(
        identity: RunIdentity,
        signing_seed: [u8; 32],
        config: Vec<u8>,
        grants: Vec<u8>,
    ) -> Self {
        Self {
            identity,
            signing_seed,
            config,
            grants,
            manifest_bytes: Vec::new(),
            claim_bytes: Vec::new(),
            physical_claim_bytes: Vec::new(),
            aggregate_claim_bytes: Vec::new(),
            // A directly-constructed run has negotiated nothing, so it must not claim the newest
            // contract. Defaulting to the host's own minor made this field track the constant: when
            // the certification minor landed, every run built without admission silently began
            // claiming it — which selects the certification run-header variant for a run that
            // composed nothing. The default is now the highest legacy minor, and admission overwrites
            // it with what the module actually declared.
            abi_minor: daemon_vhc_abi::LEGACY_CONTEXT_MAX_MINOR,
            resource_plan_bytes: Vec::new(),
            execution_grant: Vec::new(),
            channels_bytes: Vec::new(),
            device_bytes: Vec::new(),
            max_frame_bytes: 1 << 20,
            max_readback_bytes_per_slice: 1 << 20,
            advisory_depth: 64,
            payload_depth: 64,
            gossip_depth: 64,
            spool_frames: 256,
            per_sender_quota: 64,
            hard_accountable_host_bytes: 0,
            granted_artifacts: std::collections::BTreeSet::new(),
            data_read_budget_bytes: 0,
            max_live_buffer_handles: 64,
            max_live_buffer_bytes: 1 << 26,
            max_outstanding_ops: 16,
            compute_queue_depth: 1024,
            state_chunk_size: 0,
            state_streams_max: 0,
            state_emit_max_bytes: 0,
            state_write_rate_per_min: 0,
            state_store_bytes: 0,
            state_retain_roots: daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT,
            state_dir: None,
            migration_max_sections: 0,
            migration_max_section_bytes: 0,
            compute_fault_after_ops: None,
        }
    }
}

/// The host stage a device bring-up failure is attributed to.
///
/// Numbered past the five admission-funnel stages deliberately: bring-up happens *after* a module has
/// been admitted, when the instance is being stood up, and it is the first point at which a host-side
/// failure can occur with no guest phase to name. Reusing an admission stage would place the failure
/// in the funnel it had already cleared.
pub const HOST_STAGE_BACKEND_BRING_UP: u32 = 6;

/// Driver-level failures raised before/around guest execution (admission-shaped, not traps).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RunError {
    /// Engine/linker/instantiation plumbing failed.
    #[error("v2 sandbox error: {0}")]
    Sandbox(String),
    /// The module imports the retired `tabi@1` compute bridge — the typed `BridgeRetired`
    /// admission refusal, re-raised here so a caller that skipped the §1.3 front door still
    /// meets it before any guest code runs.
    #[error("BridgeRetired: {0}")]
    BridgeRetired(String),
    /// The admitted execution backend (`EngineConfig.backend`) cannot serve on this host right
    /// now (device absent/ineligible at run start, NVRTC not staged, or the process
    /// device-compute slot is occupied). A typed refusal — NEVER a silent ndarray fallback: the
    /// caller classifies it recoverable (the node reassesses; the device inventory changed).
    #[error("BackendUnavailable: {0}")]
    BackendUnavailable(String),
    /// The device backend could not be **brought up** for this instance: a host-side failure in
    /// which no guest code ran at all.
    ///
    /// It carries its own stage because it belongs to none of the guest execution contexts. That
    /// domain describes where *guest* code was executing, and a bring-up failure happens before the
    /// guest exists — so attributing it to initialization, which is what this used to do, was a
    /// classification bug rather than a conservative choice: it recorded a guest-trap fact about a
    /// phase the guest never entered, and a reader of that record would reasonably conclude the
    /// module's own initialization had failed.
    ///
    /// Inventing a twelfth guest context for it would have been the same mistake with more
    /// machinery. The honest shape is a typed host refusal outside the guest-trap surface entirely,
    /// and **no terminal guest-trap record is written**.
    #[error(
        "BackendBringUp (host stage {stage}): the {backend} lane could not be brought up for this \
         instance, before any guest code ran: {reason}"
    )]
    BackendBringUp {
        /// The host stage this failed at. See [`HOST_STAGE_BACKEND_BRING_UP`].
        stage: u32,
        /// The backend lane that failed to come up.
        backend: String,
        /// The captured reason, verbatim.
        reason: String,
    },
    /// A journal-sink write failed (journaling is load-bearing, §8.4).
    #[error(transparent)]
    Sink(#[from] SinkError),
}

/// The verdict on one authoritative-frame delivery (ABI §4.7): the reliable class NEVER drops —
/// overload back-pressures the network reader, which must hold the frame and retry. (The session
/// seam rewinds its dedup/gap cursor on back-pressure so the retry is not a duplicate.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliverVerdict {
    /// Enqueued for delivery.
    Accepted,
    /// The bounded spool is at `spool_frames` — hold + retry; the typed `SpoolExhausted` run
    /// condition (§6.7) was journaled at the episode's first refusal.
    SpoolFull,
    /// This sender is at `per_sender_quota` outstanding frames — hold + retry (the per-sender
    /// DoS bound; other senders are unaffected).
    SenderQuota,
    /// The frame exceeds the channel's `max_frame_bytes` — a protocol violation by the sender,
    /// refused outright (never enqueued, never retried).
    FrameTooLarge,
}

/// The embedder's answer to one serviced op request (see [`PumpHandle::complete_op`]).
#[derive(Debug)]
pub enum OpOutcome {
    /// A `payload_put` was durably stored; the pump computes the commitment hash itself.
    PutDone,
    /// A `payload_get` fetched these bytes; the pump hash-verifies before delivery.
    GetDone {
        /// The fetched bytes (verified against the op's requested hash by the pump).
        bytes: Vec<u8>,
    },
    /// A `data.fetch` serviced the WHOLE artifact (resolver + content cache); the pump verifies
    /// it against the op's committed hash, then slices the op's range before delivery (the
    /// sub-resource verification rule: a range is a slice OF hash-verified content).
    ///
    /// Also accepted for an [`OpRequest::ArtifactRange`] from an embedder that can only serve
    /// whole objects (an in-process content store): the pump extracts the covering span itself
    /// and chunk-verifies it — correctness is identical, only the transferred volume differs.
    FetchDone {
        /// The complete artifact bytes (verified + sliced by the pump, never the embedder).
        artifact: Vec<u8>,
    },
    /// An [`OpRequest::ArtifactRange`] serviced exactly the requested covering span. The pump
    /// verifies every covering chunk against the REGISTERED chunk hashes (the fold-committed
    /// map), then slices the guest's original range — the store stays untrusted; a lying span
    /// completes `Err(HashMismatch)` and the guest never sees the bytes.
    RangeDone {
        /// The covering-span bytes (`span_len` of them).
        bytes: Vec<u8>,
    },
    /// A `stream_open` connected; the pump mints the kind-9 handle with this receiver-granted
    /// initial writable credit (§3.3).
    OpenDone {
        /// Initial writable credit (bytes).
        credit: u64,
    },
    /// A standing `stream_accept` matched an incoming stream; the pump mints the handle.
    AcceptDone {
        /// Initial writable credit (bytes) for THIS side's writes on the accepted stream.
        credit: u64,
    },
    /// A `stream_write`'s bytes were accepted by the transport (unit completion, §3.4).
    WriteDone,
    /// A `stream_read` received these opaque bytes (journaled verbatim at completion — stream
    /// payloads are not content-addressed; ABI kind-4 record).
    ReadDone {
        /// The received bytes.
        bytes: Vec<u8>,
    },
    /// The operation failed (`COMP_ERR_*`, e.g. `NetUnreachable` for a connection failure —
    /// connection establishment/failure are completions of the op that needed the connection,
    /// architecture §3.3).
    Failed {
        /// The `comp-error` code (ABI §7.5).
        code: u64,
        /// A human-readable detail.
        detail: String,
    },
}

/// How a run ended (the guest-thread join result).
#[derive(Debug)]
pub enum RunEnd {
    /// `da_run` returned an Outcome code (ABI §4.5), journaled as terminal kind 0.
    Outcome(u32),
    /// `da_init` returned nonzero — journaled (tag 11), torn down, the join refused (§9.4 step 11).
    InitRefused(u32),
    /// `da_migrate` returned non-`Ready` (`Incompatible` or a module-defined detail ≥ 16) on a
    /// migrating instance (§10.2/§10.3 step 5): the validate step failed, the instance tore down
    /// without ever entering `da_run`, and the upgrade transaction rolls back (§10.3 step 7).
    MigrateRefused(u32),
    /// The guest trapped (typed, journaled as terminal kind 1); the subprocess survives (§7.6).
    Trapped(Trap),
    /// `da_apply_execution_grant` returned nonzero on the run instance, so `da_init` never ran.
    ///
    /// Carries the module's status verbatim. Deterministic and non-retryable for that
    /// `(module, plan, grant)` tuple: retrying needs changed admitted input, not a fresh instance.
    ExecutionGrantRejected(u32),
}

/// The accepted snapshot an upgrade transaction carries across the module switch (ABI §10.2/§10.3
/// step 2): the verbatim accepted state-manifest bytes plus its sections, in manifest order.
/// Captured by `snapshot_state` on the OLD instance (via [`PumpHandle::snapshot_capture`]);
/// consumed by [`MigrationInput`] on the NEW one.
///
/// Sections are the checkpoint-document v2 forms ([SF-6]): **inline** bytes for small state (the
/// round watermark), staged host-side; **by-reference** for already-sealed families
/// (master/ef/adamw) — zero section bytes moved, the new instance registers the fold ([SF-R2])
/// and streams it in `da_run`. A by-ref section's [`FamilyRef`](daemon_vhc_proto::det_state::FamilyRef)
/// is reconstructed host-side from the draining instance's own state store
/// ([`crate::run::state_store::StateStore::sealed_family_ref`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotCapture {
    /// The accepted state-manifest bytes, verbatim (journaled as tag 10).
    pub manifest: Vec<u8>,
    /// The checkpoint-document sections (inline or by-ref) in the manifest's declared order.
    pub sections: Vec<daemon_vhc_proto::det_state::CkptDocSection>,
}

/// One authoritative frame that spooled undelivered through a Quiesce drain (§4.4), as
/// [`PumpHandle::take_spooled_frames`] returns it — the exact [`PumpHandle::deliver_frame`]
/// argument set, so re-delivery into the new instance's pump is mechanical (§10.3 step 6).
#[derive(Debug, Clone)]
pub struct SpooledFrame {
    /// The channel the frame arrived on.
    pub channel: u32,
    /// The sender's durable channel-scoped sequence number.
    pub seq: u64,
    /// The sender identity.
    pub sender: [u8; 32],
    /// The module-authored payload bytes.
    pub payload: Vec<u8>,
    /// The complete original signed wire frame (tag-12 evidence).
    pub original_signed_frame: Vec<u8>,
}

/// The migration input to a NEW run instance (ABI §10.3 step 4): the old module's accepted
/// snapshot, the restore grant, and the migrate budget. Handing this to [`start_run_migrating`]
/// stages the sections host-side, journals the tag-13 instantiation as reason 2
/// (upgrade-activation), and calls `da_migrate(descriptor)` under budget between `da_init` and
/// `da_run`.
#[derive(Debug, Clone)]
pub struct MigrationInput {
    /// The accepted snapshot from the old instance's quiesce drain.
    pub capture: SnapshotCapture,
    /// The `migration-grant.restore` bit (ABI §2.6/§10.2): whether `read_back(kind = 3)` is
    /// granted during `da_migrate`. `false` fails closed (`GrantViolation`).
    pub restore: bool,
    /// The explicit migrate fuel budget (§10.2 "under an explicit bounded budget"); `None` uses
    /// the engine's per-call fuel. Exhaustion traps `MigrateBudget`.
    pub migrate_fuel: Option<u64>,
    /// Sealed det-state families carried from the DRAINING instance's store into this successor's
    /// store within the in-process live-module-switch transaction ([SF-6]). Empty for a
    /// content-plane late-join restore (whose chunks are fetched from the payload plane) — the
    /// switch is the one migrate where the same node retains custody of canonical state, so the
    /// successor serves these folds self-sealed ([SF-R1]) rather than fetching them from a plane
    /// the in-process switch never published to.
    pub carried_state: Vec<crate::run::state_store::CarriedFamily>,
}
