// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 event-loop driver (ABI §3–§6, §11–§12) — Phase A's closed capability subset.
//!
//! The inversion itself: the host calls `da_init` once, then `da_run` exactly once, and from then
//! on **drives nothing** — the guest pulls events through the blocking `next_event` import while
//! the host routes, journals, and enforces budgets (architecture §3.1).
//!
//! ## Threading (ABI §11)
//!
//! The guest runs on **one dedicated OS thread per role-instance** ([`start_run`] spawns it); that
//! thread owns the wasmtime `Store`, is the only thread that ever calls into wasm, and is the only
//! thread that drops the `Store` (§11.1/§11.3). The embedder (the session's async runtime; the
//! tier-1 tests) talks to the run through a [`PumpHandle`]: enqueue inbound frames, stage
//! payloads, deliver budget/stop/quiesce — a bounded, condvar-signalled queue the guest thread
//! blocks on inside `next_event` (§11.2). Timers need no external waker: the parked `next_event`
//! wait times out at the earliest armed deadline and fires due timers itself, inside the pump
//! lock, in deterministic `(fire_at, timer_id)` order.
//!
//! ## Born audited (ABI §8)
//!
//! Every observation flows through the [`JournalSink`] seam before the guest can see it: the
//! delivered event frame (tag 1, written before delivery — §8.4 rule 4) with the original signed
//! wire frame beside it (tag 12, §8.6), every `read_back` value (tag 2), every clock reading
//! (tag 3), every publish (tag 4, committed before `publish` returns — §6.2), timer arms/cancels
//! (tags 5/6), advisory drops (tag 7 via the sink's drop hook), instantiation + `da_init`
//! (tags 13/11), and the terminal fact (tag 9). There is no unjournaled mode.
//!
//! ## Budgets (ABI §5.5/§5.6)
//!
//! Fuel, the op count, and the readback-byte allowance reset at each `Delivered` return of
//! `next_event` (a `NeedCapacity` return resets nothing); the epoch deadline re-arms at the same
//! point, so a guest parked inside `next_event`/`read_back` is never epoch-killed for waiting —
//! the watchdog covers in-slice spins only, unchanged from v1.
//!
//! ## Deliberate Phase-A bounds (recorded)
//!
//! - The `tabi@1` compute bridge IS wired (the choreography sitting): the frozen dispatch is
//!   genericized over the store (`runtime::TabiHost`) and linked for any major-2 module that
//!   imports it, under the §2.5 legality rules (registration only in `da_init`; slice-class
//!   arenas cleared at each Delivered boundary; nr-class results journaled under §2.7 kinds).
//! - `snapshot_state` returns `SectionMissing` during a drain (no state-manifest verification yet
//!   — the §10.2 protocol lands with the migrate scaffolding); outside a drain it traps
//!   `PhaseViolation` per §6.6. `stage_state` is fully functional.
//! - Inbound frames arrive through [`PumpHandle::deliver_frame`] pre-verified: signature
//!   verification/dedup/gap detection are the session pump's admission-side jobs (the
//!   choreography sitting); this driver journals the original signed frame it is handed (§8.6).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ciborium::value::Value;
use wasmtime::{Caller, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder};

use daemon_vhc_abi::{
    pack_status_len, CHANNEL_DIR_RX_ONLY, COMP_ERR_GRANT_EXHAUSTED, COMP_ERR_HASH_MISMATCH,
    EV_TAG_FRAME, EV_TAG_STOP, FRAME_ENVELOPE_DOMAIN_V2, NS_COMPUTE_V2, NS_DATA_V2, NS_NET_V2,
    NS_SYS_V2, NS_TABI_V1, NS_VHC_V2, PHASE_A_DEFAULT_CHANNEL_TABLE, READBACK_KIND_STAGED_BYTES,
    RET_STATUS_DELIVERED, RET_STATUS_NEED_CAPACITY, SNAPSHOT_STATE_SECTION_MISSING,
    STAGED_KIND_BYTES,
};
use daemon_vhc_proto::{peer_id, sign_canonical, to_canonical_vec, SigningKey};

use crate::runtime::{EngineConfig, TabiHost, Worker};
use crate::trap::{Trap, TrapCode};
use crate::v2::buffer::BufferTable;
use crate::v2::completion::{CompError, CompletionResult, SuccessPayload};
use crate::v2::event::{encode_event_frame, EventV2, PayloadMeta};
use crate::v2::journal::{JournalSink, SinkError};
use crate::v2::ops::{OpRequest, OpTable};
use crate::v2::streams::StreamTable;

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
pub struct V2RunConfig {
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
    /// Run-header claim bytes (see [`V2RunConfig::manifest_bytes`]).
    pub claim_bytes: Vec<u8>,
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
    /// The claim's **hard-accountable host-tier cap** in raw bytes (`0` = uncapped): the
    /// enforceable tier the host meters EXACTLY (ABI §9.1). At Phase A the metered
    /// guest-attributable allocations are the staged bytes (`stage_state`); tensors/buffers join
    /// the meter with the bridge/Phase-B buffer layer. Breach is the typed attributable
    /// `BudgetMemory` trap — the under-claim acceptance (refactor §5 A2). The admission funnel
    /// (`v2::admission`) supplies this from the evaluated claim.
    pub hard_accountable_host_bytes: u64,
    /// The **admitted artifact set** (track B2, architecture §3.2 `data@`): the blake3 hashes of
    /// the envelope's committed artifact map, intersected with the role's artifact grants ("which
    /// artifacts a module may touch is a grant"). A `data.fetch` naming a hash outside this set
    /// traps `GrantViolation` — and because the set descends from the envelope's edge-pinned
    /// snapshot descriptors (§5.1), an unpinned source is unreachable by construction. Empty =
    /// no artifacts granted (fail closed).
    pub granted_artifacts: std::collections::BTreeSet<[u8; 32]>,
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
    /// **Deferred-device-fault injection (test/testkit seam — the simulated-providers pattern).**
    /// `Some(n)`: a synthetic device fault is latched after the `n`-th accepted `submit_op`
    /// (0-based: `n = 0` faults the first op), surfacing at the next fence (typed
    /// `ComputeFault` trap) or export (`COMP_ERR_DEVICE` completion) — the CUDA/wgpu
    /// deferred-error *timing* shape, exercised without a GPU (the ndarray tier is synchronous;
    /// real async faults ride the same `ComputeRunner` latch). `None` in production.
    pub compute_fault_after_ops: Option<u64>,
}

impl V2RunConfig {
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
            max_live_buffer_handles: 64,
            max_live_buffer_bytes: 1 << 26,
            max_outstanding_ops: 16,
            compute_queue_depth: 1024,
            compute_fault_after_ops: None,
        }
    }
}

/// Driver-level failures raised before/around guest execution (admission-shaped, not traps).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum V2Error {
    /// Engine/linker/instantiation plumbing failed.
    #[error("v2 sandbox error: {0}")]
    Sandbox(String),
    /// RETIRED at the choreography move (kept for wire/type stability this phase): the bridge is
    /// linked for every major-2 module that imports `tabi@1`; no path constructs this any more.
    #[error("tabi@1 bridge not wired into the v2 driver yet: {0}")]
    BridgeUnwired(String),
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
    FetchDone {
        /// The complete artifact bytes (verified + sliced by the pump, never the embedder).
        artifact: Vec<u8>,
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
    /// The guest trapped (typed, journaled as terminal kind 1); the subprocess survives (§7.6).
    Trapped(Trap),
}

// -- the pump (shared between the guest thread and the embedder) ----------------------------------

/// One queued, not-yet-delivered event: the decoded event plus its **frozen** frame encoding
/// (encoded once at enqueue, so a `NeedCapacity` retry sees byte-identical length/content) and,
/// for authoritative frames, the original signed wire frame for tag 12.
struct QueuedEvent {
    frame_bytes: Vec<u8>,
    tag: u64,
    /// `(channel, seq, sender, original signed frame)` for tag-12 evidence journaling.
    signed: Option<(u64, u64, [u8; 32], Vec<u8>)>,
    /// Advisory dedup key for `PayloadReady` (the staged hash).
    payload_hash: Option<[u8; 32]>,
    /// The timer id for a queued `Timer` event (depth accounting + cancel-of-queued, §4.7/§6.3).
    timer_id: Option<u64>,
    /// `Budget` marker (host-fixed depth 1, latest-wins).
    is_budget: bool,
    /// Advisory gossip-class identity `(channel, arrival seq, sender)` — present iff this is a
    /// gossip frame (drop-oldest accounting + the tag-7 drop identity, §4.7).
    gossip_id: Option<(u32, u64, [u8; 32])>,
}

struct ArmedTimer {
    id: u64,
    fire_at: u64,
}

struct PumpState {
    queue: VecDeque<QueuedEvent>,
    timers: Vec<ArmedTimer>,
    next_timer_id: u64,
    /// Host-announced staged payloads: `staging_id → (kind, bytes)`. Guest-created (`stage_state`)
    /// entries carry the §10.2 top bit.
    staged: std::collections::BTreeMap<u64, (u64, Vec<u8>)>,
    next_host_staging_id: u64,
    next_guest_staging_id: u64,
    sink: Box<dyn JournalSink>,
    /// The advisory `Timer`-queue depth (manifest-declared once the funnel lands; §4.7).
    timer_depth: usize,
    /// The advisory `PayloadReady`-queue depth (§4.7 dedup-by-hash class).
    payload_depth: usize,
    /// The advisory gossip-class queue depth (§4.7 drop-oldest).
    gossip_depth: usize,
    /// Authoritative spool bound (`spool_frames`, §4.7) + per-sender outstanding quota.
    spool_frames: usize,
    per_sender_quota: usize,
    /// Undelivered authoritative frames (the spool occupancy).
    auth_spooled: usize,
    /// Undelivered authoritative frames per sender (the quota ledger).
    auth_per_sender: std::collections::HashMap<[u8; 32], usize>,
    /// Whether the current spool-exhaustion episode was already journaled (§6.7, once per
    /// episode; cleared when the spool drains below the bound).
    spool_exhausted_reported: bool,
    /// Per-channel arrival counters for advisory (gossip-class) frames — advisory channels have
    /// no durable sequence semantics (§4.7); this dense arrival ordinal fills the frame's `seq`
    /// field deterministically for replay (the journaled delivered sequence is authoritative).
    gossip_arrivals: std::collections::HashMap<u32, u64>,
    /// Egress captured for the embedder (metrics/log are not journaled — outputs, not inputs).
    metrics: Vec<(String, f64)>,
    logs: Vec<(u32, String)>,
    /// Published frames, for embedder-side assertions: `(channel, seq, signed frame bytes)`.
    published: Vec<(u64, u64, Vec<u8>)>,
    /// The final bridge canonical state, exported by the guest thread before it drops the
    /// store (the parity-oracle hook: the digest input, §3.6 tier-3 comparisons).
    bridge_final_state: Option<Vec<u8>>,
    /// The per-instance buffer table (kind 8, architecture §3.4) — shared between the guest
    /// thread's imports and completion-arrival minting, behind this pump lock.
    buffers: BufferTable,
    /// The outstanding-op table (kind 10, ABI §7.5).
    ops: OpTable,
    /// The stream table (kind 9, §3.3 credit flow control).
    streams: StreamTable,
    /// Requests awaiting the embedder (the async-runtime bridge): `(op, request)` in issue order.
    op_requests: Vec<(u64, OpRequest)>,
    /// A `Stop` has been enqueued — no further deliveries will be accepted after it.
    stop_enqueued: bool,
    /// A `Quiesce` drain is open: Frame/PayloadReady/Timer deliveries are frozen (§4.4).
    draining: bool,
}

impl PumpState {
    /// Bound the advisory `PayloadReady` queue at its declared depth (§4.7 class 0): dedup by
    /// hash happens at staging; a distinct-hash announcement beyond the depth drops the OLDEST
    /// queued announcement — its staged bytes are unstaged with it — journaled (tag 7).
    fn enforce_payload_depth(&mut self) -> Result<(), SinkError> {
        if self.payload_depth == 0 {
            return Ok(());
        }
        let queued = self
            .queue
            .iter()
            .filter(|q| q.tag == daemon_vhc_abi::EV_TAG_PAYLOAD_READY)
            .count();
        if queued < self.payload_depth {
            return Ok(());
        }
        if let Some(pos) = self
            .queue
            .iter()
            .position(|q| q.tag == daemon_vhc_abi::EV_TAG_PAYLOAD_READY)
        {
            let old = self.queue.remove(pos).expect("position exists");
            // The frame names what was dropped (staging id + hash); unstage the bytes too.
            if let Ok(EventV2::PayloadReady {
                staging_id, hash, ..
            }) = crate::v2::event::decode_event_frame(&old.frame_bytes)
            {
                self.staged.remove(&staging_id);
                self.sink.drop_coalesced(
                    0,
                    daemon_vhc_abi::COALESCE_DEDUP_HASH,
                    crate::v2::journal::Dropped::payload(hash),
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue + journal one completion (ABI §4.6/§7.5): the tag-14 record captures the
    /// nondeterministic ARRIVAL (result + order) before the event is deliverable; the frame then
    /// rides the ordinary event queue (tag 1 at delivery, §8.4 rule 4). Completions still deliver
    /// during a `Quiesce` drain (§4.4 "already-outstanding operations") but never after `Stop`.
    fn enqueue_completion(&mut self, op: u64, result: &CompletionResult) -> Result<(), SinkError> {
        if self.stop_enqueued {
            return Ok(()); // the host delivers no further events after Stop (§4.4)
        }
        let result_bytes = result.encode().map_err(|e| SinkError(e.to_string()))?;
        self.sink.completion(op, &result_bytes)?;
        let frame_bytes = encode_event_frame(&EventV2::Completion {
            op,
            result: result.clone(),
        })
        .map_err(|e| SinkError(e.to_string()))?;
        self.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_COMPLETION,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        Ok(())
    }

    /// Enqueue one `Event::Fence(fence_id)` (ABI §4.6/§15, tag 5): the guest's compute-queue
    /// marker passed the device. Like completions, fences are precise markers — never coalesced,
    /// never dropped, still deliverable during a `Quiesce` drain (§4.4 "already-outstanding
    /// operations"), never after `Stop`. Journaled as the ordinary tag-1 delivered event; the
    /// fence *call* is deterministic guest output and needs no record of its own.
    fn enqueue_fence(&mut self, fence_id: u64) -> Result<(), SinkError> {
        if self.stop_enqueued {
            return Ok(()); // the host delivers no further events after Stop (§4.4)
        }
        let frame_bytes = encode_event_frame(&EventV2::Fence { fence_id })
            .map_err(|e| SinkError(e.to_string()))?;
        self.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_FENCE,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        Ok(())
    }
}

struct PumpShared {
    state: Mutex<PumpState>,
    wake: Condvar,
    /// Logical time zero = pump creation (≈ run join / journal open, §6.5).
    t0: Instant,
    /// Rig-controlled delivery hold (D2 back-pressure prerequisite; module docs on
    /// [`PumpHandle::hold`]). When set, the guest thread parks inside `next_event` and NO event is
    /// delivered — so the embedder can fill the authoritative spool to `SpoolFull`/`SenderQuota`
    /// deterministically, which a live guest (draining as fast as frames arrive) cannot force.
    hold: AtomicBool,
}

impl PumpShared {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.t0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// The embedder's handle onto a running v2 instance: enqueue inbound events, stage payloads,
/// read egress. Clonable; all methods are non-blocking (bounded queue overflows coalesce/drop
/// advisory events per §4.7, journaled).
#[derive(Clone)]
pub struct PumpHandle {
    shared: Arc<PumpShared>,
}

impl PumpHandle {
    /// Freeze event delivery to the guest (the D2 rig back-pressure control). While held, the guest
    /// thread parks inside `next_event` and no Frame/PayloadReady/Timer/… is delivered, so the
    /// embedder can `deliver_frame` past the authoritative spool bound and observe the typed
    /// [`DeliverVerdict::SpoolFull`] / [`DeliverVerdict::SenderQuota`] back-pressure deterministically
    /// (a live guest drains too fast to force it). Idempotent; pair with [`PumpHandle::release`].
    pub fn hold(&self) {
        self.shared.hold.store(true, Ordering::Relaxed);
    }

    /// Release a [`PumpHandle::hold`] and wake the guest to resume draining. Idempotent.
    pub fn release(&self) {
        self.shared.hold.store(false, Ordering::Relaxed);
        self.shared.wake.notify_all();
    }

    /// Deliver a **pre-verified** authoritative control frame (see module docs): `payload` is the
    /// module-authored bytes; `original_signed_frame` is the complete signed wire frame journaled
    /// as tag-12 evidence (§8.6).
    ///
    /// The reliable class is bounded but NEVER drops (§4.7): a [`DeliverVerdict::SpoolFull`] /
    /// [`DeliverVerdict::SenderQuota`] verdict back-pressures the caller, which MUST hold the
    /// frame and retry; genuine spool exhaustion journals the typed `SpoolExhausted` run
    /// condition (§6.7). Escalation policy (sustained exhaustion → `Stop{Fault}` and leave the
    /// run) is the embedder's, over these verdicts.
    pub fn deliver_frame(
        &self,
        channel: u32,
        seq: u64,
        sender: [u8; 32],
        payload: Vec<u8>,
        original_signed_frame: Vec<u8>,
    ) -> Result<DeliverVerdict, SinkError> {
        let ev = EventV2::Frame {
            channel,
            seq,
            sender,
            payload,
        };
        let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Err(SinkError("run is stopping; no further deliveries".into()));
        }
        if st.spool_frames != 0 && st.auth_spooled >= st.spool_frames {
            if !st.spool_exhausted_reported {
                st.spool_exhausted_reported = true;
                let detail = format!(
                    "authoritative spool at capacity ({} frames); back-pressuring",
                    st.spool_frames
                );
                st.sink.condition("SpoolExhausted", &detail)?;
            }
            return Ok(DeliverVerdict::SpoolFull);
        }
        if st.per_sender_quota != 0
            && st.auth_per_sender.get(&sender).copied().unwrap_or(0) >= st.per_sender_quota
        {
            return Ok(DeliverVerdict::SenderQuota);
        }
        st.auth_spooled += 1;
        *st.auth_per_sender.entry(sender).or_insert(0) += 1;
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: EV_TAG_FRAME,
            signed: Some((u64::from(channel), seq, sender, original_signed_frame)),
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(DeliverVerdict::Accepted)
    }

    /// Deliver an **advisory (gossip-class) frame** (§4.7): an unsequenced observation on an
    /// advisory channel. Bounded per-class queue at the declared depth with the fixed
    /// drop-oldest rule, every drop journaled (tag 7, class 2). The frame's `seq` is a dense
    /// per-channel arrival ordinal (advisory channels have no durable sequence semantics).
    pub fn deliver_gossip(
        &self,
        channel: u32,
        sender: [u8; 32],
        payload: Vec<u8>,
    ) -> Result<(), SinkError> {
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Err(SinkError("run is stopping; no further deliveries".into()));
        }
        if st.draining {
            // Advisory deliveries are frozen during a drain (§4.4); a gossip observation simply
            // coalesces away (journaled as a drop of the newest — nothing was ever queued).
            let seq = st.gossip_arrivals.get(&channel).copied().unwrap_or(0);
            st.sink.drop_coalesced(
                2,
                daemon_vhc_abi::COALESCE_DROP_OLDEST,
                crate::v2::journal::Dropped::gossip(u64::from(channel), sender, seq),
            )?;
            return Ok(());
        }
        let seq = {
            let c = st.gossip_arrivals.entry(channel).or_insert(0);
            let s = *c;
            *c += 1;
            s
        };
        let ev = EventV2::Frame {
            channel,
            seq,
            sender,
            payload,
        };
        let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
        // Drop-oldest at the declared depth (§4.7 gossip rule), journaled.
        let queued = st.queue.iter().filter(|q| q.gossip_id.is_some()).count();
        if st.gossip_depth != 0 && queued >= st.gossip_depth {
            if let Some(pos) = st.queue.iter().position(|q| q.gossip_id.is_some()) {
                let old = st.queue.remove(pos).expect("position exists");
                let (och, oseq, osender) = old.gossip_id.expect("gossip event has its id");
                st.sink.drop_coalesced(
                    2,
                    daemon_vhc_abi::COALESCE_DROP_OLDEST,
                    crate::v2::journal::Dropped::gossip(u64::from(och), osender, oseq),
                )?;
            }
        }
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: EV_TAG_FRAME,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: Some((channel, seq, sender)),
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Stage a content-addressed payload and announce it with a `PayloadReady` event (§4.3).
    /// Returns the staging id. Dedup-by-hash within the queue (§4.7), journaled as a drop.
    pub fn stage_payload(&self, bytes: Vec<u8>, channel: Option<u32>) -> Result<u64, SinkError> {
        let hash = *blake3::hash(&bytes).as_bytes();
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Err(SinkError("run is stopping; no further deliveries".into()));
        }
        // Advisory dedup by hash: an undelivered announcement for identical bytes coalesces.
        if st.queue.iter().any(|q| q.payload_hash == Some(hash)) {
            st.sink.drop_coalesced(
                0,
                daemon_vhc_abi::COALESCE_DEDUP_HASH,
                crate::v2::journal::Dropped::payload(hash),
            )?;
            // The staged bytes are already announced; find its id for the caller.
            if let Some((&id, _)) = st
                .staged
                .iter()
                .find(|(_, (k, b))| *k == STAGED_KIND_BYTES && blake3::hash(b).as_bytes() == &hash)
            {
                return Ok(id);
            }
        }
        st.enforce_payload_depth()?;
        let staging_id = st.next_host_staging_id;
        st.next_host_staging_id += 1;
        let size = bytes.len() as u64;
        st.staged.insert(staging_id, (STAGED_KIND_BYTES, bytes));
        let ev = EventV2::PayloadReady {
            staging_id,
            hash,
            meta: PayloadMeta {
                size,
                kind: STAGED_KIND_BYTES,
                channel,
            },
        };
        let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_PAYLOAD_READY,
            signed: None,
            payload_hash: Some(hash),
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(staging_id)
    }

    /// Stage a batch for the bridge (`read_back` kind 1, §2.5 rule 2): the tokens/shape encode as
    /// canonical CBOR `[sequences, seq_len, tokens-le-bytes]`, announced with `meta.kind = 1`.
    /// Identical batches may repeat, so no dedup applies (unlike kind-0 bytes).
    pub fn stage_batch(
        &self,
        tokens: &[u32],
        sequences: u32,
        seq_len: u32,
        channel: Option<u32>,
    ) -> Result<u64, SinkError> {
        let mut le = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            le.extend_from_slice(&t.to_le_bytes());
        }
        let v = Value::Array(vec![
            Value::from(sequences),
            Value::from(seq_len),
            Value::Bytes(le),
        ]);
        let bytes = to_canonical_vec(&v).map_err(|e| SinkError(format!("batch encode: {e}")))?;
        self.stage_kinded(bytes, daemon_vhc_abi::STAGED_KIND_BATCH, channel)
    }

    /// Stage an update-container payload for the bridge (`read_back` kind 2, §2.5 rule 2): the
    /// opaque committed payload wire bytes, announced with `meta.kind = 2`.
    pub fn stage_update(&self, payload: Vec<u8>, channel: Option<u32>) -> Result<u64, SinkError> {
        self.stage_kinded(
            payload,
            daemon_vhc_abi::STAGED_KIND_UPDATE_CONTAINER,
            channel,
        )
    }

    fn stage_kinded(
        &self,
        bytes: Vec<u8>,
        staged_kind: u64,
        channel: Option<u32>,
    ) -> Result<u64, SinkError> {
        let hash = *blake3::hash(&bytes).as_bytes();
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Err(SinkError("run is stopping; no further deliveries".into()));
        }
        st.enforce_payload_depth()?;
        let staging_id = st.next_host_staging_id;
        st.next_host_staging_id += 1;
        let size = bytes.len() as u64;
        st.staged.insert(staging_id, (staged_kind, bytes));
        let ev = EventV2::PayloadReady {
            staging_id,
            hash,
            meta: PayloadMeta {
                size,
                kind: staged_kind,
                channel,
            },
        };
        let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_PAYLOAD_READY,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(staging_id)
    }

    /// Deliver a `Budget` notification (host-fixed depth 1, latest-wins — §4.3/§4.7).
    pub fn budget(
        &self,
        fuel: u64,
        mem: u64,
        paused: bool,
        duty_pct: u64,
        vram_cap_bytes: u64,
    ) -> Result<(), SinkError> {
        let ev = EventV2::Budget {
            report: crate::v2::event::BudgetReport {
                fuel,
                mem,
                throttle: crate::v2::event::ThrottleReport {
                    paused,
                    duty_pct,
                    vram_cap_bytes,
                },
            },
        };
        let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Err(SinkError("run is stopping; no further deliveries".into()));
        }
        // Latest-wins at depth 1: replace any queued Budget, journaling the coalesce (§4.7).
        if let Some(pos) = st.queue.iter().position(|q| q.is_budget) {
            st.queue.remove(pos);
            st.sink.drop_coalesced(
                3,
                daemon_vhc_abi::COALESCE_LATEST_WINS,
                crate::v2::journal::Dropped::default(),
            )?;
        }
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_BUDGET,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: true,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Deliver the terminal `Stop{reason}` (§4.4). Queued behind already-pending events (the host
    /// never silently discards a consensus input); nothing may be enqueued after it.
    pub fn stop(&self, reason: u64) -> Result<(), SinkError> {
        let frame_bytes =
            encode_event_frame(&EventV2::Stop { reason }).map_err(|e| SinkError(e.to_string()))?;
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Ok(());
        }
        st.stop_enqueued = true;
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: EV_TAG_STOP,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Open a `Quiesce{reason, deadline_ms}` drain (§4.4): new Frame/PayloadReady/Timer
    /// deliveries freeze (spool/coalesce); the guest is expected to return `QuiesceReady`.
    pub fn quiesce(&self, reason: u64, deadline_ms: u64) -> Result<(), SinkError> {
        let frame_bytes = encode_event_frame(&EventV2::Quiesce {
            reason,
            deadline_ms,
        })
        .map_err(|e| SinkError(e.to_string()))?;
        let mut st = self.shared.state.lock().expect("pump lock");
        st.draining = true;
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_QUIESCE,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Drain the outstanding-op requests awaiting service (the async-runtime bridge, architecture
    /// §3.3: "all actual waiting lives in the host's async runtime"). Each request is handed out
    /// exactly once; the embedder answers through [`PumpHandle::complete_op`] whenever it likes —
    /// completion order is a nondeterministic input the journal captures (tag 14).
    #[must_use]
    pub fn take_op_requests(&self) -> Vec<(u64, OpRequest)> {
        std::mem::take(&mut self.shared.state.lock().expect("pump lock").op_requests)
    }

    /// Complete an outstanding op with the embedder's outcome. The pump — not the embedder — owns
    /// the trust steps (architecture §3.4): a put's hash is computed here over the op's own sealed
    /// bytes; a get's bytes are hash-verified against the requested hash BEFORE the completion is
    /// delivered (a mismatch completes `HashMismatch`, and the guest never sees the bytes);
    /// stream/buffer handles are minted here (never trusted from the embedder), and a stream
    /// read's opaque bytes are journaled verbatim at arrival (the ABI kind-4 record) — they are
    /// nondeterministic input that no content address can re-fetch. A completion for an op that
    /// is no longer outstanding (it was cancelled) is the raced-cancel no-op: the guest was
    /// already told `Cancelled`.
    ///
    /// Returns the handle the completion minted (`Some` for get/open/accept — the transport
    /// needs it to route subsequent writes/reads), else `None`.
    ///
    /// # Errors
    /// A journal-sink failure, or an outcome that contradicts the op's request shape (an embedder
    /// bug, surfaced loudly).
    pub fn complete_op(&self, op: u64, outcome: OpOutcome) -> Result<Option<u64>, SinkError> {
        let mut st = self.shared.state.lock().expect("pump lock");
        let Some(request) = st.ops.finish(op) else {
            return Ok(None); // cancelled while in service — Cancelled was already delivered
        };
        let mut minted = None;
        let result = match (request, outcome) {
            (OpRequest::PayloadPut { bytes }, OpOutcome::PutDone) => {
                CompletionResult::Ok(SuccessPayload::Hash(*blake3::hash(&bytes).as_bytes()))
            }
            (OpRequest::PayloadGet { hash }, OpOutcome::GetDone { bytes }) => {
                if blake3::hash(&bytes).as_bytes() == &hash {
                    match st.buffers.create_host(Arc::new(bytes)) {
                        Some(handle) => {
                            minted = Some(handle);
                            CompletionResult::Ok(SuccessPayload::Handle(handle))
                        }
                        None => CompletionResult::Err(CompError {
                            code: COMP_ERR_GRANT_EXHAUSTED,
                            detail: Some("buffer quota exhausted (deny new buffers)".into()),
                        }),
                    }
                } else {
                    CompletionResult::Err(CompError {
                        code: COMP_ERR_HASH_MISMATCH,
                        detail: Some("fetched bytes do not hash to the requested content".into()),
                    })
                }
            }
            (
                OpRequest::ArtifactFetch {
                    hash,
                    range_off,
                    range_len,
                },
                OpOutcome::FetchDone { artifact },
            ) => {
                // §3.4 "verification unchanged": the pump — not the embedder — verifies the
                // WHOLE artifact against the committed hash, then slices the range. A range is
                // thereby verified as a slice of hash-verified content (the sub-resource rule).
                if blake3::hash(&artifact).as_bytes() != &hash {
                    CompletionResult::Err(CompError {
                        code: COMP_ERR_HASH_MISMATCH,
                        detail: Some(
                            "fetched artifact does not hash to the committed value".into(),
                        ),
                    })
                } else {
                    let total = artifact.len() as u64;
                    let end = if range_len == 0 {
                        total
                    } else {
                        range_off.saturating_add(range_len)
                    };
                    if range_off > total || end > total {
                        CompletionResult::Err(CompError {
                            code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                            detail: Some(format!(
                                "range [{range_off}, {end}) out of bounds (artifact is {total} \
                                 bytes)"
                            )),
                        })
                    } else {
                        let slice = artifact[range_off as usize..end as usize].to_vec();
                        match st.buffers.create_host(Arc::new(slice)) {
                            Some(handle) => CompletionResult::Ok(SuccessPayload::Handle(handle)),
                            None => CompletionResult::Err(CompError {
                                code: COMP_ERR_GRANT_EXHAUSTED,
                                detail: Some("buffer quota exhausted (deny new buffers)".into()),
                            }),
                        }
                    }
                }
            }
            (OpRequest::StreamOpen { .. }, OpOutcome::OpenDone { credit }) => {
                let stream = st.streams.open(credit);
                minted = Some(stream);
                CompletionResult::Ok(SuccessPayload::Handle(stream))
            }
            (OpRequest::StreamAccept, OpOutcome::AcceptDone { credit }) => {
                let stream = st.streams.open(credit);
                minted = Some(stream);
                CompletionResult::Ok(SuccessPayload::Handle(stream))
            }
            (OpRequest::StreamWrite { .. }, OpOutcome::WriteDone) => {
                CompletionResult::Ok(SuccessPayload::Unit)
            }
            (OpRequest::StreamRead { .. }, OpOutcome::ReadDone { bytes }) => {
                // Opaque stream bytes are a nondeterministic input with NO content address:
                // journal them verbatim at arrival (the kind-4 record) so replay can
                // materialize the completion's buffer (§8.1/§8.7).
                st.sink.read_back(
                    op,
                    u64::from(daemon_vhc_abi::READBACK_KIND_STREAM_BYTES),
                    daemon_vhc_abi::RET_STATUS_DELIVERED,
                    &bytes,
                )?;
                match st.buffers.create_host(Arc::new(bytes)) {
                    Some(handle) => {
                        minted = Some(handle);
                        CompletionResult::Ok(SuccessPayload::Handle(handle))
                    }
                    None => CompletionResult::Err(CompError {
                        code: COMP_ERR_GRANT_EXHAUSTED,
                        detail: Some("buffer quota exhausted (deny new buffers)".into()),
                    }),
                }
            }
            (_, OpOutcome::Failed { code, detail }) => CompletionResult::Err(CompError {
                code,
                detail: Some(detail),
            }),
            (req, outcome) => {
                return Err(SinkError(format!(
                    "op outcome shape mismatch: request {req:?} answered with {outcome:?}"
                )))
            }
        };
        st.enqueue_completion(op, &result)?;
        drop(st);
        self.shared.wake.notify_all();
        Ok(minted)
    }

    /// Replenish a stream's writable credit (the receiver consumed bytes — transport-driven,
    /// §3.3). Held writes whose sizes now fit are released FIFO: their transport requests are
    /// emitted, and their completions will follow the embedder's service — the guest's credit
    /// signal IS those completions.
    pub fn grant_credit(&self, stream: u64, credit: u64) {
        let mut st = self.shared.state.lock().expect("pump lock");
        let released = st.streams.grant(stream, credit);
        for (op, bytes) in released {
            st.op_requests
                .push((op, OpRequest::StreamWrite { stream, bytes }));
        }
    }

    /// The bridge's final canonical state bytes (the digest input), exported by the guest thread
    /// before it dropped the store. `None` until the run ends, or when no bridge was linked.
    #[must_use]
    pub fn bridge_final_state(&self) -> Option<Vec<u8>> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .bridge_final_state
            .clone()
    }

    /// Signed frames the guest has published so far: `(channel, seq, signed frame bytes)`.
    #[must_use]
    pub fn published(&self) -> Vec<(u64, u64, Vec<u8>)> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .published
            .clone()
    }

    /// Metrics the guest emitted (egress; finite values only — §6.5).
    #[must_use]
    pub fn metrics(&self) -> Vec<(String, f64)> {
        self.shared.state.lock().expect("pump lock").metrics.clone()
    }

    /// Log lines the guest emitted (egress).
    #[must_use]
    pub fn logs(&self) -> Vec<(u32, String)> {
        self.shared.state.lock().expect("pump lock").logs.clone()
    }
}

// -- the guest-side store data --------------------------------------------------------------------

/// Per-slice budget/legality state (guest-thread-local — never behind the pump lock).
struct SliceState {
    /// Inside `da_init` (imports illegal, §6.6 rule 1).
    in_init: bool,
    /// `Stop` has been consumed (every import traps `PhaseViolation`, §4.4).
    stopped: bool,
    /// A `Quiesce` drain is open (freezes some behaviors, §4.4).
    draining: bool,
    /// The slice-constant logical `now()` (§6.5): the current slice's delivery timestamp.
    now: u64,
    /// Op-budget consumed this slice.
    op_calls: u64,
    /// Readback bytes consumed this slice.
    readback_bytes: u64,
    /// A pending mandatory `next_event` retry: the required capacity (§4.1).
    pending_next: Option<u64>,
    /// A pending mandatory `read_back` retry: `(src, kind, required)` (§6.4).
    pending_readback: Option<(u64, u32, u64)>,
    /// A pending mandatory `device_profile` retry: the required capacity (same §4.1/§6.4
    /// mandatory-retry discipline — the profile is delivered, and journaled, exactly once).
    pending_device: Option<u64>,
    /// The already-computed value behind a pending `read_back` retry: bridge kinds (1/2) mutate
    /// the tabi state exactly once (register batch / stage container), so the retry re-delivers
    /// the SAME value instead of re-registering (§6.4 "the staged value remains available").
    pending_readback_value: Option<Vec<u8>>,
}

/// The bridge staged-update ingest phase (§2.5/§5.9) — the state machine behind the v1 ingest
/// epilogue's placement (the B3-found catch-up defect):
///
/// v1 ran `snapshot_round_bases` (post-ingest master → next round's base) when
/// `da_ingest_updates` RETURNED — i.e. at the **ingest→training boundary**, before any later
/// math. The v2 bridge has no "ingest returns" moment, and deferring the epilogue to the
/// Delivered slice close diverges on the straggle **catch-up path**, where `BarrierRound`
/// ingests round r and trains round r+1 in ONE slice (round r+1 then trains against a
/// pre-ingest base). The equivalent boundary, driver-visible:
///
/// - a kind-2 read while `Idle` opens a FRESH staged window (v1's per-ingest `staged.clear()`);
/// - further kind-2 reads with no intervening non-window ops continue the SAME window (a
///   record's N entries are read back-to-back by `Staged::mint`);
/// - any other bridge op marks the window's ingest MATH as begun (aggregate + apply);
/// - the epilogue fires at the first boundary AFTER the math: the next kind-1 batch read
///   (training resumes — the catch-up path), a kind-2 read that OPENS THE NEXT window
///   (multi-round catch-up), or the slice close (the normal path, timing unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestPhase {
    /// No staged-update window is being consumed.
    Idle,
    /// A window is open; `math_seen` = a non-window bridge op ran since its last kind-2 read.
    Consuming {
        /// Whether ingest math has begun on this window.
        math_seen: bool,
    },
}

/// The wasmtime `Store` data for a v2 run instance.
struct V2Host {
    shared: Arc<PumpShared>,
    limits: StoreLimits,
    trap: Option<Trap>,
    slice: SliceState,
    // budgets (per-slice allowances)
    fuel_per_slice: u64,
    op_budget: u64,
    epoch_ticks: u64,
    max_readback_bytes: u64,
    max_frame_bytes: u32,
    // the claim's hard-accountable host-tier cap (standing, not per-slice — ABI §9.1/§5.5)
    hard_accountable_host_bytes: u64,
    accountable_staged_bytes: u64,
    // The §2.5 compute bridge: the SAME frozen tabi@1 dispatch state the v1 driver uses,
    // embedded (params/persistents/arenas/backend). None when the module imports no tabi@1.
    tabi: Option<crate::runtime::HostState>,
    // Whether a bridge slice pass is open (begin/end at Delivered boundaries, §2.5 rule 4).
    slice_pass_open: bool,
    // The bridge ingest phase (§5.9): tracks a staged-update window from its first kind-2
    // consumption to the boundary where the v1 ingest epilogue (post-ingest master → next
    // round's base) must run. See [`IngestPhase`].
    ingest_phase: IngestPhase,
    // Sealed-container watermark (B1 sealing-gap retirement): containers the bridge has built
    // that were already sealed + announced to the guest at a slice boundary.
    sealed_containers: usize,
    // signing (§12.1)
    signing: SigningKey,
    identity: RunIdentity,
    sender: [u8; 32],
    // sys@2 ambient inputs: the admitted device-profile bytes (nondeterministic input — journaled
    // tag 15 per delivery) and the identity-derived RNG seed (deterministic — never journaled).
    device_bytes: Vec<u8>,
    rng_seed: [u8; 32],
    // data@2: the admitted artifact set ("which artifacts a module may touch is a grant") — the
    // envelope's edge-pinned artifact map ∩ the role's grants. Fail closed when empty.
    granted_artifacts: std::collections::BTreeSet<[u8; 32]>,
    // compute@2 (track C1, ABI §15): the per-instance command-queue runner over the tier-1 real
    // backend, guest-thread-local like the bridge (device work belongs to the guest thread,
    // §11.1/§11.3 — the runner drops with the Store). `None` when the module imports no
    // compute@2 symbol. wgpu/cuda ride the same generic `ComputeRunner<B>` seam behind the host
    // feature lanes; driver-side backend selection is deferred with them.
    compute: Option<crate::compute::ComputeRunner<crate::compute::HostReal>>,
    // The queue-depth grant + its ledger: ops enqueued since the last successful fence.
    compute_queue_depth: u64,
    compute_ops_since_fence: u64,
    // The deferred-fault injection seam (see `V2RunConfig::compute_fault_after_ops`).
    compute_fault_after_ops: Option<u64>,
    compute_ops_total: u64,
}

impl V2Host {
    fn charge_op(&mut self, import: &'static str) -> Result<(), Trap> {
        self.slice.op_calls += 1;
        if self.slice.op_calls > self.op_budget {
            return Err(Trap::new(
                TrapCode::BudgetOps,
                import,
                None,
                "per-slice op budget exhausted",
            ));
        }
        Ok(())
    }

    /// The §6.6 temporal-legality gate + §4.1/§6.4 mandatory-retry enforcement, shared by every
    /// import. `is_next_event`/`is_read_back` let the two blocking imports pass their own retry.
    fn enter(&mut self, import: &'static str) -> Result<(), Trap> {
        if self.slice.stopped {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "import after Stop was consumed (§4.4)",
            ));
        }
        if self.slice.in_init {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "capability import during da_init (§6.6)",
            ));
        }
        if self.slice.pending_next.is_some() && import != "next_event" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from next_event requires an immediate retry (§4.1)",
            ));
        }
        if self.slice.pending_readback.is_some() && import != "read_back" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from read_back requires an immediate retry (§6.4)",
            ));
        }
        if self.slice.pending_device.is_some() && import != "device_profile" {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity from device_profile requires an immediate retry (§6.4)",
            ));
        }
        self.charge_op(import)
    }
}

// -- memory helpers (Caller<V2Host>) ---------------------------------------------------------------

fn mem_of(caller: &mut Caller<'_, V2Host>) -> Result<Memory, Trap> {
    caller
        .get_export("memory")
        .and_then(wasmtime::Extern::into_memory)
        .ok_or_else(|| Trap::bare(TrapCode::BadModule, "module has no exported memory"))
}

fn read_guest(caller: &mut Caller<'_, V2Host>, ptr: u32, len: u32) -> Result<Vec<u8>, Trap> {
    let mem = mem_of(caller)?;
    let (start, end) = (ptr as usize, ptr as usize + len as usize);
    mem.data(&caller)
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| Trap::bare(TrapCode::MemOob, "guest span out of bounds"))
}

fn write_guest(caller: &mut Caller<'_, V2Host>, ptr: u32, bytes: &[u8]) -> Result<(), Trap> {
    let mem = mem_of(caller)?;
    let start = ptr as usize;
    let data = mem.data_mut(caller);
    let end = start + bytes.len();
    data.get_mut(start..end)
        .ok_or_else(|| Trap::bare(TrapCode::MemOob, "guest span out of bounds"))?
        .copy_from_slice(bytes);
    Ok(())
}

fn stash<T>(caller: &mut Caller<'_, V2Host>, r: Result<T, Trap>) -> Result<T, wasmtime::Error> {
    r.map_err(|t| {
        let msg = t.to_string();
        caller.data_mut().trap = Some(t);
        wasmtime::Error::msg(msg)
    })
}

// -- the signed-frame envelope (§12.1) --------------------------------------------------------------

/// Build + sign the §12.1 domain-separated frame: `[envelope, payload, sig]` canonical CBOR, the
/// signature over the canonical envelope (which commits to the payload via `payload_hash`).
fn build_signed_frame(
    host: &V2Host,
    channel: u64,
    seq: u64,
    payload: &[u8],
) -> Result<Vec<u8>, Trap> {
    let payload_hash = blake3::hash(payload);
    let envelope = Value::Map(vec![
        (Value::from("domain"), Value::from(FRAME_ENVELOPE_DOMAIN_V2)),
        (
            Value::from("run_id"),
            Value::Bytes(host.identity.run_id.to_vec()),
        ),
        (Value::from("epoch"), Value::from(host.identity.epoch)),
        (
            Value::from("role"),
            Value::from(host.identity.role.as_str()),
        ),
        (Value::from("instance"), Value::from(host.identity.instance)),
        (
            Value::from("module"),
            Value::Bytes(host.identity.module.to_vec()),
        ),
        (Value::from("sender"), Value::Bytes(host.sender.to_vec())),
        (Value::from("channel"), Value::from(channel)),
        (Value::from("seq"), Value::from(seq)),
        (
            Value::from("payload_hash"),
            Value::Bytes(payload_hash.as_bytes().to_vec()),
        ),
    ]);
    let sig = sign_canonical(&host.signing, &envelope)
        .map_err(|e| Trap::bare(TrapCode::BadModule, format!("frame signing: {e}")))?;
    let frame = Value::Array(vec![
        envelope,
        Value::Bytes(payload.to_vec()),
        Value::Bytes(sig.0.to_vec()),
    ]);
    to_canonical_vec(&frame)
        .map_err(|e| Trap::bare(TrapCode::BadModule, format!("frame encoding: {e}")))
}

// -- sys@2 seeded deterministic randomness (architecture §3.2 "seeded randomness") ------------------

/// Derive the run-scoped RNG seed for one execution identity (`sys@2::rng_seed`): a **pure
/// function of the frozen §8.1 identity**, domain-separated under
/// [`daemon_vhc_abi::RNG_SEED_DOMAIN_V2`]. Deterministic per the §2.7 `dc` class — the import
/// carries **no journal record**; replay re-derives the identical seed from the run header's
/// identity (see [`super::replay`]). Two role-instances never share a seed; a trap-restart of the
/// same incarnation reproduces it (the seed is an *identity* property, not an *instantiation*
/// property — restarted policy must be able to re-derive its own randomness).
#[must_use]
pub fn derive_rng_seed(identity: &RunIdentity) -> [u8; 32] {
    // Unambiguous concatenation: fixed-width fields + a length prefix on the one variable field.
    let mut material = Vec::with_capacity(32 + 8 + 4 + identity.role.len() + 8 + 32);
    material.extend_from_slice(&identity.run_id);
    material.extend_from_slice(&identity.epoch.to_le_bytes());
    material.extend_from_slice(&(identity.role.len() as u32).to_le_bytes());
    material.extend_from_slice(identity.role.as_bytes());
    material.extend_from_slice(&identity.instance.to_le_bytes());
    material.extend_from_slice(&identity.module);
    blake3::derive_key(daemon_vhc_abi::RNG_SEED_DOMAIN_V2, &material)
}

// -- sys@2 crypto accelerations (the det/crypto-lane fast path, §3.2/§3.7) --------------------------

/// The host `sys@2::hash` acceleration body: blake3-256 over `data`, pinned by the dual-compiled
/// [`daemon_vhc_proto::crypto`] contract. Because the in-guest fallback is that *same* contract
/// compiled to wasm, host-op ≡ in-guest-op is bit-exact **by construction** (architecture §3.2, the
/// det-lane pattern). Exposed (crate-public) so the tier-1 conformance gate exercises the exact
/// body the live import runs. Deterministic → the import carries **no journal record** (§2.7 `dc`
/// class); replay re-executes it (see [`super::replay`]).
#[must_use]
pub fn host_crypto_hash(data: &[u8]) -> [u8; daemon_vhc_proto::HASH_LEN] {
    daemon_vhc_proto::crypto_hash(data)
}

/// The host `sys@2::verify_sig` acceleration body: the ABI status code of the tri-state
/// [`daemon_vhc_proto::VerifyOutcome`] (0 = valid, 1 = invalid, 2 = malformed). Same
/// dual-compiled-contract / by-construction-parity story as [`host_crypto_hash`]; deterministic,
/// not journaled.
#[must_use]
pub fn host_crypto_verify(public_key: &[u8], signature: &[u8], message: &[u8]) -> u32 {
    daemon_vhc_proto::verify_sig(public_key, signature, message).code()
}

// -- the v2 linker ----------------------------------------------------------------------------------

/// How long a parked `next_event` waits between wake checks when no timer bounds the wait.
const PARK_RECHECK: Duration = Duration::from_millis(50);

/// The pump shared state behind a caller (borrow helper for the import bodies).
fn shared_of(c: &Caller<'_, V2Host>) -> Arc<PumpShared> {
    c.data().shared.clone()
}

#[allow(clippy::too_many_lines)]
fn link_v2(linker: &mut Linker<V2Host>) -> Result<(), wasmtime::Error> {
    // ---- vhc@2::next_event — THE blocking pull (§4.1) -------------------------------------------
    linker.func_wrap(
        NS_VHC_V2,
        "next_event",
        |mut c: Caller<'_, V2Host>, buf_ptr: u32, buf_cap: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("next_event")?;
                // Mandatory-retry rule: after NeedCapacity the re-call must be big enough (§4.1).
                if let Some(required) = c.data().slice.pending_next {
                    if u64::from(buf_cap) < required {
                        return Err(Trap::new(
                            TrapCode::BadEvent,
                            "next_event",
                            None,
                            format!("retry with buf_cap {buf_cap} < required {required} (§4.1)"),
                        ));
                    }
                }
                // B1 (the sealing-gap retirement): a bridge container completed in the slice
                // that just ended is sealed HERE — at the slice boundary — and ANNOUNCED to the
                // guest as PayloadReady kind 0. Bridge-plane serialization stays a host service
                // until Phase C's compute export, but the guest now owns the commitment path:
                // read_back → create_from → payload_put → author the commitment over the
                // completion hash. Watermarked, so a NeedCapacity retry never double-stages.
                let sealed: Option<Vec<u8>> = {
                    let d = c.data_mut();
                    match d.tabi.as_ref() {
                        Some(tabi) => {
                            // GUEST-BUILT containers only: an inbound-staged container (read_back
                            // kind 2) must never be re-announced as the guest's own.
                            let n = tabi.guest_container_count();
                            if n > d.sealed_containers {
                                let bytes = tabi.seal_guest_container_of(n - 1);
                                d.sealed_containers = n;
                                bytes
                            } else {
                                None
                            }
                        }
                        None => None,
                    }
                };
                if let Some(bytes) = sealed {
                    let sh = c.data().shared.clone();
                    let mut st = sh.state.lock().expect("pump lock");
                    stage_own_bytes(&mut st, bytes)
                        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                }
                let shared = c.data().shared.clone();
                // Park until an event is deliverable, firing due timers ourselves (§6.3).
                let (frame, at) = {
                    let mut st = shared.state.lock().expect("pump lock");
                    loop {
                        // Rig delivery hold (D2 back-pressure prerequisite): freeze all delivery so
                        // the embedder can fill the spool to SpoolFull/SenderQuota deterministically.
                        if shared.hold.load(Ordering::Relaxed) {
                            let (guard, _timeout) = shared
                                .wake
                                .wait_timeout(st, PARK_RECHECK)
                                .expect("pump lock");
                            st = guard;
                            continue;
                        }
                        let now = shared.now_ms();
                        // Fire due timers (frozen during a drain, §4.4; and never behind a queued
                        // Stop — the host delivers no further events after Stop, §4.4).
                        if !st.draining && !st.stop_enqueued {
                            fire_due_timers(&mut st, now)?;
                        }
                        if let Some(front) = st.queue.front() {
                            let len = front.frame_bytes.len() as u64;
                            if len > u64::from(buf_cap) {
                                // Not consumed; no journal record; budgets do not reset (§4.1).
                                c.data_mut().slice.pending_next = Some(len);
                                return Ok(pack_status_len(RET_STATUS_NEED_CAPACITY, len as u32));
                            }
                            // Deliver: sample once, journal BEFORE the guest observes (§8.4 r4).
                            let at = shared.now_ms();
                            let ev = st.queue.pop_front().expect("front checked");
                            // Spool accounting (§4.7): a delivered authoritative frame frees its
                            // spool slot + its sender's quota; draining below the bound closes
                            // the exhaustion episode.
                            if let Some((_, _, sender, _)) = ev.signed {
                                st.auth_spooled = st.auth_spooled.saturating_sub(1);
                                if let Some(n) = st.auth_per_sender.get_mut(&sender) {
                                    *n = n.saturating_sub(1);
                                }
                                if st.auth_spooled < st.spool_frames {
                                    st.spool_exhausted_reported = false;
                                }
                            }
                            st.sink
                                .event(at, &ev.frame_bytes)
                                .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                            if let Some((ch, seq, sender, ref orig)) = ev.signed {
                                st.sink
                                    .signed_frame(ch, seq, sender, orig)
                                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                            }
                            break (ev, at);
                        }
                        // Nothing deliverable: park until a wake or the earliest timer deadline.
                        let wait = if st.draining {
                            PARK_RECHECK
                        } else {
                            st.timers
                                .iter()
                                .map(|t| t.fire_at.saturating_sub(now))
                                .min()
                                .map_or(PARK_RECHECK, |ms| {
                                    Duration::from_millis(ms.max(1)).min(PARK_RECHECK)
                                })
                        };
                        let (guard, _timeout) =
                            shared.wake.wait_timeout(st, wait).expect("pump lock");
                        st = guard;
                    }
                };
                // Copy into the guest buffer, then start the new slice (§5.5): budgets reset +
                // epoch re-arms on Delivered only.
                write_guest(c, buf_ptr, &frame.frame_bytes)?;
                let d = c.data_mut();
                // Bridge slice lifecycle (§2.5 rule 4 / §7.1 slice class): end the previous
                // slice's pass (clear step arenas wholesale, free tensors — the v1 finish_entry
                // teardown) and begin the new slice's differentiable pass.
                if let Some(tabi) = d.tabi.as_mut() {
                    // The §5.9 epilogue's slice-close backstop (the normal path, where the
                    // ingesting slice ends without further training — timing unchanged from
                    // A2). The catch-up boundaries fire earlier, in read_back (IngestPhase).
                    if d.ingest_phase != IngestPhase::Idle {
                        tabi.snapshot_round_bases();
                        d.ingest_phase = IngestPhase::Idle;
                    }
                    if d.slice_pass_open {
                        tabi.end_slice_pass_and_clear();
                    }
                    tabi.begin_slice_pass();
                    d.slice_pass_open = true;
                }
                d.slice.pending_next = None;
                d.slice.now = at;
                d.slice.op_calls = 0;
                d.slice.readback_bytes = 0;
                if frame.tag == EV_TAG_STOP {
                    d.slice.stopped = true;
                }
                if frame.tag == daemon_vhc_abi::EV_TAG_QUIESCE {
                    d.slice.draining = true;
                }
                let fuel = d.fuel_per_slice;
                let ticks = d.epoch_ticks;
                c.set_fuel(fuel)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                wasmtime::AsContextMut::as_context_mut(c).set_epoch_deadline(ticks);
                Ok(pack_status_len(
                    RET_STATUS_DELIVERED,
                    frame.frame_bytes.len() as u32,
                ))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- vhc@2::read_back — the explicit budgeted blocking readback (§6.4) ----------------------
    linker.func_wrap(
        NS_VHC_V2,
        "read_back",
        |mut c: Caller<'_, V2Host>,
         src: u64,
         kind: u32,
         out_ptr: u32,
         out_cap: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("read_back")?;
                if let Some((psrc, pkind, required)) = c.data().slice.pending_readback {
                    if psrc != src || pkind != kind {
                        return Err(Trap::new(
                            TrapCode::BadEvent,
                            "read_back",
                            None,
                            "retry must repeat the same (src, kind) (§6.4)",
                        ));
                    }
                    if u64::from(out_cap) < required {
                        return Err(Trap::new(
                            TrapCode::BadEvent,
                            "read_back",
                            None,
                            "retry with a still-too-small buffer (§6.4)",
                        ));
                    }
                }
                // kind → the staged-kind it consumes (ABI §6.4 table): 0 bytes, 1 batch (bridge),
                // 2 update container (bridge). State-section (3) arrives with migrate.
                let want_staged_kind = match kind {
                    READBACK_KIND_STAGED_BYTES => daemon_vhc_abi::STAGED_KIND_BYTES,
                    daemon_vhc_abi::READBACK_KIND_STAGED_BATCH => daemon_vhc_abi::STAGED_KIND_BATCH,
                    daemon_vhc_abi::READBACK_KIND_STAGED_UPDATE => {
                        daemon_vhc_abi::STAGED_KIND_UPDATE_CONTAINER
                    }
                    _ => {
                        return Err(Trap::new(
                            TrapCode::ReadBackUnavailable,
                            "read_back",
                            None,
                            format!("kind {kind} stages nothing in this Phase-A driver"),
                        ))
                    }
                };
                // A pending retry re-delivers the already-computed value (no re-mutation, §6.4).
                if let Some(v) = c.data().slice.pending_readback_value.clone() {
                    let len = v.len() as u64;
                    if len > u64::from(out_cap) {
                        return Ok(pack_status_len(RET_STATUS_NEED_CAPACITY, len as u32));
                    }
                    {
                        let d = c.data_mut();
                        d.slice.readback_bytes += len;
                        if d.slice.readback_bytes > d.max_readback_bytes {
                            return Err(Trap::new(
                                TrapCode::GrantViolation,
                                "read_back",
                                None,
                                "per-slice readback-byte allowance exhausted (§5.5)",
                            ));
                        }
                    }
                    {
                        let sh = shared_of(c);
                        let mut st = sh.state.lock().expect("pump lock");
                        st.sink
                            .read_back(src, u64::from(kind), RET_STATUS_DELIVERED, &v)
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                    }
                    write_guest(c, out_ptr, &v)?;
                    let d = c.data_mut();
                    d.slice.pending_readback = None;
                    d.slice.pending_readback_value = None;
                    return Ok(pack_status_len(RET_STATUS_DELIVERED, len as u32));
                }
                if want_staged_kind != daemon_vhc_abi::STAGED_KIND_BYTES && c.data().tabi.is_none()
                {
                    return Err(Trap::new(
                        TrapCode::ReadBackUnavailable,
                        "read_back",
                        None,
                        "bridge staging kinds need the tabi@1 bridge linked (§2.5)",
                    ));
                }
                let shared = c.data().shared.clone();
                let staged_bytes = {
                    let st = shared.state.lock().expect("pump lock");
                    match st.staged.get(&src) {
                        Some((k, bytes)) if *k == want_staged_kind => bytes.clone(),
                        _ => {
                            return Err(Trap::new(
                                TrapCode::ReadBackUnavailable,
                                "read_back",
                                None,
                                format!("staging id {src} names nothing stageable as kind {kind}"),
                            ))
                        }
                    }
                };
                // Bridge kinds resolve to a tiny CBOR uint (§6.4): kind 1 registers the batch and
                // yields its kind-7 handle; kind 2 stages the container and yields the upd index.
                // The staged entry is CONSUMED (a re-read would double-register — deterministic
                // refusal instead). Kind 0 delivers the bytes verbatim, re-readable.
                let value = match kind {
                    daemon_vhc_abi::READBACK_KIND_STAGED_BATCH => {
                        let v: ciborium::value::Value =
                            ciborium::de::from_reader(staged_bytes.as_slice()).map_err(|e| {
                                Trap::bare(TrapCode::BadModule, format!("staged batch: {e}"))
                            })?;
                        let ciborium::value::Value::Array(items) = v else {
                            return Err(Trap::bare(TrapCode::BadModule, "staged batch shape"));
                        };
                        let uint = |i: usize| -> u32 {
                            items
                                .get(i)
                                .and_then(ciborium::value::Value::as_integer)
                                .map(|n| u32::try_from(i128::from(n)).unwrap_or(0))
                                .unwrap_or(0)
                        };
                        let (sequences, seq_len) = (uint(0), uint(1));
                        let tokens: Vec<u32> = match items.get(2) {
                            Some(ciborium::value::Value::Bytes(b)) => b
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                            _ => return Err(Trap::bare(TrapCode::BadModule, "staged tokens")),
                        };
                        // A batch read is the ingest→TRAINING boundary (the catch-up path,
                        // §5.9 / IngestPhase docs): the epilogue fires BEFORE the batch
                        // registers, so training sees the post-ingest round base.
                        {
                            let d = c.data_mut();
                            if d.ingest_phase != IngestPhase::Idle {
                                d.tabi().snapshot_round_bases();
                                d.ingest_phase = IngestPhase::Idle;
                            }
                        }
                        let handle = c
                            .data_mut()
                            .tabi()
                            .stage_bridge_batch(tokens, sequences, seq_len);
                        let mut out = Vec::new();
                        ciborium::into_writer(&ciborium::value::Value::from(handle), &mut out)
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                        shared.state.lock().expect("pump lock").staged.remove(&src);
                        out
                    }
                    daemon_vhc_abi::READBACK_KIND_STAGED_UPDATE => {
                        // Window boundaries (IngestPhase docs): the first kind-2 read of a
                        // window — or one following the previous window's ingest MATH (the
                        // multi-round catch-up boundary, which also fires the epilogue) —
                        // opens a FRESH window (v1's per-ingest `staged.clear()`), so
                        // `upd_*@1` indices are 0-based per round.
                        let fresh_window = {
                            let d = c.data_mut();
                            match d.ingest_phase {
                                IngestPhase::Idle => true,
                                IngestPhase::Consuming { math_seen: false } => false,
                                IngestPhase::Consuming { math_seen: true } => {
                                    // The previous window's ingest finished: epilogue, then
                                    // a fresh window for this round.
                                    d.tabi().snapshot_round_bases();
                                    true
                                }
                            }
                        };
                        let idx = c
                            .data_mut()
                            .tabi()
                            .stage_bridge_update(&staged_bytes, fresh_window)
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e))?;
                        c.data_mut().ingest_phase = IngestPhase::Consuming { math_seen: false };
                        let mut out = Vec::new();
                        ciborium::into_writer(&ciborium::value::Value::from(idx), &mut out)
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                        shared.state.lock().expect("pump lock").staged.remove(&src);
                        out
                    }
                    _ => staged_bytes,
                };
                let len = value.len() as u64;
                if len > u64::from(out_cap) {
                    let d = c.data_mut();
                    d.slice.pending_readback = Some((src, kind, len));
                    d.slice.pending_readback_value = Some(value);
                    return Ok(pack_status_len(RET_STATUS_NEED_CAPACITY, len as u32));
                }
                // Charge the per-slice readback-byte budget (§5.5) before the bytes cross.
                {
                    let d = c.data_mut();
                    d.slice.readback_bytes += len;
                    if d.slice.readback_bytes > d.max_readback_bytes {
                        return Err(Trap::new(
                            TrapCode::GrantViolation,
                            "read_back",
                            None,
                            "per-slice readback-byte allowance exhausted (§5.5)",
                        ));
                    }
                }
                // Journal the Ok value (§6.4: every Ok return is journaled) then deliver.
                {
                    let mut st = shared.state.lock().expect("pump lock");
                    st.sink
                        .read_back(src, u64::from(kind), RET_STATUS_DELIVERED, &value)
                        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                }
                write_guest(c, out_ptr, &value)?;
                let d = c.data_mut();
                d.slice.pending_readback = None;
                d.slice.pending_readback_value = None;
                Ok(pack_status_len(RET_STATUS_DELIVERED, len as u32))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- vhc@2::stage_state — guest-created staged sections (§10.2) -----------------------------
    linker.func_wrap(
        NS_VHC_V2,
        "stage_state",
        |mut c: Caller<'_, V2Host>, ptr: u32, len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("stage_state")?;
                let bytes = read_guest(c, ptr, len)?;
                // The hard-accountable cap (ABI §9.1): staged bytes are the Phase-A metered
                // guest-attributable allocation. Breach is the typed ATTRIBUTABLE trap — the
                // module claimed less than it uses (the under-claim acceptance, refactor §5 A2).
                {
                    let d = c.data_mut();
                    d.accountable_staged_bytes += bytes.len() as u64;
                    if d.hard_accountable_host_bytes != 0
                        && d.accountable_staged_bytes > d.hard_accountable_host_bytes
                    {
                        return Err(Trap::new(
                            TrapCode::BudgetMemory,
                            "stage_state",
                            None,
                            format!(
                                "hard-accountable host cap breached: staged {} > claimed {} \
                                 (attributable to the module, ABI §9.1)",
                                d.accountable_staged_bytes, d.hard_accountable_host_bytes
                            ),
                        ));
                    }
                }
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let id = daemon_vhc_abi::GUEST_STAGING_ID_TOP_BIT | st.next_guest_staging_id;
                st.next_guest_staging_id += 1;
                st.staged.insert(id, (STAGED_KIND_BYTES, bytes));
                // Deterministic (counter-derived over replay-reproduced guest bytes): no record.
                Ok(id)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- vhc@2::snapshot_state — quiesce-scoped (§10.2; verification lands with migrate) --------
    linker.func_wrap(
        NS_VHC_V2,
        "snapshot_state",
        |mut c: Caller<'_, V2Host>, _ptr: u32, _len: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("snapshot_state")?;
                if !c.data().slice.draining {
                    return Err(Trap::new(
                        TrapCode::PhaseViolation,
                        "snapshot_state",
                        None,
                        "snapshot_state outside a Quiesce drain (§10.2)",
                    ));
                }
                // State-manifest verification is the migrate-scaffolding sitting; until then no
                // declared section can be verified staged, which is exactly `SectionMissing`.
                Ok(SNAPSHOT_STATE_SECTION_MISSING)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- vhc@2 minor 1: the buffer layer (architecture §3.4; ABI §7.4) --------------------------
    // create_from — the budgeted linear-memory path OUT: seal guest bytes into a kind-8 buffer.
    linker.func_wrap(
        NS_VHC_V2,
        "create_from",
        |mut c: Caller<'_, V2Host>, ptr: u32, len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("create_from")?;
                let bytes = read_guest(c, ptr, len)?;
                // The hard-accountable claim meter covers guest-initiated allocations (ABI §9.1
                // "tensors, buffers, handles the host meters exactly").
                {
                    let d = c.data_mut();
                    d.accountable_staged_bytes += u64::from(len);
                    if d.hard_accountable_host_bytes != 0
                        && d.accountable_staged_bytes > d.hard_accountable_host_bytes
                    {
                        return Err(Trap::new(
                            TrapCode::BudgetMemory,
                            "create_from",
                            None,
                            "hard-accountable host cap breached (attributable, ABI §9.1)",
                        ));
                    }
                }
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.buffers
                    .create(Arc::new(bytes))
                    .map_err(|code| Trap::new(code, "create_from", None, "buffer quota (§7.3)"))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // read_into — the budgeted linear-memory path IN: copy a window of a sealed buffer into guest
    // memory. Charged against the per-slice readback-byte allowance (§5.5); recordless — buffer
    // contents are deterministic at replay (create_from bytes) or content-addressed (payload_get).
    linker.func_wrap(
        NS_VHC_V2,
        "read_into",
        |mut c: Caller<'_, V2Host>,
         buffer: u64,
         offset: u64,
         out_ptr: u32,
         out_cap: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("read_into")?;
                let shared = c.data().shared.clone();
                let data = {
                    let st = shared.state.lock().expect("pump lock");
                    st.buffers
                        .resolve(buffer)
                        .map_err(|code| Trap::new(code, "read_into", None, "buffer handle"))?
                };
                let start = usize::try_from(offset).unwrap_or(usize::MAX);
                let window = data.len().saturating_sub(start.min(data.len()));
                let n = window.min(out_cap as usize);
                {
                    let d = c.data_mut();
                    d.slice.readback_bytes += n as u64;
                    if d.slice.readback_bytes > d.max_readback_bytes {
                        return Err(Trap::new(
                            TrapCode::GrantViolation,
                            "read_into",
                            None,
                            "per-slice readback-byte allowance exhausted (§5.5)",
                        ));
                    }
                }
                if n > 0 {
                    let slice = data[start..start + n].to_vec();
                    write_guest(c, out_ptr, &slice)?;
                }
                Ok(n as u64)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // buffer_len — deterministic bookkeeping (no journal record).
    linker.func_wrap(
        NS_VHC_V2,
        "buffer_len",
        |mut c: Caller<'_, V2Host>, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("buffer_len")?;
                let shared = c.data().shared.clone();
                let st = shared.state.lock().expect("pump lock");
                st.buffers
                    .resolve(buffer)
                    .map(|d| d.len() as u64)
                    .map_err(|code| Trap::new(code, "buffer_len", None, "buffer handle"))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // buffer_release — the guest's explicit release (§3.4 ownership); frees quota + claim meter.
    linker.func_wrap(
        NS_VHC_V2,
        "buffer_release",
        |mut c: Caller<'_, V2Host>, buffer: u64| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("buffer_release")?;
                let shared = c.data().shared.clone();
                let freed = {
                    let mut st = shared.state.lock().expect("pump lock");
                    st.buffers
                        .release(buffer)
                        .map_err(|code| Trap::new(code, "buffer_release", None, "buffer handle"))?
                };
                let d = c.data_mut();
                d.accountable_staged_bytes = d.accountable_staged_bytes.saturating_sub(freed);
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // cancel — the completion protocol's cancellation (§3.3/§7.5): an outstanding op is retired
    // NOW and its completion (reporting Cancelled) is enqueued deterministically; a late service
    // outcome is ignored. Recordless: the journaled completion result captures the race (§8.3).
    linker.func_wrap(
        NS_VHC_V2,
        "cancel",
        |mut c: Caller<'_, V2Host>, op: u64| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("cancel")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                if st.ops.finish(op).is_some() {
                    // A credit-held stream write is un-held with its op (its bytes never left).
                    st.streams.cancel_held(op);
                    st.enqueue_completion(op, &CompletionResult::cancelled())
                        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                    Ok(0) // cancel accepted: the op's completion reports Cancelled
                } else {
                    Ok(1) // already completed/cancelled or never issued
                }
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- net@2 minor 1: content-addressed payloads by handle (§3.4) — both complete async -------
    linker.func_wrap(
        NS_NET_V2,
        "payload_put",
        |mut c: Caller<'_, V2Host>, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("payload_put")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let bytes = st
                    .buffers
                    .resolve(buffer)
                    .map_err(|code| Trap::new(code, "payload_put", None, "buffer handle"))?;
                let request = OpRequest::PayloadPut { bytes };
                let op = st.ops.begin(request.clone()).map_err(|code| {
                    Trap::new(code, "payload_put", None, "max_outstanding grant (§2.3)")
                })?;
                st.op_requests.push((op, request));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "payload_get",
        |mut c: Caller<'_, V2Host>, hash_ptr: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("payload_get")?;
                let hash_bytes = read_guest(c, hash_ptr, 32)?;
                let hash: [u8; 32] = hash_bytes.as_slice().try_into().expect("32-byte span");
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::PayloadGet { hash })
                    .map_err(|code| {
                        Trap::new(code, "payload_get", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::PayloadGet { hash }));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- data@2::fetch — artifact fetch by committed hash + range (track B2; §3.2) --------------
    // The guest names CONTENT, never location: the only inputs are the committed blake3 (edge-
    // pinned in the envelope's artifact map, §5.1) and a byte range — no URL, no locator, no
    // credential crosses this boundary (the resolver + its credentials stay embedder-side).
    // Which artifacts a module may touch is a GRANT: a hash outside the admitted set traps
    // GrantViolation before any op is issued. Completes Ok(BufferHandle) via tag 6 after the
    // pump whole-artifact-verifies + range-slices (see complete_op).
    linker.func_wrap(
        NS_DATA_V2,
        "fetch",
        |mut c: Caller<'_, V2Host>,
         hash_ptr: u32,
         range_off: u64,
         range_len: u64|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("fetch")?;
                let hash_bytes = read_guest(c, hash_ptr, 32)?;
                let hash: [u8; 32] = hash_bytes.as_slice().try_into().expect("32-byte span");
                if !c.data().granted_artifacts.contains(&hash) {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "fetch",
                        None,
                        format!(
                            "artifact {} is not in the admitted artifact set (which artifacts a \
                             module may touch is a grant, architecture §3.2)",
                            hash[..4]
                                .iter()
                                .map(|b| format!("{b:02x}"))
                                .collect::<String>()
                        ),
                    ));
                }
                let request = OpRequest::ArtifactFetch {
                    hash,
                    range_off,
                    range_len,
                };
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st.ops.begin(request.clone()).map_err(|code| {
                    Trap::new(code, "fetch", None, "max_outstanding grant (§2.3)")
                })?;
                st.op_requests.push((op, request));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- net@2 minor 1: direct peer streams under credit flow control (§3.3/§3.4) ---------------
    linker.func_wrap(
        NS_NET_V2,
        "stream_open",
        |mut c: Caller<'_, V2Host>, peer_ptr: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("stream_open")?;
                let peer_bytes = read_guest(c, peer_ptr, 32)?;
                let peer: [u8; 32] = peer_bytes.as_slice().try_into().expect("32-byte span");
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::StreamOpen { peer })
                    .map_err(|code| {
                        Trap::new(code, "stream_open", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::StreamOpen { peer }));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_accept",
        |mut c: Caller<'_, V2Host>| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("stream_accept")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st.ops.begin(OpRequest::StreamAccept).map_err(|code| {
                    Trap::new(code, "stream_accept", None, "max_outstanding grant (§2.3)")
                })?;
                st.op_requests.push((op, OpRequest::StreamAccept));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_write",
        |mut c: Caller<'_, V2Host>, stream: u64, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("stream_write")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let bytes = st
                    .buffers
                    .resolve(buffer)
                    .map_err(|code| Trap::new(code, "stream_write", None, "buffer handle"))?;
                let op = st
                    .ops
                    .begin(OpRequest::StreamWrite {
                        stream,
                        bytes: bytes.clone(),
                    })
                    .map_err(|code| {
                        Trap::new(code, "stream_write", None, "max_outstanding grant (§2.3)")
                    })?;
                // Credit flow control (§3.3): the transport request is emitted only when the
                // stream's writable credit covers the bytes; otherwise the op is HELD pump-side
                // (still outstanding — the guest's OpId is live) until the receiver's reads
                // replenish credit.
                match st.streams.write(stream, op, bytes.clone()) {
                    Some(true) => {
                        st.op_requests
                            .push((op, OpRequest::StreamWrite { stream, bytes }));
                    }
                    Some(false) => { /* held for credit */ }
                    None => {
                        st.ops.finish(op);
                        return Err(Trap::new(
                            TrapCode::StaleHandle,
                            "stream_write",
                            None,
                            "unknown or stale stream handle",
                        ));
                    }
                }
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_read",
        |mut c: Caller<'_, V2Host>, stream: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("stream_read")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                if !st.streams.is_live(stream) {
                    return Err(Trap::new(
                        TrapCode::StaleHandle,
                        "stream_read",
                        None,
                        "unknown or stale stream handle",
                    ));
                }
                let op = st
                    .ops
                    .begin(OpRequest::StreamRead { stream })
                    .map_err(|code| {
                        Trap::new(code, "stream_read", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::StreamRead { stream }));
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- net@2::publish — the signed, sequenced, durable egress door (§6.2/§12) -----------------
    linker.func_wrap(
        NS_NET_V2,
        "publish",
        |mut c: Caller<'_, V2Host>,
         channel_id: u32,
         payload_ptr: u32,
         payload_len: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("publish")?;
                // The channel table decides class/direction/bounds — never the guest (§6.2).
                let decl = PHASE_A_DEFAULT_CHANNEL_TABLE
                    .iter()
                    .find(|ch| ch.id == channel_id)
                    .ok_or_else(|| {
                        Trap::new(
                            TrapCode::GrantViolation,
                            "publish",
                            None,
                            format!("undeclared channel {channel_id} (§6.2)"),
                        )
                    })?;
                if decl.direction == CHANNEL_DIR_RX_ONLY {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "publish",
                        None,
                        format!("channel {channel_id} is rx-only (§6.2)"),
                    ));
                }
                if payload_len > c.data().max_frame_bytes {
                    return Err(Trap::new(
                        TrapCode::PayloadOverflow,
                        "publish",
                        None,
                        format!(
                            "payload {payload_len} bytes > max_frame_bytes {}",
                            c.data().max_frame_bytes
                        ),
                    ));
                }
                let payload = read_guest(c, payload_ptr, payload_len)?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // Atomic commit (§6.2): seq allocation + the tag-4 record + (Phase-A) the spool
                // stand-in are covered by the sink's publish barrier before the guest sees seq.
                let seq = st.sink.next_seq(u64::from(channel_id));
                let frame = build_signed_frame(c.data(), u64::from(channel_id), seq, &payload)?;
                st.sink
                    .publish(u64::from(channel_id), seq, &payload, &frame)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                st.published.push((u64::from(channel_id), seq, frame));
                Ok(seq)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- sys@2 ------------------------------------------------------------------------------------
    linker.func_wrap(
        NS_SYS_V2,
        "set_timer",
        |mut c: Caller<'_, V2Host>, delay_ms: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("set_timer")?;
                let armed_at = c.data().slice.now;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let id = st.next_timer_id;
                st.next_timer_id += 1;
                st.sink
                    .timer_arm(id, delay_ms, armed_at)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                st.timers.push(ArmedTimer {
                    id,
                    fire_at: armed_at.saturating_add(delay_ms),
                });
                drop(st);
                shared.wake.notify_all();
                Ok(id)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "cancel_timer",
        |mut c: Caller<'_, V2Host>, timer_id: u64| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("cancel_timer")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // Cancelled iff still armed AND not already queued for delivery (§6.3): a queued
                // Timer event is removed too (the host MUST NOT deliver after a 0 return).
                let mut cancelled = false;
                if let Some(pos) = st.timers.iter().position(|t| t.id == timer_id) {
                    st.timers.remove(pos);
                    cancelled = true;
                } else if let Some(pos) = st.queue.iter().position(|q| q.timer_id == Some(timer_id))
                {
                    // Fired but not yet delivered: remove the queued Timer event too — the host
                    // MUST NOT deliver after a `0 = Cancelled` return (§6.3).
                    st.queue.remove(pos);
                    cancelled = true;
                }
                let status = if cancelled {
                    daemon_vhc_abi::CANCEL_TIMER_CANCELLED
                } else {
                    daemon_vhc_abi::CANCEL_TIMER_ALREADY_FIRED_OR_UNKNOWN
                };
                st.sink
                    .timer_cancel(timer_id, u64::from(status))
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                Ok(status)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "now",
        |mut c: Caller<'_, V2Host>| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("now")?;
                // Slice-constant (§6.5); every reading journaled (the coordinator-replay lesson).
                let now = c.data().slice.now;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.sink
                    .clock(now)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                Ok(now)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "emit_metric",
        |mut c: Caller<'_, V2Host>,
         name_ptr: u32,
         name_len: u32,
         value: f64|
         -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("emit_metric")?;
                // Egress only; non-finite / oversize-name → dropped host-side, never a trap (§6.5).
                if !value.is_finite() || name_len > 128 {
                    return Ok(());
                }
                let name =
                    String::from_utf8_lossy(&read_guest(c, name_ptr, name_len)?).into_owned();
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.metrics.push((name, value));
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "log",
        |mut c: Caller<'_, V2Host>,
         level: u32,
         msg_ptr: u32,
         msg_len: u32|
         -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("log")?;
                let msg = String::from_utf8_lossy(&read_guest(c, msg_ptr, msg_len)?).into_owned();
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.logs.push((level.min(5), msg));
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- sys@2::hash — crypto acceleration: blake3-256 over guest bytes (§3.2/§3.7) -------------
    // The det/crypto-lane fast path: the in-guest fallback (daemon_vhc_proto::crypto) is always
    // available, this host import accelerates it, and the two are bit-identical by construction.
    // `(in_ptr, in_len, out_ptr) -> status`; writes HASH_LEN bytes to `out_ptr`, returns 0 (Ok).
    // Deterministic function of the input → no journal record (§2.7 dc class).
    linker.func_wrap(
        NS_SYS_V2,
        "hash",
        |mut c: Caller<'_, V2Host>,
         in_ptr: u32,
         in_len: u32,
         out_ptr: u32|
         -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("hash")?;
                let data = read_guest(c, in_ptr, in_len)?;
                let digest = host_crypto_hash(&data);
                write_guest(c, out_ptr, &digest)?;
                Ok(0)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- sys@2::verify_sig — crypto acceleration: ed25519 verify (§3.2/§3.7) --------------------
    // `(pk_ptr, sig_ptr, msg_ptr, msg_len) -> VerifyOutcome code` (0 valid / 1 invalid / 2
    // malformed). Fixed-length key/sig spans; a bad-length span is a MemOob trap (an out-of-bounds
    // read), a structurally-bad-but-in-bounds key/sig is `Malformed` (2). Deterministic → not
    // journaled; replay re-executes it.
    linker.func_wrap(
        NS_SYS_V2,
        "verify_sig",
        |mut c: Caller<'_, V2Host>,
         pk_ptr: u32,
         sig_ptr: u32,
         msg_ptr: u32,
         msg_len: u32|
         -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("verify_sig")?;
                let pk = read_guest(c, pk_ptr, daemon_vhc_proto::VERIFY_PUBLIC_KEY_LEN as u32)?;
                let sig = read_guest(c, sig_ptr, daemon_vhc_proto::VERIFY_SIGNATURE_LEN as u32)?;
                let msg = read_guest(c, msg_ptr, msg_len)?;
                Ok(host_crypto_verify(&pk, &sig, &msg))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- sys@2::rng_seed — the run-scoped deterministic seed (architecture §3.2) ----------------
    // `(out_ptr) -> status 0`; writes RNG_SEED_LEN bytes. A pure function of the execution
    // identity (derive_rng_seed) — §2.7 dc class, no journal record; replay re-derives it.
    linker.func_wrap(
        NS_SYS_V2,
        "rng_seed",
        |mut c: Caller<'_, V2Host>, out_ptr: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("rng_seed")?;
                let seed = c.data().rng_seed;
                write_guest(c, out_ptr, &seed)?;
                Ok(0)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- sys@2::device_profile — the probe's device profile, guest-readable (§3.5) --------------
    // `(out_ptr, out_cap) -> (status << 32) | len` with the §4.1/§6.4 NeedCapacity protocol. The
    // profile is what makes module autotune possible ("micro-batch autotune moves inside the
    // module", architecture §3.5). It is a NONDETERMINISTIC INPUT — the same probe measurement
    // admission judged — so every `Ok` delivery is journaled as the §8.3 tag-15 record and replay
    // feeds the recorded bytes, never a fresh probe.
    linker.func_wrap(
        NS_SYS_V2,
        "device_profile",
        |mut c: Caller<'_, V2Host>, out_ptr: u32, out_cap: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("device_profile")?;
                // Mandatory-retry rule: after NeedCapacity the re-call must be big enough.
                if let Some(required) = c.data().slice.pending_device {
                    if u64::from(out_cap) < required {
                        return Err(Trap::new(
                            TrapCode::BadEvent,
                            "device_profile",
                            None,
                            format!("retry with out_cap {out_cap} < required {required} (§6.4)"),
                        ));
                    }
                }
                let bytes = c.data().device_bytes.clone();
                let len = bytes.len() as u64;
                if len > u64::from(out_cap) {
                    c.data_mut().slice.pending_device = Some(len);
                    return Ok(pack_status_len(RET_STATUS_NEED_CAPACITY, len as u32));
                }
                write_guest(c, out_ptr, &bytes)?;
                c.data_mut().slice.pending_device = None;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.sink
                    .device_profile(&bytes)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                Ok(pack_status_len(RET_STATUS_DELIVERED, len as u32))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ==== compute@2 (track C1, ABI §15; architecture §3.3/§3.4) — the Burn-IR command queue ======
    // The wire is CBOR(burn_ir::OperationIr) at the pinned Burn version; dispatch is the
    // ComputeRunner (burn-router runner + typed handle faults + the deferred-error latch).
    // Validation faults trap at the call (§7.6 programming errors); DEVICE faults defer to
    // fence (ComputeFault trap) / export (COMP_ERR_DEVICE completion) — §3.3.

    // ---- compute@2::submit_op — enqueue one op-blob (infallible for device faults) --------------
    linker.func_wrap(
        NS_COMPUTE_V2,
        "submit_op",
        |mut c: Caller<'_, V2Host>, op_ptr: u32, op_len: u32| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("submit_op")?;
                let op_cbor = read_guest(c, op_ptr, op_len)?;
                let d = c.data_mut();
                if d.compute.is_none() {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                }
                // The queue-depth grant (architecture §3.3): outstanding device work is bounded;
                // the guest reclaims depth by fencing.
                if d.compute_queue_depth != 0 && d.compute_ops_since_fence >= d.compute_queue_depth
                {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "submit_op",
                        None,
                        format!(
                            "compute queue depth {} reached — fence to reclaim (§3.3)",
                            d.compute_queue_depth
                        ),
                    ));
                }
                let compute = d.compute.as_mut().expect("checked above");
                compute
                    .submit_op(&op_cbor)
                    .map_err(|e| Trap::new(e.trap_code(), "submit_op", None, e.to_string()))?;
                d.compute_ops_since_fence += 1;
                d.compute_ops_total += 1;
                // The deferred-fault injection seam (V2RunConfig::compute_fault_after_ops).
                if d.compute_fault_after_ops == Some(d.compute_ops_total - 1) {
                    d.compute
                        .as_mut()
                        .expect("checked above")
                        .inject_device_fault("injected deferred device fault (test seam)");
                }
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::fence — insert a marker; Event::Fence(id) delivers when the device passes it
    linker.func_wrap(
        NS_COMPUTE_V2,
        "fence",
        |mut c: Caller<'_, V2Host>, fence_id: u64| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("fence")?;
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // Deferred device errors surface HERE, typed (§3.3): the fence event is
                // delivered only on a successful drain, so a delivered Fence is a real
                // consistency point.
                compute
                    .fence()
                    .map_err(|e| Trap::new(e.trap_code(), "fence", None, e.to_string()))?;
                d.compute_ops_since_fence = 0;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.enqueue_fence(fence_id)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::export — device tensor → sealed buffer (bulk bytes ride the BufferHandle,
    // §3.4 — never inline in the op-stream). Returns an OpId; completes Ok(BufferHandle) with the
    // CBOR(TensorData), journaled verbatim (kind-5 tag-2 — device bytes are a nondeterministic
    // input); a deferred device error completes Err(COMP_ERR_DEVICE) — the readback twin of the
    // fence trap.
    linker.func_wrap(
        NS_COMPUTE_V2,
        "export",
        |mut c: Caller<'_, V2Host>, ir_ptr: u32, ir_len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("export")?;
                let ir_cbor = read_guest(c, ir_ptr, ir_len)?;
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // Stale/invalid handles and undecodable IR are programming errors → trap at the
                // call (§7.6); only the DEVICE fault defers into the completion.
                let read = match compute.read_tensor(&ir_cbor) {
                    Ok(data) => Ok(data),
                    Err(e @ crate::compute::ComputeError::Device(_)) => Err(e),
                    Err(e) => return Err(Trap::new(e.trap_code(), "export", None, e.to_string())),
                };
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st.ops.begin(OpRequest::TensorExport).map_err(|code| {
                    Trap::new(code, "export", None, "max_outstanding grant (§2.3)")
                })?;
                // Pump-internal service AT THE CALL (the runner is host-local): the op never
                // reaches `op_requests`, so transport seats never see a TensorExport.
                st.ops.finish(op);
                let result = match read {
                    Ok(data) => {
                        // Journal the device bytes verbatim (kind 5) BEFORE the completion
                        // record — the stream-read (kind 4) discipline: replay materializes the
                        // completion's buffer from this record and re-executes no kernel (§8.7).
                        st.sink
                            .read_back(
                                op,
                                u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT),
                                daemon_vhc_abi::RET_STATUS_DELIVERED,
                                &data,
                            )
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                        match st.buffers.create_host(Arc::new(data)) {
                            Some(handle) => CompletionResult::Ok(SuccessPayload::Handle(handle)),
                            None => CompletionResult::Err(CompError {
                                code: COMP_ERR_GRANT_EXHAUSTED,
                                detail: Some("buffer quota exhausted (deny new buffers)".into()),
                            }),
                        }
                    }
                    Err(e) => CompletionResult::Err(CompError {
                        code: daemon_vhc_abi::COMP_ERR_DEVICE,
                        detail: Some(e.to_string()),
                    }),
                };
                st.enqueue_completion(op, &result)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::import — sealed buffer → device tensor under the guest-minted TensorId.
    // Returns an OpId; completes Ok(()). Deterministic (guest bytes by way of the sealed buffer):
    // no journal record beyond the tag-14 completion.
    linker.func_wrap(
        NS_COMPUTE_V2,
        "import",
        |mut c: Caller<'_, V2Host>, buffer: u64, tensor_id: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, V2Host>| {
                c.data_mut().enter("import")?;
                let shared = c.data().shared.clone();
                let bytes = {
                    let st = shared.state.lock().expect("pump lock");
                    st.buffers
                        .resolve(buffer)
                        .map_err(|code| Trap::new(code, "import", None, "buffer handle"))?
                };
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // The buffer must hold decodable CBOR(TensorData) — a malformed import is a
                // programming error at the call (§7.6), not a completion error.
                compute
                    .import_tensor(tensor_id, &bytes)
                    .map_err(|e| Trap::new(e.trap_code(), "import", None, e.to_string()))?;
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::TensorImport { tensor_id })
                    .map_err(|code| {
                        Trap::new(code, "import", None, "max_outstanding grant (§2.3)")
                    })?;
                st.ops.finish(op); // pump-internal service at the call (see export)
                st.enqueue_completion(op, &CompletionResult::Ok(SuccessPayload::Unit))
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    Ok(())
}

/// Stage the guest's own sealed container bytes and announce them as `PayloadReady` kind 0 —
/// the slice-boundary half of the B1 sealing-gap retirement (see the `next_event` body).
fn stage_own_bytes(st: &mut PumpState, bytes: Vec<u8>) -> Result<(), SinkError> {
    if st.stop_enqueued {
        return Ok(()); // no deliveries after Stop (§4.4)
    }
    let hash = *blake3::hash(&bytes).as_bytes();
    st.enforce_payload_depth()?;
    let staging_id = st.next_host_staging_id;
    st.next_host_staging_id += 1;
    let size = bytes.len() as u64;
    st.staged.insert(staging_id, (STAGED_KIND_BYTES, bytes));
    let ev = EventV2::PayloadReady {
        staging_id,
        hash,
        meta: PayloadMeta {
            size,
            kind: STAGED_KIND_BYTES,
            channel: None,
        },
    };
    let frame_bytes = encode_event_frame(&ev).map_err(|e| SinkError(e.to_string()))?;
    st.queue.push_back(QueuedEvent {
        frame_bytes,
        tag: daemon_vhc_abi::EV_TAG_PAYLOAD_READY,
        signed: None,
        payload_hash: Some(hash),
        timer_id: None,
        is_budget: false,
        gossip_id: None,
    });
    Ok(())
}

/// Move due timers into the queue as `Timer` events, in `(fire_at, timer_id)` order (§6.3).
fn fire_due_timers(st: &mut PumpState, now: u64) -> Result<(), Trap> {
    if !st.timers.iter().any(|t| t.fire_at <= now) {
        return Ok(());
    }
    let (mut fired, keep): (Vec<ArmedTimer>, Vec<ArmedTimer>) =
        st.timers.drain(..).partition(|t| t.fire_at <= now);
    st.timers = keep;
    // Deterministic firing order (§6.3).
    fired.sort_by_key(|t| (t.fire_at, t.id));
    for t in fired {
        let ev = EventV2::Timer {
            timer_id: t.id,
            fired_at: now,
        };
        let frame_bytes =
            encode_event_frame(&ev).map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
        // Bounded advisory depth, latest-wins for timers: dropping the OLDEST queued Timer beyond
        // the declared depth, journaled (§4.7).
        let queued_timers = st.queue.iter().filter(|q| q.timer_id.is_some()).count();
        if queued_timers >= st.timer_depth {
            if let Some(pos) = st.queue.iter().position(|q| q.timer_id.is_some()) {
                let dropped = st
                    .queue
                    .remove(pos)
                    .and_then(|q| q.timer_id)
                    .expect("positioned on a queued Timer");
                st.sink
                    .drop_coalesced(
                        1,
                        daemon_vhc_abi::COALESCE_LATEST_WINS,
                        crate::v2::journal::Dropped::timer(dropped),
                    )
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
            }
        }
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_TIMER,
            signed: None,
            payload_hash: None,
            timer_id: Some(t.id),
            is_budget: false,
            gossip_id: None,
        });
    }
    Ok(())
}

// -- the run ---------------------------------------------------------------------------------------

/// A live v2 run: the embedder handle plus the guest thread's join handle.
pub struct V2Run {
    /// The embedder's event/staging/egress handle.
    pub pump: PumpHandle,
    thread: JoinHandle<Result<RunEnd, V2Error>>,
}

impl V2Run {
    /// Join the guest thread and return how the run ended. The guest thread has already dropped
    /// the `Store` (guest-thread-owned teardown, §11.3) and journaled the terminal fact.
    ///
    /// # Errors
    /// [`V2Error`] for setup/journaling failures (a trap is a [`RunEnd::Trapped`], not an error).
    pub fn wait(self) -> Result<RunEnd, V2Error> {
        self.thread
            .join()
            .map_err(|_| V2Error::Sandbox("guest thread panicked".into()))?
    }
}

/// Start a major-2 run instance: journal the run header, spawn the dedicated guest thread,
/// instantiate with the real Phase-A capability providers, run `da_init` then `da_run` (§3.1,
/// §9.4 steps 10–12), journaling throughout.
///
/// The caller has already run ABI §1.3 selection (`select_driver` → `CandidateDriver::V2`).
/// A module importing the `tabi@1` bridge gets the frozen dispatch linked beside the v2
/// namespaces (§2.5).
///
/// # Errors
/// [`V2Error`] on setup/journal failure. Guest traps and init refusals are [`RunEnd`]s.
pub fn start_run(
    worker: &Worker,
    wasm: &[u8],
    run: V2RunConfig,
    mut sink: Box<dyn JournalSink>,
) -> Result<V2Run, V2Error> {
    let module = Module::new(worker.engine(), wasm).map_err(|e| V2Error::Sandbox(e.to_string()))?;
    // The §2.5 compute bridge: a major-2 module MAY link the frozen tabi@1 vocabulary; the host
    // links the SAME dispatch the v1 driver uses (genericized over the store — never forked)
    // while the bridge is advertised. This retires the sitting-3 `BridgeUnwired` bound.
    let bridge = module.imports().any(|i| i.module() == NS_TABI_V1);
    // The compute@2 command queue (track C1, ABI §15): a per-instance ComputeRunner over the
    // tier-1 real backend, constructed only for modules that import the world.
    let compute = module.imports().any(|i| i.module() == NS_COMPUTE_V2);

    let engine_cfg: EngineConfig = worker.config().clone();
    let abi_packed = u64::from(daemon_vhc_abi::DA_ABI_MAJOR_V2) << 16;
    let worlds: Vec<(String, u64)> = module
        .imports()
        .map(|i| (i.module().to_string(), 0u64))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    // tag 0 first — the run header precedes everything (§8.3).
    sink.run_header(
        abi_packed,
        &worlds,
        bridge,
        &run.manifest_bytes,
        &run.config,
        &run.grants,
        &run.claim_bytes,
        &run.channels_bytes,
        &run.device_bytes,
    )?;

    let shared = Arc::new(PumpShared {
        state: Mutex::new(PumpState {
            queue: VecDeque::new(),
            timers: Vec::new(),
            next_timer_id: 1,
            staged: std::collections::BTreeMap::new(),
            next_host_staging_id: 1,
            next_guest_staging_id: 1,
            sink,
            timer_depth: run.advisory_depth,
            payload_depth: run.payload_depth,
            gossip_depth: run.gossip_depth,
            spool_frames: run.spool_frames,
            per_sender_quota: run.per_sender_quota,
            auth_spooled: 0,
            auth_per_sender: std::collections::HashMap::new(),
            spool_exhausted_reported: false,
            gossip_arrivals: std::collections::HashMap::new(),
            metrics: Vec::new(),
            logs: Vec::new(),
            published: Vec::new(),
            bridge_final_state: None,
            // Generation-seeded by the instantiation counter (0: this driver instantiates once
            // per start_run; trap-restart re-seeding rides the tag-13 counter, ABI §7.1).
            buffers: BufferTable::new(0, run.max_live_buffer_handles, run.max_live_buffer_bytes),
            ops: OpTable::new(0, run.max_outstanding_ops),
            streams: StreamTable::new(0),
            op_requests: Vec::new(),
            stop_enqueued: false,
            draining: false,
        }),
        wake: Condvar::new(),
        t0: Instant::now(),
        hold: AtomicBool::new(false),
    });
    let pump = PumpHandle {
        shared: shared.clone(),
    };

    let mut linker: Linker<V2Host> = Linker::new(worker.engine());
    link_v2(&mut linker).map_err(|e| V2Error::Sandbox(e.to_string()))?;
    if bridge {
        // The identical frozen dispatch the v1 linker carries, monomorphized over V2Host (§2.5).
        crate::runtime::link_tabi(&mut linker).map_err(|e| V2Error::Sandbox(e.to_string()))?;
    }

    let signing = SigningKey::from_bytes(&run.signing_seed);
    let sender = peer_id(&signing).0;
    let epoch_ticks = worker.epoch_ticks_pub();
    let engine = worker.engine().clone();

    let thread = std::thread::Builder::new()
        .name(format!(
            "vhc-guest-{}-{}",
            run.identity.role, run.identity.instance
        ))
        .spawn(move || -> Result<RunEnd, V2Error> {
            let host = V2Host {
                shared: shared.clone(),
                limits: StoreLimitsBuilder::new()
                    .memory_size(engine_cfg.max_memory_bytes)
                    .build(),
                trap: None,
                slice: SliceState {
                    in_init: false,
                    stopped: false,
                    draining: false,
                    now: shared.now_ms(),
                    op_calls: 0,
                    readback_bytes: 0,
                    pending_next: None,
                    pending_readback: None,
                    pending_readback_value: None,
                    pending_device: None,
                },
                fuel_per_slice: engine_cfg.fuel_per_call,
                op_budget: engine_cfg.op_budget,
                epoch_ticks,
                max_readback_bytes: run.max_readback_bytes_per_slice,
                max_frame_bytes: run.max_frame_bytes,
                hard_accountable_host_bytes: run.hard_accountable_host_bytes,
                accountable_staged_bytes: 0,
                tabi: bridge.then(|| crate::runtime::HostState::new(&engine_cfg)),
                slice_pass_open: false,
                ingest_phase: IngestPhase::Idle,
                sealed_containers: 0,
                compute: compute.then(|| {
                    let runner = crate::compute::ComputeRunner::ndarray_cpu();
                    // Host-side RNG (Float/Random ops) seeded deterministically from the
                    // identity-derived seed: two runs of one incarnation reproduce it, and
                    // replay never re-runs kernels anyway (kind-5 records feed readbacks).
                    let seed_bytes = derive_rng_seed(&run.identity);
                    runner.seed(u64::from_le_bytes(
                        seed_bytes[..8].try_into().expect("8-byte slice"),
                    ));
                    runner
                }),
                compute_queue_depth: run.compute_queue_depth,
                compute_ops_since_fence: 0,
                compute_fault_after_ops: run.compute_fault_after_ops,
                compute_ops_total: 0,
                signing,
                rng_seed: derive_rng_seed(&run.identity),
                device_bytes: run.device_bytes.clone(),
                granted_artifacts: run.granted_artifacts.clone(),
                identity: run.identity.clone(),
                sender,
            };
            let mut store = Store::new(&engine, host);
            store.limiter(|s| &mut s.limits);
            store
                .set_fuel(engine_cfg.fuel_per_call)
                .map_err(|e| V2Error::Sandbox(e.to_string()))?;
            store.set_epoch_deadline(epoch_ticks);

            let instance = linker
                .instantiate(&mut store, &module)
                .map_err(|e| V2Error::Sandbox(format!("v2 instantiation: {e}")))?;

            // tag 13 at instantiation, before any guest code (§8.3/§10.3): counter 0, initial.
            let inst_at = shared.now_ms();
            {
                let mut st = shared.state.lock().expect("pump lock");
                st.sink.instantiation(0, 0, inst_at)?;
            }
            store.data_mut().slice.now = inst_at;

            // Write the admitted config + grants via da_alloc (outside import context, §2.4).
            let write_span = |store: &mut Store<V2Host>, bytes: &[u8]| -> Result<u32, V2Error> {
                if bytes.is_empty() {
                    return Ok(0);
                }
                let alloc = instance
                    .get_typed_func::<(u32, u32), u32>(&mut *store, "da_alloc")
                    .map_err(|_| V2Error::Sandbox("missing da_alloc".into()))?;
                let ptr = alloc
                    .call(&mut *store, (bytes.len() as u32, 1))
                    .map_err(|e| V2Error::Sandbox(format!("da_alloc: {e}")))?;
                if ptr == 0 {
                    return Err(V2Error::Sandbox("da_alloc returned 0".into()));
                }
                let mem = instance
                    .get_memory(&mut *store, "memory")
                    .ok_or_else(|| V2Error::Sandbox("no exported memory".into()))?;
                mem.write(&mut *store, ptr as usize, bytes)
                    .map_err(|e| V2Error::Sandbox(format!("config write: {e}")))?;
                Ok(ptr)
            };
            let cfg_ptr = write_span(&mut store, &run.config)?;
            let grants_ptr = write_span(&mut store, &run.grants)?;

            // da_init — once, on the run instance, imports illegal inside it (§3.1/§6.6).
            store.data_mut().slice.in_init = true;
            let da_init = instance
                .get_typed_func::<(u32, u32, u32, u32), u32>(&mut store, "da_init")
                .map_err(|_| V2Error::Sandbox("missing/mis-typed da_init".into()))?;
            let init_status = match da_init.call(
                &mut store,
                (
                    cfg_ptr,
                    run.config.len() as u32,
                    grants_ptr,
                    run.grants.len() as u32,
                ),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let trap = take_trap(&mut store, e);
                    journal_terminal_trap(&shared, &trap)?;
                    return Ok(RunEnd::Trapped(trap));
                }
            };
            store.data_mut().slice.in_init = false;
            {
                let mut st = shared.state.lock().expect("pump lock");
                st.sink.init(
                    *blake3::hash(&run.config).as_bytes(),
                    *blake3::hash(&run.grants).as_bytes(),
                    u64::from(init_status),
                )?;
            }
            if init_status != 0 {
                // Journal, tear down, refuse the join (§9.4 step 11). Store drops on this thread.
                return Ok(RunEnd::InitRefused(init_status));
            }

            // da_run — exactly once; the module owns its loop from here (§3.1).
            let da_run = instance
                .get_typed_func::<(), u32>(&mut store, "da_run")
                .map_err(|_| V2Error::Sandbox("missing/mis-typed da_run".into()))?;
            let run_result = da_run.call(&mut store, ());
            // The parity-oracle hook: export the bridge's final canonical state through the pump
            // BEFORE this thread drops the store (guest-thread-owned teardown, §11.3) — the
            // digest input the det-lane comparisons consume (§3.6).
            {
                let final_state = store
                    .data()
                    .tabi
                    .as_ref()
                    .map(crate::runtime::HostState::canonical_state_bytes_of);
                let mut st = shared.state.lock().expect("pump lock");
                st.bridge_final_state = final_state;
                // Force-reclaim the instance's buffers + outstanding ops + streams through the
                // per-instance tables (architecture §3.4; ABI §7.3) — guest-thread-owned teardown.
                st.buffers.clear();
                st.ops.clear();
                st.streams.clear();
                st.op_requests.clear();
            }
            match run_result {
                Ok(outcome) => {
                    let mut st = shared.state.lock().expect("pump lock");
                    st.sink.terminal(0, Some(u64::from(outcome)), None)?;
                    Ok(RunEnd::Outcome(outcome))
                }
                Err(e) => {
                    let trap = take_trap(&mut store, e);
                    journal_terminal_trap(&shared, &trap)?;
                    Ok(RunEnd::Trapped(trap))
                } // `store` (instance, handle table, device allocations) drops HERE, on the guest
                  // thread — the only thread allowed to (§11.3).
            }
        })
        .map_err(|e| V2Error::Sandbox(format!("guest thread spawn: {e}")))?;

    Ok(V2Run { pump, thread })
}

/// Map a wasmtime error into the typed taxonomy: prefer the stashed host trap, else classify the
/// engine trap (fuel/epoch/unreachable/oob), mirroring the v1 driver's mapping (§7.6).
fn take_trap(store: &mut Store<V2Host>, e: wasmtime::Error) -> Trap {
    if let Some(t) = store.data_mut().trap.take() {
        return t;
    }
    let msg = e
        .chain()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ");
    let low = msg.to_lowercase();
    let code = if low.contains("fuel") {
        TrapCode::BudgetFuel
    } else if low.contains("epoch") {
        TrapCode::BudgetEpoch
    } else if low.contains("unreachable") {
        TrapCode::GuestPanic
    } else if low.contains("out of bounds") {
        TrapCode::MemOob
    } else if low.contains("memory") {
        TrapCode::BudgetMemory
    } else {
        TrapCode::BadModule
    };
    Trap::bare(code, msg)
}

impl crate::runtime::TabiHost for V2Host {
    fn tabi(&mut self) -> &mut crate::runtime::HostState {
        self.tabi
            .as_mut()
            .expect("bridge linked only when tabi imports exist")
    }
    fn tabi_ref(&self) -> &crate::runtime::HostState {
        self.tabi
            .as_ref()
            .expect("bridge linked only when tabi imports exist")
    }
    /// The §2.5 temporal-legality rules (the v1 phase table does NOT apply): registration imports
    /// are legal ONLY during `da_init`; every other bridge import is legal in any `da_run` slice;
    /// every call charges the v2 per-slice op budget (§5.5) and honors the mandatory-retry rules.
    fn enter_tabi(&mut self, import: &'static str) -> Result<(), Trap> {
        if self.slice.stopped {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                "bridge import after Stop was consumed (§4.4)",
            ));
        }
        if self.slice.pending_next.is_some() || self.slice.pending_readback.is_some() {
            return Err(Trap::new(
                TrapCode::BadEvent,
                import,
                None,
                "NeedCapacity requires an immediate retry before any other import (§4.1/§6.4)",
            ));
        }
        let registration = matches!(import, "param@1" | "persistent@1" | "det_persistent@1");
        if registration != self.slice.in_init {
            return Err(Trap::new(
                TrapCode::PhaseViolation,
                import,
                None,
                if registration {
                    "bridge registration imports are legal only during da_init (§2.5)"
                } else {
                    "non-registration bridge imports are illegal during da_init (§2.5)"
                },
            ));
        }
        // Ingest-phase tracking (§5.9 / IngestPhase docs): any bridge op that is not a pure
        // window reader marks the open staged window's ingest MATH as begun — the next window
        // (or batch read, or slice close) is then the epilogue boundary.
        if let IngestPhase::Consuming { math_seen: false } = self.ingest_phase {
            let window_reader = matches!(
                import,
                "upd_sections@1"
                    | "upd_kind@1"
                    | "upd_bytes_len@1"
                    | "upd_read_bytes@1"
                    | "upd_tensor@1"
                    | "drop@1"
                    | "metric@1"
                    | "log@1"
                    | "abi_minor@1"
            );
            if !window_reader {
                self.ingest_phase = IngestPhase::Consuming { math_seen: true };
            }
        }
        self.charge_op(import)
    }
    fn stash_trap(&mut self, t: Trap) {
        self.trap = Some(t);
    }
    /// Bridge nr-class results are journaled verbatim under the §2.7 reserved kinds (≥ 128) so
    /// input replay can feed them back without re-running native kernels.
    fn journal_bridge_nr(&mut self, kind: u32, value: &[u8]) {
        let shared = self.shared.clone();
        let mut st = shared.state.lock().expect("pump lock");
        // A sink failure here is a host fault; surface it as a stashed trap on the next boundary
        // rather than swallowing (journaling is load-bearing, §8.4).
        if let Err(e) = st.sink.read_back(0, u64::from(kind), 0, value) {
            drop(st);
            self.trap = Some(Trap::bare(TrapCode::BadModule, e.to_string()));
        }
    }
}

fn journal_terminal_trap(shared: &Arc<PumpShared>, trap: &Trap) -> Result<(), SinkError> {
    let mut st = shared.state.lock().expect("pump lock");
    st.sink.terminal(
        1,
        None,
        Some((
            trap.code.slug().to_string(),
            trap.import.to_string(),
            "da_run".to_string(),
            trap.detail.clone(),
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::journal::{MemorySink, SinkEntry};
    use daemon_vhc_proto::sign::verify_bytes;

    fn test_state(sink: Box<dyn JournalSink>) -> PumpState {
        PumpState {
            queue: VecDeque::new(),
            timers: Vec::new(),
            next_timer_id: 1,
            staged: std::collections::BTreeMap::new(),
            next_host_staging_id: 1,
            next_guest_staging_id: 1,
            sink,
            timer_depth: 2,
            payload_depth: 4,
            gossip_depth: 2,
            spool_frames: 4,
            per_sender_quota: 2,
            auth_spooled: 0,
            auth_per_sender: std::collections::HashMap::new(),
            spool_exhausted_reported: false,
            gossip_arrivals: std::collections::HashMap::new(),
            metrics: Vec::new(),
            logs: Vec::new(),
            published: Vec::new(),
            bridge_final_state: None,
            buffers: BufferTable::new(0, 0, 0),
            ops: OpTable::new(0, 0),
            streams: StreamTable::new(0),
            op_requests: Vec::new(),
            stop_enqueued: false,
            draining: false,
        }
    }

    fn test_pump(sink: Box<dyn JournalSink>) -> PumpHandle {
        PumpHandle {
            shared: Arc::new(PumpShared {
                state: Mutex::new(test_state(sink)),
                wake: Condvar::new(),
                t0: Instant::now(),
                hold: AtomicBool::new(false),
            }),
        }
    }

    fn signed_stub() -> Vec<u8> {
        b"signed-frame-stub".to_vec()
    }

    #[test]
    fn authoritative_spool_backpressures_and_journals_the_typed_stall() {
        // test_state: spool_frames = 4, per_sender_quota = 2 (§4.7: bounded, never drops).
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink.clone()));
        let s1 = [1u8; 32];
        let s2 = [2u8; 32];
        let s3 = [3u8; 32];
        // Per-sender quota: sender 1's third undelivered frame back-pressures HIM only.
        assert_eq!(
            pump.deliver_frame(0, 0, s1, b"a".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::Accepted
        );
        assert_eq!(
            pump.deliver_frame(0, 1, s1, b"b".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::Accepted
        );
        assert_eq!(
            pump.deliver_frame(0, 2, s1, b"c".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::SenderQuota,
            "per-sender quota bounds the DoS vector (§4.7)"
        );
        // Other senders proceed until the SPOOL bound (4).
        assert_eq!(
            pump.deliver_frame(0, 0, s2, b"d".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::Accepted
        );
        assert_eq!(
            pump.deliver_frame(0, 1, s2, b"e".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::Accepted
        );
        assert_eq!(
            pump.deliver_frame(0, 0, s3, b"f".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::SpoolFull,
            "genuine spool exhaustion back-pressures (never a drop)"
        );
        // The typed stall was journaled ONCE for the episode (§6.7 tag 16), even on a re-hit.
        assert_eq!(
            pump.deliver_frame(0, 0, s3, b"f".to_vec(), signed_stub())
                .unwrap(),
            DeliverVerdict::SpoolFull
        );
        let entries = &sink.lock().unwrap().entries;
        let stalls: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e, SinkEntry::Condition { code, .. } if code == "SpoolExhausted"))
            .collect();
        assert_eq!(stalls.len(), 1, "one condition per exhaustion episode");
        // Nothing was dropped: the reliable class holds every accepted frame.
        assert!(!entries.iter().any(|e| matches!(e, SinkEntry::Drop { .. })));
    }

    #[test]
    fn gossip_class_drops_oldest_at_depth_and_journals_identity() {
        // test_state: gossip_depth = 2 (§4.7 drop-oldest, journaled tag 7 class 2).
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink.clone()));
        let g = [9u8; 32];
        pump.deliver_gossip(5, g, b"g0".to_vec()).unwrap();
        pump.deliver_gossip(5, g, b"g1".to_vec()).unwrap();
        pump.deliver_gossip(5, g, b"g2".to_vec()).unwrap();
        let entries = &sink.lock().unwrap().entries;
        let drops: Vec<&SinkEntry> = entries
            .iter()
            .filter(|e| matches!(e, SinkEntry::Drop { class: 2, .. }))
            .collect();
        assert_eq!(drops.len(), 1, "third arrival drops the OLDEST");
        let SinkEntry::Drop { rule, dropped, .. } = drops[0] else {
            unreachable!()
        };
        assert_eq!(*rule, daemon_vhc_abi::COALESCE_DROP_OLDEST);
        assert_eq!(
            (dropped.channel, dropped.sender, dropped.seq),
            (Some(5), Some(g), Some(0)),
            "the drop names the oldest arrival's full identity"
        );
    }

    #[test]
    fn payload_ready_dedups_by_hash_and_bounds_depth() {
        // test_state: payload_depth = 4 (§4.7 class 0: dedup-by-hash + bounded queue).
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink.clone()));
        // Identical bytes coalesce: one announcement, one journaled dedup.
        let id1 = pump.stage_payload(b"same".to_vec(), None).unwrap();
        let id2 = pump.stage_payload(b"same".to_vec(), None).unwrap();
        assert_eq!(id1, id2, "dedup returns the already-staged id");
        // Distinct hashes beyond the depth drop the OLDEST announcement (and unstage it).
        for i in 0u8..4 {
            pump.stage_payload(vec![i], None).unwrap();
        }
        let entries = &sink.lock().unwrap().entries;
        let dedups = entries
            .iter()
            .filter(
                |e| matches!(e, SinkEntry::Drop { class: 0, rule, .. } if *rule == daemon_vhc_abi::COALESCE_DEDUP_HASH),
            )
            .count();
        assert!(
            dedups >= 2,
            "the dedup + the depth drop are journaled: {entries:?}"
        );
    }

    #[test]
    fn due_timers_fire_in_deterministic_fire_at_then_id_order() {
        let mut st = test_state(Box::new(MemorySink::new()));
        st.timer_depth = 16;
        st.timers = vec![
            ArmedTimer { id: 3, fire_at: 10 },
            ArmedTimer { id: 1, fire_at: 10 },
            ArmedTimer { id: 2, fire_at: 5 },
            ArmedTimer { id: 4, fire_at: 99 }, // not due
        ];
        fire_due_timers(&mut st, 20).unwrap();
        let fired: Vec<u64> = st.queue.iter().filter_map(|q| q.timer_id).collect();
        assert_eq!(fired, vec![2, 1, 3], "(fire_at, id) ascending");
        assert_eq!(st.timers.len(), 1, "undue timer stays armed");
    }

    #[test]
    fn timer_queue_depth_drops_oldest_and_journals_it() {
        let mut st = test_state(Box::new(MemorySink::new()));
        st.timers = vec![
            ArmedTimer { id: 1, fire_at: 1 },
            ArmedTimer { id: 2, fire_at: 2 },
            ArmedTimer { id: 3, fire_at: 3 },
        ];
        // Depth 2: firing all three drops the oldest queued Timer (id 1), journaled (§4.7).
        fire_due_timers(&mut st, 10).unwrap();
        let queued: Vec<u64> = st.queue.iter().filter_map(|q| q.timer_id).collect();
        assert_eq!(queued, vec![2, 3]);
    }

    #[test]
    fn signed_frame_carries_the_full_scope_tuple_and_verifies() {
        // §12.1: [envelope, payload, sig]; the signature over the canonical envelope; every scope
        // field host-built. Verify with the plain proto primitives a third party would use.
        let signing = SigningKey::from_bytes(&[9u8; 32]);
        let sender = peer_id(&signing).0;
        let host = V2Host {
            shared: Arc::new(PumpShared {
                state: Mutex::new(test_state(Box::new(MemorySink::new()))),
                wake: Condvar::new(),
                t0: Instant::now(),
                hold: AtomicBool::new(false),
            }),
            limits: StoreLimitsBuilder::new().build(),
            trap: None,
            slice: SliceState {
                in_init: false,
                stopped: false,
                draining: false,
                now: 0,
                op_calls: 0,
                readback_bytes: 0,
                pending_next: None,
                pending_readback: None,
                pending_readback_value: None,
                pending_device: None,
            },
            fuel_per_slice: 0,
            op_budget: 0,
            epoch_ticks: 1,
            max_readback_bytes: 0,
            max_frame_bytes: 0,
            hard_accountable_host_bytes: 0,
            accountable_staged_bytes: 0,
            tabi: None,
            slice_pass_open: false,
            ingest_phase: IngestPhase::Idle,
            sealed_containers: 0,
            compute: None,
            compute_queue_depth: 0,
            compute_ops_since_fence: 0,
            compute_fault_after_ops: None,
            compute_ops_total: 0,
            signing,
            rng_seed: [0u8; 32],
            device_bytes: Vec::new(),
            granted_artifacts: std::collections::BTreeSet::new(),
            identity: RunIdentity {
                run_id: [1u8; 32],
                epoch: 4,
                role: "trainer".into(),
                instance: 7,
                module: [2u8; 32],
            },
            sender,
        };
        let payload = b"opaque-payload";
        let frame = build_signed_frame(&host, 0, 42, payload).unwrap();

        let v: Value = ciborium::de::from_reader(frame.as_slice()).unwrap();
        let Value::Array(parts) = v else {
            panic!("frame is [envelope, payload, sig]")
        };
        assert_eq!(parts.len(), 3);
        let Value::Map(env) = &parts[0] else {
            panic!("envelope is a map")
        };
        let get = |k: &str| {
            env.iter()
                .find(|(key, _)| matches!(key, Value::Text(t) if t == k))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("envelope field {k}"))
        };
        assert_eq!(get("domain"), Value::from(FRAME_ENVELOPE_DOMAIN_V2));
        assert_eq!(get("epoch"), Value::from(4u64));
        assert_eq!(get("instance"), Value::from(7u64));
        assert_eq!(get("channel"), Value::from(0u64));
        assert_eq!(get("seq"), Value::from(42u64));
        assert_eq!(
            get("payload_hash"),
            Value::Bytes(blake3::hash(payload).as_bytes().to_vec())
        );
        // The payload is carried verbatim; the signature verifies over the canonical envelope.
        assert_eq!(parts[1], Value::Bytes(payload.to_vec()));
        let Value::Bytes(sig) = &parts[2] else {
            panic!("sig bytes")
        };
        let env_bytes = to_canonical_vec(&parts[0]).unwrap();
        let sig64: [u8; 64] = sig.as_slice().try_into().unwrap();
        verify_bytes(
            &daemon_vhc_proto::PeerId(sender),
            &daemon_vhc_proto::Signature(sig64),
            &env_bytes,
        )
        .expect("§12.1 signature verifies");
    }
}
