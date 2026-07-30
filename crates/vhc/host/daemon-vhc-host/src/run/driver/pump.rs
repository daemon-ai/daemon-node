// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The event pump — the bounded, condvar-signalled queue between the embedder and the guest
//! thread (ABI §11.2), plus its §4.7 class policies: authoritative spool bounds + per-sender
//! quotas (back-pressure, never drops), advisory dedup/drop-oldest/latest-wins coalescing
//! (journaled), timer firing in deterministic `(fire_at, timer_id)` order (§6.3), and the
//! [`PumpHandle`] the embedder drives (deliveries, staging, completions, stop/quiesce, egress).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use daemon_vhc_abi::{
    COMP_ERR_GRANT_EXHAUSTED, COMP_ERR_HASH_MISMATCH, EV_TAG_FRAME, EV_TAG_STOP, STAGED_KIND_BYTES,
};

use crate::run::buffer::BufferTable;
use crate::run::completion::{CompError, CompletionResult, SuccessPayload};
use crate::run::driver::chunks::verify_covering_span;
use crate::run::driver::config::{DeliverVerdict, OpOutcome, SnapshotCapture, SpooledFrame};
use crate::run::event::{encode_event_frame, PayloadMeta, RunEvent};
use crate::run::journal::{JournalSink, SinkError};
use crate::run::ops::{OpRequest, OpTable};
use crate::run::streams::StreamTable;
use crate::trap::{Trap, TrapCode};

/// One queued, not-yet-delivered event: the decoded event plus its **frozen** frame encoding
/// (encoded once at enqueue, so a `NeedCapacity` retry sees byte-identical length/content) and,
/// for authoritative frames, the original signed wire frame for tag 12.
pub(crate) struct QueuedEvent {
    pub(crate) frame_bytes: Vec<u8>,
    pub(crate) tag: u64,
    /// `(channel, seq, sender, original signed frame)` for tag-12 evidence journaling.
    pub(crate) signed: Option<(u64, u64, [u8; 32], Vec<u8>)>,
    /// Advisory dedup key for `PayloadReady` (the staged hash).
    pub(crate) payload_hash: Option<[u8; 32]>,
    /// The timer id for a queued `Timer` event (depth accounting + cancel-of-queued, §4.7/§6.3).
    pub(crate) timer_id: Option<u64>,
    /// `Budget` marker (host-fixed depth 1, latest-wins).
    pub(crate) is_budget: bool,
    /// Advisory gossip-class identity `(channel, arrival seq, sender)` — present iff this is a
    /// gossip frame (drop-oldest accounting + the tag-7 drop identity, §4.7).
    pub(crate) gossip_id: Option<(u32, u64, [u8; 32])>,
}

pub(crate) struct ArmedTimer {
    pub(crate) id: u64,
    pub(crate) fire_at: u64,
}

pub(crate) struct PumpState {
    pub(crate) queue: VecDeque<QueuedEvent>,
    pub(crate) timers: Vec<ArmedTimer>,
    pub(crate) next_timer_id: u64,
    /// Host-announced staged payloads: `staging_id → (kind, bytes)`. Guest-created (`stage_state`)
    /// entries carry the §10.2 top bit.
    pub(crate) staged: std::collections::BTreeMap<u64, (u64, Vec<u8>)>,
    pub(crate) next_host_staging_id: u64,
    pub(crate) next_guest_staging_id: u64,
    pub(crate) sink: Box<dyn JournalSink>,
    /// The advisory `Timer`-queue depth (manifest-declared once the funnel lands; §4.7).
    pub(crate) timer_depth: usize,
    /// The advisory `PayloadReady`-queue depth (§4.7 dedup-by-hash class).
    pub(crate) payload_depth: usize,
    /// The advisory gossip-class queue depth (§4.7 drop-oldest).
    pub(crate) gossip_depth: usize,
    /// Authoritative spool bound (`spool_frames`, §4.7) + per-sender outstanding quota.
    pub(crate) spool_frames: usize,
    pub(crate) per_sender_quota: usize,
    /// Undelivered authoritative frames (the spool occupancy).
    pub(crate) auth_spooled: usize,
    /// Undelivered authoritative frames per sender (the quota ledger).
    pub(crate) auth_per_sender: std::collections::HashMap<[u8; 32], usize>,
    /// Whether the current spool-exhaustion episode was already journaled (§6.7, once per
    /// episode; cleared when the spool drains below the bound).
    pub(crate) spool_exhausted_reported: bool,
    /// Per-channel arrival counters for advisory (gossip-class) frames — advisory channels have
    /// no durable sequence semantics (§4.7); this dense arrival ordinal fills the frame's `seq`
    /// field deterministically for replay (the journaled delivered sequence is authoritative).
    pub(crate) gossip_arrivals: std::collections::HashMap<u32, u64>,
    /// Egress captured for the embedder (metrics/log are not journaled — outputs, not inputs).
    pub(crate) metrics: Vec<(String, f64)>,
    pub(crate) logs: Vec<(u32, String)>,
    /// The last log line the guest stamped with [`daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX`] — the
    /// SDK panic hook's forwarded message, held aside so the trap that follows a beat later can
    /// carry it as its typed detail instead of reaching the embedder as a bare `unreachable`.
    /// The forwarded guest-panic line, **tagged with the execution context it was emitted in**
    /// (`[LX-10]`).
    ///
    /// Emitting a prefixed line does not imply trapping: a guest may log one and continue, or log
    /// one during initialization and trap much later for an unrelated reason. An unscoped slot lifts
    /// that stale line into the later trap's detail, producing an authoritative-looking source
    /// location that belongs to a different failure — a diagnostic that names a different bug
    /// convincingly, which is worse than no diagnostic. So the context travels with the message and
    /// the lift happens only on an exact match.
    pub(crate) guest_panic: Option<(daemon_vhc_abi::ExecutionContext, String)>,
    /// Published frames, for embedder-side assertions: `(channel, seq, signed frame bytes)`.
    pub(crate) published: Vec<(u64, u64, Vec<u8>)>,
    /// The per-instance buffer table (kind 8, architecture §3.4) — shared between the guest
    /// thread's imports and completion-arrival minting, behind this pump lock.
    pub(crate) buffers: BufferTable,
    /// The outstanding-op table (kind 10, ABI §7.5).
    pub(crate) ops: OpTable,
    /// The registered chunk maps (the chunk-addressed corpus contract): fold identity → the
    /// module-registered, host-verified chunk map. Registration is deterministic guest output
    /// (`data@2::register_chunks` re-derives the fold and admits only granted identities), so
    /// this table is replay-reconstructible and carries no journal record.
    pub(crate) chunk_maps: std::collections::HashMap<[u8; 32], daemon_vhc_proto::ChunkMap>,
    /// The registered **det-state** chunk maps ([SF-R2], ABI §12.14): fold identity → the
    /// length-aware map a module registered for an externally-sourced family fold (artifact-form
    /// init, restore roots). Per-parameter chunking is not a uniform grid, so these carry explicit
    /// per-chunk lengths and `fetch` resolves the covering span by walking actual offsets. Like
    /// `chunk_maps` this is deterministic guest output (`register_state_chunks` re-derives the fold
    /// and admits only granted identities) — replay-reconstructible, no journal record.
    pub(crate) state_chunk_maps:
        std::collections::HashMap<[u8; 32], daemon_vhc_proto::det_state::DetStateChunkMap>,
    /// The cumulative `data@2` read budget (raw bytes; `0` = unbounded) + its ledger. Charged
    /// at the fetch CALL from the requested range — guest-call-order deterministic.
    pub(crate) data_read_budget: u64,
    pub(crate) data_read_used: u64,
    /// The stream table (kind 9, §3.3 credit flow control).
    pub(crate) streams: StreamTable,
    /// The host state store (ABI §12.14): chunk-addressed canonical det-lane state the guest
    /// writes via `state_open`/`state_emit`/`state_seal` and reads via `data@2::fetch`
    /// ([SF-R1] — a self-sealed fold is fetchable by construction, ahead of the grant check).
    /// Instance-scoped by construction: a restart starts empty, so torn (unsealed) folds can
    /// never survive a crash ([SF-4]).
    pub(crate) state: crate::run::state_store::StateStore,
    /// The high-water mark of the guest's linear memory in bytes, sampled at the `next_event` seam
    /// (wasm memory never shrinks, so this IS the run's peak residency). The measured half of the
    /// module's claimed host-accountable footprint: a gate asserts the measurement against the
    /// admitted claim instead of inferring the footprint from the absence of a trap.
    pub(crate) guest_memory_high_water: u64,
    /// Backend-allocator occupancy readings, one per phase boundary, in the order they were taken.
    ///
    /// The measurement every workspace, pooling, compilation and staging term in a Backend Execution
    /// Profile is calibrated against, and which nothing in this tree took before — so no run left an
    /// allocator record on any backend. Kept in order because the SHAPE across boundaries is the
    /// evidence: a pool that never returns memory to the driver looks identical to one that does, at
    /// any single point.
    pub(crate) allocator_samples:
        Vec<(crate::compute::SamplePoint, crate::compute::AllocatorSample)>,
    /// Open incremental buffer streams (`buffer_open`/`buffer_append`/`buffer_seal`): the host-side
    /// accumulation a guest builds a large sealed buffer through without ever holding it whole.
    pub(crate) buffer_streams: BufferStreams,
    /// Requests awaiting the embedder (the async-runtime bridge): `(op, request)` in issue order.
    pub(crate) op_requests: Vec<(u64, OpRequest)>,
    /// A `Stop` has been enqueued — no further deliveries will be accepted after it.
    pub(crate) stop_enqueued: bool,
    /// A registered stop cut (`PumpHandle::stop_at_publishes`): when the guest's total publish
    /// count reaches `.0`, a `Stop{.1}` enqueues atomically with that publish's commit — stop
    /// intent registered at a deterministic point in the guest's own output stream (§4.4).
    pub(crate) stop_cut: Option<(usize, u64)>,
    /// A `Quiesce` drain is open: Frame/PayloadReady/Timer deliveries are frozen (§4.4).
    pub(crate) draining: bool,
    /// The drain's wall-clock deadline (§4.4/§11.3), as logical pump time: `now_ms()` at the
    /// `quiesce()` registration plus its `deadline_ms`. A guest still pulling past it is forcibly
    /// interrupted with the typed `QuiesceDeadlineExceeded` trap. Live-pump enforcement only —
    /// replay has no wall clock; the trap lands in the journal as the tag-9 terminal fact.
    pub(crate) drain_deadline_at: Option<u64>,
    /// The snapshot `snapshot_state` accepted during this drain (§10.2): the upgrade transaction
    /// reads it through [`PumpHandle::snapshot_capture`]. At most one per drain (a second
    /// successful submission is `BadEvent`).
    pub(crate) accepted_snapshot: Option<SnapshotCapture>,
    /// The registered embedder egress wake ([`PumpHandle::set_egress_hook`]): fired whenever
    /// guest egress lands (a publish, an op request awaiting service, a metric, a log line) so
    /// an async embedder can wait for a wake instead of interval-polling `published()` /
    /// `take_op_requests()`. Host-internal — never a wire surface. The hook runs under the pump
    /// lock, so it MUST be wait-free and MUST NOT call back into the pump (a channel/notify
    /// signal, nothing more).
    pub(crate) egress_hook: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Set once `da_migrate` returned `Ready` on a migrating instance (§10.3 step 5): the
    /// embedder-visible VALIDATE marker the upgrade transaction gates activation on, independent
    /// of what (or whether) the migrated module publishes. Never set on a non-migrating start.
    pub(crate) migrate_validated: bool,
}

/// The open incremental buffer streams of one instance (ABI minor 4:
/// `buffer_open`/`buffer_append`/`buffer_seal`).
///
/// Deterministic by construction: stream ids are a per-instance counter, so a replay that
/// re-executes the same guest code opens the same ids in the same order — which is why the three
/// imports carry no journal record (the dc class, exactly like `create_from`).
#[derive(Debug, Default)]
pub(crate) struct BufferStreams {
    open: std::collections::BTreeMap<u64, Vec<u8>>,
    next: u64,
}

impl BufferStreams {
    /// Open a fresh stream and return its counter-deterministic id.
    pub(crate) fn open(&mut self) -> u64 {
        self.next += 1;
        self.open.insert(self.next, Vec::new());
        self.next
    }

    /// Append to an open stream; returns the stream's accumulated length.
    pub(crate) fn append(&mut self, stream: u64, bytes: &[u8]) -> Result<u64, String> {
        let buf = self
            .open
            .get_mut(&stream)
            .ok_or_else(|| format!("buffer stream {stream} is not open"))?;
        buf.extend_from_slice(bytes);
        Ok(buf.len() as u64)
    }

    /// Close a stream and take its bytes (the seal input). An unsealed stream at instance teardown
    /// simply drops — nothing durable was minted, the `state_open` torn-fold rule in miniature.
    pub(crate) fn take(&mut self, stream: u64) -> Result<Vec<u8>, String> {
        self.open
            .remove(&stream)
            .ok_or_else(|| format!("buffer stream {stream} is not open"))
    }
}

impl PumpState {
    /// Fire the registered egress wake, if any (see [`PumpHandle::set_egress_hook`]).
    pub(crate) fn note_egress(&self) {
        if let Some(hook) = &self.egress_hook {
            hook();
        }
    }

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
            if let Ok(RunEvent::PayloadReady {
                staging_id, hash, ..
            }) = crate::run::event::decode_event_frame(&old.frame_bytes)
            {
                self.staged.remove(&staging_id);
                self.sink.drop_coalesced(
                    0,
                    daemon_vhc_abi::COALESCE_DEDUP_HASH,
                    crate::run::journal::Dropped::payload(hash),
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue + journal one completion (ABI §4.6/§7.5): the tag-14 record captures the
    /// nondeterministic ARRIVAL (result + order) before the event is deliverable; the frame then
    /// rides the ordinary event queue (tag 1 at delivery, §8.4 rule 4). Completions still deliver
    /// during a `Quiesce` drain (§4.4 "already-outstanding operations") but never after `Stop`.
    pub(crate) fn enqueue_completion(
        &mut self,
        op: u64,
        result: &CompletionResult,
    ) -> Result<(), SinkError> {
        if self.stop_enqueued {
            return Ok(()); // the host delivers no further events after Stop (§4.4)
        }
        let result_bytes = result.encode().map_err(|e| SinkError(e.to_string()))?;
        self.sink.completion(op, &result_bytes)?;
        let frame_bytes = encode_event_frame(&RunEvent::Completion {
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
    pub(crate) fn enqueue_fence(&mut self, fence_id: u64) -> Result<(), SinkError> {
        if self.stop_enqueued {
            return Ok(()); // the host delivers no further events after Stop (§4.4)
        }
        let frame_bytes = encode_event_frame(&RunEvent::Fence { fence_id })
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

    /// Register stop intent: enqueue the terminal `Stop{reason}` behind already-pending events
    /// and refuse all further deliveries (§4.4). Idempotent. Because timers fire only inside
    /// `next_event` under this same lock, no timer fire can enter the delivered (recorded)
    /// stream after this returns.
    pub(crate) fn enqueue_stop(&mut self, reason: u64) -> Result<(), SinkError> {
        if self.stop_enqueued {
            return Ok(());
        }
        let frame_bytes =
            encode_event_frame(&RunEvent::Stop { reason }).map_err(|e| SinkError(e.to_string()))?;
        self.stop_enqueued = true;
        self.stop_cut = None;
        self.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: EV_TAG_STOP,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: false,
            gossip_id: None,
        });
        Ok(())
    }
}

pub(crate) struct PumpShared {
    pub(crate) state: Mutex<PumpState>,
    pub(crate) wake: Condvar,
    /// Logical time zero = pump creation (≈ run join / journal open, §6.5).
    pub(crate) t0: Instant,
    /// Rig-controlled delivery hold (D2 back-pressure prerequisite; module docs on
    /// [`PumpHandle::hold`]). When set, the guest thread parks inside `next_event` and NO event is
    /// delivered — so the embedder can fill the authoritative spool to `SpoolFull`/`SenderQuota`
    /// deterministically, which a live guest (draining as fast as frames arrive) cannot force.
    pub(crate) hold: AtomicBool,
}

impl PumpShared {
    pub(crate) fn now_ms(&self) -> u64 {
        u64::try_from(self.t0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// The embedder's handle onto a running v2 instance: enqueue inbound events, stage payloads,
/// read egress. Clonable; all methods are non-blocking (bounded queue overflows coalesce/drop
/// advisory events per §4.7, journaled).
#[derive(Clone)]
pub struct PumpHandle {
    pub(crate) shared: Arc<PumpShared>,
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
        let ev = RunEvent::Frame {
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
                crate::run::journal::Dropped::gossip(u64::from(channel), sender, seq),
            )?;
            return Ok(());
        }
        let seq = {
            let c = st.gossip_arrivals.entry(channel).or_insert(0);
            let s = *c;
            *c += 1;
            s
        };
        let ev = RunEvent::Frame {
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
                    crate::run::journal::Dropped::gossip(u64::from(och), osender, oseq),
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
                crate::run::journal::Dropped::payload(hash),
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
        let ev = RunEvent::PayloadReady {
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

    /// Deliver a `Budget` notification (host-fixed depth 1, latest-wins — §4.3/§4.7).
    pub fn budget(
        &self,
        fuel: u64,
        mem: u64,
        paused: bool,
        duty_pct: u64,
        vram_cap_bytes: u64,
    ) -> Result<(), SinkError> {
        let ev = RunEvent::Budget {
            report: crate::run::event::BudgetReport {
                fuel,
                mem,
                throttle: crate::run::event::ThrottleReport {
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
                crate::run::journal::Dropped::default(),
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
    ///
    /// This registers stop intent at a **wall-clock** point: it races the guest's own event loop,
    /// so a due timer may still fire and be delivered before the intent lands. An embedder whose
    /// stop condition is "the guest published its Nth output" should register the intent at that
    /// cut instead ([`PumpHandle::stop_at_publishes`]) — the recorded stream is then deterministic.
    pub fn stop(&self, reason: u64) -> Result<(), SinkError> {
        let mut st = self.shared.state.lock().expect("pump lock");
        st.enqueue_stop(reason)?;
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Register stop intent at a deterministic cut in the guest's own output stream (§4.4): when
    /// the guest's total publish count reaches `publishes`, the terminal `Stop{reason}` enqueues
    /// atomically with that publish's commit — under the pump lock, before the publish import
    /// returns. Timers fire only inside `next_event` under the same lock, so no timer fire (nor
    /// any other new delivery) can enter the recorded stream between the cut and the `Stop`.
    ///
    /// This is the race-free form of "stop when the run completes": run completion is a fact of
    /// the guest's output, and observing it from the embedder thread ([`PumpHandle::published`] +
    /// [`PumpHandle::stop`]) loses the lock race to the guest's next timer under load. If the cut
    /// has already passed at registration time, the `Stop` enqueues immediately. Idempotent
    /// against an already-registered stop.
    ///
    /// # Errors
    /// A journal-sink/encode failure.
    pub fn stop_at_publishes(&self, publishes: usize, reason: u64) -> Result<(), SinkError> {
        let mut st = self.shared.state.lock().expect("pump lock");
        if st.stop_enqueued {
            return Ok(());
        }
        if st.published.len() >= publishes {
            st.enqueue_stop(reason)?;
        } else {
            st.stop_cut = Some((publishes, reason));
        }
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Open a `Quiesce{reason, deadline_ms}` drain (§4.4): new Frame/PayloadReady/Timer
    /// deliveries freeze (spool/coalesce); the guest is expected to return `QuiesceReady`.
    /// The snapshot `snapshot_state` accepted during this drain, if any (§10.2): the upgrade
    /// transaction's step-2 capture. `None` until the guest achieves one successful submission.
    #[must_use]
    pub fn snapshot_capture(&self) -> Option<SnapshotCapture> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .accepted_snapshot
            .clone()
    }

    /// Drain the authoritative frames that spooled undelivered during the Quiesce drain (§4.4),
    /// in arrival order — the upgrade transaction re-delivers them into the NEW instance's pump
    /// at activation (§10.3 step 6 "spooled frames drain into the new instance"). Each
    /// [`SpooledFrame`] carries exactly the [`PumpHandle::deliver_frame`] argument tuple.
    #[must_use]
    pub fn take_spooled_frames(&self) -> Vec<SpooledFrame> {
        let mut st = self.shared.state.lock().expect("pump lock");
        let mut out = Vec::new();
        let mut i = 0;
        while i < st.queue.len() {
            if st.queue[i].tag == EV_TAG_FRAME && st.queue[i].signed.is_some() {
                let ev = st.queue.remove(i).expect("index checked");
                let (channel, seq, sender, original) = ev.signed.expect("checked signed");
                st.auth_spooled = st.auth_spooled.saturating_sub(1);
                if let Some(n) = st.auth_per_sender.get_mut(&sender) {
                    *n = n.saturating_sub(1);
                }
                let payload = match crate::run::event::decode_event_frame(&ev.frame_bytes) {
                    Ok(RunEvent::Frame { payload, .. }) => payload,
                    _ => Vec::new(),
                };
                out.push(SpooledFrame {
                    channel: u32::try_from(channel).unwrap_or(u32::MAX),
                    seq,
                    sender,
                    payload,
                    original_signed_frame: original,
                });
            } else {
                i += 1;
            }
        }
        out
    }

    /// Open a `Quiesce{reason, deadline_ms}` drain (§4.4): new Frame/Timer deliveries freeze
    /// (spool/coalesce); the guest is expected to snapshot and return `QuiesceReady`. The
    /// `deadline_ms` is not advisory: the pump enforces it wall-clock, and a guest still pulling
    /// past it is forcibly interrupted with the typed `QuiesceDeadlineExceeded` trap
    /// (§4.4/§11.3) — the upgrade orchestrator treats that as a failed quiesce and leaves.
    ///
    /// # Errors
    /// A journal-sink/encode failure.
    pub fn quiesce(&self, reason: u64, deadline_ms: u64) -> Result<(), SinkError> {
        let frame_bytes = encode_event_frame(&RunEvent::Quiesce {
            reason,
            deadline_ms,
        })
        .map_err(|e| SinkError(e.to_string()))?;
        let mut st = self.shared.state.lock().expect("pump lock");
        st.draining = true;
        // Host-side enforcement of the deadline the event advertises (§4.4/§11.3): a guest that
        // has not returned from the drain by then is forcibly interrupted inside `next_event`
        // with the typed `QuiesceDeadlineExceeded` trap. The clock lives here, on the live pump
        // (`PumpShared::t0`-anchored logical ms) — never on the deterministic replay surface.
        st.drain_deadline_at = Some(self.shared.now_ms().saturating_add(deadline_ms));
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

    /// Re-pin the freshest (role,kind) checkpoint's referenced folds (design §8.2, C6): keep the
    /// current checkpoint's families exempt from retention eviction and release a superseded
    /// checkpoint's now-unreferenced folds. Driven by the checkpoint publication seam.
    pub fn repin_checkpoint(&self, folds: &[[u8; 32]]) {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .state
            .repin_checkpoint(folds);
    }

    /// The `(chunk hash, chunk bytes)` list of a fold this instance self-sealed ([SF-R1]) — the
    /// content a checkpoint publisher uploads to the payload plane, and what the golden harness
    /// reads from the draining instance to stand in for that plane. The refusal is typed
    /// ([`crate::run::state_store::SealedReadError`]) so the publisher can distinguish
    /// "not sealed here" from a custody hole — never a silent skip.
    ///
    /// # Errors
    /// See [`crate::run::state_store::StateStore::sealed_chunks`].
    pub fn sealed_fold_chunks(
        &self,
        fold: &[u8; 32],
    ) -> Result<Vec<(daemon_vhc_proto::Hash, Vec<u8>)>, crate::run::state_store::SealedReadError>
    {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .state
            .sealed_chunks(fold)
    }

    /// The registered **det-state** chunk map for `fold`, if the guest registered one ([SF-R2]).
    /// The chunk-keyed resolver ([SF-6] restore carriage) consults this — the single source of
    /// truth for a family's covering geometry — to decompose an `ArtifactRange` covering span into
    /// its constituent `(chunk hash, len)` list and fetch each chunk content-addressed, symmetric
    /// with the replay-side chunk-keyed materialization. Returns a clone (small: hashes + lengths),
    /// so the resolver holds no pump lock while it awaits the store.
    #[must_use]
    pub fn state_chunk_map(
        &self,
        fold: &[u8; 32],
    ) -> Option<daemon_vhc_proto::det_state::DetStateChunkMap> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .state_chunk_maps
            .get(fold)
            .cloned()
    }

    /// Lift a sealed det-state family out of this instance's store for carriage into a successor
    /// within the SAME node (the in-process live-module-switch transaction, [SF-6]). Called on the
    /// DRAINING instance while it is still alive (its store not yet dropped), so the successor can
    /// serve the drain snapshot's folds self-sealed ([SF-R1]) instead of fetching them from a
    /// content plane the in-process switch never published to. `None` if the fold is not sealed
    /// here.
    #[must_use]
    pub fn export_sealed_family(
        &self,
        fold: &[u8; 32],
    ) -> Option<crate::run::state_store::CarriedFamily> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .state
            .export_sealed_family(fold)
    }

    /// Whether a migrating instance's `da_migrate` returned `Ready` (§10.3 step 5) — the
    /// embedder-visible VALIDATE marker the upgrade transaction gates activation on. `false`
    /// until validation, and forever on a non-migrating instance.
    #[must_use]
    pub fn migrate_validated(&self) -> bool {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .migrate_validated
    }

    /// Register the embedder's **egress wake**: `hook` fires whenever guest egress lands — a
    /// publish, an op request awaiting service, a metric, or a log line — so an event-driven
    /// embedder (the role session) can await a wake instead of interval-polling
    /// [`PumpHandle::published`] / [`PumpHandle::take_op_requests`]. Host-internal plumbing,
    /// never a wire surface.
    ///
    /// The hook runs under the pump lock: it MUST be wait-free and MUST NOT call back into the
    /// pump (signal a channel/notify and return). It fires once at registration so egress that
    /// landed before the hook was set is never silently unannounced.
    pub fn set_egress_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        let mut st = self.shared.state.lock().expect("pump lock");
        st.egress_hook = Some(hook);
        st.note_egress();
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
            (
                OpRequest::ArtifactRange {
                    hash,
                    range_off,
                    range_len,
                    span_off,
                    span_len,
                },
                outcome @ (OpOutcome::RangeDone { .. } | OpOutcome::FetchDone { .. }),
            ) => {
                // The chunk-addressed verification rule: the pump — never the embedder —
                // verifies every covering chunk against the REGISTERED (fold-committed) chunk
                // map, then slices the guest's original range. A whole-object answer
                // (`FetchDone`, the in-process content-store seat) has its span extracted
                // first; a span answer (`RangeDone`) must be exactly the requested span.
                //
                // [SF-R2]: an externally-registered DET-STATE fold verifies LENGTH-AWARE against
                // its per-chunk offsets (per-parameter chunking is not a uniform grid). Same
                // covering-span discipline, walked over actual lengths rather than a grid.
                if let Some(smap) = st.state_chunk_maps.get(&hash).cloned() {
                    let span = match outcome {
                        OpOutcome::RangeDone { bytes } => {
                            if bytes.len() as u64 == span_len {
                                Ok(bytes)
                            } else {
                                Err(format!(
                                    "span answer is {} bytes, the covering span is {span_len}",
                                    bytes.len()
                                ))
                            }
                        }
                        OpOutcome::FetchDone { artifact } => {
                            if artifact.len() as u64 == smap.byte_len {
                                let s = span_off as usize;
                                Ok(artifact[s..s + span_len as usize].to_vec())
                            } else {
                                Err(format!(
                                    "whole-object answer is {} bytes, the registered fold is {}",
                                    artifact.len(),
                                    smap.byte_len
                                ))
                            }
                        }
                        _ => unreachable!("outer match arm admits only Range/FetchDone"),
                    };
                    match span.and_then(|bytes| {
                        smap.verify_covering_span(span_off, &bytes)
                            .map(|()| bytes)
                            .map_err(|e| e.to_string())
                    }) {
                        Err(detail) => CompletionResult::Err(CompError {
                            code: COMP_ERR_HASH_MISMATCH,
                            detail: Some(detail),
                        }),
                        Ok(span_bytes) => {
                            let end = if range_len == 0 {
                                smap.byte_len
                            } else {
                                range_off + range_len
                            };
                            let lo = (range_off - span_off) as usize;
                            let hi = lo + (end - range_off) as usize;
                            let slice = span_bytes[lo..hi].to_vec();
                            match st.buffers.create_host(Arc::new(slice)) {
                                Some(handle) => {
                                    minted = Some(handle);
                                    CompletionResult::Ok(SuccessPayload::Handle(handle))
                                }
                                None => CompletionResult::Err(CompError {
                                    code: COMP_ERR_GRANT_EXHAUSTED,
                                    detail: Some(
                                        "buffer quota exhausted (deny new buffers)".into(),
                                    ),
                                }),
                            }
                        }
                    }
                } else {
                    let map = st.chunk_maps.get(&hash).cloned();
                    match map {
                        None => CompletionResult::Err(CompError {
                            code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                            detail: Some(
                                "chunk map unregistered at completion (pump invariant)".into(),
                            ),
                        }),
                        Some(map) => {
                            let span = match outcome {
                                OpOutcome::RangeDone { bytes } => {
                                    if bytes.len() as u64 == span_len {
                                        Ok(bytes)
                                    } else {
                                        Err(format!(
                                            "span answer is {} bytes, the covering span is \
                                         {span_len}",
                                            bytes.len()
                                        ))
                                    }
                                }
                                OpOutcome::FetchDone { artifact } => {
                                    if artifact.len() as u64 == map.byte_len {
                                        let s = span_off as usize;
                                        Ok(artifact[s..s + span_len as usize].to_vec())
                                    } else {
                                        Err(format!(
                                            "whole-object answer is {} bytes, the registered \
                                         shard is {}",
                                            artifact.len(),
                                            map.byte_len
                                        ))
                                    }
                                }
                                _ => unreachable!("outer match arm admits only Range/FetchDone"),
                            };
                            match span.and_then(|bytes| verify_covering_span(&map, span_off, bytes))
                            {
                                Err(detail) => CompletionResult::Err(CompError {
                                    code: COMP_ERR_HASH_MISMATCH,
                                    detail: Some(detail),
                                }),
                                Ok(span_bytes) => {
                                    let end = if range_len == 0 {
                                        map.byte_len
                                    } else {
                                        range_off + range_len
                                    };
                                    let lo = (range_off - span_off) as usize;
                                    let hi = lo + (end - range_off) as usize;
                                    let slice = span_bytes[lo..hi].to_vec();
                                    match st.buffers.create_host(Arc::new(slice)) {
                                        Some(handle) => {
                                            minted = Some(handle);
                                            CompletionResult::Ok(SuccessPayload::Handle(handle))
                                        }
                                        None => CompletionResult::Err(CompError {
                                            code: COMP_ERR_GRANT_EXHAUSTED,
                                            detail: Some(
                                                "buffer quota exhausted (deny new buffers)".into(),
                                            ),
                                        }),
                                    }
                                }
                            }
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
        let any = !released.is_empty();
        for (op, bytes) in released {
            st.op_requests
                .push((op, OpRequest::StreamWrite { stream, bytes }));
        }
        if any {
            st.note_egress();
        }
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

    /// The host state store's introspection snapshot (ABI §12.14): sealed folds, open streams,
    /// live retained bytes — the torn-fold-GC / retention evidence surface (tests, the
    /// fleet-preflight disk/RAM line item).
    #[must_use]
    pub fn state_store_stats(&self) -> crate::run::state_store::StateStoreStats {
        self.shared.state.lock().expect("pump lock").state.stats()
    }

    /// The guest's peak linear memory in bytes, sampled at every event slice (wasm memory never
    /// shrinks). This is the MEASUREMENT behind a module's claimed host-accountable footprint — the
    /// number a real-geometry gate asserts stays under the admitted claim, and the number an
    /// honest `decl_for_config` is derived from in the first place.
    #[must_use]
    pub fn guest_memory_high_water(&self) -> u64 {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .guest_memory_high_water
    }
}

/// Move due timers into the queue as `Timer` events, in `(fire_at, timer_id)` order (§6.3).
pub(crate) fn fire_due_timers(st: &mut PumpState, now: u64) -> Result<(), Trap> {
    if !st.timers.iter().any(|t| t.fire_at <= now) {
        return Ok(());
    }
    let (mut fired, keep): (Vec<ArmedTimer>, Vec<ArmedTimer>) =
        st.timers.drain(..).partition(|t| t.fire_at <= now);
    st.timers = keep;
    // Deterministic firing order (§6.3).
    fired.sort_by_key(|t| (t.fire_at, t.id));
    for t in fired {
        let ev = RunEvent::Timer {
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
                        crate::run::journal::Dropped::timer(dropped),
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

impl PumpHandle {
    /// The backend-allocator occupancy readings taken at this instance's phase boundaries.
    ///
    /// Empty when this build and backend cannot report them, which is **not** the same as an
    /// instance that allocated nothing — a caller that conflated the two would be reporting a
    /// measurement it does not have.
    #[must_use]
    pub fn allocator_samples(
        &self,
    ) -> Vec<(crate::compute::SamplePoint, crate::compute::AllocatorSample)> {
        self.shared
            .state
            .lock()
            .expect("pump lock")
            .allocator_samples
            .clone()
    }
}
