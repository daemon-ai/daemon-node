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
//! - The `tabi@1` compute bridge is **not yet wired** into this driver: a major-2 module that
//!   imports `tabi@1` is refused [`V2Error::BridgeUnwired`] at [`start_run`] (a clear, typed,
//!   pre-instantiation error). Selection (§1.3) already admits such modules as major-2 candidates;
//!   the bridge dispatch rides the choreography sitting, where the first bridge consumer
//!   (TinyLlama on `BarrierRound`) arrives. Escalated in the sitting report.
//! - `snapshot_state` returns `SectionMissing` during a drain (no state-manifest verification yet
//!   — the §10.2 protocol lands with the migrate scaffolding); outside a drain it traps
//!   `PhaseViolation` per §6.6. `stage_state` is fully functional.
//! - Inbound frames arrive through [`PumpHandle::deliver_frame`] pre-verified: signature
//!   verification/dedup/gap detection are the session pump's admission-side jobs (the
//!   choreography sitting); this driver journals the original signed frame it is handed (§8.6).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ciborium::value::Value;
use wasmtime::{Caller, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder};

use daemon_vhc_abi::{
    pack_status_len, CHANNEL_DIR_RX_ONLY, EV_TAG_FRAME, EV_TAG_STOP, FRAME_ENVELOPE_DOMAIN_V2,
    NS_NET_V2, NS_SYS_V2, NS_TABI_V1, NS_VHC_V2, PHASE_A_DEFAULT_CHANNEL_TABLE,
    READBACK_KIND_STAGED_BYTES, RET_STATUS_DELIVERED, RET_STATUS_NEED_CAPACITY,
    SNAPSHOT_STATE_SECTION_MISSING, STAGED_KIND_BYTES,
};
use daemon_vhc_proto::{peer_id, sign_canonical, to_canonical_vec, SigningKey};

use crate::runtime::{EngineConfig, Worker};
use crate::trap::{Trap, TrapCode};
use crate::v2::event::{encode_event_frame, EventV2, PayloadMeta};
use crate::v2::journal::{JournalSink, SinkError};

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
    /// Bounded advisory queue depth (manifest-declared once the funnel lands; a default here).
    pub advisory_depth: usize,
    /// The claim's **hard-accountable host-tier cap** in raw bytes (`0` = uncapped): the
    /// enforceable tier the host meters EXACTLY (ABI §9.1). At Phase A the metered
    /// guest-attributable allocations are the staged bytes (`stage_state`); tensors/buffers join
    /// the meter with the bridge/Phase-B buffer layer. Breach is the typed attributable
    /// `BudgetMemory` trap — the under-claim acceptance (refactor §5 A2). The admission funnel
    /// (`v2::admission`) supplies this from the evaluated claim.
    pub hard_accountable_host_bytes: u64,
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
            hard_accountable_host_bytes: 0,
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
    /// The module imports `tabi@1` under major 2, and this driver has not wired the bridge
    /// dispatch yet (a deliberate Phase-A bound — see the module docs; the bridge lands with the
    /// choreography move).
    #[error("tabi@1 bridge not wired into the v2 driver yet: {0}")]
    BridgeUnwired(String),
    /// A journal-sink write failed (journaling is load-bearing, §8.4).
    #[error(transparent)]
    Sink(#[from] SinkError),
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
    /// Egress captured for the embedder (metrics/log are not journaled — outputs, not inputs).
    metrics: Vec<(String, f64)>,
    logs: Vec<(u32, String)>,
    /// Published frames, for embedder-side assertions: `(channel, seq, signed frame bytes)`.
    published: Vec<(u64, u64, Vec<u8>)>,
    /// A `Stop` has been enqueued — no further deliveries will be accepted after it.
    stop_enqueued: bool,
    /// A `Quiesce` drain is open: Frame/PayloadReady/Timer deliveries are frozen (§4.4).
    draining: bool,
}

struct PumpShared {
    state: Mutex<PumpState>,
    wake: Condvar,
    /// Logical time zero = pump creation (≈ run join / journal open, §6.5).
    t0: Instant,
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
    /// Deliver a **pre-verified** authoritative control frame (see module docs): `payload` is the
    /// module-authored bytes; `original_signed_frame` is the complete signed wire frame journaled
    /// as tag-12 evidence (§8.6).
    pub fn deliver_frame(
        &self,
        channel: u32,
        seq: u64,
        sender: [u8; 32],
        payload: Vec<u8>,
        original_signed_frame: Vec<u8>,
    ) -> Result<(), SinkError> {
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
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: EV_TAG_FRAME,
            signed: Some((u64::from(channel), seq, sender, original_signed_frame)),
            payload_hash: None,
            timer_id: None,
            is_budget: false,
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
            st.sink.drop_coalesced(0, 0, None, Some(hash))?;
            // The staged bytes are already announced; find its id for the caller.
            if let Some((&id, _)) = st
                .staged
                .iter()
                .find(|(_, (k, b))| *k == STAGED_KIND_BYTES && blake3::hash(b).as_bytes() == &hash)
            {
                return Ok(id);
            }
        }
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
            st.sink.drop_coalesced(3, 1, None, None)?;
        }
        st.queue.push_back(QueuedEvent {
            frame_bytes,
            tag: daemon_vhc_abi::EV_TAG_BUDGET,
            signed: None,
            payload_hash: None,
            timer_id: None,
            is_budget: true,
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
        });
        drop(st);
        self.shared.wake.notify_all();
        Ok(())
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
    // signing (§12.1)
    signing: SigningKey,
    identity: RunIdentity,
    sender: [u8; 32],
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

// -- the v2 linker ----------------------------------------------------------------------------------

/// How long a parked `next_event` waits between wake checks when no timer bounds the wait.
const PARK_RECHECK: Duration = Duration::from_millis(50);

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
                let shared = c.data().shared.clone();
                // Park until an event is deliverable, firing due timers ourselves (§6.3).
                let (frame, at) = {
                    let mut st = shared.state.lock().expect("pump lock");
                    loop {
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
                if kind != READBACK_KIND_STAGED_BYTES {
                    // Bridge kinds (1/2) arrive with the bridge; state-section (3) with migrate.
                    return Err(Trap::new(
                        TrapCode::ReadBackUnavailable,
                        "read_back",
                        None,
                        format!("kind {kind} stages nothing in this Phase-A driver"),
                    ));
                }
                let shared = c.data().shared.clone();
                let value = {
                    let st = shared.state.lock().expect("pump lock");
                    match st.staged.get(&src) {
                        Some((k, bytes)) if *k == STAGED_KIND_BYTES => bytes.clone(),
                        _ => {
                            return Err(Trap::new(
                                TrapCode::ReadBackUnavailable,
                                "read_back",
                                None,
                                format!("staging id {src} names nothing stageable"),
                            ))
                        }
                    }
                };
                let len = value.len() as u64;
                if len > u64::from(out_cap) {
                    c.data_mut().slice.pending_readback = Some((src, kind, len));
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
                c.data_mut().slice.pending_readback = None;
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
                let dropped = st.queue.remove(pos).and_then(|q| q.timer_id);
                st.sink
                    .drop_coalesced(1, 1, dropped, None)
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
/// The caller has already run ABI §1.3 selection (`select_driver` → `CandidateDriver::V2`);
/// this function additionally refuses `tabi@1`-importing modules ([`V2Error::BridgeUnwired`] —
/// see the module docs).
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
    // Deliberate Phase-A bound: the bridge dispatch is not yet generic over the v2 store.
    let bridge_imports: Vec<String> = module
        .imports()
        .filter(|i| i.module() == NS_TABI_V1)
        .map(|i| i.name().to_string())
        .collect();
    if !bridge_imports.is_empty() {
        return Err(V2Error::BridgeUnwired(format!(
            "module imports {} tabi@1 symbol(s) (e.g. `{}`); the bridge lands with the \
             choreography move",
            bridge_imports.len(),
            bridge_imports[0]
        )));
    }

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
        false,
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
            metrics: Vec::new(),
            logs: Vec::new(),
            published: Vec::new(),
            stop_enqueued: false,
            draining: false,
        }),
        wake: Condvar::new(),
        t0: Instant::now(),
    });
    let pump = PumpHandle {
        shared: shared.clone(),
    };

    let mut linker: Linker<V2Host> = Linker::new(worker.engine());
    link_v2(&mut linker).map_err(|e| V2Error::Sandbox(e.to_string()))?;

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
                },
                fuel_per_slice: engine_cfg.fuel_per_call,
                op_budget: engine_cfg.op_budget,
                epoch_ticks,
                max_readback_bytes: run.max_readback_bytes_per_slice,
                max_frame_bytes: run.max_frame_bytes,
                hard_accountable_host_bytes: run.hard_accountable_host_bytes,
                accountable_staged_bytes: 0,
                signing,
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
            match da_run.call(&mut store, ()) {
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
    use crate::v2::journal::MemorySink;
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
            metrics: Vec::new(),
            logs: Vec::new(),
            published: Vec::new(),
            stop_enqueued: false,
            draining: false,
        }
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
            },
            fuel_per_slice: 0,
            op_budget: 0,
            epoch_ticks: 1,
            max_readback_bytes: 0,
            max_frame_bytes: 0,
            hard_accountable_host_bytes: 0,
            accountable_staged_bytes: 0,
            signing,
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
