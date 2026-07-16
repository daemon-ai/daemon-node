// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The worker input-replay execution engine (ABI companion §8.7; refactor §5 A1→A2): re-drive a
//! major-2 module from a recorded journal and observe every outbound decision it makes.
//!
//! This is the HOST half of the verifier: `daemon-vhc-observe`'s `journal::verifier` owns the
//! typed §8.7 contract (`ReplayPlan` / `run_replay` / `ReplayOutcome`) and stays wasm-free; this
//! module owns the wasm execution it drives — the dependency direction is observe → host
//! (dev-side today: the tier-1 lane adapts the two in `tests/v2_replay.rs`), and neither links
//! the SDK.
//!
//! **Replay semantics (the fixed A1 contract):** re-feeding the recorded inputs reproduces every
//! guest decision bit-for-bit. Inputs answered FROM THE JOURNAL, never re-executed:
//!
//! - delivered event frames (tag 1) — re-fed verbatim through `next_event`; **completion events
//!   (tag 6 frames) re-fed in journaled order** — completion order is a nondeterministic input
//!   (§8.1), and the journaled order IS the replayed order. A delivered completion retires its op;
//!   a `payload_get` success materializes its buffer from the script's content-addressed payload
//!   table (§8.7 re-fetch; a missing payload is the typed `ReplayMissingPayload` divergence);
//! - guest-requested staged read-backs (tag 2, kinds < 128) — the recorded value, after
//!   verifying the guest asked for the same `(src, kind)`;
//! - §2.7 bridge nr-class readouts (tag 2, kinds 128–136) — the recorded value, after verifying
//!   the guest called the same nr import;
//! - clock readings (tag 3) and timer arms/cancels (tags 5/6) — the recorded ids/values, after
//!   verifying the armed delay matches;
//! - buffers + `OpId`s (kinds 8/10) — re-derived from deterministic bookkeeping (the same
//!   §7.1-seeded arenas as the recording), never journaled; `cancel` returns are recordless-
//!   deterministic (the journaled completion result captures the race — `v2::ops` docs).
//!
//! Bridge COMPUTE imports (`ones@1`, `add@1`, `backward@1`, …) are pure state transformers whose
//! only guest-visible products are opaque handles and the nr readouts above; replay stubs them
//! with synthesized handles and re-executes no kernel (§8.7: "recorded results are replayed,
//! kernels are not re-executed"). A **decision** is a publish: `(channel, seq, payload hash)`,
//! attributed to the slice (delivered event) it happened in; the run's terminal outcome is
//! compared too. Any mismatch between what the guest asks/does and what the journal recorded is
//! a typed [`ReplayEnd::Diverged`] carrying the first divergence.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use wasmtime::{Caller, Extern, ExternType, Linker, Memory, Module, Store, Val};

use super::journal::SinkEntry;
use crate::runtime::Worker;
use crate::v2::buffer::BufferTable;
use crate::v2::completion::{CompletionResult, SuccessPayload};
use crate::v2::ops::{OpRequest, OpTable};
use crate::v2::V2Error;

/// The §2.7 nr-class bridge imports and their journal kinds (runtime.rs `journal_bridge_nr`).
const NR_IMPORTS: &[(&str, u64)] = &[
    ("scalar@1", 128),
    ("abi_minor@1", 129),
    ("batch_size@1", 130),
    ("batch_seq_len@1", 131),
    ("upd_sections@1", 132),
    ("upd_kind@1", 133),
    ("upd_bytes_len@1", 134),
    ("upd_read_bytes@1", 135),
    ("det_l2norm@1", 136),
];

/// The recorded inputs a replay re-feeds, split by answering mechanism.
#[derive(Debug, Default, Clone)]
pub struct ReplayScript {
    /// Delivered event frames (tag 1): `(at, frame bytes)` in delivery order.
    pub events: VecDeque<(u64, Vec<u8>)>,
    /// Guest-requested staged read-backs (tag 2, kind < 128): `(src, kind, value)`.
    pub readbacks: VecDeque<(u64, u64, Vec<u8>)>,
    /// Bridge nr-class readouts (tag 2, kind ≥ 128): `(kind, value)`.
    pub nr: VecDeque<(u64, Vec<u8>)>,
    /// Clock readings (tag 3).
    pub clocks: VecDeque<u64>,
    /// Timer arms (tag 5): `(id, delay_ms)`.
    pub timer_arms: VecDeque<(u64, u64)>,
    /// Timer cancels (tag 6): `(id, status)`.
    pub timer_cancels: VecDeque<(u64, u64)>,
    /// Device-profile deliveries (tag 15): the recorded profile bytes, in delivery order — replay
    /// feeds the recorded observation, never a fresh probe.
    pub device_profiles: VecDeque<Vec<u8>>,
    /// The execution identity behind the journal (the tag-0 run header): what `sys@2::rng_seed`
    /// re-derives from (the seed is deterministic — §2.7 dc class — so it is re-computed at
    /// replay, not recorded). `None` is fine for guests that never read the seed.
    pub identity: Option<super::driver::RunIdentity>,
    /// Content-addressed payload bytes by blake3 — the §8.7 "bulk payloads are not copied; they
    /// are content-addressed and re-fetched at replay" table. A `payload_get`-completed buffer's
    /// bytes come from here; a missing entry is the typed `ReplayMissingPayload` divergence,
    /// never a silent pass.
    pub payloads: HashMap<[u8; 32], Vec<u8>>,
}

impl ReplayScript {
    /// Split a recorded journal (sink mirror form) into the replay inputs.
    #[must_use]
    pub fn from_entries(entries: &[SinkEntry]) -> Self {
        let mut s = Self::default();
        for e in entries {
            match e {
                SinkEntry::Event { at, frame } => s.events.push_back((*at, frame.clone())),
                SinkEntry::ReadBack {
                    src, kind, value, ..
                } => {
                    if *kind >= 128 {
                        s.nr.push_back((*kind, value.clone()));
                    } else {
                        s.readbacks.push_back((*src, *kind, value.clone()));
                    }
                }
                SinkEntry::Clock { now } => s.clocks.push_back(*now),
                SinkEntry::TimerArm { id, delay, .. } => s.timer_arms.push_back((*id, *delay)),
                SinkEntry::TimerCancel { id, status } => {
                    s.timer_cancels.push_back((*id, *status));
                }
                SinkEntry::DeviceProfile { profile } => {
                    s.device_profiles.push_back(profile.clone());
                }
                _ => {}
            }
        }
        s
    }
}

/// One outbound decision the guest made during replay: a publish, attributed to its slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedDecision {
    /// Index of the delivered event whose slice produced this publish (0-based; publishes
    /// before the first delivery — none exist under §3.1 — would carry 0).
    pub event_index: usize,
    /// The channel published on.
    pub channel: u64,
    /// The durable channel-scoped seq the replay assigned (dense from 0, as the recording did).
    pub seq: u64,
    /// blake3 of the guest payload bytes.
    pub payload_hash: [u8; 32],
}

/// How the replayed run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayEnd {
    /// `da_run` returned this outcome code.
    Outcome(u32),
    /// `da_init` refused with this status.
    InitRefused(u32),
    /// The guest's requests stopped matching the journal — the first divergence, described.
    Diverged(String),
    /// The guest trapped for a non-divergence reason (wasm fault, missing export, …).
    Trapped(String),
}

/// The observable product of a replayed run.
#[derive(Debug, Clone)]
pub struct ReplayedRun {
    /// Every publish the guest made, in order, slice-attributed.
    pub decisions: Vec<ReplayedDecision>,
    /// How many recorded events were actually delivered.
    pub events_delivered: usize,
    /// How the run ended.
    pub end: ReplayEnd,
}

struct ReplayHost {
    script: ReplayScript,
    decisions: Vec<ReplayedDecision>,
    events_delivered: usize,
    /// NeedCapacity retry state for `next_event` (the frame must not be consumed twice).
    pending_event: Option<(u64, Vec<u8>)>,
    /// Synthesized opaque handles for stubbed bridge compute imports.
    next_handle: u64,
    /// Per-channel dense seq counters (mirrors the recording driver's allocation).
    seqs: std::collections::HashMap<u64, u64>,
    /// Guest-partition buffers: the SAME deterministic arena as the recording (quotas were
    /// enforced then; replay re-derives identical `create_from` handle values).
    buffers: BufferTable,
    /// The op mint (monotone indices — identical `OpId` values to the recording).
    ops: OpTable,
    /// `payload_get` op → requested hash (guest-authored, deterministic).
    op_hashes: HashMap<u64, [u8; 32]>,
    /// Host-partition (completion-minted) buffers, keyed by the handle the journaled completion
    /// frame carries, materialized from `script.payloads` at delivery.
    host_buffers: HashMap<u64, Arc<Vec<u8>>>,
}

fn diverged(msg: impl std::fmt::Display) -> wasmtime::Error {
    wasmtime::Error::msg(format!("{DIVERGENCE_MARKER}{msg}"))
}

/// Resolve a buffer in either replay partition: the deterministic guest arena (`create_from`) or
/// the completion-materialized host map.
fn resolve_replay_buffer(host: &ReplayHost, handle: u64) -> Option<Arc<Vec<u8>>> {
    host.buffers
        .resolve(handle)
        .ok()
        .or_else(|| host.host_buffers.get(&handle).cloned())
}

fn hex8(hash: &[u8; 32]) -> String {
    hash[..4].iter().map(|b| format!("{b:02x}")).collect()
}

const DIVERGENCE_MARKER: &str = "replay divergence: ";

fn mem_of(caller: &mut Caller<'_, ReplayHost>) -> Result<Memory, wasmtime::Error> {
    caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| wasmtime::Error::msg("module has no exported memory"))
}

fn p_u32(params: &[Val], i: usize) -> u32 {
    match params[i] {
        Val::I32(v) => v as u32,
        _ => 0,
    }
}

fn p_u64(params: &[Val], i: usize) -> u64 {
    match params[i] {
        Val::I64(v) => v as u64,
        _ => 0,
    }
}

fn packed(status: u64, len: usize) -> Val {
    Val::I64((((status << 32) | len as u64) as i64).to_owned())
}

/// Deliver `bytes` into `(ptr, cap)` with the §4.1 NeedCapacity protocol; on success runs `ok`.
fn deliver_span(
    caller: &mut Caller<'_, ReplayHost>,
    bytes: &[u8],
    ptr: u32,
    cap: u32,
    results: &mut [Val],
) -> Result<bool, wasmtime::Error> {
    if (cap as usize) < bytes.len() {
        results[0] = packed(1, bytes.len());
        return Ok(false);
    }
    let mem = mem_of(caller)?;
    mem.write(caller, ptr as usize, bytes)
        .map_err(|e| wasmtime::Error::msg(format!("span write: {e}")))?;
    results[0] = packed(0, bytes.len());
    Ok(true)
}

#[allow(clippy::too_many_lines)]
fn dispatch(
    caller: &mut Caller<'_, ReplayHost>,
    ns: &str,
    name: &str,
    params: &[Val],
    results: &mut [Val],
    result_types: &[wasmtime::ValType],
) -> Result<(), wasmtime::Error> {
    match (ns, name) {
        ("vhc@2", "next_event") => {
            let (ptr, cap) = (p_u32(params, 0), p_u32(params, 1));
            let host = caller.data_mut();
            let frame = match host.pending_event.take() {
                Some(f) => f,
                None => host.script.events.pop_front().ok_or_else(|| {
                    diverged("guest pulled an event beyond the recorded stream (tag 1 exhausted)")
                })?,
            };
            if deliver_span(caller, &frame.1.clone(), ptr, cap, results)? {
                caller.data_mut().events_delivered += 1;
                // A delivered Completion retires its op and, for a payload_get success,
                // materializes the completion-minted buffer from the content-addressed payload
                // table (§8.7 — bulk payloads are re-fetched, never copied into the journal).
                if let Ok(crate::v2::event::EventV2::Completion { op, result }) =
                    crate::v2::event::decode_event_frame(&frame.1)
                {
                    let host = caller.data_mut();
                    host.ops.finish(op);
                    if let CompletionResult::Ok(SuccessPayload::Handle(h)) = result {
                        if let Some(hash) = host.op_hashes.remove(&op) {
                            let Some(bytes) = host.script.payloads.get(&hash) else {
                                return Err(diverged(format!(
                                    "ReplayMissingPayload: completion for op {op:#x} names \
                                     content {} but the replay payload table lacks it",
                                    hex8(&hash)
                                )));
                            };
                            host.host_buffers.insert(h, Arc::new(bytes.clone()));
                        }
                    }
                }
            } else {
                caller.data_mut().pending_event = Some(frame);
            }
            Ok(())
        }
        ("vhc@2", "create_from") => {
            let (ptr, len) = (p_u32(params, 0), p_u32(params, 1));
            let mem = mem_of(caller)?;
            let mut bytes = vec![0u8; len as usize];
            mem.read(&mut *caller, ptr as usize, &mut bytes)
                .map_err(|e| wasmtime::Error::msg(format!("create_from read: {e}")))?;
            let handle = caller
                .data_mut()
                .buffers
                .create(Arc::new(bytes))
                .map_err(|c| diverged(format!("create_from refused at replay: {c:?}")))?;
            results[0] = Val::I64(handle as i64);
            Ok(())
        }
        ("vhc@2", "read_into") => {
            let (buffer, offset) = (p_u64(params, 0), p_u64(params, 1));
            let (ptr, cap) = (p_u32(params, 2), p_u32(params, 3));
            let data = resolve_replay_buffer(caller.data(), buffer)
                .ok_or_else(|| diverged(format!("read_into on unknown buffer {buffer:#x}")))?;
            let start = usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(data.len());
            let n = (data.len() - start).min(cap as usize);
            if n > 0 {
                let window = data[start..start + n].to_vec();
                let mem = mem_of(caller)?;
                mem.write(&mut *caller, ptr as usize, &window)
                    .map_err(|e| wasmtime::Error::msg(format!("read_into write: {e}")))?;
            }
            results[0] = Val::I64(n as i64);
            Ok(())
        }
        ("vhc@2", "buffer_len") => {
            let buffer = p_u64(params, 0);
            let data = resolve_replay_buffer(caller.data(), buffer)
                .ok_or_else(|| diverged(format!("buffer_len on unknown buffer {buffer:#x}")))?;
            results[0] = Val::I64(data.len() as i64);
            Ok(())
        }
        ("vhc@2", "buffer_release") => {
            let buffer = p_u64(params, 0);
            let host = caller.data_mut();
            if host.buffers.release(buffer).is_err() && host.host_buffers.remove(&buffer).is_none()
            {
                return Err(diverged(format!(
                    "buffer_release on unknown buffer {buffer:#x}"
                )));
            }
            Ok(())
        }
        ("vhc@2", "cancel") => {
            // Recordless-deterministic (v2::ops docs): the journaled completion result captures
            // the race — cancel returned 0 iff the op's (yet-undelivered) completion is Cancelled.
            let op = p_u64(params, 0);
            let host = caller.data_mut();
            let accepted = host.ops.is_outstanding(op)
                && host.script.events.iter().any(|(_, f)| {
                    matches!(
                        crate::v2::event::decode_event_frame(f),
                        Ok(crate::v2::event::EventV2::Completion { op: o, result })
                            if o == op && result == CompletionResult::cancelled()
                    )
                });
            if accepted {
                host.ops.finish(op);
                host.op_hashes.remove(&op);
            }
            results[0] = Val::I32(i32::from(!accepted));
            Ok(())
        }
        ("net@2", "payload_put") => {
            let buffer = p_u64(params, 0);
            let bytes = resolve_replay_buffer(caller.data(), buffer)
                .ok_or_else(|| diverged(format!("payload_put on unknown buffer {buffer:#x}")))?;
            let op = caller
                .data_mut()
                .ops
                .begin(OpRequest::PayloadPut { bytes })
                .map_err(|c| diverged(format!("payload_put refused at replay: {c:?}")))?;
            results[0] = Val::I64(op as i64);
            Ok(())
        }
        ("net@2", "payload_get") => {
            let hash_ptr = p_u32(params, 0);
            let mem = mem_of(caller)?;
            let mut hash = [0u8; 32];
            mem.read(&mut *caller, hash_ptr as usize, &mut hash)
                .map_err(|e| wasmtime::Error::msg(format!("payload_get read: {e}")))?;
            let host = caller.data_mut();
            let op = host
                .ops
                .begin(OpRequest::PayloadGet { hash })
                .map_err(|c| diverged(format!("payload_get refused at replay: {c:?}")))?;
            host.op_hashes.insert(op, hash);
            results[0] = Val::I64(op as i64);
            Ok(())
        }
        ("vhc@2", "read_back") => {
            let (src, kind) = (p_u64(params, 0), u64::from(p_u32(params, 1)));
            let (ptr, cap) = (p_u32(params, 2), p_u32(params, 3));
            // Peek (don't consume) so a NeedCapacity retry re-reads the same record.
            let (r_src, r_kind, value) = caller
                .data()
                .script
                .readbacks
                .front()
                .cloned()
                .ok_or_else(|| diverged("guest requested a read_back with none recorded"))?;
            if (r_src, r_kind) != (src, kind) {
                return Err(diverged(format!(
                    "read_back(src {src}, kind {kind}) but the journal recorded \
                     (src {r_src}, kind {r_kind})"
                )));
            }
            if deliver_span(caller, &value, ptr, cap, results)? {
                caller.data_mut().script.readbacks.pop_front();
            }
            Ok(())
        }
        ("net@2", "publish") => {
            let (channel, ptr, len) = (
                u64::from(p_u32(params, 0)),
                p_u32(params, 1),
                p_u32(params, 2),
            );
            let mem = mem_of(caller)?;
            let mut payload = vec![0u8; len as usize];
            mem.read(&mut *caller, ptr as usize, &mut payload)
                .map_err(|e| wasmtime::Error::msg(format!("payload read: {e}")))?;
            let host = caller.data_mut();
            let seq = {
                let c = host.seqs.entry(channel).or_insert(0);
                let s = *c;
                *c += 1;
                s
            };
            host.decisions.push(ReplayedDecision {
                event_index: host.events_delivered.saturating_sub(1),
                channel,
                seq,
                payload_hash: *blake3::hash(&payload).as_bytes(),
            });
            results[0] = Val::I64(seq as i64);
            Ok(())
        }
        ("sys@2", "set_timer") => {
            let delay = p_u64(params, 0);
            let (id, r_delay) = caller
                .data_mut()
                .script
                .timer_arms
                .pop_front()
                .ok_or_else(|| diverged("guest armed a timer with none recorded (tag 5)"))?;
            if delay != r_delay {
                return Err(diverged(format!(
                    "set_timer({delay} ms) but the journal recorded {r_delay} ms"
                )));
            }
            results[0] = Val::I64(id as i64);
            Ok(())
        }
        ("sys@2", "cancel_timer") => {
            let id = p_u64(params, 0);
            let (r_id, status) = caller
                .data_mut()
                .script
                .timer_cancels
                .pop_front()
                .ok_or_else(|| diverged("guest cancelled a timer with none recorded (tag 6)"))?;
            if id != r_id {
                return Err(diverged(format!(
                    "cancel_timer({id}) but the journal recorded id {r_id}"
                )));
            }
            results[0] = Val::I32(status as i32);
            Ok(())
        }
        ("sys@2", "now") => {
            let now = caller
                .data_mut()
                .script
                .clocks
                .pop_front()
                .ok_or_else(|| diverged("guest read the clock with none recorded (tag 3)"))?;
            results[0] = Val::I64(now as i64);
            Ok(())
        }
        // Advisory sinks: no recorded product, no decision (§6.3) — accepted and dropped.
        ("sys@2", "emit_metric" | "log") => Ok(()),
        // The run-scoped RNG seed is DETERMINISTIC (a pure function of the execution identity,
        // §2.7 dc class): re-derive it — nothing was recorded, exactly like the crypto accels.
        ("sys@2", "rng_seed") => {
            let out_ptr = p_u32(params, 0);
            let identity = caller.data().script.identity.clone().ok_or_else(|| {
                diverged("guest read rng_seed but the replay script carries no identity")
            })?;
            let seed = super::driver::derive_rng_seed(&identity);
            let mem = mem_of(caller)?;
            mem.write(&mut *caller, out_ptr as usize, &seed)
                .map_err(|e| wasmtime::Error::msg(format!("rng_seed write: {e}")))?;
            results[0] = Val::I32(0);
            Ok(())
        }
        // The device profile is a RECORDED nondeterministic input (tag 15): feed the recorded
        // bytes with the NeedCapacity protocol (peek, pop on delivery — a retry re-reads).
        ("sys@2", "device_profile") => {
            let (ptr, cap) = (p_u32(params, 0), p_u32(params, 1));
            let profile = caller
                .data()
                .script
                .device_profiles
                .front()
                .cloned()
                .ok_or_else(|| {
                    diverged("guest read the device profile with none recorded (tag 15)")
                })?;
            if deliver_span(caller, &profile, ptr, cap, results)? {
                caller.data_mut().script.device_profiles.pop_front();
            }
            Ok(())
        }
        // Crypto accelerations are deterministic functions of guest memory (no host observation, so
        // never journaled, §2.7 dc class): replay RE-EXECUTES them over the reproduced linear
        // memory and gets the identical answer — the same `daemon_vhc_proto::crypto` contract the
        // live host accel uses.
        ("sys@2", "hash") => {
            let (in_ptr, in_len, out_ptr) = (p_u32(params, 0), p_u32(params, 1), p_u32(params, 2));
            let mem = mem_of(caller)?;
            let mut data = vec![0u8; in_len as usize];
            mem.read(&mut *caller, in_ptr as usize, &mut data)
                .map_err(|e| wasmtime::Error::msg(format!("hash input read: {e}")))?;
            let digest = super::driver::host_crypto_hash(&data);
            let mem = mem_of(caller)?;
            mem.write(&mut *caller, out_ptr as usize, &digest)
                .map_err(|e| wasmtime::Error::msg(format!("hash output write: {e}")))?;
            results[0] = Val::I32(0);
            Ok(())
        }
        ("sys@2", "verify_sig") => {
            let (pk_ptr, sig_ptr, msg_ptr, msg_len) = (
                p_u32(params, 0),
                p_u32(params, 1),
                p_u32(params, 2),
                p_u32(params, 3),
            );
            let mem = mem_of(caller)?;
            let mut pk = vec![0u8; daemon_vhc_proto::VERIFY_PUBLIC_KEY_LEN];
            let mut sig = vec![0u8; daemon_vhc_proto::VERIFY_SIGNATURE_LEN];
            let mut msg = vec![0u8; msg_len as usize];
            mem.read(&mut *caller, pk_ptr as usize, &mut pk)
                .map_err(|e| wasmtime::Error::msg(format!("verify pk read: {e}")))?;
            mem.read(&mut *caller, sig_ptr as usize, &mut sig)
                .map_err(|e| wasmtime::Error::msg(format!("verify sig read: {e}")))?;
            mem.read(&mut *caller, msg_ptr as usize, &mut msg)
                .map_err(|e| wasmtime::Error::msg(format!("verify msg read: {e}")))?;
            results[0] = Val::I32(super::driver::host_crypto_verify(&pk, &sig, &msg) as i32);
            Ok(())
        }
        ("tabi@1", _) => {
            if let Some((_, want_kind)) = NR_IMPORTS.iter().find(|(n, _)| *n == name) {
                let (kind, value) = caller.data_mut().script.nr.pop_front().ok_or_else(|| {
                    diverged(format!("guest called nr import {name} with none recorded"))
                })?;
                if kind != *want_kind {
                    return Err(diverged(format!(
                        "nr import {name} (kind {want_kind}) but the journal recorded kind {kind}"
                    )));
                }
                match name {
                    "scalar@1" | "det_l2norm@1" => {
                        let v = f64::from_le_bytes(value.as_slice().try_into().map_err(|_| {
                            diverged(format!("nr kind {kind}: malformed f64 value"))
                        })?);
                        results[0] = Val::F64(v.to_bits());
                    }
                    "upd_read_bytes@1" => {
                        // (i, s, ptr, cap) -> written; the recorded value IS what was written.
                        let ptr = p_u32(params, 2);
                        let mem = mem_of(caller)?;
                        mem.write(&mut *caller, ptr as usize, &value)
                            .map_err(|e| wasmtime::Error::msg(format!("nr write: {e}")))?;
                        results[0] = Val::I32(value.len() as i32);
                    }
                    _ => {
                        let v = u32::from_le_bytes(value.as_slice().try_into().map_err(|_| {
                            diverged(format!("nr kind {kind}: malformed u32 value"))
                        })?);
                        results[0] = Val::I32(v as i32);
                    }
                }
                Ok(())
            } else {
                // A bridge COMPUTE import: pure state transformer — synthesize opaque results,
                // re-execute nothing (§8.7). Handles are counter-derived; scalars are unused by
                // conforming guests outside the nr set.
                for (i, t) in result_types.iter().enumerate() {
                    results[i] = match t {
                        wasmtime::ValType::I64 => {
                            let h = caller.data_mut().next_handle;
                            caller.data_mut().next_handle += 1;
                            Val::I64(h as i64)
                        }
                        wasmtime::ValType::I32 => Val::I32(0),
                        wasmtime::ValType::F64 => Val::F64(0f64.to_bits()),
                        wasmtime::ValType::F32 => Val::F32(0f32.to_bits()),
                        other => {
                            return Err(wasmtime::Error::msg(format!(
                                "unsupported tabi result type {other} in replay"
                            )))
                        }
                    };
                }
                Ok(())
            }
        }
        _ => Err(diverged(format!(
            "guest called `{ns}::{name}`, which the Phase-A replay does not model"
        ))),
    }
}

/// Re-drive `wasm` from a recorded journal: `da_init(config, grants)` then `da_run`, every
/// nondeterministic input answered from `script`, every publish collected as a decision.
///
/// Synchronous and single-threaded — a replay needs no pump: the guest can never block, because
/// every input it may wait for is already in the script (a pull beyond the script is itself a
/// divergence).
///
/// # Errors
/// [`V2Error`] only for harness-level failures (compile/instantiate/missing exports). Guest-level
/// endings — outcome, init refusal, divergence, trap — are the [`ReplayedRun::end`] variants.
pub fn replay_v2(
    worker: &Worker,
    wasm: &[u8],
    config: &[u8],
    grants: &[u8],
    script: ReplayScript,
) -> Result<ReplayedRun, V2Error> {
    let module = Module::new(worker.engine(), wasm)
        .map_err(|e| V2Error::Sandbox(format!("replay compile: {e}")))?;

    let mut linker: Linker<ReplayHost> = Linker::new(worker.engine());
    for import in module.imports() {
        let ExternType::Func(func_ty) = import.ty() else {
            return Err(V2Error::Sandbox(format!(
                "non-function import `{}::{}`",
                import.module(),
                import.name()
            )));
        };
        let ns = import.module().to_string();
        let name = import.name().to_string();
        let result_types: Vec<wasmtime::ValType> = func_ty.results().collect();
        linker
            .func_new(
                import.module(),
                import.name(),
                func_ty.clone(),
                move |mut caller, params, results| {
                    dispatch(&mut caller, &ns, &name, params, results, &result_types)
                },
            )
            .map_err(|e| V2Error::Sandbox(format!("replay link: {e}")))?;
    }

    let host = ReplayHost {
        script,
        decisions: Vec::new(),
        events_delivered: 0,
        pending_event: None,
        next_handle: 0x5EED_0000_0000,
        seqs: std::collections::HashMap::new(),
        // Quotas 0 (unbounded): the recording already enforced them; replay re-derives handles.
        buffers: BufferTable::new(0, 0, 0),
        ops: OpTable::new(0, 0),
        op_hashes: HashMap::new(),
        host_buffers: HashMap::new(),
    };
    let mut store = Store::new(worker.engine(), host);
    // Replay is not the sandbox gate — the recording already enforced budgets; give the replay
    // room to complete and rely on divergence detection for runaway guests.
    store
        .set_fuel(u64::MAX / 2)
        .map_err(|e| V2Error::Sandbox(e.to_string()))?;
    store.set_epoch_deadline(u64::MAX / 2);

    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| V2Error::Sandbox(format!("replay instantiation: {e}")))?;

    // Config + grants spans via da_alloc, exactly as the recording driver wrote them (§2.4).
    let write_span = |store: &mut Store<ReplayHost>, bytes: &[u8]| -> Result<u32, V2Error> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let alloc = instance
            .get_typed_func::<(u32, u32), u32>(&mut *store, "da_alloc")
            .map_err(|_| V2Error::Sandbox("missing da_alloc".into()))?;
        let ptr = alloc
            .call(&mut *store, (bytes.len() as u32, 1))
            .map_err(|e| V2Error::Sandbox(format!("da_alloc: {e}")))?;
        let mem = instance
            .get_memory(&mut *store, "memory")
            .ok_or_else(|| V2Error::Sandbox("no exported memory".into()))?;
        mem.write(&mut *store, ptr as usize, bytes)
            .map_err(|e| V2Error::Sandbox(format!("span write: {e}")))?;
        Ok(ptr)
    };
    let cfg_ptr = write_span(&mut store, config)?;
    let grants_ptr = write_span(&mut store, grants)?;

    let finish = |store: Store<ReplayHost>, end: ReplayEnd| {
        let host = store.into_data();
        Ok(ReplayedRun {
            decisions: host.decisions,
            events_delivered: host.events_delivered,
            end,
        })
    };
    let classify = |e: &wasmtime::Error| -> ReplayEnd {
        let msg = format!("{e:#}");
        match msg.find(DIVERGENCE_MARKER) {
            Some(i) => ReplayEnd::Diverged(msg[i + DIVERGENCE_MARKER.len()..].to_string()),
            None => ReplayEnd::Trapped(msg),
        }
    };

    let da_init = instance
        .get_typed_func::<(u32, u32, u32, u32), u32>(&mut store, "da_init")
        .map_err(|_| V2Error::Sandbox("missing/mis-typed da_init".into()))?;
    match da_init.call(
        &mut store,
        (
            cfg_ptr,
            config.len() as u32,
            grants_ptr,
            grants.len() as u32,
        ),
    ) {
        Ok(0) => {}
        Ok(status) => return finish(store, ReplayEnd::InitRefused(status)),
        Err(e) => {
            let end = classify(&e);
            return finish(store, end);
        }
    }

    let da_run = instance
        .get_typed_func::<(), u32>(&mut store, "da_run")
        .map_err(|_| V2Error::Sandbox("missing/mis-typed da_run".into()))?;
    match da_run.call(&mut store, ()) {
        Ok(outcome) => finish(store, ReplayEnd::Outcome(outcome)),
        Err(e) => {
            let end = classify(&e);
            finish(store, end)
        }
    }
}
