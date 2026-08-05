// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `sys@2` world: timers (§6.3), the slice-constant journaled clock (§6.5), metric/log
//! egress, the deterministic crypto accelerations (`hash`/`verify_sig`, §3.2/§3.7), the
//! identity-derived `rng_seed`, and the journaled `device_profile` delivery (§3.5).

use wasmtime::{Caller, Linker};

use daemon_vhc_abi::{pack_status_len, NS_SYS_V2, RET_STATUS_DELIVERED, RET_STATUS_NEED_CAPACITY};

use crate::run::driver::host::{
    host_crypto_hash, host_crypto_verify, read_guest, stash, write_guest, Host,
};
use crate::run::driver::pump::ArmedTimer;
use crate::trap::{Trap, TrapCode};

/// Link the `sys@2` imports.
#[allow(clippy::too_many_lines)]
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    // ---- sys@2 ------------------------------------------------------------------------------------
    linker.func_wrap(
        NS_SYS_V2,
        "set_timer",
        |mut c: Caller<'_, Host>, delay_ms: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("set_timer")?;
                let armed_at = c.data().slice.now;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let id = st.next_timer_id;
                st.next_timer_id += 1;
                st.sink
                    .timer_arm(id, delay_ms, armed_at)
                    .map_err(Trap::from)?;
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
        |mut c: Caller<'_, Host>, timer_id: u64| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
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
                    .map_err(Trap::from)?;
                Ok(status)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "now",
        |mut c: Caller<'_, Host>| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("now")?;
                // Slice-constant (§6.5); every reading journaled (the coordinator-replay lesson).
                let now = c.data().slice.now;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.sink.clock(now).map_err(Trap::from)?;
                Ok(now)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "emit_metric",
        |mut c: Caller<'_, Host>,
         name_ptr: u32,
         name_len: u32,
         value: f64|
         -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, Host>| {
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
                st.note_egress();
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_SYS_V2,
        "log",
        |mut c: Caller<'_, Host>,
         level: u32,
         msg_ptr: u32,
         msg_len: u32|
         -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("log")?;
                // [LX-6]: the bounds are applied BEFORE the guest-memory read, and in this order.
                // The guest-supplied length is untrusted input and MUST NOT size a host allocation,
                // so a call is admitted or dropped against the phase's call budget using only the
                // arguments, the raw length is clamped, and only the accepted prefix is read. A host
                // that reads first and limits afterwards has already spent the memory the limit
                // exists to protect.
                let context = c.data().slice.execution_context();
                let exempt_phase = matches!(
                    context,
                    daemon_vhc_abi::ExecutionContext::Init
                        | daemon_vhc_abi::ExecutionContext::Migrate
                );
                if exempt_phase {
                    let d = c.data_mut();
                    // A call past either budget is DROPPED — counted, never delivered, never a trap.
                    // Dropping must never trap: two peers must reach identical decisions whether or
                    // not either host dropped a line.
                    if d.slice.log_calls_this_phase >= daemon_vhc_abi::LOG_CALLS_PER_PHASE_MAX {
                        return Ok(());
                    }
                    d.slice.log_calls_this_phase += 1;
                }
                let remaining = if exempt_phase {
                    daemon_vhc_abi::LOG_BYTES_PER_PHASE_MAX
                        .saturating_sub(c.data().slice.log_bytes_this_phase)
                } else {
                    daemon_vhc_abi::LOG_BYTES_PER_PHASE_MAX
                };
                let accepted = daemon_vhc_abi::log_accepted_prefix_len(msg_len, remaining);
                if accepted == 0 {
                    return Ok(());
                }
                // Only now is guest memory touched, and only for the accepted prefix. A pointer
                // genuinely outside the guest's memory still traps `MemOob`; an absurd length is a
                // truncation, not a trap, because it was clamped above.
                let raw = read_guest(c, msg_ptr, accepted)?;
                if exempt_phase {
                    c.data_mut().slice.log_bytes_this_phase += u64::from(accepted);
                }
                // Lossy conversion and truncation at a character boundary; neither is a trap.
                let msg = String::from_utf8_lossy(&raw).into_owned();
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // The SDK panic hook's forwarded message (ABI §3.6): hold it aside so the
                // `unreachable` that follows a beat later traps WITH the message rather than
                // as an anonymous `GuestPanic` (`take_trap`). It is tagged with the context it was
                // emitted in, and is lifted only into a trap carrying that same context ([LX-10]).
                if let Some(detail) = msg.strip_prefix(daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX) {
                    st.guest_panic = Some((context, detail.to_string()));
                }
                st.logs.push((level.min(5), msg));
                st.note_egress();
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
        |mut c: Caller<'_, Host>,
         in_ptr: u32,
         in_len: u32,
         out_ptr: u32|
         -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
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
        |mut c: Caller<'_, Host>,
         pk_ptr: u32,
         sig_ptr: u32,
         msg_ptr: u32,
         msg_len: u32|
         -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
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
        |mut c: Caller<'_, Host>, out_ptr: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
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
        |mut c: Caller<'_, Host>, out_ptr: u32, out_cap: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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
                st.sink.device_profile(&bytes).map_err(Trap::from)?;
                Ok(pack_status_len(RET_STATUS_DELIVERED, len as u32))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    Ok(())
}
