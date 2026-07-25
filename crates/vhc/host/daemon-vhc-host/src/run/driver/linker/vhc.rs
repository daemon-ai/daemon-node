// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `vhc@2` world: `next_event` (THE blocking pull, §4.1 — parking, due-timer firing, the
//! drain-deadline watchdog, budget reset at Delivered), `read_back` (§6.4 budgeted readback with
//! the mandatory-retry protocol), `stage_state`/`snapshot_state` (§10.2), the buffer layer
//! (§3.4/§7.4), and `cancel` (§7.5).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use wasmtime::{Caller, Linker};

use daemon_vhc_abi::{
    pack_status_len, EV_TAG_FRAME, EV_TAG_STOP, NS_VHC_V2, READBACK_KIND_STAGED_BYTES,
    RET_STATUS_DELIVERED, RET_STATUS_NEED_CAPACITY, SNAPSHOT_STATE_SECTION_MISSING,
    STAGED_KIND_BYTES,
};

use crate::run::completion::CompletionResult;
use crate::run::driver::config::SnapshotCapture;
use crate::run::driver::host::{read_guest, shared_of, stash, write_guest, Host, PARK_RECHECK};
use crate::run::driver::migration::decode_manifest_sections;
use crate::run::driver::pump::fire_due_timers;
use crate::run::state_store::StateStoreError;
use crate::trap::{Trap, TrapCode};

/// Map a typed state-store refusal onto its trap (ABI §12.14 [SF-4]/[SF-7]): framing faults get
/// the two dedicated codes; grant breaches are `GrantViolation` (guest-driven, attributable);
/// unknown streams are handle faults; an unprovisioned state plane (no genesis state contract)
/// is a grant fault too — the capability was never provisioned for this run.
fn state_trap(e: StateStoreError, import: &'static str) -> Trap {
    let code = match &e {
        StateStoreError::MisframedEmit { .. } => TrapCode::StateMisframedEmit,
        StateStoreError::IncompleteSeal { .. } => TrapCode::StateIncompleteSeal,
        StateStoreError::UnknownStream => TrapCode::InvalidHandle,
        StateStoreError::EmptyFamily => TrapCode::BadEnum,
        StateStoreError::NotProvisioned
        | StateStoreError::StreamsExhausted { .. }
        | StateStoreError::WriteBudget { .. }
        | StateStoreError::StoreBytes { .. } => TrapCode::GrantViolation,
    };
    Trap::new(code, import, None, e.to_string())
}

/// Link the `vhc@2` imports.
#[allow(clippy::too_many_lines)]
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    // ---- vhc@2::next_event — THE blocking pull (§4.1) -------------------------------------------
    linker.func_wrap(
        NS_VHC_V2,
        "next_event",
        |mut c: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("next_event")?;
                // The bounded-guest-memory MEASUREMENT (the declared-budget evidence surface): wasm
                // linear memory never shrinks, so its size at the one seam every event slice passes
                // through is the run's high-water. Recorded here so a run's real footprint is a
                // measured number a gate can assert against its DECLARED budget, instead of being
                // inferred from "it did not trap".
                if let Some(size) = c
                    .get_export("memory")
                    .and_then(wasmtime::Extern::into_memory)
                    .map(|m| m.data_size(&c) as u64)
                {
                    let shared = c.data().shared.clone();
                    let mut st = shared.state.lock().expect("pump lock");
                    st.guest_memory_high_water = st.guest_memory_high_water.max(size);
                }
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
                        // Forced interruption at the drain deadline (§4.4/§11.3): a guest that
                        // has not returned `QuiesceReady` by the advertised deadline is trapped
                        // typed — the host never waits on a drain indefinitely. Checked before
                        // candidate selection, so a guest busy consuming still-deliverable events
                        // cannot ride past the deadline either.
                        if st.draining {
                            if let Some(deadline) = st.drain_deadline_at {
                                if now >= deadline {
                                    return Err(Trap::new(
                                        TrapCode::QuiesceDeadlineExceeded,
                                        "next_event",
                                        None,
                                        format!(
                                            "the Quiesce drain's deadline passed ({now}ms >= \
                                             {deadline}ms) without QuiesceReady (§4.4/§11.3)"
                                        ),
                                    ));
                                }
                            }
                        }
                        // Fire due timers (frozen during a drain, §4.4; and never behind a queued
                        // Stop — the host delivers no further events after Stop, §4.4).
                        if !st.draining && !st.stop_enqueued {
                            fire_due_timers(&mut st, now)?;
                        }
                        // During a Quiesce drain, authoritative `Frame` deliveries are FROZEN:
                        // they spool undelivered (§4.4) and drain into the NEW instance after an
                        // upgrade activation (§10.3 step 6, `PumpHandle::take_spooled_frames`).
                        // Everything else (Quiesce itself, Budget, Fence, Completion) delivers.
                        let candidate = if st.draining {
                            st.queue.iter().position(|q| q.tag != EV_TAG_FRAME)
                        } else if st.queue.is_empty() {
                            None
                        } else {
                            Some(0)
                        };
                        if let Some(pos) = candidate {
                            let len = st.queue[pos].frame_bytes.len() as u64;
                            if len > u64::from(buf_cap) {
                                // Not consumed; no journal record; fuel/op budgets do not reset
                                // (§4.1/§5.5 — they reset on Delivered only). The epoch WATCHDOG
                                // does re-arm: the deadline armed at the previous Delivered may
                                // have lapsed during a long park (live planes idle for seconds),
                                // and the mandatory realloc+retry executes guest code — killing
                                // it here would epoch-kill a guest FOR WAITING, exactly what
                                // §5.6 rules out. Unloopable: a retry below the required length
                                // is the typed BadEvent trap above.
                                drop(st);
                                let d = c.data_mut();
                                d.slice.pending_next = Some(len);
                                let ticks = d.epoch_ticks;
                                wasmtime::AsContextMut::as_context_mut(c).set_epoch_deadline(ticks);
                                return Ok(pack_status_len(RET_STATUS_NEED_CAPACITY, len as u32));
                            }
                            // Deliver: sample once, journal BEFORE the guest observes (§8.4 r4).
                            let at = shared.now_ms();
                            let ev = st.queue.remove(pos).expect("candidate position exists");
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
                            // Bounded by the drain deadline, so its expiry is noticed promptly.
                            st.drain_deadline_at.map_or(PARK_RECHECK, |d| {
                                Duration::from_millis(d.saturating_sub(now).max(1))
                                    .min(PARK_RECHECK)
                            })
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
        |mut c: Caller<'_, Host>,
         src: u64,
         kind: u32,
         out_ptr: u32,
         out_cap: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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
                // kind → the staged-kind it consumes (ABI §6.4 table): 0 bytes, 3 state-section
                // (migrate restore, §10.2). Kinds 1/2 retired with the compute bridge; their
                // assignments are permanent and never again valid call arguments.
                let want_staged_kind = match kind {
                    READBACK_KIND_STAGED_BYTES => daemon_vhc_abi::STAGED_KIND_BYTES,
                    daemon_vhc_abi::READBACK_KIND_STATE_SECTION => {
                        // Legal EXACTLY during `da_migrate` (§6.6's one exception), and only
                        // under the migration grant's restore bit (§10.2 — fail closed).
                        if !c.data().slice.in_migrate {
                            return Err(Trap::new(
                                TrapCode::PhaseViolation,
                                "read_back",
                                None,
                                "state-section readback outside da_migrate (§6.6/§10.2)",
                            ));
                        }
                        if !c.data().migration_restore {
                            return Err(Trap::new(
                                TrapCode::GrantViolation,
                                "read_back",
                                None,
                                "migration-grant.restore is not granted (§2.6/§10.2)",
                            ));
                        }
                        daemon_vhc_abi::STAGED_KIND_STATE_SECTION
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
                // Conversely, inside da_migrate ONLY the state-section kind is legal (§6.6).
                if c.data().slice.in_migrate && kind != daemon_vhc_abi::READBACK_KIND_STATE_SECTION
                {
                    return Err(Trap::new(
                        TrapCode::PhaseViolation,
                        "read_back",
                        None,
                        "only read_back(kind = state-section) is legal during da_migrate (§6.6)",
                    ));
                }
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
                // Kind 0 delivers the bytes verbatim, re-readable; kind 3 delivers the staged
                // state section (§10.2).
                let value = staged_bytes;
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
        |mut c: Caller<'_, Host>, ptr: u32, len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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

    // ---- vhc@2::snapshot_state — quiesce-scoped (§10.2): verify + journal + capture ------------
    linker.func_wrap(
        NS_VHC_V2,
        "snapshot_state",
        |mut c: Caller<'_, Host>, ptr: u32, len: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("snapshot_state")?;
                if !c.data().slice.draining {
                    return Err(Trap::new(
                        TrapCode::PhaseViolation,
                        "snapshot_state",
                        None,
                        "snapshot_state outside a Quiesce drain (§10.2)",
                    ));
                }
                let manifest_bytes = read_guest(c, ptr, len)?;
                // Decode the §10.2 state-manifest at the CBOR-value level (the host never links
                // the SDK's typed manifest — dependency wall; the bytes are journaled verbatim
                // and handed to `da_migrate` verbatim, so no re-encode ever happens host-side).
                let sections = match decode_manifest_sections(&manifest_bytes) {
                    Ok(s) => s,
                    Err(detail) => {
                        return Err(Trap::new(
                            TrapCode::BadEvent,
                            "snapshot_state",
                            None,
                            format!("malformed state-manifest: {detail}"),
                        ))
                    }
                };
                let (max_sections, max_section_bytes) = {
                    let d = c.data();
                    (d.migration_max_sections, d.migration_max_section_bytes)
                };
                if max_sections != 0 && sections.len() as u64 > max_sections {
                    return Ok(daemon_vhc_abi::SNAPSHOT_STATE_GRANT_EXCEEDED);
                }
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // A second successful submission in one drain is BadEvent (§10.2).
                if st.accepted_snapshot.is_some() {
                    return Err(Trap::new(
                        TrapCode::BadEvent,
                        "snapshot_state",
                        None,
                        "a snapshot was already accepted in this drain (§10.2)",
                    ));
                }
                // Verify + capture every declared section in one of the two §10.2/[SF-6] forms;
                // rejected attempts MAY be corrected and retried within the drain (never "exactly
                // once"). INLINE: a guest-staged plain-bytes entry matching by content hash (the
                // small round watermark). BY-REFERENCE: the decl's `hash` is a family FOLD the
                // instance itself sealed — the host reconstructs the FamilyRef from its own state
                // store (chunk hashes recorded on emit; §7.2), moving zero section bytes, so the
                // per-section byte cap does not apply to it.
                let mut captured: Vec<daemon_vhc_proto::det_state::CkptDocSection> =
                    Vec::with_capacity(sections.len());
                for decl in &sections {
                    let inline = st
                        .staged
                        .iter()
                        .find(|(id, (kind, bytes))| {
                            *id & daemon_vhc_abi::GUEST_STAGING_ID_TOP_BIT != 0
                                && *kind == STAGED_KIND_BYTES
                                && bytes.len() as u64 == decl.size
                                && blake3::hash(bytes).as_bytes() == &decl.hash
                        })
                        .map(|(_, (_, bytes))| bytes.clone());
                    if let Some(bytes) = inline {
                        if max_section_bytes != 0 && decl.size > max_section_bytes {
                            return Ok(daemon_vhc_abi::SNAPSHOT_STATE_GRANT_EXCEEDED);
                        }
                        captured.push(daemon_vhc_proto::det_state::CkptDocSection::Inline(
                            decl.name.clone(),
                            bytes,
                        ));
                        continue;
                    }
                    if let Some(fref) = st.state.sealed_family_ref(&decl.hash) {
                        // The manifest's declared `size` for a by-ref section is the family
                        // byte_len; a disagreement is the §10.2 hash/geometry-mismatch status.
                        if fref.byte_len != decl.size {
                            return Ok(daemon_vhc_abi::SNAPSHOT_STATE_HASH_MISMATCH);
                        }
                        captured.push(daemon_vhc_proto::det_state::CkptDocSection::ByRef(
                            decl.name.clone(),
                            fref,
                        ));
                        continue;
                    }
                    // Neither staged bytes nor a known self-sealed fold: distinguish the §10.2
                    // statuses (staged-but-mutated vs never staged).
                    let name_sized = st.staged.values().any(|(kind, bytes)| {
                        *kind == STAGED_KIND_BYTES && bytes.len() as u64 == decl.size
                    });
                    return Ok(if name_sized {
                        daemon_vhc_abi::SNAPSHOT_STATE_HASH_MISMATCH
                    } else {
                        SNAPSHOT_STATE_SECTION_MISSING
                    });
                }
                // Accepted: journal the manifest verbatim (tag 10) under the sink's durability
                // barrier (§8.4 rule 2), then capture for the upgrade transaction.
                st.sink
                    .snapshot(&manifest_bytes)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                st.accepted_snapshot = Some(SnapshotCapture {
                    manifest: manifest_bytes,
                    sections: captured,
                });
                Ok(daemon_vhc_abi::SNAPSHOT_STATE_ACCEPTED)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- vhc@2 minor 1: the buffer layer (architecture §3.4; ABI §7.4) --------------------------
    // create_from — the budgeted linear-memory path OUT: seal guest bytes into a kind-8 buffer.
    linker.func_wrap(
        NS_VHC_V2,
        "create_from",
        |mut c: Caller<'_, Host>, ptr: u32, len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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
    // buffer_open / buffer_append / buffer_seal — the INCREMENTAL twin of `create_from` (ABI
    // minor 4, the host-resource layer only: no wire, no contract, no new event). `create_from`
    // requires the whole object in linear memory at once, which is precisely the residency a
    // producer of a large committed update cannot afford (the emit-side mirror of [SF-R3]: the
    // consuming side range-reads the payload out of a host buffer, so the producing side must be
    // able to build one without ever holding it). The shape mirrors `state_open/emit/seal`: open a
    // stream, append bounded spans, seal into exactly the kind-8 `BufferHandle` `create_from`
    // would have returned — usable by `payload_put` unchanged.
    //
    // dc class throughout: stream ids are counter-deterministic per instance, the appended bytes
    // come from replay-reproduced guest memory, and the sealed handle is the buffer table's next
    // deterministic id — so no journal record, exactly like `create_from`.
    linker.func_wrap(
        NS_VHC_V2,
        "buffer_open",
        |mut c: Caller<'_, Host>| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("buffer_open")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                Ok(st.buffer_streams.open())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_VHC_V2,
        "buffer_append",
        |mut c: Caller<'_, Host>,
         stream: u64,
         ptr: u32,
         len: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("buffer_append")?;
                let bytes = read_guest(c, ptr, len)?;
                // Same hard-accountable meter `create_from` charges: the growing host-side object
                // is a guest-initiated allocation the host meters exactly (ABI §9.1).
                {
                    let d = c.data_mut();
                    d.accountable_staged_bytes += u64::from(len);
                    if d.hard_accountable_host_bytes != 0
                        && d.accountable_staged_bytes > d.hard_accountable_host_bytes
                    {
                        return Err(Trap::new(
                            TrapCode::BudgetMemory,
                            "buffer_append",
                            None,
                            "hard-accountable host cap breached (attributable, ABI §9.1)",
                        ));
                    }
                }
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.buffer_streams
                    .append(stream, &bytes)
                    .map_err(|e| Trap::new(TrapCode::InvalidHandle, "buffer_append", None, e))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_VHC_V2,
        "buffer_seal",
        |mut c: Caller<'_, Host>, stream: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("buffer_seal")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let bytes = st
                    .buffer_streams
                    .take(stream)
                    .map_err(|e| Trap::new(TrapCode::InvalidHandle, "buffer_seal", None, e))?;
                st.buffers
                    .create(Arc::new(bytes))
                    .map_err(|code| Trap::new(code, "buffer_seal", None, "buffer quota (§7.3)"))
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
        |mut c: Caller<'_, Host>,
         buffer: u64,
         offset: u64,
         out_ptr: u32,
         out_cap: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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
        |mut c: Caller<'_, Host>, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
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
        |mut c: Caller<'_, Host>, buffer: u64| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, Host>| {
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
    // ---- vhc@2 minor 3: the det-state write surface (ABI §12.14 [SF-4]) -------------------------
    // state_open — open a family write stream: `tag` names the family (e.g. "master"), `byte_len`
    // declares its total length. Stream ids are counter-deterministic (top-bit namespace, §2.7 dc
    // class — no journal record; replay re-derives them from call order).
    linker.func_wrap(
        NS_VHC_V2,
        "state_open",
        |mut c: Caller<'_, Host>,
         tag_ptr: u32,
         tag_len: u32,
         byte_len: u64|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("state_open")?;
                let tag_bytes = read_guest(c, tag_ptr, tag_len)?;
                let tag = String::from_utf8(tag_bytes).map_err(|_| {
                    Trap::new(
                        TrapCode::BadEnum,
                        "state_open",
                        None,
                        "the family tag is not UTF-8",
                    )
                })?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.state
                    .open(&tag, byte_len)
                    .map_err(|e| state_trap(e, "state_open"))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // state_emit — append exactly one chunk: the host copies the span out of linear memory,
    // blake3-hashes it, and stores it content-addressed ([SF-4]). Coarse framing (`0 < len ≤
    // chunk_size`, never past the declared byte_len) traps typed; the write budget ([SF-7]
    // `state-write-budget`) traps `GrantViolation` (guest-driven, attributable). dc class over
    // replay-reproduced guest memory: no journal record; replay re-executes the emit into a
    // replay-side state chunk store.
    linker.func_wrap(
        NS_VHC_V2,
        "state_emit",
        |mut c: Caller<'_, Host>,
         stream: u64,
         ptr: u32,
         len: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("state_emit")?;
                let bytes = read_guest(c, ptr, len)?;
                let shared = c.data().shared.clone();
                let now = shared.now_ms();
                let mut st = shared.state.lock().expect("pump lock");
                st.state
                    .emit(stream, &bytes, now)
                    .map_err(|e| state_trap(e, "state_emit"))
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    // state_seal — close the stream: the host computes the domain-separated family fold over the
    // accumulated chunk hashes, registers it retained + fetchable ([SF-R1]), enforces retention +
    // `state-store-bytes` ([SF-7]), and writes the 32-byte fold into guest memory. nr class: the
    // fold is journaled verbatim (the journal-record-only READBACK_KIND_STATE_SEAL) BEFORE the
    // guest observes it — replay re-derives the fold over the re-emitted chunks and compares,
    // the O(1) fold-divergence cross-check. An incomplete seal traps typed and the stream stays
    // open (complete and retry); a store-bytes refusal rolls the seal back (nothing durable).
    linker.func_wrap(
        NS_VHC_V2,
        "state_seal",
        |mut c: Caller<'_, Host>, stream: u64, out_ptr: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("state_seal")?;
                let shared = c.data().shared.clone();
                let fold = {
                    let mut st = shared.state.lock().expect("pump lock");
                    let fold = st
                        .state
                        .seal(stream)
                        .map_err(|e| state_trap(e, "state_seal"))?;
                    // The nr record (§8.4 rule 4 discipline: journal before the guest observes).
                    st.sink
                        .read_back(
                            stream,
                            u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL),
                            RET_STATUS_DELIVERED,
                            &fold,
                        )
                        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                    fold
                };
                write_guest(c, out_ptr, &fold)?;
                Ok(0)
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
        |mut c: Caller<'_, Host>, op: u64| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
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

    Ok(())
}
