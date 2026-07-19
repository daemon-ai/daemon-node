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
//! - The `tabi@1` compute bridge is RETIRED: a module importing the namespace is refused typed
//!   (`BridgeRetired`) at the §1.3 front door and again here at start; compute crosses the
//!   boundary through the `compute@2` world only.
//! - `snapshot_state` returns `SectionMissing` during a drain (no state-manifest verification yet
//!   — the §10.2 protocol lands with the migrate scaffolding); outside a drain it traps
//!   `PhaseViolation` per §6.6. `stage_state` is fully functional.
//! - Inbound frames arrive through [`PumpHandle::deliver_frame`] pre-verified: signature
//!   verification/dedup/gap detection are the session pump's admission-side jobs (the
//!   choreography sitting); this driver journals the original signed frame it is handed (§8.6).

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use wasmtime::{Linker, Module, Store, StoreLimitsBuilder};

use daemon_vhc_abi::{NS_COMPUTE_V2, NS_TABI_V1};
use daemon_vhc_proto::{peer_id, SigningKey};

use crate::run::buffer::BufferTable;
use crate::run::journal::{JournalSink, SinkError};
use crate::run::ops::OpTable;
use crate::run::streams::StreamTable;
use crate::runtime::{EngineConfig, Worker};
use crate::trap::{Trap, TrapCode};

mod chunks;
mod config;
mod host;
mod linker;
mod migration;
mod pump;

pub(crate) use chunks::decode_chunk_descriptor;
pub use config::{
    DeliverVerdict, MigrationInput, OpOutcome, RunConfig, RunEnd, RunError, RunIdentity,
    SnapshotCapture, SpooledFrame,
};
pub use host::{derive_rng_seed, host_crypto_hash, host_crypto_verify};
pub(crate) use migration::build_migration_descriptor;
pub use pump::PumpHandle;

use host::{Host, SliceState};
use linker::link_v2;
use pump::{PumpShared, PumpState};

// -- the run ---------------------------------------------------------------------------------------

/// A live v2 run: the embedder handle plus the guest thread's join handle.
pub struct Run {
    /// The embedder's event/staging/egress handle.
    pub pump: PumpHandle,
    thread: JoinHandle<Result<RunEnd, RunError>>,
}

impl Run {
    /// Join the guest thread and return how the run ended. The guest thread has already dropped
    /// the `Store` (guest-thread-owned teardown, §11.3) and journaled the terminal fact.
    ///
    /// # Errors
    /// [`RunError`] for setup/journaling failures (a trap is a [`RunEnd::Trapped`], not an error).
    pub fn wait(self) -> Result<RunEnd, RunError> {
        self.thread
            .join()
            .map_err(|_| RunError::Sandbox("guest thread panicked".into()))?
    }

    /// Whether the guest thread has ended (non-blocking): the upgrade transaction's migrate step
    /// polls this to distinguish "migrated and running" from "tore down before `da_run`"
    /// (`InitRefused`/`MigrateRefused`/trapped) without consuming the run.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.thread.is_finished()
    }
}

/// Start a major-2 run instance: journal the run header, spawn the dedicated guest thread,
/// instantiate with the real Phase-A capability providers, run `da_init` then `da_run` (§3.1,
/// §9.4 steps 10–12), journaling throughout.
///
/// The caller has already run ABI §1.3 selection (`select_driver` → `CandidateDriver::V2`).
/// A module importing the retired `tabi@1` bridge is refused typed ([`RunError::BridgeRetired`]).
///
/// # Errors
/// [`RunError`] on setup/journal failure. Guest traps and init refusals are [`RunEnd`]s.
pub fn start_run(
    worker: &Worker,
    wasm: &[u8],
    run: RunConfig,
    sink: Box<dyn JournalSink>,
) -> Result<Run, RunError> {
    start_run_migrating(worker, wasm, run, sink, None)
}

/// [`start_run`] with an optional **migration input** (ABI §10.3 step 4): when `migration` is
/// `Some`, the instantiation record is journaled as tag-13 **reason 2** (upgrade-activation),
/// the snapshot's sections are staged host-side after `da_init`, and `da_migrate(descriptor)`
/// runs under its explicit budget before `da_run`. A non-`Ready` return tears the instance down
/// as [`RunEnd::MigrateRefused`] (the transaction's validate failure — roll back, §10.3 step 7);
/// budget exhaustion inside `da_migrate` traps the typed `MigrateBudget`.
///
/// # Errors
/// [`RunError`] on setup/journal failure. Guest traps, init refusals, and migrate refusals are
/// [`RunEnd`]s.
pub fn start_run_migrating(
    worker: &Worker,
    wasm: &[u8],
    run: RunConfig,
    mut sink: Box<dyn JournalSink>,
    migration: Option<MigrationInput>,
) -> Result<Run, RunError> {
    let module =
        Module::new(worker.engine(), wasm).map_err(|e| RunError::Sandbox(e.to_string()))?;
    // The retired compute bridge: any tabi@1 import is refused typed here as well as at the
    // §1.3 front door (`validate_imports`), so a caller that skips selection still never links
    // or runs a bridge module.
    if module.imports().any(|i| i.module() == NS_TABI_V1) {
        return Err(RunError::BridgeRetired(
            "the module imports the retired tabi@1 compute bridge — compute crosses the \
             boundary through compute@2 only"
                .to_string(),
        ));
    }
    // The compute@2 command queue (track C1, ABI §15): a per-instance runner over the ADMITTED
    // backend, constructed only for modules that import the world.
    let compute = module.imports().any(|i| i.module() == NS_COMPUTE_V2);

    let engine_cfg: EngineConfig = worker.config().clone();
    // Backend claim revalidation at run start (fail fast, BEFORE the run header is journaled):
    // the admitted backend must still be servable — feature compiled AND the runtime probe
    // passing AND (device lanes) the process device-compute slot free. Unavailability is the
    // typed refusal, never a silent ndarray run. The guest-thread construction below stays the
    // authoritative backstop for the race window (a device dying between this check and
    // bring-up surfaces as a typed compute fault, classified recoverable).
    if compute {
        crate::compute::backend_available(engine_cfg.backend)
            .map_err(RunError::BackendUnavailable)?;
        if engine_cfg.backend.is_device() && crate::compute::DeviceComputeGuard::is_held() {
            return Err(RunError::BackendUnavailable(
                "a device-backed compute instance is already live in this process (one \
                 device-compute instance per process)"
                    .to_string(),
            ));
        }
    }
    let abi_packed = u64::from(daemon_vhc_abi::DA_ABI_MAJOR_V2) << 16;
    let worlds: Vec<(String, u64)> = module
        .imports()
        .map(|i| (i.module().to_string(), 0u64))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    // tag 0 first — the run header precedes everything (§8.3). The header's `bridge` field is
    // keep-reserved (always `false`: no bridge exists; the field stays so the record grammar is
    // unchanged and pre-existing journals stay parseable).
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
            // Generation-seeded by the instantiation counter (0: this driver instantiates once
            // per start_run; trap-restart re-seeding rides the tag-13 counter, ABI §7.1).
            buffers: BufferTable::new(0, run.max_live_buffer_handles, run.max_live_buffer_bytes),
            ops: OpTable::new(0, run.max_outstanding_ops),
            chunk_maps: std::collections::HashMap::new(),
            data_read_budget: run.data_read_budget_bytes,
            data_read_used: 0,
            streams: StreamTable::new(0),
            op_requests: Vec::new(),
            stop_enqueued: false,
            stop_cut: None,
            draining: false,
            drain_deadline_at: None,
            accepted_snapshot: None,
            egress_hook: None,
            migrate_validated: false,
        }),
        wake: Condvar::new(),
        t0: Instant::now(),
        hold: AtomicBool::new(false),
    });
    let pump = PumpHandle {
        shared: shared.clone(),
    };

    let mut linker: Linker<Host> = Linker::new(worker.engine());
    link_v2(&mut linker).map_err(|e| RunError::Sandbox(e.to_string()))?;

    let signing = SigningKey::from_bytes(&run.signing_seed);
    let sender = peer_id(&signing).0;
    let epoch_ticks = worker.epoch_ticks_pub();
    let engine = worker.engine().clone();

    let thread = std::thread::Builder::new()
        .name(format!(
            "vhc-guest-{}-{}",
            run.identity.role, run.identity.instance
        ))
        .spawn(move || -> Result<RunEnd, RunError> {
            // Construct the compute runner ON THIS THREAD (the pinned device thread): GPU
            // backends derive their stream + memory-pool registry from the constructing thread
            // and are driven single-threaded — every compute@2 import is a synchronous host
            // call on this same thread, so affinity holds for the instance's lifetime. A
            // bring-up failure here (the race the pre-spawn check cannot close: the device died
            // in between) is a typed compute fault, journaled terminal — classified recoverable
            // by the session, never a silent CPU run and never a host abort.
            let compute_runner = if compute {
                match crate::compute::HostCompute::build(&engine_cfg) {
                    Ok(runner) => {
                        // Host-side RNG (Float/Random ops) seeded deterministically from the
                        // identity-derived seed: two runs of one incarnation reproduce it, and
                        // replay never re-runs kernels anyway (kind-5 records feed readbacks).
                        let seed_bytes = derive_rng_seed(&run.identity);
                        runner.seed(u64::from_le_bytes(
                            seed_bytes[..8].try_into().expect("8-byte slice"),
                        ));
                        Some(runner)
                    }
                    Err(reason) => {
                        let trap = Trap::bare(
                            TrapCode::ComputeFault,
                            format!(
                                "backend unavailable at device bring-up ({}): {reason}",
                                engine_cfg.backend.slug()
                            ),
                        );
                        journal_terminal_trap(&shared, &trap)?;
                        return Ok(RunEnd::Trapped(trap));
                    }
                }
            } else {
                None
            };
            let host = Host {
                shared: shared.clone(),
                limits: StoreLimitsBuilder::new()
                    .memory_size(engine_cfg.max_memory_bytes)
                    .build(),
                trap: None,
                slice: SliceState {
                    in_init: false,
                    in_migrate: false,
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
                migration_max_sections: run.migration_max_sections,
                migration_max_section_bytes: run.migration_max_section_bytes,
                migration_restore: migration.as_ref().is_some_and(|m| m.restore),
                compute: compute_runner,
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
                .map_err(|e| RunError::Sandbox(e.to_string()))?;
            store.set_epoch_deadline(epoch_ticks);

            let instance = linker
                .instantiate(&mut store, &module)
                .map_err(|e| RunError::Sandbox(format!("v2 instantiation: {e}")))?;

            // tag 13 at instantiation, before any guest code (§8.3/§10.3): counter 0; reason 0
            // (initial) — or reason 2 (upgrade-activation) on a migrating instance, journaled at
            // instantiation, BEFORE `da_init`/`da_migrate` (§10.3 step 4, never deferred).
            let inst_at = shared.now_ms();
            {
                let mut st = shared.state.lock().expect("pump lock");
                let reason = if migration.is_some() { 2 } else { 0 };
                st.sink.instantiation(0, reason, inst_at)?;
            }
            store.data_mut().slice.now = inst_at;

            // Write the admitted config + grants via da_alloc (outside import context, §2.4).
            let write_span = |store: &mut Store<Host>, bytes: &[u8]| -> Result<u32, RunError> {
                if bytes.is_empty() {
                    return Ok(0);
                }
                let alloc = instance
                    .get_typed_func::<(u32, u32), u32>(&mut *store, "da_alloc")
                    .map_err(|_| RunError::Sandbox("missing da_alloc".into()))?;
                let ptr = alloc
                    .call(&mut *store, (bytes.len() as u32, 1))
                    .map_err(|e| RunError::Sandbox(format!("da_alloc: {e}")))?;
                if ptr == 0 {
                    return Err(RunError::Sandbox("da_alloc returned 0".into()));
                }
                let mem = instance
                    .get_memory(&mut *store, "memory")
                    .ok_or_else(|| RunError::Sandbox("no exported memory".into()))?;
                mem.write(&mut *store, ptr as usize, bytes)
                    .map_err(|e| RunError::Sandbox(format!("config write: {e}")))?;
                Ok(ptr)
            };
            let cfg_ptr = write_span(&mut store, &run.config)?;
            let grants_ptr = write_span(&mut store, &run.grants)?;

            // da_init — once, on the run instance, imports illegal inside it (§3.1/§6.6).
            store.data_mut().slice.in_init = true;
            let da_init = instance
                .get_typed_func::<(u32, u32, u32, u32), u32>(&mut store, "da_init")
                .map_err(|_| RunError::Sandbox("missing/mis-typed da_init".into()))?;
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

            // -- the migrate step (§10.3 steps 4–5), on a migrating instance only ----------------
            if let Some(mig) = &migration {
                // Stage the snapshot's sections host-side under kind-3 staging IDs; the restore
                // IDs travel IN the descriptor (§10.2 — the module is not in `da_run` and sees
                // no PayloadReady).
                let bindings: Vec<(String, u64)> = {
                    let mut st = shared.state.lock().expect("pump lock");
                    mig.capture
                        .sections
                        .iter()
                        .map(|(name, bytes)| {
                            let id = st.next_host_staging_id;
                            st.next_host_staging_id += 1;
                            st.staged.insert(
                                id,
                                (daemon_vhc_abi::STAGED_KIND_STATE_SECTION, bytes.clone()),
                            );
                            (name.clone(), id)
                        })
                        .collect()
                };
                let descriptor = build_migration_descriptor(&mig.capture.manifest, &bindings)
                    .map_err(|e| RunError::Sandbox(format!("migration descriptor: {e}")))?;
                let desc_ptr = write_span(&mut store, &descriptor)?;

                let da_migrate = instance
                    .get_typed_func::<(u32, u32), u32>(&mut store, "da_migrate")
                    .map_err(|_| RunError::Sandbox("missing/mis-typed da_migrate".into()))?;
                // The explicit bounded budget (§10.2): fuel + the epoch deadline; exceeding it is
                // the typed `MigrateBudget` trap and the host rolls back.
                store
                    .set_fuel(mig.migrate_fuel.unwrap_or(engine_cfg.fuel_per_call))
                    .map_err(|e| RunError::Sandbox(e.to_string()))?;
                store.set_epoch_deadline(epoch_ticks);
                store.data_mut().slice.in_migrate = true;
                let migrate_status =
                    match da_migrate.call(&mut store, (desc_ptr, descriptor.len() as u32)) {
                        Ok(s) => s,
                        Err(e) => {
                            let mut trap = take_trap(&mut store, e);
                            // Budget exhaustion inside da_migrate is the typed MigrateBudget (§10.2).
                            if matches!(trap.code, TrapCode::BudgetFuel | TrapCode::BudgetEpoch) {
                                trap = Trap::new(
                                    TrapCode::MigrateBudget,
                                    "da_migrate",
                                    None,
                                    format!("migrate budget exhausted: {}", trap.detail),
                                );
                            }
                            journal_terminal_trap(&shared, &trap)?;
                            return Ok(RunEnd::Trapped(trap));
                        }
                    };
                store.data_mut().slice.in_migrate = false;
                store
                    .set_fuel(engine_cfg.fuel_per_call)
                    .map_err(|e| RunError::Sandbox(e.to_string()))?;
                if migrate_status != daemon_vhc_abi::DA_MIGRATE_READY {
                    // Validate failed (§10.3 step 5): journal the fact (a typed condition + the
                    // forced-interruption terminal — the instance never entered da_run) and tear
                    // down; the upgrade transaction rolls back and retries or leaves (step 7).
                    let mut st = shared.state.lock().expect("pump lock");
                    st.sink.condition(
                        "MigrateIncompatible",
                        &format!("da_migrate returned {migrate_status} (§10.2)"),
                    )?;
                    st.sink.terminal(
                        2,
                        None,
                        Some((
                            "MigrateIncompatible".to_string(),
                            "da_migrate".to_string(),
                            "da_migrate".to_string(),
                            format!("da_migrate returned {migrate_status}"),
                        )),
                    )?;
                    return Ok(RunEnd::MigrateRefused(migrate_status));
                }
                // Validate passed (§10.3 step 5): mark it embedder-visible — the upgrade
                // transaction gates activation on this, not on module-specific egress — and
                // wake any egress waiter.
                {
                    let mut st = shared.state.lock().expect("pump lock");
                    st.migrate_validated = true;
                    st.note_egress();
                }
            }

            // da_run — exactly once; the module owns its loop from here (§3.1).
            let da_run = instance
                .get_typed_func::<(), u32>(&mut store, "da_run")
                .map_err(|_| RunError::Sandbox("missing/mis-typed da_run".into()))?;
            let run_result = da_run.call(&mut store, ());
            {
                let mut st = shared.state.lock().expect("pump lock");
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
        .map_err(|e| RunError::Sandbox(format!("guest thread spawn: {e}")))?;

    Ok(Run { pump, thread })
}

/// Map a wasmtime error into the typed taxonomy: prefer the stashed host trap, else classify the
/// engine trap (fuel/epoch/unreachable/oob), mirroring the v1 driver's mapping (§7.6).
fn take_trap(store: &mut Store<Host>, e: wasmtime::Error) -> Trap {
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
    use std::sync::atomic::Ordering;

    use ciborium::value::Value;
    use daemon_vhc_abi::{COMP_ERR_HASH_MISMATCH, EV_TAG_STOP, FRAME_ENVELOPE_DOMAIN_V2};
    use daemon_vhc_proto::sign::verify_bytes;
    use daemon_vhc_proto::to_canonical_vec;
    use wasmtime::StoreLimitsBuilder;

    use crate::run::completion::{CompError, CompletionResult};
    use crate::run::journal::{MemorySink, SinkEntry};
    use crate::run::ops::OpRequest;

    use super::chunks::verify_covering_span;
    use super::host::build_signed_frame;
    use super::migration::decode_manifest_sections;
    use super::pump::{fire_due_timers, ArmedTimer};
    use super::*;

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
            buffers: BufferTable::new(0, 0, 0),
            ops: OpTable::new(0, 0),
            chunk_maps: std::collections::HashMap::new(),
            data_read_budget: 0,
            data_read_used: 0,
            streams: StreamTable::new(0),
            op_requests: Vec::new(),
            stop_enqueued: false,
            stop_cut: None,
            draining: false,
            drain_deadline_at: None,
            accepted_snapshot: None,
            egress_hook: None,
            migrate_validated: false,
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

    /// A chunked fixture: 80 bytes at chunk_size 32 (two full chunks + one short).
    fn chunk_fixture() -> (daemon_vhc_proto::ChunkMap, Vec<u8>) {
        let bytes: Vec<u8> = (0u8..80).collect();
        let map = daemon_vhc_proto::ChunkMap {
            chunk_size: 32,
            token_count: 40,
            byte_len: 80,
            chunk_hashes: daemon_vhc_proto::chunk_hashes(&bytes, 32),
        };
        (map, bytes)
    }

    #[test]
    fn covering_span_verification_accepts_true_chunks_and_refuses_lies() {
        let (map, bytes) = chunk_fixture();
        // The full span verifies; a mid-span range's covering chunks verify.
        assert_eq!(verify_covering_span(&map, 0, bytes.clone()).unwrap(), bytes);
        assert!(verify_covering_span(&map, 32, bytes[32..].to_vec()).is_ok());
        // One flipped byte in any covering chunk is a described refusal.
        let mut tampered = bytes.clone();
        tampered[40] ^= 0xFF;
        let err = verify_covering_span(&map, 0, tampered).unwrap_err();
        assert!(err.contains("chunk 1"), "{err}");
        // A truncated span is refused, never partially accepted.
        assert!(verify_covering_span(&map, 0, bytes[..40].to_vec())
            .unwrap_err()
            .contains("truncates chunk 1"));
        // A span past the chunk list is refused.
        let mut overlong = bytes.clone();
        overlong.extend_from_slice(&[0u8; 32]);
        assert!(verify_covering_span(&map, 0, overlong)
            .unwrap_err()
            .contains("past the chunk list"));
    }

    #[test]
    fn chunk_descriptor_decode_round_trips_and_rejects_malformed() {
        let (map, _) = chunk_fixture();
        let hashes: Vec<ciborium::value::Value> = map
            .chunk_hashes
            .iter()
            .map(|h| ciborium::value::Value::Bytes(h.0.to_vec()))
            .collect();
        let doc = ciborium::value::Value::Array(vec![
            ciborium::value::Value::from(map.chunk_size),
            ciborium::value::Value::from(map.token_count),
            ciborium::value::Value::from(map.byte_len),
            ciborium::value::Value::Array(hashes),
        ]);
        let desc = daemon_vhc_proto::to_canonical_vec(&doc).unwrap();
        let decoded = decode_chunk_descriptor(&desc).unwrap();
        assert_eq!(decoded, map);
        assert_eq!(decoded.fold(), map.fold());

        assert!(decode_chunk_descriptor(b"junk").is_err(), "not CBOR");
        // Degenerate geometry (chunk list shorter than the byte length needs) is refused.
        let bad = ciborium::value::Value::Array(vec![
            ciborium::value::Value::from(32u64),
            ciborium::value::Value::from(40u64),
            ciborium::value::Value::from(80u64),
            ciborium::value::Value::Array(vec![ciborium::value::Value::Bytes(vec![0u8; 32])]),
        ]);
        let bad_desc = daemon_vhc_proto::to_canonical_vec(&bad).unwrap();
        assert!(decode_chunk_descriptor(&bad_desc)
            .unwrap_err()
            .contains("degenerate"));
    }

    #[test]
    fn chunked_completion_verifies_covering_chunks_then_slices_the_range() {
        let (map, bytes) = chunk_fixture();
        let fold = map.fold();
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink.clone()));
        {
            let mut st = pump.shared.state.lock().unwrap();
            st.chunk_maps.insert(fold.0, map.clone());
        }
        // The guest asked for [40, 60); chunk 1 ([32, 64)) covers it entirely.
        let (span_off, span_len) =
            daemon_vhc_proto::covering_span(map.byte_len, map.chunk_size, 40, 60);
        assert_eq!((span_off, span_len), (32, 32));
        let request = OpRequest::ArtifactRange {
            hash: fold.0,
            range_off: 40,
            range_len: 20,
            span_off,
            span_len,
        };
        let op = {
            let mut st = pump.shared.state.lock().unwrap();
            let op = st.ops.begin(request.clone()).unwrap();
            st.op_requests.push((op, request));
            op
        };
        // A span answer with true chunks completes Ok(handle) carrying exactly the range.
        let handle = pump
            .complete_op(
                op,
                OpOutcome::RangeDone {
                    bytes: bytes[32..64].to_vec(),
                },
            )
            .unwrap()
            .expect("range completion mints a buffer");
        let st = pump.shared.state.lock().unwrap();
        let buf = st.buffers.resolve(handle).unwrap();
        assert_eq!(buf.as_slice(), &bytes[40..60]);
    }

    #[test]
    fn chunked_completion_refuses_tampered_spans_typed() {
        let (map, bytes) = chunk_fixture();
        let fold = map.fold();
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink.clone()));
        {
            let mut st = pump.shared.state.lock().unwrap();
            st.chunk_maps.insert(fold.0, map.clone());
        }
        let request = OpRequest::ArtifactRange {
            hash: fold.0,
            range_off: 0,
            range_len: 16,
            span_off: 0,
            span_len: 32,
        };
        let op = {
            let mut st = pump.shared.state.lock().unwrap();
            st.ops.begin(request).unwrap()
        };
        let mut lied = bytes[..32].to_vec();
        lied[3] ^= 0x01;
        let minted = pump
            .complete_op(op, OpOutcome::RangeDone { bytes: lied })
            .unwrap();
        assert!(minted.is_none(), "no buffer for a refused span");
        // The journaled completion is the typed HashMismatch — the guest never saw the bytes.
        let entries = sink.lock().unwrap().entries.clone();
        let completion = entries
            .iter()
            .find_map(|e| match e {
                SinkEntry::Completion { result, .. } => Some(result.clone()),
                _ => None,
            })
            .expect("completion journaled");
        let decoded = CompletionResult::decode(&completion).unwrap();
        assert!(matches!(
            decoded,
            CompletionResult::Err(CompError { code, .. })
                if code == COMP_ERR_HASH_MISMATCH
        ));
    }

    /// A whole-object answer for a chunked request (the in-process content-store seat) is
    /// span-extracted + chunk-verified by the pump — same trust path, different transfer shape.
    #[test]
    fn chunked_completion_accepts_whole_object_answers() {
        let (map, bytes) = chunk_fixture();
        let fold = map.fold();
        let pump = test_pump(Box::new(MemorySink::new()));
        {
            let mut st = pump.shared.state.lock().unwrap();
            st.chunk_maps.insert(fold.0, map.clone());
        }
        let request = OpRequest::ArtifactRange {
            hash: fold.0,
            range_off: 70,
            range_len: 0, // to the end
            span_off: 64,
            span_len: 16,
        };
        let op = {
            let mut st = pump.shared.state.lock().unwrap();
            st.ops.begin(request).unwrap()
        };
        let handle = pump
            .complete_op(
                op,
                OpOutcome::FetchDone {
                    artifact: bytes.clone(),
                },
            )
            .unwrap()
            .expect("whole-object answer verifies");
        let st = pump.shared.state.lock().unwrap();
        assert_eq!(st.buffers.resolve(handle).unwrap().as_slice(), &bytes[70..]);
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
    fn the_egress_hook_fires_on_registration_and_on_guest_egress() {
        // The embedder egress wake: registering fires once (nothing already landed is silently
        // unannounced), and every subsequent guest-egress landing fires it again. The hook is a
        // pure signal — the embedder still drains through `published`/`take_op_requests`.
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let pump = test_pump(Box::new(sink));
        let fires = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = fires.clone();
        pump.set_egress_hook(Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(
            fires.load(Ordering::SeqCst),
            1,
            "fires once at registration"
        );
        // Simulate guest egress landing under the pump lock (the import-body path).
        {
            let mut st = pump.shared.state.lock().unwrap();
            st.published.push((0, 0, b"frame".to_vec()));
            st.note_egress();
            st.metrics.push(("loss".into(), 1.0));
            st.note_egress();
        }
        assert_eq!(
            fires.load(Ordering::SeqCst),
            3,
            "each landing wakes the embedder"
        );
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
    fn stop_cut_already_passed_enqueues_stop_immediately_and_fences_timers() {
        let pump = test_pump(Box::new(MemorySink::new()));
        {
            let mut st = pump.shared.state.lock().unwrap();
            st.published.push((0, 0, b"frame".to_vec()));
            // A due timer armed at registration time: it must never fire past the cut.
            st.timers.push(ArmedTimer { id: 7, fire_at: 0 });
        }
        pump.stop_at_publishes(1, 0).unwrap();
        let mut st = pump.shared.state.lock().unwrap();
        assert!(st.stop_enqueued, "cut already passed: stop registers now");
        assert_eq!(st.queue.len(), 1);
        assert_eq!(st.queue[0].tag, EV_TAG_STOP);
        // The delivery loop's gate: with stop enqueued, due timers never fire (§4.4).
        fire_due_timers_gated(&mut st, 100).unwrap();
        assert_eq!(st.queue.len(), 1, "no Timer enters the stream behind Stop");
    }

    /// The `next_event` loop's exact firing condition, extracted for the gate assertion above.
    fn fire_due_timers_gated(st: &mut PumpState, now: u64) -> Result<(), Trap> {
        if !st.draining && !st.stop_enqueued {
            fire_due_timers(st, now)?;
        }
        Ok(())
    }

    #[test]
    fn stop_cut_pending_yields_to_explicit_stop_and_stays_idempotent() {
        let pump = test_pump(Box::new(MemorySink::new()));
        pump.stop_at_publishes(5, 0).unwrap();
        {
            let st = pump.shared.state.lock().unwrap();
            assert!(!st.stop_enqueued, "cut not reached: no stop yet");
            assert_eq!(st.stop_cut, Some((5, 0)));
        }
        pump.stop(1).unwrap();
        pump.stop(1).unwrap(); // idempotent
        pump.stop_at_publishes(0, 2).unwrap(); // registration after stop is a no-op
        let st = pump.shared.state.lock().unwrap();
        let stops = st.queue.iter().filter(|q| q.tag == EV_TAG_STOP).count();
        assert_eq!(stops, 1, "exactly one terminal Stop");
        assert_eq!(st.stop_cut, None, "an explicit stop clears the cut");
    }

    #[test]
    fn manifest_sections_decode_and_descriptor_round_trips() {
        use ciborium::value::Value;
        // A minimal §10.2 state-manifest: schema/module + one section decl.
        let section_bytes = b"counter-state".to_vec();
        let manifest = Value::Map(vec![
            (Value::Text("schema".into()), Value::Integer(1.into())),
            (Value::Text("module".into()), Value::Bytes(vec![7u8; 32])),
            (
                Value::Text("sections".into()),
                Value::Array(vec![Value::Map(vec![
                    (Value::Text("name".into()), Value::Text("counter".into())),
                    (Value::Text("schema".into()), Value::Integer(1.into())),
                    (
                        Value::Text("hash".into()),
                        Value::Bytes(blake3::hash(&section_bytes).as_bytes().to_vec()),
                    ),
                    (
                        Value::Text("size".into()),
                        Value::Integer((section_bytes.len() as u64).into()),
                    ),
                    (Value::Text("class".into()), Value::Integer(0.into())),
                ])]),
            ),
        ]);
        let mut manifest_bytes = Vec::new();
        ciborium::into_writer(&manifest, &mut manifest_bytes).unwrap();

        let decls = decode_manifest_sections(&manifest_bytes).expect("decodes");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].name, "counter");
        assert_eq!(decls[0].size, section_bytes.len() as u64);
        assert_eq!(&decls[0].hash, blake3::hash(&section_bytes).as_bytes());

        // The descriptor embeds the manifest value verbatim + the restore bindings (§10.2).
        let desc =
            build_migration_descriptor(&manifest_bytes, &[("counter".into(), 42)]).expect("builds");
        let v: Value = ciborium::de::from_reader(desc.as_slice()).unwrap();
        let Value::Map(entries) = v else {
            panic!("descriptor is a map")
        };
        let get = |name: &str| {
            entries
                .iter()
                .find_map(|(k, val)| match k {
                    Value::Text(t) if t == name => Some(val.clone()),
                    _ => None,
                })
                .expect("descriptor key")
        };
        assert_eq!(get("manifest"), manifest, "manifest embedded verbatim");
        let Value::Array(sections) = get("sections") else {
            panic!("sections is an array")
        };
        assert_eq!(sections.len(), 1);

        // Malformed manifests are refused, not misread.
        assert!(decode_manifest_sections(b"not-cbor").is_err());
        assert!(
            decode_manifest_sections(&[0xa0]).is_err(),
            "empty map: no sections"
        );
    }

    #[test]
    fn signed_frame_carries_the_full_scope_tuple_and_verifies() {
        // §12.1: [envelope, payload, sig]; the signature over the canonical envelope; every scope
        // field host-built. Verify with the plain proto primitives a third party would use.
        let signing = SigningKey::from_bytes(&[9u8; 32]);
        let sender = peer_id(&signing).0;
        let host = Host {
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
                in_migrate: false,
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
            migration_max_sections: 0,
            migration_max_section_bytes: 0,
            migration_restore: false,
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
