// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The run lifecycle: journal the run header, spawn the dedicated guest thread (one OS thread
//! per role-instance, §11.1), instantiate against the per-world linker, drive
//! `da_init` → (`da_migrate` under the §10.2 budget on a migrating instance) → `da_run`, and
//! journal the terminal fact — plus the wasmtime-error → typed-trap mapping (§7.6).

use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use wasmtime::{Linker, Module, Store, StoreLimitsBuilder};

use daemon_vhc_abi::{ExecutionContext, NS_COMPUTE_V2, NS_TABI_V1};
use daemon_vhc_proto::{peer_id, SigningKey};

use crate::run::buffer::BufferTable;
use crate::run::driver::config::{MigrationInput, RunConfig, RunEnd, RunError};
use crate::run::driver::host::{derive_rng_seed, Host, SliceState};
use crate::run::driver::linker::link_v2;
use crate::run::driver::migration::{build_migration_descriptor, RestoreBinding};
use crate::run::driver::pump::{BufferStreams, PumpHandle, PumpShared, PumpState};
use crate::run::journal::{JournalSink, SinkError};
use crate::run::ops::OpTable;
use crate::run::streams::StreamTable;
use crate::runtime::{EngineConfig, Worker};
use crate::trap::{Trap, TrapCode};

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
    // The negotiated major-2 minor. It selects the terminal-context rendering: a journal written at a
    // legacy minor keeps the bare string it has always carried, and the certification minor is the
    // first that records the truthful eleven-value context.
    let abi_minor = run.abi_minor;
    let abi_packed = (u64::from(daemon_vhc_abi::DA_ABI_MAJOR_V2) << 16) | u64::from(abi_minor);
    let worlds: Vec<(String, u64)> = module
        .imports()
        .map(|i| (i.module().to_string(), 0u64))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    // tag 0 first — the run header precedes everything (§8.3). The header's `bridge` field is
    // keep-reserved (always `false`: no bridge exists; the field stays so the record grammar is
    // unchanged and pre-existing journals stay parseable).
    let resources = run_header_resources(&run)?;
    sink.run_header(
        abi_packed,
        &worlds,
        false,
        &run.manifest_bytes,
        &run.config,
        &run.grants,
        resources,
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
            guest_panic: None,
            published: Vec::new(),
            // Generation-seeded by the instantiation counter (0: this driver instantiates once
            // per start_run; trap-restart re-seeding rides the tag-13 counter, ABI §7.1).
            buffers: BufferTable::new(0, run.max_live_buffer_handles, run.max_live_buffer_bytes),
            ops: OpTable::new(0, run.max_outstanding_ops),
            chunk_maps: std::collections::HashMap::new(),
            state_chunk_maps: std::collections::HashMap::new(),
            data_read_budget: run.data_read_budget_bytes,
            data_read_used: 0,
            streams: StreamTable::new(0),
            state: crate::run::state_store::StateStore::new_with_spill(
                crate::run::state_store::StateStoreConfig {
                    chunk_size: run.state_chunk_size,
                    streams_max: run.state_streams_max,
                    emit_max_bytes: run.state_emit_max_bytes,
                    write_rate_per_min: run.state_write_rate_per_min,
                    store_bytes_max: run.state_store_bytes,
                    retain_roots: run.state_retain_roots,
                },
                // Disk-back the state store when the run pins a state directory (design §8.1);
                // a spill that cannot open refuses the run (a provisioned state plane that
                // cannot persist must not silently fall back to RAM on the memory-floor peer).
                match &run.state_dir {
                    Some(dir) => Some(
                        crate::run::state_spill::SpillStore::open(dir)
                            .map_err(|e| RunError::Sandbox(format!("open state spill: {e}")))?,
                    ),
                    None => None,
                },
            ),
            guest_memory_high_water: 0,
            allocator_samples: Vec::new(),
            buffer_streams: BufferStreams::default(),
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

    // Seed the successor's state store with the sealed families the in-process live-module-switch
    // transaction carried from the draining instance ([SF-6]). Done here, before `da_run`, so the
    // streamed restore walk resolves the drain snapshot's folds self-sealed ([SF-R1], host-local)
    // — the switch is the one migrate where the same node keeps custody of canonical state, so no
    // content-plane fetch is needed (and none would resolve: the in-process switch publishes
    // nothing). Empty for a content-plane late-join restore.
    if let Some(mig) = &migration {
        if !mig.carried_state.is_empty() {
            let mut st = shared.state.lock().expect("pump lock");
            for fam in &mig.carried_state {
                st.state.inject_sealed_family(fam);
            }
        }
    }

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
                        // A host-side typed refusal, and **no guest-trap record at all**: no guest
                        // code has run, so there is no guest execution context to name and nothing
                        // about the module to report.
                        //
                        // This used to journal a terminal trap attributed to initialization. That was
                        // a classification bug, not a conservative choice — it recorded a guest-trap
                        // fact about a phase the guest never entered, and a reader would reasonably
                        // have concluded the module's own initialization failed when the truth was
                        // that this host could not bring up its device. Inventing a context for it
                        // would have been the same mistake with more machinery.
                        return Err(RunError::BackendBringUp {
                            stage: crate::run::driver::config::HOST_STAGE_BACKEND_BRING_UP,
                            backend: engine_cfg.backend.slug().to_string(),
                            reason,
                        });
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
                    in_run: false,
                    slice_ordinal: None,
                    delivered_completion_failure: None,
                    slices_delivered: 0,
                    log_calls_this_phase: 0,
                    log_bytes_this_phase: 0,
                    import_calls: 0,
                    import_calls_at_epoch_check: 0,
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
            // The epoch watchdog is a WEDGE detector, not a wall bound on legitimate work: a
            // device-lane slice spends its wall inside host compute imports (a ceremony-geometry
            // round is ONE slice, and its device time grows with the granted geometry — no
            // constant survives that), while a wedged guest spins in pure wasm making no import
            // calls at all. So on expiry the deadline EXTENDS while the guest's import-entry
            // count is advancing ([`SliceState::import_calls`], §5.6's never-kill-for-waiting
            // principle) and interrupts only a full budget with zero host contact — pure-wasm
            // wedges still die within two budgets, and the deterministic fuel/op budgets are
            // untouched.
            store.epoch_deadline_callback(move |mut cx| {
                let d = cx.data_mut();
                if d.slice.import_calls == d.slice.import_calls_at_epoch_check {
                    return Ok(wasmtime::UpdateDeadline::Interrupt);
                }
                d.slice.import_calls_at_epoch_check = d.slice.import_calls;
                Ok(wasmtime::UpdateDeadline::Continue(epoch_ticks))
            });

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

            // da_apply_execution_grant — exactly once, on the admitted run instance, BEFORE
            // `da_init` (ABI §2.1 at the certification minor; [RC-12]).
            //
            // The span is host-written and BORROWED, on the same lifetime convention as the
            // configuration and Capability Grants above: the guest decodes or copies it
            // synchronously and never frees it, the host does not free it either, and it is
            // reclaimed with the instance. That is why nothing below pairs a `da_free` with it —
            // an unpaired free here would be a double free the moment the store drops.
            //
            // The grant is deliberately NOT part of the Capability Grants: those are the bytes the
            // plan was derived from, and inserting the grant into them would make the grant an
            // input to its own derivation.
            if !run.execution_grant.is_empty() {
                let grant_ptr = write_span(&mut store, &run.execution_grant)?;
                let apply = instance
                    .get_typed_func::<(u32, u32), u32>(
                        &mut store,
                        daemon_vhc_abi::DA_APPLY_EXECUTION_GRANT_EXPORT,
                    )
                    .map_err(|_| {
                        RunError::Sandbox(format!(
                            "missing/mis-typed {}",
                            daemon_vhc_abi::DA_APPLY_EXECUTION_GRANT_EXPORT
                        ))
                    })?;
                let status =
                    match apply.call(&mut store, (grant_ptr, run.execution_grant.len() as u32)) {
                        Ok(status) => status,
                        Err(e) => {
                            // A trap here means initialization never occurs. The terminal record carries
                            // the grant-application context, and tag 18 is absent — the grammar admits
                            // exactly one of the result record or this trap.
                            let trap = take_trap(&mut store, e);
                            journal_terminal_trap(
                                &shared,
                                &trap,
                                &ExecutionContext::ExecutionGrant,
                                abi_minor,
                            )?;
                            return Ok(RunEnd::Trapped(trap));
                        }
                    };
                {
                    // Written exactly once after the export RETURNS, whatever the status, and before
                    // the tag-11 init record.
                    let mut st = shared.state.lock().expect("pump lock");
                    st.sink.execution_grant(
                        *blake3::hash(&run.execution_grant).as_bytes(),
                        u64::from(status),
                    )?;
                }
                if status != 0 {
                    // A deterministic, non-retryable refusal for this (module, plan, grant) tuple:
                    // a retry needs changed admitted input, not a fresh instance.
                    return Ok(RunEnd::ExecutionGrantRejected(status));
                }
            }

            // da_init — once, on the run instance. Every capability import is illegal inside it
            // except the observational `sys@2::log` exemption (§3.1/§6.6, §6.6.1). Each exempt phase
            // gets its OWN budget, reset as the phase is entered.
            store.data_mut().slice.in_init = true;
            store.data_mut().slice.log_calls_this_phase = 0;
            store.data_mut().slice.log_bytes_this_phase = 0;
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
                    journal_terminal_trap(&shared, &trap, &ExecutionContext::Init, abi_minor)?;
                    return Ok(RunEnd::Trapped(trap));
                }
            };
            store.data_mut().slice.in_init = false;
            sample_allocator_at(&store, &shared, crate::compute::SamplePoint::AfterInit);
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
                // Build the restore bindings, in manifest order ([SF-6]): INLINE sections stage
                // host-side under kind-3 staging IDs (read via `read_back(kind=3)` in
                // `da_migrate`); BY-REFERENCE families carry their FamilyRef so the new instance
                // registers the fold ([SF-R2]) and streams it in `da_run` — zero section bytes move
                // through the migrate seam. The bindings travel IN the descriptor (§10.2 — the
                // module is not in `da_run` and sees no PayloadReady).
                use daemon_vhc_proto::det_state::CkptDocSection;
                let bindings: Vec<RestoreBinding> = {
                    let mut st = shared.state.lock().expect("pump lock");
                    mig.capture
                        .sections
                        .iter()
                        .map(|section| match section {
                            CkptDocSection::Inline(name, bytes) => {
                                let id = st.next_host_staging_id;
                                st.next_host_staging_id += 1;
                                st.staged.insert(
                                    id,
                                    (daemon_vhc_abi::STAGED_KIND_STATE_SECTION, bytes.clone()),
                                );
                                RestoreBinding::Inline {
                                    name: name.clone(),
                                    staging_id: id,
                                }
                            }
                            CkptDocSection::ByRef(name, family) => RestoreBinding::ByRef {
                                name: name.clone(),
                                family: family.clone(),
                            },
                        })
                        .collect()
                };
                // Grant the by-ref family folds on the NEW instance before `da_run`, so its
                // `register_state_chunks` ([SF-R2]) + streamed `data@2::fetch` resolve (the family
                // chunks are served from the content plane by the chunk-keyed resolver).
                for section in &mig.capture.sections {
                    if let CkptDocSection::ByRef(_, family) = section {
                        store.data_mut().granted_artifacts.insert(family.fold.0);
                    }
                }
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
                store.data_mut().slice.log_calls_this_phase = 0;
                store.data_mut().slice.log_bytes_this_phase = 0;
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
                            journal_terminal_trap(
                                &shared,
                                &trap,
                                &ExecutionContext::Migrate,
                                abi_minor,
                            )?;
                            return Ok(RunEnd::Trapped(trap));
                        }
                    };
                store.data_mut().slice.in_migrate = false;
                sample_allocator_at(&store, &shared, crate::compute::SamplePoint::AfterMigrate);
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
                    // A chain-founding migration (late join / crash reconstruction) journals the
                    // validated restore manifest as the new chain's anchoring tag-10 BEFORE any
                    // event lands, so the successor chain is self-contained: replay and
                    // reconstruction re-anchor here instead of walking predecessor-chain state
                    // (§8.3/§8.8). The live switch never sets this — its seam continues the
                    // retiring chain, which already carries the drain snapshot.
                    if mig.anchor {
                        st.sink.snapshot(&mig.capture.manifest)?;
                    }
                    st.migrate_validated = true;
                    st.note_egress();
                }
            }

            // da_run — exactly once; the module owns its loop from here (§3.1).
            let da_run = instance
                .get_typed_func::<(), u32>(&mut store, "da_run")
                .map_err(|_| RunError::Sandbox("missing/mis-typed da_run".into()))?;
            store.data_mut().slice.in_run = true;
            sample_allocator_at(&store, &shared, crate::compute::SamplePoint::AfterBringUp);
            let run_result = da_run.call(&mut store, ());
            sample_allocator_at(&store, &shared, crate::compute::SamplePoint::AtTeardown);
            {
                let mut st = shared.state.lock().expect("pump lock");
                // Force-reclaim the instance's buffers + outstanding ops + streams through the
                // per-instance tables (architecture §3.4; ABI §7.3) — guest-thread-owned teardown.
                st.buffers.clear();
                st.ops.clear();
                st.streams.clear();
                st.op_requests.clear();
                // Torn-fold GC (ABI §12.14 [SF-4]): opened-but-unsealed state streams are never
                // durable — their staged chunks drop here; only sealed folds outlive the slice
                // (and the store itself is instance-scoped).
                st.state.clear_open();
            }
            match run_result {
                Ok(outcome) => {
                    let mut st = shared.state.lock().expect("pump lock");
                    st.sink.terminal(0, Some(u64::from(outcome)), None)?;
                    Ok(RunEnd::Outcome(outcome))
                }
                Err(e) => {
                    // The run-phase context is read from the slice state BEFORE the trap is taken,
                    // so it is the state the trap actually occurred in — one of four distinct
                    // values, never an invented slice ordinal.
                    let context = store.data().slice.execution_context();
                    let trap = take_trap(&mut store, e);
                    journal_terminal_trap(&shared, &trap, &context, abi_minor)?;
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
///
/// A guest that panicked forwarded its message through `sys@2::log` a beat before the panic
/// runtime executed `unreachable` (ABI [`daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX`]); that message
/// is lifted to the FRONT of the detail here, so a `GuestPanic` names the assertion that failed
/// and the `file:line:col` it failed at instead of only the engine's backtrace.
/// [`ExecutionContext`]-scoped: the held message is lifted **only** when the trap's context is
/// identical to the context the message was emitted in. A held message whose context does not match
/// is discarded, and the trap keeps the detail it would otherwise have had.
///
/// The check lives at the single point of consumption rather than as a sweep at every phase
/// boundary, because boundaries get added over time and a missed one fails silently and
/// misleadingly — the failure mode being an authoritative-looking source location that belongs to a
/// different phase and a different bug.
pub(super) fn take_trap(store: &mut Store<Host>, e: wasmtime::Error) -> Trap {
    let context = store.data().slice.execution_context();
    let forwarded = {
        let shared = store.data().shared.clone();
        let mut st = shared.state.lock().expect("pump lock");
        // Consuming clears the slot either way: a message that did not match this trap is not left
        // behind to be mis-lifted into the next one.
        match st.guest_panic.take() {
            Some((emitted_in, message)) if emitted_in == context => Some(message),
            _ => None,
        }
    };
    let mut trap = classify_trap(store, e);
    if let Some(message) = forwarded {
        trap.detail = if trap.detail.is_empty() {
            message
        } else {
            format!("{message} — {}", trap.detail)
        };
    }
    // The failed completion the trapping slice consumed, if any (REL-4 attribution evidence).
    // No context comparison is needed: the evidence lives on the ACTIVE slice and is cleared the
    // moment the guest asks for the next event, so whatever is here at trap time belongs to the
    // slice the trap occurred in — a between-slices or later-slice trap reads `None` by
    // construction.
    trap.env_completion = store.data().slice.delivered_completion_failure.clone();
    trap
}

/// Which resources the run header records — selected by the **negotiated minor**, and fail-closed.
///
/// The minor decides, not which fields happen to be populated. Reading it off the data would make an
/// empty declared claim on a certification-minor run look like a legacy record, and a legacy run that
/// never populated the composed fields look like a broken certification one.
///
/// Which leaves the case the minor alone cannot handle: a run that declares the certification minor and
/// carries no composition. Writing the composed branch there would produce a header whose members claim
/// a composition happened over empty bytes — a record asserting a fact about a run that has none. In
/// production admission refuses such a run before it starts, so this is the second gate on the same
/// rule, and it exists because the first one can be bypassed by any caller assembling a `RunConfig`
/// directly.
///
/// # Errors
/// [`RunError::CompositionMissing`] naming the first absent member.
pub(crate) fn run_header_resources(
    run: &RunConfig,
) -> Result<crate::run::RunHeaderResources<'_>, RunError> {
    if !daemon_vhc_abi::run_header_is_certification_variant(run.abi_minor) {
        return Ok(crate::run::RunHeaderResources::Declared(&run.claim_bytes));
    }
    for (member, bytes) in [
        ("resource_plan", &run.resource_plan_bytes),
        ("physical_estimate", &run.physical_estimate_bytes),
        ("aggregate_estimate", &run.aggregate_claim_bytes),
        ("execution_grant", &run.execution_grant),
    ] {
        if bytes.is_empty() {
            return Err(RunError::CompositionMissing {
                minor: run.abi_minor,
                member,
            });
        }
    }
    Ok(crate::run::RunHeaderResources::Composed {
        resource_plan: &run.resource_plan_bytes,
        physical_estimate: &run.physical_estimate_bytes,
        aggregate_estimate: &run.aggregate_claim_bytes,
        execution_grant: &run.execution_grant,
    })
}

fn classify_trap(store: &mut Store<Host>, e: wasmtime::Error) -> Trap {
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
    } else if low.contains("epoch") || low.contains("interrupt") {
        // wasmtime reports an epoch-deadline expiry as `wasm trap: interrupt` (the epoch trap
        // variant IS `Trap::Interrupt`); nothing else arms interruption on this engine, so an
        // interrupt is the epoch watchdog — without this arm it fell through to `BadModule`,
        // filing a wall-clock budget exhaustion as a malformed module.
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

/// Journal a terminal trap with its **real** execution context.
///
/// The context used to be the literal `"da_run"` for every trap, whatever phase it occurred in, so an
/// initialization trap was recorded as a run-loop trap. That falsified the one field a replay verdict
/// can compare — the detail string is diagnostic text and is deliberately not comparable — and it
/// made a correct in-memory diagnosis read as a misattribution bug in the forwarding, sending the
/// reader to investigate the wrong mechanism.
///
/// Rendering is selected by the negotiated ABI minor: a journal written at a legacy minor keeps the
/// bare string it has always carried, because those bytes are evidence and evidence is not rewritten.
pub(super) fn journal_terminal_trap(
    shared: &Arc<PumpShared>,
    trap: &Trap,
    context: &ExecutionContext,
    abi_minor: u32,
) -> Result<(), SinkError> {
    let mut st = shared.state.lock().expect("pump lock");
    st.sink.terminal(
        1,
        None,
        Some((
            trap.code.slug().to_string(),
            trap.import.to_string(),
            context.render_for_minor(abi_minor),
            trap.detail.clone(),
        )),
    )
}

/// Take one backend-allocator reading at a phase boundary and record it in order.
///
/// Boundaries rather than a timer, and **in process**: an external sampler cannot see a phase
/// boundary at all — on one fleet platform the device phase is around two milliseconds inside a
/// process lasting fifty, so sampling from outside misses the shape entirely, which is the shape a
/// pooling term is calibrated against.
///
/// A backend that cannot report occupancy records nothing. It deliberately does not record a zero: a
/// zero is a measurement, absence is not, and a profile calibrated against a manufactured zero would
/// be calibrated against nothing at all.
fn sample_allocator_at(
    store: &Store<Host>,
    shared: &Arc<PumpShared>,
    point: crate::compute::SamplePoint,
) {
    let Some(compute) = store.data().compute.as_ref() else {
        return;
    };
    let Some(sample) = compute.sample_allocator() else {
        return;
    };
    shared
        .state
        .lock()
        .expect("pump lock")
        .allocator_samples
        .push((point, sample));
}
