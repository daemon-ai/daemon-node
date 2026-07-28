// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **fit-probe mode** (`DAEMON_TRAIN_FIT_PROBE`, `[RC-15]`): the device is the oracle.
//!
//! The composed estimate is demoted to a conservative figure — it refuses cheaply and sizes the
//! enforced budget, and nothing more. The authority on "does this (module, backend revision,
//! plan, grant, budget) tuple fit on THIS box?" is the box: this mode runs the actual module on
//! the actual measured backend at the actual granted geometry under the actual enforced budget,
//! in the same sandbox that contains an admitted run, and records what happened as a
//! content-addressed, memoized [`FitVerdict`](daemon_vhc_resource::FitVerdict).
//!
//! It lives in the worker binary and nowhere else because the verdict's backend-revision digest
//! names a **sealed binary identity** — the record admission compares is the one this very
//! executable makes about itself, so a probe hosted by any other process would record evidence
//! about an executable nobody runs.
//!
//! The worker authors nothing: every drive input (config, requirements, control frame, staged
//! batches, completion condition) arrives pre-authored through the
//! [`daemon_vhc_resource::probe`] directory contract, and the worker delivers the bytes exactly
//! the way production frames reach it — opaque. The round vocabulary stays outside this binary
//! (the dep-check-enforced opaque-host boundary).
//!
//! ## What is and is not a verdict
//!
//! - The **estimate refusing** (funnel refusal, un-provisioned box, unauthenticated profile) is a
//!   typed probe *failure* before the probe ran — sound refusal needs no device time, and no
//!   verdict is recorded (an absent verdict never means "the estimate answered instead").
//! - The run reaching its committed voice is a **green** verdict carrying the measured peak.
//! - The run ending contained — a typed trap, a nonzero outcome, a refused init or grant — is a
//!   **red** verdict carrying the trap's stable slug: evidence about a configuration, not an
//!   outage.
//! - A run that neither completes nor ends inside the deadline is a probe **failure** (no
//!   verdict): a wedged probe is not evidence.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    admit, start_run, Dropped, JournalSink, OpOutcome, OpRequest, OwnerPolicy, RunConfig, RunEnd,
    RunIdentity, SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::{from_canonical_slice, Hash};
use daemon_vhc_resource::probe as contract;
use daemon_vhc_resource::{
    revision_record_digest, FitOutcome, FitProbeKey, FitVerdict, FIT_VERDICT_SCHEMA,
};
use daemon_vhc_session::protocol::{AdmittedResource, ComposedResource};

use crate::backend;

/// The env switch: names the probe directory ([`daemon_vhc_resource::probe`] layout).
pub(crate) const FIT_PROBE_ENV: &str = "DAEMON_TRAIN_FIT_PROBE";

/// Run one fit probe over the directory's inputs and write its verdict there.
///
/// # Errors
/// A human-readable failure when the probe could not produce a verdict — unreadable inputs, a
/// refused admission (the estimate's sound refusal), a host-lane failure, or a wedged drive.
/// A **red verdict is not an error**: a contained run records `FitOutcome::Contained` and this
/// returns `Ok`.
pub(crate) fn run_fit_probe(dir: &Path) -> Result<(), String> {
    let read = |name: &str| {
        std::fs::read(dir.join(name)).map_err(|e| format!("fit probe: read {name}: {e}"))
    };
    let module = read(contract::MODULE_FILE)?;
    let config = read(contract::CONFIG_FILE)?;
    let requirements_bytes = read(contract::REQUIREMENTS_FILE)?;
    let open_frame = read(contract::OPEN_FRAME_FILE)?;
    let drive: contract::FitProbeDrive = from_canonical_slice(&read(contract::DRIVE_FILE)?)
        .map_err(|e| format!("fit probe: decode {}: {e}", contract::DRIVE_FILE))?;
    let requirements: daemon_vhc_proto::RoleExecutionRequirements =
        from_canonical_slice(&requirements_bytes)
            .map_err(|e| format!("fit probe: decode {}: {e}", contract::REQUIREMENTS_FILE))?;
    let staged = read_staged(&dir.join(contract::STAGE_DIR))?;
    let module_hash = *blake3::hash(&module).as_bytes();

    // -- admission: the estimate refuses cheaply and sizes the budget; it proves nothing ---------
    // The same assembly the worker's assess runs (measured selection, provisioned resource
    // authority, the certification funnel), so the probe stands exactly where an admitted join
    // stands — with the Admission kept, because the probe needs its budget figures and its
    // composed identity, not an eligibility projection of them.
    let (admission, grants, composed, revision_digest) =
        admit_for_probe(&module, module_hash, &config, &drive.role, &requirements)?;

    // -- the engine an admitted join builds; its cap is the enforced budget the key names --------
    let engine = backend::engine_for_join(None, admission.admitted_host_bytes())?;
    let budget_bytes = engine.max_memory_bytes as u64;
    let key = FitProbeKey {
        module_hash: Hash(module_hash),
        backend_revision_digest: revision_digest,
        plan_hash: Hash(composed.logical_resource_plan_hash),
        grant_hash: Hash(composed.execution_grant_hash),
        budget_bytes,
    };
    let key_digest = key
        .digest()
        .map_err(|e| format!("fit probe: address the probe key: {e}"))?;
    eprintln!(
        "fit-probe: key {} (module {}, revision {}, plan {}, grant {}, budget {budget_bytes} B)",
        key_digest.to_hex(),
        Hash(module_hash).to_hex(),
        key.backend_revision_digest.to_hex(),
        key.plan_hash.to_hex(),
        key.grant_hash.to_hex(),
    );

    // -- the run: the admitted wiring, driven over the pre-authored inputs -----------------------
    let outcome = drive_probe(
        &engine,
        &admission,
        grants,
        &module,
        module_hash,
        &config,
        &drive,
        DriveInputs {
            open_frame,
            staged,
            state_dir: dir.join("state"),
        },
    )?;

    let verdict = FitVerdict {
        schema: FIT_VERDICT_SCHEMA,
        key,
        outcome,
    };
    let bytes = verdict
        .to_canonical_bytes()
        .map_err(|e| format!("fit probe: encode the verdict: {e}"))?;
    let path = contract::verdict_path(dir, &key_digest.to_hex());
    std::fs::write(&path, bytes)
        .map_err(|e| format!("fit probe: write {}: {e}", path.display()))?;
    let verdict_digest = verdict
        .digest()
        .map_err(|e| format!("fit probe: address the verdict: {e}"))?;
    match &verdict.outcome {
        FitOutcome::Fits { measured_peak_bytes } => println!(
            "fit-verdict GREEN {} (peak {measured_peak_bytes} B under budget {budget_bytes} B) -> {}",
            verdict_digest.to_hex(),
            path.display()
        ),
        FitOutcome::Contained { trap_slug } => println!(
            "fit-verdict RED {} (contained: {trap_slug}) -> {}",
            verdict_digest.to_hex(),
            path.display()
        ),
    }
    Ok(())
}

/// The staged payloads, in ascending file-name order (the orchestrator's delivery order).
fn read_staged(stage: &Path) -> Result<Vec<Vec<u8>>, String> {
    if !stage.exists() {
        return Ok(Vec::new());
    }
    let mut names: Vec<std::path::PathBuf> = std::fs::read_dir(stage)
        .map_err(|e| format!("fit probe: read {}: {e}", stage.display()))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("fit probe: read {}: {e}", stage.display()))?;
    names.sort();
    names
        .into_iter()
        .map(|p| std::fs::read(&p).map_err(|e| format!("fit probe: read {}: {e}", p.display())))
        .collect()
}

/// The probe's admission: the worker's own assess assembly, with the [`daemon_vhc_host::run::
/// admission::Admission`] kept.
fn admit_for_probe(
    module: &[u8],
    module_hash: [u8; 32],
    config: &[u8],
    role: &str,
    requirements: &daemon_vhc_proto::RoleExecutionRequirements,
) -> Result<
    (
        daemon_vhc_host::run::admission::Admission,
        Vec<u8>,
        Box<ComposedResource>,
        Hash,
    ),
    String,
> {
    // The roomy real-scale admission engine (the training-step gate's choice): assessment stays
    // CPU-cheap, but a real-geometry parameter layout exceeds the tiny-model default budgets.
    let admission_worker = Worker::new(EngineConfig::real_model(BackendKind::Cpu, None))
        .map_err(|e| format!("fit probe: admission engine: {e}"))?;
    let selection = backend::measured_backend(None)?;
    let lane = backend::selected_lane();
    let hw = backend::hardware();
    let dl = backend::device_limits();
    let device = backend::admission_device_profile(&hw, &dl);
    let owner = OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    let default_role = daemon_vhc_proto::genesis::RoleGrants::default();
    let grants = backend::derive_grants(&admission_worker, module, &default_role)?;
    let parts = backend::resource_authority_parts(&selection, role, Some(requirements))?
        .ok_or_else(|| {
            format!(
                "fit probe: this box assembles no resource authority for role `{role}` — \
                 provision it (`DAEMON_VHC_PROFILE_DIR`, `xtask vhc-provision-dev-profile`) \
                 before probing"
            )
        })?;
    let revision_digest = revision_record_digest(&parts.running)
        .map_err(|e| format!("fit probe: digest the running revision record: {e}"))?;
    let profile = parts.select().map_err(|refusal| {
        format!("fit probe: the provisioned profile did not authenticate: {refusal}")
    })?;
    let authority = parts.authority(&profile);
    let admission = admit(
        &admission_worker,
        module,
        Some(&module_hash),
        config,
        &grants,
        &lane,
        &device,
        &owner,
        None,
        None,
        Some(&authority),
    )
    .map_err(|refusal| {
        format!(
            "fit probe: the estimate refused before the probe ran (sound refusal — no device \
             time spent, no verdict recorded): {refusal}"
        )
    })?;
    let composed = match AdmittedResource::from_admission(&admission)
        .map_err(|e| format!("fit probe: state the composed resource identity: {e}"))?
    {
        AdmittedResource::Composed(c) => c,
        AdmittedResource::Declared { .. } => {
            return Err(
                "fit probe: the module admits by declared claim (below the certification \
                 minor); the fit probe is certification-minor machinery — there is no plan or \
                 grant to key a verdict by"
                    .into(),
            )
        }
    };
    Ok((admission, grants, composed, revision_digest))
}

/// The opaque drive inputs.
struct DriveInputs {
    open_frame: Vec<u8>,
    staged: Vec<Vec<u8>>,
    state_dir: std::path::PathBuf,
}

/// Run the admitted instance over the pre-authored inputs and say what the device said.
///
/// `Ok(FitOutcome)` is a verdict either way — green with the measured peak, or red with the
/// containing trap's slug. `Err` is a probe failure: nothing was learned about the fit.
#[allow(clippy::too_many_arguments)] // the admitted run's own wiring, spelled once
fn drive_probe(
    engine: &EngineConfig,
    admission: &daemon_vhc_host::run::admission::Admission,
    grants: Vec<u8>,
    module: &[u8],
    module_hash: [u8; 32],
    config: &[u8],
    drive: &contract::FitProbeDrive,
    inputs: DriveInputs,
) -> Result<FitOutcome, String> {
    let worker =
        Worker::new(engine.clone()).map_err(|e| format!("fit probe: probe engine: {e}"))?;
    let identity = RunIdentity {
        run_id: [0xF1; 32],
        epoch: 0,
        role: drive.role.clone(),
        instance: 1,
        module: module_hash,
    };
    // The admission-derived run wiring, mirrored from the join path (`role_binding`): the claim
    // and manifest bytes, the exactly-metered staging ceiling, the composed members the header
    // records, the admitted quotas — then the genesis-pinned members the orchestrator states as
    // data (state geometry, lane grants).
    let mut run_cfg = RunConfig::new(identity, [0x9D; 32], config.to_vec(), grants);
    run_cfg.claim_bytes = admission.claim_bytes.clone();
    run_cfg.manifest_bytes = admission.manifest_bytes.clone();
    run_cfg.hard_accountable_host_bytes = admission.hard_accountable_host_bytes();
    admission
        .apply_composition(&mut run_cfg)
        .map_err(|e| format!("fit probe: record the composed estimate: {e}"))?;
    admission.apply_quotas(&mut run_cfg);
    if drive.state_chunk_size > 0 {
        run_cfg.state_chunk_size = drive.state_chunk_size;
    }
    if let Some(depth) = drive.compute_queue_depth {
        run_cfg.compute_queue_depth = depth;
    }
    if let Some(bytes) = drive.max_readback_bytes_per_slice {
        run_cfg.max_readback_bytes_per_slice = bytes;
    }
    if let Some(bytes) = drive.max_live_buffer_bytes {
        run_cfg.max_live_buffer_bytes = bytes;
    }
    if let Some(handles) = drive.max_live_buffer_handles {
        run_cfg.max_live_buffer_handles = handles;
    }
    // Disk-back the state plane (the join does the same under a run-state home): the retained
    // det-lane roots at a real geometry are GiB-scale, and a probe that held them resident would
    // measure its own harness.
    run_cfg.state_dir = Some(inputs.state_dir);

    let sink = ProbeSink::shared();
    let t0 = Instant::now();
    let run = start_run(&worker, module, run_cfg, Box::new(sink.clone()))
        .map_err(|e| format!("fit probe: start: {e}"))?;
    let pump = run.pump.clone();

    // Stage, then open: both queue while the guest streams its init; it observes them at its
    // first `next_event`.
    for wrapper in inputs.staged {
        pump.stage_payload(wrapper, None)
            .map_err(|e| format!("fit probe: stage payload: {e}"))?;
    }
    let verdict = pump
        .deliver_frame(
            0,
            0,
            [9u8; 32],
            inputs.open_frame.clone(),
            inputs.open_frame,
        )
        .map_err(|e| format!("fit probe: deliver the open frame: {e}"))?;
    if verdict != daemon_vhc_host::run::DeliverVerdict::Accepted {
        return Err(format!(
            "fit probe: the open frame was not accepted ({verdict:?}) — the drive inputs do not \
             match the run they were authored for"
        ));
    }

    let deadline = t0 + Duration::from_secs(drive.deadline_s);
    // The guest's own `sys@2::log` lines, streamed to stderr as they land: the probe's slice
    // timeline. A RED verdict names WHAT contained the run (the trap slug); these lines are the
    // only record of WHERE the wall time went before it did — without them a `BudgetEpoch` on a
    // device lane is unactionable evidence.
    let mut log_cursor = 0usize;
    let mut drain_logs = |pump: &daemon_vhc_host::run::PumpHandle| {
        let logs = pump.logs();
        for (level, line) in logs.iter().skip(log_cursor) {
            eprintln!(
                "fit-probe: guest[{level}] {line} (+{:.1} s)",
                t0.elapsed().as_secs_f64()
            );
        }
        log_cursor = logs.len();
    };
    loop {
        drain_logs(&pump);
        // The drive answers exactly one embedder op: the committed container's durable PUT. Any
        // other request means these inputs need a plane this runner does not carry — a probe
        // failure, not a verdict.
        for (op, request) in pump.take_op_requests() {
            match request {
                OpRequest::PayloadPut { .. } => {
                    pump.complete_op(op, OpOutcome::PutDone)
                        .map_err(|e| format!("fit probe: complete the container PUT: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "fit probe: the drive cannot answer this op request (the probe carries \
                         no content plane): {other:?}"
                    ))
                }
            }
        }
        let committed = sink
            .lock()
            .expect("probe sink")
            .publishes
            .iter()
            .any(|&(tag, round)| tag == drive.commit_tag && round == drive.commit_round);
        if committed {
            let peak = pump.guest_memory_high_water();
            eprintln!(
                "fit-probe: committed ({}, {}) at {:.1} s; peak guest linear memory {peak} B",
                drive.commit_tag,
                drive.commit_round,
                t0.elapsed().as_secs_f64()
            );
            pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
                .map_err(|e| format!("fit probe: stop: {e}"))?;
            return match run.wait() {
                Ok(RunEnd::Outcome(0)) => Ok(FitOutcome::Fits {
                    measured_peak_bytes: peak,
                }),
                Ok(other) => Err(format!(
                    "fit probe: the run committed but did not end cleanly: {other:?}"
                )),
                Err(e) => Err(format!(
                    "fit probe: the run committed but the host lane failed: {e}"
                )),
            };
        }
        if run.is_finished() {
            // The run ended before its commitment: a contained, typed end is a RED verdict; a
            // host-lane failure is a probe failure (evidence about the host, not the fit).
            let terminal = sink.lock().expect("probe sink").terminal.clone();
            return match run.wait() {
                Ok(RunEnd::Trapped(trap)) => {
                    // The verdict memoizes only the stable slug; the trap's full context (import,
                    // entry point, detail) is one-shot forensic evidence, so surface it where the
                    // probe's operator is already looking. A RED verdict that names only a slug
                    // sends the reader back to a re-run with extra instrumentation — this line IS
                    // that instrumentation, always on.
                    eprintln!("fit-probe: contained: {trap}");
                    Ok(FitOutcome::Contained {
                        trap_slug: trap.code.slug().to_string(),
                    })
                }
                Ok(RunEnd::Outcome(code)) => Ok(FitOutcome::Contained {
                    trap_slug: format!("Outcome:{code}"),
                }),
                Ok(RunEnd::InitRefused(status)) => Ok(FitOutcome::Contained {
                    trap_slug: format!("InitRefused:{status}"),
                }),
                Ok(RunEnd::ExecutionGrantRejected(status)) => Ok(FitOutcome::Contained {
                    trap_slug: format!("ExecutionGrantRejected:{status}"),
                }),
                Ok(RunEnd::MigrateRefused(status)) => Err(format!(
                    "fit probe: a fresh instance reported MigrateRefused({status}) — that end \
                     cannot come from this drive; terminal {terminal:?}"
                )),
                Err(e) => Err(format!(
                    "fit probe: the run ended with a HOST-lane failure, not a typed containment \
                     (terminal {terminal:?}): {e}"
                )),
            };
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "fit probe: the drive neither committed nor ended within {} s (published {:?}, \
                 peak guest linear memory {} B) — a wedged probe is not evidence",
                drive.deadline_s,
                sink.lock().expect("probe sink").publishes,
                pump.guest_memory_high_water(),
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A journal sink that keeps only what the drive reads: the `(tag, round)` head of each publish
/// and the terminal record. A real-geometry probe reads GiB of θ back through the journal seam;
/// retaining it would measure the harness, not the module.
///
/// It also echoes each lifecycle record to stderr with its elapsed offset: the probe's slice
/// timeline. A `BudgetEpoch` verdict on a device lane is unactionable without knowing WHICH slice
/// spent the wall — the guest under probe emits no logs of its own, and these journal seams are
/// the only host-visible slice boundaries.
struct ProbeSink {
    publishes: Vec<(u64, u64)>,
    terminal: Option<String>,
    next_seq: u64,
    started: Instant,
}

impl ProbeSink {
    fn shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            publishes: Vec::new(),
            terminal: None,
            next_seq: 0,
            started: Instant::now(),
        }))
    }

    fn mark(&self, what: &std::fmt::Arguments<'_>) {
        eprintln!(
            "fit-probe: +{:.1} s {what}",
            self.started.elapsed().as_secs_f64()
        );
    }
}

impl JournalSink for ProbeSink {
    fn run_header(
        &mut self,
        _abi: u64,
        _worlds: &[(String, u64)],
        _bridge: bool,
        _manifest: &[u8],
        _config: &[u8],
        _grants: &[u8],
        _resources: daemon_vhc_host::run::RunHeaderResources<'_>,
        _channels: &[u8],
        _device: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn instantiation(&mut self, counter: u64, reason: u64, _at: u64) -> Result<(), SinkError> {
        self.mark(&format_args!("instantiation #{counter} (reason {reason})"));
        Ok(())
    }
    fn init(&mut self, _c: [u8; 32], _g: [u8; 32], status: u64) -> Result<(), SinkError> {
        self.mark(&format_args!("init journaled (status {status})"));
        Ok(())
    }
    fn execution_grant(&mut self, _hash: [u8; 32], status: u64) -> Result<(), SinkError> {
        self.mark(&format_args!("execution grant (status {status})"));
        Ok(())
    }
    fn event(&mut self, _at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.mark(&format_args!("event delivered ({} B)", frame.len()));
        Ok(())
    }
    fn signed_frame(
        &mut self,
        _channel: u64,
        _seq: u64,
        _sender: [u8; 32],
        _frame: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn next_seq(&mut self, _channel: u64) -> u64 {
        self.next_seq
    }
    fn publish(
        &mut self,
        _channel: u64,
        seq: u64,
        payload: &[u8],
        _frame: &[u8],
    ) -> Result<(), SinkError> {
        self.next_seq = seq + 1;
        // The generic `[uint, uint, ...]` head peek the drive's completion condition compares
        // against — data handed to the worker, never a schema linked by it.
        if let Ok(ciborium::value::Value::Array(items)) =
            ciborium::de::from_reader::<ciborium::value::Value, _>(payload)
        {
            let uint = |i: usize| -> Option<u64> {
                items
                    .get(i)
                    .and_then(ciborium::value::Value::as_integer)
                    .and_then(|n| u64::try_from(i128::from(n)).ok())
            };
            if let (Some(tag), Some(round)) = (uint(0), uint(1)) {
                self.mark(&format_args!("publish (tag {tag}, round {round})"));
                self.publishes.push((tag, round));
            }
        }
        Ok(())
    }
    fn clock(&mut self, _now: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn timer_arm(&mut self, _id: u64, _delay: u64, _armed_at: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn timer_cancel(&mut self, _id: u64, _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn read_back(
        &mut self,
        _src: u64,
        _kind: u64,
        _status: u64,
        _value: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn device_profile(&mut self, _profile: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn drop_coalesced(&mut self, _c: u64, _r: u64, _d: Dropped) -> Result<(), SinkError> {
        Ok(())
    }
    fn condition(&mut self, _code: &str, _detail: &str) -> Result<(), SinkError> {
        Ok(())
    }
    fn completion(&mut self, _op: u64, _result: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn snapshot(&mut self, _manifest: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.mark(&format_args!(
            "terminal (kind {kind}, outcome {outcome:?}, trap {trap:?})"
        ));
        self.terminal = Some(format!("kind {kind}, outcome {outcome:?}, trap {trap:?}"));
        Ok(())
    }
}
