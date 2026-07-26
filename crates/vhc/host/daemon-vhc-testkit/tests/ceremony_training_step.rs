// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The REAL-GEOMETRY guest ROUND gate WITH TRAINING MATH: the production trainer guest runs a round
// whose inner loop actually trains — forward, backward, `inner_update` (AdamW over 786_507_264
// parameters), the round-final fence, the streamed θ export and the committed container — at the
// FROZEN fleet-ceremony geometry and the FROZEN compression profile (`topk = 64`), under the
// production sandbox budgets, on the CPU lane.
//
// Why this suite exists (the gap it closes). Its siblings split the fleet's round in half and each
// left the other half ungated at scale:
//
//   * `ceremony_geometry` runs the real geometry, INIT ONLY.
//   * `ceremony_round` runs the real geometry's whole ROUND WALK, but with `steps_per_round = 0` —
//     the barrier opens, commits, fences and exports at full size, and the optimizer never runs.
//   * `trainer_goldens` runs the training math bit-exactly, at a TOY geometry (64-dim, 2 blocks).
//
// So nothing in the battery ever observed what a fleet peer does between "init finished" and
// "round 0 committed": a real optimizer step over the real parameter set. That is a live seam — a
// fleet trainer reached exactly it and died there — and this lane is the one that walks it.
//
// The bound, and why it still covers the seam. `seq_len` is shortened (see
// `ceremony_trainer_config_training_step`, which owns the argument): a step's arithmetic is
// O(parameters × tokens), so the frozen 2048-token sequence is ~9.7 TFLOP per step on a CPU lane —
// hours, not a gate. Everything the seam is ABOUT is O(parameters) and unaffected: the parameter
// layout, the gradient buffers, the AdamW moments, the per-parameter device traffic, the profile's
// `topk = 64` selection over 512_049 compression rows, and the export/commit walks all run at full
// ceremony size at any sequence length. Only the activations shrink. Two steps, not one, so the
// accumulation boundary (`inner_update`) is a real boundary.
//
// The three assertions:
//
//  1. THE ROUND COMMITS. Reaching the tag-3 commitment means init streamed, both training steps
//     ran over the full parameter set, the fence passed, θ exported window-by-window and the
//     committed container was built and PUT — the whole fleet round-0 path bar the data plane.
//  2. THE MEASURED PEAK still fits the admitted claim. Training allocates gradients and steps the
//     moments; if any of that became guest-resident at this geometry the run would trap, and if it
//     merely grew, this measurement says so.
//  3. THE READBACKS STAY WINDOWED. Every device export crossed as one `state_chunk_size` window,
//     with the training math running — a gradient or θ readback that scales with a parameter is
//     the residency class the round-path gates exist to catch.
//
// And the RED LINE that keeps assertion 1 from being vacuous: the SAME round is driven TWICE, once
// with the inner loop OFF (`steps_per_round = 0` — the `ceremony_round` shape), and the two
// committed containers must DIFFER. Every other property of this lane is identical either way — the
// same walks run, the same windows are read back, the same memory profile is measured, and the
// untrained round reaches the very same tag-3 commitment — so without that comparison a "training
// gate" stays green while the optimizer at this geometry never executes. The container is this
// peer's compressed progress, so it is the one artifact that can only differ if θ moved.
//
// Cost: two rounds, minutes (each dominated by the ~50 s real-geometry init), single, CPU.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::{
    admit, start_run, DeviceProfile, Dropped, JournalSink, MemoryClaim, OwnerPolicy,
    ParticipationLane, RunConfig, RunEnd, RunIdentity, SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::{to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{BatchWindow, RoundOpen, VhcMessage};
use daemon_vhc_testkit::ceremony::{
    ceremony_param_numels, ceremony_state_chunk_size, ceremony_trainer_config_training_step,
    CEREMONY_PARAM_COUNT, CEREMONY_VOCAB,
};

/// The sole trainer identity the harness form pins as `peer`/`roster[0]`.
const PEER: [u8; 32] = [0x3b; 32];

/// Training steps in the gated round. Two, so the AdamW accumulation boundary is a real boundary
/// (the fleet runs 30; the count is not what the geometry seam is about — see the module docs).
const STEPS_PER_ROUND: u64 = 2;

/// The shortened sequence (the ONE documented deviation from the frozen model). 64 tokens keeps
/// the activation arithmetic to ~0.3 TFLOP per step while every O(parameters) part of the step —
/// which is all of what this lane gates — runs at full ceremony size.
const GATE_SEQ_LEN: u32 = 64;

/// One micro-batch per step (the frozen ceremony value).
const MICRO_BATCH: u32 = 1;

/// The per-slice readback allowance the production TRAINER lane grants.
const TRAINER_LANE_READBACK_BYTES: u64 = 64 << 20;

/// The live-buffer grants the production TRAINER lane carries.
const TRAINER_LANE_BUFFER_BYTES: u64 = 1 << 30;
const TRAINER_LANE_BUFFER_HANDLES: u64 = 1024;

/// The wall the whole gated round is given: the ~100 s real-geometry init, two real optimizer steps
/// over 786 M parameters, and the streamed θ export + commit.
const ROUND_DEADLINE: Duration = Duration::from_secs(2700);

/// A journal sink that MEASURES instead of retaining (the `ceremony_round` discipline: this round
/// reads ~3 GiB of θ back through the journal seam).
#[derive(Default)]
struct MeasuringSink {
    max_export_readback: u64,
    export_readbacks: u64,
    publishes: Vec<(u64, u64, usize)>,
    next_seq: u64,
    terminal: Option<String>,
}

impl MeasuringSink {
    fn shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }
}

impl JournalSink for MeasuringSink {
    fn run_header(
        &mut self,
        _abi: u64,
        _worlds: &[(String, u64)],
        _bridge: bool,
        _manifest: &[u8],
        _config: &[u8],
        _grants: &[u8],
        _claim: &[u8],
        _channels: &[u8],
        _device: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn instantiation(&mut self, _counter: u64, _reason: u64, _at: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn init(&mut self, _c: [u8; 32], _g: [u8; 32], _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn execution_grant(&mut self, _hash: [u8; 32], _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn event(&mut self, _at: u64, _frame: &[u8]) -> Result<(), SinkError> {
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
        if let Ok(Value::Array(items)) = ciborium::de::from_reader::<Value, _>(payload) {
            let uint = |i: usize| -> Option<u64> {
                items
                    .get(i)
                    .and_then(Value::as_integer)
                    .and_then(|n| u64::try_from(i128::from(n)).ok())
            };
            if let (Some(tag), Some(round)) = (uint(0), uint(1)) {
                self.publishes.push((tag, round, payload.len()));
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
        kind: u64,
        _status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        if kind == u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT) {
            self.export_readbacks += 1;
            self.max_export_readback = self.max_export_readback.max(value.len() as u64);
        }
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
        self.terminal = Some(format!("kind {kind}, outcome {outcome:?}, trap {trap:?}"));
        Ok(())
    }
}

/// One host-staged batch, in the guest's documented absent-`live` wrapper shape
/// `[0, round, step, sequences, seq_len, tokens_le]`.
fn batch_wrapper(round: u64, step: u32) -> Vec<u8> {
    let n = u64::from(MICRO_BATCH) * u64::from(GATE_SEQ_LEN);
    let mut le = Vec::with_capacity(usize::try_from(n).expect("token count fits usize") * 4);
    for i in 0..n {
        let x = i + 1_000 * u64::from(step) + 100_000 * round + 1;
        let token = (x.wrapping_mul(2_654_435_761) % u64::from(CEREMONY_VOCAB)) as u32;
        le.extend_from_slice(&token.to_le_bytes());
    }
    to_canonical_vec(&Value::Array(vec![
        Value::from(0u8),
        Value::from(round),
        Value::from(step),
        Value::from(MICRO_BATCH),
        Value::from(GATE_SEQ_LEN),
        Value::Bytes(le),
    ]))
    .expect("batch wrapper")
}

/// Run the REAL admission funnel over this config and return the claim the module derives for it —
/// the number the production join engine enforces as the sandbox's memory cap (the `ceremony_round`
/// derivation, unchanged: the lane is the production trainer lane with its DEVICE floor relaxed for
/// the CPU backend, every ceiling production).
fn admitted_claim(wasm: &[u8], cfg_bytes: &[u8]) -> MemoryClaim {
    let mut lane = ParticipationLane::trainer_launch_defaults();
    lane.gpu = 1;
    lane.vram_bytes = 0;
    lane.ram_bytes = 0;
    lane.disk_bytes = 0;
    let device = DeviceProfile {
        gpu: false,
        vram_bytes: 0,
        ram_bytes: 64 << 30,
        disk_bytes: 512 << 30,
    };
    let owner = OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    let assessment =
        Worker::new(EngineConfig::real_model(BackendKind::Cpu, None)).expect("assessment engine");
    admit(
        &assessment,
        wasm,
        Some(blake3::hash(wasm).as_bytes()),
        cfg_bytes,
        &[],
        &lane,
        &device,
        &owner,
        None,
        None,
    )
    .expect("the ceremony trainer's own claim admits under the production trainer lane")
    .claim
}

/// What one gated round produced.
struct RoundOutcome {
    /// The committed container the guest PUT.
    container: Vec<u8>,
    /// Peak guest linear memory over the whole run (wasm memory never shrinks, so the sample IS
    /// the high-water).
    peak: u64,
    /// The claim the module derived for this exact config (the enforced linear-memory cap).
    claim_host: u64,
    /// The largest single `READBACK_KIND_TENSOR_EXPORT`, and how many there were.
    max_export_readback: u64,
    export_readbacks: u64,
    /// Wall to the commitment.
    wall: Duration,
}

/// Drive ONE ceremony-geometry round to its committed-container PUT with `steps` inner training
/// steps, and report what it produced.
fn drive_round(wasm: &[u8], steps: u64) -> RoundOutcome {
    let roster = [PeerId(PEER)];
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_training_step(
        &roster,
        steps,
        GATE_SEQ_LEN,
    ))
    .expect("ceremony trainer config (training-step form)");

    // The cap is the module's OWN admitted claim for this config, through the real funnel.
    let claim = admitted_claim(wasm, &cfg_bytes);
    let engine = EngineConfig::real_model(BackendKind::Cpu, None)
        .with_claimed_memory(claim.hard_accountable.host);
    let worker = Worker::new(engine).expect("engine");

    let identity = RunIdentity {
        run_id: [0xcd; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = ceremony_state_chunk_size();
    run_cfg.compute_queue_depth = 1 << 20;
    // The production trainer lane's grants, not the driver defaults.
    run_cfg.max_readback_bytes_per_slice = TRAINER_LANE_READBACK_BYTES;
    run_cfg.max_live_buffer_bytes = TRAINER_LANE_BUFFER_BYTES;
    run_cfg.max_live_buffer_handles = TRAINER_LANE_BUFFER_HANDLES;
    run_cfg.hard_accountable_host_bytes = claim.declared_peak.host;

    let sink = MeasuringSink::shared();
    let t0 = Instant::now();
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    // Stage this round's batches, then open it. Both are queued while the guest is still streaming
    // its init; it observes them at its first `next_event`, which is after init completes.
    for step in 0..steps {
        pump.stage_payload(
            batch_wrapper(0, u32::try_from(step).expect("step fits u32")),
            None,
        )
        .expect("stage batch");
    }
    let open = to_canonical_vec(&VhcMessage::RoundOpen(RoundOpen {
        round: 0,
        seed: Seed([0; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow {
            start: 0,
            end: u64::from(MICRO_BATCH) * steps,
        },
        deadline_unix_s: 0,
    }))
    .expect("round open");
    assert_eq!(
        pump.deliver_frame(0, 0, [9u8; 32], open.clone(), open)
            .expect("deliver round open"),
        daemon_vhc_host::run::DeliverVerdict::Accepted
    );

    let deadline = Instant::now() + ROUND_DEADLINE;
    let mut container: Option<Vec<u8>> = None;
    loop {
        // This round asks the embedder for exactly one thing: the committed container's durable
        // PUT (no corpus, no peer payloads — the data plane is not this lane's subject).
        for (op, request) in pump.take_op_requests() {
            match request {
                daemon_vhc_host::run::OpRequest::PayloadPut { bytes } => {
                    container = Some(bytes.to_vec());
                    pump.complete_op(op, daemon_vhc_host::run::OpOutcome::PutDone)
                        .map(|_| ())
                        .expect("put done");
                }
                other => panic!("unexpected op request from the round: {other:?}"),
            }
        }
        let committed = sink
            .lock()
            .expect("sink")
            .publishes
            .iter()
            .any(|(tag, round, _)| *tag == 3 && *round == 0);
        if committed {
            break;
        }
        let terminal = sink.lock().expect("sink").terminal.clone();
        if let Some(terminal) = terminal {
            let peak = pump.guest_memory_high_water();
            panic!(
                "the {steps}-step round-0 ended before its commitment: {terminal}. Peak guest \
                 linear memory {peak} B against the admitted claim {} B. A GuestPanic here is an \
                 allocation the claim does not cover once the optimizer runs at this parameter \
                 count; an OutOfFuel is the per-slice fuel budget under a REAL training slice — \
                 which was measured on init and on the training-free round walk, and is measured \
                 with the math ON only here",
                claim.hard_accountable.host
            );
        }
        assert!(
            Instant::now() < deadline,
            "the {steps}-step round-0 did not commit within {ROUND_DEADLINE:?}; published {:?}, \
             peak guest linear memory {} B",
            sink.lock().expect("sink").publishes,
            pump.guest_memory_high_water(),
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let wall = t0.elapsed();
    let peak = pump.guest_memory_high_water();

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("the {steps}-step round must end cleanly, got {other:?}"),
    }

    let measured = sink.lock().expect("sink");
    RoundOutcome {
        container: container.expect("the round PUTs its committed container"),
        peak,
        claim_host: claim.hard_accountable.host,
        max_export_readback: measured.max_export_readback,
        export_readbacks: measured.export_readbacks,
        wall,
    }
}

/// The gate. **Release-only, and therefore `#[ignore]`d for a bare `cargo test`** — the merge lane
/// runs it explicitly (`xtask vhc-ci-t2`, the entry that passes `--release -- --ignored`).
///
/// Why: unlike its training-free siblings, whose cost is guest-side (cranelift-JIT'd) det expansion
/// and memory traffic, this lane's optimizer steps are real fp32 matmuls executed by HOST
/// burn-ndarray. Under the unoptimized test profile that arithmetic runs an order of magnitude
/// slower — measured at over nine minutes and still going, against 229 s in release — which would
/// make the tier-2 lane's wall a function of the host profile rather than of the code under test.
#[test]
#[ignore = "release-only: the optimizer steps are host fp32 matmuls (229 s release, >9 min debug); \
            `xtask vhc-ci-t2` runs it with --release --ignored"]
fn ceremony_geometry_trainer_round_trains_at_the_frozen_profile() {
    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let numels = ceremony_param_numels();
    assert_eq!(
        numels.iter().map(|&n| n as u64).sum::<u64>(),
        CEREMONY_PARAM_COUNT,
        "the gate trains the frozen ceremony parameter set"
    );

    // THE RED LINE, first: the SAME round with the inner loop switched off. It is the
    // `ceremony_round` shape at this config, it reaches the same commitment, and it satisfies every
    // other assertion below — which is exactly why the trained run has to be compared against it.
    let untrained = drive_round(&wasm, 0);
    let trained = drive_round(&wasm, STEPS_PER_ROUND);

    // 1. The trained round committed (both `drive_round` calls fail loudly if they do not).
    // 2. It committed something DIFFERENT. The container is the compressed progress this peer
    //    voices, so a round whose optimizer never ran — or ran and was discarded — commits the
    //    untrained container byte-for-byte. Nothing else in the lane can tell those apart: the
    //    walks, the readbacks and the memory profile are identical either way, which is how a
    //    "round gate" can be green while the training math at this geometry has never executed.
    assert_ne!(
        trained.container,
        untrained.container,
        "the {STEPS_PER_ROUND}-step round committed the SAME {} B container as the 0-step round: \
         the inner loop did not move θ, so this lane is not gating the training math at all",
        trained.container.len()
    );
    assert_eq!(
        trained.container.len(),
        untrained.container.len(),
        "the committed container's length is the frozen geometry's, training or not"
    );

    // 3. The measured peak still fits the admitted claim WITH the optimizer running: training
    //    allocates gradients and steps both AdamW moments, and none of that may become
    //    guest-resident at this parameter count.
    assert!(
        trained.peak <= trained.claim_host,
        "measured peak guest linear memory {} B exceeds the admitted claim {} B with the training \
         math on — the module under-claims for a round that actually trains (fix \
         `decl_for_config`, do not raise the cap)",
        trained.peak,
        trained.claim_host
    );

    // 4. Every device export crossed as a WINDOW while training: a gradient or θ readback that
    //    scales with a PARAMETER (the tied embedding is 192 MiB here) is the residency class the
    //    round-path gates exist to catch, and the training math is the one code path that had
    //    never been observed against it at this size.
    let window = ceremony_state_chunk_size();
    assert!(
        trained.max_export_readback <= window + 4096,
        "device exports must stay WINDOWED while training: largest readback {} B over a {window} \
         B window (+framing) across {} exports",
        trained.max_export_readback,
        trained.export_readbacks,
    );

    eprintln!(
        "ceremony_training_step: {STEPS_PER_ROUND} real optimizer step(s) over {} parameters at \
         seq {GATE_SEQ_LEN} committed in {:.1} s (the 0-step red-line round: {:.1} s); {} windowed \
         device exports, largest readback {} B (window {window} B); MEASURED peak guest linear \
         memory {} B ({:.1} % of the admitted claim {} B, vs {} B untrained); committed container \
         {} B, differing from the untrained one",
        CEREMONY_PARAM_COUNT,
        trained.wall.as_secs_f64(),
        untrained.wall.as_secs_f64(),
        trained.export_readbacks,
        trained.max_export_readback,
        trained.peak,
        100.0 * trained.peak as f64 / trained.claim_host as f64,
        trained.claim_host,
        untrained.peak,
        trained.container.len(),
    );
}
