// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The REAL-GEOMETRY **device-lane** gate: the production trainer guest brings its state plane up at
// the FROZEN fleet-ceremony geometry (`daemon_vhc_testkit::ceremony` — 786_507_264 parameters)
// against a REAL GPU device runner, then walks the first round's device ops (θ export ->
// `make_update`) — the seam a host-side device fault kills.
//
// Why this suite exists (the class it locks down). Every real-geometry lane in the battery is a CPU
// lane (`ceremony_geometry`, `ceremony_round`), and every DEVICE lane in the battery is a toy
// geometry (`trainer_goldens`' wgpu/cuda tiers, `compute_wgpu`). Neither combination exercises what
// the fleet actually does: allocate the ceremony model's fp32 working set ON a device and stream
// ~3 GiB of init windows into it. A device-lane fault that only appears at that scale therefore
// passed everything and stopped a fleet smoke at the trainer's FIRST device operation — a host-side
// device-task failure that poisoned the backend router's lock, after which the next router call
// unwrapped the poison and took the guest thread down with an unrelated secondary panic.
//
// The two properties it asserts, both of which that failure violated:
//
//  1. THE DEVICE LANE COMPLETES. The guest's init streams to completion (and its own seed-init
//     `expected_root` cross-check passes) with the model resident on the device, and the round's
//     first device walk reaches its commitment. This is the availability property.
//  2. NO HOST-LANE PANIC — a device failure is a TYPED fault. A host panic anywhere under the
//     compute seam is a defect regardless of the device's own health (ABI §7.6/§15: a device fault
//     surfaces through the deferred-error latch as `ComputeError::Device` -> `ComputeFault`, never
//     as a host crash). The run's end is inspected for exactly that: a guest-thread panic (which is
//     how a host-side compute panic reaches the caller) fails the gate with the panic text, while a
//     genuine device refusal must arrive as a typed trap/outcome.
//
// # How to invoke it (it is hardware-gated AND env-gated)
//
// The lane is compiled only with the device feature and RUN only when explicitly asked for, because
// it needs ~12 GiB of device working set and minutes of real GPU time — it is not part of any
// default gate:
//
// ```text
// # Apple Metal (the M4 fleet seat), over ssh, from the repo root on the box:
// ssh m1@<m4-host>
// cd ~/daemon-node-ceremony
// . ~/dev-env.sh                       # or: nix develop --command <cmd>
// DAEMON_VHC_CEREMONY_DEVICE_LANE=1 \
//   cargo test -p daemon-vhc-testkit --features wgpu --release \
//   --test ceremony_device_lane -- --nocapture --test-threads=1
// ```
//
// On macOS `--features wgpu` comes up on Metal (`cubecl_wgpu` reports `backend: Metal`); on Linux
// the same lane comes up on Vulkan, and the `cuda` feature selects the CUDA lane instead. Without
// `DAEMON_VHC_CEREMONY_DEVICE_LANE=1` the lane self-skips with a note; without a usable adapter it
// self-skips too (the established `trainer_goldens` / `compute_wgpu` hardware-gate convention).
// `--release` is not optional in practice: a debug-profile host allocates and streams the ceremony
// working set an order of magnitude slower than the epoch watchdog allows.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]
// The lane exists only where a device backend is compiled in.
#![cfg(any(feature = "wgpu", feature = "cuda"))]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    start_run, Dropped, JournalSink, RunConfig, RunEnd, RunIdentity, SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::det_state::family_byte_len;
use daemon_vhc_proto::{to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{BatchWindow, RoundOpen, VhcMessage};
use daemon_vhc_testkit::ceremony::{
    ceremony_param_numels, ceremony_state_chunk_size, ceremony_trainer_config_round_walk,
    CEREMONY_PARAM_COUNT,
};

/// The env switch that opts a box into this lane (see the module docs).
const LANE_ENV: &str = "DAEMON_VHC_CEREMONY_DEVICE_LANE";

/// The sole trainer identity the round-walk config pins as `peer`/`roster[0]`.
const PEER: [u8; 32] = [0x3b; 32];

/// The wall the device init is given. The DX12 fleet seat streams this init in 62.6 s and the CPU
/// lane in ~100 s; a device lane that has not finished in ten minutes is wedged, not slow.
const INIT_DEADLINE: Duration = Duration::from_secs(600);

/// The wall the first round's device walk (θ export -> `make_update`) is given on top of init.
const ROUND_DEADLINE: Duration = Duration::from_secs(900);

/// The device lane under test, and whether this build/host can run it.
struct Lane {
    backend: BackendKind,
    /// The device PLACEMENT the product passes (`EngineConfig::gpu_index`). The worker's measured
    /// selection always materializes one for a device lane
    /// (`gpu_index = backend.is_device().then_some(sel.gpu_index)`), so a gate that left it `None`
    /// would drive a device-selection path no fleet peer ever takes — and would miss a placement
    /// that cannot be brought up on this box's adapter class.
    placement: Option<u32>,
    name: &'static str,
}

/// Select the compiled device lane, preferring an explicitly-available adapter. Returns `None` when
/// no device lane can run here (the self-skip).
///
/// The placement mirrors the worker's: the probed device index (0 on every single-accelerator fleet
/// box), threaded through exactly as `engine_for_join` threads it.
fn device_lane() -> Option<Lane> {
    #[cfg(feature = "cuda")]
    if daemon_vhc_host::cuda_adapter_available() && daemon_vhc_host::probe::cuda_nvrtc_ready() {
        return Some(Lane {
            backend: BackendKind::Cuda,
            placement: Some(0),
            name: "cuda",
        });
    }
    #[cfg(feature = "wgpu")]
    if daemon_vhc_host::wgpu_adapter_available() {
        return Some(Lane {
            backend: BackendKind::Wgpu,
            placement: Some(0),
            name: "wgpu",
        });
    }
    None
}

/// Whether the operator asked for this lane.
fn lane_requested() -> bool {
    std::env::var(LANE_ENV).is_ok_and(|v| v == "1")
}

/// A journal sink that keeps only what the gate reads: the guest's publishes (the round's
/// commitment voice) and the terminal record. The device lane reads ~3 GiB of θ back through the
/// journal seam; retaining it would double the host footprint beside the device working set.
#[derive(Default)]
struct TerseSink {
    publishes: Vec<u64>,
    conditions: Vec<(String, String)>,
    terminal: Option<String>,
    next_seq: u64,
}

impl TerseSink {
    fn shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }
}

impl JournalSink for TerseSink {
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
        if let Ok(ciborium::value::Value::Array(items)) =
            ciborium::de::from_reader::<ciborium::value::Value, _>(payload)
        {
            if let Some(tag) = items
                .first()
                .and_then(ciborium::value::Value::as_integer)
                .and_then(|n| u64::try_from(i128::from(n)).ok())
            {
                self.publishes.push(tag);
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
    fn condition(&mut self, code: &str, detail: &str) -> Result<(), SinkError> {
        self.conditions.push((code.to_string(), detail.to_string()));
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

/// The REAL-GEOMETRY device-lane gate (see the module docs for how to invoke it).
#[test]
fn ceremony_geometry_trainer_device_lane_streams_init_and_the_first_round_walk() {
    if !lane_requested() {
        eprintln!(
            "SKIP ceremony device lane: set {LANE_ENV}=1 to run it (it needs a real GPU and \
             ~12 GiB of device working set)"
        );
        return;
    }
    let Some(lane) = device_lane() else {
        eprintln!("SKIP ceremony device lane: no usable device adapter on this host");
        return;
    };
    eprintln!(
        "ceremony_device_lane: driving the `{}` lane at placement {:?}",
        lane.name, lane.placement
    );

    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let numels = ceremony_param_numels();
    let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
    let family_bytes = family_byte_len(&numels_u64);
    assert_eq!(
        numels_u64.iter().sum::<u64>(),
        CEREMONY_PARAM_COUNT,
        "the device lane drives the frozen ceremony geometry"
    );

    // The production join-lane profile on the DEVICE backend — the same budgets
    // `daemon-vhc-worker`'s `engine_for_join` builds, including the unraised linear-memory cap.
    let engine = EngineConfig::real_model(lane.backend, lane.placement);
    let worker = Worker::new(engine).expect("engine");

    let roster = [PeerId(PEER)];
    // The round-walk config: the frozen model + profile + state contract, `steps_per_round = 0`
    // (the training math is the CPU goldens' job; this lane owns the DEVICE ops at real geometry).
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_round_walk(&roster))
        .expect("ceremony trainer config (round-walk form)");
    let identity = RunIdentity {
        run_id: [0xce; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = ceremony_state_chunk_size();
    run_cfg.compute_queue_depth = 1 << 20;
    run_cfg.max_readback_bytes_per_slice = 64 << 20;

    let sink = TerseSink::shared();
    let t0 = Instant::now();
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    // -- stage 1: the streamed seed init, on the device ------------------------------------------
    // Init completes when the guest first reaches `next_event`; the round-0 open queued here is the
    // first thing it observes, so its acceptance IS the init-complete signal and the round walk
    // starts in the same drive.
    let open = to_canonical_vec(&VhcMessage::RoundOpen(RoundOpen {
        round: 0,
        seed: Seed([0; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow { start: 0, end: 0 },
        deadline_unix_s: 0,
    }))
    .expect("round open");
    assert_eq!(
        pump.deliver_frame(0, 0, [9u8; 32], open.clone(), open)
            .expect("deliver round open"),
        daemon_vhc_host::run::DeliverVerdict::Accepted
    );

    let init_done = wait_for(&run, &pump, &sink, INIT_DEADLINE, |pump| {
        pump.state_store_stats().retained_bytes >= family_bytes
    });
    let init_wall = t0.elapsed();
    assert!(
        init_done,
        "the device-lane seed init must stream the whole master family to the state store within \
         {INIT_DEADLINE:?}; retained {} B of {family_bytes} B",
        pump.state_store_stats().retained_bytes
    );
    eprintln!(
        "ceremony_device_lane[{}]: init streamed {} B in {:.1} s",
        lane.name,
        pump.state_store_stats().retained_bytes,
        init_wall.as_secs_f64()
    );

    // -- stage 2: the first round's device ops (θ export -> `make_update` -> the committed PUT) ---
    // `payload_put` is the only op this walk asks the embedder for; answering it inline keeps the
    // gate to the DEVICE seam under test with no content plane.
    let round_done = wait_for(&run, &pump, &sink, ROUND_DEADLINE, |pump| {
        for (op, request) in pump.take_op_requests() {
            match request {
                daemon_vhc_host::run::OpRequest::PayloadPut { .. } => {
                    pump.complete_op(op, daemon_vhc_host::run::OpOutcome::PutDone)
                        .expect("put done");
                }
                other => panic!("unexpected op request on the device lane: {other:?}"),
            }
        }
        !sink.lock().expect("sink").publishes.is_empty()
    });
    assert!(
        round_done,
        "the first round's device walk (θ export -> make_update) must reach its commitment voice \
         within {ROUND_DEADLINE:?}"
    );
    eprintln!(
        "ceremony_device_lane[{}]: round-0 device walk voiced {:?} at {:.1} s",
        lane.name,
        sink.lock().expect("sink").publishes,
        t0.elapsed().as_secs_f64()
    );

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    let end = run.wait();

    // -- the no-host-panic assertion -------------------------------------------------------------
    // A host-side compute panic reaches the caller as a guest-thread panic, which is exactly how
    // the fleet's Metal-lane failure presented (a device task failed host-side, poisoned the
    // backend router's lock, and the NEXT router call unwrapped the poison). A device fault must
    // instead arrive as a typed compute fault (`ComputeError::Device` -> `ComputeFault` ->
    // `COMP_ERR_DEVICE`), so the run's own record says what the device did.
    match end {
        Ok(RunEnd::Outcome(0)) => {}
        Ok(other) => panic!(
            "the ceremony-geometry device lane must complete: got {other:?}. Conditions: {:?}, \
             terminal: {:?}",
            sink.lock().expect("sink").conditions,
            sink.lock().expect("sink").terminal
        ),
        Err(e) => panic!(
            "the device lane ended with a HOST-lane failure, not a typed device fault: {e}. A \
             panic under the compute seam is a defect even when the device itself is at fault — \
             ABI §7.6/§15 requires a device execution error to surface through the deferred-error \
             latch as a typed `ComputeFault` carrying the device's own message. Conditions: {:?}",
            sink.lock().expect("sink").conditions
        ),
    }
}

/// Pump until `cond` holds or `timeout` elapses, failing FAST (returning `false` immediately) when
/// the run has already ended — at this geometry the deadlines are minutes, and waiting one out
/// hides the terminal record that says what actually happened.
fn wait_for(
    run: &daemon_vhc_host::run::Run,
    pump: &daemon_vhc_host::run::PumpHandle,
    sink: &Arc<Mutex<TerseSink>>,
    timeout: Duration,
    mut cond: impl FnMut(&daemon_vhc_host::run::PumpHandle) -> bool,
) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if cond(pump) {
            return true;
        }
        if run.is_finished() {
            // Let the caller's assertion report the terminal record; a finished run will never
            // satisfy `cond`.
            eprintln!(
                "ceremony_device_lane: the run ended early — terminal {:?}, conditions {:?}",
                sink.lock().expect("sink").terminal,
                sink.lock().expect("sink").conditions
            );
            return false;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
