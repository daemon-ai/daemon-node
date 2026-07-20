// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **The Phase-C compute acceptance** (track C1; refactor §7, ABI §15): the `test-compute-v2`
//! guest — an ordinary Burn model over the guest-side `HostBackend` — runs as a REAL wasm32
//! module against the real event-loop driver, and:
//!
//! - the pinned op wire round-trips end to end (guest `CBOR(OperationIr)` → `submit_op` →
//!   `ComputeRunner` → device) with the gradient extracted through the fence → `export` →
//!   `Completion(BufferHandle)` → `read_into` path, **bit-exact vs native `Autodiff<NdArray>`**
//!   (the wasm32 + driver counterpart of `compute_conformance.rs`'s in-process gate);
//! - the journal + replay treatment holds: the recorded run replays bit-exact (`Event::Fence`
//!   re-fed from tag-1 records; the export completion's buffer materialized from the kind-5
//!   record; no kernel re-executed);
//! - the typed fault surface holds under the driver: `InvalidHandle` at a bad export, the
//!   queue-depth `GrantViolation`, and the deferred device fault at fence (`ComputeFault` trap)
//!   and at export (`DeviceError` completion) via the injectable-fault seam.
//!
//! Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed
//! file-wide.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    replay, start_run, MemorySink, ReplayEnd, ReplayScript, RunConfig, RunEnd, RunIdentity,
    SinkEntry,
};
use daemon_vhc_host::{select_driver, EngineConfig, TrapCode, Worker};

fn compute_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("test_compute_v2")
}

/// The guest's deterministic inputs (kept in lockstep with `test-compute-v2/src/lib.rs`).
fn inputs() -> (Vec<f32>, [usize; 2], Vec<f32>, [usize; 2]) {
    let a = vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5]; // [2,3]
    let w = vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5]; // [3,2]
    (a, [2, 3], w, [3, 2])
}

/// The native oracle: the identical forward+backward on `Autodiff<NdArray>`.
fn native_grad() -> Vec<f32> {
    use burn::backend::Autodiff;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = Autodiff<NdArray<f32, i64, i8>>;
    let (a_data, a_shape, w_data, w_shape) = inputs();
    let dev = NdArrayDevice::Cpu;
    let a = Tensor::<B, 2>::from_data(TensorData::new(a_data, a_shape), &dev).require_grad();
    let w = Tensor::<B, 2>::from_data(TensorData::new(w_data, w_shape), &dev);
    let loss = burn::tensor::activation::relu(a.clone().matmul(w).add_scalar(0.75_f32)).sum();
    let grads = loss.backward();
    let ga = a.grad(&grads).expect("grad exists");
    ga.into_data().to_vec::<f32>().expect("f32 grads")
}

fn identity(instance: u64, module: [u8; 32]) -> RunIdentity {
    RunIdentity {
        run_id: [0xC1; 32],
        epoch: 0,
        role: "compute".to_string(),
        instance,
        module,
    }
}

struct Ran {
    end: RunEnd,
    entries: Vec<SinkEntry>,
    published: Vec<(u64, u64, Vec<u8>)>,
}

/// Run one scenario to its natural end: stop once `published` reaches `expect_publishes` (0 =
/// the guest is expected to trap on its own — no stop is sent).
fn run_scenario(
    wasm: &[u8],
    config: Vec<u8>,
    tune: impl FnOnce(&mut RunConfig),
    expect_publishes: usize,
) -> Ran {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let module_hash = *blake3::hash(wasm).as_bytes();
    let mut cfg = RunConfig::new(identity(1, module_hash), [0x5C; 32], config, Vec::new());
    tune(&mut cfg);
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, cfg, Box::new(sink.clone())).expect("start");
    if expect_publishes > 0 {
        let deadline = Instant::now() + Duration::from_secs(60);
        while run.pump.published().len() < expect_publishes {
            assert!(
                Instant::now() < deadline,
                "scenario stalled awaiting publishes; logs: {:?}",
                run.pump.logs()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        run.pump
            .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .expect("stop");
    }
    let pump = run.pump.clone();
    let end = run.wait().expect("guest thread");
    let published = pump.published();
    let entries = sink.lock().expect("sink").entries.clone();
    Ran {
        end,
        entries,
        published,
    }
}

fn frame_payload(frame: &[u8]) -> Vec<u8> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).expect("frame cbor");
    let ciborium::value::Value::Array(parts) = v else {
        panic!("frame shape");
    };
    let ciborium::value::Value::Bytes(payload) = &parts[1] else {
        panic!("payload");
    };
    payload.clone()
}

#[test]
fn selection_admits_the_compute_guest_at_the_phase_c_minor() {
    // The end-to-end witness of the C1-owned DA_ABI_MINOR_V2 bump: a real module importing
    // compute@2 selects major 2 at the Phase-C minor (its imports force declared minor 2).
    let wasm = compute_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("compute guest admitted");
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::COMPUTE_MINOR_V2)
    );
}

#[test]
fn wasm_guest_backward_is_bit_exact_and_replays() {
    // Scenario 0: forward+backward over the wasm32 HostBackend; gradient extracted via
    // fence → export → Completion(BufferHandle) → read_into; published verbatim.
    let wasm = compute_wasm();
    let ran = run_scenario(&wasm, vec![0, 0], |_| {}, 1);
    assert!(matches!(ran.end, RunEnd::Outcome(0)), "{:?}", ran.end);

    // Bit-exact vs the native oracle (both sides run identical ndarray kernels; only the
    // wasm32 + CBOR + driver path differs — the strongest possible pass).
    let exported = frame_payload(&ran.published[0].2);
    let data: burn::tensor::TensorData =
        ciborium::from_reader(exported.as_slice()).expect("TensorData decodes");
    let got = data.to_vec::<f32>().expect("f32 tensor");
    let want = native_grad();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits(), "grad must be bit-exact");
    }

    // The journal + replay treatment (§8.7): Fence re-fed from tag-1, the export completion's
    // buffer materialized from the kind-5 record, decisions bit-exact, no kernel re-executed.
    let script = ReplayScript::from_entries(&ran.entries);
    assert!(
        !script.tensor_exports.is_empty(),
        "the export journaled its kind-5 TensorData record"
    );
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let replayed = replay(&worker, &wasm, &[0, 0], &[], script).expect("replay harness");
    assert_eq!(replayed.end, ReplayEnd::Outcome(0));
    let recorded: Vec<[u8; 32]> = ran
        .entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Publish { payload_hash, .. } => Some(*payload_hash),
            _ => None,
        })
        .collect();
    let redriven: Vec<[u8; 32]> = replayed.decisions.iter().map(|d| d.payload_hash).collect();
    assert_eq!(recorded, redriven, "decisions replay bit-exact");
}

#[test]
fn invalid_handle_export_is_a_typed_trap() {
    // Scenario 1 (ABI §15): a never-registered TensorId at export is the typed InvalidHandle
    // trap — never a host crash.
    let wasm = compute_wasm();
    let ran = run_scenario(&wasm, vec![1, 0], |_| {}, 0);
    match ran.end {
        RunEnd::Trapped(t) => assert_eq!(t.code, TrapCode::InvalidHandle, "{t}"),
        other => panic!("expected the InvalidHandle trap, got {other:?}"),
    }
}

#[test]
fn queue_depth_grant_bounds_outstanding_device_work() {
    // Scenario 2 (architecture §3.3): depth+1 enqueues without a fence trap GrantViolation.
    let wasm = compute_wasm();
    let depth = 3u8;
    let ran = run_scenario(
        &wasm,
        vec![2, depth],
        |cfg| cfg.compute_queue_depth = u64::from(depth),
        0,
    );
    match ran.end {
        RunEnd::Trapped(t) => {
            assert_eq!(t.code, TrapCode::GrantViolation, "{t}");
            assert!(t.detail.contains("queue depth"), "{t}");
        }
        other => panic!("expected the queue-depth GrantViolation, got {other:?}"),
    }
}

#[test]
fn injected_device_fault_surfaces_at_the_fence_as_compute_fault() {
    // Scenario 3 (the deferred-error timing shape, §3.3): a device fault latched after the
    // first submitted op NEVER surfaces at submit_op; the next fence reports it as the typed
    // ComputeFault trap — so a delivered Event::Fence is a real consistency point.
    let wasm = compute_wasm();
    let ran = run_scenario(
        &wasm,
        vec![3, 0],
        |cfg| cfg.compute_fault_after_ops = Some(0),
        0,
    );
    match ran.end {
        RunEnd::Trapped(t) => assert_eq!(t.code, TrapCode::ComputeFault, "{t}"),
        other => panic!("expected the ComputeFault trap at the fence, got {other:?}"),
    }
}

#[test]
fn injected_device_fault_surfaces_at_export_as_device_error_completion() {
    // Scenario 4: the same latched fault at an EXPORT is the typed Err(DeviceError) completion
    // (the readback twin of the fence trap) — the guest observes it and reports.
    let wasm = compute_wasm();
    let ran = run_scenario(
        &wasm,
        vec![4, 0],
        |cfg| cfg.compute_fault_after_ops = Some(0),
        1,
    );
    assert!(matches!(ran.end, RunEnd::Outcome(0)), "{:?}", ran.end);
    assert_eq!(frame_payload(&ran.published[0].2), b"device-error");
}
