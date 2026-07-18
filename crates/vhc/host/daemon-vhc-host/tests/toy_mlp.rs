// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The Phase-C **model-agnostic** acceptance (refactor §7: "a non-LLaMA toy authored with zero host
// changes … proving the compute ABI is model-agnostic"): the `toy-mlp` guest — a two-layer MLP
// trained by SGD, authored purely over `daemon-vhc-sdk-compute` + `daemon-vhc-sdk` — runs as a
// REAL wasm32 module against the SAME `compute@2` runner + event-loop driver + journal the LLaMA
// reference uses. No host code was added for it: the host dispatches its `CBOR(OperationIr)` op
// stream by tensor shape, never by model. The trained `W1` it exports is **bit-exact** vs a native
// `Autodiff<NdArray>` run of the identical loop (both sides run ndarray; only the wasm32 + CBOR +
// driver path differs), and its recorded run replays bit-for-bit through the §8.7 engine.
//
// Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use daemon_vhc_host::run::{
    replay, start_run, MemorySink, ReplayEnd, ReplayScript, RunConfig, RunEnd, RunIdentity,
    SinkEntry,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};

// -- model constants (kept in lockstep with guests/toy-mlp/src/lib.rs) --------------------------

const IN: usize = 3;
const HID: usize = 4;
const OUT: usize = 2;
const BATCH: usize = 2;
const LR: f32 = 0.1;

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn toy_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/toy_mlp.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The deterministic inputs + initial parameters (identical to the guest's).
fn params() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.5];
    let y = vec![1.0, -0.5, 0.0, 2.0];
    let w1 = vec![
        0.1, -0.2, 0.3, 0.05, //
        0.4, -0.1, 0.2, -0.3, //
        -0.25, 0.15, 0.35, 0.1,
    ];
    let w2 = vec![
        0.2, -0.4, //
        0.1, 0.3, //
        -0.15, 0.25, //
        0.05, -0.35,
    ];
    (x, y, w1, w2)
}

/// The native oracle: the identical SGD loop on `Autodiff<NdArray>`, returning the trained `W1`.
fn native_trained_w1(steps: u8) -> Vec<f32> {
    use burn::backend::Autodiff;
    use burn::tensor::{Tensor, TensorData};
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = Autodiff<NdArray<f32, i64, i8>>;
    let dev = NdArrayDevice::Cpu;
    let (x_d, y_d, w1_d, w2_d) = params();
    let x = Tensor::<B, 2>::from_data(TensorData::new(x_d, [BATCH, IN]), &dev);
    let y = Tensor::<B, 2>::from_data(TensorData::new(y_d, [BATCH, OUT]), &dev);
    let mut w1 = Tensor::<B, 2>::from_data(TensorData::new(w1_d, [IN, HID]), &dev).require_grad();
    let mut w2 = Tensor::<B, 2>::from_data(TensorData::new(w2_d, [HID, OUT]), &dev).require_grad();
    for _ in 0..steps {
        let hidden = burn::tensor::activation::relu(x.clone().matmul(w1.clone()));
        let logits = hidden.matmul(w2.clone());
        let diff = logits.sub(y.clone());
        let loss = diff.clone().mul(diff).sum();
        let grads = loss.backward();
        let g1 = w1.grad(&grads).expect("g1");
        let g2 = w2.grad(&grads).expect("g2");
        w1 = Tensor::from_inner(w1.inner().sub(g1.mul_scalar(LR))).require_grad();
        w2 = Tensor::from_inner(w2.inner().sub(g2.mul_scalar(LR))).require_grad();
    }
    w1.inner().into_data().to_vec::<f32>().expect("f32 W1")
}

fn identity(module: [u8; 32]) -> RunIdentity {
    RunIdentity {
        run_id: [0x70; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module,
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

struct Ran {
    end: RunEnd,
    entries: Vec<SinkEntry>,
    published: Vec<(u64, u64, Vec<u8>)>,
}

fn run_toy(wasm: &[u8], steps: u8) -> Ran {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let module_hash = *blake3::hash(wasm).as_bytes();
    let cfg = RunConfig::new(identity(module_hash), [0x77; 32], vec![steps], Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, cfg, Box::new(sink.clone())).expect("start");
    let deadline = Instant::now() + Duration::from_secs(60);
    while run.pump.published().is_empty() {
        assert!(
            Instant::now() < deadline,
            "toy-mlp stalled awaiting its export publish; logs: {:?}",
            run.pump.logs()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    run.pump
        .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
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

#[test]
fn toy_mlp_selects_major_2_at_the_phase_c_minor() {
    // Same admission as the LLaMA compute reference — nothing about a distinct model differs at the
    // ABI: any compute@2 importer selects major 2 at the Phase-C minor.
    let wasm = toy_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("toy-mlp admitted");
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::COMPUTE_MINOR_V2),
        "a non-LLaMA compute@2 model is admitted by the same major-2 path"
    );
}

#[test]
fn toy_mlp_trains_bit_exact_and_replays_with_zero_host_changes() {
    let wasm = toy_wasm();
    let steps = 3u8;
    let ran = run_toy(&wasm, steps);
    assert!(matches!(ran.end, RunEnd::Outcome(0)), "{:?}", ran.end);

    // The trained W1, exported over the fence → export → completion → read path, is bit-exact vs a
    // native Autodiff<NdArray> run of the identical loop — the model-agnostic-lowering proof for a
    // model that is NOT the LLaMA reference (a plain MLP), authored with zero host edits.
    let exported = frame_payload(&ran.published[0].2);
    let data: burn::tensor::TensorData =
        ciborium::from_reader(exported.as_slice()).expect("TensorData decodes");
    let got = data.to_vec::<f32>().expect("f32 tensor");
    let want = native_trained_w1(steps);
    assert_eq!(got.len(), want.len(), "W1 element count");
    assert_eq!(got.len(), IN * HID);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "trained W1 must lower bit-exact vs native Autodiff<NdArray>"
        );
    }

    // Journal + §8.7 replay: the recorded run replays bit-for-bit (Fence re-fed from tag-1, the
    // export completion's buffer materialized from the kind-5 tensor record, no kernel re-run).
    let script = ReplayScript::from_entries(&ran.entries);
    assert!(
        !script.tensor_exports.is_empty(),
        "the export journaled its kind-5 TensorData record"
    );
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let replayed = replay(&worker, &wasm, &[steps], &[], script).expect("replay harness");
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
    assert_eq!(recorded, redriven, "decisions replay bit-for-bit");
    assert!(
        !recorded.is_empty(),
        "the toy decided things (published W1)"
    );
}
