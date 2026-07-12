// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Host runtime ↔ real wasm guest integration (ABI §11): the da_abi gate + da_build round trip, T3
// re-instantiation (HOST-14), an op call, and typed budget/phase traps (HOST-10).
//
// The `.wasm` artifacts land in `guests/target/wasm32-unknown-unknown/release/<name>.wasm` (the
// guests mini-workspace's own target dir, gitignored). Tests locate them via `SWARM_TEST_GUEST_DIR`
// if set, else the conventional path relative to this crate; if absent they are BUILT ON DEMAND
// (exactly what `xtask build-guests` does), so `cargo test --workspace` never silently skips. The
// dev-shell `wasm32-unknown-unknown` rust-std is required (a bare host cargo cannot cross-compile).

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_train::{EngineConfig, TrapCode, Worker};
use serde::Serialize;

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        // Skip the build if the caller pre-staged artifacts via SWARM_TEST_GUEST_DIR.
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
}

fn wasm(name: &str) -> Vec<u8> {
    let path = guest_dir().join(format!("{}.wasm", name.replace('-', "_")));
    if !path.exists() {
        ensure_built();
    }
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn cbor<T: Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(v, &mut b).expect("cbor");
    b
}

#[derive(Serialize)]
struct TinyCfg {
    d_model: u32,
    vocab: u32,
}

#[derive(Serialize)]
struct ModeCfg {
    mode: u32,
}

fn tiny_cfg() -> Vec<u8> {
    cbor(&TinyCfg {
        d_model: 8,
        vocab: 16,
    })
}

#[test]
fn abi_gate_and_build_round_trip() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap(); // da_abi gate runs here

    let manifest = inst.manifest(&tiny_cfg()).unwrap();
    assert_eq!(manifest.name, "tiny-llama");
    assert_eq!(manifest.steps_per_round, 1);
    assert!(manifest.round_modes.iter().any(|m| m == "barrier"));

    inst.build(&tiny_cfg()).unwrap();
    let params = inst.params();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].name, "tok.weight");
    assert_eq!(params[0].shape, vec![16, 8]);
    assert_eq!(params[1].name, "norm.weight");
    assert_eq!(params[1].shape, vec![8]);
    // norm.weight was Ones-initialized ⇒ its master is all ones.
    assert!(inst
        .param_master("norm.weight")
        .unwrap()
        .iter()
        .all(|&x| x == 1.0));
}

/// HOST-14 / T3: drop the instance mid-"round", re-instantiate from the same `InstancePre`, re-run
/// `da_build`, and assert an identical handle layout, registration list, and round-base state.
#[test]
fn reinstantiate_rebuilds_identical_state() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();

    let mut i1 = worker.instantiate(&module).unwrap();
    i1.build(&tiny_cfg()).unwrap();
    let params1 = i1.params();
    let masters1: Vec<Vec<f32>> = params1
        .iter()
        .map(|p| i1.param_master(&p.name).unwrap())
        .collect();
    drop(i1);

    let mut i2 = worker.instantiate(&module).unwrap();
    i2.build(&tiny_cfg()).unwrap();
    let params2 = i2.params();

    // Identical registration list + stable handle layout (deterministic function of order, T3).
    assert_eq!(params1, params2);
    let masters2: Vec<Vec<f32>> = params2
        .iter()
        .map(|p| i2.param_master(&p.name).unwrap())
        .collect();
    // Deterministic init ⇒ identical round-base masters across re-instantiation.
    assert_eq!(masters1, masters2);
}

#[test]
fn op_call_in_step_reports_metric() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("test_abi_basic")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let cfg = cbor(&ModeCfg { mode: 0 });
    inst.build(&cfg).unwrap();

    let batch = inst.register_batch(vec![0; 4], 2, 2);
    inst.step(batch, 0, 0, 1, 2).unwrap();

    assert!(
        inst.metrics().iter().any(|(n, _)| n == "probe"),
        "the happy-path step should report the `probe` metric"
    );
}

/// HOST-12 shape: a det-lane op in `da_step` is illegal (ABI §3.5) ⇒ typed `PhaseViolation`.
#[test]
fn phase_violation_traps_typed() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("test_abi_basic")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&cbor(&ModeCfg { mode: 1 })).unwrap();

    let batch = inst.register_batch(vec![0; 4], 2, 2);
    let err = inst.step(batch, 0, 0, 1, 2).unwrap_err();
    assert_eq!(err.trap_code(), Some(TrapCode::PhaseViolation), "{err}");
}

/// HOST-10: fuel exhaustion in a pure-guest spin traps typed `BudgetFuel` (worker intact).
#[test]
fn budget_exhaustion_traps_typed() {
    let worker = Worker::new(EngineConfig {
        fuel_per_call: 1_000_000,
        ..EngineConfig::default()
    })
    .unwrap();
    let module = worker.load_module(&wasm("test_abi_basic")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&cbor(&ModeCfg { mode: 2 })).unwrap();

    let batch = inst.register_batch(vec![0; 4], 2, 2);
    let err = inst.step(batch, 0, 0, 1, 2).unwrap_err();
    assert_eq!(err.trap_code(), Some(TrapCode::BudgetFuel), "{err}");

    // The worker survives a trapping module: a fresh instance still works.
    let mut ok = worker.instantiate(&module).unwrap();
    ok.build(&cbor(&ModeCfg { mode: 0 })).unwrap();
    assert_eq!(ok.params().len(), 1);
}

/// The lifecycle loop shape end to end: build → make_update → stage (self) → ingest, and the
/// post-ingest master is re-derivable (the barrier snapshot advances the round base).
#[test]
fn full_round_shape_runs() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&tiny_cfg()).unwrap();

    let before = inst.param_master("tok.weight").unwrap();
    let container = inst.make_update(0).unwrap();
    inst.stage(container);
    inst.ingest(0, 1).unwrap();

    // Single self-inclusive peer, α = 1: the outer step returns the (unchanged) local params.
    let after = inst.param_master("tok.weight").unwrap();
    assert_eq!(before, after);
}
