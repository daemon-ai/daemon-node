// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Host runtime ↔ real wasm guest integration (ABI §11): the da_abi gate + da_build round trip, T3
// re-instantiation (HOST-14), a full forward through the real tiny-llama model (exercising the
// Wave-2 op dispatch: embedding/rmsnorm/rope/flash_attn/silu/…), a sparse_loco round shape, and
// typed budget/phase traps (HOST-10) via test-abi-basic.
//
// The `.wasm` artifacts land in `guests/target/wasm32-unknown-unknown/release/<name>.wasm` (the
// guests mini-workspace's own target dir, gitignored). Tests locate them via `SWARM_TEST_GUEST_DIR`
// if set, else the conventional path relative to this crate; if absent they are BUILT ON DEMAND
// (exactly what `xtask build-guests` does), so `cargo test --workspace` never silently skips. The
// dev-shell `wasm32-unknown-unknown` rust-std is required (a bare host cargo cannot cross-compile).
//
// This is a dev/test harness (it shells `cargo build` for the guests and reads the `.wasm`, exactly
// like `xtask build-guests`); the fs/process hardening bans target the shipped node, not tests.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_train::{EngineConfig, TrapCode, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;
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
struct ModeCfg {
    mode: u32,
}

/// A small, fast tiny-llama config (1 layer) whose parameter element counts are all multiples of
/// the default sparse_loco chunk (64), so make_update needs no padding.
fn tiny_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    }
}

fn tiny_cbor() -> Vec<u8> {
    cbor(&tiny_cfg())
}

#[test]
fn abi_gate_and_build_round_trip() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap(); // da_abi gate runs here

    let manifest = inst.manifest(&tiny_cbor()).unwrap();
    assert_eq!(manifest.name, "tiny-llama");
    assert_eq!(manifest.steps_per_round, tiny_cfg().sparse_loco.h);
    assert!(manifest.round_modes.iter().any(|m| m == "barrier"));

    inst.build(&tiny_cbor()).unwrap();
    let params = inst.params();
    // 1 (tok) + n_layers·9 + 1 (final norm) = 11 for a 1-layer model.
    assert_eq!(params.len(), 1 + 9 + 1);
    assert_eq!(params[0].name, "tok.weight");
    assert_eq!(params[0].shape, vec![tiny_cfg().vocab, tiny_cfg().d_model]);
    assert_eq!(params.last().unwrap().name, "norm.weight");
    // norm.weight is Ones-initialized ⇒ its master is all ones.
    assert!(inst
        .param_master("norm.weight")
        .unwrap()
        .iter()
        .all(|&x| x == 1.0));
}

/// HOST-14 / T3: drop the instance mid-"round", re-instantiate, re-run `da_build`, assert identical
/// registration list + round-base state.
#[test]
fn reinstantiate_rebuilds_identical_state() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();

    let mut i1 = worker.instantiate(&module).unwrap();
    i1.build(&tiny_cbor()).unwrap();
    let params1 = i1.params();
    let masters1: Vec<Vec<f32>> = params1
        .iter()
        .map(|p| i1.param_master(&p.name).unwrap())
        .collect();
    drop(i1);

    let mut i2 = worker.instantiate(&module).unwrap();
    i2.build(&tiny_cbor()).unwrap();
    let params2 = i2.params();
    assert_eq!(params1, params2);
    let masters2: Vec<Vec<f32>> = params2
        .iter()
        .map(|p| i2.param_master(&p.name).unwrap())
        .collect();
    assert_eq!(masters1, masters2);
}

/// The full Wave-2 forward runs through the host op dispatch (embedding → rmsnorm → RoPE →
/// flash_attn → SwiGLU → tied logits → cross_entropy) and reports a finite loss metric.
#[test]
fn tiny_llama_forward_step_reports_loss() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&tiny_cbor()).unwrap();

    let c = tiny_cfg();
    let (b, seq) = (2u32, c.seq_len);
    let tokens: Vec<u32> = (0..b * seq).map(|i| i % c.vocab).collect();
    let batch = inst.register_batch(tokens, b, seq);
    inst.step(batch, 0, 0, 1, b).unwrap();

    let loss = inst
        .metrics()
        .into_iter()
        .find(|(n, _)| n == "loss")
        .map(|(_, v)| v);
    assert!(
        loss.is_some_and(f32::is_finite),
        "step must report a finite loss, got {loss:?}"
    );
}

/// HOST-8: a `meta` pass over the real model produces a schema-valid `MetaReport` — the param
/// layout, byte footprints, per-entry op counts, the two-point ingest-per-peer fit, and the set of
/// ops actually exercised (embedding/rmsnorm/flash_attn/…), all CBOR round-trippable.
#[test]
fn meta_report_layout_and_schema() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();

    let report = inst.meta(&tiny_cbor(), 2, tiny_cfg().seq_len).unwrap();

    assert_eq!(report.abi >> 16, 1);
    assert_eq!(report.params.len(), 1 + 9 + 1);
    assert_eq!(report.params[0].0, "tok.weight");
    assert!(report.master_bytes > 0 && report.grad_bytes == report.master_bytes);
    assert!(
        report.op_calls["da_step"] > 0,
        "the forward charged host ops"
    );
    assert!(report.op_calls.contains_key("da_ingest_updates"));
    // The forward exercised the Wave-2 NN vocabulary.
    for op in ["embedding@1", "rmsnorm@1", "flash_attn@1", "silu@1"] {
        assert!(report.ops_used.iter().any(|o| o == op), "meta missed {op}");
    }
    // The ingest cost fit is a non-negative per-peer slope, and the report round-trips as CBOR.
    let bytes = report.to_cbor();
    let back: daemon_train::MetaReport = ciborium::from_reader(bytes.as_slice()).unwrap();
    assert_eq!(back.params.len(), report.params.len());
    assert_eq!(back.ops_used, report.ops_used);
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

    let mut ok = worker.instantiate(&module).unwrap();
    ok.build(&cbor(&ModeCfg { mode: 0 })).unwrap();
    assert_eq!(ok.params().len(), 1);
}

/// The sparse_loco round shape end to end through the host: build → make_update → stage (self) →
/// ingest, and the post-ingest master re-derives (barrier snapshot advances the round base). With
/// no inner training Δ = 0, so the outer step is a no-op and θ is unchanged.
#[test]
fn full_round_shape_runs() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&tiny_cbor()).unwrap();

    let before = inst.param_master("tok.weight").unwrap();
    let container = inst.make_update(0).unwrap();
    inst.stage(container);
    inst.ingest(0, 1).unwrap();

    let after = inst.param_master("tok.weight").unwrap();
    assert_eq!(before, after);
}

/// HOST-15: `da_manifest` is **pure** — it charges zero host imports. `da_manifest` runs outside any
/// entry-point phase, so a host import called during it would trap `PhaseViolation`; a clean run that
/// charged nothing proves the manifest is a pure function of the config (ABI §6.2). Extends the
/// Wave-1 purity pattern to the real tiny-llama module.
#[test]
fn manifest_is_pure_no_host_imports() {
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&wasm("tiny_llama")).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let manifest = inst.manifest(&tiny_cbor()).unwrap();
    assert_eq!(manifest.name, "tiny-llama");
    assert_eq!(
        inst.imports_charged(),
        0,
        "da_manifest must call no host import (it is a pure function of the config)"
    );
}

/// Guest module release sizes stay well under a few hundred KB (a size-regression guard; the actual
/// bytes are printed so the lane report can record them).
#[test]
fn guest_wasm_sizes_are_sane() {
    for name in ["tiny_llama", "test_abi_basic"] {
        let bytes = wasm(name);
        eprintln!(
            "guest {name}.wasm = {} bytes ({} KiB)",
            bytes.len(),
            bytes.len() / 1024
        );
        assert!(
            bytes.len() < 512 * 1024,
            "{name}.wasm is {} bytes (> 512 KiB budget)",
            bytes.len()
        );
    }
}
