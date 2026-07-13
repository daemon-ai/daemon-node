// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// G2 — lifecycle-level wgpu tests: HOST-8 (`meta_mode_vram_ram_estimates` against real device
// numbers) and HOST-10 (`ingest_budget_scales_with_count` re-run on the wgpu backend).
//
// GPU-skip convention (TDD §8.1 tier-2): each test checks `wgpu_adapter_available()` and skips
// with a loud stderr note when absent. The `.#vulkan` devShell is the runnable lane.
//
// Guest loading mirrors guest_lifecycle.rs (always-rebuild stale-guest guard); this is a dev/test
// harness, so the fs/process bans (which target the shipped node) are allowed file-wide.
#![cfg(feature = "wgpu")]
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_train::autotune::{probe_wgpu, Autotune, DeviceLimits, DEFAULT_MAX_MICROBATCH};
use daemon_train::{wgpu_adapter_available, BackendKind, EngineConfig, TrapCode, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;
use serde::Serialize;

macro_rules! require_gpu {
    () => {
        if !wgpu_adapter_available() {
            eprintln!(
                "SKIP {}: no usable wgpu adapter on this runner (run in the .#vulkan devShell — \
                 TDD §8.1 tier-2)",
                module_path!()
            );
            return;
        }
    };
}

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

fn tiny_llama_wasm() -> Vec<u8> {
    ensure_built();
    let path = guest_dir().join("tiny_llama.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn cbor<T: Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(v, &mut b).expect("cbor");
    b
}

fn tiny_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    }
}

/// HOST-8 `meta_mode_vram_ram_estimates`: the meta-pass byte footprints feed the G2 autotune, whose
/// verdict against THIS machine's real device numbers (wgpu adapter probe + /proc/meminfo host RAM)
/// is eligible with internally consistent estimates. The estimates themselves are backend-
/// independent (shapes/dtypes), so a CPU meta pass compared against real GPU limits is the honest
/// §6.5 admission shape.
#[test]
fn meta_mode_vram_ram_estimates() {
    require_gpu!();
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let report = inst
        .meta(&cbor(&tiny_cfg()), 1, tiny_cfg().seq_len)
        .unwrap();

    // Byte-footprint consistency: fp32 masters == grads; params in storage dtype ≤ masters (all
    // tiny-llama params are F32 here, so equal); every estimate is non-zero for a real model.
    assert_eq!(report.master_bytes, report.grad_bytes);
    assert!(report.param_bytes > 0 && report.master_bytes > 0);
    assert!(report.act_bytes_est > 0 && report.host_ram_bytes_est > 0);
    assert!(report.payload_bytes_est > 0);

    // The autotune model derived from the report: fixed VRAM = params (storage dtype) + fp32
    // masters + fp32 grads + the registered persistents (tiny-llama registers Adam moments etc.).
    let autotune = Autotune::from_meta(&report);
    let core = report.param_bytes + report.master_bytes + report.grad_bytes;
    assert!(
        autotune.fixed_vram_bytes >= core,
        "fixed VRAM covers at least params+masters+grads ({} >= {core})",
        autotune.fixed_vram_bytes
    );
    assert!(
        !report.persistent.is_empty(),
        "tiny-llama registers optimizer persistents (they enter the fixed VRAM term)"
    );

    // Real device numbers: the wgpu adapter probe (max single allocation — the one wgpu-queryable
    // budget number) + real host RAM.
    let probe = probe_wgpu().expect("adapter probed (require_gpu passed)");
    assert!(probe.gpus >= 1);
    assert!(probe.max_alloc_mb > 0, "max_buffer_size is queryable");
    eprintln!(
        "wgpu probe: adapter={} backend={} max_alloc={} MiB",
        probe.adapter, probe.backend, probe.max_alloc_mb
    );

    let limits = DeviceLimits {
        vram_mb: probe.max_alloc_mb,
        ram_mb: 8192, // a conservative host-RAM floor; the worker probes /proc/meminfo
        max_alloc_mb: probe.max_alloc_mb,
    };
    let v = autotune.verdict(&limits, DEFAULT_MAX_MICROBATCH);
    eprintln!(
        "meta_mode_vram_ram_estimates: verdict={v:?} (fixed={} MiB, act/mb={} B, host_ram={} MiB)",
        autotune.fixed_vram_bytes >> 20,
        autotune.act_bytes_per_mb,
        autotune.host_ram_bytes >> 20
    );
    assert!(
        v.eligible,
        "the 1-layer tiny-llama must fit this machine's real device budget: {:?}",
        v.reasons
    );
    assert!(v.micro_batch >= 1);
    assert!(v.vram_mb_estimate >= 1 && v.ram_mb_estimate >= 1);
    assert_eq!(v.payload_bytes_estimate, report.payload_bytes_est);
}

/// HOST-10 `ingest_budget_scales_with_count` (re-run on wgpu): the per-peer ingest cost measured by
/// the meta two-point fit scales the execute op budget (ABI §8) — a budget sized for the staged
/// count passes on the wgpu backend, and the SAME staged count under a budget sized for 1 peer
/// traps `BudgetOps` (typed, worker intact).
#[test]
fn ingest_budget_scales_with_count() {
    require_gpu!();
    let config = cbor(&tiny_cfg());
    let wasm = tiny_llama_wasm();

    // Meta two-point fit (CPU meta pass; op counts are backend-independent — the guest's import
    // stream is bit-deterministic, ABI §7.1).
    let meta_worker = Worker::new(EngineConfig::default()).unwrap();
    let module = meta_worker.load_module(&wasm).unwrap();
    let mut meta_inst = meta_worker.instantiate(&module).unwrap();
    let report = meta_inst.meta(&config, 1, tiny_cfg().seq_len).unwrap();
    let base = report.op_calls["da_ingest_updates"]; // cost at count = 1
    let slope = report.ingest_op_calls_per_peer;
    let op_build = report.op_calls["da_build"];
    let op_make = report.op_calls["da_make_update"];
    assert!(slope > 0, "ingest cost must grow with the staged count");

    // Pick a count whose ingest cost strictly exceeds every other entry point's cost, so the tight
    // budget below can pass build/make_update yet trap ONLY on the under-scaled ingest.
    let n = usize::try_from((op_build.max(op_make) / slope) + 4).unwrap();
    let cost = |count: u64| base + (count - 1) * slope;

    let run = |op_budget: u64| -> Result<(), daemon_train::TrainError> {
        let worker = Worker::new(EngineConfig {
            backend: BackendKind::Wgpu,
            op_budget,
            ..EngineConfig::default()
        })
        .unwrap();
        let module = worker.load_module(&wasm).unwrap();
        let mut inst = worker.instantiate(&module).unwrap();
        inst.build(&config)?;
        let container = inst.make_update(0)?;
        let payload = inst.update_bytes(container)?;
        let staged: Vec<Vec<u8>> = vec![payload; n];
        inst.ingest_payloads(0, &staged)
    };

    // Budget scaled for count = n (the ABI §8 shape: base + count × per-peer) → passes on wgpu.
    let scaled = cost(n as u64) + slope; // headroom of one peer
    run(scaled).expect("a count-scaled ingest budget must pass");

    // Budget scaled for count = 1 (still enough for build/make_update by construction of n) → the
    // count-n ingest traps typed BudgetOps.
    let tight = cost(1).max(op_build).max(op_make) + 1;
    assert!(
        tight < cost(n as u64),
        "n was chosen so the tight budget under-scales ingest"
    );
    let err = run(tight).expect_err("an unscaled budget must trap on the count-n ingest");
    assert_eq!(
        err.trap_code(),
        Some(TrapCode::BudgetOps),
        "typed BudgetOps, worker intact: {err}"
    );
    eprintln!(
        "ingest_budget_scales_with_count (wgpu): base={base} slope={slope} n={n} \
         scaled_budget={scaled} tight_budget={tight}"
    );
}
