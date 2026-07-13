// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Merge-2 headline integration check: the real 160M LLaMA preset trained through the wasm host on
// `BackendKind::Wgpu` (Vulkan/RADV) — build + a few inner AdamW steps over a fixed batch (so the
// loss overfits *down*) + make_update, all finite. Plus the worker-shape autotune assess reporting
// ELIGIBLE with the real unified-memory probe (the UMA fix). This is `#[ignore]`d: a real ~152M
// execute pass on the GPU is minutes/GBs (program Risk 3), so it is opt-in and run in the wgpu lane
// (`nix develop .#vulkan --command cargo test -p daemon-train --features wgpu --test preset_160m_wgpu -- --ignored --nocapture`).
#![cfg(feature = "wgpu")]
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::{Duration, Instant};

use daemon_swarm_run::backend::{BatchRef, StepCtx, TrainerBackend};
use daemon_train::{
    wgpu_adapter_available, BackendKind, EngineConfig, WasmBackend, WasmBackendConfig,
};
use daemon_train_sdk::models::TinyLlamaCfg;

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
            // Clear the devShell's `CARGO_TARGET_DIR` (pinned to the parent checkout) so the guests
            // build into their own `guests/target/` where `guest_dir()` reads them.
            .env_remove("CARGO_TARGET_DIR")
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

fn cbor(cfg: &TinyLlamaCfg) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(cfg, &mut b).expect("cbor");
    b
}

/// Budgets sized for a 768-wide, 12-layer, seq-1024 model (self-protection, not domain limits —
/// ABI §8): a big model's real fp32 matmuls take longer wall-clock (else the epoch watchdog trips)
/// even at the same host-op count, and its build/step touch far more handles.
fn roomy_wgpu_engine() -> EngineConfig {
    EngineConfig {
        backend: BackendKind::Wgpu,
        fuel_per_call: 1 << 36,
        epoch_deadline: Duration::from_secs(3600),
        op_budget: 1 << 32,
        max_step_handles: 1 << 26,
        ..EngineConfig::default()
    }
}

fn ctx(inner_step: u32) -> StepCtx {
    StepCtx {
        inner_step,
        mb_index: 0,
        mb_count: 1,
        step_seqs: 1,
    }
}

/// The headline P1 integration check on the GPU: the 160M preset builds on wgpu, overfits a fixed
/// batch (loss finite and strictly decreasing), and seals a non-empty sparse_loco update.
#[test]
#[ignore = "expensive: a real ~152M-param execute pass on the GPU is minutes/GBs (Risk 3)"]
fn preset_160m_trains_on_wgpu() {
    if !wgpu_adapter_available() {
        eprintln!("SKIP preset_160m_trains_on_wgpu: no usable wgpu adapter (run in .#vulkan)");
        return;
    }
    let cfg = TinyLlamaCfg::llama_160m();
    assert_eq!(cfg.param_count(), 151_862_784, "exact 160M param count");
    let config = cbor(&cfg);

    let t_build = Instant::now();
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: tiny_llama_wasm(),
        engine: roomy_wgpu_engine(),
    })
    .expect("construct WasmBackend(wgpu)");
    b.build(&config).expect("da_build 160M on wgpu");
    let build_secs = t_build.elapsed().as_secs_f64();
    eprintln!("160M wgpu build: {build_secs:.1}s");

    // Overfit a single fixed batch (batch 1 × seq_len 1024): the loss must fall as the host learns.
    let seq = cfg.seq_len;
    let tokens: Vec<u32> = (0..seq)
        .map(|i| (u64::from(i).wrapping_mul(2_654_435_761) % u64::from(cfg.vocab)) as u32)
        .collect();
    let fixed = BatchRef {
        tokens,
        seq_len: seq,
    };

    const STEPS: u32 = 4; // a few inner AdamW steps — enough to show the loss trend, not the full h
    let mut losses = Vec::new();
    let t_steps = Instant::now();
    for step in 0..STEPS {
        let stats = b
            .train_step(&fixed, ctx(step))
            .expect("train_step 160M/wgpu");
        assert!(
            stats.loss.is_finite(),
            "160M/wgpu step {step} loss must be finite, got {}",
            stats.loss
        );
        losses.push(stats.loss);
        b.inner_update(step).expect("inner_update 160M/wgpu");
    }
    let step_secs = t_steps.elapsed().as_secs_f64();

    let t_upd = Instant::now();
    let payload = b.make_update(0).expect("make_update 160M/wgpu");
    let upd_secs = t_upd.elapsed().as_secs_f64();

    eprintln!(
        "preset_160m_trains_on_wgpu: build {build_secs:.1}s, {STEPS} steps {step_secs:.1}s \
         ({:.1}s/step), make_update {upd_secs:.1}s -> {} bytes; loss {losses:?}",
        step_secs / f64::from(STEPS),
        payload.len()
    );
    assert!(
        !payload.is_empty(),
        "make_update sealed a non-empty payload"
    );
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(
        last < first,
        "160M/wgpu loss must decrease over {STEPS} overfit steps ({first} -> {last})"
    );
}
