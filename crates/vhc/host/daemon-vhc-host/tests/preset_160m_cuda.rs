// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// P3 Lane G headline integration check (the CUDA analogue of `preset_160m_wgpu.rs`): the real 160M
// LLaMA preset trained through the wasm host on `BackendKind::Cuda` (NVIDIA/NVRTC) — build + a few
// inner AdamW steps over a fixed batch (so the loss overfits *down*) + make_update, all finite. This
// is `#[ignore]`d: a real ~152M execute pass on the GPU is minutes/GBs (Risk 3), so it is opt-in and
// run in the CUDA lane on the RunPod 4090:
//   nix develop .#cuda-train --command cargo test -p daemon-train --features cuda \
//     --test preset_160m_cuda -- --ignored --nocapture
// (DAEMON_CUDA_RUNTIME_DIR=/root/cuda-rt-124 for NVRTC 12.4 — swarm-ledger-p3-g).
#![cfg(feature = "cuda")]
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use std::time::{Duration, Instant};

use daemon_swarm_run::backend::{BatchRef, StepCtx, TrainerBackend};
use daemon_train::{
    cuda_adapter_available, BackendKind, EngineConfig, WasmBackend, WasmBackendConfig,
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

/// RUSTFLAGS that make the guest `.wasm` byte-reproducible across checkouts/machines by remapping the
/// absolute prefixes rustc embeds in panic locations. MUST match `xtask build-guests`.
fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.parent().unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

/// Stale-guest guard (Merge-1 adjudication): missing/unreadable fails loud; a hash mismatch only
/// WARNS (path-keyed codegen ordering across worktrees, not a stale artifact). Callers rebuild first.
fn verify_guest_manifest(dir: &Path) {
    let manifest = guests_root().join("guests.blake3");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
        panic!(
            "read guest manifest {}: {e} — run `cargo run -p xtask -- build-guests`",
            manifest.display()
        )
    });
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (hex, name) = line
            .split_once("  ")
            .expect("guests.blake3 line must be `<blake3-hex>  <name>.wasm`");
        let bytes = std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("read guest module {}/{name}: {e}", dir.display()));
        let got = blake3::hash(&bytes).to_hex();
        if got.as_str() != hex {
            eprintln!(
                "warning: guest `{name}` in {} hashes {got} but committed guests.blake3 records \
                 {hex}. Expected across worktrees/machines (path-keyed codegen ordering); the \
                 freshly-built module is used.",
                dir.display()
            );
        }
    }
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            verify_guest_manifest(&guest_dir());
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
        verify_guest_manifest(&guest_dir());
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
fn roomy_cuda_engine() -> EngineConfig {
    EngineConfig {
        backend: BackendKind::Cuda,
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

/// The headline integration check on the GPU: the 160M preset builds on CUDA, overfits a fixed batch
/// (loss finite and strictly decreasing), and seals a non-empty sparse_loco update.
#[test]
#[ignore = "expensive: a real ~152M-param execute pass on the GPU is minutes/GBs (Risk 3)"]
fn preset_160m_trains_on_cuda() {
    if !cuda_adapter_available() {
        eprintln!("SKIP preset_160m_trains_on_cuda: no usable CUDA device (run in .#cuda-train on the 4090)");
        return;
    }
    let cfg = TinyLlamaCfg::llama_160m();
    assert_eq!(cfg.param_count(), 151_862_784, "exact 160M param count");
    let config = cbor(&cfg);

    let t_build = Instant::now();
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: tiny_llama_wasm(),
        engine: roomy_cuda_engine(),
    })
    .expect("construct WasmBackend(cuda)");
    b.build(&config).expect("da_build 160M on cuda");
    let build_secs = t_build.elapsed().as_secs_f64();
    eprintln!("160M cuda build: {build_secs:.1}s");

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
            .expect("train_step 160M/cuda");
        assert!(
            stats.loss.is_finite(),
            "160M/cuda step {step} loss must be finite, got {}",
            stats.loss
        );
        losses.push(stats.loss);
        b.inner_update(step).expect("inner_update 160M/cuda");
    }
    let step_secs = t_steps.elapsed().as_secs_f64();

    let t_upd = Instant::now();
    let payload = b.make_update(0).expect("make_update 160M/cuda");
    let upd_secs = t_upd.elapsed().as_secs_f64();

    eprintln!(
        "preset_160m_trains_on_cuda: build {build_secs:.1}s, {STEPS} steps {step_secs:.1}s \
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
        "160M/cuda loss must decrease over {STEPS} overfit steps ({first} -> {last})"
    );
}
