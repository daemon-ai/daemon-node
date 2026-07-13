// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// M2 — reference-parity + throughput on the **burn-ndarray** (CPU) lane (P1 numeric gate, spec §17 /
// TDD §8 "P1"). An independent burn LLaMA reference (`tests/reference/mod.rs`) is matched-init to the
// tabi (module) path's own weights and fed identical token batches; per-step loss + final-weights
// parity is asserted within the Optimizer tolerance class (the outer bound), and the achieved deltas
// are recorded (swarm-ledger-m2.md / swarm-p1-throughput.md).
//
// - `reference_parity_reduced_ndarray` + `throughput_reduced_ndarray` are **always-on** (a reduced
//   2-layer config, fast) so per-PR CI carries the seam.
// - `loss_parity_within_tolerance_ndarray` (≥20 steps, medium config, real TinyStories tokens) and
//   `throughput_medium_ndarray` are `#[ignore]`d-expensive (run once in the session).
//
// Full 160M parity runs on the **wgpu** lane (`reference_parity_wgpu.rs`): a 160M fp32 execute pass
// on CPU ndarray is impractically slow (program Risk 3), which is exactly why 160M needs a GPU. The
// ndarray lane proves many-step loss-curve tracking on CPU where it is cheap and deterministic.
#![cfg(feature = "burn-ndarray")]
#![allow(clippy::disallowed_methods)]

mod reference;
mod tolerance;

use daemon_train::BackendKind;
use daemon_train_sdk::models::{AdamWCfg, TinyLlamaCfg};

use reference::{assert_parity, drive_reference, drive_tabi, TokenBatch};
use tolerance::OpClass;

type Ndarray = burn::backend::Autodiff<burn::backend::NdArray>;

/// The reduced always-on config: the tiny 2-layer default (d_model 64, seq 9, vocab 64) — a real
/// transformer, fast enough for per-PR CI.
fn reduced_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg::default()
}

/// A medium config for the ≥20-step ndarray evidence: big enough to be a meaningful transformer,
/// small enough to run 20 CPU steps quickly. `vocab = 50257` so real TinyStories (GPT-2) tokens are
/// in range; `chunk 64` divides every param (build's sparse_loco registration is happy).
fn medium_cfg() -> TinyLlamaCfg {
    use daemon_train_sdk::profiles::SparseLocoCfg;
    TinyLlamaCfg {
        d_model: 256,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 4,
        head_dim: 64,
        vocab: 50257,
        seq_len: 128,
        ffn_mult: 4,
        rope_theta: 10_000.0,
        rmsnorm_eps: 1.0e-5,
        inner: AdamWCfg::default(),
        profile: "sparse_loco".to_string(),
        sparse_loco: SparseLocoCfg {
            h: 30,
            chunk: 64,
            topk: 4,
            bits: 2,
            ef_decay: 0.95,
            outer_alpha: 1.0,
            clip: true,
        },
        diloco: Default::default(),
        demo: Default::default(),
    }
}

/// tokens/s = next-token positions per second = `b·(seq−1)·steps / secs`.
fn tokens_per_s(b: u32, seq: u32, steps: u32, secs: f64) -> f64 {
    f64::from(b) * f64::from(seq - 1) * f64::from(steps) / secs
}

#[test]
fn reference_parity_reduced_ndarray() {
    let cfg = reduced_cfg();
    let batch = TokenBatch::deterministic(2, cfg.seq_len, cfg.vocab);
    let steps = 8;
    let tabi = drive_tabi(&cfg, BackendKind::BurnNdarray, &batch, steps);
    let reference =
        drive_reference::<Ndarray>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    let report = assert_parity(&tabi, &reference, OpClass::Optimizer, "reduced/ndarray");
    eprintln!(
        "reference_parity_reduced_ndarray: tabi losses {:?}",
        tabi.losses
    );
    eprintln!("  per-step |Δloss| {:?}", report.per_step_delta);
    eprintln!(
        "  final-weight max Δ = {:.3e} (class {:?})",
        report.final_weight_max_delta, report.class
    );
    // The overfit loss must fall on both paths (a sanity check the step actually learns).
    assert!(
        tabi.losses.last().unwrap() < tabi.losses.first().unwrap(),
        "reduced tabi loss must decrease"
    );
}

#[test]
fn throughput_reduced_ndarray() {
    let cfg = reduced_cfg();
    let batch = TokenBatch::deterministic(2, cfg.seq_len, cfg.vocab);
    let steps = 8;
    let tabi = drive_tabi(&cfg, BackendKind::BurnNdarray, &batch, steps);
    let reference =
        drive_reference::<Ndarray>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    let tps_tabi = tokens_per_s(batch.b, batch.seq, steps, tabi.step_secs);
    let tps_ref = tokens_per_s(batch.b, batch.seq, steps, reference.step_secs);
    let overhead = tabi.step_secs / reference.step_secs;
    eprintln!(
        "throughput_reduced_ndarray: tabi {:.1} tok/s ({:.3}s), reference {:.1} tok/s ({:.3}s), \
         tabi/reference wall = {overhead:.2}×",
        tps_tabi, tabi.step_secs, tps_ref, reference.step_secs
    );
    assert!(tps_tabi.is_finite() && tps_ref.is_finite() && overhead.is_finite());
    assert!(tps_tabi > 0.0 && tps_ref > 0.0);
}

/// ≥20-step loss-curve parity on ndarray at the medium config over real TinyStories tokens.
#[test]
#[ignore = "expensive: 20 CPU steps of a medium transformer over real TinyStories tokens"]
fn loss_parity_within_tolerance_ndarray() {
    let cfg = medium_cfg();
    let steps: u32 = std::env::var("M2_NDARRAY_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    // Real TinyStories tokens, truncated to the model's seq_len (fixture window is 1024).
    let batch = TokenBatch::tinystories(2).truncate_seq(cfg.seq_len);
    let tabi = drive_tabi(&cfg, BackendKind::BurnNdarray, &batch, steps);
    let reference =
        drive_reference::<Ndarray>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    let report = assert_parity(&tabi, &reference, OpClass::Optimizer, "medium/ndarray");
    eprintln!("loss_parity_within_tolerance_ndarray ({steps} steps, medium cfg, TinyStories):");
    for (i, ((lt, lr), d)) in tabi
        .losses
        .iter()
        .zip(reference.losses.iter())
        .zip(report.per_step_delta.iter())
        .enumerate()
    {
        eprintln!("  step {i:2}: tabi {lt:.6}  ref {lr:.6}  |Δ| {d:.3e}");
    }
    eprintln!(
        "  final-weight max Δ = {:.3e} (Optimizer class rtol 2e-4/atol 2e-5)",
        report.final_weight_max_delta
    );
}

#[test]
#[ignore = "expensive: throughput at the medium config (many CPU matmuls)"]
fn throughput_medium_ndarray() {
    let cfg = medium_cfg();
    let steps = 8;
    let batch = TokenBatch::tinystories(2).truncate_seq(cfg.seq_len);
    let tabi = drive_tabi(&cfg, BackendKind::BurnNdarray, &batch, steps);
    let reference =
        drive_reference::<Ndarray>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    eprintln!(
        "throughput_medium_ndarray: tabi {:.1} tok/s ({:.3}s/{steps}), reference {:.1} tok/s \
         ({:.3}s/{steps}), tabi/reference wall = {:.2}×",
        tokens_per_s(batch.b, batch.seq, steps, tabi.step_secs),
        tabi.step_secs,
        tokens_per_s(batch.b, batch.seq, steps, reference.step_secs),
        reference.step_secs,
        tabi.step_secs / reference.step_secs
    );
}
