// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// P3 Lane G — the CUDA (NVIDIA/NVRTC) analogue of `reference_parity_wgpu.rs`: the 160M preset
// pretrains through the tabi (module) path on `BackendKind::Cuda` with loss curves matching a
// straight-burn `Autodiff<Cuda>` reference (the same independent `RefLlama` harness), and tokens/s is
// measured + reported (CUDA vs the P2 wgpu figures — swarm-p2-throughput.md). All tests are
// `#[ignore]`d (a real ~152M fp32 execute pass on the GPU is minutes/GBs — Risk 3) and use the CUDA
// GPU-skip convention, so the default gate stays green GPU-less and the full gate runs on the RunPod
// 4090 in `.#cuda-train`:
//   nix develop .#cuda-train --command cargo test -p daemon-train --features cuda \
//     --test reference_parity_cuda -- --ignored --nocapture --test-threads=1
// (with DAEMON_CUDA_RUNTIME_DIR=/root/cuda-rt-124 so NVRTC 12.4 is on LD_LIBRARY_PATH — see the C3
// ledger; without a device / runtime dir the tests skip loudly.)
#![cfg(feature = "cuda")]
#![allow(clippy::disallowed_methods)]

mod reference;
mod tolerance;

use daemon_train::{cuda_adapter_available, BackendKind, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;

use reference::{
    assert_parity, cfg_cbor, drive_reference, drive_tabi, engine_for, throughput_stats,
    tiny_llama_wasm, TokenBatch,
};
use tolerance::OpClass;

type Cuda = burn::backend::Autodiff<burn::backend::Cuda>;

/// The GPU-skip convention (Lane G): bail loudly when no usable CUDA device exists.
macro_rules! require_gpu {
    () => {
        if !cuda_adapter_available() {
            eprintln!(
                "SKIP {}: no usable CUDA device (run on a CUDA box in the .#cuda-train devShell \
                 with DAEMON_CUDA_RUNTIME_DIR set — swarm-ledger-p3-g)",
                module_path!()
            );
            return;
        }
    };
}

/// **The exit-criterion numeric gate**: the full 160M preset, matched-init to the tabi path, over
/// real TinyStories tokens, run on both the tabi (module) path and the independent burn reference on
/// CUDA; per-step loss + final-weights parity within the Optimizer tolerance class (outer bound).
#[test]
#[ignore = "expensive: two ~152M fp32 execute passes on the GPU (Lane G exit criterion, Risk 3)"]
fn loss_parity_within_tolerance_160m_cuda() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    assert_eq!(cfg.param_count(), 151_862_784, "exact 160M param count");
    let steps: u32 = std::env::var("M2_CUDA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let batch = TokenBatch::tinystories(1);

    let tabi = drive_tabi(&cfg, BackendKind::Cuda, &batch, steps);
    let reference =
        drive_reference::<Cuda>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    let report = assert_parity(&tabi, &reference, OpClass::Optimizer, "160m/cuda");

    eprintln!("loss_parity_within_tolerance_160m_cuda ({steps} steps, cuda, TinyStories, b=1):");
    for (i, ((lt, lr), d)) in tabi
        .losses
        .iter()
        .zip(reference.losses.iter())
        .zip(report.per_step_delta.iter())
        .enumerate()
    {
        eprintln!("  step {i}: tabi {lt:.6}  ref {lr:.6}  |Δ| {d:.3e}");
    }
    eprintln!(
        "  final-weight max Δ = {:.3e} (Optimizer class rtol 2e-4/atol 2e-5)",
        report.final_weight_max_delta
    );
    assert!(
        tabi.losses.last().unwrap() < tabi.losses.first().unwrap(),
        "160M/cuda tabi loss must decrease"
    );
}

/// tokens/s of the tabi path vs the reference on CUDA at 160M (warmup + measured, mean ± sd) — the
/// throughput record compared against the P2 wgpu RADV figures (swarm-p2-throughput.md). This is a
/// numeric gate, not perf work: the tabi/reference overhead is **documented**, not asserted.
#[test]
#[ignore = "expensive: 160M throughput probe on the GPU"]
fn throughput_160m_cuda_documented() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    // Warmup drops leading steps (lazy device bringup + NVRTC kernel JIT + cubecl autotune); measured
    // is the sample the mean±sd is taken over. Defaults: 3 warmup + 10 measured (the B3 low-variance
    // evidence shape); `M2_CUDA_WARMUP` / `M2_CUDA_MEASURED` override.
    let warmup: usize = std::env::var("M2_CUDA_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let measured: u32 = std::env::var("M2_CUDA_MEASURED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let steps = measured + warmup as u32;
    let batch = TokenBatch::tinystories(1);

    let tabi = drive_tabi(&cfg, BackendKind::Cuda, &batch, steps);
    let reference =
        drive_reference::<Cuda>(&cfg, Default::default(), &tabi.init_state, &batch, steps);

    let (tps_t, mean_t, sd_t) = throughput_stats(&tabi, batch.b, batch.seq, warmup);
    let (tps_r, mean_r, sd_r) = throughput_stats(&reference, batch.b, batch.seq, warmup);
    let overhead = mean_t / mean_r;
    eprintln!("throughput_160m_cuda_documented (160M/cuda, b=1, {measured} measured steps after {warmup} warmup):");
    eprintln!("  tabi      {tps_t:8.1} tok/s   step {mean_t:.3}s ± {sd_t:.3}s");
    eprintln!("  reference {tps_r:8.1} tok/s   step {mean_r:.3}s ± {sd_r:.3}s");
    eprintln!("  tabi/reference wall = {overhead:.2}×  (documented; the record vs P2 wgpu RADV 383.9 tok/s)");
    assert!(tps_t.is_finite() && tps_r.is_finite() && overhead.is_finite());
    assert!(tps_t > 0.0 && tps_r > 0.0);
}

/// The **loss-curve evidence run**: drive the 160M preset on CUDA through ≥2 full rounds (H inner
/// AdamW steps + `make_update` + self-`ingest` per round), recording the loss series — "160M trains
/// through the swarm stack on CUDA". The byte-identical cpu-vs-cuda det-lane digest invariant is
/// covered separately by `wasm_backend_determinism::cross_backend_cuda`.
#[test]
#[ignore = "expensive: multiple full 160M rounds on the GPU"]
fn loss_curve_160m_cuda() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    let rounds: u32 = std::env::var("M2_CUDA_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let h: u32 = std::env::var("M2_CUDA_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cfg.sparse_loco.h);
    let batch = TokenBatch::tinystories(1);

    let worker = Worker::new(engine_for(&cfg, BackendKind::Cuda)).expect("worker");
    let module = worker.load_module(&tiny_llama_wasm()).expect("module");
    let mut inst = worker.instantiate(&module).expect("instantiate");
    let t_build = std::time::Instant::now();
    inst.build(&cfg_cbor(&cfg)).expect("da_build 160M/cuda");
    eprintln!(
        "loss_curve_160m_cuda: build {:.1}s",
        t_build.elapsed().as_secs_f64()
    );

    let mut inner = 0u32;
    let mut series: Vec<f32> = Vec::new();
    for r in 0..rounds {
        let t_round = std::time::Instant::now();
        for _ in 0..h {
            let bh = inst.register_batch(batch.tokens.clone(), batch.b, batch.seq);
            inst.step(bh, inner, 0, 1, batch.b).expect("da_step");
            let loss = inst
                .metrics()
                .into_iter()
                .rev()
                .find(|(n, _)| n == "loss")
                .map_or(f32::NAN, |(_, v)| v);
            assert!(
                loss.is_finite(),
                "round {r} step {inner} loss must be finite"
            );
            series.push(loss);
            inst.inner_update(inner).expect("da_inner_update");
            inner += 1;
        }
        let t_upd = std::time::Instant::now();
        let container = inst.make_update(u64::from(r)).expect("da_make_update");
        let payload = inst.update_bytes(container).expect("update bytes");
        inst.stage(container);
        inst.ingest(u64::from(r), 1)
            .expect("da_ingest_updates (self)");
        eprintln!(
            "  round {r}: {h} steps in {:.1}s, make_update+ingest {:.1}s -> {} byte payload, \
             loss {:.4} -> {:.4}",
            t_round.elapsed().as_secs_f64() - t_upd.elapsed().as_secs_f64(),
            t_upd.elapsed().as_secs_f64(),
            payload.len(),
            series[(r * h) as usize],
            series[series.len() - 1],
        );
    }
    eprintln!(
        "loss_curve_160m_cuda: full series ({} inner steps): {series:?}",
        series.len()
    );
    let (first, last) = (series[0], *series.last().unwrap());
    assert!(
        last < first,
        "160M/cuda loss must fall over the run ({first} -> {last})"
    );
}
