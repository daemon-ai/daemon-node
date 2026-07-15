// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// M2 — the **P1 numeric exit-gate** on the wgpu (Vulkan/RADV) lane at the real 160M preset (spec
// §17 / TDD §8 "P1"): 160M pretrains through the tabi (module) path with loss curves matching a
// straight-burn reference, and tokens/s is measured + reported. All tests here are `#[ignore]`d
// (a real ~152M fp32 execute pass on the GPU is minutes/GBs — program Risk 3) and use the G2
// `require_gpu!` skip convention, so the default gate stays green GPU-less and the full gate runs in
// `nix develop .#vulkan --command cargo test -p daemon-train --features wgpu --test reference_parity_wgpu -- --ignored --nocapture`.
#![cfg(feature = "wgpu")]
#![allow(clippy::disallowed_methods)]

mod reference;
mod tolerance;

use daemon_train::{wgpu_adapter_available, BackendKind, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;

use reference::{
    assert_parity, cfg_cbor, drive_reference, drive_tabi, engine_for, throughput_stats,
    tiny_llama_wasm, TokenBatch,
};
use tolerance::OpClass;

type Wgpu = burn::backend::Autodiff<burn::backend::Wgpu>;

/// The GPU-skip convention (G2): bail loudly when no usable wgpu adapter exists.
macro_rules! require_gpu {
    () => {
        if !wgpu_adapter_available() {
            eprintln!(
                "SKIP {}: no usable wgpu adapter (run in the .#vulkan devShell / on a GPU box)",
                module_path!()
            );
            return;
        }
    };
}

/// **The P1 numeric gate**: the full 160M preset, matched-init to the tabi path, over real
/// TinyStories tokens, run on both the tabi (module) path and the independent burn reference on
/// wgpu; per-step loss + final-weights parity within the Optimizer tolerance class (outer bound).
#[test]
#[ignore = "expensive: two ~152M fp32 execute passes on the GPU (P1 gate, Risk 3)"]
fn loss_parity_within_tolerance_160m() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    assert_eq!(cfg.param_count(), 151_862_784, "exact 160M param count");
    let steps: u32 = std::env::var("M2_WGPU_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let batch = TokenBatch::tinystories(1);

    let tabi = drive_tabi(&cfg, BackendKind::Wgpu, &batch, steps);
    let reference =
        drive_reference::<Wgpu>(&cfg, Default::default(), &tabi.init_state, &batch, steps);
    let report = assert_parity(&tabi, &reference, OpClass::Optimizer, "160m/wgpu");

    eprintln!("loss_parity_within_tolerance_160m ({steps} steps, wgpu, TinyStories, b=1):");
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
        "160M/wgpu tabi loss must decrease"
    );
}

/// tokens/s of the tabi path vs the reference on wgpu at 160M (warmup + measured, mean ± sd). P1 is a
/// numeric gate, not perf work: the tabi/reference overhead is **documented**, not asserted within
/// 25% (spec §15.1 claims <1% dispatch overhead — reported here honestly at 160M).
#[test]
#[ignore = "expensive: 160M throughput probe on the GPU"]
fn throughput_within_budget_or_documented() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    // Warmup drops leading steps (lazy GPU bringup + cubecl autotune kernel compile); measured is the
    // sample the mean±sd is taken over. Defaults keep the P1 gate identical (1 warmup + 4 measured);
    // `M2_WGPU_WARMUP` / `M2_WGPU_MEASURED` raise them for a low-variance evidence run (B3 ledger).
    let warmup: usize = std::env::var("M2_WGPU_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let measured: u32 = std::env::var("M2_WGPU_MEASURED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let steps = measured + warmup as u32;
    let batch = TokenBatch::tinystories(1);

    let tabi = drive_tabi(&cfg, BackendKind::Wgpu, &batch, steps);
    let reference =
        drive_reference::<Wgpu>(&cfg, Default::default(), &tabi.init_state, &batch, steps);

    let (tps_t, mean_t, sd_t) = throughput_stats(&tabi, batch.b, batch.seq, warmup);
    let (tps_r, mean_r, sd_r) = throughput_stats(&reference, batch.b, batch.seq, warmup);
    let overhead = mean_t / mean_r;
    eprintln!("throughput_within_budget_or_documented (160M/wgpu, b=1, {measured} measured steps after {warmup} warmup):");
    eprintln!("  tabi      {tps_t:8.1} tok/s   step {mean_t:.3}s ± {sd_t:.3}s");
    eprintln!("  reference {tps_r:8.1} tok/s   step {mean_r:.3}s ± {sd_r:.3}s");
    eprintln!(
        "  tabi/reference wall = {overhead:.2}×  (documented; P1 is a numeric gate, not perf)"
    );
    assert!(tps_t.is_finite() && tps_r.is_finite() && overhead.is_finite());
    assert!(tps_t > 0.0 && tps_r > 0.0);
}

/// The **loss-curve evidence run**: drive the 160M preset on wgpu through ≥2 full rounds
/// (H inner AdamW steps + `make_update` + self-`ingest` per round), recording the loss series. This
/// is the "160M trains through the swarm stack" evidence; the byte-identical cpu-vs-wgpu det-lane
/// digest invariant is covered by the G2 cross-backend digest tests (`wasm_backend_determinism.rs`).
#[test]
#[ignore = "expensive: multiple full 160M rounds on the GPU (~15-25 min, program budget)"]
fn loss_curve_160m_wgpu() {
    require_gpu!();
    let cfg = TinyLlamaCfg::llama_160m();
    let rounds: u32 = std::env::var("M2_WGPU_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let h: u32 = std::env::var("M2_WGPU_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cfg.sparse_loco.h);
    let batch = TokenBatch::tinystories(1);

    let worker = Worker::new(engine_for(&cfg, BackendKind::Wgpu)).expect("worker");
    let module = worker.load_module(&tiny_llama_wasm()).expect("module");
    let mut inst = worker.instantiate(&module).expect("instantiate");
    let t_build = std::time::Instant::now();
    inst.build(&cfg_cbor(&cfg)).expect("da_build 160M/wgpu");
    eprintln!(
        "loss_curve_160m_wgpu: build {:.1}s",
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
        "loss_curve_160m_wgpu: full series ({} inner steps): {series:?}",
        series.len()
    );
    let (first, last) = (series[0], *series.last().unwrap());
    assert!(
        last < first,
        "160M loss must fall over the run ({first} -> {last})"
    );
}
