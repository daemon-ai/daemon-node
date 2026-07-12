// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Profile golden/property tests (TDD §3.4, SDK-1..5) driven through the `sim` backend: each profile
// round-trips make_update → stage (2 simulated peers) → ingest to a deterministic, bit-reproducible
// state, with per-profile sparsity/compression assertions. sim-only.
#![cfg(feature = "sim")]

use daemon_train_sdk::profiles::{Demo, DemoCfg, DiLoCo, DiLoCoCfg, SparseLoco, SparseLocoCfg};
use daemon_train_sdk::sim;
use daemon_train_sdk::{Dtype, Init, Param, Persistent, Tensor, UpdatesView};

/// A minimal one-weight model with AdamW inner state — enough to drive a real pseudo-gradient
/// (Δ = θ⁽ᵗ⁾ − θ) through a profile's compress → ingest loop.
struct Model {
    w: Vec<Param>,
    m: Vec<Persistent>,
    v: Vec<Persistent>,
    dims: Vec<u32>,
}

impl Model {
    fn build(dims: &[u32]) -> Self {
        let w = vec![Param::new("w", dims, Dtype::F32, Init::Normal, 0.0, 0.1)];
        let m = vec![Persistent::local("m0", dims, Dtype::F32)];
        let v = vec![Persistent::local("v0", dims, Dtype::F32)];
        Self {
            w,
            m,
            v,
            dims: dims.to_vec(),
        }
    }

    /// `h` inner AdamW steps minimizing `Σ (w − 0.5)²` — a real backward so θ moves off θ⁽ᵗ⁾.
    fn train(&mut self, h: u32) {
        let numel: u32 = self.dims.iter().product();
        for s in 0..h {
            let target = Tensor::full(&self.dims, Dtype::F32, 0.5);
            let diff = self.w[0].tensor().sub(&target);
            let sq = diff.mul(&diff);
            // Σ sq via [1, numel]·[numel, 1] ones → a rank-2 [1,1] scalar the tape can differentiate.
            let loss = sq
                .reshape(&[1, numel])
                .matmul(&Tensor::ones(&[numel, 1], Dtype::F32));
            loss.backward();
            self.w[0].adamw_step(
                &self.w[0].grad(),
                &self.m[0],
                &self.v[0],
                s + 1,
                0.1,
                0.9,
                0.999,
                1e-8,
                0.0,
            );
            daemon_train_sdk::zero_grads();
        }
    }
}

fn w_master() -> Vec<f32> {
    sim::param_master("w").unwrap()
}

// -- SDK-1: sparse_loco full round ---------------------------------------------------------------

fn sparse_loco_round(seed: u64, dims: &[u32], cfg: SparseLocoCfg) -> Vec<f32> {
    sim::reset(seed);
    let mut model = Model::build(dims);
    let mut sl = SparseLoco::new(cfg.clone(), &model.w);
    model.train(cfg.h);
    let u1 = sl.make_update(&model.w);
    sim::stage(&u1);
    let u2 = sl.make_update(&model.w); // self-inclusive second peer
    sim::stage(&u2);
    sl.ingest(&model.w, &UpdatesView::with_count(2));
    sim::snapshot_round_base();
    w_master()
}

#[test]
fn sdk1_sparse_loco_round_moves_and_is_reproducible() {
    let dims = [64u32]; // numel 64, chunk 16 ⇒ 4 chunks
    let cfg = SparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let a = sparse_loco_round(0xDAE0_7E57, &dims, cfg.clone());
    let b = sparse_loco_round(0xDAE0_7E57, &dims, cfg);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "same seed ⇒ bit-identical");
    }
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn sdk4_median_norm_clip_alters_dominated_aggregate() {
    let dims = [32u32];
    let run = |clip: bool| -> Vec<f32> {
        sim::reset(0x1234);
        let mut model = Model::build(&dims);
        let cfg = SparseLocoCfg {
            h: 2,
            chunk: 8,
            topk: 4,
            clip,
            ..SparseLocoCfg::default()
        };
        let mut sl = SparseLoco::new(cfg, &model.w);
        model.train(2);
        let u1 = sl.make_update(&model.w);
        sim::stage(&u1);
        model.train(8); // peer 2 trained much harder ⇒ a dominant-norm Δ
        let u2 = sl.make_update(&model.w);
        sim::stage(&u2);
        sl.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        w_master()
    };
    let clipped = run(true);
    let unclipped = run(false);
    assert!(clipped.iter().all(|v| v.is_finite()));
    assert_ne!(
        clipped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        unclipped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "median-norm clip must alter the aggregate when one peer dominates"
    );
}

// -- SDK-2/3: sparse payload compression ratio ---------------------------------------------------

#[test]
fn sdk2_sparse_loco_payload_beats_dense() {
    sim::reset(7);
    let dims = [256u32];
    let mut model = Model::build(&dims);
    let cfg = SparseLocoCfg {
        h: 2,
        chunk: 16,
        topk: 2,
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let mut sl = SparseLoco::new(cfg, &model.w);
    model.train(2);
    let ub = sl.make_update(&model.w);
    let packed = sim::section_len(&ub, 0); // packed 2-bit values
    assert!(
        packed < 256 * 4,
        "packed payload {packed} B must beat dense {} B",
        256 * 4
    );
}

// -- diloco --------------------------------------------------------------------------------------

#[test]
fn diloco_nesterov_differs_from_plain_and_is_finite() {
    let dims = [16u32];
    let run = |nesterov: bool| -> Vec<f32> {
        sim::reset(0xABCD);
        let mut model = Model::build(&dims);
        let cfg = DiLoCoCfg {
            h: 3,
            nesterov,
            ..DiLoCoCfg::default()
        };
        let mut dl = DiLoCo::new(cfg, &model.w);
        model.train(3);
        let u1 = dl.make_update(&model.w);
        sim::stage(&u1);
        let u2 = dl.make_update(&model.w);
        sim::stage(&u2);
        dl.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        w_master()
    };
    let nes = run(true);
    let plain = run(false);
    assert!(nes.iter().all(|v| v.is_finite()) && plain.iter().all(|v| v.is_finite()));
    assert_ne!(
        nes.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
}

#[test]
fn diloco_int8_payload_beats_dense() {
    sim::reset(11);
    let dims = [64u32];
    let mut model = Model::build(&dims);
    let cfg = DiLoCoCfg {
        h: 2,
        quant_bits: 8,
        ..DiLoCoCfg::default()
    };
    let mut dl = DiLoCo::new(cfg, &model.w);
    model.train(2);
    let ub = dl.make_update(&model.w);
    let bytes = sim::section_len(&ub, 0);
    assert!(bytes < 64 * 4, "int8 payload {bytes} B beats dense 256 B");
}

// -- demo ----------------------------------------------------------------------------------------

#[test]
fn demo_per_step_round_trips_and_is_reproducible() {
    let dims = [64u32]; // one 8×8 DCT tile
    let run = || -> Vec<f32> {
        sim::reset(0x5150);
        let mut model = Model::build(&dims);
        let cfg = DemoCfg {
            tile: 8,
            topk: 8,
            ..DemoCfg::default()
        };
        let mut demo = Demo::new(cfg, &model.w);
        model.train(1); // demo is H = 1 (per step)
        let u1 = demo.make_update(&model.w);
        sim::stage(&u1);
        let u2 = demo.make_update(&model.w);
        sim::stage(&u2);
        demo.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        w_master()
    };
    let a = run();
    let b = run();
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "demo must be bit-reproducible");
    }
    assert!(a.iter().all(|v| v.is_finite()));
}
