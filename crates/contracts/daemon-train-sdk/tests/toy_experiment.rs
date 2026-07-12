// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// A toy 2-layer dense experiment driven end to end through the SDK's `sim` backend (ABI §10.4):
// build → step → inner_update → make_update → ingest. Proves the whole lifecycle runs natively, the
// det-lane outer step returns the trained params under a self-inclusive single-peer aggregate, and
// the run is bit-reproducible for a fixed seed (the §7 within-peer replay property).
//
// The whole file is `sim`-only: under the default (native, no-sim) `cargo test --workspace` gate the
// tensor surface is `cfg`-gated out, so this compiles to an empty test crate.
#![cfg(feature = "sim")]

use daemon_train_sdk::prelude::*;
use daemon_train_sdk::sim;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
struct ToyCfg {
    d_in: u32,
    d_hidden: u32,
    d_out: u32,
    batch: u32,
    inner_steps: u32,
    lr: f64,
}

/// A trivial dense profile: `logits = relu(x·W1 + b1)·W2 + b2`, AdamW inner, DiLoCo-shaped outer.
struct Toy {
    cfg: ToyCfg,
    params: Vec<Param>, // [w1, b1, w2, b2] — registration order = canonical state dict
    m: Vec<Persistent>,
    v: Vec<Persistent>,
}

impl Experiment for Toy {
    fn manifest(cfg: &Config) -> Manifest {
        let cfg: ToyCfg = cfg.parse();
        Manifest::new("toy", "0.1.0", cfg.inner_steps)
    }

    fn build(cfg: &Config) -> Self {
        let cfg: ToyCfg = cfg.parse();
        let w1 = Param::new(
            "w1",
            &[cfg.d_in, cfg.d_hidden],
            Dtype::F32,
            Init::Normal,
            0.0,
            0.02,
        );
        let b1 = Param::new("b1", &[cfg.d_hidden], Dtype::F32, Init::Zeros, 0.0, 0.0);
        let w2 = Param::new(
            "w2",
            &[cfg.d_hidden, cfg.d_out],
            Dtype::F32,
            Init::Normal,
            0.0,
            0.02,
        );
        let b2 = Param::new("b2", &[cfg.d_out], Dtype::F32, Init::Zeros, 0.0, 0.0);
        let params = vec![w1, b1, w2, b2];
        let m = params
            .iter()
            .enumerate()
            .map(|(i, p)| Persistent::local(&format!("m{i}"), p.shape(), Dtype::F32))
            .collect();
        let v = params
            .iter()
            .enumerate()
            .map(|(i, p)| Persistent::local(&format!("v{i}"), p.shape(), Dtype::F32))
            .collect();
        Self { cfg, params, m, v }
    }

    fn step(&mut self, batch: &Batch, ctx: &StepCtx) {
        let b = batch.size();
        let x = Tensor::ones(&[b, self.cfg.d_in], Dtype::F32);
        let h = x
            .matmul(self.params[0].tensor())
            .add(self.params[1].tensor())
            .relu();
        let logits = h
            .matmul(self.params[2].tensor())
            .add(self.params[3].tensor());
        let targets = Tensor::zeros(&[b], Dtype::I32);
        let loss = logits.cross_entropy(&targets, -1);
        loss.metric("loss");
        loss.mul_s(ctx.loss_scale(batch)).backward();
    }

    fn inner_update(&mut self, inner_step: u32) {
        for (i, p) in self.params.iter().enumerate() {
            let g = p.grad();
            p.adamw_step(
                &g,
                &self.m[i],
                &self.v[i],
                inner_step + 1,
                self.cfg.lr,
                0.9,
                0.95,
                1.0e-8,
                0.1,
            );
        }
        zero_grads();
    }

    fn make_update(&mut self, _round: u64) -> UpdateBuilder {
        let mut ub = UpdateBuilder::new();
        for p in &self.params {
            // pseudo-gradient Δ = θ⁽ᵗ⁾ − θ_local (native lane, local math)
            let delta = p.round_base().sub(p.tensor());
            ub.push_tensor(&delta);
        }
        ub
    }

    fn ingest(&mut self, _round: u64, updates: &UpdatesView) {
        let count = updates.len().max(1) as f64;
        for (s, p) in self.params.iter().enumerate() {
            let mut acc = det_zeros(p.shape());
            for i in 0..updates.len() {
                let delta = updates.get(i).tensor(s as u32);
                acc = acc.add(&delta);
            }
            let mean = acc.scale(1.0 / count);
            // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·mean(Δ), α = 1 (det lane, canonical inputs, ABI §5.9)
            p.det_reset_to_base();
            p.det_axpy(&mean, -1.0);
        }
    }
}

/// (init masters, post-inner masters, post-round masters, reported metrics) for the four params.
type RoundSnapshot = (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<(String, f32)>,
);

fn cfg() -> Config {
    Config::from_value(&ToyCfg {
        d_in: 3,
        d_hidden: 4,
        d_out: 2,
        batch: 5,
        inner_steps: 3,
        lr: 0.1,
    })
}

/// Drive one full round; return the four params' post-round masters + the reported metrics, plus
/// the params captured right after `build` (init) and right after the inner steps (pre-outer).
fn drive_round(seed: u64) -> RoundSnapshot {
    sim::reset(seed);
    let cfg = cfg();

    // manifest is a pure function of config (ABI §6.2).
    assert_eq!(Toy::manifest(&cfg).steps_per_round, 3);

    let mut exp = Toy::build(&cfg);
    let names = ["w1", "b1", "w2", "b2"];
    let init: Vec<Vec<f32>> = names
        .iter()
        .map(|n| sim::param_master(n).unwrap())
        .collect();

    let batch = sim::make_batch(vec![0u32; 5], 5, 1);
    for s in 0..3 {
        exp.step(
            &batch,
            &StepCtx {
                inner_step: s,
                mb_index: 0,
                mb_count: 1,
                step_seqs: batch.size(),
            },
        );
        exp.inner_update(s);
    }
    let trained: Vec<Vec<f32>> = names
        .iter()
        .map(|n| sim::param_master(n).unwrap())
        .collect();

    // round end: compress → stage (self-inclusive) → ingest (outer step) — round_base is still the
    // build snapshot, so this is the θ⁽ᵗ⁾ baseline.
    let ub = exp.make_update(0);
    sim::stage(&ub);
    exp.ingest(0, &UpdatesView::with_count(1));
    // barrier: snapshot for the next round.
    sim::snapshot_round_base();

    let after: Vec<Vec<f32>> = names
        .iter()
        .map(|n| sim::param_master(n).unwrap())
        .collect();

    (init, trained, after, sim::metrics())
}

#[test]
fn toy_round_runs_end_to_end() {
    let (init, trained, after, metrics) = drive_round(0xDAE0_7E57);

    // Training moved the weights off their init.
    assert_ne!(init[0], trained[0], "w1 should change after inner steps");
    assert_ne!(init[2], trained[2], "w2 should change after inner steps");

    // Single self-inclusive peer, α = 1: the outer step returns exactly the locally-trained params
    // (θ_new = base − 1·(base − local) = local) — a clean, deterministic fixed point.
    for (t, a) in trained.iter().zip(after.iter()) {
        for (x, y) in t.iter().zip(a.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "outer step must reproduce local θ"
            );
        }
    }

    // metric@1 reported a finite loss on every inner step.
    let losses: Vec<f32> = metrics
        .iter()
        .filter(|(n, _)| n == "loss")
        .map(|(_, v)| *v)
        .collect();
    assert_eq!(losses.len(), 3);
    assert!(losses.iter().all(|l| l.is_finite() && *l > 0.0));
}

#[test]
fn toy_round_is_bit_reproducible() {
    let (_, _, a1, m1) = drive_round(0xDAE0_7E57);
    let (_, _, a2, m2) = drive_round(0xDAE0_7E57);
    for (p1, p2) in a1.iter().zip(a2.iter()) {
        for (x, y) in p1.iter().zip(p2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "same seed ⇒ bit-identical replay");
        }
    }
    assert_eq!(m1, m2);

    // A different seed changes the init (hence the whole run).
    let (_, _, a3, _) = drive_round(0x1234_5678);
    assert_ne!(a1[0], a3[0]);
}

#[test]
fn ingest_averages_multiple_peers() {
    // Two staged updates (self + one phantom) — the mean of two identical Δ is that Δ, so the outer
    // step still lands on the trained params; this exercises the multi-peer det aggregation path.
    sim::reset(0xABCD);
    let cfg = cfg();
    let mut exp = Toy::build(&cfg);
    let batch = sim::make_batch(vec![0u32; 5], 5, 1);
    for s in 0..3 {
        exp.step(
            &batch,
            &StepCtx {
                inner_step: s,
                mb_index: 0,
                mb_count: 1,
                step_seqs: 5,
            },
        );
        exp.inner_update(s);
    }
    let trained = sim::param_master("w2").unwrap();

    let ub1 = exp.make_update(0);
    sim::stage(&ub1);
    let ub2 = exp.make_update(0); // identical Δ (params unchanged since round base)
    sim::stage(&ub2);
    exp.ingest(0, &UpdatesView::with_count(2));

    let after = sim::param_master("w2").unwrap();
    for (t, a) in trained.iter().zip(after.iter()) {
        assert!((t - a).abs() < 1e-5, "mean of identical Δ ⇒ trained params");
    }
}

// The macro must at least expand (to nothing under `sim`) for a real Experiment type.
daemon_train_sdk::experiment!(Toy);
