// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// HOST-11 (TDD §3.5): the LLaMA-preset numerics — RMSNorm / RoPE / SwiGLU / attention — checked
// against an **independent oracle**. The oracle here is a from-definition naive implementation
// written directly in this test (distinct code from the `sim` backend's implementation in
// `sim.rs`), plus a handful of hand-verifiable literal anchors (e.g. silu(0)=0, rmsnorm of a
// constant vector). This is the documented alternative to a llama-burn/tch fixture (the task's
// "compute via an independent implementation … document the oracle"): it needs no GPU, no wasm host,
// and no heavyweight dep, and it exercises the exact ops the 160M preset composes (models.rs).
//
// sim-only: under the default no-sim gate the tensor surface is cfg'd out.
#![cfg(feature = "sim")]

use daemon_train_sdk::sim;
use daemon_train_sdk::{Dtype, Init, Param, Tensor};

const SEED: u64 = 0xDAE0_7E57;
/// Relative+absolute closeness for the fp32 op vs the fp64 oracle (native lane is fp32).
const TOL: f64 = 5.0e-4;

fn assert_close(got: &[f32], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let g = f64::from(g);
        let diff = (g - w).abs();
        let tol = TOL * (1.0 + w.abs());
        assert!(
            diff <= tol,
            "{ctx}[{i}]: got {g}, want {w} (|Δ|={diff} > {tol})"
        );
    }
}

/// Register a param with deterministic Normal-init values and return (param, its fp32 values).
fn normal_param(name: &str, dims: &[u32], mean: f64, std: f64) -> (Param, Vec<f32>) {
    let p = Param::new(name, dims, Dtype::F32, Init::Normal, mean, std);
    let v = sim::param_master(name).expect("registered");
    (p, v)
}

/// Copy a computed step tensor into a fresh param so its values can be read back via the sim store.
fn readback(name: &str, dims: &[u32], t: &Tensor) -> Vec<f32> {
    let out = Param::new(name, dims, Dtype::F32, Init::Zeros, 0.0, 0.0);
    out.assign(t);
    sim::param_master(name).expect("readback param")
}

#[test]
fn rmsnorm_golden() {
    sim::reset(SEED);
    let (rows, d) = (3u32, 8u32);
    let eps = 1.0e-5;
    let (x, xv) = normal_param("rms.x", &[rows, d], 0.0, 1.0);
    let (w, wv) = normal_param("rms.w", &[d], 1.0, 0.1);

    let got = readback("rms.out", &[rows, d], &x.tensor().rmsnorm(w.tensor(), eps));

    // Oracle: per row, out_i = x_i / sqrt(mean(x^2)+eps) * w_i.
    let d = d as usize;
    let mut want = vec![0.0f64; xv.len()];
    for r in 0..rows as usize {
        let row = &xv[r * d..(r + 1) * d];
        let ms = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            want[r * d + i] = f64::from(row[i]) * inv * f64::from(wv[i]);
        }
    }
    assert_close(&got, &want, "rmsnorm");

    // Hand anchor: a constant vector normalizes to (value/|value|)·w when eps→0.
    sim::reset(SEED);
    let cst = Param::new("rms.c", &[1, 4], Dtype::F32, Init::Ones, 0.0, 0.0);
    let w1 = Param::new("rms.cw", &[4], Dtype::F32, Init::Ones, 0.0, 0.0);
    let got = readback("rms.co", &[1, 4], &cst.tensor().rmsnorm(w1.tensor(), 0.0));
    assert_close(&got, &[1.0, 1.0, 1.0, 1.0], "rmsnorm-ones");
}

#[test]
fn rope_golden() {
    sim::reset(SEED);
    let (s, hd) = (4u32, 8u32);
    let theta = 10_000.0;
    let (x, xv) = normal_param("rope.x", &[1, 1, s, hd], 0.0, 1.0);

    let got = readback(
        "rope.out",
        &[1, 1, s, hd],
        &x.tensor().rope(0, theta, false),
    );

    // Oracle: non-interleaved RoPE, pairs (j, j+hd/2), angle = pos · theta^(-2j/hd), pos = row.
    let (s, hd) = (s as usize, hd as usize);
    let mut want = vec![0.0f64; xv.len()];
    for r in 0..s {
        let pos = r as f64;
        for j in 0..hd / 2 {
            let freq = theta.powf(-2.0 * j as f64 / hd as f64);
            let (c, si) = (pos * freq).cos_sin();
            let (ia, ib) = (j, j + hd / 2);
            let (a, b) = (f64::from(xv[r * hd + ia]), f64::from(xv[r * hd + ib]));
            want[r * hd + ia] = a * c - b * si;
            want[r * hd + ib] = a * si + b * c;
        }
    }
    assert_close(&got, &want, "rope");

    // Hand anchor: row 0 (pos 0) is the identity (cos 0 = 1, sin 0 = 0).
    assert_close(
        &got[..hd],
        &xv[..hd].iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        "rope-pos0",
    );
}

#[test]
fn swiglu_golden() {
    sim::reset(SEED);
    let (rows, hidden) = (2u32, 6u32);
    let (a, av) = normal_param("sg.a", &[rows, hidden], 0.0, 1.0);
    let (b, bv) = normal_param("sg.b", &[rows, hidden], 0.0, 1.0);

    // SwiGLU gate: silu(a) · b (models.rs: gate.silu() then mul(&up)).
    let got = readback(
        "sg.out",
        &[rows, hidden],
        &a.tensor().silu().mul(b.tensor()),
    );

    // Oracle: silu(x) = x·sigmoid(x); then elementwise product.
    let want: Vec<f64> = av
        .iter()
        .zip(bv.iter())
        .map(|(&x, &y)| {
            let x = f64::from(x);
            let silu = x / (1.0 + (-x).exp());
            silu * f64::from(y)
        })
        .collect();
    assert_close(&got, &want, "swiglu");

    // Hand anchors: silu(0)=0; silu(large)≈large.
    sim::reset(SEED);
    let z = Param::new("sg.z", &[1, 2], Dtype::F32, Init::Zeros, 0.0, 0.0);
    let got = readback("sg.zo", &[1, 2], &z.tensor().silu());
    assert_close(&got, &[0.0, 0.0], "silu-zero");
}

#[test]
fn attention_golden() {
    sim::reset(SEED);
    let (b, h, s, hd) = (1u32, 2u32, 4u32, 8u32);
    let scale = 1.0 / f64::from(hd).sqrt();
    let (q, qv) = normal_param("at.q", &[b, h, s, hd], 0.0, 1.0);
    let (k, kv) = normal_param("at.k", &[b, h, s, hd], 0.0, 1.0);
    let (v, vv) = normal_param("at.v", &[b, h, s, hd], 0.0, 1.0);

    let got = readback(
        "at.out",
        &[b, h, s, hd],
        &q.tensor().flash_attn(k.tensor(), v.tensor(), true, scale),
    );

    // Oracle: per (b·h) group, causal softmax attention (mask j>i), out = Σ_j p_ij · v_j.
    let (h, s, hd) = (h as usize, s as usize, hd as usize);
    let bh = b as usize * h;
    let mut want = vec![0.0f64; qv.len()];
    for g in 0..bh {
        let base = g * s * hd;
        for i in 0..s {
            let mut scores = vec![f64::NEG_INFINITY; s];
            let mut maxv = f64::NEG_INFINITY;
            for j in 0..=i {
                let mut dot = 0.0;
                for e in 0..hd {
                    dot += f64::from(qv[base + i * hd + e]) * f64::from(kv[base + j * hd + e]);
                }
                scores[j] = dot * scale;
                maxv = maxv.max(scores[j]);
            }
            let mut denom = 0.0;
            let mut probs = vec![0.0f64; s];
            for j in 0..=i {
                probs[j] = (scores[j] - maxv).exp();
                denom += probs[j];
            }
            for e in 0..hd {
                let mut acc = 0.0;
                for j in 0..=i {
                    acc += probs[j] / denom * f64::from(vv[base + j * hd + e]);
                }
                want[base + i * hd + e] = acc;
            }
        }
    }
    assert_close(&got, &want, "attention");

    // Hand anchor: query row 0 attends only to key 0 (causal) ⇒ out row 0 == v row 0.
    let row0: Vec<f64> = vv[..hd].iter().map(|&x| f64::from(x)).collect();
    assert_close(&got[..hd], &row0, "attn-causal-row0");
}

/// Small helper: `(cos, sin)` of an angle.
trait CosSin {
    fn cos_sin(self) -> (f64, f64);
}
impl CosSin for f64 {
    fn cos_sin(self) -> (f64, f64) {
        (self.cos(), self.sin())
    }
}
