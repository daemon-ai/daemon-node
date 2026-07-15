// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// G1 — `BurnBackend(ndarray)` vs `CpuBackend` parity (HOST-3 tolerance classes + HOST-9 autodiff).
//
// The native lane is a *tolerance class*, not bit-identical (burn's autodiff vs the CpuBackend tape,
// program ledger "Determinism story", spec §7.2). These tests pin the per-op rtol/atol via the
// shared `tolerance` harness and prove `BurnBackend` maps every native op onto burn tensor ops with
// matching forward outputs + backward grads, while the det lane / compression natives stay
// **bit-exact** (both delegate to det-core). G2 reuses the harness with a wgpu backend.
#![cfg(feature = "burn-ndarray")]

mod tolerance;

use daemon_train::{AdamwHp, BurnNdarrayBackend, CpuBackend, OpBackend};
use tolerance::{assert_close, assert_parity, Fixture, OpClass};

fn backends() -> (BurnNdarrayBackend, CpuBackend) {
    (BurnNdarrayBackend::new(), CpuBackend::new())
}

// -- forward parity (per op, per class) ---------------------------------------------------------

#[test]
fn matmul_forward_parity() {
    let mut fx = Fixture::new();
    let (m, k, n) = (3, 4, 5);
    let a = fx.vec(m * k);
    let b = fx.vec(k * n);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::MatmulReduce, "matmul", |bk| {
        let ta = bk.create(a.clone());
        let tb = bk.create(b.clone());
        let c = bk.matmul(ta, m, k, tb, n);
        vec![bk.view(c).to_vec()]
    });
}

#[test]
fn elementwise_forward_parity() {
    let mut fx = Fixture::new();
    let n = 32;
    let a = fx.vec(n);
    let b = fx.vec(n);
    let (mut ut, mut rf) = backends();
    assert_parity(
        &mut ut,
        &mut rf,
        OpClass::Elementwise,
        "elementwise",
        |bk| {
            let ta = bk.create(a.clone());
            let tb = bk.create(b.clone());
            let add = bk.add(ta, tb);
            let sub = bk.sub(ta, tb);
            let mul = bk.mul(ta, tb);
            let muls = bk.mul_s(ta, 1.5);
            let relu = bk.relu(tb);
            vec![
                bk.view(add).to_vec(),
                bk.view(sub).to_vec(),
                bk.view(mul).to_vec(),
                bk.view(muls).to_vec(),
                bk.view(relu).to_vec(),
            ]
        },
    );
}

#[test]
fn add_bias_forward_parity() {
    let mut fx = Fixture::new();
    let (rows, cols) = (4, 6);
    let a = fx.vec(rows * cols);
    let bias = fx.vec(cols);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Elementwise, "add_bias", |bk| {
        let ta = bk.create(a.clone());
        let tb = bk.create(bias.clone());
        let out = bk.add_bias(ta, tb, rows, cols);
        vec![bk.view(out).to_vec()]
    });
}

#[test]
fn embedding_forward_parity() {
    let mut fx = Fixture::new();
    let (vocab, d) = (10, 4);
    let w = fx.vec(vocab * d);
    let ids = vec![2usize, 0, 7, 3, 3];
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::MatmulReduce, "embedding", |bk| {
        let tw = bk.create(w.clone());
        let out = bk.embedding(tw, &ids, d);
        vec![bk.view(out).to_vec()]
    });
}

#[test]
fn rmsnorm_forward_parity() {
    let mut fx = Fixture::new();
    let (rows, d) = (3, 8);
    let x = fx.vec(rows * d);
    let w = fx.vec(d);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Normalization, "rmsnorm", |bk| {
        let tx = bk.create(x.clone());
        let tw = bk.create(w.clone());
        let out = bk.rmsnorm(tx, tw, rows, d, 1.0e-5);
        vec![bk.view(out).to_vec()]
    });
}

#[test]
fn silu_softmax_forward_parity() {
    let mut fx = Fixture::new();
    let (outer, dimlen, inner) = (2, 5, 3);
    let x = fx.vec(outer * dimlen * inner);
    let (mut ut, mut rf) = backends();
    assert_parity(
        &mut ut,
        &mut rf,
        OpClass::Normalization,
        "silu_softmax",
        |bk| {
            let tx = bk.create(x.clone());
            let silu = bk.silu(tx);
            let sm = bk.softmax(tx, outer, dimlen, inner);
            vec![bk.view(silu).to_vec(), bk.view(sm).to_vec()]
        },
    );
}

#[test]
fn rope_forward_parity() {
    let mut fx = Fixture::new();
    // [rows, hd] with rows = b*nh*s periodic in seq.
    let (rows, seq, hd) = (6, 3, 8);
    let x = fx.vec(rows * hd);
    let (mut ut, mut rf) = backends();
    for interleaved in [false, true] {
        assert_parity(&mut ut, &mut rf, OpClass::Normalization, "rope", |bk| {
            let tx = bk.create(x.clone());
            let out = bk.rope(tx, rows, seq, hd, 0, 10_000.0, interleaved);
            vec![bk.view(out).to_vec()]
        });
    }
}

#[test]
fn flash_attn_forward_parity() {
    let mut fx = Fixture::new();
    let (bh, s, d) = (2, 4, 3);
    let q = fx.vec(bh * s * d);
    let k = fx.vec(bh * s * d);
    let v = fx.vec(bh * s * d);
    let scale = 1.0 / (d as f64).sqrt();
    let (mut ut, mut rf) = backends();
    for causal in [true, false] {
        assert_parity(&mut ut, &mut rf, OpClass::Attention, "flash_attn", |bk| {
            let tq = bk.create(q.clone());
            let tk = bk.create(k.clone());
            let tv = bk.create(v.clone());
            let out = bk.flash_attn(tq, tk, tv, bh, s, d, causal, scale);
            vec![bk.view(out).to_vec()]
        });
    }
}

#[test]
fn cross_entropy_forward_parity() {
    let mut fx = Fixture::new();
    let (rows, cols) = (5, 7);
    let logits = fx.vec(rows * cols);
    let targets: Vec<i64> = vec![1, 3, -1, 6, 0]; // one ignored (-1)
    let (mut ut, mut rf) = backends();
    assert_parity(
        &mut ut,
        &mut rf,
        OpClass::Attention,
        "cross_entropy",
        |bk| {
            let tl = bk.create(logits.clone());
            let loss = bk.cross_entropy(tl, rows, cols, &targets, -1);
            vec![bk.view(loss).to_vec()]
        },
    );
}

#[test]
fn shape_ops_bit_exact() {
    // reshape/transpose/slice are pure data moves — bit-exact across backends.
    let mut fx = Fixture::new();
    let shape = [2usize, 3, 4]; // rank 3
    let x = fx.vec(shape.iter().product());
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Shape, "shape_ops", |bk| {
        let tx = bk.create(x.clone());
        let r = bk.reshape(tx);
        let t = bk.transpose(tx, &shape, 0, 2);
        let s = bk.slice(tx, &shape, 1, 1, 3);
        vec![
            bk.view(r).to_vec(),
            bk.view(t).to_vec(),
            bk.view(s).to_vec(),
        ]
    });
    // rank-4 transpose (the tiny-llama attention layout) + rank-2 transpose (tied logits).
    let s4 = [2usize, 3, 4, 5];
    let x4 = fx.vec(s4.iter().product());
    assert_parity(&mut ut, &mut rf, OpClass::Shape, "transpose4", |bk| {
        let tx = bk.create(x4.clone());
        let t = bk.transpose(tx, &s4, 1, 2);
        vec![bk.view(t).to_vec()]
    });
}

// -- HOST-9 autodiff parity (backward) ----------------------------------------------------------

#[test]
fn abi_matmul_backward() {
    // matmul → cross_entropy → backward: grads of both matmul inputs match the reference tape.
    let mut fx = Fixture::new();
    let (rows, k, cols) = (4, 6, 5);
    let h = fx.vec(rows * k);
    let w = fx.vec(k * cols);
    let targets: Vec<i64> = (0..rows).map(|i| (i % cols) as i64).collect();
    let (mut ut, mut rf) = backends();
    assert_parity(
        &mut ut,
        &mut rf,
        OpClass::Attention,
        "matmul_backward",
        |bk| {
            bk.begin_pass();
            let th = bk.create(h.clone());
            let tw = bk.create(w.clone());
            let logits = bk.matmul(th, rows, k, tw, cols);
            let loss = bk.cross_entropy(logits, rows, cols, &targets, -1);
            bk.backward(loss);
            let gh = bk.grad_of(th).expect("grad of h");
            let gw = bk.grad_of(tw).expect("grad of w");
            bk.end_pass();
            vec![gh, gw]
        },
    );
}

#[test]
fn attention_stack_backward_parity() {
    // A deeper chain (rmsnorm → matmul → silu → matmul → cross_entropy) exercises the full backward
    // graph, the HOST-9 "autodiff parity vs compiled-in Burn" acceptance across op classes.
    let mut fx = Fixture::new();
    let (rows, d, hidden, vocab) = (4, 8, 16, 6);
    let x = fx.vec(rows * d);
    let norm_w = fx.vec_scaled(d, 0.5);
    let w1 = fx.vec_scaled(d * hidden, 0.3);
    let w2 = fx.vec_scaled(hidden * vocab, 0.3);
    let targets: Vec<i64> = (0..rows).map(|i| (i % vocab) as i64).collect();
    let (mut ut, mut rf) = backends();
    assert_parity(
        &mut ut,
        &mut rf,
        OpClass::Attention,
        "stack_backward",
        |bk| {
            bk.begin_pass();
            let tx = bk.create(x.clone());
            let tnw = bk.create(norm_w.clone());
            let tw1 = bk.create(w1.clone());
            let tw2 = bk.create(w2.clone());
            let normed = bk.rmsnorm(tx, tnw, rows, d, 1.0e-5);
            let gate = bk.matmul(normed, rows, d, tw1, hidden);
            let act = bk.silu(gate);
            let logits = bk.matmul(act, rows, hidden, tw2, vocab);
            let loss = bk.cross_entropy(logits, rows, vocab, &targets, -1);
            bk.backward(loss);
            let g = vec![
                bk.grad_of(tx).expect("grad x"),
                bk.grad_of(tnw).expect("grad norm_w"),
                bk.grad_of(tw1).expect("grad w1"),
                bk.grad_of(tw2).expect("grad w2"),
            ];
            bk.end_pass();
            g
        },
    );
}

#[test]
fn grads_invariant_to_accumulation_split() {
    // HOST-9 (re-run against BurnBackend): the accumulated W grad is invariant to host micro-batch
    // slicing — mb_count=1 (whole batch) vs mb_count=4 (mean-loss micro-batches scaled by their row
    // fraction and summed, exactly the runtime's `op_backward` accumulation). Proven for both
    // backends; BurnBackend is the lane under test.
    fn w_grad(bk: &mut dyn OpBackend, mb: usize) -> Vec<f32> {
        let mut fx = Fixture::new();
        let (rows, k, cols) = (8usize, 6usize, 5usize);
        let h = fx.vec(rows * k);
        let w = fx.vec(k * cols);
        let targets: Vec<i64> = (0..rows).map(|i| (i % cols) as i64).collect();
        let tw = bk.create(w);
        let per = rows / mb;
        let mut acc = vec![0.0_f32; k * cols];
        for q in 0..mb {
            let r0 = q * per;
            let hs = h[r0 * k..(r0 + per) * k].to_vec();
            let ts = targets[r0..r0 + per].to_vec();
            bk.begin_pass();
            let th = bk.create(hs);
            let logits = bk.matmul(th, per, k, tw, cols);
            let loss = bk.cross_entropy(logits, per, cols, &ts, -1);
            // Scale each micro-batch mean loss by its row fraction so the sum equals the full mean.
            let scaled = bk.mul_s(loss, per as f64 / rows as f64);
            bk.backward(scaled);
            if let Some(g) = bk.grad_of(tw) {
                for (a, v) in acc.iter_mut().zip(g) {
                    *a += v;
                }
            }
            bk.end_pass();
        }
        acc
    }

    let mut burn = BurnNdarrayBackend::new();
    let full = w_grad(&mut burn, 1);
    let split = w_grad(&mut burn, 4);
    assert_close(&split, &full, OpClass::Optimizer, "burn accumulation split");

    let mut cpu = CpuBackend::new();
    let full_c = w_grad(&mut cpu, 1);
    let split_c = w_grad(&mut cpu, 4);
    assert_close(
        &split_c,
        &full_c,
        OpClass::Optimizer,
        "cpu accumulation split",
    );
}

#[test]
fn abi_adamw_step_matches_burn() {
    // The ABI's fused `adamw_step` on the burn native path matches the reference closed-form AdamW
    // (CpuBackend) within the Optimizer tolerance class, over five steps (moments accumulate). "burn"
    // is the native engine under test; CpuBackend is the closed-form oracle.
    let mut fx = Fixture::new();
    let n = 64;
    let w = fx.vec_scaled(n, 0.1);
    let g = fx.vec_scaled(n, 0.05);
    let hp = AdamwHp {
        step: 1,
        lr: 4.0e-4,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1.0e-8,
        wd: 0.1,
    };
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Optimizer, "adamw", |bk| {
        let master = bk.create(w.clone());
        let grad = bk.create(g.clone());
        let m = bk.zeros(n);
        let v = bk.zeros(n);
        for step in 1..=5 {
            bk.adamw_step(master, grad, m, v, AdamwHp { step, ..hp });
        }
        vec![
            bk.view(master).to_vec(),
            bk.view(m).to_vec(),
            bk.view(v).to_vec(),
        ]
    });
}

// -- det lane + compression: bit-exact across backends (§5.9, HOST-3 golden) --------------------

#[test]
fn det_lane_bit_exact() {
    let mut fx = Fixture::new();
    let n = 64;
    let x = fx.vec(n);
    let y = fx.vec(n);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Exact, "det_lane", |bk| {
        let a = bk.create(x.clone());
        let b = bk.create(y.clone());
        let sum = bk.det_sum(&[a, b]).unwrap();
        let add = bk.det_add(a, b).unwrap();
        let sub = bk.det_sub(a, b).unwrap();
        let mul = bk.det_mul(a, b).unwrap();
        let scale = bk.det_scale(a, 0.25);
        let sign = bk.det_sign(a);
        let norm = bk.det_l2norm(a);
        vec![
            bk.view(sum).to_vec(),
            bk.view(add).to_vec(),
            bk.view(sub).to_vec(),
            bk.view(mul).to_vec(),
            bk.view(scale).to_vec(),
            bk.view(sign).to_vec(),
            vec![norm],
        ]
    });
}

#[test]
fn det_axpy_bit_exact() {
    let mut fx = Fixture::new();
    let n = 32;
    let y0 = fx.vec(n);
    let x = fx.vec(n);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Exact, "det_axpy", |bk| {
        let y = bk.create(y0.clone());
        let tx = bk.create(x.clone());
        bk.det_axpy(y, -0.5, tx).unwrap();
        vec![bk.view(y).to_vec()]
    });
}

#[test]
fn compression_natives_bit_exact() {
    let mut fx = Fixture::new();
    let x = fx.vec(64);
    let (mut ut, mut rf) = backends();
    assert_parity(&mut ut, &mut rf, OpClass::Exact, "compression", |bk| {
        let t = bk.create(x.clone());
        let (vals, idx) = bk.topk_chunk(t, 16, 4).unwrap();
        let packed = bk.absmax_pack(t, 16, 8).unwrap();
        let dct = bk.dct2(t, 8).unwrap();
        let inv = bk.idct2(dct, 8).unwrap();
        let unpacked = bk.det_absmax_unpack(packed, 16, 8).unwrap();
        vec![
            bk.view(vals).to_vec(),
            bk.view(idx).to_vec(),
            bk.view(packed).to_vec(),
            bk.view(dct).to_vec(),
            bk.view(inv).to_vec(),
            bk.view(unpacked).to_vec(),
        ]
    });
}
