// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// G2 — `BurnBackend(wgpu)` vs `CpuBackend` parity on a real Vulkan device (HOST-3 tolerance
// classes on GPU + the §6.6 absmax layout golden).
//
// This is G1's harness (`tests/tolerance/mod.rs`) with the backend factory swapped to
// `BurnWgpuBackend` — the exact reuse the harness was built for. The native lane is a *tolerance
// class* (GPU kernels vs the CpuBackend fixed-order fp32 tape, spec §7.2); the det lane +
// compression natives are **bit-exact** because `BurnBackend` materializes host-side and delegates
// to the same `det_core` kernels regardless of the burn backend (ABI §5.9 residency contract).
//
// GPU-skip convention (TDD §8.1 tier-2): every test that needs a device first checks
// `wgpu_adapter_available()` and skips with a loud stderr note when absent, so the default CI gate
// stays green on GPU-less runners while the `.#vulkan` devShell runs the full suite.
#![cfg(feature = "wgpu")]

mod tolerance;

use daemon_train::{wgpu_adapter_available, AdamwHp, BurnWgpuBackend, CpuBackend, OpBackend};
use tolerance::{assert_close, assert_parity, Fixture, OpClass};

/// The GPU-skip convention: bail loudly (stderr) when no usable wgpu adapter exists.
macro_rules! require_gpu {
    () => {
        if !wgpu_adapter_available() {
            eprintln!(
                "SKIP {}: no usable wgpu adapter on this runner (run in the .#vulkan devShell / \
                 on a GPU box for the full G2 suite — TDD §8.1 tier-2)",
                module_path!()
            );
            return;
        }
    };
}

fn backends() -> (BurnWgpuBackend, CpuBackend) {
    (BurnWgpuBackend::new(), CpuBackend::new())
}

// -- forward parity (per op, per class) ---------------------------------------------------------

#[test]
fn matmul_forward_parity() {
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
    let mut fx = Fixture::new();
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
    require_gpu!();
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
    require_gpu!();
    let mut fx = Fixture::new();
    let (rows, cols) = (5, 7);
    let logits = fx.vec(rows * cols);
    let targets: Vec<i64> = vec![1, 3, -1, 6, 0];
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
    require_gpu!();
    // reshape/transpose/slice are pure data moves — bit-exact even across a device round-trip.
    let mut fx = Fixture::new();
    let shape = [2usize, 3, 4];
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
    let s4 = [2usize, 3, 4, 5];
    let x4 = fx.vec(s4.iter().product());
    assert_parity(&mut ut, &mut rf, OpClass::Shape, "transpose4", |bk| {
        let tx = bk.create(x4.clone());
        let t = bk.transpose(tx, &s4, 1, 2);
        vec![bk.view(t).to_vec()]
    });
}

// -- HOST-9 autodiff parity (backward) on the GPU ------------------------------------------------

#[test]
fn abi_matmul_backward() {
    require_gpu!();
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
    require_gpu!();
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
    require_gpu!();
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

    let mut wgpu = BurnWgpuBackend::new();
    let full = w_grad(&mut wgpu, 1);
    let split = w_grad(&mut wgpu, 4);
    assert_close(&split, &full, OpClass::Optimizer, "wgpu accumulation split");
}

#[test]
fn abi_adamw_step_matches_burn() {
    require_gpu!();
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

// -- det lane + compression: bit-exact on the GPU build too (§5.9) -------------------------------
//
// Honesty note (G1's delegation, re-affirmed here): the compression natives (absmax_pack/unpack,
// dct2/idct2, topk_chunk) and every det_* op run **host-side det-core** in `BurnBackend` regardless
// of the burn backend — there is no GPU-native absmax kernel in this build (recorded as future work
// in G1's ledger). So HOST-3's "GPU-vs-CPU parity" is asserted at the `OpBackend` seam of the wgpu
// *build* (byte-exact, because both delegate), plus the §6.6 layout golden below, which pins the
// packed byte layout itself.

#[test]
fn det_lane_bit_exact() {
    require_gpu!();
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

/// HOST-3 `absmax_pack_golden`: the 2-bit and 1-bit absmax pack paths on the wgpu build are
/// byte-identical to `CpuBackend`, and pack∘unpack round-trips bit-exactly on both.
#[test]
fn absmax_pack_golden() {
    require_gpu!();
    let mut fx = Fixture::new();
    let x = fx.vec(64);
    let (mut ut, mut rf) = backends();
    for bits in [1u32, 2, 8] {
        assert_parity(&mut ut, &mut rf, OpClass::Exact, "absmax_pack", |bk| {
            let t = bk.create(x.clone());
            let packed = bk.absmax_pack(t, 16, bits).unwrap();
            let unpacked = bk.det_absmax_unpack(packed, 16, bits).unwrap();
            // pack(unpack(pack(x))) is a fixed point (det-core's stored-scale discipline).
            let repacked = bk.absmax_pack(unpacked, 16, bits).unwrap();
            vec![
                bk.view(packed).to_vec(),
                bk.view(unpacked).to_vec(),
                bk.view(repacked).to_vec(),
            ]
        });
    }
}

/// HOST-3 `absmax_layout_bytes_golden`: the §6.6 packed layout, pinned to literal bytes — per chunk
/// a little-endian f16 absmax codebook scalar, then `bits`-wide codes packed LSB-first, chunk-major.
/// Asserted through the wgpu build's `OpBackend::absmax_pack` (bytes carried as f32 values at the
/// seam) AND against `det_core::absmax_pack` directly, so the golden pins both the layout and the
/// delegation.
#[test]
fn absmax_layout_bytes_golden() {
    require_gpu!();
    // 2-bit, chunk 4, two chunks. absmax(c0) = 1.0 (f16 0x3C00), absmax(c1) = 0.25 (f16 0x3400).
    // codes(c0): 1.0→3, -1.0→0, 0.5→2, 0.0→2  → LSB-first byte 0b10_10_00_11 = 0xA3
    // codes(c1): 0.25→3, -0.25→0, 0.125→2, -0.125→1 → byte 0b01_10_00_11 = 0x63
    let x = vec![1.0_f32, -1.0, 0.5, 0.0, 0.25, -0.25, 0.125, -0.125];
    let want: Vec<u8> = vec![0x00, 0x3C, 0xA3, 0x00, 0x34, 0x63];
    assert_eq!(
        det_core::absmax_pack(&x, 4, 2).unwrap(),
        want,
        "det-core §6.6 2-bit layout golden"
    );

    let mut bk = BurnWgpuBackend::new();
    let t = bk.create(x.clone());
    let packed = bk.absmax_pack(t, 4, 2).unwrap();
    let got: Vec<u8> = bk.view(packed).iter().map(|&f| f as u8).collect();
    assert_eq!(got, want, "wgpu-build OpBackend::absmax_pack layout golden");

    // 1-bit, chunk 8, one chunk. absmax = 3.0 (f16 0x4200); signs +−+−+−+− → LSB-first 0x55.
    let x1 = vec![3.0_f32, -3.0, 1.5, -1.5, 0.75, -0.75, 3.0, -3.0];
    let want1: Vec<u8> = vec![0x00, 0x42, 0x55];
    assert_eq!(
        det_core::absmax_pack(&x1, 8, 1).unwrap(),
        want1,
        "det-core §6.6 1-bit layout golden"
    );
    let t1 = bk.create(x1);
    let packed1 = bk.absmax_pack(t1, 8, 1).unwrap();
    let got1: Vec<u8> = bk.view(packed1).iter().map(|&f| f as u8).collect();
    assert_eq!(got1, want1, "wgpu-build 1-bit layout golden");
}

#[test]
fn compression_natives_bit_exact() {
    require_gpu!();
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
