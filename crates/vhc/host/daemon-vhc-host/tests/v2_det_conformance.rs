// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The det-lane reclassification conformance gate (tier-1; architecture §3.2/§3.6, refactor §7).
//
// Phase C reframes the det lane: `daemon-vhc-det` is the **normative** definition of the consensus
// math, and the host `det_*` imports are an **acceleration** of that same crate — not a second
// implementation that happens to agree. The always-available compatibility path is the crate
// compiled *in-guest* (the SDK `sim` wasm build): because wasm core semantics are deterministic
// (wasmtime canonicalises NaNs) and the det kernels are fixed-order fp32, the in-guest crate and
// the native crate are bit-identical by construction. This gate asserts the remaining half — that
// the **host acceleration** the worker actually runs (the `OpBackend` `det_*` methods, exercised
// here through the reference `CpuBackend`; `BurnBackend<B>` delegates through the identical
// `daemon_vhc_det` calls) is **bit-identical** to calling the normative crate directly, for
// **every** det import, over a wide deterministic sweep.
//
// This is the det twin of the `sys@2` crypto conformance gate (`v2_crypto.rs`): one dual-compiled
// contract, host accel ≡ in-guest contract asserted standingly rather than trusting two paths to
// agree. It guards against a future host-side det reimplementation (or a marshalling bug in the
// handle round-trip) drifting from the crate the guest fallback pins. It is CPU-deterministic (no
// wasm host, no GPU, no network) — a `swarm-ci-det` citizen. Bit-identity is asserted on the raw
// f32 bit pattern (`to_bits`), the equality-class contract the swarm's agree-path leans on.

use daemon_vhc_abi::DET_ACCEL_OPS;
use daemon_vhc_host::{CpuBackend, OpBackend};

/// A tiny deterministic xorshift64* — test data only, never consensus-relevant (mirrors the
/// `daemon-vhc-det` host-kernel suite's generator so the sweeps are comparable).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.signed()).collect()
    }
}

/// Bit-exact f32-slice equality (equality class, not tolerance): compares the raw bit pattern so
/// `-0.0`/`NaN` payloads must match too. Names the op for a legible failure.
#[track_caller]
fn assert_bits_eq(op: &str, host: &[f32], crate_ref: &[f32]) {
    assert_eq!(
        host.len(),
        crate_ref.len(),
        "{op}: host len {} ≠ crate len {}",
        host.len(),
        crate_ref.len()
    );
    for (i, (h, c)) in host.iter().zip(crate_ref).enumerate() {
        assert_eq!(
            h.to_bits(),
            c.to_bits(),
            "{op}: element {i} diverged: host {h} ({:#010x}) ≠ crate {c} ({:#010x})",
            h.to_bits(),
            c.to_bits()
        );
    }
}

#[test]
fn host_det_ops_are_bit_identical_to_the_normative_crate() {
    let mut rng = Rng::new(0xDAE0_C2DE);
    // A wide length sweep; chunk-shaped ops use divisor-clean lengths below.
    for &n in &[0usize, 1, 2, 3, 7, 8, 16, 33, 64, 100, 257, 1024] {
        let a = rng.vec(n);
        let b = rng.vec(n);
        let mut be = CpuBackend::new();

        // det_add / det_sub / det_mul (same-shape elementwise).
        let (ha, hb) = (be.create(a.clone()), be.create(b.clone()));
        let hs = be.det_add(ha, hb).unwrap();
        assert_bits_eq(
            "det_add",
            be.view(hs),
            &daemon_vhc_det::det_add(&a, &b).unwrap(),
        );
        let (ha, hb) = (be.create(a.clone()), be.create(b.clone()));
        let hd = be.det_sub(ha, hb).unwrap();
        assert_bits_eq(
            "det_sub",
            be.view(hd),
            &daemon_vhc_det::det_sub(&a, &b).unwrap(),
        );
        let (ha, hb) = (be.create(a.clone()), be.create(b.clone()));
        let hm = be.det_mul(ha, hb).unwrap();
        assert_bits_eq(
            "det_mul",
            be.view(hm),
            &daemon_vhc_det::det_mul(&a, &b).unwrap(),
        );

        // det_sum over a record-ordered set (0, 1, and 3 operands — order is normative).
        for set in [
            vec![a.clone()],
            vec![a.clone(), b.clone()],
            vec![a.clone(), b.clone(), a.clone()],
        ] {
            let ids: Vec<_> = set.iter().map(|v| be.create(v.clone())).collect();
            let refs: Vec<&[f32]> = set.iter().map(Vec::as_slice).collect();
            let hsum = be.det_sum(&ids).unwrap();
            assert_bits_eq(
                "det_sum",
                be.view(hsum),
                &daemon_vhc_det::det_sum(&refs).unwrap(),
            );
        }

        // det_scale / det_sign / det_l2norm (the last returns a scalar f32).
        for &alpha in &[0.0_f64, 1.0, -2.5, 0.333_333_333_333] {
            let hx = be.create(a.clone());
            let hsc = be.det_scale(hx, alpha);
            assert_bits_eq(
                "det_scale",
                be.view(hsc),
                &daemon_vhc_det::det_scale(&a, alpha),
            );
        }
        let hx = be.create(a.clone());
        let hsg = be.det_sign(hx);
        assert_bits_eq("det_sign", be.view(hsg), &daemon_vhc_det::det_sign(&a));
        let hx = be.create(a.clone());
        assert_eq!(
            be.det_l2norm(hx).to_bits(),
            daemon_vhc_det::det_l2norm(&a).to_bits(),
            "det_l2norm diverged at n={n}"
        );

        // det_axpy — in-place `y += alpha*x` (normative accumulation order).
        for &alpha in &[1.0_f64, -0.75, 3.0] {
            let hy = be.create(b.clone());
            let hx = be.create(a.clone());
            be.det_axpy(hy, alpha, hx).unwrap();
            let mut yref = b.clone();
            daemon_vhc_det::det_axpy(&mut yref, alpha, &a).unwrap();
            assert_bits_eq("det_axpy", be.view(hy), &yref);
        }
    }
}

#[test]
fn host_chunk_and_transform_det_ops_match_the_crate() {
    let mut rng = Rng::new(0x0C2D_7E57);
    // Chunk-shaped ops: length a multiple of `chunk`. DCT ops: length a multiple of `tile²`.
    for &chunk in &[4usize, 8, 16, 64] {
        for nchunks in [1usize, 2, 5] {
            let n = chunk * nchunks;
            let x = rng.vec(n);
            let mut be = CpuBackend::new();

            // absmax_pack (bytes, exposed as f32) then det_absmax_unpack, for each supported width.
            for &bits in &[1u32, 2, 4, 8] {
                let hx = be.create(x.clone());
                let hpacked = be.absmax_pack(hx, chunk, bits).unwrap();
                let packed_ref = daemon_vhc_det::absmax_pack(&x, chunk, bits).unwrap();
                let packed_as_f32: Vec<f32> = packed_ref.iter().map(|&b| f32::from(b)).collect();
                assert_bits_eq("absmax_pack", be.view(hpacked), &packed_as_f32);

                // det_absmax_unpack consumes the packed bytes (host: f32-encoded per byte).
                let hpk = be.create(packed_as_f32.clone());
                let hunp = be.det_absmax_unpack(hpk, chunk, bits).unwrap();
                let unp_ref = daemon_vhc_det::det_absmax_unpack(&packed_ref, chunk, bits).unwrap();
                assert_bits_eq("det_absmax_unpack", be.view(hunp), &unp_ref);
            }

            // topk_chunk (k ≤ chunk): values bit-exact, indices exact (host exposes them as f32).
            for k in [1usize, chunk / 2 + 1, chunk] {
                let k = k.clamp(1, chunk);
                let hx = be.create(x.clone());
                let (hv, hi) = be.topk_chunk(hx, chunk, k).unwrap();
                let (vref, iref) = daemon_vhc_det::topk_chunk(&x, chunk, k).unwrap();
                assert_bits_eq("topk_chunk values", be.view(hv), &vref);
                let iref_f32: Vec<f32> = iref.iter().map(|&i| i as f32).collect();
                assert_bits_eq("topk_chunk indices", be.view(hi), &iref_f32);
            }

            // det_chunk_scatter / det_chunk_scatter_add — `[nchunks, picks]` sparse payload
            // scattered into `[nchunks, chunk]` dense; indices within [0, chunk); out_len = n.
            let picks = chunk / 2 + 1;
            let vals: Vec<f32> = rng.vec(nchunks * picks);
            let idx: Vec<u32> = (0..nchunks * picks).map(|j| (j % chunk) as u32).collect();
            let idx_f32: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
            let (hvals, hidx) = (be.create(vals.clone()), be.create(idx_f32.clone()));
            let hscat = be.det_chunk_scatter(hvals, hidx, chunk, n).unwrap();
            let scat_ref = daemon_vhc_det::det_chunk_scatter(&vals, &idx, chunk, n).unwrap();
            assert_bits_eq("det_chunk_scatter", be.view(hscat), &scat_ref);

            let acc0 = rng.vec(n);
            let (hacc, hvals, hidx) = (
                be.create(acc0.clone()),
                be.create(vals.clone()),
                be.create(idx_f32.clone()),
            );
            be.det_chunk_scatter_add(hacc, hvals, hidx, chunk).unwrap();
            let mut acc_ref = acc0.clone();
            daemon_vhc_det::det_chunk_scatter_add(&mut acc_ref, &vals, &idx, chunk).unwrap();
            assert_bits_eq("det_chunk_scatter_add", be.view(hacc), &acc_ref);
        }
    }

    // dct2 / idct2 across the specced tile ladder — length a multiple of tile².
    for &tile in &[8usize, 16] {
        for nblocks in [1usize, 2, 3] {
            let n = tile * tile * nblocks;
            let x = rng.vec(n);
            let mut be = CpuBackend::new();
            let hx = be.create(x.clone());
            let hf = be.dct2(hx, tile).unwrap();
            assert_bits_eq(
                "dct2",
                be.view(hf),
                &daemon_vhc_det::dct2(&x, tile).unwrap(),
            );
            let hf2 = be.create(x.clone());
            let hi = be.idct2(hf2, tile).unwrap();
            assert_bits_eq(
                "idct2",
                be.view(hi),
                &daemon_vhc_det::idct2(&x, tile).unwrap(),
            );
        }
    }
}

/// The registry-coverage guard (twin of the §2.7 `TABI_IMPORTS`-coverage assertion): every det
/// acceleration op named in the ABI `DET_ACCEL_OPS` vocabulary is exercised by the sweeps above.
/// If a new det import lands, this list forces a matching conformance arm rather than letting the
/// gate silently miss it.
#[test]
fn det_accel_op_vocabulary_is_covered() {
    let covered = [
        "det_sum@1",
        "det_scale@1",
        "det_l2norm@1",
        "det_sign@1",
        "det_add@1",
        "det_sub@1",
        "det_mul@1",
        "det_axpy@1",
        "det_chunk_scatter@1",
        "det_chunk_scatter_add@1",
        "det_absmax_unpack@1",
        "absmax_pack@1",
        "topk_chunk@1",
        "dct2@1",
        "idct2@1",
    ];
    for op in DET_ACCEL_OPS {
        assert!(
            covered.contains(op),
            "det accel op `{op}` is in DET_ACCEL_OPS but not exercised by v2_det_conformance"
        );
    }
    assert_eq!(
        DET_ACCEL_OPS.len(),
        covered.len(),
        "DET_ACCEL_OPS and the covered set diverged — add a conformance arm for the new det op"
    );
}
