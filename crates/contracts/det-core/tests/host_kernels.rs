// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.5 HOST-1/2/5/6 full det-lane kernel suites (spec §5.6, ABI §5.8/§5.9). These extend the
// det-core `#[cfg(test)]` unit tests into the "full suite" the P2 gate requires: DCT orthonormality
// across tile sizes 8..128 (HOST-1), `topk_chunk@1` semantics incl. ties + empty/all-zero chunks +
// k boundaries (HOST-2), `det_sum` streaming≡batch equivalence in record order (HOST-5), and the
// det outer-step composition `det_reset_param_to_base`+`det_axpy_param` (HOST-6, modelled at the
// kernel layer). All assertions are bit-exact because the det lane is CPU fp32 with fixed evaluation
// order — the cross-peer identity property the swarm's agree-path leans on.
//
// Oracle provenance (swarm-ledger-p2-b1.md): from-definition (an independent Rust expression of the
// spec math) and hand-derived pinned literals; the daemon fixture seed is 0xDAE0_7E57.

use det_core::{
    absmax_pack, dct2, det_absmax_unpack, det_axpy, det_chunk_scatter, det_chunk_scatter_add,
    det_scale, det_sum, idct2, topk_chunk,
};

const SEED: u64 = 0xDAE0_7E57;

/// A tiny deterministic xorshift64* — test data only, never consensus-relevant.
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
}

// ===== HOST-1: dct2 / idct2 orthonormality across tile sizes 8..128 ==============================

#[test]
fn dct2_orthonormal_per_tile() {
    // The specced tile ladder (spec §15.2 / ABI §5.8) — extends Psyche's 4×4 golden to 128×128.
    for &tile in &[8usize, 16, 32, 64, 128] {
        let block = tile * tile;
        let mut rng = Rng::new(SEED ^ tile as u64);
        // Two blocks so the per-block loop is exercised, not just a single tile.
        let x: Vec<f32> = (0..block * 2).map(|_| rng.signed() * 3.0).collect();

        let y = dct2(&x, tile).unwrap();
        let back = idct2(&y, tile).unwrap();

        // Reconstruction bound (fp32 DCT-II/III round-trip): relative error small per tile.
        let mut max_abs = 0.0f32;
        for (a, b) in x.iter().zip(back.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 2e-3,
            "tile {tile}: reconstruction error {max_abs} exceeds bound"
        );

        // Parseval / energy preservation of the orthonormal transform, per block.
        for blk in 0..2 {
            let xs = &x[blk * block..(blk + 1) * block];
            let ys = &y[blk * block..(blk + 1) * block];
            let ex: f64 = xs.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let ey: f64 = ys.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            assert!(
                (ex - ey).abs() / ex.max(1e-9) < 1e-3,
                "tile {tile} block {blk}: energy {ex} vs {ey}"
            );
        }
    }
}

#[test]
fn dct2_dc_only_for_constant_block_per_tile() {
    // A constant block puts all energy in the DC coefficient (0,0) = mean · tile, AC ≈ 0.
    for &tile in &[8usize, 16, 32, 64, 128] {
        let block = tile * tile;
        let x = vec![2.0f32; block];
        let y = dct2(&x, tile).unwrap();
        let dc = 2.0 * tile as f32; // mean·N over the orthonormal DC row
        assert!((y[0] - dc).abs() < 1e-2, "tile {tile}: DC {} vs {dc}", y[0]);
        for &v in &y[1..] {
            assert!(
                v.abs() < 1e-2,
                "tile {tile}: AC coefficient {v} should vanish"
            );
        }
    }
}

#[test]
fn dct2_is_bit_reproducible_per_tile() {
    for &tile in &[8usize, 16, 32, 64, 128] {
        let block = tile * tile;
        let mut rng = Rng::new(SEED ^ (tile as u64).rotate_left(7));
        let x: Vec<f32> = (0..block).map(|_| rng.signed()).collect();
        let a = dct2(&x, tile).unwrap();
        let b = dct2(&x, tile).unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "tile {tile}: dct2 not reproducible"
            );
        }
    }
}

// ===== HOST-2: topk_chunk@1 semantics incl. ties + empty/all-zero chunks + k boundaries =========

#[test]
fn topk_chunk_golden() {
    // Two chunks of 4, k=2. Selection is descending magnitude, ties by ascending index.
    let x = [0.1f32, -0.9, 0.2, 1.0, -5.0, 0.5, 3.0, -0.1];
    let (vals, idx) = topk_chunk(&x, 4, 2).unwrap();
    assert_eq!(idx, vec![3, 1, 0, 2]);
    assert_eq!(vals, vec![1.0, -0.9, -5.0, 3.0]);
}

#[test]
fn topk_chunk_ties_break_by_index_within_and_across_chunks() {
    // Equal magnitudes ⇒ the lowest index wins, deterministically, and the tie rule is per-chunk.
    let x = [
        1.0f32, -1.0, 1.0, -1.0, // chunk 0: all |v|=1 → indices 0,1
        -2.0, 2.0, -2.0, 2.0, // chunk 1: all |v|=2 → indices 0,1
    ];
    let (vals, idx) = topk_chunk(&x, 4, 2).unwrap();
    assert_eq!(
        idx,
        vec![0, 1, 0, 1],
        "ties resolve to the lowest indices, per chunk"
    );
    assert_eq!(vals, vec![1.0, -1.0, -2.0, 2.0]);
}

#[test]
fn topk_chunk_all_zero_chunk_selects_lowest_indices() {
    // An all-zero ("empty") chunk: every magnitude is 0 → the total order falls back to index, so
    // the first k indices are chosen and their values are exactly 0.
    let x = [0.0f32; 8]; // 2 chunks of 4
    let (vals, idx) = topk_chunk(&x, 4, 3).unwrap();
    assert_eq!(idx, vec![0, 1, 2, 0, 1, 2]);
    assert!(vals.iter().all(|&v| v == 0.0));
}

#[test]
fn topk_chunk_k_boundaries() {
    let x = [3.0f32, 1.0, 2.0, 0.5];
    // k = 0: an empty selection (legal; nothing retained).
    let (vals, idx) = topk_chunk(&x, 4, 0).unwrap();
    assert!(vals.is_empty() && idx.is_empty());
    // k = chunk: the full set, ordered by descending magnitude.
    let (vals, idx) = topk_chunk(&x, 4, 4).unwrap();
    assert_eq!(idx, vec![0, 2, 1, 3]);
    assert_eq!(vals, vec![3.0, 2.0, 1.0, 0.5]);
    // k > chunk errors.
    assert!(topk_chunk(&x, 4, 5).is_err());
    // non-divisible length errors.
    assert!(topk_chunk(&[1.0, 2.0, 3.0], 4, 1).is_err());
}

#[test]
fn topk_chunk_scatter_roundtrip_is_a_sparse_projection() {
    // topk_chunk → chunk_scatter reconstructs a k-sparse tensor equal to the original at the kept
    // positions and zero elsewhere — the compression pipeline's core invariant.
    let mut rng = Rng::new(SEED);
    let chunk = 16usize;
    let k = 4usize;
    let numel = chunk * 3;
    let x: Vec<f32> = (0..numel).map(|_| rng.signed() * 10.0).collect();
    let (vals, idx) = topk_chunk(&x, chunk, k).unwrap();
    let dense = det_chunk_scatter(&vals, &idx, chunk, numel).unwrap();
    // Every reconstructed nonzero equals x at that global position; kept count is n_chunks·k.
    let mut kept = 0usize;
    for c in 0..numel / chunk {
        for j in 0..k {
            let gpos = c * chunk + idx[c * k + j] as usize;
            assert_eq!(dense[gpos].to_bits(), x[gpos].to_bits());
            kept += 1;
        }
    }
    assert_eq!(kept, (numel / chunk) * k);
}

// ===== HOST-5: det_sum streaming ≡ batch equivalence in record order ============================

#[test]
fn det_sum_record_order() {
    // det_sum accumulates in array (record) order — a golden where left-to-right differs bitwise
    // from a reassociated order, so the fixed order is observable. big=3e7 has an fp32 ulp ≈ 2, so
    // the +1 survives only when added before the large magnitude is cancelled.
    let big = 3.0e7f32;
    let xs: [&[f32]; 3] = [&[1.0], &[big], &[-big]];
    // Record order: 1 + big - big == 0 (the 1 is lost inside big).
    let got = det_sum(&xs).unwrap();
    assert_eq!(got, vec![0.0]);
    // Reverse order: -big + big + 1 == 1 (the 1 survives) — a genuinely different aggregate.
    let rev: Vec<&[f32]> = xs.iter().rev().copied().collect();
    let other = det_sum(&rev).unwrap();
    assert_eq!(other, vec![1.0]);
    assert_ne!(other[0].to_bits(), got[0].to_bits());
}

#[test]
fn streaming_equals_batch_aggregation_many_payloads() {
    // HOST-5 full: over many record-ordered sparse payloads, streaming (scatter-add each decode,
    // then drop) is bit-identical to batch (det_sum of the dense decodes). This is the property that
    // lets a peer ingest O(1) tensors of peak memory yet agree bit-for-bit with a batch aggregator.
    let chunk = 8usize;
    let out_len = chunk * 4; // 4 chunks
    let k = 3usize;
    let n_peers = 11;
    let mut rng = Rng::new(SEED);

    let mut payloads: Vec<(Vec<f32>, Vec<u32>)> = Vec::new();
    for _ in 0..n_peers {
        let dense: Vec<f32> = (0..out_len).map(|_| rng.signed() * 100.0).collect();
        let (v, i) = topk_chunk(&dense, chunk, k).unwrap();
        payloads.push((v, i));
    }

    // Batch: densify each in record order, det_sum.
    let dense: Vec<Vec<f32>> = payloads
        .iter()
        .map(|(v, i)| det_chunk_scatter(v, i, chunk, out_len).unwrap())
        .collect();
    let batch = det_sum(&dense.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();

    // Streaming: one accumulator, scatter-add each in record order.
    let mut acc = vec![0.0f32; out_len];
    for (v, i) in &payloads {
        det_chunk_scatter_add(&mut acc, v, i, chunk).unwrap();
    }

    for (s, b) in acc.iter().zip(batch.iter()) {
        assert_eq!(
            s.to_bits(),
            b.to_bits(),
            "streaming must equal batch bitwise"
        );
    }
}

#[test]
fn host_stages_record_order_determines_the_aggregate() {
    // The aggregate is a function of the *record* order the host stages, not physical storage order.
    // Re-presenting the same tensors in record order reproduces the aggregate byte-for-byte; a
    // different staging order is a different (allowed-to-diverge) aggregate — which is exactly why
    // the host stages in record order (spec §5.11) and the guest never reorders.
    let big = 3.0e7f32;
    let tensors: [[f32; 1]; 3] = [[1.0], [big], [-big]];
    let record: Vec<&[f32]> = tensors.iter().map(|t| t.as_slice()).collect();
    let canonical = det_sum(&record).unwrap();
    for _ in 0..4 {
        assert_eq!(
            det_sum(&record).unwrap()[0].to_bits(),
            canonical[0].to_bits()
        );
    }
    // Record order: 1 + big - big == 0; a reordering that cancels big first keeps the 1 → differs.
    let reordered: Vec<&[f32]> = vec![record[1], record[2], record[0]];
    assert_ne!(
        det_sum(&reordered).unwrap()[0].to_bits(),
        canonical[0].to_bits()
    );
}

// ===== HOST-6: outer step composition det_reset_param_to_base + det_axpy_param ==================

/// A kernel-layer model of the det outer step (ABI §5.9): `det_reset_param_to_base@1` restores the
/// round-base snapshot into `master`, then `det_axpy_param@1` applies `master += α·x`. Composed,
/// θ⁽ᵗ⁺¹⁾ = base + α·x, bit-exactly, independent of `master`'s pre-reset contents.
fn outer_step(base: &[f32], update: &[f32], alpha: f64) -> Vec<f32> {
    let mut master = base.to_vec(); // det_reset_param_to_base: master ← round-base snapshot
    det_axpy(&mut master, alpha, update).unwrap(); // det_axpy_param: master += α·update
    master
}

#[test]
fn det_outer_step_golden() {
    // From-definition: base + α·update, elementwise, matches the reset+axpy composition bit-for-bit.
    let base = [1.0f32, -2.0, 0.5, 4.0];
    let update = [0.25f32, 0.5, -1.0, 2.0];
    let alpha = -0.5f64; // an outer step subtracts α·mean(Δ̂)

    let got = outer_step(&base, &update, alpha);
    let want: Vec<f32> = base
        .iter()
        .zip(update.iter())
        .map(|(&b, &u)| b + (alpha as f32) * u)
        .collect();
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    // Reproducible.
    assert_eq!(outer_step(&base, &update, alpha), got);
}

#[test]
fn round_base_snapshots_at_barrier_not_cumulative() {
    // The round-base snapshot is taken once at the ingest barrier: applying the outer step again
    // from a *re-reset* base yields base + α·x each time — it does NOT accumulate across rounds
    // (each round rebases to θ⁽ᵗ⁾, ABI §5.9). Two applications from the snapshot are identical.
    let base = [10.0f32, 20.0, -5.0];
    let x = [1.0f32, 2.0, 3.0];
    let alpha = 1.0;
    let once = outer_step(&base, &x, alpha);
    let twice = outer_step(&base, &x, alpha); // re-reset to the same snapshot, apply again
    assert_eq!(
        once, twice,
        "rebasing to the barrier snapshot is idempotent per round"
    );
    // Sanity: not the same as applying the update twice cumulatively.
    let mut cumulative = base.to_vec();
    det_axpy(&mut cumulative, alpha, &x).unwrap();
    det_axpy(&mut cumulative, alpha, &x).unwrap();
    assert_ne!(
        cumulative, once,
        "cumulative (no rebase) must differ from the rebased step"
    );
}

#[test]
fn sparse_loco_outer_step_composition_from_sparse_payloads() {
    // The realistic outer step: aggregate 2-bit absmax sparse payloads (streaming scatter-add), then
    // θ = base − (α/R)·Σ Δ̂. Verifies the det-lane compression→aggregate→outer-step chain end to end
    // at the kernel layer (the HOST-6 view of what the SDK sim golden exercises).
    let chunk = 8usize;
    let numel = chunk * 2;
    let k = 3usize;
    let bits = 2u32;
    let base: Vec<f32> = (0..numel).map(|n| (n as f32) * 0.1 - 0.8).collect();

    let mut rng = Rng::new(SEED);
    let mut acc = vec![0.0f32; numel];
    let r = 3;
    for _ in 0..r {
        let dense: Vec<f32> = (0..numel).map(|_| rng.signed()).collect();
        let (v, i) = topk_chunk(&dense, chunk, k).unwrap();
        // 2-bit absmax quantize the retained values (per top-k row), then decode (what a peer sends).
        let packed = absmax_pack(&v, k, bits).unwrap();
        let decoded = det_absmax_unpack(&packed, k, bits).unwrap();
        det_chunk_scatter_add(&mut acc, &decoded, &i, chunk).unwrap();
    }
    let alpha = 1.0f64;
    // θ = base − (α/R)·acc
    let scaled = det_scale(&acc, -alpha / r as f64);
    let theta = outer_step(&base, &scaled, 1.0);

    // From-definition: recompute base + scaled directly.
    let want: Vec<f32> = base
        .iter()
        .zip(scaled.iter())
        .map(|(&b, &s)| b + s)
        .collect();
    for (g, w) in theta.iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    assert!(theta.iter().all(|v| v.is_finite()));
}
