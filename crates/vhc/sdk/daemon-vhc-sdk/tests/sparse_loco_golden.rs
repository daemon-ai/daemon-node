// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.4 SDK-1..5 — the `sparse_loco` flagship golden/conformance suite (plus the `demo`/`diloco`
// golden completion). This is the P2 golden completion of the profile machinery P1 shipped: the
// DCT/chunk/top-k pipeline, error-feedback accumulation across rounds, quantization legality,
// compression-ratio invariants, and cross-round determinism — driven through the `sim` backend
// (ABI §10.4), whose det lane delegates to the shared `det-core` kernels so "sim ≡ host".
//
// Oracle provenance (swarm-ledger-p2-b1.md "Golden oracle provenance"):
//  - Pinned-literal goldens are recorded from the bit-reproducible `sim` reference at the daemon
//    seed 0xDAE0_7E57 (matches det-core's SEED). The det lane is CPU fp32 with fixed evaluation
//    order (spec §5.6), so these literals are stable across targets/vendors by construction; a drift
//    is a deliberate break in the profile math.
//  - From-definition goldens recompute the expected value by an independent expression of the spec
//    math (a direct `det-core` call path) and assert bit-for-bit against the profile pipeline.
#![cfg(feature = "sim")]

use daemon_train_sdk::profiles::{Demo, DemoCfg, DiLoCo, DiLoCoCfg, SparseLoco, SparseLocoCfg};
use daemon_train_sdk::sim;
use daemon_train_sdk::{Dtype, Init, Param, Persistent, Tensor, UpdatesView};

const SEED: u64 = 0xDAE0_7E57;

/// A one-weight model with AdamW inner state — a real backward so θ moves off θ⁽ᵗ⁾ each round
/// (mirrors `tests/profiles.rs::Model`, the P1 driver).
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

    fn train(&mut self, h: u32) {
        let numel: u32 = self.dims.iter().product();
        for s in 0..h {
            let target = Tensor::full(&self.dims, Dtype::F32, 0.5);
            let diff = self.w[0].tensor().sub(&target);
            let sq = diff.mul(&diff);
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

/// Assert `got` matches the pinned golden `want`, or (when `want` is empty) print a copy-pasteable
/// literal so the golden can be recorded once. The det lane is bit-reproducible, so a recorded
/// literal is stable; see the module oracle-provenance note.
fn golden_bits(label: &str, got: &[f32], want: &[u32]) {
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    if want.is_empty() {
        eprintln!("GOLDEN[{label}] = {got_bits:?};");
        panic!("golden {label} not yet pinned (see stderr for the recorded literal)");
    }
    assert_eq!(
        got_bits, want,
        "golden {label} drifted from the pinned oracle"
    );
    assert!(got.iter().all(|v| v.is_finite()));
}

// ===== SDK-1: sparse_loco full round golden ======================================================

/// One full `sparse_loco` round: H AdamW inner steps → Δ=θ⁽ᵗ⁾−θᵣ → acc=β·e+Δ → chunk-top-k →
/// 2-bit Q → residual e; two self-inclusive peers stage, then the det-lane ingest rebases + applies
/// the outer step. The post-round canonical master is pinned.
fn sparse_loco_round(cfg: SparseLocoCfg, dims: &[u32]) -> Vec<f32> {
    sim::reset(SEED);
    let mut model = Model::build(dims);
    let mut sl = SparseLoco::new(cfg.clone(), &model.w);
    model.train(cfg.h);
    let u1 = sl.make_update(&model.w);
    sim::stage(&u1);
    let u2 = sl.make_update(&model.w);
    sim::stage(&u2);
    sl.ingest(&model.w, &UpdatesView::with_count(2));
    sim::snapshot_round_base();
    w_master()
}

#[test]
fn sdk1_sparse_loco_round_golden() {
    let dims = [64u32]; // numel 64, chunk 16 ⇒ 4 chunks, k=4
    let cfg = SparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let got = sparse_loco_round(cfg, &dims);
    // GOLDEN: recorded from the sim reference @ seed 0xDAE07E57 (see module note).
    const WANT: &[u32] = &[
        1022866996, 1035905016, 1049099383, 1046773475, 1032789417, 1042697249, 1047616210,
        1037443080, 1034583887, 1022732346, 1048585438, 1045264907, 3189781144, 1003467168,
        1024065251, 1022694712, 1023940144, 3189389204, 1024670989, 1017043157, 1037873341,
        1045168788, 1000092354, 1044458016, 1020865956, 1043555106, 1046660398, 1039907904,
        1023845362, 1041321735, 1041618040, 1019425688, 1048488971, 1043485881, 1039702757,
        1026832117, 3181922874, 1026173014, 1019562548, 1011522219, 1039729050, 1036564907,
        1016671272, 1025777068, 1049534257, 1045039832, 1024173708, 1048626832, 1045448963,
        1027019828, 1049390157, 1042388084, 1038197015, 1016931240, 1032671187, 1038918388,
        1029413506, 1036349793, 1038537856, 1049399815, 1050967702, 1025913804, 1024394021,
        1048850918,
    ];
    golden_bits("sdk1_round", &got, WANT);
}

#[test]
fn sdk1_round_is_cross_run_bit_identical() {
    let dims = [64u32];
    let cfg = SparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let a = sparse_loco_round(cfg.clone(), &dims);
    let b = sparse_loco_round(cfg, &dims);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sparse_loco round must be bit-identical across runs"
        );
    }
}

// ===== SDK-1: error-feedback accumulation across rounds ==========================================

/// The error-feedback residual `e` carries the un-transmitted mass from round to round: a second
/// round starting from a non-zero `e` reaches a different (still finite, still reproducible) state
/// than the first, and the two-round trajectory is bit-reproducible.
#[test]
fn sdk1_error_feedback_accumulates_across_rounds() {
    let dims = [64u32];
    let cfg = SparseLocoCfg {
        h: 2,
        chunk: 16,
        topk: 2, // aggressive sparsity ⇒ meaningful residual carried in e
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let two_rounds = || -> (Vec<f32>, Vec<f32>) {
        sim::reset(SEED);
        let mut model = Model::build(&dims);
        let mut sl = SparseLoco::new(cfg.clone(), &model.w);
        // round 0
        model.train(cfg.h);
        let u1 = sl.make_update(&model.w);
        sim::stage(&u1);
        let u2 = sl.make_update(&model.w);
        sim::stage(&u2);
        sl.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        let after_r0 = w_master();
        sim::clear_staged();
        // round 1 (e is now non-zero, carrying the round-0 residual)
        model.train(cfg.h);
        let u1 = sl.make_update(&model.w);
        sim::stage(&u1);
        let u2 = sl.make_update(&model.w);
        sim::stage(&u2);
        sl.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        let after_r1 = w_master();
        (after_r0, after_r1)
    };
    let (r0a, r1a) = two_rounds();
    let (r0b, r1b) = two_rounds();
    // Bit-reproducible across the whole trajectory.
    for (x, y) in r0a.iter().zip(r0b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    for (x, y) in r1a.iter().zip(r1b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    // The two rounds are distinct states (e carried mass forward, not a fixed point).
    assert_ne!(
        r0a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        r1a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "round 1 must move past round 0 (error feedback carried residual)"
    );
    assert!(r1a.iter().all(|v| v.is_finite()));
}

// ===== SDK-2: 2-bit quantization legality + compression ratio ===================================

/// The `sparse_loco` payload is a per-top-k-row 2-bit absmax codebook: for `k`-wide rows the packed
/// section is `n_chunks · (2 + ceil(k·bits/8))` bytes (ABI §6.6 layout), which must be far smaller
/// than the dense fp32 alternative. Also asserts the index section is present (values + indices).
#[test]
fn sdk2_quantization_legality_and_compression_ratio() {
    sim::reset(SEED);
    let dims = [256u32];
    let (chunk, k, bits) = (16u32, 2u32, 2u32);
    let mut model = Model::build(&dims);
    let cfg = SparseLocoCfg {
        h: 2,
        chunk,
        topk: k,
        bits,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let mut sl = SparseLoco::new(cfg, &model.w);
    model.train(2);
    let ub = sl.make_update(&model.w);

    let numel = 256u32;
    let n_chunks = numel / chunk; // 16 rows
                                  // Each packed absmax row: 2-byte f16 codebook + ceil(k·bits/8) code bytes, over n_chunks rows.
    let code_bytes = (k * bits).div_ceil(8);
    let expected_packed = (n_chunks * (2 + code_bytes)) as usize;
    let packed = sim::section_len(&ub, 0);
    assert_eq!(
        packed, expected_packed,
        "2-bit absmax layout: {n_chunks} rows · (2 + {code_bytes}) B"
    );
    // Index section (one u32 index per retained value): present + sized n_chunks·k.
    let idx = sim::section_len(&ub, 1);
    assert_eq!(idx, (n_chunks * k) as usize, "one index per retained value");

    let dense = (numel * 4) as usize;
    assert!(
        packed * 8 < dense,
        "2-bit sparse payload {packed} B must be >8x smaller than dense {dense} B"
    );
}

// ===== SDK-3: chunk-top-k index codec fits within 12 bits per value =============================

/// TDD SDK-3: indices are `< chunk`, so within the paper's 4096 chunk each retained index fits in
/// ≤12 bits (`2^12 = 4096`). Verified from-definition against the shared `det-core::topk_chunk`
/// (the exact kernel `sparse_loco` composes) over a deterministic sweep of chunk contents.
#[test]
fn sdk3_topk_indices_fit_within_12_bits() {
    let chunk = 4096usize;
    let k = 64usize;
    // A deterministic pseudo-random tensor of two 4096-chunks.
    let mut state = SEED;
    let numel = chunk * 2;
    let x: Vec<f32> = (0..numel)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let (_vals, idx) = det_core::topk_chunk(&x, chunk, k).unwrap();
    assert_eq!(idx.len(), (numel / chunk) * k);
    for &i in &idx {
        assert!(i < chunk as u32, "index must be within the chunk");
        assert!(i < (1 << 12), "index must fit in 12 bits (chunk = 4096)");
    }
}

// ===== SDK-4: median-norm clip golden ===========================================================

/// TDD SDK-4: with a dominant-norm peer present, the median-norm clip changes the aggregate vs. the
/// unclipped run. Pinned literal captures the *clipped* post-round master (the hardened default).
#[test]
fn sdk4_median_norm_clip_golden() {
    let dims = [32u32];
    let run = |clip: bool| -> Vec<f32> {
        sim::reset(0x1234);
        let mut model = Model::build(&dims);
        let cfg = SparseLocoCfg {
            h: 2,
            chunk: 8,
            topk: 4,
            bits: 2,
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
    assert_ne!(
        clipped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        unclipped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "median-norm clip must alter the aggregate when one peer dominates"
    );
    // GOLDEN: the clipped post-round master @ seed 0x1234.
    const WANT_CLIPPED: &[u32] = &[
        1048399937, 1041325501, 1034513755, 1047491600, 1050745946, 1043743709, 1049443532,
        1034007017, 1051962560, 1050680617, 1042843334, 1036992154, 1040400891, 1047303413,
        1042268554, 1039713937, 1030559002, 1043517125, 1048682410, 1049387044, 1036787643,
        1037653402, 1048716156, 1032137136, 1050615722, 1047449897, 1025146051, 1048731680,
        1031314266, 1049655924, 3170958056, 1043224015,
    ];
    golden_bits("sdk4_clipped", &clipped, WANT_CLIPPED);
}

// ===== SDK-5: outer step golden (θ − α·mean(Δ̂)) =================================================

/// TDD SDK-5: with clip off, the ingest is exactly the outer step θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂.
/// Pinned literal captures the post-round master at α=1 vs. a lowered late-training α (open q.2), and
/// asserts the two differ (α is load-bearing on the step magnitude).
#[test]
fn sdk5_sparse_loco_outer_step_golden() {
    let dims = [48u32];
    let run = |alpha: f64| -> Vec<f32> {
        sim::reset(SEED);
        let mut model = Model::build(&dims);
        let cfg = SparseLocoCfg {
            h: 3,
            chunk: 12,
            topk: 4,
            bits: 2,
            outer_alpha: alpha,
            clip: false,
            ..SparseLocoCfg::default()
        };
        let mut sl = SparseLoco::new(cfg, &model.w);
        model.train(3);
        let u1 = sl.make_update(&model.w);
        sim::stage(&u1);
        let u2 = sl.make_update(&model.w);
        sim::stage(&u2);
        sl.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        w_master()
    };
    let a_full = run(1.0);
    let a_late = run(0.65);
    assert_ne!(
        a_full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        a_late.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "outer α scales the step; α=1 vs α=0.65 must differ"
    );
    // GOLDEN: the α=1 post-round master.
    const WANT_FULL: &[u32] = &[
        1022539316, 1035905016, 1049074807, 1034501574, 1052198762, 1042697249, 1047616210,
        1037443080, 1052647380, 1050860164, 1038144377, 1031170092, 3189781144, 1003467168,
        1024065251, 1022694712, 1023940144, 3189389204, 1024670989, 1050537357, 1037873341,
        1045168788, 1050046667, 1044458016, 1020538276, 1024265354, 1046644014, 1039907904,
        1023681522, 1041321735, 1041618040, 1019098008, 1048472587, 1043485881, 1053951673,
        1051354847, 3181922874, 1026173014, 1019562548, 1050301653, 1039729050, 1036564907,
        1016671272, 1025777068, 1049534257, 1045039832, 1024173708, 1048626832,
    ];
    golden_bits("sdk5_outer_alpha1", &a_full, WANT_FULL);
}

// ===== demo / diloco golden completion ==========================================================

/// `demo` (§5.3.3) per-step round: DCT energy extraction + top-k coefficients (native) → det-lane
/// coefficient sum → inverse-DCT → sign-SGD + decoupled decay. Pinned post-round master.
#[test]
fn demo_per_step_round_golden() {
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
        model.train(1);
        let u1 = demo.make_update(&model.w);
        sim::stage(&u1);
        let u2 = demo.make_update(&model.w);
        sim::stage(&u2);
        demo.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        w_master()
    };
    let got = run();
    let again = run();
    for (x, y) in got.iter().zip(again.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "demo must be bit-reproducible");
    }
    const WANT: &[u32] = &[
        1041233614, 3160447752, 3159305194, 3157755456, 3197556803, 3162992298, 3186678993,
        3163121865, 1024933292, 1042354737, 3185639862, 1016324450, 3170773243, 3185332349,
        3180567922, 3169461521, 3156531778, 3176117672, 1022718796, 1047154333, 1036782663,
        1033076633, 3188449712, 1009615390, 1042223381, 1036841973, 1040890333, 1025350400,
        1036628266, 1028204524, 3190675651, 1026568202, 1040212552, 3181437482, 1043522942,
        1032634148, 3191838637, 1043816554, 1044660189, 1026650692, 1044370633, 1032411733,
        3185569236, 1042967702, 1042608690, 1031362148, 1036493399, 1022135713, 1039526524,
        3167785143, 3190970421, 1037517045, 3179632063, 1038207926, 1031287552, 3178213456,
        1043895317, 1017040106, 3185731352, 994257008, 3193362007, 3181076024, 3171432182,
        3176529634,
    ];
    golden_bits("demo_round", &got, WANT);
}

/// `diloco` (§5.3.2) outer Nesterov step over a replicated momentum: dense pseudo-gradient →
/// aggregate → m ← μ·m + g → step = g + μ·m (Nesterov) → θ − outer_lr·step. Pinned master, and the
/// plain-heavy-ball vs. Nesterov ablation differs.
#[test]
fn diloco_outer_nesterov_golden() {
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
    assert_ne!(
        nes.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "Nesterov vs plain heavy-ball must differ"
    );
    const WANT_NESTEROV: &[u32] = &[
        1053950547, 1052981238, 1053352725, 1055679126, 1051100703, 1052352578, 1051630824,
        1052295540, 1055795133, 1055222456, 1056767783, 1045158268, 1054637190, 1057391135,
        1051720249, 1047298411,
    ];
    golden_bits("diloco_nesterov", &nes, WANT_NESTEROV);
}
