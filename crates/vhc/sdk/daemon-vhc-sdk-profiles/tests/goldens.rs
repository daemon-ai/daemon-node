// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The C3a golden suite: the re-expressed profiles (this crate) reproduce the CURRENT SDK profile
// implementation (`daemon-vhc-sdk::profiles` over the `sim` backend) **bit-for-bit**, on the same
// inputs, at the same pinned literals.
//
// ## Oracle provenance / regeneration
//
// Two oracles per scenario, both from the current implementation:
//  - **Live A/B**: the old profile runs in the same test over the identical θ/round-base
//    trajectory (the sim `Model`, seeds below); its post-round master is read back and asserted
//    bit-equal to this crate's — the old code IS the oracle, executed fresh.
//  - **Pinned literals**: the `WANT` bit-pattern arrays are copied verbatim from
//    `crates/vhc/sdk/daemon-vhc-sdk/tests/sparse_loco_golden.rs` (recorded from the
//    bit-reproducible sim reference at the documented seeds: 0xDAE0_7E57 / 0x1234 / 0x5150 /
//    0xABCD). Regeneration command (only when the profile math deliberately changes):
//    empty a `WANT` array and run
//        cargo test -p daemon-vhc-sdk --features sim --test sparse_loco_golden -- --nocapture
//    then copy the `GOLDEN[...]` literal from stderr into BOTH suites (they pin one math).
//
// The det lane is CPU fp32 with fixed evaluation order (both implementations delegate every
// kernel to `daemon-vhc-det`), so bit-equality across the two implementations is the designed
// property, not luck — a mismatch is a porting bug, never tolerance.

use daemon_vhc_sdk::profiles::{
    Demo as OldDemo, DemoCfg as OldDemoCfg, DiLoCo as OldDiLoCo, DiLoCoCfg as OldDiLoCoCfg,
    SparseLoco as OldSparseLoco, SparseLocoCfg as OldSparseLocoCfg,
};
use daemon_vhc_sdk::{sim, Dtype, Init, Param, Persistent, Tensor, UpdatesView};

use daemon_vhc_sdk_profiles::{
    decode_payload, encode_payload, Demo, DemoCfg, DiLoCo, DiLoCoCfg, IngestParam, ParamView,
    Section, SparseLoco, SparseLocoCfg,
};

const SEED: u64 = 0xDAE0_7E57;

/// The sim one-weight model with AdamW inner state — identical to the current golden suite's
/// `Model`, so the θ trajectories (hence the pinned literals) are the same.
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
            daemon_vhc_sdk::zero_grads();
        }
    }
}

fn w_master() -> Vec<f32> {
    sim::param_master("w").unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bit_equal(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{ctx}[{i}]: {g} vs {w} — the port must be bit-exact"
        );
    }
}

// ===== sparse_loco: the SDK-1 round golden, A/B + pinned ========================================

/// The pinned SDK-1 literal (copied from `sparse_loco_golden.rs::sdk1_sparse_loco_round_golden`,
/// recorded from the sim reference @ seed 0xDAE0_7E57 — see the module provenance note).
const SDK1_WANT: &[u32] = &[
    1022866996, 1035905016, 1049099383, 1046773475, 1032789417, 1042697249, 1047616210, 1037443080,
    1034583887, 1022732346, 1048585438, 1045264907, 3189781144, 1003467168, 1024065251, 1022694712,
    1023940144, 3189389204, 1024670989, 1017043157, 1037873341, 1045168788, 1000092354, 1044458016,
    1020865956, 1043555106, 1046660398, 1039907904, 1023845362, 1041321735, 1041618040, 1019425688,
    1048488971, 1043485881, 1039702757, 1026832117, 3181922874, 1026173014, 1019562548, 1011522219,
    1039729050, 1036564907, 1016671272, 1025777068, 1049534257, 1045039832, 1024173708, 1048626832,
    1045448963, 1027019828, 1049390157, 1042388084, 1038197015, 1016931240, 1032671187, 1038918388,
    1029413506, 1036349793, 1038537856, 1049399815, 1050967702, 1025913804, 1024394021, 1048850918,
];

#[test]
fn sparse_loco_round_matches_current_sdk_bit_for_bit() {
    let dims = [64u32];
    let numel = 64usize;
    let old_cfg = OldSparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        bits: 2,
        clip: false,
        ..OldSparseLocoCfg::default()
    };
    let new_cfg = SparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };

    // The shared θ trajectory (the sim Model at the documented seed).
    sim::reset(SEED);
    let mut model = Model::build(&dims);
    let base0 = w_master(); // round_base == master at registration
    let mut old = OldSparseLoco::new(old_cfg, &model.w);
    let mut new = SparseLoco::new(new_cfg, &[numel]);
    model.train(3);
    let theta = w_master(); // trained θ (adamw writes the master)

    // OLD (the live oracle): two self-inclusive peers → det ingest → post-round master.
    let u1 = old.make_update(&model.w);
    sim::stage(&u1);
    let u2 = old.make_update(&model.w);
    sim::stage(&u2);
    old.ingest(&model.w, &UpdatesView::with_count(2));
    sim::snapshot_round_base();
    let old_master = w_master();

    // NEW (this crate): identical inputs as f32 slices.
    let view = ParamView {
        theta: &theta,
        round_base: &base0,
    };
    let p1 = new.make_update(&[view]);
    let p2 = new.make_update(&[view]);
    let mut master = vec![0.0f32; numel];
    new.ingest(
        &mut [IngestParam {
            master: &mut master,
            round_base: &base0,
        }],
        &[p1, p2],
    )
    .expect("ingest");

    assert_bit_equal(&master, &old_master, "sparse_loco A/B");
    assert_eq!(bits(&master), SDK1_WANT, "sparse_loco pinned golden");
}

// ===== sparse_loco: error feedback across rounds (A/B) ==========================================

#[test]
fn sparse_loco_error_feedback_carries_across_rounds_like_the_current_sdk() {
    let dims = [64u32];
    let numel = 64usize;
    let old_cfg = OldSparseLocoCfg {
        h: 2,
        chunk: 16,
        topk: 2, // aggressive sparsity ⇒ meaningful residual carried in ef
        bits: 2,
        clip: false,
        ..OldSparseLocoCfg::default()
    };
    let new_cfg = SparseLocoCfg {
        h: 2,
        chunk: 16,
        topk: 2,
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };

    sim::reset(SEED);
    let mut model = Model::build(&dims);
    let mut base = w_master();
    let mut old = OldSparseLoco::new(old_cfg, &model.w);
    let mut new = SparseLoco::new(new_cfg, &[numel]);
    let mut new_master = vec![0.0f32; numel];

    for round in 0..2u32 {
        model.train(2);
        let theta = w_master();

        let u1 = old.make_update(&model.w);
        sim::stage(&u1);
        let u2 = old.make_update(&model.w);
        sim::stage(&u2);
        old.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        let old_master = w_master();
        sim::clear_staged();

        let view = ParamView {
            theta: &theta,
            round_base: &base,
        };
        let p1 = new.make_update(&[view]);
        let p2 = new.make_update(&[view]);
        new.ingest(
            &mut [IngestParam {
                master: &mut new_master,
                round_base: &base,
            }],
            &[p1, p2],
        )
        .expect("ingest");

        assert_bit_equal(&new_master, &old_master, &format!("ef round {round}"));
        base = new_master.clone(); // the post-ingest master is the next round base
    }
}

// ===== sparse_loco: median-norm clip golden (A/B + pinned) ======================================

/// Copied from `sparse_loco_golden.rs::sdk4_median_norm_clip_golden` (@ seed 0x1234).
const SDK4_WANT_CLIPPED: &[u32] = &[
    1048399937, 1041325501, 1034513755, 1047491600, 1050745946, 1043743709, 1049443532, 1034007017,
    1051962560, 1050680617, 1042843334, 1036992154, 1040400891, 1047303413, 1042268554, 1039713937,
    1030559002, 1043517125, 1048682410, 1049387044, 1036787643, 1037653402, 1048716156, 1032137136,
    1050615722, 1047449897, 1025146051, 1048731680, 1031314266, 1049655924, 3170958056, 1043224015,
];

#[test]
fn sparse_loco_median_clip_matches_current_sdk_bit_for_bit() {
    let dims = [32u32];
    let numel = 32usize;

    sim::reset(0x1234);
    let mut model = Model::build(&dims);
    let base0 = w_master();
    let mut old = OldSparseLoco::new(
        OldSparseLocoCfg {
            h: 2,
            chunk: 8,
            topk: 4,
            bits: 2,
            clip: true,
            ..OldSparseLocoCfg::default()
        },
        &model.w,
    );
    let mut new = SparseLoco::new(
        SparseLocoCfg {
            h: 2,
            chunk: 8,
            topk: 4,
            bits: 2,
            clip: true,
            ..SparseLocoCfg::default()
        },
        &[numel],
    );

    // Peer 1 trains 2 steps; peer 2 trains 8 more (a dominant-norm Δ) — the clip must bite.
    model.train(2);
    let theta_a = w_master();
    let u1 = old.make_update(&model.w);
    sim::stage(&u1);
    model.train(8);
    let theta_b = w_master();
    let u2 = old.make_update(&model.w);
    sim::stage(&u2);
    old.ingest(&model.w, &UpdatesView::with_count(2));
    sim::snapshot_round_base();
    let old_master = w_master();

    let p1 = new.make_update(&[ParamView {
        theta: &theta_a,
        round_base: &base0,
    }]);
    let p2 = new.make_update(&[ParamView {
        theta: &theta_b,
        round_base: &base0,
    }]);
    let mut master = vec![0.0f32; numel];
    new.ingest(
        &mut [IngestParam {
            master: &mut master,
            round_base: &base0,
        }],
        &[p1, p2],
    )
    .expect("ingest");

    assert_bit_equal(&master, &old_master, "clip A/B");
    assert_eq!(bits(&master), SDK4_WANT_CLIPPED, "clip pinned golden");
}

// ===== diloco: outer Nesterov golden (A/B + pinned) =============================================

/// Copied from `sparse_loco_golden.rs::diloco_outer_nesterov_golden` (@ seed 0xABCD).
const DILOCO_WANT_NESTEROV: &[u32] = &[
    1053950547, 1052981238, 1053352725, 1055679126, 1051100703, 1052352578, 1051630824, 1052295540,
    1055795133, 1055222456, 1056767783, 1045158268, 1054637190, 1057391135, 1051720249, 1047298411,
];

#[test]
fn diloco_nesterov_matches_current_sdk_bit_for_bit() {
    for nesterov in [true, false] {
        let dims = [16u32];
        let numel = 16usize;
        sim::reset(0xABCD);
        let mut model = Model::build(&dims);
        let base0 = w_master();
        let mut old = OldDiLoCo::new(
            OldDiLoCoCfg {
                h: 3,
                nesterov,
                ..OldDiLoCoCfg::default()
            },
            &model.w,
        );
        let mut new = DiLoCo::new(
            DiLoCoCfg {
                h: 3,
                nesterov,
                ..DiLoCoCfg::default()
            },
            &[numel],
        );
        model.train(3);
        let theta = w_master();

        let u1 = old.make_update(&model.w);
        sim::stage(&u1);
        let u2 = old.make_update(&model.w);
        sim::stage(&u2);
        old.ingest(&model.w, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
        let old_master = w_master();

        let view = ParamView {
            theta: &theta,
            round_base: &base0,
        };
        let p1 = new.make_update(&[view]);
        let p2 = new.make_update(&[view]);
        let mut master = vec![0.0f32; numel];
        new.ingest(
            &mut [IngestParam {
                master: &mut master,
                round_base: &base0,
            }],
            &[p1, p2],
        )
        .expect("ingest");

        assert_bit_equal(
            &master,
            &old_master,
            &format!("diloco A/B nesterov={nesterov}"),
        );
        if nesterov {
            assert_eq!(bits(&master), DILOCO_WANT_NESTEROV, "diloco pinned golden");
        }
        // The replicated momentum (digest-covered det state) exists and moved.
        assert!(new.replicated_state()[0].iter().any(|&v| v != 0.0));
    }
}

// ===== demo: per-step round golden (A/B + pinned) ===============================================

/// Copied from `sparse_loco_golden.rs::demo_per_step_round_golden` (@ seed 0x5150).
const DEMO_WANT: &[u32] = &[
    1041233614, 3160447752, 3159305194, 3157755456, 3197556803, 3162992298, 3186678993, 3163121865,
    1024933292, 1042354737, 3185639862, 1016324450, 3170773243, 3185332349, 3180567922, 3169461521,
    3156531778, 3176117672, 1022718796, 1047154333, 1036782663, 1033076633, 3188449712, 1009615390,
    1042223381, 1036841973, 1040890333, 1025350400, 1036628266, 1028204524, 3190675651, 1026568202,
    1040212552, 3181437482, 1043522942, 1032634148, 3191838637, 1043816554, 1044660189, 1026650692,
    1044370633, 1032411733, 3185569236, 1042967702, 1042608690, 1031362148, 1036493399, 1022135713,
    1039526524, 3167785143, 3190970421, 1037517045, 3179632063, 1038207926, 1031287552, 3178213456,
    1043895317, 1017040106, 3185731352, 994257008, 3193362007, 3181076024, 3171432182, 3176529634,
];

#[test]
fn demo_per_step_round_matches_current_sdk_bit_for_bit() {
    let dims = [64u32]; // one 8×8 DCT tile
    let numel = 64usize;
    sim::reset(0x5150);
    let mut model = Model::build(&dims);
    let base0 = w_master();
    let mut old = OldDemo::new(
        OldDemoCfg {
            tile: 8,
            topk: 8,
            ..OldDemoCfg::default()
        },
        &model.w,
    );
    let mut new = Demo::new(
        DemoCfg {
            tile: 8,
            topk: 8,
            ..DemoCfg::default()
        },
        &[numel],
    );
    model.train(1);
    let theta = w_master();

    let u1 = old.make_update(&model.w);
    sim::stage(&u1);
    let u2 = old.make_update(&model.w);
    sim::stage(&u2);
    old.ingest(&model.w, &UpdatesView::with_count(2));
    sim::snapshot_round_base();
    let old_master = w_master();

    let view = ParamView {
        theta: &theta,
        round_base: &base0,
    };
    let p1 = new.make_update(&[view]);
    let p2 = new.make_update(&[view]);
    let mut master = vec![0.0f32; numel];
    new.ingest(
        &mut [IngestParam {
            master: &mut master,
            round_base: &base0,
        }],
        &[p1, p2],
    )
    .expect("ingest");

    assert_bit_equal(&master, &old_master, "demo A/B");
    assert_eq!(bits(&master), DEMO_WANT, "demo pinned golden");
}

// ===== the payload wire =========================================================================

/// The `Section` CBOR is byte-identical to the v1 host container wire (`SectionWire`): an
/// externally-tagged enum — `{"Bytes": bytes}` / `{"Tensor": {"data": […], "shape": […]}}` — so
/// payloads interoperate across the v1/C3 boundary without a re-encode.
#[test]
fn section_wire_matches_the_v1_container_encoding() {
    use ciborium::value::Value;

    let sections = vec![
        Section::Tensor {
            data: vec![1.5f32, -2.0],
            shape: vec![2],
        },
        Section::Bytes(vec![7, 8, 9]),
    ];
    let encoded = encode_payload(&sections);

    // The reference encoding, built by hand as the externally-tagged serde shape.
    let reference = Value::Array(vec![
        Value::Map(vec![(
            Value::Text("Tensor".into()),
            Value::Map(vec![
                (
                    Value::Text("data".into()),
                    Value::Array(vec![Value::Float(1.5), Value::Float(-2.0)]),
                ),
                (
                    Value::Text("shape".into()),
                    Value::Array(vec![Value::Integer(2.into())]),
                ),
            ]),
        )]),
        Value::Map(vec![(
            Value::Text("Bytes".into()),
            Value::Array(vec![
                Value::Integer(7.into()),
                Value::Integer(8.into()),
                Value::Integer(9.into()),
            ]),
        )]),
    ]);
    let decoded: Value = ciborium::from_reader(encoded.as_slice()).expect("decode");
    assert_eq!(decoded, reference, "externally-tagged v1 container shape");

    let back = decode_payload(&encoded).expect("roundtrip");
    assert_eq!(back, sections);
}
