// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The profile golden suite: the profiles reproduce their PINNED bit-pattern outputs over pinned
// trajectory inputs — bit-for-bit, forever.
//
// ## Oracle provenance / regeneration
//
// The pinned inputs (`*_BASE0` / `*_THETA*` bit-pattern arrays) are the exact deterministic
// θ/round-base trajectories the retired v1 reference tape produced at the documented seeds
// (0xDAE0_7E57 / 0x1234 / 0xABCD / 0x5150), captured as literals when that tape retired. The
// pinned outputs (`*_WANT*`) are the post-ingest masters recorded when this crate's profiles
// were proven bit-equal to the retired reference implementation in a live A/B at the same
// commit lineage — so the input→output pairs carry the equivalence proof forward as standing
// literals. The end-to-end drift oracle for the profile math is the trainer-goldens bundle
// (`daemon-vhc-host/tests/trainer_goldens.rs`), which pins digests + committed payloads through
// the production trainer guest.
//
// The det lane is CPU fp32 with fixed evaluation order (every kernel delegates to
// `daemon-vhc-det`), so bit-equality is the designed property, not luck — a mismatch is a
// porting bug, never tolerance.

use daemon_vhc_sdk_profiles::{
    decode_payload, encode_payload, Demo, DemoCfg, DiLoCo, DiLoCoCfg, IngestParam, ParamView,
    Section, SparseLoco, SparseLocoCfg,
};

fn f32s(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ===== the pinned trajectory inputs (module provenance note) ====================================

const SL_BASE0: &[u32] = &[
    3186823795, 1035905016, 3166582922, 3179708986, 1032789417, 1042697249, 1047616210, 1037443080,
    1034583887, 1022732346, 3172849934, 3182726122, 3197917516, 3188851315, 1024065251, 3186866866,
    1023940144, 3197721546, 1024670989, 1017043157, 1037873341, 3182951129, 1000092354, 3184372672,
    3187324055, 3186178491, 3179967907, 1039907904, 3186470407, 1041321735, 1041618040, 3187677581,
    3173339091, 1043485881, 1039702757, 1026832117, 3194774813, 3185273813, 1019562548, 1011522219,
    1039729050, 1036564907, 3188005499, 3185471786, 3157006803, 3183209039, 1024173708, 3172584317,
    1045448963, 1027019828, 3160831576, 1042388084, 1038197015, 3187948427, 1032671187, 3171301911,
    1029413506, 1036349793, 3172062976, 3160522523, 1023866034, 3185354266, 1024394021, 3170296224,
];
const SL_THETA: &[u32] = &[
    1043740780, 1053137388, 1049323867, 1047260950, 1052382660, 1055354710, 1057270040, 1053508223,
    1052817891, 1051077005, 1048817408, 1045769414, 3153587079, 1042150729, 1051198569, 1043719442,
    1051183272, 3135599087, 1051272618, 1050728753, 1053611725, 1045658103, 1050247367, 1044954630,
    1043492915, 1044060451, 1047133031, 1054099572, 1043915850, 1054706319, 1054846634, 1043314486,
    1048757111, 1055722573, 1054050508, 1051536588, 1032347163, 1044508489, 1050883036, 1050497628,
    1054056798, 1053296648, 1042989439, 1044410456, 1049743795, 1045530502, 1051211829, 1048850147,
    1056621912, 1051559498, 1049626215, 1055209672, 1053689512, 1043046017, 1052353937, 1049008171,
    1051851417, 1053244758, 1048914396, 1049635717, 1051174211, 1044468651, 1051238763, 1049095249,
];
const EF_BASE0: &[u32] = &[
    3186823795, 1035905016, 3166582922, 3179708986, 1032789417, 1042697249, 1047616210, 1037443080,
    1034583887, 1022732346, 3172849934, 3182726122, 3197917516, 3188851315, 1024065251, 3186866866,
    1023940144, 3197721546, 1024670989, 1017043157, 1037873341, 3182951129, 1000092354, 3184372672,
    3187324055, 3186178491, 3179967907, 1039907904, 3186470407, 1041321735, 1041618040, 3187677581,
    3173339091, 1043485881, 1039702757, 1026832117, 3194774813, 3185273813, 1019562548, 1011522219,
    1039729050, 1036564907, 3188005499, 3185471786, 3157006803, 3183209039, 1024173708, 3172584317,
    1045448963, 1027019828, 3160831576, 1042388084, 1038197015, 3187948427, 1032671187, 3171301911,
    1029413506, 1036349793, 3172062976, 3160522523, 1023866034, 3185354266, 1024394021, 3170296224,
];
const EF_THETA0: &[u32] = &[
    1034210883, 1049965263, 1043588353, 1040747884, 1049191614, 1052267951, 1054679461, 1050346835,
    1049637316, 1047144565, 1042563878, 1038299128, 3185111708, 1030220410, 1047391867, 1034167905,
    1047360742, 3184328643, 1047542560, 1046436685, 1050453530, 1038074675, 1045459480, 1036656530,
    1033711685, 1034854791, 1040618779, 1050957738, 1034563510, 1051588391, 1051734903, 1033352371,
    1042441977, 1052656818, 1050906923, 1048080106, 3170962244, 1035757457, 1046750188, 1045967329,
    1050913436, 1050129004, 1032697884, 1035559930, 1044438651, 1037817395, 1047418849, 1042630070,
    1053621674, 1048126788, 1044200484, 1052115346, 1050533778, 1032811796, 1049162239, 1042949632,
    1048648994, 1050075633, 1042759986, 1044219729, 1047342305, 1035677186, 1047473659, 1043125768,
];
const EF_THETA1: &[u32] = &[
    1052003126, 1057738128, 1055033240, 1053697381, 1057406798, 1058639518, 1059317687, 1057897369,
    1057598960, 1056667419, 1054553979, 1052981913, 1043245643, 1051232175, 1056778952, 1051992800,
    1056764934, 1043630421, 1056846755, 1056346447, 1057941332, 1052928366, 1055899531, 1052589509,
    1051883146, 1052157746, 1053636178, 1058145363, 1052087821, 1058390590, 1058445725, 1051796730,
    1054496748, 1058773242, 1058125097, 1057026095, 1048141306, 1052374225, 1056488902, 1056132315,
    1058127698, 1057806834, 1051639203, 1052326883, 1055428477, 1052866958, 1056791102, 1054585038,
    1059069938, 1057036512, 1055318021, 1058585197, 1057974224, 1051666632, 1057394020, 1054734808,
    1057168748, 1057784500, 1054645962, 1055326955, 1056756626, 1052354988, 1056815769, 1054817229,
];
const EF_MASTER1: &[u32] = &[
    1033547149, 1035905016, 1058754108, 1052049266, 1032789417, 1042697249, 1047616210, 1037443080,
    1034583887, 1022732346, 1058497135, 1051294982, 3193082520, 3174009292, 1024065251, 1033504078,
    1023940144, 3192690580, 1024670989, 1017043157, 1037873341, 1057685669, 1000092354, 1050899728,
    1033046889, 1050448273, 1058058572, 1039907904, 1033900537, 1041321735, 1041618040, 3167734888,
    1052898822, 1043485881, 1039702757, 1026832117, 3188086045, 1035097131, 1019562548, 1011522219,
    1039729050, 1036564907, 3170391000, 1034899158, 1058967449, 1051174252, 1024173708, 1058513736,
    1045448963, 1027019828, 1053715533, 1042388084, 1038197015, 3169967192, 1032671187, 1042616698,
    1029413506, 1036349793, 1042426432, 1058887940, 1059671883, 3154456784, 1024394021, 1053176294,
];
const CLIP_BASE0: &[u32] = &[
    3187607674, 1041325501, 3163179669, 3188547694, 1026396379, 1043743709, 996755040, 3165206619,
    1033107624, 1024160920, 1042843334, 1036992154, 1021939672, 3189168183, 3194203042, 1018337859,
    3173722854, 3193482259, 3171462675, 3159009537, 1036787643, 1037653402, 3188143071, 3171806367,
    1020049373, 3189495599, 3179119421, 3170844861, 3172951206, 3144473637, 3188080314, 1043224015,
];
const CLIP_THETA_A: &[u32] = &[
    1033428660, 1051590254, 1044012258, 1031432536, 1047971734, 1052783812, 1045378789, 1043759790,
    1049270669, 1047415668, 1052340029, 1050234994, 1047045949, 1028955352, 3166462736, 1046597797,
    1042346338, 3158904069, 1042909572, 1044313944, 1050184262, 1050398992, 1032423297, 1042823930,
    1046810762, 1027648129, 1041001205, 1043057416, 1042538641, 1044899276, 1032548557, 1052527762,
];
const CLIP_THETA_B: &[u32] = &[
    1060696499, 1058974718, 1060602701, 1060659699, 1060215400, 1058352563, 1060498030, 1060618791,
    1059921042, 1060285464, 1058593756, 1059571654, 1060329169, 1060629077, 1060233263, 1060379080,
    1060690643, 1060304633, 1060665679, 1060582157, 1059591615, 1059505934, 1060677289, 1060669786,
    1060355780, 1060611178, 1060731139, 1060658328, 1060682661, 1060538169, 1060679846, 1058493086,
];
const DILOCO_BASE0: &[u32] = &[
    1016350619, 3158069443, 3106915430, 1032979902, 3180100288, 3170556791, 3176606528, 3171190574,
    1033462571, 1030370563, 1037533157, 3192452376, 1025532617, 1041088703, 3175878563, 3190292399,
];
const DILOCO_THETA: &[u32] = &[
    1050686326, 1049711132, 1050084743, 1052428924, 1047067624, 1049079202, 1048132247, 1049021887,
    1052546087, 1051968002, 1053529901, 1036964165, 1051377902, 1054595744, 1048311877, 1040720843,
];
const DILOCO_PLAIN_MASTER: &[u32] = &[
    1046861703, 1044900618, 1045651701, 1049474194, 1041106002, 1043630810, 1042174610, 1043515676,
    1049592408, 1049009406, 1050586372, 1010930944, 1048253826, 1051666726, 1042354948, 1026791372,
];
const DEMO_BASE0: &[u32] = &[
    1040571298, 3166857934, 3166286083, 3165510439, 3197902579, 3168375266, 3187854677, 3168504962,
    1021102168, 1041693543, 3186998143, 1006898567, 3173528912, 3186690322, 3181921126, 3172872394,
    3164897987, 3178818339, 1017360815, 1046497943, 1035452528, 1031686792, 3189130649, 974837989,
    1041562055, 1035511898, 1040227673, 1021937219, 1035297977, 1025530679, 3191358816, 1023892718,
    1038911035, 3182791557, 1042862917, 1030800936, 3192522967, 1043156823, 1044001303, 1023975291,
    1043711457, 1030355661, 3186927447, 1042307121, 1041947750, 1028691463, 1035162975, 1016777148,
    1038199136, 3172033366, 3191653881, 1036187645, 3180984331, 1036879218, 1028616792, 3180099327,
    1043235665, 1008331313, 3187089725, 3152434387, 3194047861, 3182429737, 3174128159, 3179230714,
];
const DEMO_THETA: &[u32] = &[
    1047282184, 1033646617, 1033789580, 1033983491, 3193034624, 1033267284, 3169007477, 1033234860,
    1040318321, 1048404429, 3164846793, 1037913726, 1030841417, 3163615509, 1016210843, 1031497935,
    1034136604, 1025551990, 1039513912, 1050892415, 1044530846, 1042675976, 3175054539, 1036911230,
    1048272941, 1044560531, 1046938559, 1040422703, 1044453571, 1041136948, 3181624819, 1040727458,
    1046260100, 1010436669, 1049074902, 1042454512, 3183953121, 1049221855, 1049644095, 1040748101,
    1049499172, 1042343194, 3164564009, 1048797004, 1048617318, 1041927144, 1044386070, 1039367996,
    1045904150, 1032067874, 3182214949, 1044898405, 1019958023, 1045244191, 1041908476, 1023454107,
    1049261276, 1038092819, 3165213121, 1035888511, 3187002909, 1013331229, 1030242170, 1025139615,
];

// ===== sparse_loco: the pinned round golden =====================================================

/// The pinned round output (recorded from the bit-reproducible reference at seed 0xDAE0_7E57 —
/// see the module provenance note).
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
fn sparse_loco_round_reproduces_the_pinned_golden() {
    let numel = 64usize;
    let cfg = SparseLocoCfg {
        h: 3,
        chunk: 16,
        topk: 4,
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let (base0, theta) = (f32s(SL_BASE0), f32s(SL_THETA));
    let mut new = SparseLoco::new(cfg, &[numel]);
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
    assert_eq!(bits(&master), SDK1_WANT, "sparse_loco pinned golden");
}

// ===== sparse_loco: error feedback across rounds (pinned) =======================================

#[test]
fn sparse_loco_error_feedback_carries_across_rounds() {
    let numel = 64usize;
    let cfg = SparseLocoCfg {
        h: 2,
        chunk: 16,
        topk: 2, // aggressive sparsity ⇒ meaningful residual carried in ef
        bits: 2,
        clip: false,
        ..SparseLocoCfg::default()
    };
    let mut new = SparseLoco::new(cfg, &[numel]);
    let base0 = f32s(EF_BASE0);

    // Round 0.
    let theta0 = f32s(EF_THETA0);
    let v0 = ParamView {
        theta: &theta0,
        round_base: &base0,
    };
    let (p1, p2) = (new.make_update(&[v0]), new.make_update(&[v0]));
    let mut master0 = vec![0.0f32; numel];
    new.ingest(
        &mut [IngestParam {
            master: &mut master0,
            round_base: &base0,
        }],
        &[p1, p2],
    )
    .expect("ingest round 0");

    // Round 1 trains from the pinned trajectory; the post-ingest master is the next round base,
    // and the profile's error-feedback residual carries between the rounds (the pinned output
    // is only reproducible when it does).
    let theta1 = f32s(EF_THETA1);
    let v1 = ParamView {
        theta: &theta1,
        round_base: &master0,
    };
    let (p1, p2) = (new.make_update(&[v1]), new.make_update(&[v1]));
    let mut master1 = vec![0.0f32; numel];
    new.ingest(
        &mut [IngestParam {
            master: &mut master1,
            round_base: &master0,
        }],
        &[p1, p2],
    )
    .expect("ingest round 1");
    assert_eq!(bits(&master1), EF_MASTER1, "ef-carry pinned golden");
}

// ===== sparse_loco: median-norm clip golden (pinned) ============================================

/// The pinned clipped output (recorded at seed 0x1234 — module provenance note).
const SDK4_WANT_CLIPPED: &[u32] = &[
    1048399937, 1041325501, 1034513755, 1047491600, 1050745946, 1043743709, 1049443532, 1034007017,
    1051962560, 1050680617, 1042843334, 1036992154, 1040400891, 1047303413, 1042268554, 1039713937,
    1030559002, 1043517125, 1048682410, 1049387044, 1036787643, 1037653402, 1048716156, 1032137136,
    1050615722, 1047449897, 1025146051, 1048731680, 1031314266, 1049655924, 3170958056, 1043224015,
];

#[test]
fn sparse_loco_median_clip_reproduces_the_pinned_golden() {
    let numel = 32usize;
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
    // Peer 1's θ is 2 steps in; peer 2's is 8 more (a dominant-norm Δ) — the clip must bite.
    let base0 = f32s(CLIP_BASE0);
    let (theta_a, theta_b) = (f32s(CLIP_THETA_A), f32s(CLIP_THETA_B));
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
    assert_eq!(bits(&master), SDK4_WANT_CLIPPED, "clip pinned golden");
}

// ===== diloco: outer Nesterov golden (pinned, both legs) ========================================

/// The pinned Nesterov output (recorded at seed 0xABCD — module provenance note).
const DILOCO_WANT_NESTEROV: &[u32] = &[
    1053950547, 1052981238, 1053352725, 1055679126, 1051100703, 1052352578, 1051630824, 1052295540,
    1055795133, 1055222456, 1056767783, 1045158268, 1054637190, 1057391135, 1051720249, 1047298411,
];

#[test]
fn diloco_reproduces_the_pinned_goldens() {
    for nesterov in [true, false] {
        let numel = 16usize;
        let base0 = f32s(DILOCO_BASE0);
        let theta = f32s(DILOCO_THETA);
        let mut new = DiLoCo::new(
            DiLoCoCfg {
                h: 3,
                nesterov,
                ..DiLoCoCfg::default()
            },
            &[numel],
        );
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
        let want = if nesterov {
            DILOCO_WANT_NESTEROV
        } else {
            DILOCO_PLAIN_MASTER
        };
        assert_eq!(bits(&master), want, "diloco pinned golden ({nesterov})");
        // The replicated momentum (digest-covered det state) exists and moved.
        assert!(new.replicated_state()[0].iter().any(|&v| v != 0.0));
    }
}

// ===== demo: per-step round golden (pinned) =====================================================

/// The pinned demo output (recorded at seed 0x5150 — module provenance note).
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
fn demo_per_step_round_reproduces_the_pinned_golden() {
    let numel = 64usize; // one 8×8 DCT tile
    let base0 = f32s(DEMO_BASE0);
    let theta = f32s(DEMO_THETA);
    let mut new = Demo::new(
        DemoCfg {
            tile: 8,
            topk: 8,
            ..DemoCfg::default()
        },
        &[numel],
    );
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
    assert_eq!(bits(&master), DEMO_WANT, "demo pinned golden");
}

// ===== the payload wire =========================================================================

/// The `Section` CBOR is byte-identical to the retired v1 host container wire: an
/// externally-tagged enum — `{"Bytes": bytes}` / `{"Tensor": {"data": […], "shape": […]}}` — the
/// committed-payload encoding the pinned goldens (and the trainer-goldens payload bytes) ride.
#[test]
fn section_wire_matches_the_container_encoding() {
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
    assert_eq!(decoded, reference, "externally-tagged container shape");

    let back = decode_payload(&encoded).expect("roundtrip");
    assert_eq!(back, sections);
}
