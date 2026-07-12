// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Deterministic-assignment golden + property suite (spec §6.3; TDD PROTO-3/4/8).
//!
//! Golden vectors are pinned from the daemon seed `0xDAE07E57`. They are this lane's own vectors
//! (the upstream Psyche swap-or-not sources are not vendored, ledger-P2 deviation): they lock the
//! LCG/shuffle output so any future change to the assignment math is a visible, deliberate break.

use daemon_swarm_proto::assignment::{
    assign_batches, deterministic_shuffle, seeded_lcg, select_committee, witness_quorum,
    ASSIGN_SALT, WITNESS_SALT,
};
use daemon_swarm_proto::envelope::GlobalBatch;
use daemon_swarm_proto::messages::{BatchWindow, ThroughputClass};
use daemon_swarm_proto::{advance_cursor, global_batch_at, Lcg, PeerId, Seed};

const GOLDEN_SEED_RAW: u64 = 0xDAE0_7E57;

fn seed() -> Seed {
    let mut b = [0u8; 32];
    b[..8].copy_from_slice(&GOLDEN_SEED_RAW.to_le_bytes());
    Seed(b)
}

fn peer(n: u32) -> PeerId {
    let mut b = [0u8; 32];
    b[..4].copy_from_slice(&n.to_be_bytes());
    PeerId(b)
}

#[test]
fn golden_lcg_stream_from_pinned_seed() {
    // GOLDEN: the first eight outputs of the MMIX LCG seeded with 0xDAE07E57.
    let mut rng = Lcg::new(GOLDEN_SEED_RAW);
    let got: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    let want: [u64; 8] = [
        12_170_999_644_640_108_442,
        2_502_666_265_089_750_369,
        1_704_155_204_856_578_652,
        2_505_965_762_792_623_995,
        9_042_357_207_960_960_750,
        18_423_543_759_526_109_989,
        6_084_815_828_060_277_200,
        17_184_138_304_411_953_887,
    ];
    assert_eq!(
        got,
        want.to_vec(),
        "LCG stream drifted from the pinned golden"
    );
}

#[test]
fn golden_shuffle_of_sixteen() {
    // GOLDEN: a 0..16 shuffle under the witness salt + pinned seed.
    let mut v: Vec<u32> = (0..16).collect();
    let mut rng = seeded_lcg(&seed(), WITNESS_SALT);
    deterministic_shuffle(&mut v, &mut rng);

    // A permutation of the domain…
    let mut sorted = v.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    // …pinned exactly.
    let want: [u32; 16] = [10, 15, 14, 5, 12, 7, 6, 11, 3, 13, 9, 0, 1, 2, 4, 8];
    assert_eq!(v, want.to_vec(), "shuffle drifted from the pinned golden");
}

#[test]
fn golden_witness_quorum_ladder() {
    // PROTO-4: n = 1..=32, ⌈⅔·n⌉ with the small-n specials.
    let got: Vec<u32> = (1..=32).map(witness_quorum).collect();
    let want: Vec<u32> = vec![
        1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18,
        18, 19, 20, 20, 21, 22,
    ];
    assert_eq!(got, want, "witness-quorum ladder drifted");
}

#[test]
fn committee_witnesses_are_distinct_roster_members() {
    let roster: Vec<PeerId> = (0..12).map(peer).collect();
    let c = select_committee(&roster, &seed(), 4);
    assert_eq!(c.trainers.len(), 12);
    assert_eq!(c.witnesses.len(), 4);
    for w in &c.witnesses {
        assert!(c.trainers.contains(w), "witness must be a roster member");
    }
    // Distinct witnesses.
    let mut ws = c.witnesses.clone();
    ws.sort_unstable();
    ws.dedup();
    assert_eq!(ws.len(), 4);
}

#[test]
fn cursor_advances_by_global_batch_each_round() {
    // PROTO-3: data_index threads across rounds by the (possibly ramping) global batch.
    let gb = GlobalBatch {
        start: 100,
        end: 200,
        ramp_rounds: 4,
    };
    let mut data_index = 0u64;
    let mut expected = 0u64;
    for round in 0..8 {
        assert_eq!(data_index, expected, "cursor at round {round}");
        expected += global_batch_at(gb, round);
        data_index = advance_cursor(data_index, gb, round);
    }
    // Rounds 0..4 ramp 100→200; rounds 4+ sit at 200.
    assert_eq!(global_batch_at(gb, 0), 100);
    assert_eq!(global_batch_at(gb, 2), 150);
    assert_eq!(global_batch_at(gb, 4), 200);
}

#[test]
fn assignment_weighted_by_class() {
    // PROTO-8: a c4 peer is assigned far more of the window than a c1 peer.
    let roster = vec![
        (peer(1), ThroughputClass::C1),
        (peer(2), ThroughputClass::C4),
    ];
    let window = BatchWindow { start: 0, end: 650 };
    let out = assign_batches(&roster, &seed(), window, 0);
    let c1 = out.iter().find(|(p, _)| *p == peer(1)).unwrap().1;
    let c4 = out.iter().find(|(p, _)| *p == peer(2)).unwrap().1;
    let c1_size = c1.end - c1.start;
    let c4_size = c4.end - c4.start;
    assert_eq!(c1_size + c4_size, 650);
    // weights 1:64 → c1 gets 10, c4 gets 640.
    assert_eq!(c1_size, 10);
    assert_eq!(c4_size, 640);
}

#[test]
fn overlap_10pct_covers_churn() {
    // PROTO-8: a positive overlap makes neighbouring windows overlap (no coverage gap on a drop).
    let roster: Vec<(PeerId, ThroughputClass)> =
        (0..4).map(|n| (peer(n), ThroughputClass::C2)).collect();
    let window = BatchWindow { start: 0, end: 400 };
    // overlap_bps = 1000 (10%); the union still spans the whole window.
    let out = assign_batches(&roster, &seed(), window, 1000);
    let min = out.iter().map(|(_, w)| w.start).min().unwrap();
    let max = out.iter().map(|(_, w)| w.end).max().unwrap();
    assert_eq!(min, 0);
    assert_eq!(max, 400);
    // …and total covered length exceeds the window (overlap present) except where clamped at the end.
    let covered: u64 = out.iter().map(|(_, w)| w.end - w.start).sum();
    assert!(
        covered > 400,
        "overlap should extend coverage, got {covered}"
    );
}

#[test]
fn assignment_covers_window_exactly_once_property() {
    // Across many rosters/seeds, overlap=0 is always an exact single cover.
    for s in 0u8..30 {
        let mut sd = [0u8; 32];
        sd[0] = s;
        let seed = Seed(sd);
        let n = 1 + (s as u32 % 9);
        let classes = [
            ThroughputClass::C1,
            ThroughputClass::C2,
            ThroughputClass::C3,
            ThroughputClass::C4,
        ];
        let roster: Vec<(PeerId, ThroughputClass)> = (0..n)
            .map(|i| (peer(i), classes[(i as usize) % 4]))
            .collect();
        let total = 37 + u64::from(s) * 13;
        let window = BatchWindow {
            start: 5,
            end: 5 + total,
        };
        let out = assign_batches(&roster, &seed, window, 0);
        let mut cover = vec![0u32; total as usize];
        for (_, w) in &out {
            for i in w.start..w.end {
                cover[(i - 5) as usize] += 1;
            }
        }
        assert!(
            cover.iter().all(|&c| c == 1),
            "seed {s}: not an exact single cover"
        );
    }
}

#[test]
fn assignment_and_committee_are_reproducible() {
    // The whole derivation is a pure function of (seed, roster) — re-running is byte-identical.
    let roster: Vec<PeerId> = (0..10).map(peer).collect();
    let weighted: Vec<(PeerId, ThroughputClass)> =
        roster.iter().map(|p| (*p, ThroughputClass::C3)).collect();
    let window = BatchWindow {
        start: 0,
        end: 1000,
    };

    let c1 = select_committee(&roster, &seed(), 4);
    let c2 = select_committee(&roster, &seed(), 4);
    assert_eq!(c1, c2);

    let a1 = assign_batches(&weighted, &seed(), window, 500);
    let a2 = assign_batches(&weighted, &seed(), window, 500);
    assert_eq!(a1, a2);

    // Independent salts give independent permutations.
    let _ = ASSIGN_SALT;
    let mut wit = seeded_lcg(&seed(), WITNESS_SALT);
    let mut asg = seeded_lcg(&seed(), ASSIGN_SALT);
    assert_ne!(wit.next_u64(), asg.next_u64());
}
