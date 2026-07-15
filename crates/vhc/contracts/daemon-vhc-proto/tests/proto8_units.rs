// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.1 PROTO-8 unit suite — throughput-class-weighted assignment + deliberate overlap + the
// class-ladder boundaries (spec §6.3). Completes the named PROTO-8 IDs alongside
// `assignment_golden.rs` (`assignment_weighted_by_class`, `overlap_10pct_covers_churn`):
// `overlap_zero_is_partition` and `class_ladder_boundaries`.
//
// Oracle provenance (swarm-ledger-p2-b1.md): hand-derived pinned literals for the class-weight
// ladder + from-definition cover-counting for the partition property; daemon seed 0xDAE0_7E57.

use daemon_swarm_proto::assignment::{assign_batches, class_weight};
use daemon_swarm_proto::messages::{BatchWindow, ThroughputClass};
use daemon_swarm_proto::{PeerId, Seed};

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
fn class_ladder_boundaries() {
    // The ~4× class ladder (c1<1k, c2 1–4k, c3 4–16k, c4 >16k tok/s) → integer weights 1/4/16/64,
    // frozen with the proto version. A c4 peer is assigned ~64× a c1 peer's data.
    assert_eq!(class_weight(ThroughputClass::C1), 1);
    assert_eq!(class_weight(ThroughputClass::C2), 4);
    assert_eq!(class_weight(ThroughputClass::C3), 16);
    assert_eq!(class_weight(ThroughputClass::C4), 64);

    // The ladder holds end to end in an assignment: one peer of each class over a window sized to
    // the weight sum (1+4+16+64 = 85) gives sizes exactly proportional to the weights.
    let roster = vec![
        (peer(1), ThroughputClass::C1),
        (peer(2), ThroughputClass::C2),
        (peer(3), ThroughputClass::C3),
        (peer(4), ThroughputClass::C4),
    ];
    let window = BatchWindow {
        start: 0,
        end: 85 * 10,
    };
    let out = assign_batches(&roster, &seed(), window, 0);
    let size = |p: PeerId| {
        out.iter().find(|(q, _)| *q == p).unwrap().1.end
            - out.iter().find(|(q, _)| *q == p).unwrap().1.start
    };
    assert_eq!(size(peer(1)), 10);
    assert_eq!(size(peer(2)), 40);
    assert_eq!(size(peer(3)), 160);
    assert_eq!(size(peer(4)), 640);
    assert_eq!(
        size(peer(1)) + size(peer(2)) + size(peer(3)) + size(peer(4)),
        850
    );
}

#[test]
fn overlap_zero_is_partition() {
    // overlap_bps == 0 yields an exact partition of the window: every sequence covered exactly once,
    // across a mixed-class roster.
    let roster = vec![
        (peer(1), ThroughputClass::C1),
        (peer(2), ThroughputClass::C4),
        (peer(3), ThroughputClass::C2),
        (peer(4), ThroughputClass::C3),
    ];
    let window = BatchWindow {
        start: 100,
        end: 100 + 850,
    };
    let out = assign_batches(&roster, &seed(), window, 0);

    let total: u64 = out.iter().map(|(_, w)| w.end - w.start).sum();
    assert_eq!(total, 850, "sizes sum to the window (no overlap, no gap)");
    let mut cover = vec![0u32; 850];
    for (_, w) in &out {
        for i in w.start..w.end {
            cover[(i - 100) as usize] += 1;
        }
    }
    assert!(cover.iter().all(|&c| c == 1), "exact single cover");
}
