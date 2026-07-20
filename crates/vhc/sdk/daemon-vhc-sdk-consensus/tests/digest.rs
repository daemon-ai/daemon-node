// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Round state digest schedule conformance (TDD PROTO-18, spec §5.6).

use daemon_vhc_proto::bytes::Seed;
use daemon_vhc_sdk_consensus::digest::{derive_schedule, digest_state, DigestCarry, StateLayout};
use proptest::prelude::*;

const BLOCK: u32 = 64;
const SAMPLES: u32 = 8;

fn state(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + 7) as u8).collect()
}

#[test]
fn digest_sampling_schedule_deterministic() {
    let seed = Seed([0x5a; 32]);
    let layout = StateLayout::of(&state(4096), BLOCK);
    let a = derive_schedule(&seed, layout, SAMPLES);
    let b = derive_schedule(&seed, layout, SAMPLES);
    assert_eq!(a, b, "same seed + layout must yield the same schedule");
    assert_eq!(a.blocks.len(), SAMPLES as usize);
    // Sorted, distinct, in range.
    for w in a.blocks.windows(2) {
        assert!(w[0] < w[1]);
    }
    assert!(a.blocks.iter().all(|&b| b < layout.num_blocks()));

    // A different seed almost certainly picks a different set of blocks.
    let other = derive_schedule(&Seed([0x5b; 32]), layout, SAMPLES);
    assert_ne!(a, other);
}

#[test]
fn digest_changes_on_one_bit() {
    let seed = Seed([0x11; 32]);
    let data = state(4096);
    let base = digest_state(&seed, BLOCK, SAMPLES, &data);

    // Flip one bit inside a block that IS in the schedule → digest changes.
    let layout = StateLayout::of(&data, BLOCK);
    let schedule = derive_schedule(&seed, layout, SAMPLES);
    let sampled_block = schedule.blocks[0];
    let mut flipped = data.clone();
    let byte = (sampled_block as usize) * BLOCK as usize;
    flipped[byte] ^= 0x01;
    let changed = digest_state(&seed, BLOCK, SAMPLES, &flipped);
    assert_ne!(
        base, changed,
        "a bit flip in a sampled block must change the digest"
    );
}

#[test]
fn digest_ignores_unsampled_blocks() {
    // Demonstrates the sampling property: a bit outside the schedule does not affect the digest.
    let seed = Seed([0x22; 32]);
    let data = state(4096);
    let layout = StateLayout::of(&data, BLOCK);
    let schedule = derive_schedule(&seed, layout, SAMPLES);
    let unsampled = (0..layout.num_blocks())
        .find(|b| !schedule.blocks.contains(b))
        .expect("some block is unsampled at 8/64");
    let base = digest_state(&seed, BLOCK, SAMPLES, &data);
    let mut flipped = data.clone();
    flipped[(unsampled as usize) * BLOCK as usize] ^= 0x01;
    let same = digest_state(&seed, BLOCK, SAMPLES, &flipped);
    assert_eq!(base, same);
}

#[test]
fn digest_is_cross_peer_stable() {
    // Two peers with identical (seed, layout, bytes) compute the identical digest.
    let seed = Seed([0x33; 32]);
    let data = state(9000);
    assert_eq!(
        digest_state(&seed, BLOCK, SAMPLES, &data),
        digest_state(&seed, BLOCK, SAMPLES, &data),
    );
}

proptest! {
    /// The streaming carry reproduces the full-coverage digest bit-for-bit for ARBITRARY
    /// (unaligned) update splits: any state length (incl. partial final blocks), any split
    /// points. The equivalence the pinned digest-carry vectors extend with fixed golden values.
    #[test]
    fn carry_equals_full_coverage_digest_for_arbitrary_splits(
        seed_fill in 0u8..=255,
        len in 0usize..3000,
        splits in proptest::collection::vec(1usize..257, 0..40),
    ) {
        let seed = Seed([seed_fill; 32]);
        let data = state(len);
        let want = digest_state(&seed, BLOCK, u32::MAX, &data);

        let mut carry = DigestCarry::new(&seed, BLOCK);
        let mut off = 0usize;
        for s in splits {
            if off >= data.len() {
                break;
            }
            let end = (off + s).min(data.len());
            carry.update(&data[off..end]);
            off = end;
        }
        carry.update(&data[off..]);
        prop_assert_eq!(carry.finalize(), want);
        prop_assert_eq!(carry.bytes_folded(), data.len() as u64);
    }
}
