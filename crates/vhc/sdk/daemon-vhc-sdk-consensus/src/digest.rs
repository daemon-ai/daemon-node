// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The round state digest schedule (spec §5.6; TDD PROTO-18).
//!
//! After each ingest the host computes the **round state digest**: an xxh3-128 over *sampled*
//! blocks of the canonical state (params + `replicated` persistents). The sampling schedule is
//! derived from the round seed, so every peer hashes the identical blocks and a mismatch means true
//! divergence (the [`Digest`](crate::messages::Digest) message, §6.4) rather than sampling noise.
//! Content addressing of artifacts/checkpoints is full blake3 ([`daemon_vhc_proto::hash`]); xxh3
//! here is *only* the cheap per-round comparison digest.
//!
//! The state is opaque byte ranges at this layer — the host supplies the real tensor bytes later.
//! Both the schedule and the digest are **pure functions** of `(seed, layout)` (+ the bytes), so the
//! result is bit-identical across peers by construction.

use xxhash_rust::xxh3::{xxh3_64_with_seed, Xxh3};

use daemon_vhc_proto::bytes::{Seed, StateDigest};

// Domain-separating salts so the schedule PRNG and the digest hasher are seeded independently.
const SCHEDULE_SALT: u64 = 0x5761_726d_5363_6864; // "WarmSchd"
const DIGEST_SALT: u64 = 0x5761_726d_4469_6773; // "WarmDigs"

/// The layout of the opaque state being digested: total byte length and the sampling block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateLayout {
    /// Total state length in bytes.
    pub total_len: u64,
    /// Block granularity for sampling (bytes).
    pub block_size: u32,
}

impl StateLayout {
    /// The layout of `state` at `block_size`.
    #[must_use]
    pub fn of(state: &[u8], block_size: u32) -> Self {
        Self {
            total_len: state.len() as u64,
            block_size,
        }
    }

    /// The number of blocks (the last one may be partial).
    #[must_use]
    pub fn num_blocks(&self) -> u64 {
        if self.block_size == 0 {
            return 0;
        }
        self.total_len.div_ceil(u64::from(self.block_size))
    }
}

/// A deterministic set of block indices to sample, sorted ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigestSchedule {
    /// The sampled block indices (sorted, distinct).
    pub blocks: Vec<u64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive the sampling schedule — a pure function of `(seed, layout, sample_count)`.
///
/// If the state has no more than `sample_count` blocks, every block is sampled. Otherwise
/// `sample_count` distinct indices are drawn with a seed-keyed splitmix64 PRNG (rejection sampling
/// of duplicates), so the schedule is identical for every peer that shares the round seed.
#[must_use]
pub fn derive_schedule(seed: &Seed, layout: StateLayout, sample_count: u32) -> DigestSchedule {
    let num_blocks = layout.num_blocks();
    if num_blocks == 0 {
        return DigestSchedule { blocks: Vec::new() };
    }
    let want = u64::from(sample_count).min(num_blocks);
    if want == num_blocks {
        return DigestSchedule {
            blocks: (0..num_blocks).collect(),
        };
    }
    let mut prng = xxh3_64_with_seed(seed.as_bytes(), SCHEDULE_SALT);
    let mut chosen = std::collections::BTreeSet::new();
    while (chosen.len() as u64) < want {
        let idx = splitmix64(&mut prng) % num_blocks;
        chosen.insert(idx);
    }
    DigestSchedule {
        blocks: chosen.into_iter().collect(),
    }
}

/// Digest the sampled blocks of `state` under a pre-derived `schedule`.
///
/// Each sampled block is bound to its index (so identical bytes at different offsets differ) and
/// folded into a seed-keyed xxh3-128.
#[must_use]
pub fn digest_with_schedule(
    seed: &Seed,
    layout: StateLayout,
    schedule: &DigestSchedule,
    state: &[u8],
) -> StateDigest {
    let hash_seed = xxh3_64_with_seed(seed.as_bytes(), DIGEST_SALT);
    let mut hasher = Xxh3::with_seed(hash_seed);
    let block_size = u64::from(layout.block_size);
    for &block in &schedule.blocks {
        let start = (block * block_size).min(state.len() as u64) as usize;
        let end = ((block + 1) * block_size).min(state.len() as u64) as usize;
        hasher.update(&block.to_le_bytes());
        hasher.update(&state[start..end]);
    }
    StateDigest(hasher.digest128().to_le_bytes())
}

/// Derive the schedule and digest `state` in one call — the host's per-round entry point.
#[must_use]
pub fn digest_state(seed: &Seed, block_size: u32, sample_count: u32, state: &[u8]) -> StateDigest {
    let layout = StateLayout::of(state, block_size);
    let schedule = derive_schedule(seed, layout, sample_count);
    digest_with_schedule(seed, layout, &schedule, state)
}

/// The **streaming digest carry**: the full-coverage round state digest computed incrementally,
/// without ever materializing the state.
///
/// With `sample_count = u32::MAX` the schedule is *all blocks, ascending* — a sequential fold
/// over the state image — so the exact [`digest_state`] value is computable as a carry: the
/// seeded streaming hasher plus the absolute byte offset, injecting the 8-byte LE block-index
/// frame at each `block_size` boundary independent of how the update calls slice the state.
/// Feeding the identical bytes in the identical order through ANY split
/// (chunk-sized windows, parameter tails, mid-block splits) reproduces
/// `digest_state(seed, block_size, u32::MAX, full_state)` **bit-for-bit** — the equivalence the
/// carry's shared vectors pin.
///
/// Full coverage holds while the block count fits `u32::MAX` (at the pinned `block_size = 64`
/// that is state < 256 GiB); beyond it [`digest_state`] itself stops covering every block and
/// the carry no longer corresponds.
#[derive(Clone)]
pub struct DigestCarry {
    hasher: Xxh3,
    block_size: u64,
    offset: u64,
}

impl core::fmt::Debug for DigestCarry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DigestCarry")
            .field("block_size", &self.block_size)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl DigestCarry {
    /// A fresh carry for the round `seed` at `block_size` (the pinned digest granularity, 64).
    ///
    /// # Panics
    /// `block_size == 0` is a contract violation (no digest is defined over zero-byte blocks).
    #[must_use]
    pub fn new(seed: &Seed, block_size: u32) -> Self {
        assert!(block_size > 0, "digest block size must be > 0");
        Self {
            hasher: Xxh3::with_seed(xxh3_64_with_seed(seed.as_bytes(), DIGEST_SALT)),
            block_size: u64::from(block_size),
            offset: 0,
        }
    }

    /// Feed the next `bytes` of the state image (any split — block-index framing is injected
    /// from the absolute offset, independent of update boundaries).
    pub fn update(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let within = self.offset % self.block_size;
            if within == 0 {
                self.hasher
                    .update(&(self.offset / self.block_size).to_le_bytes());
            }
            let room = self.block_size - within;
            let take = usize::try_from(room.min(bytes.len() as u64)).expect("span fits usize");
            self.hasher.update(&bytes[..take]);
            self.offset += take as u64;
            bytes = &bytes[take..];
        }
    }

    /// The total bytes folded so far (the absolute state offset).
    #[must_use]
    pub fn bytes_folded(&self) -> u64 {
        self.offset
    }

    /// Finish: the 16-byte round state digest, bit-identical to
    /// `digest_state(seed, block_size, u32::MAX, state)` over the same bytes.
    #[must_use]
    pub fn finalize(&self) -> StateDigest {
        StateDigest(self.hasher.digest128().to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_blocks_rounds_up() {
        assert_eq!(StateLayout::of(&[0u8; 4096], 64).num_blocks(), 64);
        assert_eq!(StateLayout::of(&[0u8; 4097], 64).num_blocks(), 65);
        assert_eq!(StateLayout::of(&[], 64).num_blocks(), 0);
    }

    #[test]
    fn schedule_all_blocks_when_sample_exceeds() {
        let s = derive_schedule(&Seed([1; 32]), StateLayout::of(&[0u8; 256], 64), 100);
        assert_eq!(s.blocks, vec![0, 1, 2, 3]);
    }

    /// The carry reproduces the full-coverage digest bit-for-bit regardless of update splits —
    /// one-shot, per-byte, and a mid-block/partial-tail mix all agree with `digest_state`.
    #[test]
    fn carry_equals_full_coverage_digest_for_any_split() {
        let seed = Seed([9; 32]);
        let state: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let want = digest_state(&seed, 64, u32::MAX, &state);

        let mut one_shot = DigestCarry::new(&seed, 64);
        one_shot.update(&state);
        assert_eq!(one_shot.finalize(), want);
        assert_eq!(one_shot.bytes_folded(), 1000);

        let mut per_byte = DigestCarry::new(&seed, 64);
        for b in &state {
            per_byte.update(core::slice::from_ref(b));
        }
        assert_eq!(per_byte.finalize(), want);

        let mut mixed = DigestCarry::new(&seed, 64);
        for chunk in state.chunks(37) {
            mixed.update(chunk);
        }
        assert_eq!(mixed.finalize(), want);
    }

    /// Empty state: the carry agrees with the zero-block digest (no frames injected).
    #[test]
    fn carry_of_empty_state_matches() {
        let seed = Seed([3; 32]);
        assert_eq!(
            DigestCarry::new(&seed, 64).finalize(),
            digest_state(&seed, 64, u32::MAX, &[])
        );
    }
}
