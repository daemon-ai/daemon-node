// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Deterministic per-round assignment (spec §6.3; TDD PROTO-3/4/8).
//!
//! Every peer independently derives the same committee roles and batch intervals from
//! `(round_seed, roster)` — no per-batch RPC. This module is the **pure** authority for that math,
//! so the coordinator, every peer, and (later) the replay oracle all re-derive byte-identical
//! assignments. It lives in `daemon-swarm-proto` (not the coordinator) precisely so the oracle can
//! consume it without a coordinator dependency; the coordinator re-exports what it uses.
//!
//! Randomness is a documented 64-bit [`Lcg`] (Knuth MMIX constants) seeded from
//! `blake3(seed ‖ salt)`, driving a Fisher–Yates [`deterministic_shuffle`]. Salted shuffles
//! (`WITNESS_SALT` / `ASSIGN_SALT`) give committee selection and batch layout independent
//! permutations of the same round seed (§6.3, Appendix A.2/A.3).
//!
//! **No floats.** Class weighting, the global-batch ramp, and interval math are all integer-only, so
//! the assignment is bit-reproducible on any target (including the `wasm32`/zkVM coordinator
//! substrate, §11.2).

use crate::bytes::{PeerId, Seed};
use crate::envelope::GlobalBatch;
use crate::hash::blake3_hash;
use crate::messages::{BatchWindow, ThroughputClass};

/// Salt for the witness-committee shuffle (§6.3).
pub const WITNESS_SALT: &[u8] = b"daemon-swarm/witness/v1";
/// Salt for the batch-assignment shuffle (§6.3).
pub const ASSIGN_SALT: &[u8] = b"daemon-swarm/assign/v1";

/// Default witness-committee size (§6.3 — "witness count default 4"). `0` means "all peers witness".
pub const WITNESS_TARGET_DEFAULT: u32 = 4;

/// A deterministic 64-bit linear congruential generator (Knuth MMIX constants).
///
/// Not cryptographic — its only job is a reproducible permutation. Seed it via [`seeded_lcg`] so a
/// round seed + salt produce an independent stream. `daemon-swarm-proto`'s golden vectors pin its
/// output; the constants are frozen with [`crate::SwarmProtoVersion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lcg {
    state: u64,
}

const LCG_MULT: u64 = 6_364_136_223_846_793_005;
const LCG_INCR: u64 = 1_442_695_040_888_963_407;

impl Lcg {
    /// A generator seeded with a raw 64-bit state.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64-bit output. Advances the state (MMIX recurrence) then returns it.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(LCG_MULT).wrapping_add(LCG_INCR);
        self.state
    }

    /// A uniform integer in `[0, bound)` via rejection sampling (unbiased; `bound` must be > 0).
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0, "bound must be positive");
        // Reject the short tail above the largest multiple of `bound` so every residue is equally
        // likely (modulo bias elimination).
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % bound;
            }
        }
    }
}

/// Seed an [`Lcg`] from a round `seed` and a domain `salt`: `blake3(seed ‖ salt)`, first 8 bytes LE.
#[must_use]
pub fn seeded_lcg(seed: &Seed, salt: &[u8]) -> Lcg {
    let mut buf = Vec::with_capacity(Seed::LEN + salt.len());
    buf.extend_from_slice(seed.as_bytes());
    buf.extend_from_slice(salt);
    let h = blake3_hash(&buf);
    let mut s = [0u8; 8];
    s.copy_from_slice(&h.as_bytes()[..8]);
    Lcg::new(u64::from_le_bytes(s))
}

/// In-place Fisher–Yates shuffle driven by `rng` — deterministic for a given seeded [`Lcg`].
pub fn deterministic_shuffle<T>(items: &mut [T], rng: &mut Lcg) {
    let n = items.len();
    if n <= 1 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        items.swap(i, j);
    }
}

/// The witness quorum for `n` witnesses: `⌈⅔·n⌉` with the adopted small-n specials (§6.3,
/// Appendix A.3). The `1→1 / 2→2 / 3→2` arms are exactly the ceiling result, kept explicit so the
/// spec-verbatim table is visible and any future ratio change stays deliberate.
#[must_use]
pub fn witness_quorum(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 2,
        n => (2 * u64::from(n)).div_ceil(3) as u32,
    }
}

/// The integer assignment weight of a throughput class (§6.3). Ratios track the `~4×` class ladder
/// (`c1<1k, c2 1–4k, c3 4–16k, c4 >16k` tok/s) so a `c4` peer is assigned ~64× a `c1` peer's data.
/// Frozen with [`crate::SwarmProtoVersion`].
#[must_use]
pub fn class_weight(class: ThroughputClass) -> u64 {
    match class {
        ThroughputClass::C1 => 1,
        ThroughputClass::C2 => 4,
        ThroughputClass::C3 => 16,
        ThroughputClass::C4 => 64,
    }
}

/// The per-round committee derived from `(seed, roster)` (§6.3). Every peer is a **trainer**; the
/// **witnesses** are a seed-shuffled subset whose attestations count toward the commit rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committee {
    /// All roster members (canonical pubkey-byte order).
    pub trainers: Vec<PeerId>,
    /// The witness subset (seed-shuffled, first `witness_target`).
    pub witnesses: Vec<PeerId>,
}

/// Salt for the verifier-committee shuffle (§12).
pub const VERIFIER_SALT: &[u8] = b"daemon-swarm/verifier/v1";
/// Salt for the checkpointer (tie-breaker) election (§9).
pub const CHECKPOINTER_SALT: &[u8] = b"daemon-swarm/checkpointer/v1";

/// Select the verifier committee — a seed-shuffled `⌈n·percent/100⌉` subset (§12; TDD PROTO-15).
///
/// **`percent == 0` returns the empty set** — the shipped default keeps verification a no-op seam
/// (Psyche's `verification_percent = 0`), designed but never faked.
#[must_use]
pub fn select_verifiers(roster: &[PeerId], seed: &Seed, percent: u32) -> Vec<PeerId> {
    if percent == 0 || roster.is_empty() {
        return Vec::new();
    }
    let mut pool = roster.to_vec();
    pool.sort_unstable();
    pool.dedup();
    let n = pool.len() as u64;
    let count = ((n * u64::from(percent)).div_ceil(100) as usize).clamp(1, pool.len());
    let mut rng = seeded_lcg(seed, VERIFIER_SALT);
    deterministic_shuffle(&mut pool, &mut rng);
    pool.truncate(count);
    pool
}

/// Elect a single checkpointer deterministically from `(seed, roster)` — the tie-breaker committee
/// of one (§9; TDD PROTO-10). Order-independent (roster sorted first); `None` for an empty roster.
#[must_use]
pub fn elect_checkpointer(roster: &[PeerId], seed: &Seed) -> Option<PeerId> {
    if roster.is_empty() {
        return None;
    }
    let mut pool = roster.to_vec();
    pool.sort_unstable();
    pool.dedup();
    let mut rng = seeded_lcg(seed, CHECKPOINTER_SALT);
    let idx = rng.below(pool.len() as u64) as usize;
    Some(pool[idx])
}

/// Elect `count` checkpointers deterministically from `(seed, roster)` — the checkpointer committee
/// (§9; TDD RUN-6). The spec runs **two** elected checkpointers that upload independently and whose
/// manifests must agree (both-match registration); this returns the first `count` of the
/// checkpointer-salted shuffle (sorted + deduped roster first, so it is order-independent). Fewer
/// than `count` members ⇒ all of them (a single-uploader roster degrades, RUN-6).
#[must_use]
pub fn elect_checkpointers(roster: &[PeerId], seed: &Seed, count: usize) -> Vec<PeerId> {
    if roster.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut pool = roster.to_vec();
    pool.sort_unstable();
    pool.dedup();
    let mut rng = seeded_lcg(seed, CHECKPOINTER_SALT);
    deterministic_shuffle(&mut pool, &mut rng);
    pool.truncate(count.min(pool.len()));
    pool
}

/// Select the round committee. `witness_target == 0` makes every peer a witness (§6.3); otherwise
/// the first `min(witness_target, roster)` of the witness-salted shuffle are witnesses.
///
/// The roster is sorted by pubkey bytes before shuffling, so the result is independent of the
/// caller's input order (only the set + seed matter) — invariant for cross-peer agreement.
#[must_use]
pub fn select_committee(roster: &[PeerId], seed: &Seed, witness_target: u32) -> Committee {
    let mut trainers = roster.to_vec();
    trainers.sort_unstable();
    trainers.dedup();

    let mut pool = trainers.clone();
    let mut rng = seeded_lcg(seed, WITNESS_SALT);
    deterministic_shuffle(&mut pool, &mut rng);

    let count = if witness_target == 0 {
        pool.len()
    } else {
        (witness_target as usize).min(pool.len())
    };
    let witnesses = pool[..count].to_vec();
    Committee {
        trainers,
        witnesses,
    }
}

/// The global batch size (sequences/round) at `round`, ramped linearly `start → end` over
/// `ramp_rounds` (§6.1 `[data].global_batch`; TDD PROTO-9). Integer interpolation; clamps to `end`
/// at and beyond `ramp_rounds`, and when `ramp_rounds == 0`.
#[must_use]
pub fn global_batch_at(gb: GlobalBatch, round: u64) -> u64 {
    let start = u64::from(gb.start);
    let end = u64::from(gb.end);
    let ramp = u64::from(gb.ramp_rounds);
    if ramp == 0 || round >= ramp {
        return end;
    }
    if end >= start {
        start + (end - start) * round / ramp
    } else {
        start - (start - end) * round / ramp
    }
}

/// Advance the data cursor by one round: `data_index' = data_index + global_batch_at(gb, round)`
/// (§6.2/§6.3 — the `data_index` threading, TDD PROTO-3).
#[must_use]
pub fn advance_cursor(data_index: u64, gb: GlobalBatch, round: u64) -> u64 {
    data_index.saturating_add(global_batch_at(gb, round))
}

/// Split `window` into contiguous per-peer [`BatchWindow`]s, weighted by throughput class, with a
/// deliberate `overlap_bps` (basis points, 0–10000) so a dropped trainer only delays coverage,
/// never loses it (§6.3; TDD PROTO-8).
///
/// `overlap_bps == 0` yields an **exact partition** of `window` (every sequence covered once). A
/// positive overlap extends each peer's end by `size · overlap_bps / 10000`, clamped to the window
/// end, so neighbouring intervals overlap. The peer→interval mapping is deterministic (sorted +
/// assign-salted shuffle); the returned order is the left-to-right layout order.
#[must_use]
pub fn assign_batches(
    roster: &[(PeerId, ThroughputClass)],
    seed: &Seed,
    window: BatchWindow,
    overlap_bps: u32,
) -> Vec<(PeerId, BatchWindow)> {
    let total = window.end.saturating_sub(window.start);
    if roster.is_empty() || total == 0 {
        return Vec::new();
    }

    // Canonicalize by pubkey bytes, then shuffle so interval positions depend only on the set+seed.
    let mut peers: Vec<(PeerId, ThroughputClass)> = roster.to_vec();
    peers.sort_unstable_by_key(|a| a.0);
    peers.dedup_by(|a, b| a.0 == b.0);
    let mut rng = seeded_lcg(seed, ASSIGN_SALT);
    deterministic_shuffle(&mut peers, &mut rng);

    let weights: Vec<u128> = peers
        .iter()
        .map(|(_, c)| u128::from(class_weight(*c)))
        .collect();
    let sum_w: u128 = weights.iter().sum();
    let total_u = u128::from(total);

    // Base (floored) share + largest-remainder apportionment so the sizes sum to exactly `total`.
    let mut sizes: Vec<u64> = Vec::with_capacity(peers.len());
    let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(peers.len());
    let mut assigned: u128 = 0;
    for (i, w) in weights.iter().enumerate() {
        let scaled = total_u * w;
        let base = scaled / sum_w;
        remainders.push((scaled % sum_w, i));
        assigned += base;
        sizes.push(base as u64);
    }
    let mut leftover = (total_u - assigned) as usize;
    // Largest fractional remainder first; ties broken by (already-shuffled) index for determinism.
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for &(_, idx) in remainders.iter() {
        if leftover == 0 {
            break;
        }
        sizes[idx] += 1;
        leftover -= 1;
    }

    let mut out = Vec::with_capacity(peers.len());
    let mut cursor = window.start;
    for (i, (peer, _)) in peers.into_iter().enumerate() {
        let size = sizes[i];
        let start = cursor;
        let base_end = start + size;
        let overlap = (u128::from(size) * u128::from(overlap_bps) / 10_000) as u64;
        let end = (base_end + overlap).min(window.end);
        out.push((peer, BatchWindow { start, end }));
        cursor = base_end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(n: u8) -> Seed {
        Seed([n; 32])
    }

    fn peer(n: u32) -> PeerId {
        let mut b = [0u8; 32];
        b[..4].copy_from_slice(&n.to_be_bytes());
        PeerId(b)
    }

    #[test]
    fn lcg_is_deterministic() {
        let mut a = Lcg::new(0xDAE0_7E57);
        let mut b = Lcg::new(0xDAE0_7E57);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut rng = Lcg::new(1);
        for _ in 0..1000 {
            assert!(rng.below(7) < 7);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<u32> = (0..50).collect();
        let mut rng = seeded_lcg(&seed(3), WITNESS_SALT);
        deterministic_shuffle(&mut v, &mut rng);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_ne!(v, (0..50).collect::<Vec<_>>(), "should reorder");
    }

    #[test]
    fn witness_quorum_small_n_specials() {
        assert_eq!(witness_quorum(0), 0);
        assert_eq!(witness_quorum(1), 1);
        assert_eq!(witness_quorum(2), 2);
        assert_eq!(witness_quorum(3), 2);
        assert_eq!(witness_quorum(4), 3);
        assert_eq!(witness_quorum(6), 4);
        assert_eq!(witness_quorum(32), 22);
    }

    #[test]
    fn committee_is_seed_stable_and_order_independent() {
        let roster: Vec<PeerId> = (0..7).map(peer).collect();
        let mut reversed = roster.clone();
        reversed.reverse();
        let a = select_committee(&roster, &seed(9), 4);
        let b = select_committee(&reversed, &seed(9), 4);
        assert_eq!(a, b, "input order must not matter");
        assert_eq!(a.trainers.len(), 7);
        assert_eq!(a.witnesses.len(), 4);
    }

    #[test]
    fn committee_target_zero_is_all_witnesses() {
        let roster: Vec<PeerId> = (0..5).map(peer).collect();
        let c = select_committee(&roster, &seed(1), 0);
        assert_eq!(c.witnesses.len(), 5);
    }

    #[test]
    fn global_batch_ramps_linearly() {
        let gb = GlobalBatch {
            start: 256,
            end: 512,
            ramp_rounds: 2000,
        };
        assert_eq!(global_batch_at(gb, 0), 256);
        assert_eq!(global_batch_at(gb, 1000), 384);
        assert_eq!(global_batch_at(gb, 2000), 512);
        assert_eq!(global_batch_at(gb, 5000), 512);
    }

    #[test]
    fn assignment_zero_overlap_is_exact_partition() {
        let roster = vec![
            (peer(1), ThroughputClass::C1),
            (peer(2), ThroughputClass::C4),
            (peer(3), ThroughputClass::C2),
        ];
        let window = BatchWindow {
            start: 100,
            end: 100 + 690,
        };
        let out = assign_batches(&roster, &seed(2), window, 0);
        let total_size: u64 = out.iter().map(|(_, w)| w.end - w.start).sum();
        assert_eq!(total_size, 690, "sizes sum to the window");

        // Every sequence covered exactly once.
        let mut cover = vec![0u32; 690];
        for (_, w) in &out {
            for i in w.start..w.end {
                cover[(i - 100) as usize] += 1;
            }
        }
        assert!(cover.iter().all(|&c| c == 1), "exact single cover");
    }
}
