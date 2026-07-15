// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.5 HOST-7 — the round state digest as the cross-peer agreement tripwire (spec §5.6). This
// is the "full suite beyond the P1 tripwire" (which lives in daemon-train), asserted here at the
// pure-function digest layer it shares with PROTO-18: the digest covers the *canonical* state image
// (params ++ `replicated` persistents — never `local` persistents), is bit-identical across peers
// that share the round seed, and flips on a one-bit change anywhere the schedule samples (params or
// replicated region alike). It is CPU xxh3-128 over sampled blocks: deterministic by construction.
//
// Oracle provenance (swarm-ledger-p2-b1.md): from-definition — the state image is assembled exactly
// as the host does (params, then replicated persistents, in registration order, ABI §6.3), and the
// digest is the shared `digest_state` pure function; two peers computing it must agree.

use daemon_swarm_proto::bytes::Seed;
use daemon_swarm_proto::digest::{derive_schedule, digest_state, StateLayout};

const BLOCK: u32 = 64;

/// The canonical digested state image: params, then the `replicated` persistents, in registration
/// order (ABI §6.3). `local` persistents (inner moments, error feedback) are deliberately excluded —
/// they are legitimately peer-divergent and never digested (spec §5.1).
fn canonical_state(params: &[u8], replicated: &[u8]) -> Vec<u8> {
    let mut s = Vec::with_capacity(params.len() + replicated.len());
    s.extend_from_slice(params);
    s.extend_from_slice(replicated);
    s
}

fn deterministic(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
        .collect()
}

#[test]
fn digest_stable_across_peers() {
    // Two peers with identical (seed, layout, canonical state) compute the identical digest — the
    // property the whole agree-path rests on. Full coverage (sample_count ≥ num_blocks) so the
    // entire params+replicated image is bound.
    let seed = Seed([0x77; 32]);
    let params = deterministic(4096, 0x10);
    let replicated = deterministic(2048, 0x20);
    let peer_a = canonical_state(&params, &replicated);
    let peer_b = canonical_state(&params, &replicated);
    let samples = (peer_a.len() as u32).div_ceil(BLOCK); // sample every block

    assert_eq!(
        digest_state(&seed, BLOCK, samples, &peer_a),
        digest_state(&seed, BLOCK, samples, &peer_b),
        "identical canonical state ⇒ identical digest across peers"
    );

    // Reproducible across repeated computation (no hidden nondeterminism).
    let d = digest_state(&seed, BLOCK, samples, &peer_a);
    for _ in 0..4 {
        assert_eq!(digest_state(&seed, BLOCK, samples, &peer_a), d);
    }
}

#[test]
fn digest_covers_replicated_persistents() {
    // A one-bit change inside the *replicated persistents* region (not just the params) changes the
    // digest, when that block is sampled — the digest must bind replicated consensus state, or two
    // peers with divergent outer-optimizer momentum would silently agree (spec §5.1 / §5.6).
    let seed = Seed([0x33; 32]);
    let params = deterministic(2048, 0x01);
    let replicated = deterministic(2048, 0x02);
    let base = canonical_state(&params, &replicated);
    // Full coverage so the replicated region is definitely sampled.
    let samples = (base.len() as u32).div_ceil(BLOCK);
    let base_digest = digest_state(&seed, BLOCK, samples, &base);

    // Flip one bit in the replicated region (second half of the image).
    let mut flipped = base.clone();
    let repl_off = params.len(); // first byte of the replicated region
    flipped[repl_off] ^= 0x01;
    assert_ne!(
        base_digest,
        digest_state(&seed, BLOCK, samples, &flipped),
        "a one-bit flip in the replicated region must change the digest"
    );

    // And a flip in the params region likewise changes it (both are canonical state).
    let mut flipped_param = base.clone();
    flipped_param[0] ^= 0x80;
    assert_ne!(
        base_digest,
        digest_state(&seed, BLOCK, samples, &flipped_param)
    );
}

#[test]
fn digest_changes_on_one_bit_in_a_sampled_block() {
    // Sub-sampled digest (fewer samples than blocks): a flip inside a sampled block changes the
    // digest; the schedule is a pure function of the seed so both peers sample the same blocks.
    let seed = Seed([0x5a; 32]);
    let state = deterministic(8192, 0x00);
    let samples = 16u32;
    let layout = StateLayout::of(&state, BLOCK);
    let schedule = derive_schedule(&seed, layout, samples);

    let base = digest_state(&seed, BLOCK, samples, &state);
    let sampled_block = schedule.blocks[0] as usize;
    let mut flipped = state.clone();
    flipped[sampled_block * BLOCK as usize] ^= 0x01;
    assert_ne!(base, digest_state(&seed, BLOCK, samples, &flipped));
}
