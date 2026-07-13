// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Merkle set-commitment conformance (TDD PROTO-5 root/proof half, spec §6.4).

use daemon_swarm_proto::merkle::{commit_set, MembershipProof};
use daemon_swarm_proto::{Hash, PeerId};

fn peer(n: u32) -> PeerId {
    let mut b = [0u8; 32];
    b[..4].copy_from_slice(&n.to_be_bytes());
    PeerId(b)
}

fn hash(n: u32) -> Hash {
    let mut b = [0u8; 32];
    b[28..].copy_from_slice(&n.to_be_bytes());
    Hash(b)
}

fn set(n: u32) -> Vec<(PeerId, Hash)> {
    (0..n).map(|i| (peer(i), hash(i * 7 + 1))).collect()
}

#[test]
fn record_root_matches_set() {
    // Root is a deterministic function of the set (recompute → identical).
    for n in [1u32, 2, 3, 4, 5, 8, 13, 64, 512] {
        let entries = set(n);
        let a = commit_set(&entries).commitment();
        let b = commit_set(&entries).commitment();
        assert_eq!(a, b, "n={n}");
        assert_eq!(a.count, n);
    }
}

#[test]
fn set_order_is_pubkey_bytes() {
    // Two different input orderings of the same set commit to the same root, and the tree's stored
    // entries are sorted by peer bytes.
    let mut forward = set(16);
    let mut reversed = forward.clone();
    reversed.reverse();
    let t1 = commit_set(&forward);
    let t2 = commit_set(&reversed);
    assert_eq!(t1.commitment().root, t2.commitment().root);

    forward.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(t1.entries(), forward.as_slice());
    // Explicitly assert ascending peer order.
    for w in t1.entries().windows(2) {
        assert!(w[0].0 <= w[1].0);
    }
}

#[test]
fn membership_proof_verifies() {
    // Every element of every roster size proves against the committed root.
    for n in [1u32, 2, 3, 4, 5, 7, 8, 15, 16, 64, 512] {
        let entries = set(n);
        let tree = commit_set(&entries);
        let commitment = tree.commitment();
        for (i, (p, h)) in tree.entries().iter().enumerate() {
            let proof = tree.prove(i).expect("in-range proof");
            commitment
                .verify_membership(p, h, &proof)
                .unwrap_or_else(|e| panic!("n={n} i={i}: {e}"));
        }
        assert!(tree.prove(n as usize).is_none(), "out-of-range proof");
    }
}

#[test]
fn tampered_proof_and_absent_peer_rejected() {
    let entries = set(32);
    let tree = commit_set(&entries);
    let commitment = tree.commitment();
    let (p0, h0) = tree.entries()[0];
    let good = tree.prove(0).unwrap();

    // A flipped sibling no longer reconstructs the root.
    let mut bad = good.clone();
    bad.siblings[0].0[0] ^= 0xff;
    assert!(commitment.verify_membership(&p0, &h0, &bad).is_err());

    // A peer that is not in the set cannot be shown as a member (its leaf differs).
    let absent = peer(9999);
    assert!(commitment.verify_membership(&absent, &h0, &good).is_err());

    // A wrong payload hash for a real peer also fails.
    let wrong_hash = hash(123_456);
    assert!(commitment
        .verify_membership(&p0, &wrong_hash, &good)
        .is_err());
}

#[test]
fn proof_is_log_sized() {
    let tree = commit_set(&set(512));
    let proof: MembershipProof = tree.prove(0).unwrap();
    // 512 leaves → 9 levels of siblings.
    assert_eq!(proof.siblings.len(), 9);
}

proptest::proptest! {
    /// For arbitrary rosters of arbitrary peer keys, every committed member proves against the root.
    #[test]
    fn prop_every_member_verifies(
        keys in proptest::collection::vec(proptest::array::uniform32(proptest::prelude::any::<u8>()), 1..40)
    ) {
        let entries: Vec<(PeerId, Hash)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (PeerId(*k), hash(i as u32)))
            .collect();
        let tree = commit_set(&entries);
        let commitment = tree.commitment();
        for (i, (p, h)) in tree.entries().iter().enumerate() {
            let proof = tree.prove(i).unwrap();
            proptest::prop_assert!(commitment.verify_membership(p, h, &proof).is_ok());
        }
    }
}
