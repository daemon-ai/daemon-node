// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `record-set.cbor` object: roundtrip, content-hash stability, commitment equality, and
//! membership-proof spot-checks (spec §6.4/§11.3; TDD PROTO-5 / RUN-2).

use daemon_swarm_proto::merkle::commit_set;
use daemon_swarm_proto::messages::{Locator, RecordEntry, RoundRecord};
use daemon_swarm_proto::{to_canonical_vec, Hash, PeerId, RecordSet, Root, Seed};

const CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/daemon-swarm.cddl"));

fn peer(n: u8) -> PeerId {
    PeerId([n; 32])
}

fn hash(n: u8) -> Hash {
    Hash([n; 32])
}

fn entry(p: u8, h: u8, size: u64) -> RecordEntry {
    RecordEntry {
        peer: peer(p),
        hash: hash(h),
        size,
    }
}

fn sample() -> RecordSet {
    RecordSet::new([entry(1, 10, 100), entry(2, 20, 200), entry(3, 30, 300)])
}

#[test]
fn record_set_roundtrip_canonical() {
    let rs = sample();
    let bytes = rs.to_canonical_vec().unwrap();

    // Structurally valid against the authoritative CDDL.
    cddl_cat::validate_cbor_bytes("record-set", CDDL, &bytes)
        .unwrap_or_else(|e| panic!("record-set failed to validate: {e:?}"));

    // Lossless decode.
    let decoded = RecordSet::from_canonical_slice(&bytes).unwrap();
    assert_eq!(decoded, rs);

    // Canonical: re-encoding the decode is byte-identical.
    assert_eq!(decoded.to_canonical_vec().unwrap(), bytes);
}

#[test]
fn record_set_hash_stable_across_input_order() {
    // Same committed set, presented in two different orders (+ a duplicate) → identical object.
    let a = RecordSet::new([entry(1, 10, 100), entry(2, 20, 200), entry(3, 30, 300)]);
    let b = RecordSet::new([
        entry(3, 30, 300),
        entry(2, 20, 200),
        entry(1, 10, 100),
        entry(2, 20, 200),
    ]);
    assert_eq!(
        a.entries(),
        b.entries(),
        "canonical order is input-independent"
    );
    assert_eq!(a.to_canonical_vec().unwrap(), b.to_canonical_vec().unwrap());
    assert_eq!(a.content_hash().unwrap(), b.content_hash().unwrap());

    // The content hash actually changes when the set changes (not a constant).
    let c = RecordSet::new([entry(1, 10, 100), entry(2, 20, 201)]);
    assert_ne!(a.content_hash().unwrap(), c.content_hash().unwrap());
}

#[test]
fn record_set_commitment_matches_record() {
    let rs = sample();
    // The object's commitment is byte-for-byte the record's signed set commitment.
    let pairs: Vec<(PeerId, Hash)> = rs.entries().iter().map(|e| (e.peer, e.hash)).collect();
    let expected = commit_set(&pairs).commitment();
    assert_eq!(rs.commitment(), expected);

    // A RoundRecord built from the same committed set carries that exact root; verify_against holds.
    let record = RoundRecord {
        round: 7,
        set: rs.commitment(),
        drops: vec![],
        next_seed: Seed([9; 32]),
        set_locator: Locator::StoreKey("runs/x/rounds/7/record-set.cbor".into()),
        inline: None,
    };
    assert!(rs.verify_against(&record.set).is_ok());

    // A tampered object (extra member) no longer reconstructs the record's root.
    let tampered = RecordSet::new(rs.entries().iter().copied().chain([entry(4, 40, 400)]));
    assert!(tampered.verify_against(&record.set).is_err());
    // …and its content address differs too.
    assert_ne!(tampered.content_hash().unwrap(), rs.content_hash().unwrap());
}

#[test]
fn record_set_membership_proof_spotcheck() {
    let rs = sample();
    let pairs: Vec<(PeerId, Hash)> = rs.entries().iter().map(|e| (e.peer, e.hash)).collect();
    let tree = commit_set(&pairs);
    let commitment = tree.commitment();

    // Every member proves against the record's root; a wrong hash for the same index does not.
    for (i, (p, h)) in tree.entries().iter().enumerate() {
        let proof = tree.prove(i).expect("in-range proof");
        assert!(commitment.verify_membership(p, h, &proof).is_ok());
        assert!(
            commitment
                .verify_membership(p, &hash(0xff), &proof)
                .is_err(),
            "a non-member hash must not verify at a member's index"
        );
    }

    // An out-of-range index yields no proof.
    assert!(tree.prove(rs.len()).is_none());
}

#[test]
fn record_set_empty_object_roundtrips() {
    let rs = RecordSet::new([]);
    let bytes = rs.to_canonical_vec().unwrap();
    cddl_cat::validate_cbor_bytes("record-set", CDDL, &bytes).unwrap();
    assert_eq!(RecordSet::from_canonical_slice(&bytes).unwrap(), rs);
    // The empty set's commitment is the well-known empty root with count 0.
    assert_eq!(rs.commitment().count, 0);
    assert_ne!(rs.commitment().root, Root([0; 32]));
    // to_canonical_vec is the same as the free function on the value.
    assert_eq!(to_canonical_vec(&rs).unwrap(), bytes);
}
