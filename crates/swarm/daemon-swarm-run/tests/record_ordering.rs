// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Cross-crate consensus invariant (§6.4 I3): record staging order == proto set-commitment order.
//!
//! `TrainerBackend::ingest` folds staged payloads in caller-supplied order, and the Wave-2 round
//! loop MUST stage in `RoundRecord` order — the committed set sorted by node public-key bytes. The
//! proto crate is the single authority for that order: `commit_set` sorts the `(peer, hash)` set by
//! `PeerId` byte order (then hash) before committing. This test pins the two to the *same* key so a
//! future reorder in either lane is a loud failure, not a silent consensus fork.

use daemon_swarm_proto::{blake3_hash, commit_set, Hash, PeerId};
use daemon_swarm_run::backend::StagedPayload;

fn peer(b: u8) -> PeerId {
    PeerId([b; 32])
}

/// The staging order the round loop uses: the committed set sorted by node pubkey bytes (I3).
fn staged_in_record_order(mut staged: Vec<StagedPayload>) -> Vec<StagedPayload> {
    staged.sort_by_key(|a| a.peer);
    staged
}

#[test]
fn staging_order_matches_proto_commit_set() {
    // A deliberately unsorted staged set (peers 0x42, 0x01, 0x99).
    let staged = vec![
        StagedPayload {
            peer: peer(0x42),
            hash: blake3_hash(b"gamma"),
            bytes: b"gamma".to_vec(),
        },
        StagedPayload {
            peer: peer(0x01),
            hash: blake3_hash(b"alpha"),
            bytes: b"alpha".to_vec(),
        },
        StagedPayload {
            peer: peer(0x99),
            hash: blake3_hash(b"beta"),
            bytes: b"beta".to_vec(),
        },
    ];

    // Runtime side: stage in RoundRecord (node-pubkey-byte) order.
    let staged_peers: Vec<PeerId> = staged_in_record_order(staged.clone())
        .iter()
        .map(|p| p.peer)
        .collect();

    // Proto side: commit_set is the single authority for the committed-set ordering.
    let entries: Vec<(PeerId, Hash)> = staged.iter().map(|p| (p.peer, p.hash)).collect();
    let tree = commit_set(&entries);
    let proto_peers: Vec<PeerId> = tree.entries().iter().map(|(p, _)| *p).collect();

    assert_eq!(
        staged_peers, proto_peers,
        "record staging order must equal proto commit_set peer order (§6.4 I3)"
    );
    // Sanity: the shared key really is ascending node-pubkey bytes.
    assert_eq!(staged_peers, vec![peer(0x01), peer(0x42), peer(0x99)]);
}

#[test]
fn ingest_is_sensitive_to_record_order() {
    use daemon_swarm_run::backend::{StubBackend, TrainerBackend};

    let mk = |peers: [u8; 3]| -> Vec<StagedPayload> {
        peers
            .iter()
            .map(|&b| StagedPayload {
                peer: peer(b),
                hash: blake3_hash(&[b]),
                bytes: vec![b],
            })
            .collect()
    };

    // Ingesting in record order vs a permutation yields a different digest — so a staging reorder
    // cannot silently agree across peers (the invariant the ordering test protects).
    let mut a = StubBackend::new();
    a.build(b"cfg").unwrap();
    let ordered = a
        .ingest(1, &staged_in_record_order(mk([0x03, 0x01, 0x02])))
        .unwrap();

    let mut b = StubBackend::new();
    b.build(b"cfg").unwrap();
    let permuted = b.ingest(1, &mk([0x03, 0x01, 0x02])).unwrap();

    assert_ne!(
        ordered, permuted,
        "unordered staging must diverge from record-ordered staging"
    );
}
