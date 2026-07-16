// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 record archive (architecture §4.4; refactor §8/D2): signed, hash-chained journal segments
// published content-addressed, with declared replication / retention / GC, and fork detection over
// gossiped signed chain heads (two non-extending heads = portable evidence).
//
// Sanctioned raw-fs test home (the segment writer needs real files): same pattern as the journal
// tests.
#![allow(clippy::disallowed_methods)]

use daemon_vhc_observe::journal::record::{Body, ClockRec, ExecIdentity, Record};
use daemon_vhc_observe::journal::segment::{SegmentHeader, SegmentWriter};
use daemon_vhc_observe::{
    detect_fork, ArchiveError, AttestedHead, ChainHead, ForkEvidence, RecordArchive,
    ReplicationPolicy, RetentionPolicy,
};
use daemon_vhc_proto::{peer_id, Hash, PeerId, SigningKey};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

fn h(b: u8) -> Hash {
    Hash([b; 32])
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: h(1),
        epoch: 4,
        role: "coordinator".into(),
        instance: 42,
        module: h(2),
    }
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-archive-test-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// Build a cleanly-sealed segment carrying one distinguishing record; returns its bytes and the
/// content address (BLAKE3 of the complete sealed file — the §8.2 chain link).
fn sealed_segment(seg: u64, prev: [u8; 32], marker: u64) -> (Vec<u8>, Hash) {
    let dir = tempdir();
    let path = dir.join(format!("seg-{seg}.dvhcjrn"));
    let header = SegmentHeader {
        id: ident(),
        segment: seg,
        prev_blake3: prev,
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    w.append(&Record::new(0, Body::Clock(ClockRec { now: marker })))
        .unwrap();
    let hash = w.seal().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (bytes, Hash(hash))
}

/// An UNsealed (committed but not sealed) segment — not archivable.
fn unsealed_segment(marker: u64) -> Vec<u8> {
    let dir = tempdir();
    let path = dir.join("seg-unsealed.dvhcjrn");
    let header = SegmentHeader {
        id: ident(),
        segment: 0,
        prev_blake3: [0u8; 32],
    };
    let mut w = SegmentWriter::create(&path, &header).unwrap();
    w.append(&Record::new(0, Body::Clock(ClockRec { now: marker })))
        .unwrap();
    w.commit().unwrap();
    std::fs::read(&path).unwrap()
}

fn head(signer: &SigningKey, seg: u64, seg_hash: Hash, prev: Hash) -> AttestedHead {
    AttestedHead::single(
        signer,
        ChainHead {
            run_id: h(1),
            epoch: 4,
            role: "coordinator".into(),
            instance: 42,
            module: h(2),
            segment: seg,
            segment_hash: seg_hash,
            prev_hash: prev,
            records: 1,
        },
    )
    .unwrap()
}

fn single_key(authority: PeerId) -> AuthorityConfig {
    AuthorityConfig {
        topology: Topology::SingleKey(SingleKey::new(authority)),
        records_channel: DEFAULT_RECORDS_CHANNEL,
    }
}

fn archive(authority: PeerId) -> RecordArchive {
    RecordArchive::new(
        single_key(authority),
        ReplicationPolicy { factor: 2 },
        RetentionPolicy::default(),
    )
}

#[test]
fn publishes_content_addressed_and_replicates_to_durable() {
    let coord = key(1);
    let mut arc = archive(peer_id(&coord));
    let (bytes, addr) = sealed_segment(0, [0u8; 32], 100);

    let stored = arc.publish_segment(bytes.clone()).unwrap();
    assert_eq!(stored, addr, "content address == BLAKE3 of the sealed file");
    assert_eq!(arc.fetch(&addr), Some(bytes.as_slice()));
    assert!(!arc.is_durable(&addr), "one replica < factor 2");

    assert_eq!(arc.replicate(&addr), 2);
    assert!(arc.is_durable(&addr), "reached the replication factor");
    // Re-publishing the identical bytes is idempotent (bumps replica count, capped at factor).
    assert_eq!(arc.publish_segment(bytes).unwrap(), addr);
    assert_eq!(arc.len(), 1);
}

#[test]
fn unsealed_segment_is_refused() {
    let coord = key(1);
    let mut arc = archive(peer_id(&coord));
    assert_eq!(
        arc.publish_segment(unsealed_segment(7)).unwrap_err(),
        ArchiveError::NotSealed
    );
}

#[test]
fn divergent_heads_are_portable_fork_evidence() {
    let coord = key(1);
    let authority = peer_id(&coord);
    // Two different sealed segment-0 histories → two different content addresses.
    let (_, addr_a) = sealed_segment(0, [0u8; 32], 100);
    let (_, addr_b) = sealed_segment(0, [0u8; 32], 200);
    assert_ne!(addr_a, addr_b);

    let head_a = head(&coord, 0, addr_a, h(0));
    let head_b = head(&coord, 0, addr_b, h(0));

    // Stand-alone detection over gossiped heads: the equivocation-drill primitive, judged
    // through D1's AuthorityConfig::authorize.
    let config = single_key(authority);
    match detect_fork(&[head_a.clone(), head_b.clone()], &config) {
        Some(ForkEvidence::DivergentHead { a, b }) => {
            // Portable: both heads authorize on their own under the declared topology, needing
            // nothing beyond the two heads + the run's AuthorityConfig (§4.3).
            let ok = |head: &AttestedHead| {
                config
                    .authorize(&head.preimage().unwrap(), &head.sigs)
                    .is_ok()
            };
            assert!(ok(&a) && ok(&b));
            assert_eq!(a.body.segment, b.body.segment);
            assert_ne!(a.body.segment_hash, b.body.segment_hash);
        }
        other => panic!("expected DivergentHead, got {other:?}"),
    }

    // End-to-end through the archive: the first head is accepted, the second yields evidence.
    let mut arc = archive(authority);
    assert_eq!(arc.ingest_head(head_a).unwrap(), None);
    assert!(matches!(
        arc.ingest_head(head_b).unwrap(),
        Some(ForkEvidence::DivergentHead { .. })
    ));
}

#[test]
fn non_extending_head_is_detected_and_extending_head_accepted() {
    let coord = key(1);
    let authority = peer_id(&coord);
    let mut arc = archive(authority);

    let (_, addr0) = sealed_segment(0, [0u8; 32], 100);
    let (_, addr1) = sealed_segment(1, addr0.0, 101);

    assert_eq!(arc.ingest_head(head(&coord, 0, addr0, h(0))).unwrap(), None);

    // A segment-1 head whose prev link does not match the accepted segment 0 → non-extending.
    let bad = head(&coord, 1, addr1, h(0xEE));
    assert!(matches!(
        arc.ingest_head(bad).unwrap(),
        Some(ForkEvidence::NonExtending { .. })
    ));

    // The correctly-linked segment-1 head extends the chain and is accepted.
    let good = head(&coord, 1, addr1, addr0);
    assert_eq!(arc.ingest_head(good).unwrap(), None);
    assert!(arc.head_at(1).is_some());
}

#[test]
fn unauthoritative_head_is_refused_and_ignored() {
    let coord = key(1);
    let impostor = key(9);
    let authority = peer_id(&coord);
    let mut arc = archive(authority);

    let (_, addr) = sealed_segment(0, [0u8; 32], 100);
    let forged = head(&impostor, 0, addr, h(0));

    assert!(matches!(
        arc.ingest_head(forged.clone()),
        Err(ArchiveError::Unauthoritative(_))
    ));
    // An unsigned/forged head cannot manufacture a fork: detect_fork ignores it even paired with a
    // genuine authoritative head at the same height with different content.
    let (_, addr2) = sealed_segment(0, [0u8; 32], 200);
    let genuine = head(&coord, 0, addr2, h(0));
    assert_eq!(
        detect_fork(&[forged, genuine], &single_key(authority)),
        None
    );
}

#[test]
fn gc_respects_the_retention_horizon_and_attested_window() {
    let coord = key(1);
    let mut arc = archive(peer_id(&coord));
    // Publish 11 segments (ordinals 0..=10), each a distinct content address.
    for seg in 0..=10u64 {
        let (bytes, _) = sealed_segment(seg, [0u8; 32], 1000 + seg);
        arc.publish_segment(bytes).unwrap();
    }
    assert_eq!(arc.head_segment(), Some(10));

    // Attestation-keyed GC with nothing attested keeps everything (never collect an unwitnessed
    // prefix).
    assert_eq!(arc.gc(), 0);
    assert_eq!(arc.len(), 11);

    // Attest segment 8: the GC floor is min(head - horizon, attested) = min(10 - 4, 8) = 6, so
    // segments 0..6 are collectable (6 dropped); the attested catch-up window (>= 6) is kept.
    arc.set_attested_checkpoint(8);
    assert_eq!(arc.gc(), 6);
    assert_eq!(arc.len(), 5, "segments 6..=10 retained");
}
