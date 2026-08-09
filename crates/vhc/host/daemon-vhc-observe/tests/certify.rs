// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The lineage semantic fold's unit gate (certification kernel, observe half): across a whole
//! coordinator lineage the published decisions must tell ONE non-equivocating, continuous
//! story. GREEN tolerates exactly the duplication replay-forward re-publishing produces
//! (identical records deduplicate); everything else is a typed archive-material refusal.
//!
//! The records are synthesized in the PRODUCTION wire form (tag-4 publishes / tag-12 signed
//! frames carrying §12.1 `[envelope, payload, sig]` frames) — the same projection
//! `extract_wire_capture` runs over a real pulled archive.

use ciborium::value::Value;

use daemon_vhc_observe::journal::record::{Body, PublishRec, Record, SignedFrameRec};
use daemon_vhc_observe::{semantic_fold, SemanticFoldError};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed, StateDigest};
use daemon_vhc_sdk_consensus::messages::{Digest, Locator, RoundRecord, VhcMessage};

/// A §12.1 signed wire frame around `msg` (the sig is not consulted by the fold's projection).
fn wire_frame(sender: [u8; 32], msg: &VhcMessage) -> Vec<u8> {
    let payload = to_canonical_vec(msg).expect("payload encode");
    let frame = Value::Array(vec![
        Value::Map(vec![(Value::from("sender"), Value::Bytes(sender.to_vec()))]),
        Value::Bytes(payload),
        Value::Bytes(vec![0u8; 64]),
    ]);
    to_canonical_vec(&frame).expect("frame encode")
}

/// A committed `RoundRecord` for `round`; `flavor` perturbs the set root so two records for one
/// round can be made deliberately DIFFERENT (the equivocation shape).
fn round_record(round: u64, flavor: u8) -> VhcMessage {
    VhcMessage::RoundRecord(RoundRecord {
        round,
        set: daemon_vhc_proto::SetCommitment {
            root: daemon_vhc_proto::Root(blake3_hash(&[flavor, round as u8]).0),
            count: 2,
        },
        drops: Vec::new(),
        next_seed: Seed([round as u8; 32]),
        set_locator: Locator::StoreKey(format!("round/{round}")),
        inline: None,
    })
}

/// A tag-4 publish of `msg` (the coordinator's outbound decision record).
fn publish(ord: u64, seq: u64, msg: &VhcMessage) -> Record {
    let frame = wire_frame([0xC0; 32], msg);
    Record::new(
        ord,
        Body::Publish(PublishRec {
            channel: 0,
            seq,
            hash: blake3_hash(&frame),
            frame,
        }),
    )
}

/// A tag-12 inbound signed frame carrying a peer's `Digest` for `round`.
fn digest_frame(ord: u64, seq: u64, sender: [u8; 32], round: u64, digest: [u8; 16]) -> Record {
    let msg = VhcMessage::Digest(Digest {
        round,
        digest: StateDigest(digest),
    });
    Record::new(
        ord,
        Body::SignedFrame(SignedFrameRec {
            channel: 0,
            seq,
            sender: Hash(sender),
            frame: Some(wire_frame(sender, &msg)),
            evidence: None,
        }),
    )
}

// -- GREEN: replay-forward duplication is legal and counted, never conflated -----------------

/// Case 4 of the certification matrix: a successor chain re-publishing its predecessor's
/// retained records (replay-forward) produces IDENTICAL duplicates — they deduplicate, count
/// once, and the fold stays GREEN with the full continuous round story.
#[test]
fn identical_duplicate_round_records_deduplicate_and_count_once() {
    let r0 = round_record(0, 1);
    let r1 = round_record(1, 1);
    let records = vec![
        publish(0, 0, &r0),
        publish(1, 1, &r1),
        // The successor span's replay-forward re-publish of BOTH.
        publish(2, 2, &r0),
        publish(3, 3, &r1),
    ];
    let fold = semantic_fold(&records).expect("identical duplicates are legal");
    assert_eq!(fold.records.len(), 2, "each round counted once");
    assert_eq!(fold.duplicates_deduped, 2, "both re-publishes deduplicated");
    assert_eq!(
        fold.records.iter().map(|r| r.round).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

/// Identical duplicate peer digests (a rejoining trainer re-sending) collapse; the per-peer
/// transcript keeps one value per round.
#[test]
fn identical_duplicate_peer_digests_deduplicate() {
    let records = vec![
        publish(0, 0, &round_record(0, 1)),
        digest_frame(1, 0, [0xA1; 32], 0, [7; 16]),
        digest_frame(2, 1, [0xA1; 32], 0, [7; 16]),
    ];
    let fold = semantic_fold(&records).expect("identical digests are legal");
    assert_eq!(fold.digest_duplicates_deduped, 1);
    let peer = PeerId([0xA1; 32]);
    assert_eq!(fold.by_peer[&peer][&0], [7; 16]);
}

// -- RED: equivocation / continuity / digest conflicts ----------------------------------------

/// Case 11: two DIFFERENT `RoundRecord`s published for one round = equivocation, RED.
#[test]
fn two_different_round_records_for_one_round_refuse_as_equivocation() {
    let records = vec![
        publish(0, 0, &round_record(0, 1)),
        publish(1, 1, &round_record(0, 2)),
    ];
    let err = semantic_fold(&records).expect_err("equivocation is RED");
    assert!(
        matches!(err, SemanticFoldError::Equivocation { round: 0 }),
        "typed equivocation at the offending round, got: {err}"
    );
}

/// Case 12a: the deduplicated committed rounds must be dense `0..=max` — a skip is RED.
#[test]
fn a_skipped_committed_round_refuses_as_continuity() {
    let records = vec![
        publish(0, 0, &round_record(0, 1)),
        publish(1, 1, &round_record(2, 1)),
    ];
    let err = semantic_fold(&records).expect_err("a skipped round is RED");
    match err {
        SemanticFoldError::Continuity { detail } => {
            assert!(
                detail.contains("round 1"),
                "names the missing round: {detail}"
            )
        }
        other => panic!("expected a continuity refusal, got: {other}"),
    }
}

/// Case 12b: a lineage whose first committed round is not 0 is RED (the story must be whole).
#[test]
fn a_lineage_starting_past_round_zero_refuses_as_continuity() {
    let records = vec![publish(0, 0, &round_record(1, 1))];
    let err = semantic_fold(&records).expect_err("a truncated-front story is RED");
    assert!(matches!(err, SemanticFoldError::Continuity { .. }));
}

/// Case 12c: a NEW (non-duplicate) record for an earlier round appearing after a later one is a
/// regression — a successor re-derived a DIFFERENT past (its identical re-publish would have
/// deduplicated instead).
#[test]
fn a_regressing_first_occurrence_refuses_as_continuity() {
    let records = vec![
        publish(0, 0, &round_record(1, 1)),
        publish(1, 1, &round_record(0, 1)),
    ];
    let err = semantic_fold(&records).expect_err("a regression is RED");
    assert!(matches!(err, SemanticFoldError::Continuity { .. }));
}

/// Case 13: one `(peer, round)`, two DIFFERENT state digests = RED (the silent-overwrite
/// defect this fold replaces: last-write-wins in the archive assembler).
#[test]
fn conflicting_peer_digests_for_one_round_refuse_typed() {
    let records = vec![
        publish(0, 0, &round_record(0, 1)),
        digest_frame(1, 0, [0xA1; 32], 0, [7; 16]),
        digest_frame(2, 1, [0xA1; 32], 0, [8; 16]),
    ];
    let err = semantic_fold(&records).expect_err("a digest conflict is RED");
    match err {
        SemanticFoldError::DigestConflict { peer, round } => {
            assert_eq!(peer, PeerId([0xA1; 32]));
            assert_eq!(round, 0);
        }
        other => panic!("expected a digest-conflict refusal, got: {other}"),
    }
}
