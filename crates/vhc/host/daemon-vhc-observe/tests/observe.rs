// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-observe` behavior tests (spec §6.4/§14; TDD §3.9 + PROTO-20):
//! `log_roundtrip_canonical`, `replay_matches_recorded_run`, `replay_detects_tampered_record`,
//! `digest_quorum_flags_outlier`, plus run-health projection.
//!
//! Consensus is re-derived inside the real `coordinator-quorum` module (the sandbox), never a native
//! tick — both the recorded fixture and its replay drive the same content-addressed module .

// The sandbox shells `cargo build` for the guests workspace (the established testkit pattern).
#![allow(clippy::disallowed_methods)]

use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, PeerId, SigningKey, StateDigest, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::messages::{
    Digest, RoundRecord, SignedMessage, Straggle, StraggleStatus, VhcMessage,
};

use daemon_vhc_sdk_consensus::coordinator::Input;

use daemon_vhc_observe::desync::digest_tally_from_log;
use daemon_vhc_observe::{
    digest_tally, replay_capture, replay_from_state, MessageKind, MessageLog, ReplayError,
    RunCapture, RunHealth,
};

mod common;
use common::{coord_key, coordinator_sandbox, key, run_fixture, Fixture};

const RUN_ID: &str = "obs-run";

fn pid(seed: u8) -> PeerId {
    peer_id(&key(seed))
}

/// Sign the fixture's published `RoundRecord`s as the coordinator authority broadcasts them.
fn signed_records(fx: &Fixture, coord: &SigningKey) -> Vec<SignedMessage> {
    fx.records
        .iter()
        .map(|r| {
            SignedMessage::sign(coord, VHC_PROTO_VERSION, VhcMessage::RoundRecord(r.clone()))
                .expect("sign record")
        })
        .collect()
}

/// The node-visible wire log: the driving frames + the coordinator's published RoundRecords.
fn wire_log(fx: &Fixture, coord: &SigningKey) -> MessageLog {
    let mut log = MessageLog::new(&fx.run_id);
    for sm in &fx.driving {
        log.append(sm.clone());
    }
    for sm in signed_records(fx, coord) {
        log.append(sm);
    }
    log
}

/// The driving capture inputs (messages only; the module owns its clock, and its RoundRecords are
/// the oracle, re-supplied from the log).
fn driving_inputs(fx: &Fixture) -> Vec<Input> {
    fx.driving.iter().cloned().map(Input::Message).collect()
}

/// The resync anchor the report carries: blake3 of the module's published decision stream.
fn anchor(records: &[RoundRecord]) -> daemon_vhc_proto::Hash {
    blake3_hash(&to_canonical_vec(records).unwrap())
}

fn digest_msg(k: &SigningKey, round: u64, d: StateDigest) -> SignedMessage {
    let x = Digest { round, digest: d };
    SignedMessage::sign(k, VHC_PROTO_VERSION, VhcMessage::Digest(x)).unwrap()
}

fn straggle_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let s = Straggle {
        round,
        status: StraggleStatus::Stalled,
    };
    SignedMessage::sign(k, VHC_PROTO_VERSION, VhcMessage::Straggle(s)).unwrap()
}

// ----- OBS: message-log roundtrip -----

#[test]
fn log_roundtrip_canonical() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 3);
    let log = wire_log(&fx, &coord_key());
    assert!(!log.is_empty());

    // Write → read is lossless and preserves arrival order + run id.
    let mut bytes = Vec::new();
    log.write_to(&mut bytes).unwrap();
    let read = MessageLog::read_from(&mut bytes.as_slice()).unwrap();
    assert_eq!(read, log);
    assert_eq!(read.run_id(), RUN_ID);

    // Framing is canonical: a second write is byte-identical.
    let mut bytes2 = Vec::new();
    read.write_to(&mut bytes2).unwrap();
    assert_eq!(bytes, bytes2);

    // (round, kind) index: every round has exactly one published RoundRecord.
    for round in log.rounds() {
        let records = log.by_round_kind(round, MessageKind::RoundRecord).count();
        assert_eq!(records, 1, "one record per round");
    }
    assert_eq!(log.by_kind(MessageKind::RoundRecord).count(), 3);
    // Joins are roster-scoped (no round) — not returned by `by_round`.
    assert_eq!(
        log.by_round(0)
            .filter(|m| matches!(m.payload, VhcMessage::Join(_)))
            .count(),
        0
    );
}

// ----- OBS / PROTO-20: replay reproduces the recorded run (in the sandbox) -----

#[test]
fn replay_matches_recorded_run() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 3);
    let log = wire_log(&fx, &coord_key());

    let capture = RunCapture::new(fx.initial.clone(), driving_inputs(&fx));
    let report = replay_capture(&sandbox, capture, &log).expect("replay must reproduce the run");
    assert_eq!(report.rounds_verified, 3, "all recorded records re-derived");
    assert_eq!(report.records.len(), 3);
    // The module re-derived exactly the records the recorded run published.
    assert_eq!(report.records, fx.records);
    assert_eq!(report.final_state_hash, anchor(&fx.records));
}

// ----- OBS: RunCapture replays a recorded run (the --observe / replay path) -----

#[test]
fn run_capture_replays_recorded_run() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 3);
    let log = wire_log(&fx, &coord_key());

    let capture = RunCapture::new(fx.initial.clone(), driving_inputs(&fx));

    // The capture round-trips through its on-disk framing byte-identically.
    let mut bytes = Vec::new();
    capture.write_to(&mut bytes).unwrap();
    let read = RunCapture::read_from(&mut bytes.as_slice()).unwrap();
    assert_eq!(read, capture);

    // replay_capture re-derives every logged RoundRecord byte-identically (digest equality).
    let report = replay_capture(&sandbox, capture, &log).expect("recorded run must re-derive");
    assert_eq!(
        report.rounds_verified, 3,
        "all 3 recorded records re-derived"
    );
    assert_eq!(report.final_state_hash, anchor(&fx.records));

    // replay_from_state over driving-inputs-only re-derives the records with nothing to compare
    // (no oracle in the stream) — it still re-derives 3 records, verifying 0.
    let bare = replay_from_state(
        &sandbox,
        fx.initial.clone(),
        driving_inputs(&fx).into_iter(),
    )
    .expect("bare replay");
    assert_eq!(bare.records.len(), 3);
    assert_eq!(bare.rounds_verified, 0);
    assert_eq!(bare.final_state_hash, anchor(&fx.records));
}

// ----- OBS: a tampered record is caught, first-divergence pinpointed -----

#[test]
fn replay_detects_tampered_record() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 3);
    let coord = coord_key();

    // Tamper the round-0 record in the wire log (the oracle): claim a spurious drop, re-sign so the
    // frame is still valid — only the consensus content diverges from what the module re-derives.
    let mut log = MessageLog::new(RUN_ID);
    for sm in &fx.driving {
        log.append(sm.clone());
    }
    for (i, r) in fx.records.iter().enumerate() {
        let mut rec = r.clone();
        if i == 0 {
            rec.drops.push(pid(9));
        }
        log.append(
            SignedMessage::sign(&coord, VHC_PROTO_VERSION, VhcMessage::RoundRecord(rec)).unwrap(),
        );
    }

    let capture = RunCapture::new(fx.initial.clone(), driving_inputs(&fx));
    match replay_capture(&sandbox, capture, &log) {
        Err(ReplayError::Diverged(d)) => {
            assert_eq!(d.round, 0);
            assert!(d.rederived.is_some());
            assert!(d.recorded.drops.contains(&pid(9)));
            assert!(!d.rederived.unwrap().drops.contains(&pid(9)));
        }
        other => panic!("expected a divergence at round 0, got {other:?}"),
    }
}

// ----- OBS: digest tally flags the outlier -----

#[test]
fn digest_quorum_flags_outlier() {
    let good = StateDigest([0xAA; 16]);
    let bad = StateDigest([0xBB; 16]);
    let reports = vec![(pid(1), good), (pid(2), good), (pid(3), bad)];

    let verdict = digest_tally(5, reports, 2); // quorum = 2
    assert_eq!(verdict.quorum_digest, Some(good));
    assert_eq!(verdict.outliers, vec![pid(3)]);
    assert_eq!(verdict.reporters, 3);
    assert!(!verdict.agreed);
    assert!(verdict.is_desync());

    // Full agreement → no outliers, not a desync.
    let all_good = vec![(pid(1), good), (pid(2), good), (pid(3), good)];
    let ok = digest_tally(5, all_good, 2);
    assert!(ok.agreed);
    assert!(ok.outliers.is_empty());
    assert!(!ok.is_desync());

    // Same, folded straight from a message log.
    let mut log = MessageLog::new(RUN_ID);
    log.append(digest_msg(&key(1), 5, good));
    log.append(digest_msg(&key(2), 5, good));
    log.append(digest_msg(&key(3), 5, bad));
    let from_log = digest_tally_from_log(&log, 5, 2);
    assert_eq!(from_log.quorum_digest, Some(good));
    assert_eq!(from_log.outliers, vec![pid(3)]);
    assert!(from_log.is_desync());
}

// ----- OBS: run-health projection -----

#[test]
fn run_health_projects_per_round_facts() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 2);
    let mut log = wire_log(&fx, &coord_key());

    // Add per-round observability messages the coordinator run doesn't itself emit.
    let good = StateDigest([7; 16]);
    for r in 0..2u64 {
        log.append(digest_msg(&key(2), r, good));
        log.append(digest_msg(&key(3), r, good));
    }
    log.append(straggle_msg(&key(3), 1));

    let health = RunHealth::from_log(&log);
    assert_eq!(health.run_id, RUN_ID);
    assert_eq!(health.rounds.len(), 2);
    for rh in &health.rounds {
        assert_eq!(rh.committed, 2, "both peers committed + evidenced");
        assert!(rh.finalized);
        assert!(rh.drops.is_empty());
        assert_eq!(rh.digest_reporters, 2);
        assert!(rh.digest_agreed);
    }
    assert_eq!(health.rounds[1].stragglers, vec![pid(3)]);
    assert!(health.rounds[0].stragglers.is_empty());
}
