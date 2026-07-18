// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator-oracle migration onto the journal substrate (refactor §5 A1 acceptance): the
//! replay oracle passes **unchanged in semantics** over journal-backed capture. This test drives a
//! real coordinator run (through the sandboxed `coordinator-quorum` module), replays it both
//! the in-memory way ([`replay_capture`]) and over the crash-safe segmented [`Journal`]
//! ([`replay_over_journal`]), and asserts the two [`ReplayReport`]s are byte-identical — proving the
//! oracle's pinned behavior is preserved on the new substrate.

// Uses a throwaway temp dir for the journal fixtures; test-scoped fs allow (Phase-4 guardrail
// targets production paths).
#![allow(clippy::disallowed_methods)]

use std::sync::atomic::{AtomicU64, Ordering};

use daemon_vhc_proto::{Hash, VHC_PROTO_VERSION};
use daemon_vhc_sdk_consensus::messages::{SignedMessage, VhcMessage};

use daemon_vhc_sdk_consensus::coordinator::Input;

use daemon_vhc_observe::journal::oracle::{record_capture, replay_over_journal};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::sidecar::StaticKey;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::{replay_capture, MessageLog, RunCapture};

mod common;
use common::{coord_key, coordinator_sandbox, run_fixture};

const RUN_ID: &str = "oracle-journal-run";

fn tempdir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-journal-oracle-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: Hash([0xC0; 32]),
        epoch: 0,
        role: "coordinator".into(),
        instance: 1,
        module: Hash([0xCD; 32]),
    }
}

/// The oracle re-derives the run identically over journal-backed capture as it does in-memory.
#[test]
fn oracle_parity_over_journal_substrate() {
    let sandbox = coordinator_sandbox();
    let fx = run_fixture(&sandbox, RUN_ID, 3);

    // The wire log: every signed message (the driving frames + the coordinator's own published
    // RoundRecords, re-signed by the coordinator authority as it would broadcast them).
    let coord = coord_key();
    let mut log = MessageLog::new(RUN_ID);
    for sm in &fx.driving {
        log.append(sm.clone());
    }
    for r in &fx.records {
        let signed = SignedMessage::sign(
            &coord,
            VHC_PROTO_VERSION,
            VhcMessage::RoundRecord(r.clone()),
        )
        .expect("sign record");
        log.append(signed);
    }

    // The driving capture: initial state + driving inputs (no RoundRecord/RoundOpen — those are the
    // oracle, re-supplied from the log).
    let driving: Vec<Input> = fx.driving.iter().cloned().map(Input::Message).collect();
    let capture = RunCapture::new(fx.initial.clone(), driving);

    // In-memory oracle (the pinned path), re-derived inside the sandbox.
    let report_mem = replay_capture(&sandbox, capture.clone(), &log).expect("in-memory replay");
    assert_eq!(report_mem.rounds_verified, 3);

    // Journal-backed oracle: record the capture onto a crash-safe segmented journal, then replay.
    let dir = tempdir();
    let mut journal = Journal::create(
        dir.join("j"),
        ident(),
        StaticKey::new([5u8; 32]),
        RotatePolicy { max_records: 4 },
    )
    .unwrap();
    record_capture(&mut journal, &ident(), &capture).expect("record capture onto journal");
    let report_journal =
        replay_over_journal(&sandbox, &journal, &log).expect("journal-backed replay");

    // Byte-identical: same records, same rounds_verified, same decision-stream anchor (pinned).
    assert_eq!(
        report_journal, report_mem,
        "oracle parity over the journal substrate"
    );
    assert_eq!(report_journal.rounds_verified, 3);

    // And it survives a reopen (crash-safe substrate): reopening the journal replays identically.
    drop(journal);
    let reopened = Journal::open(
        dir.join("j"),
        ident(),
        StaticKey::new([5u8; 32]),
        RotatePolicy { max_records: 4 },
    )
    .unwrap();
    let report_reopened =
        replay_over_journal(&sandbox, &reopened, &log).expect("replay after reopen");
    assert_eq!(
        report_reopened, report_mem,
        "oracle parity persists across reopen"
    );
}
