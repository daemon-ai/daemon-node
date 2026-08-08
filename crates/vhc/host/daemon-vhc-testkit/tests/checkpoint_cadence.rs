// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The genesis-authoring cadence gate (D-SF3, ABI §12.14 [SF-5]): a remote checkpoint cadence
//! that could strand a rejoiner — past the payload-retention floor, or past the retained RECORD
//! horizon replay-forward bridges (defect 7c of the c15 drills) — is a TYPED authoring refusal.
//! The enforcement locus is authoring, since the host cannot decode the trainer's opaque module
//! config at admission. The ceremony-tier genesis inherits this gate against explicit inputs.

use daemon_vhc_testkit::genesis_run::author_check_checkpoint_cadence;

#[test]
fn authoring_refuses_a_cadence_that_strands_a_rejoiner() {
    // remote cadence 4 + one publisher-churn slot (4) needs retention >= 8.
    author_check_checkpoint_cadence(4, 8).expect("cadence + churn slot fits retention");
    author_check_checkpoint_cadence(2, 100).expect("comfortable margin");

    // 7 < 8: a rejoiner replaying from the freshest reachable remote checkpoint could fall off
    // the retention floor — refused at authoring, typed.
    let err = author_check_checkpoint_cadence(4, 7)
        .expect_err("a stranding cadence is refused at authoring");
    assert!(
        err.contains("genesis authoring refused") && err.contains("payload_retention_rounds"),
        "the refusal names the authoring gate + the retention bound: {err}"
    );

    // Unbounded retention (0) and disabled remote publication (0) are unconstrained on the
    // payload lane (the record-horizon bound below still applies to any enabled cadence).
    author_check_checkpoint_cadence(4, 0).expect("unbounded retention: no constraint");
    author_check_checkpoint_cadence(0, 8).expect("remote publication disabled: nothing to bound");
}

/// Defects 7c (c15f) and 14 (c15h): the cadence plus one in-flight-upload slot must fit the
/// retained record horizon, or a crashed trainer restores a fence deeper than replay-forward
/// can bridge — every rejoin refuses `CheckpointStale`, churn recovery impossible by
/// construction (c15f: cadence 8 vs horizon 4; c15h: cadence 4 vs horizon 4 — zero slack, the
/// newest checkpoint upload was still in flight at the crash and the 2/2-quorum run wedged
/// permanently). Refused at authoring, regardless of retention.
#[test]
fn authoring_refuses_a_cadence_past_the_record_horizon() {
    let max = daemon_vhc_proto::det_state::RETAINED_RECORD_HORIZON_ROUNDS / 2;
    author_check_checkpoint_cadence(max, 64)
        .expect("a cadence at half the horizon (one slot of upload slack) authors");
    for retention in [0, 64] {
        let err = author_check_checkpoint_cadence(max + 1, retention)
            .expect_err("a cadence past half the record horizon is refused at authoring");
        assert!(
            err.contains("retained record horizon"),
            "the refusal names the record-horizon bound: {err}"
        );
    }
}
