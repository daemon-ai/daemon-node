// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The genesis-authoring cadence↔retention gate (D-SF3, ABI §12.14 [SF-5]): a remote checkpoint
//! cadence that could strand a rejoiner past the payload-retention floor is a TYPED authoring
//! refusal — the enforcement locus, since the host cannot decode the trainer's opaque module
//! config at admission. The ceremony-tier genesis inherits this gate against explicit inputs.

use daemon_vhc_testkit::genesis_run::author_check_checkpoint_cadence;

#[test]
fn authoring_refuses_a_cadence_that_strands_a_rejoiner() {
    // remote cadence 20 + one publisher-churn slot (20) needs retention >= 40.
    author_check_checkpoint_cadence(20, 40).expect("cadence + churn slot fits retention");
    author_check_checkpoint_cadence(10, 100).expect("comfortable margin");

    // 39 < 40: a rejoiner replaying from the freshest reachable remote checkpoint could fall off
    // the retention floor — refused at authoring, typed.
    let err = author_check_checkpoint_cadence(20, 39)
        .expect_err("a stranding cadence is refused at authoring");
    assert!(
        err.contains("genesis authoring refused") && err.contains("payload_retention_rounds"),
        "the refusal names the authoring gate + the retention bound: {err}"
    );

    // Unbounded retention (0) and disabled remote publication (0) are unconstrained.
    author_check_checkpoint_cadence(20, 0).expect("unbounded retention: no constraint");
    author_check_checkpoint_cadence(0, 8).expect("remote publication disabled: nothing to bound");
}
