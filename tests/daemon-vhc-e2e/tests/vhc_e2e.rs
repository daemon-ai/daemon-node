// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Stub vhc end-to-end record/replay (spec §6.4).
//!
//! N = 3 peers over `LoopbackGossip` + a shared `FsPayloadStore` + the deterministic
//! `StubBackend`, coordinated by the production `coordinator-quorum` module (the harness's
//! wasm-coordinator recording drive — consensus never runs natively, in recording or
//! verification) for 20 rounds with a stall at round 7 and catch-up at round 8 (§6.4 stall
//! ladder). The recorded run is then re-derived offline through the SAME content-addressed
//! module (`verify_observe_dir`) — the gate-ceremony record + replay path.

use daemon_vhc_session::harness::{run_vhc, verify_observe_dir, StallFault, VhcConfig};

/// The 20-round, 3-peer scenario with a stall at round 7 and catch-up at round 8.
fn scenario() -> VhcConfig {
    VhcConfig {
        num_rounds: 20,
        fault: Some(StallFault {
            // Peer 1 cannot fetch peer 0's round-7 payload for its first 2 gets (prefetch +
            // barrier), stalls, and catches up on its next attempt (round 8 open).
            peer_index: 1,
            missing_peer_index: 0,
            round: 7,
            first_n_gets: 2,
        }),
        ..VhcConfig::small(20)
    }
}

/// `--observe` records the run (message log + replay capture) and `replay` re-derives every
/// round record byte-identically (`verify_observe_dir`) — the gate-ceremony record + replay path.
///
/// The run is recorded by driving the production `coordinator-quorum` module — the SAME
/// content-addressed module `verify_observe_dir` re-derives through, on the SAME
/// one-tick-per-frame logical clock. So the captured driving trace is reproducible from its
/// frames alone and the record/replay is green end to end (consensus never runs natively, in
/// recording or verification).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn observe_record_and_replay_green() {
    let run = run_vhc(scenario()).await.expect("vhc run");
    assert!(
        run.all_agree(),
        "peers agree so the digest tally is unanimous"
    );

    // The observe message log captured every round record on the wire.
    assert_eq!(
        run.message_log
            .by_kind(daemon_vhc_observe::MessageKind::RoundRecord)
            .count(),
        20,
        "one round record per round on the wire"
    );
    // Digest tally over the peers' reported digests shows unanimous agreement, no desync outliers.
    for round in 0..20u64 {
        assert!(
            run.desync_outliers(round).is_empty(),
            "round {round} has no desync outlier"
        );
    }

    // Write the artifacts, then replay + verify them offline (what `replay <dir>` does).
    let dir = std::env::temp_dir().join(format!(
        "daemon-vhc-observe-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    run.write_observe(&dir).expect("write observe artifacts");

    let report = verify_observe_dir(&dir).expect("replay must re-derive the recorded run");
    assert!(
        report.all_verified(),
        "all recorded round records re-derive ({}/{})",
        report.rounds_verified,
        report.logged_records
    );
    assert_eq!(report.rounds_verified, 20, "20 rounds re-derived");
    assert_eq!(
        report.health.rounds.len(),
        20,
        "run health projects 20 rounds"
    );
    assert!(
        report.health.rounds.iter().all(|r| r.finalized),
        "every round finalized in the health projection"
    );
}
