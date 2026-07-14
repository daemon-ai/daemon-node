// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Stub swarm end-to-end (spec §6.4; the Merge-2 P0 milestone test).
//!
//! N = 3 peers + the **real** `daemon-swarm-coordinator` pure `tick` loop (signed + published by the
//! harness shell) drive the full round protocol over `LoopbackGossip` + a shared `FsPayloadStore` +
//! the deterministic `StubBackend` for 20 rounds:
//!
//! - every peer's post-ingest digest is **equal every round** (the §5.6 agree-path);
//! - one peer stalls round 7 (an injected payload miss) and catches up round 8 (§6.4 stall ladder);
//! - two runs of the same config produce a **byte-identical** digest transcript (determinism);
//! - the coordinator's `tick` trajectory replays byte-identically from its recorded inputs (I1 /
//!   PROTO-20 spirit).
//!
//! This is the P0 milestone: the swap from R2's TEST-ONLY scripted coordinator to the real tick
//! loop landed at Merge 2.

use daemon_swarm_proto::peer_id;
use daemon_swarm_run::engine::EngineEvent;
use daemon_swarm_run::harness::{peer_key, run_swarm, verify_observe_dir, StallFault, SwarmConfig};

/// The 20-round, 3-peer scenario with a stall at round 7 and catch-up at round 8.
fn scenario() -> SwarmConfig {
    SwarmConfig {
        num_rounds: 20,
        fault: Some(StallFault {
            // Peer 1 cannot fetch peer 0's round-7 payload for its first 2 gets (prefetch +
            // barrier), stalls, and catches up on its next attempt (round 8 open).
            peer_index: 1,
            missing_peer_index: 0,
            round: 7,
            first_n_gets: 2,
        }),
        ..SwarmConfig::small(20)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn twenty_rounds_all_agree_with_stall_and_catchup() {
    let run = run_swarm(scenario()).await.expect("swarm run");

    // No peer leaves; the stalling peer recovers within budget.
    assert!(run.left_peers().is_empty(), "no peer should leave the run");

    // Every round's post-ingest digest is shared by all three peers.
    assert!(run.all_agree(), "all peers agree on every round's digest");
    let by_round = run.digests_by_round();
    assert_eq!(by_round.len(), 20, "20 rounds ingested");
    for (round, peers) in &by_round {
        assert_eq!(
            peers.len(),
            3,
            "all 3 peers report a digest for round {round}"
        );
    }

    // The stall ladder: peer 1 straggles round 7 and catches it up (its round-7 digest still
    // matches the others', proving the late ingest reconverges).
    let stalled = peer_id(&peer_key(1));
    let stalled_events: Vec<&EngineEvent> = run
        .events
        .iter()
        .filter(|(p, _)| *p == stalled)
        .map(|(_, e)| e)
        .collect();
    assert!(
        stalled_events
            .iter()
            .any(|e| matches!(e, EngineEvent::Straggling { round: 7, .. })),
        "peer 1 straggles round 7"
    );
    assert!(
        stalled_events
            .iter()
            .any(|e| matches!(e, EngineEvent::CaughtUp { round: 7, .. })),
        "peer 1 catches up round 7"
    );

    // The coordinator's pure `tick` trajectory replays byte-identically from its recorded inputs
    // (canonical-CBOR state per round; I1 / PROTO-20 spirit).
    let replay = run.replay.as_ref().expect("coordinator replay captured");
    assert_eq!(replay.recorded_rounds(), 20, "a state snapshot per round");
    assert!(
        replay.verify(),
        "replaying tick over the recorded inputs reproduces the state trajectory"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn digest_transcript_is_byte_identical_across_runs() {
    // Same seeds → byte-identical agreed digest transcript (determinism, incl. the stall path).
    let a = run_swarm(scenario()).await.expect("run a");
    let b = run_swarm(scenario()).await.expect("run b");
    assert!(a.all_agree() && b.all_agree());

    let ta = a.agreed_transcript();
    let tb = b.agreed_transcript();
    assert_eq!(ta.len(), 20);
    assert_eq!(
        ta, tb,
        "the digest transcript must be reproducible byte-for-byte"
    );
}

/// B2: `--observe` records the run (message log + replay capture) and `swarm-replay` re-derives every
/// round record byte-identically (`verify_observe_dir`) — the gate-ceremony record + replay path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn observe_record_and_replay_green() {
    let run = run_swarm(scenario()).await.expect("swarm run");
    assert!(
        run.all_agree(),
        "peers agree so the digest tally is unanimous"
    );

    // The observe message log captured every round record on the wire.
    assert_eq!(
        run.message_log
            .by_kind(daemon_swarm_observe::MessageKind::RoundRecord)
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

    // Write the artifacts, then replay + verify them offline (what `swarm-replay <dir>` does).
    let dir = std::env::temp_dir().join(format!(
        "daemon-swarm-observe-e2e-{}-{}",
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
