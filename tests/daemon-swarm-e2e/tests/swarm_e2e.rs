// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Stub swarm end-to-end (spec §6.4; the Merge-2 P0 milestone test).
//!
//! N = 3 peers + a TEST-ONLY scripted coordinator drive the full round protocol over
//! `LoopbackGossip` + a shared `FsPayloadStore` + the deterministic `StubBackend` for 20 rounds:
//!
//! - every peer's post-ingest digest is **equal every round** (the §5.6 agree-path);
//! - one peer stalls round 7 (an injected payload miss) and catches up round 8 (§6.4 stall ladder);
//! - two runs of the same config produce a **byte-identical** digest transcript (determinism).
//!
//! `// MERGE-2: swap the scripted coordinator for the real daemon-swarm-coordinator tick loop` —
//! this becomes the P0 gate test unchanged except for that swap.

use daemon_swarm_proto::peer_id;
use daemon_swarm_run::engine::EngineEvent;
use daemon_swarm_run::harness::{peer_key, run_swarm, StallFault, SwarmConfig};

/// The 20-round, 3-peer scenario with a stall at round 7 and catch-up at round 8.
fn scenario() -> SwarmConfig {
    SwarmConfig {
        num_peers: 3,
        num_rounds: 20,
        steps_per_round: 2,
        micro_batch: 2,
        stall_rounds_max: 2,
        checkpoint_every_rounds: 0,
        corpus_seed: 0xDAE0_7E57,
        fault: Some(StallFault {
            // Peer 1 cannot fetch peer 0's round-7 payload for its first 2 gets (prefetch +
            // barrier), stalls, and catches up on its next attempt (round 8 open).
            peer_index: 1,
            missing_peer_index: 0,
            round: 7,
            first_n_gets: 2,
        }),
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
