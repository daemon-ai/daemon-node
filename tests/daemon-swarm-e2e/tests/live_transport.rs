// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
//! The **live-transport exit gate** (spec §6.4, §7.1; TDD §3.8, RUN-5/8, PROTO-20 over a real
//! plane; B3). Every test here drives the frozen `RoundEngine` + the real `daemon-swarm-coordinator`
//! `tick` loop over a **real per-node `IrohGossip` mesh** (QUIC gossip on loopback, explicit roster
//! addressing, no public discovery) + a shared `FsPayloadStore` — the transports swapped behind the
//! frozen `ControlPlane`/`PayloadStore` traits, nothing else changed. This is where B2's
//! plane-level partition/rejoin proof becomes an **engine-level** recovery proof: the stall ladder,
//! late-join resync, and mid-run drop all recover over the real gossip mesh.
//!
//! Gated behind the `iroh` feature so the default e2e gate stays fast + iroh-free; run with
//! `cargo test -p daemon-swarm-e2e --features iroh`. The relay variant additionally shells the
//! devShell `iroh-relay --dev` binary and skips cleanly when it is absent.
#![cfg(feature = "iroh")]
// The relay-path test shells the dev relay binary (a known dev tool from a test — the sanctioned
// exception to the workspace `Command` ban, mirroring B2's `spawn_dev_relay`).
#![allow(clippy::disallowed_methods)]

use std::time::Duration;

use daemon_swarm_proto::peer_id;
use daemon_swarm_run::engine::EngineEvent;
use daemon_swarm_run::harness::{peer_key, LateJoin, SilentDeath, StallFault, SwarmConfig};
use daemon_swarm_run::live_harness::{run_live_swarm, LiveSwarmConfig};

/// Assert every round in `run` has a single digest shared by all peers that reported it.
fn assert_all_agree(run: &daemon_swarm_run::harness::SwarmRun) {
    assert!(
        run.all_agree(),
        "surviving peers must agree on every round's digest over the live plane"
    );
}

// ---- flagship: 3 peers × ≥10 rounds, real iroh mesh, byte-identical per-round digests -----------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_flagship_three_peers_ten_rounds_all_agree() {
    // 3 peers form a real iroh gossip mesh on loopback, run 12 rounds over a shared FS store with
    // the concurrent barrier fetch enabled, agree on every round's post-ingest digest, and the run
    // terminates on the envelope stop condition (`Rounds(12)`).
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 12,
        ..SwarmConfig::small(12)
    });
    let run = run_live_swarm(cfg).await.expect("live flagship run");

    assert!(run.left_peers().is_empty(), "no peer should leave");
    assert_all_agree(&run);

    let by_round = run.digests_by_round();
    assert_eq!(
        by_round.len(),
        12,
        "12 rounds ingested over the live mesh (stop condition honored)"
    );
    for (round, peers) in &by_round {
        assert_eq!(peers.len(), 3, "all 3 peers report round {round}");
    }

    // PROTO-20 over a live-transport run: the coordinator's tick trajectory replays byte-identically
    // from its recorded inputs.
    let replay = run.replay.as_ref().expect("coordinator replay captured");
    assert_eq!(replay.recorded_rounds(), 12, "a state snapshot per round");
    assert!(
        replay.verify(),
        "the live-run tick trajectory replays byte-identically"
    );
}

// ---- stall ladder over the real plane (RUN-8, engine-level recovery) ----------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_stall_ladder_recovers_over_iroh() {
    // Peer 1 cannot fetch peer 0's round-5 payload for its first 2 gets (prefetch + barrier), stalls
    // over the real mesh, and catches it up on a later round — the ENGINE-level recovery of the
    // gossip-delivery gap B2 proved at the plane level. Concurrent fetch OFF so the injected-miss
    // count matches the barrier semantics.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 12,
        stall_rounds_max: 3,
        fault: Some(StallFault {
            peer_index: 1,
            missing_peer_index: 0,
            round: 5,
            first_n_gets: 2,
        }),
        ..SwarmConfig::small(12)
    })
    .with_concurrent_fetch(false);
    let run = run_live_swarm(cfg).await.expect("live stall run");

    assert!(
        run.left_peers().is_empty(),
        "the stall ladder absorbs the miss within budget"
    );
    assert_all_agree(&run);

    let stalled = peer_id(&peer_key(1));
    let mine: Vec<&EngineEvent> = run
        .events
        .iter()
        .filter(|(p, _)| *p == stalled)
        .map(|(_, e)| e)
        .collect();
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::Straggling { round: 5, .. })),
        "peer 1 straggles round 5 over the live plane"
    );
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::CaughtUp { round: 5, .. })),
        "peer 1 catches round 5 up over the live plane"
    );
    assert_eq!(
        run.digests_by_round()[&5].len(),
        3,
        "all 3 peers report round 5"
    );
}

// ---- late-join resync over the real plane (§6.4 admission + §9 rejoin) ---------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_late_join_resyncs_over_iroh() {
    // Epoch 0 = rounds 0..2 (3 peers); a 4th peer joins epoch 1 through the real admission path over
    // the live mesh, resyncs from the round-2 checkpoint (pulled from the shared store), and
    // contributes rounds 3..5 with consensus digests.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 6,
        steps_per_round: 1,
        micro_batch: 1,
        epoch_rounds: 3,
        checkpoint_every_rounds: 3, // checkpoint at round 2 (and 5)
        min_peers: Some(3),
        late_join: Some(LateJoin { resume_round: 2 }),
        ..SwarmConfig::small(6)
    });
    let run = run_live_swarm(cfg).await.expect("live late-join run");

    assert_all_agree(&run);
    assert!(run.left_peers().is_empty(), "no peer leaves");

    let late = peer_id(&peer_key(3));
    let by_round = run.digests_by_round();
    for round in 3..6 {
        let peers = by_round.get(&round).expect("round reported");
        assert!(
            peers.contains_key(&late),
            "the late peer reports round {round}"
        );
        assert_eq!(peers.len(), 4, "all 4 peers report round {round}");
    }
    let quorum = run.quorum_digests();
    for round in 3..6 {
        assert_eq!(
            by_round[&round][&late], quorum[&round],
            "late peer's round-{round} digest matches consensus (resync correct)"
        );
    }
}

// ---- mid-run drop over the real plane (§6.4, hard peer death) ------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_mid_run_drop_dropped_after_absences() {
    // Peer 2 goes silent after round 2 (its engine stops publishing over its iroh node). With
    // k_absences = 2 the coordinator drops it; min_peers = 1 keeps the run alive on peers 0 and 1.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 8,
        min_peers: Some(1),
        k_absences: 2,
        silent_death: Some(SilentDeath {
            peer_index: 2,
            after_round: 2,
        }),
        ..SwarmConfig::small(8)
    });
    let run = run_live_swarm(cfg).await.expect("live drop run");

    assert_all_agree(&run);
    let dead = peer_id(&peer_key(2));
    assert!(
        run.dropped_peers().contains(&dead),
        "the silent peer is dropped after K record-absences, got {:?}",
        run.dropped_peers()
    );
    let by_round = run.digests_by_round();
    let last = &by_round[&7];
    assert_eq!(last.len(), 2, "two survivors report the last round");
    assert!(
        !last.contains_key(&dead),
        "the dropped peer contributes nothing past its death"
    );
}

// ---- relay variant: route through B2's self-hosted dev relay (skip-clean if absent) --------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_run_through_self_hosted_relay() {
    // Spawn the devShell `iroh-relay --dev` (plain HTTP, default port 3340 — B2's dev runner) and
    // run a small flagship with the nodes configured for `RelayMode::Custom(<that relay>)`. Skips
    // cleanly when the binary is not on PATH (a standalone checkout without the devShell). On
    // loopback the direct path dominates (B2 finding 1), so the run is robust even if the relay port
    // is busy — the value here is exercising the relay-configured construction end to end.
    let Some(mut relay) = spawn_dev_relay() else {
        eprintln!("SKIP live_run_through_self_hosted_relay: iroh-relay not on PATH");
        return;
    };
    // Give the relay a moment to bind its HTTP listener.
    tokio::time::sleep(Duration::from_millis(800)).await;

    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 6,
        ..SwarmConfig::small(6)
    })
    .with_relay("http://127.0.0.1:3340");
    let result = run_live_swarm(cfg).await;

    let _ = relay.kill();
    let run = result.expect("live relay run");
    assert_all_agree(&run);
    assert_eq!(
        run.digests_by_round().len(),
        6,
        "6 rounds ingested through the relay-configured mesh"
    );
}

/// Spawn `iroh-relay --dev` (plain HTTP, default port 3340), returning the child (or `None` if the
/// binary is absent — the skip-clean path for a standalone checkout).
fn spawn_dev_relay() -> Option<std::process::Child> {
    std::process::Command::new("iroh-relay")
        .args(["--dev"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()
}
