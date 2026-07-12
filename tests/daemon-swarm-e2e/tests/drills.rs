// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Wave-3 churn / failure drills (spec §6.4, §13; TDD §3.8 E2E, RUN-7/8) over the local runner
//! machinery ([`daemon_swarm_run::harness`] + [`daemon_swarm_run::local_coordinator`]).
//!
//! Each drill injects one failure mode into the in-process N-peer swarm and asserts the run
//! **completes with all surviving peers' digests equal every round**:
//!
//! - [`late_join_mid_run_syncs_and_contributes`] — a peer joins at the epoch boundary via the real
//!   admission path, resyncs from the previous epoch's checkpoint, and contributes;
//! - [`hard_peer_death_dropped_after_absences`] — a peer goes silent (no `Straggle`) and is dropped
//!   after K record-absences; the run continues on the survivors;
//! - [`payload_store_outage_absorbed_by_stall_ladder`] — a peer's whole-round `get`s are denied for
//!   a window; the §6.4 stall ladder absorbs it;
//! - [`desync_injection_detected_and_resynced`] — one peer's ingest is corrupted, the digest outlier
//!   is detected by a local quorum-digest fold (`MERGE-3`: observe `DesyncVerdict`), and it is
//!   recovered via checkpoint + record replay;
//! - [`coordinator_restart_mid_run_completes`] — the coordinator shell reloads its `CoordinatorState`
//!   from canonical CBOR mid-run (PROTO-20 in anger) and the run completes.

use daemon_swarm_net::PayloadStore;
use daemon_swarm_proto::peer_id;
use daemon_swarm_run::backend::{
    AssessMeta, Assessment, BatchRef, StagedPayload, StateDigest, StepCtx, StepStats, StubBackend,
    TrainerBackend,
};
use daemon_swarm_run::checkpoint::{resync_by_replay, ReplayStep, CHECKPOINT_PEER};
use daemon_swarm_run::engine::EngineEvent;
use daemon_swarm_run::harness::{
    peer_key, run_swarm, run_swarm_with, LateJoin, SilentDeath, StoreOutage, SwarmConfig,
    EXPERIMENT_CONFIG,
};
use daemon_swarm_run::seam::{PayloadKey, RoundId};

/// Assert every round in `run` has a single digest shared by all peers that reported it.
fn assert_all_agree(run: &daemon_swarm_run::harness::SwarmRun) {
    assert!(
        run.all_agree(),
        "surviving peers must agree on every round's digest"
    );
}

// ----- drill 1: late join mid-run -----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn late_join_mid_run_syncs_and_contributes() {
    // Epoch 0 = rounds 0..2 (3 peers), epoch boundary at round 3; a 4th peer joins epoch 1 via the
    // real admission path, resyncs from the round-2 checkpoint, and contributes rounds 3..5.
    let cfg = SwarmConfig {
        num_peers: 3,
        num_rounds: 6,
        steps_per_round: 1,
        micro_batch: 1,
        epoch_rounds: 3,
        checkpoint_every_rounds: 3, // checkpoints at round 2 (and 5)
        min_peers: Some(3),
        late_join: Some(LateJoin { resume_round: 2 }),
        ..SwarmConfig::small(6)
    };
    let run = run_swarm(cfg).await.expect("late-join run");

    assert_all_agree(&run);
    assert!(run.left_peers().is_empty(), "no peer leaves");

    let late = peer_id(&peer_key(3));
    let by_round = run.digests_by_round();
    // The late peer contributes from the epoch boundary (round 3) through the last round.
    for round in 3..6 {
        let peers = by_round.get(&round).expect("round reported");
        assert!(
            peers.contains_key(&late),
            "the late peer reports a digest for round {round}"
        );
        assert_eq!(peers.len(), 4, "all 4 peers report round {round}");
    }
    // Its late digests match the consensus (it resynced from the checkpoint correctly).
    let quorum = run.quorum_digests();
    for round in 3..6 {
        assert_eq!(
            by_round[&round][&late], quorum[&round],
            "late peer's round-{round} digest matches consensus"
        );
    }
}

// ----- drill 2: hard peer death (silent) -----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hard_peer_death_dropped_after_absences() {
    // Peer 2 goes silent after round 2 (no Straggle). With k_absences = 2 the coordinator drops it;
    // min_peers = 1 keeps the run alive on peers 0 and 1 through the last round.
    let cfg = SwarmConfig {
        num_peers: 3,
        num_rounds: 8,
        min_peers: Some(1),
        k_absences: 2,
        silent_death: Some(SilentDeath {
            peer_index: 2,
            after_round: 2,
        }),
        ..SwarmConfig::small(8)
    };
    let run = run_swarm(cfg).await.expect("silent-death run");

    assert_all_agree(&run);

    let dead = peer_id(&peer_key(2));
    assert!(
        run.dropped_peers().contains(&dead),
        "the silent peer is dropped after K record-absences, got {:?}",
        run.dropped_peers()
    );

    // The survivors complete the final round.
    let by_round = run.digests_by_round();
    let last = &by_round[&7];
    assert_eq!(last.len(), 2, "two survivors report the last round");
    assert!(
        !last.contains_key(&dead),
        "the dead peer contributes nothing past its death"
    );
}

// ----- drill 3: payload-store outage window -----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payload_store_outage_absorbed_by_stall_ladder() {
    // Peer 1's whole-round get()s for round 4 are denied for a window, then the store recovers; the
    // stall ladder absorbs it and peer 1 catches up within budget.
    let cfg = SwarmConfig {
        num_peers: 3,
        num_rounds: 10,
        stall_rounds_max: 3,
        outage: Some(StoreOutage {
            peer_index: 1,
            round: 4,
            first_n_gets: 3,
        }),
        ..SwarmConfig::small(10)
    };
    let run = run_swarm(cfg).await.expect("store-outage run");

    assert_all_agree(&run);
    assert!(
        run.left_peers().is_empty(),
        "the stall ladder absorbs the outage"
    );

    let stalled = peer_id(&peer_key(1));
    let mine: Vec<&EngineEvent> = run
        .events
        .iter()
        .filter(|(p, _)| *p == stalled)
        .map(|(_, e)| e)
        .collect();
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::Straggling { round: 4, .. })),
        "peer 1 straggles round 4 during the outage"
    );
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::CaughtUp { round: 4, .. })),
        "peer 1 catches round 4 up once the store recovers"
    );
    // Round 4 is still reported by all three peers.
    assert_eq!(run.digests_by_round()[&4].len(), 3);
}

// ----- drill 4: desync injection + resync via checkpoint + replay -----

/// A [`TrainerBackend`] that corrupts its `ingest` at `corrupt_round` by reordering the staged set
/// (a consensus-input violation, §6.4 I3), diverging its state + digest from that round on.
struct DesyncBackend {
    inner: StubBackend,
    corrupt_round: RoundId,
}

impl TrainerBackend for DesyncBackend {
    type Error = <StubBackend as TrainerBackend>::Error;

    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error> {
        self.inner.build(config)
    }
    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error> {
        self.inner.assess(meta)
    }
    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error> {
        self.inner.train_step(batch, ctx)
    }
    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error> {
        self.inner.inner_update(inner_step)
    }
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error> {
        self.inner.make_update(round)
    }
    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error> {
        if round == self.corrupt_round && staged.len() >= 2 {
            // Stage in the WRONG order → a divergent fold → a divergent state + digest.
            let mut bad = staged.to_vec();
            bad.swap(0, 1);
            self.inner.ingest(round, &bad)
        } else {
            self.inner.ingest(round, staged)
        }
    }
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
        self.inner.checkpoint_save()
    }
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.inner.checkpoint_load(bytes)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn desync_injection_detected_and_resynced() {
    const CORRUPT: RoundId = 4;
    let cfg = SwarmConfig {
        num_peers: 3,
        num_rounds: 8,
        checkpoint_every_rounds: 1, // a checkpoint every round → round-3 checkpoint available
        ..SwarmConfig::small(8)
    };
    // Peer 2 desyncs at round 4; peers 0 and 1 stay healthy.
    let run = run_swarm_with(cfg, |i| {
        let mut inner = StubBackend::new();
        inner.build(EXPERIMENT_CONFIG).expect("build");
        DesyncBackend {
            inner,
            corrupt_round: if i == 2 { CORRUPT } else { u64::MAX },
        }
    })
    .await
    .expect("desync run");

    let healthy0 = peer_id(&peer_key(0));
    let healthy1 = peer_id(&peer_key(1));
    let desynced = peer_id(&peer_key(2));

    // The two healthy peers agree on every round.
    let by_round = run.digests_by_round();
    for (round, peers) in &by_round {
        assert_eq!(
            peers[&healthy0], peers[&healthy1],
            "healthy peers agree on round {round}"
        );
    }

    // Detection stand-in: a local quorum-digest fold flags the minority peer at the corrupt round.
    // `MERGE-3`: replace this fold with `daemon-swarm-observe`'s `DesyncVerdict`.
    let outliers = run.desync_outliers(CORRUPT);
    assert!(
        outliers.contains(&desynced),
        "the desynced peer is the round-{CORRUPT} digest outlier, got {outliers:?}"
    );
    assert!(
        !outliers.contains(&healthy0) && !outliers.contains(&healthy1),
        "the healthy peers are not flagged"
    );

    // Recovery: resync the outlier via the R2 checkpoint + record replay machinery. Reload the
    // round-(CORRUPT-1) checkpoint, replay round CORRUPT's committed set in record order, and confirm
    // it recovers the consensus (quorum) digest.
    let ckpt_round = CORRUPT - 1;
    let manifest = run
        .events
        .iter()
        .find_map(|(_, ev)| match ev {
            EngineEvent::Checkpointed { round, manifest } if *round == ckpt_round => {
                Some(*manifest)
            }
            _ => None,
        })
        .expect("a round-3 checkpoint was published");

    let ckpt_key = PayloadKey::new(run.run.clone(), ckpt_round, CHECKPOINT_PEER);
    let ckpt_bytes = run
        .store
        .get(&ckpt_key, &manifest.blake3)
        .await
        .expect("fetch checkpoint bytes");

    // Rebuild round CORRUPT's committed set (in record order) from the captured record + store.
    let entries = run.records.get(&CORRUPT).expect("round record captured");
    let mut staged = Vec::new();
    for e in entries {
        let key = PayloadKey::new(run.run.clone(), CORRUPT, e.peer);
        let bytes = run.store.get(&key, &e.hash).await.expect("fetch payload");
        staged.push(StagedPayload {
            peer: e.peer,
            hash: e.hash,
            bytes,
        });
    }

    let mut recovering = StubBackend::new();
    let recovered = resync_by_replay(
        &mut recovering,
        &ckpt_bytes,
        &[ReplayStep {
            round: CORRUPT,
            staged,
        }],
    )
    .expect("resync replay");

    let quorum = run.quorum_digests();
    assert_eq!(
        recovered, quorum[&CORRUPT],
        "checkpoint + record replay recovers the consensus round-{CORRUPT} digest"
    );
}

// ----- drill 5: coordinator restart mid-run -----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coordinator_restart_mid_run_completes() {
    // The coordinator shell reloads its CoordinatorState from canonical CBOR after round 4 (a
    // process restart, practical PROTO-20); the run completes and stays in agreement.
    let cfg = SwarmConfig {
        num_peers: 3,
        num_rounds: 10,
        restart_after_round: Some(4),
        ..SwarmConfig::small(10)
    };
    let run = run_swarm(cfg).await.expect("restart run");

    assert_all_agree(&run);
    assert!(
        run.left_peers().is_empty(),
        "no peer leaves across the restart"
    );

    let by_round = run.digests_by_round();
    assert_eq!(by_round.len(), 10, "all 10 rounds ingested");
    for (round, peers) in &by_round {
        assert_eq!(peers.len(), 3, "all 3 peers report round {round}");
    }

    let replay = run.replay.as_ref().expect("replay captured");
    assert_eq!(
        replay.reloads(),
        1,
        "the coordinator reloaded its state once"
    );
    assert_eq!(replay.recorded_rounds(), 10);
    assert!(
        replay.verify(),
        "the reloaded trajectory still replays byte-identically (PROTO-20)"
    );
}
