// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// TDD §3.3 RUN-6/7 — checkpoint registration + desync→resync (spec §9, §6.4). Completes the named
// IDs on top of the P1 checkpoint save/load + replay fold: two-checkpointer both-match registration
// and single-uploader degraded mode (RUN-6), fp32-exact checkpoint roundtrip (RUN-6), the
// digest-mismatch → replay-resync recovery (RUN-7), and the retention-floor decision that sends a
// too-old desync to an epoch rejoin instead of a replay (RUN-7). RUN-10's staged-assess prescreen +
// manifest-cadence checks are unit-tested in `daemon_swarm_run::assess`.
//
// Oracle provenance (swarm-ledger-p2-b1.md): from-definition — the StubBackend outer step is
// deterministic + record-ordered, so resync replay recovering the in-sync digest is a bit-exact
// property; registration/retention are pure decision functions asserted directly.

use daemon_swarm_proto::assignment::elect_checkpointers;
use daemon_swarm_proto::{blake3_hash, PeerId, Seed};

use daemon_swarm_run::backend::{StagedPayload, StateDigest, StubBackend, TrainerBackend};
use daemon_swarm_run::checkpoint::{
    plan_resync, register_checkpoint, resync_by_replay, CheckpointManifest, CheckpointRegistration,
    ReplayStep, ResyncPlan,
};
use daemon_swarm_run::seam::RoundId;

fn pk(n: u8) -> PeerId {
    PeerId([n; 32])
}

fn staged(peer: u8, tag: &[u8]) -> StagedPayload {
    StagedPayload {
        peer: PeerId([peer; 32]),
        hash: blake3_hash(tag),
        bytes: tag.to_vec(),
    }
}

fn built(config: &[u8]) -> StubBackend {
    let mut b = StubBackend::new();
    b.build(config).unwrap();
    b
}

fn manifest(round: RoundId, tag: &[u8], digest: StateDigest) -> CheckpointManifest {
    CheckpointManifest {
        round,
        blake3: blake3_hash(tag),
        digest,
    }
}

// ===== RUN-6: two-checkpointer election + both-match registration + degraded ====================

#[test]
fn checkpoint_registers_on_both_match() {
    // The coordinator elects exactly two checkpointers, deterministically from (seed, roster).
    let roster: Vec<PeerId> = (0..6).map(pk).collect();
    let seed = Seed([0x9a; 32]);
    let checkpointers = elect_checkpointers(&roster, &seed, 2);
    assert_eq!(checkpointers.len(), 2, "two elected checkpointers");
    assert_ne!(checkpointers[0], checkpointers[1], "distinct checkpointers");
    // Order-independent + reproducible.
    let mut reversed = roster.clone();
    reversed.reverse();
    assert_eq!(elect_checkpointers(&reversed, &seed, 2), checkpointers);

    // Both upload byte-identical manifests → registered with cross-check.
    let digest = StateDigest([7; 16]);
    let a = manifest(4, b"ckpt-r4", digest);
    let b = manifest(4, b"ckpt-r4", digest);
    assert_eq!(
        register_checkpoint(&[a, b]),
        CheckpointRegistration::Registered(a)
    );

    // Divergent manifests (a faulty checkpointer) → rejected.
    let c = manifest(4, b"a-different-checkpoint", digest);
    assert_eq!(
        register_checkpoint(&[a, c]),
        CheckpointRegistration::Mismatch
    );
}

#[test]
fn single_uploader_degraded_flag() {
    // Only one checkpointer uploaded (the other churned) → registered, but flagged degraded (no
    // cross-check this round). No upload at all → missing.
    let digest = StateDigest([3; 16]);
    let only = manifest(7, b"solo", digest);
    assert_eq!(
        register_checkpoint(&[only]),
        CheckpointRegistration::Degraded(only)
    );
    assert_eq!(register_checkpoint(&[]), CheckpointRegistration::Missing);
}

#[test]
fn checkpoint_roundtrips_replicated_fp32() {
    // A checkpoint captures the canonical state (params + replicated persistents) exactly: save →
    // load into a fresh backend → re-save is byte-identical, and a further ingest from the reloaded
    // state matches what the original reaches (fp32-exact resume, §9).
    let mut a = built(b"cfg");
    a.ingest(0, &[staged(1, b"x"), staged(2, b"y")]).unwrap();

    let bytes = a.checkpoint_save().unwrap();
    let mut b = built(b"totally-different-config");
    b.checkpoint_load(&bytes).unwrap();

    // Re-serializing the reloaded state is byte-identical (exact roundtrip).
    assert_eq!(
        b.checkpoint_save().unwrap(),
        bytes,
        "checkpoint roundtrip is exact"
    );

    // And the reloaded backend reaches the identical next-round digest.
    let next = [staged(1, b"p"), staged(2, b"q")];
    assert_eq!(b.ingest(1, &next).unwrap(), a.ingest(1, &next).unwrap());
}

// ===== RUN-7: desync → resync-from-checkpoint =================================================

#[test]
fn digest_mismatch_triggers_replay_resync() {
    // A peer diverges (reordered round-1 set → wrong digest); the digest mismatch is the desync
    // trigger, and replaying the retained records from a checkpoint recovers the in-sync digest (I1).
    let s0 = [staged(1, b"a0"), staged(2, b"b0")];
    let s1 = [staged(1, b"a1"), staged(2, b"b1")];
    let s2 = [staged(1, b"a2"), staged(2, b"b2")];

    let mut good = built(b"cfg");
    good.ingest(0, &s0).unwrap();
    let checkpoint = good.checkpoint_save().unwrap(); // checkpoint after round 0
    good.ingest(1, &s1).unwrap();
    let target = good.ingest(2, &s2).unwrap();

    let mut bad = built(b"cfg");
    bad.ingest(0, &s0).unwrap();
    let diverged = bad.ingest(1, &[s1[1].clone(), s1[0].clone()]).unwrap();
    // The mismatch trigger: this peer's round-1 digest disagrees with the consensus one.
    assert_ne!(diverged, good_round1_digest(&s0, &s1), "desync detected");

    // Within retention → replay from the checkpoint (RUN-7 plan).
    match plan_resync(0, 2, 4) {
        ResyncPlan::ReplayFromCheckpoint { from_round, steps } => {
            assert_eq!((from_round, steps), (0, 2));
        }
        other => panic!("expected replay, got {other:?}"),
    }

    let recovered = resync_by_replay(
        &mut bad,
        &checkpoint,
        &[
            ReplayStep {
                round: 1,
                staged: s1.to_vec(),
            },
            ReplayStep {
                round: 2,
                staged: s2.to_vec(),
            },
        ],
    )
    .unwrap();
    assert_eq!(recovered, target, "replay recovers the in-sync digest");
}

/// The consensus round-1 digest (an in-sync reference peer) — the value a desynced peer compares to.
fn good_round1_digest(s0: &[StagedPayload], s1: &[StagedPayload]) -> StateDigest {
    let mut good = built(b"cfg");
    good.ingest(0, s0).unwrap();
    good.ingest(1, s1).unwrap()
}

#[test]
fn resync_beyond_retention_waits_for_epoch() {
    // A desync older than the payload-retention floor cannot replay (the records/payloads are gone),
    // so the peer waits for the next epoch checkpoint to rejoin.
    let retention = 2u64;
    // checkpoint at round 0, currently at round 5 → 5 rounds to replay > retention → wait.
    assert_eq!(plan_resync(0, 5, retention), ResyncPlan::WaitForEpoch);
    // Exactly at the floor → still replayable.
    assert_eq!(
        plan_resync(0, 2, retention),
        ResyncPlan::ReplayFromCheckpoint {
            from_round: 0,
            steps: 2
        }
    );
    // Already current (checkpoint at/after the round) → a zero-step replay, never a wait.
    assert_eq!(
        plan_resync(5, 5, retention),
        ResyncPlan::ReplayFromCheckpoint {
            from_round: 5,
            steps: 0
        }
    );
}
