// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Round-boundary checkpointing + desync-recovery replay (spec §9; TDD RUN-6/7 subset).
//!
//! A checkpoint is `TrainerBackend::checkpoint_save` bytes plus a [`CheckpointManifest`] — the
//! round, the blake3 of the bytes (content address, §9), and the post-round state digest (§5.6).
//! Checkpoints are stored on the payload plane under a reserved key ([`CHECKPOINT_PEER`]), so
//! [`save_checkpoint`] / [`load_checkpoint`] round-trip through the same [`PayloadStore`] the round
//! payloads use, blake3-verified on load.
//!
//! Desync recovery is **record replay** (§6.4 I1, §9): a peer whose post-round digest disagrees
//! with the consensus reloads the latest checkpoint and replays the retained `RoundRecord`s (their
//! root-verified committed sets) + payloads forward to the current round. [`resync_by_replay`] is
//! that pure fold — `checkpoint_load` then `ingest` each retained round in order — the offline
//! resync oracle. The *trigger* (this peer's digest vs the quorum/consensus digest) is
//! `daemon_swarm_observe::digest_tally` / `DesyncVerdict` (folded over the run's `Digest` messages,
//! §9) — consumed by the harness + drills, which drive this replay on `DesyncVerdict::is_desync()`;
//! the replay fold itself is here.

use std::sync::Arc;

use daemon_swarm_proto::{blake3_hash, Hash, PeerId};

use crate::backend::{StagedPayload, StateDigest, TrainerBackend};
use crate::seam::{PayloadKey, RoundId, RunId};
use daemon_swarm_net::PayloadStore;

use crate::SwarmRunError;

/// The reserved payload-plane peer id under which a run's checkpoints are stored (never a real node
/// identity — node pubkeys are ed25519 points, this sentinel is not).
pub const CHECKPOINT_PEER: PeerId = PeerId([0xCC; PeerId::LEN]);

/// The manifest of one checkpoint (§9): the round it captures, the blake3 of its bytes, and the
/// post-round state digest (§5.6) it should reproduce on reload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointManifest {
    /// The round this checkpoint captures (post-ingest state).
    pub round: RoundId,
    /// blake3 of the `checkpoint_save` bytes (content address).
    pub blake3: Hash,
    /// The post-round state digest this checkpoint reproduces.
    pub digest: StateDigest,
}

/// One replayed round: its committed set staged in record order (§6.4 I3), the input `ingest`
/// consumes during resync.
#[derive(Clone, Debug)]
pub struct ReplayStep {
    /// The round being replayed.
    pub round: RoundId,
    /// Its committed set, staged in record order.
    pub staged: Vec<StagedPayload>,
}

/// Save a round-boundary checkpoint: serialize the backend, content-address it, PUT it to the
/// payload plane, and return the [`CheckpointManifest`].
pub async fn save_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    backend: &B,
    round: RoundId,
    digest: StateDigest,
) -> Result<CheckpointManifest, SwarmRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let bytes = backend
        .checkpoint_save()
        .map_err(|e| SwarmRunError::Lifecycle(format!("checkpoint_save: {e}")))?;
    let blake3 = blake3_hash(&bytes);
    let key = PayloadKey::new(run.clone(), round, CHECKPOINT_PEER);
    store.put(&key, &bytes).await?;
    Ok(CheckpointManifest {
        round,
        blake3,
        digest,
    })
}

/// Load a checkpoint named by `manifest` from the payload plane (blake3-verified against the
/// manifest) into `backend`.
pub async fn load_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    backend: &mut B,
    manifest: &CheckpointManifest,
) -> Result<(), SwarmRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let key = PayloadKey::new(run.clone(), manifest.round, CHECKPOINT_PEER);
    let bytes = store.get(&key, &manifest.blake3).await?;
    backend
        .checkpoint_load(&bytes)
        .map_err(|e| SwarmRunError::Lifecycle(format!("checkpoint_load: {e}")))?;
    Ok(())
}

/// Desync recovery (§9, I1): reload `checkpoint_bytes` into `backend`, then replay `steps` forward
/// (each round's committed set → `ingest`), returning the final post-replay digest.
///
/// A pure fold of `(checkpoint, records, payloads)` — the resync oracle. Since ingest is
/// deterministic and record-ordered, the replayed digest equals the digest an in-sync peer reached
/// (the property this recovers to).
pub fn resync_by_replay<B: TrainerBackend>(
    backend: &mut B,
    checkpoint_bytes: &[u8],
    steps: &[ReplayStep],
) -> Result<StateDigest, SwarmRunError> {
    backend
        .checkpoint_load(checkpoint_bytes)
        .map_err(|e| SwarmRunError::Lifecycle(format!("resync checkpoint_load: {e}")))?;
    let mut last = None;
    for step in steps {
        let digest = backend
            .ingest(step.round, &step.staged)
            .map_err(|e| SwarmRunError::Lifecycle(format!("resync ingest r{}: {e}", step.round)))?;
        last = Some(digest);
    }
    last.ok_or_else(|| SwarmRunError::Lifecycle("resync replay had no steps".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StubBackend;
    use daemon_swarm_net::FsPayloadStore;

    fn temp_store() -> Arc<FsPayloadStore> {
        let root = std::env::temp_dir().join(format!(
            "daemon-swarm-ckpt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        Arc::new(FsPayloadStore::open(&root, 64).unwrap())
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

    #[tokio::test]
    async fn checkpoint_save_load_roundtrips_through_store() {
        // RUN-6 subset: save a checkpoint, reload it into a fresh backend, and reach the same digest
        // on the next ingest.
        let store = temp_store();
        let run = RunId::new("ckpt-run");
        let mut a = built(b"cfg");
        let d0 = a.ingest(0, &[staged(1, b"x"), staged(2, b"y")]).unwrap();

        let manifest = save_checkpoint(&store, &run, &a, 0, d0).await.unwrap();
        assert_eq!(manifest.round, 0);
        assert_eq!(manifest.digest, d0);

        // Reload into a fresh (differently-built) backend, then a further identical ingest must
        // match what the original produces from the same point.
        let mut b = built(b"totally-different-config");
        load_checkpoint(&store, &run, &mut b, &manifest)
            .await
            .unwrap();
        let next = [staged(1, b"p"), staged(2, b"q")];
        assert_eq!(b.ingest(1, &next).unwrap(), a.ingest(1, &next).unwrap());
    }

    #[tokio::test]
    async fn load_rejects_tampered_checkpoint() {
        // A manifest whose blake3 does not match the stored bytes is rejected on load (§9 content
        // addressing).
        let store = temp_store();
        let run = RunId::new("ckpt-tamper");
        let a = built(b"cfg");
        let mut manifest = save_checkpoint(&store, &run, &a, 0, StateDigest([0; 16]))
            .await
            .unwrap();
        manifest.blake3 = blake3_hash(b"not the checkpoint");
        let mut fresh = StubBackend::new();
        let err = load_checkpoint(&store, &run, &mut fresh, &manifest)
            .await
            .unwrap_err();
        assert!(matches!(err, SwarmRunError::Net(_)), "got {err:?}");
    }

    #[test]
    fn desync_replay_recovers_the_in_sync_digest() {
        // RUN-7 subset: a peer diverges (wrong ingest), then resyncs from a checkpoint + replays the
        // retained records/payloads → recovers the exact digest the in-sync peer reached (I1).
        let s0 = [staged(1, b"a0"), staged(2, b"b0")];
        let s1 = [staged(1, b"a1"), staged(2, b"b1")];
        let s2 = [staged(1, b"a2"), staged(2, b"b2")];

        // The in-sync reference peer.
        let mut good = built(b"cfg");
        good.ingest(0, &s0).unwrap();
        let checkpoint = good.checkpoint_save().unwrap(); // checkpoint after round 0
        good.ingest(1, &s1).unwrap();
        let target = good.ingest(2, &s2).unwrap();

        // The diverged peer: same round 0, then a *reordered* round-1 set → wrong digest.
        let mut bad = built(b"cfg");
        bad.ingest(0, &s0).unwrap();
        let diverged = bad.ingest(1, &[s1[1].clone(), s1[0].clone()]).unwrap();
        assert_ne!(diverged, target, "the peer has desynced");

        // Resync: reload the round-0 checkpoint, replay rounds 1 and 2 in record order.
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
        assert_eq!(recovered, target, "replay recovers the in-sync digest (I1)");
    }
}
