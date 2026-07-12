// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`RoundEngine`] — the peer-side round state machine (spec §6.4; TDD RUN-1..5, RUN-8).
//!
//! One async state machine drives a single peer through rounds over the frozen Wave-1 seams —
//! [`ControlPlane`] + [`PayloadStore`] (net) and [`TrainerBackend`] (run) + the node ed25519
//! [`SigningKey`] (proto). It consumes the seven signed round messages and emits an
//! [`EngineEvent`] stream, per the §6.4 peer-side flow:
//!
//! - `RoundOpen(r)` → derive the assignment window → train (`train_step` × micro-batches →
//!   `inner_update` per inner step → `make_update`) → PUT the payload → publish a signed
//!   `Commitment`.
//! - **Witness duty**: as peers' `Commitment`s arrive, prefetch + blake3-verify their payloads and
//!   (if a witness) publish an `Attestation` whose signed field is the merkle root over the
//!   fetch-verified set (proto's `commit_set`).
//! - `RoundRecord(r)` is the **barrier** (invariant I2): verify the committed set against the
//!   record root, stage it in record order (node-pubkey bytes — invariant I3), run `ingest`, then
//!   publish the post-ingest `Digest`. The first `train_step` of r+1 cannot begin until `ingest(r)`
//!   returns because the engine owns `&mut backend` and processes messages sequentially.
//! - **Stall ladder** (RUN-8): a committed payload missing at the barrier → publish `Straggle`,
//!   skip training, keep fetching across rounds, late-ingest to catch up within `stall_rounds_max`,
//!   else leave for the next epoch.
//!
//! Downloads overlap compute at the swarm level: while this peer trains round r, its peers commit
//! and this peer prefetches their payloads reactively as the `Commitment`s arrive, so the barrier
//! usually finds the set already local. (`// MERGE-2`: a dedicated in-peer concurrent fetch task
//! becomes worthwhile once the real iroh/r2 payload plane replaces the fast `FsPayloadStore`.)
//!
//! Because lane P2's coordinator is built in parallel and unavailable, tests drive a **TEST-ONLY
//! scripted coordinator** (`// MERGE-2: replace with daemon-swarm-coordinator tick loop`); this
//! engine builds only the peer side against the frozen proto message types.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use tokio::sync::mpsc::UnboundedSender;

use daemon_swarm_net::{ControlPlane, ControlSubscription, PayloadStore};
use daemon_swarm_proto::messages::{
    Attestation, Commitment, Digest as DigestMsg, Locator, RecordEntry, RoundOpen, RoundRecord,
    Straggle, StraggleStatus,
};
use daemon_swarm_proto::{
    commit_set, from_canonical_slice, to_canonical_vec, Hash, PeerId, Root, SigningKey,
    SwarmMessage, SwarmProtoVersion,
};

use crate::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use crate::data::{slice_interval, BatchInterval, Corpus};
use crate::seam::{PayloadKey, RoundId, RunId};
use crate::SwarmRunError;

/// Deterministic equal-split assignment of a global batch window across the roster.
///
/// `// MERGE-2`: replace with `daemon-swarm-coordinator`'s throughput-weighted deterministic
/// assignment (§6.3, PROTO-8). The split site is isolated here so the swap is one function.
pub mod assignment {
    use super::{BatchInterval, PeerId};
    use daemon_swarm_proto::messages::BatchWindow;

    /// The `[start, end)` sub-interval assigned to `peer` — a contiguous equal split of `window`
    /// across the (sorted) `roster`, with the last peer absorbing the remainder.
    #[must_use]
    pub fn interval_for(window: BatchWindow, roster: &[PeerId], peer: &PeerId) -> BatchInterval {
        let n = roster.len().max(1) as u64;
        let idx = roster.iter().position(|p| p == peer).unwrap_or(0) as u64;
        let total = window.end.saturating_sub(window.start);
        let per = total / n;
        let start = window.start + idx * per;
        let end = if idx + 1 == n {
            window.end
        } else {
            start + per
        };
        BatchInterval::new(start, end)
    }
}

/// Static configuration for one peer's [`RoundEngine`] (frozen for the epoch).
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// The run this peer participates in (payload-store key component).
    pub run: RunId,
    /// The epoch roster (every participating node identity). Sorted internally for I3 order.
    pub roster: Vec<PeerId>,
    /// Whose `Attestation`s count as availability evidence (§6.4). Default: the whole roster.
    pub witnesses: Vec<PeerId>,
    /// Inner steps per round (§5.1 cadence, uniform across peers).
    pub steps_per_round: u32,
    /// Micro-batch size (sequences) within an inner step.
    pub micro_batch: u32,
    /// Fetch-recovery budget before a stalled peer must leave for the epoch (§6.4 rung 2).
    pub stall_rounds_max: u32,
    /// The run's pinned swarm proto version (exact-match join gate, §16).
    pub version: SwarmProtoVersion,
}

/// An observable outcome of the peer's round loop (the engine's event stream).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineEvent {
    /// This peer published its `Commitment` for `round`.
    Committed {
        /// The round committed.
        round: RoundId,
        /// The committed payload's blake3.
        hash: Hash,
    },
    /// This (witness) peer published an `Attestation` over its fetch-verified set.
    Attested {
        /// The round attested.
        round: RoundId,
        /// The set-commitment root.
        root: Root,
        /// The number of `(peer, hash)` pairs in the set.
        count: u32,
    },
    /// This peer ingested `round`'s committed set and published its `Digest`.
    RoundComplete {
        /// The round ingested.
        round: RoundId,
        /// The post-ingest state digest.
        digest: StateDigest,
    },
    /// This peer is stalled on `round` (a committed payload was missing at the barrier).
    Straggling {
        /// The round being recovered.
        round: RoundId,
        /// The recovery status reported on the heartbeat.
        status: StraggleStatus,
    },
    /// This peer late-ingested a previously-stalled `round` and caught up.
    CaughtUp {
        /// The round caught up.
        round: RoundId,
        /// The (late) post-ingest state digest.
        digest: StateDigest,
    },
    /// This peer left the run (stall budget exhausted); it rejoins at the next epoch.
    Left {
        /// The round at which it left.
        round: RoundId,
        /// Why it left.
        reason: String,
    },
}

/// How the round loop ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// The control plane closed with the peer current.
    Finished {
        /// The last round this peer ingested (or `None` if it never ingested one).
        last_round: Option<RoundId>,
    },
    /// The peer left mid-run (stall budget exhausted) for an epoch rejoin.
    LeftForEpoch {
        /// The round at which it left.
        round: RoundId,
    },
}

/// Per-round working state: received commitments + verified payloads + this witness's attest set.
#[derive(Default)]
struct RoundState {
    commits: BTreeMap<PeerId, Commitment>,
    fetched: BTreeMap<PeerId, Vec<u8>>,
    attested: BTreeSet<PeerId>,
}

/// The peer-side round state machine (§6.4). Generic over the transport + engine seams.
pub struct RoundEngine<C, P, B> {
    control: Arc<C>,
    store: Arc<P>,
    backend: B,
    key: SigningKey,
    corpus: Arc<Corpus>,
    cfg: EngineConfig,
    events: UnboundedSender<EngineEvent>,
    sub: ControlSubscription,
    peer: PeerId,
    roster: Vec<PeerId>,
    rounds: BTreeMap<RoundId, RoundState>,
    /// Verified round records awaiting ingest, keyed by round. Ingest is strictly in ascending
    /// round order (the barrier, I2): a record whose committed set cannot yet be fetched blocks
    /// every later record behind it until it is fetched (the stall ladder), so the outer step is
    /// never applied out of order.
    pending: BTreeMap<RoundId, Vec<RecordEntry>>,
    /// Whether the peer is currently stalled (the head of `pending` could not be fetched).
    straggling: bool,
    /// Consecutive `RoundOpen`s observed while stalled (the §6.4 rung-2 budget).
    stalled_rounds: u32,
    last_ingested: Option<RoundId>,
}

impl<C, P, B> RoundEngine<C, P, B>
where
    C: ControlPlane,
    P: PayloadStore,
    B: TrainerBackend,
{
    /// Construct an engine over the transport + engine seams. Subscribes to the control plane
    /// immediately so no message published after construction is missed.
    pub fn new(
        control: Arc<C>,
        store: Arc<P>,
        backend: B,
        key: SigningKey,
        corpus: Arc<Corpus>,
        cfg: EngineConfig,
        events: UnboundedSender<EngineEvent>,
    ) -> Self {
        let sub = control.subscribe();
        let peer = daemon_swarm_proto::peer_id(&key);
        let mut roster = cfg.roster.clone();
        roster.sort_unstable();
        Self {
            control,
            store,
            backend,
            key,
            corpus,
            cfg,
            events,
            sub,
            peer,
            roster,
            rounds: BTreeMap::new(),
            pending: BTreeMap::new(),
            straggling: false,
            stalled_rounds: 0,
            last_ingested: None,
        }
    }

    /// This peer's node identity.
    #[must_use]
    pub fn peer_id(&self) -> PeerId {
        self.peer
    }

    /// Run the message-driven round loop until the control plane closes or the stall budget is
    /// exhausted (`LeftForEpoch`).
    pub async fn run(&mut self) -> Result<RunOutcome, SwarmRunError> {
        while let Some(bytes) = self.sub.recv().await {
            let Ok(msg) = from_canonical_slice::<daemon_swarm_proto::SignedMessage>(&bytes) else {
                continue; // undecodable frame — gossip is best-effort dissemination
            };
            // Verification lives here, not in the transport (§7.1): drop bad-sig / wrong-version.
            if msg.verify_for_run(self.cfg.version).is_err() {
                continue;
            }
            match msg.payload {
                SwarmMessage::RoundOpen(ro) => {
                    if let Some(outcome) = self.on_round_open(&ro).await? {
                        return Ok(outcome);
                    }
                }
                SwarmMessage::Commitment(c) => self.on_commitment(msg.signer, c).await?,
                SwarmMessage::RoundRecord(rr) => self.on_round_record(&rr).await?,
                // Attestations / receipts / other peers' digests / straggles are coordinator inputs
                // (or observability); the peer side does not act on them in the MVP round loop.
                _ => {}
            }
        }
        Ok(RunOutcome::Finished {
            last_round: self.last_ingested,
        })
    }

    /// Handle `RoundOpen(r)`: first make progress on any stalled round (in-order catch-up), then
    /// either skip (still stalled) or train + commit this round.
    async fn on_round_open(&mut self, ro: &RoundOpen) -> Result<Option<RunOutcome>, SwarmRunError> {
        self.advance(None).await?;

        if self.straggling {
            // Still stalled: skip training, keep heartbeating Straggle, and check the budget.
            self.stalled_rounds += 1;
            let round = self.pending.keys().next().copied().unwrap_or(ro.round);
            self.publish(SwarmMessage::Straggle(Straggle {
                round: ro.round,
                status: StraggleStatus::Stalled,
            }))
            .await?;
            self.emit(EngineEvent::Straggling {
                round: ro.round,
                status: StraggleStatus::Stalled,
            });
            if self.stalled_rounds > self.cfg.stall_rounds_max {
                self.emit(EngineEvent::Left {
                    round,
                    reason: format!(
                        "stall budget {} exceeded recovering round {round}",
                        self.cfg.stall_rounds_max
                    ),
                });
                return Ok(Some(RunOutcome::LeftForEpoch { round }));
            }
            return Ok(None);
        }

        self.train_and_commit(ro).await?;
        Ok(None)
    }

    /// Derive this peer's interval, train it, seal + PUT the payload, and publish the `Commitment`.
    async fn train_and_commit(&mut self, ro: &RoundOpen) -> Result<(), SwarmRunError> {
        let interval = assignment::interval_for(ro.batch, &self.roster, &self.peer);
        let steps = slice_interval(interval, self.cfg.steps_per_round, self.cfg.micro_batch)?;
        let seq_len = self.corpus.manifest().seq_len;

        for step in &steps {
            let mb_count = step.micro_batches.len() as u32;
            for (mb_index, mb) in step.micro_batches.iter().enumerate() {
                let mut tokens = Vec::new();
                let mut step_seqs = 0u32;
                for batch in mb.start..mb.end {
                    tokens.extend(self.corpus.sequence(batch)?);
                    step_seqs += 1;
                }
                let batch_ref = BatchRef { tokens, seq_len };
                let ctx = StepCtx {
                    inner_step: step.index,
                    mb_index: mb_index as u32,
                    mb_count,
                    step_seqs,
                };
                self.backend
                    .train_step(&batch_ref, ctx)
                    .map_err(lifecycle)?;
            }
            self.backend.inner_update(step.index).map_err(lifecycle)?;
        }

        let payload = self.backend.make_update(ro.round).map_err(lifecycle)?;
        let peer = self.peer;
        let key = self.payload_key(ro.round, peer);
        let hash = self.store.put(&key, &payload).await?;
        // Cache our own payload so the barrier need not re-fetch it.
        self.round_mut(ro.round)
            .fetched
            .insert(peer, payload.clone());

        let commitment = Commitment {
            round: ro.round,
            payload: hash,
            size: payload.len() as u64,
            locators: vec![Locator::StoreKey(self.locator_key(ro.round, self.peer))],
        };
        self.publish(SwarmMessage::Commitment(commitment)).await?;
        self.emit(EngineEvent::Committed {
            round: ro.round,
            hash,
        });
        Ok(())
    }

    /// Handle a peer's `Commitment(r)`: record it and reactively prefetch + verify its payload
    /// (overlapping other peers' training), folding it into this witness's attestation.
    async fn on_commitment(
        &mut self,
        signer: PeerId,
        commitment: Commitment,
    ) -> Result<(), SwarmRunError> {
        let round = commitment.round;
        self.round_mut(round).commits.insert(signer, commitment);
        self.prefetch(round, signer).await?;
        Ok(())
    }

    /// Best-effort fetch + blake3-verify of `peer`'s committed payload for `round`, caching it and
    /// (if this node is a witness) re-publishing the cumulative attestation. A miss is tolerated
    /// (the barrier retries); a hash mismatch propagates (tamper, §12).
    async fn prefetch(&mut self, round: RoundId, peer: PeerId) -> Result<(), SwarmRunError> {
        if self.round_mut(round).fetched.contains_key(&peer) {
            return Ok(());
        }
        let Some(commitment) = self.round_mut(round).commits.get(&peer).cloned() else {
            return Ok(());
        };
        match self.fetch_verified(round, peer, &commitment.payload).await {
            Ok(bytes) => {
                self.round_mut(round).fetched.insert(peer, bytes);
                self.maybe_attest(round, peer, commitment.payload).await?;
            }
            Err(SwarmRunError::Net(daemon_swarm_net::SwarmNetError::PayloadMiss(_))) => {}
            Err(e) => return Err(e),
        }
        Ok(())
    }

    /// If this node is a witness, fold `(peer, hash)` into its round attest set and publish the
    /// updated `Attestation` (signed field = the `commit_set` root over the verified set).
    async fn maybe_attest(
        &mut self,
        round: RoundId,
        peer: PeerId,
        hash: Hash,
    ) -> Result<(), SwarmRunError> {
        if !self.cfg.witnesses.contains(&self.peer) {
            return Ok(());
        }
        self.round_mut(round).attested.insert(peer);
        let entries: Vec<(PeerId, Hash)> = {
            let rs = self.round_mut(round);
            rs.attested
                .iter()
                .filter_map(|p| rs.commits.get(p).map(|c| (*p, c.payload)))
                .collect()
        };
        let tree = commit_set(&entries);
        let commitment = tree.commitment();
        let inline = entries
            .iter()
            .map(|(p, h)| daemon_swarm_proto::messages::AttestEntry { peer: *p, hash: *h })
            .collect();
        let _ = (peer, hash);
        self.publish(SwarmMessage::Attestation(Attestation {
            round,
            set: commitment,
            inline: Some(inline),
        }))
        .await?;
        self.emit(EngineEvent::Attested {
            round,
            root: commitment.root,
            count: commitment.count,
        });
        Ok(())
    }

    /// Handle `RoundRecord(r)` — the barrier. Verify the committed set against the record root,
    /// enqueue it, and try to ingest as far as the queue allows (in order). If `r` itself cannot be
    /// ingested yet (its set — or an earlier round's — is unfetchable), enter the stall ladder.
    async fn on_round_record(&mut self, rr: &RoundRecord) -> Result<(), SwarmRunError> {
        let entries = verify_record_set(rr)?;
        self.pending.insert(rr.round, entries);
        self.advance(Some(rr.round)).await?;
        if self.pending.contains_key(&rr.round) {
            // This round could not be ingested yet (a committed payload is missing, here or behind
            // an earlier stalled round) → stall ladder.
            self.straggling = true;
            self.publish(SwarmMessage::Straggle(Straggle {
                round: rr.round,
                status: StraggleStatus::Fetching,
            }))
            .await?;
            self.emit(EngineEvent::Straggling {
                round: rr.round,
                status: StraggleStatus::Fetching,
            });
        }
        Ok(())
    }

    /// Ingest queued records in strictly ascending round order, stopping at the first whose
    /// committed set cannot yet be fetched (the barrier + stall ladder). `trigger` is the round
    /// whose record just arrived, if any — a round ingested "on time" (its own record, no prior
    /// stall) emits `RoundComplete`; a round ingested late (catch-up) emits `CaughtUp`.
    async fn advance(&mut self, trigger: Option<RoundId>) -> Result<(), SwarmRunError> {
        while let Some(round) = self.pending.keys().next().copied() {
            let entries = self.pending[&round].clone();
            match self.try_ingest(round, &entries).await? {
                Some(digest) => {
                    self.pending.remove(&round);
                    let on_time = !self.straggling && trigger == Some(round);
                    if on_time {
                        self.emit(EngineEvent::RoundComplete { round, digest });
                    } else {
                        self.emit(EngineEvent::CaughtUp { round, digest });
                    }
                }
                None => break, // head unfetchable — stay (or become) stalled
            }
        }
        if self.pending.is_empty() {
            self.straggling = false;
            self.stalled_rounds = 0;
        }
        Ok(())
    }

    /// Fetch every committed payload (from cache or the store) and, if all present, stage them in
    /// record order and `ingest`, returning the digest. A single miss returns `None` (the caller
    /// stalls); a hash mismatch propagates.
    async fn try_ingest(
        &mut self,
        round: RoundId,
        entries: &[RecordEntry],
    ) -> Result<Option<StateDigest>, SwarmRunError> {
        let mut staged = Vec::with_capacity(entries.len());
        for entry in entries {
            let bytes = match self.round_mut(round).fetched.get(&entry.peer).cloned() {
                Some(b) => b,
                None => match self.fetch_verified(round, entry.peer, &entry.hash).await {
                    Ok(b) => {
                        self.round_mut(round).fetched.insert(entry.peer, b.clone());
                        b
                    }
                    Err(SwarmRunError::Net(daemon_swarm_net::SwarmNetError::PayloadMiss(_))) => {
                        return Ok(None);
                    }
                    Err(e) => return Err(e),
                },
            };
            staged.push(StagedPayload {
                peer: entry.peer,
                hash: entry.hash,
                bytes,
            });
        }

        let digest = self.backend.ingest(round, &staged).map_err(lifecycle)?;
        self.last_ingested = Some(round);
        self.publish(SwarmMessage::Digest(DigestMsg {
            round,
            digest: daemon_swarm_proto::StateDigest::new(*digest.as_bytes()),
        }))
        .await?;
        // The round's transport scratch is no longer needed once ingested.
        self.rounds.remove(&round);
        Ok(Some(digest))
    }

    /// Fetch `peer`'s payload for `round` from the store, verifying it against `hash` (blake3).
    async fn fetch_verified(
        &self,
        round: RoundId,
        peer: PeerId,
        hash: &Hash,
    ) -> Result<Vec<u8>, SwarmRunError> {
        let key = self.payload_key(round, peer);
        Ok(self.store.get(&key, hash).await?)
    }

    fn payload_key(&self, round: RoundId, peer: PeerId) -> PayloadKey {
        PayloadKey::new(self.cfg.run.clone(), round, peer)
    }

    fn locator_key(&self, round: RoundId, peer: PeerId) -> String {
        format!("{}/{}/{}", self.cfg.run.as_str(), round, peer.to_hex())
    }

    fn round_mut(&mut self, round: RoundId) -> &mut RoundState {
        self.rounds.entry(round).or_default()
    }

    /// Sign `payload` with the node identity and publish the canonical-CBOR frame on the control
    /// plane (already-signed bytes, §7.1).
    async fn publish(&self, payload: SwarmMessage) -> Result<(), SwarmRunError> {
        let signed = daemon_swarm_proto::SignedMessage::sign(&self.key, self.cfg.version, payload)
            .map_err(|e| SwarmRunError::Lifecycle(format!("sign control message: {e}")))?;
        let bytes = to_canonical_vec(&signed)
            .map_err(|e| SwarmRunError::Lifecycle(format!("encode control message: {e}")))?;
        self.control.publish(&bytes).await?;
        Ok(())
    }

    fn emit(&self, event: EngineEvent) {
        let _ = self.events.send(event);
    }
}

/// Map a backend error into the run-lifecycle error (the backend's error is boxed by `Display`).
fn lifecycle<E: std::error::Error>(e: E) -> SwarmRunError {
    SwarmRunError::Lifecycle(format!("trainer backend: {e}"))
}

/// Verify a round record's inline committed set against its signed root and return it totally
/// ordered by node-pubkey bytes (invariant I3 staging order — the same order proto's `commit_set`
/// uses). Rejects a record whose inline set does not reconstruct the committed root + count.
///
/// A pure function of the record (no I/O), so the commit-side check is auditable and unit-testable.
/// `// MERGE-2`: at large rosters the set rides in `record-set.cbor` (fetched via `set_locator`)
/// rather than inline — wire that object fetch + root-verify here.
pub(crate) fn verify_record_set(rr: &RoundRecord) -> Result<Vec<RecordEntry>, SwarmRunError> {
    let inline = rr.inline.as_ref().ok_or_else(|| {
        SwarmRunError::Lifecycle(format!(
            "round {} record has no inline set (record-set.cbor fetch is a MERGE-2 seam)",
            rr.round
        ))
    })?;
    let pairs: Vec<(PeerId, Hash)> = inline.iter().map(|e| (e.peer, e.hash)).collect();
    let tree = commit_set(&pairs);
    let recomputed = tree.commitment();
    if recomputed.root != rr.set.root || recomputed.count != rr.set.count {
        return Err(SwarmRunError::Lifecycle(format!(
            "round {} record set does not match its committed root (I3)",
            rr.round
        )));
    }
    // Entries in commit_set (node-pubkey-byte) order, regardless of the inline list's order.
    let ordered = tree
        .entries()
        .iter()
        .filter_map(|(p, h)| {
            inline
                .iter()
                .find(|e| e.peer == *p && e.hash == *h)
                .copied()
        })
        .collect();
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AssessMeta, Assessment, BatchRef, StepStats, StubBackend};
    use crate::harness::{run_swarm, run_swarm_with, StallFault, SwarmConfig, EXPERIMENT_CONFIG};
    use daemon_swarm_proto::messages::RoundRecord;
    use daemon_swarm_proto::{blake3_hash, commit_set, Seed};
    use std::sync::{Arc, Mutex};

    fn peer(b: u8) -> PeerId {
        PeerId([b; 32])
    }

    fn entry(b: u8, tag: &[u8]) -> RecordEntry {
        RecordEntry {
            peer: peer(b),
            hash: blake3_hash(tag),
            size: tag.len() as u64,
        }
    }

    fn record_for(round: RoundId, entries: Vec<RecordEntry>) -> RoundRecord {
        let pairs: Vec<(PeerId, Hash)> = entries.iter().map(|e| (e.peer, e.hash)).collect();
        let tree = commit_set(&pairs);
        RoundRecord {
            round,
            set: tree.commitment(),
            drops: Vec::new(),
            next_seed: Seed([0; 32]),
            set_locator: daemon_swarm_proto::messages::Locator::StoreKey("s".into()),
            inline: Some(entries),
        }
    }

    #[test]
    fn verify_record_set_orders_by_pubkey_bytes() {
        // Unsorted inline set; verified entries must come back in node-pubkey-byte order (I3).
        let rr = record_for(
            4,
            vec![entry(0x99, b"c"), entry(0x11, b"a"), entry(0x55, b"b")],
        );
        let ordered = verify_record_set(&rr).unwrap();
        let peers: Vec<PeerId> = ordered.iter().map(|e| e.peer).collect();
        assert_eq!(peers, vec![peer(0x11), peer(0x55), peer(0x99)]);
    }

    #[test]
    fn verify_record_set_rejects_root_mismatch() {
        // A record whose signed root disagrees with its inline set is rejected (I3 exactness).
        let mut rr = record_for(4, vec![entry(0x11, b"a"), peer_entry_tampered()]);
        // Tamper the committed root so it no longer matches the inline set.
        rr.set.root = daemon_swarm_proto::Root([0xAB; 32]);
        let err = verify_record_set(&rr).unwrap_err();
        assert!(
            matches!(err, SwarmRunError::Lifecycle(m) if m.contains("does not match")),
            "expected root-mismatch rejection"
        );
    }

    fn peer_entry_tampered() -> RecordEntry {
        entry(0x22, b"b")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn three_peers_agree_each_round() {
        // RUN-1: 3 peers over the round loop; equal post-ingest digest every round.
        let run = run_swarm(SwarmConfig::small(6)).await.unwrap();
        assert!(run.left_peers().is_empty(), "no peer should leave");
        let by_round = run.digests_by_round();
        assert_eq!(by_round.len(), 6, "one digest set per round");
        for (round, peers) in &by_round {
            assert_eq!(peers.len(), 3, "all 3 peers report round {round}");
        }
        assert!(
            run.all_agree(),
            "every round's digest is shared by all peers"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn transcript_is_deterministic_across_runs() {
        // Same config → byte-identical agreed digest transcript.
        let a = run_swarm(SwarmConfig::small(8)).await.unwrap();
        let b = run_swarm(SwarmConfig::small(8)).await.unwrap();
        assert!(a.all_agree() && b.all_agree());
        assert_eq!(
            a.agreed_transcript(),
            b.agreed_transcript(),
            "the digest transcript must be reproducible"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stalled_peer_catches_up_within_budget() {
        // RUN-8: peer 1 misses peer 0's round-3 payload (first 2 gets), stalls, catches up next
        // round, and still agrees on every round's digest.
        let cfg = SwarmConfig {
            fault: Some(StallFault {
                peer_index: 1,
                missing_peer_index: 0,
                round: 3,
                first_n_gets: 2,
            }),
            ..SwarmConfig::small(6)
        };
        let run = run_swarm(cfg).await.unwrap();
        assert!(run.left_peers().is_empty(), "peer recovers within budget");
        assert!(run.all_agree(), "digests agree despite the stall");

        // Round 3 is still reported by all three peers (the stalled one via CaughtUp).
        let by_round = run.digests_by_round();
        assert_eq!(by_round[&3].len(), 3, "the stalled peer catches up round 3");

        let stalled = daemon_swarm_proto::peer_id(&crate::harness::peer_key(1));
        let mine: Vec<&EngineEvent> = run
            .events
            .iter()
            .filter(|(p, _)| *p == stalled)
            .map(|(_, e)| e)
            .collect();
        assert!(
            mine.iter()
                .any(|e| matches!(e, EngineEvent::Straggling { round: 3, .. })),
            "the stalled peer straggles round 3"
        );
        assert!(
            mine.iter()
                .any(|e| matches!(e, EngineEvent::CaughtUp { round: 3, .. })),
            "the stalled peer catches up round 3"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stall_budget_exhausted_leaves() {
        // RUN-8: the missing payload never arrives within budget → the peer leaves for the epoch;
        // the rest keep agreeing.
        let cfg = SwarmConfig {
            stall_rounds_max: 2,
            fault: Some(StallFault {
                peer_index: 1,
                missing_peer_index: 0,
                round: 3,
                first_n_gets: 1000,
            }),
            ..SwarmConfig::small(8)
        };
        let run = run_swarm(cfg).await.unwrap();
        let stalled = daemon_swarm_proto::peer_id(&crate::harness::peer_key(1));
        assert!(
            run.left_peers().contains(&stalled),
            "the peer leaves once the stall budget is exhausted"
        );
        assert!(run.all_agree(), "the remaining peers still agree");
        // The peer that left never ingested past the round it stalled on (no out-of-order apply).
        let stalled_digests: Vec<RoundId> = run
            .digests_by_round()
            .into_iter()
            .filter(|(_, peers)| peers.contains_key(&stalled))
            .map(|(r, _)| r)
            .collect();
        assert!(
            stalled_digests.iter().all(|r| *r < 3),
            "the left peer must not apply any round >= its stall point (barrier), got {stalled_digests:?}"
        );
    }

    /// A [`TrainerBackend`] wrapper that records the round each `train_step` / `ingest` belongs to,
    /// so a test can assert the ingest barrier (I2): the first `train_step` of round r+1
    /// happens-after `ingest(r)`.
    struct RecordingBackend {
        inner: StubBackend,
        round: RoundId,
        log: Arc<Mutex<Vec<Op>>>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Op {
        Train(RoundId),
        Ingest(RoundId),
    }

    impl TrainerBackend for RecordingBackend {
        type Error = crate::backend::StubError;

        fn build(&mut self, config: &[u8]) -> Result<(), Self::Error> {
            self.inner.build(config)
        }
        fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error> {
            self.inner.assess(meta)
        }
        fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error> {
            self.log.lock().unwrap().push(Op::Train(self.round));
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
            let d = self.inner.ingest(round, staged)?;
            self.log.lock().unwrap().push(Op::Ingest(round));
            self.round = round + 1; // subsequent train_steps belong to the next round
            Ok(d)
        }
        fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
            self.inner.checkpoint_save()
        }
        fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
            self.inner.checkpoint_load(bytes)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ingest_barrier_orders_next_round() {
        // RUN-5 / I2: with one peer over several rounds, the first train_step of round r+1 is
        // recorded strictly after ingest(r).
        let log: Arc<Mutex<Vec<Op>>> = Arc::new(Mutex::new(Vec::new()));
        let cfg = SwarmConfig {
            num_peers: 1,
            ..SwarmConfig::small(4)
        };
        let log_for_factory = log.clone();
        let _ = run_swarm_with(cfg, move |_i| {
            let mut inner = StubBackend::new();
            inner.build(EXPERIMENT_CONFIG).unwrap();
            RecordingBackend {
                inner,
                round: 0,
                log: log_for_factory.clone(),
            }
        })
        .await
        .unwrap();

        let ops = log.lock().unwrap().clone();
        let mut first_train: std::collections::BTreeMap<RoundId, usize> =
            std::collections::BTreeMap::new();
        let mut ingest_at: std::collections::BTreeMap<RoundId, usize> =
            std::collections::BTreeMap::new();
        for (i, op) in ops.iter().enumerate() {
            match op {
                Op::Train(r) => {
                    first_train.entry(*r).or_insert(i);
                }
                Op::Ingest(r) => {
                    ingest_at.insert(*r, i);
                }
            }
        }
        // For each round r ≥ 1, the first train_step of r comes after ingest(r-1).
        for (&r, &train_idx) in &first_train {
            if r == 0 {
                continue;
            }
            let prev_ingest = ingest_at
                .get(&(r - 1))
                .copied()
                .unwrap_or_else(|| panic!("round {} had no ingest for {}", r, r - 1));
            assert!(
                train_idx > prev_ingest,
                "train_step(round {r}) must happen-after ingest(round {})",
                r - 1
            );
        }
        assert!(ingest_at.len() >= 3, "several rounds ingested");
    }
}
