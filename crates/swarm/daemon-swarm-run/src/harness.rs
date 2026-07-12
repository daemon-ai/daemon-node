// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! In-process multi-peer harness + a **TEST-ONLY scripted coordinator** (spec §6.4; e2e support).
//!
//! Lane P2's real coordinator is built in parallel and unavailable this wave, so the peer-side
//! [`RoundEngine`](crate::engine::RoundEngine) is exercised against a scripted stand-in:
//! [`ScriptedCoordinator`] emits a hardcoded `RoundOpen`/`RoundRecord` sequence over
//! [`LoopbackGossip`](daemon_swarm_net::LoopbackGossip), observing peers' `Commitment`s /
//! `Straggle`s to build each round's committed set. `// MERGE-2: replace with the
//! daemon-swarm-coordinator tick loop` — the peer-set / event-collection shape here stays; only the
//! coordinator swaps. This is the shape the Merge-2 P0 milestone test keeps.
//!
//! [`run_swarm`] spins up N peer engines + the scripted coordinator over one gossip mesh and one
//! shared [`FsPayloadStore`](daemon_swarm_net::FsPayloadStore), runs a fixed number of rounds, and
//! returns the collected [`SwarmRun`] event log (per-peer digests per round). An optional
//! [`StallFault`] injects a payload miss (via [`FaultyStore`]) to drive the §6.4 stall ladder
//! deterministically. Everything is seeded, so two runs of the same [`SwarmConfig`] produce a
//! byte-identical digest transcript.

#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc::unbounded_channel;
use tokio::task::JoinHandle;

use daemon_swarm_net::{
    ControlPlane, FsPayloadStore, LoopbackGossip, PayloadStat, PayloadStore, SwarmNetError,
};
use daemon_swarm_proto::messages::{
    BatchWindow, Commitment, Locator, RecordEntry, RoundOpen, RoundRecord, Straggle, StraggleStatus,
};
use daemon_swarm_proto::{
    blake3_hash, commit_set, from_canonical_slice, to_canonical_vec, Hash, PeerId, Seed,
    SigningKey, SwarmMessage, SwarmProtoVersion, SWARM_PROTO_VERSION,
};

use crate::backend::{StateDigest, StubBackend, TrainerBackend};
use crate::engine::{EngineConfig, EngineEvent, RoundEngine};
use crate::seam::{PayloadKey, RoundId, RunId};
use crate::SwarmRunError;

/// A [`PayloadStore`] wrapper that withholds one object for its first `n` `get`s, to drive the
/// §6.4 stall ladder deterministically (TEST-ONLY).
///
/// The withheld key returns a typed [`SwarmNetError::PayloadMiss`] for the first `n` `get` calls
/// (the stalling peer's prefetch + barrier attempts), then delegates — so a subsequent cross-round
/// catch-up attempt succeeds. All other keys (and `put`/`head`) delegate unchanged. Independent of
/// wall-clock timing, so the stall/catch-up is reproducible regardless of task scheduling.
pub struct FaultyStore {
    inner: Arc<FsPayloadStore>,
    withhold: Option<(PayloadKey, u32)>,
    gets: AtomicU32,
}

impl FaultyStore {
    /// A transparent wrapper (no fault).
    #[must_use]
    pub fn transparent(inner: Arc<FsPayloadStore>) -> Self {
        Self {
            inner,
            withhold: None,
            gets: AtomicU32::new(0),
        }
    }

    /// Withhold `key` for its first `first_n_gets` `get` calls, then serve it.
    #[must_use]
    pub fn withholding(inner: Arc<FsPayloadStore>, key: PayloadKey, first_n_gets: u32) -> Self {
        Self {
            inner,
            withhold: Some((key, first_n_gets)),
            gets: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl PayloadStore for FaultyStore {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<Hash, SwarmNetError> {
        self.inner.put(key, bytes).await
    }

    async fn get(&self, key: &PayloadKey, expected: &Hash) -> Result<Vec<u8>, SwarmNetError> {
        if let Some((withheld, n)) = &self.withhold {
            if key == withheld {
                let seen = self.gets.fetch_add(1, Ordering::SeqCst);
                if seen < *n {
                    return Err(SwarmNetError::PayloadMiss(format!(
                        "injected miss #{} for {}@r{}/{}",
                        seen + 1,
                        key.run.as_str(),
                        key.round,
                        key.peer.to_hex()
                    )));
                }
            }
        }
        self.inner.get(key, expected).await
    }

    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, SwarmNetError> {
        self.inner.head(key).await
    }
}

/// A payload miss injected at a specific peer/round to drive the stall ladder (TEST-ONLY).
#[derive(Clone, Copy, Debug)]
pub struct StallFault {
    /// The roster index of the peer that will stall.
    pub peer_index: usize,
    /// The roster index of the peer whose payload it cannot fetch.
    pub missing_peer_index: usize,
    /// The round at which the miss occurs.
    pub round: RoundId,
    /// How many of the stalling peer's `get`s of that object are withheld (prefetch + barrier).
    pub first_n_gets: u32,
}

/// Configuration for an in-process swarm run.
#[derive(Clone, Debug)]
pub struct SwarmConfig {
    /// Number of peers.
    pub num_peers: usize,
    /// Number of rounds the scripted coordinator drives.
    pub num_rounds: u64,
    /// Inner steps per round.
    pub steps_per_round: u32,
    /// Micro-batch size (sequences).
    pub micro_batch: u32,
    /// Stall-recovery budget (rounds) before a stalled peer leaves.
    pub stall_rounds_max: u32,
    /// Save a round-boundary checkpoint every N rounds (0 = off).
    pub checkpoint_every_rounds: u32,
    /// Seed for the synthetic corpus.
    pub corpus_seed: u64,
    /// Optional injected stall.
    pub fault: Option<StallFault>,
}

impl SwarmConfig {
    /// A small, fault-free default (3 peers, 20 rounds).
    #[must_use]
    pub fn small(num_rounds: u64) -> Self {
        Self {
            num_peers: 3,
            num_rounds,
            steps_per_round: 2,
            micro_batch: 2,
            stall_rounds_max: 2,
            checkpoint_every_rounds: 0,
            corpus_seed: 0xDAE0_7E57,
            fault: None,
        }
    }
}

/// The collected outcome of a swarm run: every `(peer, event)` the engines emitted.
#[derive(Clone, Debug)]
pub struct SwarmRun {
    /// The sorted roster (node identities).
    pub roster: Vec<PeerId>,
    /// Every emitted event, tagged with the peer that emitted it (collection order).
    pub events: Vec<(PeerId, EngineEvent)>,
}

impl SwarmRun {
    /// The digest each peer reported for each round (from `RoundComplete` or `CaughtUp`).
    #[must_use]
    pub fn digests_by_round(&self) -> BTreeMap<RoundId, BTreeMap<PeerId, StateDigest>> {
        let mut out: BTreeMap<RoundId, BTreeMap<PeerId, StateDigest>> = BTreeMap::new();
        for (peer, ev) in &self.events {
            match ev {
                EngineEvent::RoundComplete { round, digest }
                | EngineEvent::CaughtUp { round, digest } => {
                    out.entry(*round).or_default().insert(*peer, *digest);
                }
                _ => {}
            }
        }
        out
    }

    /// The single agreed digest per round (the first digest seen for that round). Pair with
    /// [`SwarmRun::all_agree`] to assert every peer matched it.
    #[must_use]
    pub fn agreed_transcript(&self) -> BTreeMap<RoundId, StateDigest> {
        self.digests_by_round()
            .into_iter()
            .filter_map(|(round, peers)| peers.values().next().map(|d| (round, *d)))
            .collect()
    }

    /// Whether every round has one distinct digest shared by all peers that reported it.
    #[must_use]
    pub fn all_agree(&self) -> bool {
        self.digests_by_round().values().all(|peers| {
            let mut vals = peers.values();
            let first = vals.next();
            first.is_some_and(|f| vals.all(|d| d == f))
        })
    }

    /// The set of peers that left the run (stall budget exhausted).
    #[must_use]
    pub fn left_peers(&self) -> BTreeSet<PeerId> {
        self.events
            .iter()
            .filter_map(|(peer, ev)| matches!(ev, EngineEvent::Left { .. }).then_some(*peer))
            .collect()
    }
}

/// A scripted coordinator (TEST-ONLY): drives rounds over a control plane, building each round's
/// committed set from observed `Commitment`s. `// MERGE-2: replace with daemon-swarm-coordinator`.
pub struct ScriptedCoordinator<C> {
    control: Arc<C>,
    key: SigningKey,
    version: SwarmProtoVersion,
    run: RunId,
    roster: Vec<PeerId>,
    steps_per_round: u32,
    micro_batch: u32,
    stall_rounds_max: u32,
    safety_timeout: Duration,
}

impl<C: ControlPlane> ScriptedCoordinator<C> {
    /// Build a scripted coordinator over `control` for `roster`.
    #[must_use]
    pub fn new(
        control: Arc<C>,
        key: SigningKey,
        run: RunId,
        mut roster: Vec<PeerId>,
        steps_per_round: u32,
        micro_batch: u32,
        stall_rounds_max: u32,
    ) -> Self {
        roster.sort_unstable();
        Self {
            control,
            key,
            version: SWARM_PROTO_VERSION,
            run,
            roster,
            steps_per_round,
            micro_batch,
            stall_rounds_max,
            // A large guard against a genuine hang only — round finalization is driven by
            // deterministic all-accounted logic, never by this timeout, so it never fires in a
            // correct run (which is what keeps the digest transcript reproducible).
            safety_timeout: Duration::from_secs(10),
        }
    }

    fn global_batch(&self) -> u64 {
        self.roster.len() as u64 * u64::from(self.steps_per_round) * u64::from(self.micro_batch)
    }

    fn window(&self, round: RoundId) -> BatchWindow {
        let g = self.global_batch();
        BatchWindow {
            start: round * g,
            end: (round + 1) * g,
        }
    }

    fn seed(&self, round: RoundId) -> Seed {
        Seed(*blake3_hash(&round.to_le_bytes()).as_bytes())
    }

    fn roster_digest(&self) -> Hash {
        let mut buf = Vec::with_capacity(self.roster.len() * PeerId::LEN);
        for p in &self.roster {
            buf.extend_from_slice(p.as_bytes());
        }
        blake3_hash(&buf)
    }

    /// Drive `num_rounds` rounds: open each round, collect commitments/straggles until every
    /// still-expected roster peer is accounted for (committed **or** straggled this round), then
    /// publish the round record.
    ///
    /// Finalization is **deterministic**, never wall-clock driven: a peer that publishes more than
    /// `stall_rounds_max` consecutive `Stalled` straggles has exhausted its budget and left, so it
    /// is dropped from the expected set (mirroring the peer's own leave rule) — no timeout is ever
    /// needed to make progress, which is what keeps the digest transcript reproducible.
    pub async fn run_rounds(&self, num_rounds: u64) -> Result<(), SwarmRunError> {
        let mut sub = self.control.subscribe();
        let mut expected: BTreeSet<PeerId> = self.roster.iter().copied().collect();
        let mut straggles: BTreeMap<PeerId, u32> = BTreeMap::new();
        for round in 0..num_rounds {
            let open = RoundOpen {
                round,
                seed: self.seed(round),
                roster_digest: self.roster_digest(),
                batch: self.window(round),
                deadline_unix_s: 0,
            };
            self.publish(SwarmMessage::RoundOpen(open)).await?;

            let mut committed: BTreeMap<PeerId, (Hash, u64)> = BTreeMap::new();
            let mut accounted: BTreeSet<PeerId> = BTreeSet::new();
            while !expected.iter().all(|p| accounted.contains(p)) {
                let recv = tokio::time::timeout(self.safety_timeout, sub.recv()).await;
                let bytes = match recv {
                    Ok(Some(bytes)) => bytes,
                    // Plane closed, or the safety guard fired (never in a correct run).
                    Ok(None) | Err(_) => break,
                };
                let Ok(msg) = from_canonical_slice::<daemon_swarm_proto::SignedMessage>(&bytes)
                else {
                    continue;
                };
                if msg.verify_for_run(self.version).is_err() {
                    continue;
                }
                match msg.payload {
                    SwarmMessage::Commitment(Commitment {
                        round: r,
                        payload,
                        size,
                        ..
                    }) if r == round => {
                        committed.insert(msg.signer, (payload, size));
                        accounted.insert(msg.signer);
                        straggles.insert(msg.signer, 0);
                    }
                    SwarmMessage::Straggle(Straggle {
                        round: r, status, ..
                    }) if r == round => {
                        accounted.insert(msg.signer);
                        // Only a `Stalled` straggle (skipped training) counts toward the leave
                        // budget; `Fetching` means the peer committed but can't yet ingest.
                        if status == StraggleStatus::Stalled {
                            let c = straggles.entry(msg.signer).or_insert(0);
                            *c += 1;
                            if *c > self.stall_rounds_max {
                                expected.remove(&msg.signer);
                            }
                        }
                    }
                    _ => {}
                }
            }

            let pairs: Vec<(PeerId, Hash)> = committed.iter().map(|(p, (h, _))| (*p, *h)).collect();
            let tree = commit_set(&pairs);
            let entries: Vec<RecordEntry> = tree
                .entries()
                .iter()
                .map(|(p, h)| RecordEntry {
                    peer: *p,
                    hash: *h,
                    size: committed.get(p).map_or(0, |(_, s)| *s),
                })
                .collect();
            let record = RoundRecord {
                round,
                set: tree.commitment(),
                drops: Vec::new(),
                next_seed: self.seed(round + 1),
                set_locator: Locator::StoreKey(format!("record-set/{}/{round}", self.run.as_str())),
                inline: Some(entries),
            };
            self.publish(SwarmMessage::RoundRecord(record)).await?;
        }
        Ok(())
    }

    async fn publish(&self, payload: SwarmMessage) -> Result<(), SwarmRunError> {
        let signed = daemon_swarm_proto::SignedMessage::sign(&self.key, self.version, payload)
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator sign: {e}")))?;
        let bytes = to_canonical_vec(&signed)
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator encode: {e}")))?;
        self.control.publish(&bytes).await?;
        Ok(())
    }
}

/// A deterministic per-peer node identity key (index `i`).
#[must_use]
pub fn peer_key(i: usize) -> SigningKey {
    SigningKey::from_bytes(&[0x11 + i as u8; 32])
}

/// The coordinator's node identity key.
#[must_use]
pub fn coordinator_key() -> SigningKey {
    SigningKey::from_bytes(&[0xC0; 32])
}

/// The experiment config bytes every peer builds its backend from (identical across peers, so the
/// consensus round base coincides — the reconvergence precondition, §5.6).
pub const EXPERIMENT_CONFIG: &[u8] = b"e2e-experiment-config";

/// Run an in-process swarm (N peers + scripted coordinator) for `cfg.num_rounds` rounds over the
/// deterministic [`StubBackend`]. Deterministic: same `cfg` → byte-identical digest transcript.
pub async fn run_swarm(cfg: SwarmConfig) -> Result<SwarmRun, SwarmRunError> {
    run_swarm_with(cfg, |_i| {
        let mut backend = StubBackend::new();
        backend
            .build(EXPERIMENT_CONFIG)
            .expect("build stub backend");
        backend
    })
    .await
}

/// Like [`run_swarm`], but each peer's [`TrainerBackend`] is produced by `make_backend(index)` — so
/// tests can inject an instrumented backend (e.g. to assert the ingest barrier, RUN-5). The factory
/// MUST return an already-`build`-ed backend (base snapshot set from the shared config).
pub async fn run_swarm_with<B, F>(
    cfg: SwarmConfig,
    make_backend: F,
) -> Result<SwarmRun, SwarmRunError>
where
    B: TrainerBackend + Send + Sync + 'static,
    F: Fn(usize) -> B,
{
    let run = RunId::new("e2e-run");
    let version = SWARM_PROTO_VERSION;

    let keys: Vec<SigningKey> = (0..cfg.num_peers).map(peer_key).collect();
    let peer_ids: Vec<PeerId> = keys
        .iter()
        .map(daemon_swarm_proto::peer_id)
        .collect::<Vec<_>>();
    let mut roster = peer_ids.clone();
    roster.sort_unstable();

    // A synthetic corpus large enough that assignment intervals are non-empty; the engine wraps
    // batch ids, so a monotonically advancing cursor never runs off the end.
    let corpus = Arc::new(crate::data::Corpus::synthetic(cfg.corpus_seed, 4, 256, 8)?);

    let gossip = Arc::new(LoopbackGossip::new());

    // One shared filesystem payload store (all peers PUT/GET here), rooted under a unique temp dir.
    let root = std::env::temp_dir().join(format!(
        "daemon-swarm-e2e-{}-{}",
        std::process::id(),
        fastcounter()
    ));
    let fs = Arc::new(FsPayloadStore::open(&root, cfg.num_rounds + 8)?);

    let (col_tx, mut col_rx) = unbounded_channel::<(PeerId, EngineEvent)>();
    let mut peer_handles: Vec<JoinHandle<Result<crate::engine::RunOutcome, SwarmRunError>>> =
        Vec::new();
    let mut fwd_handles: Vec<JoinHandle<()>> = Vec::new();

    for (i, key) in keys.into_iter().enumerate() {
        let peer = peer_ids[i];

        let store = match cfg.fault {
            Some(f) if f.peer_index == i => {
                let missing = peer_ids[f.missing_peer_index];
                let withheld = PayloadKey::new(run.clone(), f.round, missing);
                Arc::new(FaultyStore::withholding(
                    fs.clone(),
                    withheld,
                    f.first_n_gets,
                ))
            }
            _ => Arc::new(FaultyStore::transparent(fs.clone())),
        };

        let backend = make_backend(i);

        let engine_cfg = EngineConfig {
            run: run.clone(),
            roster: roster.clone(),
            witnesses: roster.clone(),
            steps_per_round: cfg.steps_per_round,
            micro_batch: cfg.micro_batch,
            stall_rounds_max: cfg.stall_rounds_max,
            checkpoint_every_rounds: cfg.checkpoint_every_rounds,
            version,
        };

        let (ev_tx, mut ev_rx) = unbounded_channel::<EngineEvent>();
        let col = col_tx.clone();
        fwd_handles.push(tokio::spawn(async move {
            while let Some(ev) = ev_rx.recv().await {
                let _ = col.send((peer, ev));
            }
        }));

        let mut engine = RoundEngine::new(
            gossip.clone(),
            store,
            backend,
            key,
            corpus.clone(),
            engine_cfg,
            ev_tx,
        );
        peer_handles.push(tokio::spawn(async move { engine.run().await }));
    }
    drop(col_tx);

    let coordinator = ScriptedCoordinator::new(
        gossip.clone(),
        coordinator_key(),
        run.clone(),
        roster.clone(),
        cfg.steps_per_round,
        cfg.micro_batch,
        cfg.stall_rounds_max,
    );
    let coord_handle = tokio::spawn(async move { coordinator.run_rounds(cfg.num_rounds).await });

    // Collect events until every peer has completed the final round (or left), with a safety
    // timeout so a silent/left peer cannot hang the harness.
    let last_round = cfg.num_rounds.saturating_sub(1);
    let mut events: Vec<(PeerId, EngineEvent)> = Vec::new();
    let mut done: BTreeSet<PeerId> = BTreeSet::new();
    loop {
        if roster.iter().all(|p| done.contains(p)) {
            break;
        }
        match tokio::time::timeout(Duration::from_secs(10), col_rx.recv()).await {
            Ok(Some((peer, ev))) => {
                match &ev {
                    EngineEvent::RoundComplete { round, .. }
                    | EngineEvent::CaughtUp { round, .. }
                        if *round == last_round =>
                    {
                        done.insert(peer);
                    }
                    EngineEvent::Left { .. } => {
                        done.insert(peer);
                    }
                    _ => {}
                }
                events.push((peer, ev));
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }

    // Coordinator has published every round by now; tear the peers down and drain any tail events.
    let _ = coord_handle.await;
    for h in &peer_handles {
        h.abort();
    }
    for h in fwd_handles {
        h.abort();
    }
    while let Ok((peer, ev)) = col_rx.try_recv() {
        events.push((peer, ev));
    }

    Ok(SwarmRun { roster, events })
}

/// A tiny process-lifetime counter for unique temp-dir names (no external `tempfile` dep).
fn fastcounter() -> u64 {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos ^ N.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9)
}
