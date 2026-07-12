// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! In-process multi-peer harness driven by the **real** `daemon-swarm-coordinator` pure `tick`
//! (spec §6.2/§6.4; the Merge-2 P0 milestone support).
//!
//! Merge 2 swapped R2's TEST-ONLY `ScriptedCoordinator` for [`TickCoordinator`], the impure shell
//! around lane P2's pure [`tick`](daemon_swarm_coordinator::tick): it holds a
//! [`CoordinatorState`], feeds it signed peer messages / `StorageReceipt` availability evidence /
//! scripted `Clock` inputs, and **signs + publishes** the coordinator's own unsigned
//! `RoundOpen`/`RoundRecord` outputs over [`LoopbackGossip`](daemon_swarm_net::LoopbackGossip). The
//! peer-set / event-collection shape from R2 is unchanged — only the coordinator swapped.
//!
//! ## Why a `StorageReceipt` evidence path
//!
//! The pure commit rule (§6.4 I6) admits a payload only with signed availability evidence — a
//! `StorageReceipt` **or** a witness-quorum of `Attestation`s. R2's peer engine never attests its
//! **own** payload (a peer's self-prefetch short-circuits), so witness-quorum alone cannot evidence
//! every payload at small rosters / under a stall (e.g. a single peer, or a straggler that cannot
//! fetch one peer's object). The shell therefore acts as the coordinator-as-storage-client: on each
//! `Commitment` it `HEAD`s the shared store and feeds a signed `StorageReceipt` (P2's intended
//! primary evidence path — "the coordinator's HEADs already arrived as signed StorageReceipt
//! inputs"). This is decided **outside** `tick`, keeping the state machine pure.
//!
//! ## Deterministic finalization
//!
//! `tick` is pure, so the same input sequence yields the same `(state, outputs)`. Round records are
//! order-independent functions of accumulated evidence, so the digest transcript does not depend on
//! message arrival order. Rounds finalize **event-driven** (the last commitment + its receipt make
//! the round fully committed + evidenced → `tick` finalizes with no clock). A round blocked by a
//! straggler that will not commit is forced by a `Clock` input only once every still-expected peer
//! is **accounted** (committed **or** `Straggle(Stalled)` this round) — the same deterministic rule
//! R2's scripted coordinator used; a quiescence guard covers a peer that has gone fully silent
//! (left). Neither ever fires on the happy path, so two runs of the same [`SwarmConfig`] produce a
//! byte-identical digest transcript, and [`CoordinatorReplay`] re-runs `tick` over the recorded
//! inputs to prove an identical canonical-CBOR state trajectory (PROTO-20 spirit).

#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc::unbounded_channel;
use tokio::task::JoinHandle;

use daemon_swarm_coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, Phase, RunConfig,
};
use daemon_swarm_net::{
    ControlPlane, FsPayloadStore, LoopbackGossip, PayloadStat, PayloadStore, SwarmNetError,
};
use daemon_swarm_proto::envelope::{GlobalBatch, StopCondition};
use daemon_swarm_proto::messages::{
    Commitment, Join, RecordEntry, StorageReceipt, Straggle, StraggleStatus, ThroughputClass,
};
use daemon_swarm_proto::{
    from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, Seed,
    SignedMessage, SigningKey, SwarmMessage, SwarmProtoVersion, SWARM_PROTO_VERSION,
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
    /// Number of rounds the coordinator drives (`[data].stop = Rounds(num_rounds)`).
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

/// A recorded coordinator run trajectory for the offline replay assertion (PROTO-20 spirit).
///
/// Holds the exact ordered [`Input`] sequence the [`TickCoordinator`] fed its pure `tick`, the
/// initial [`CoordinatorState`], and a canonical-CBOR snapshot of the state taken right after each
/// `RoundRecord` was emitted. [`CoordinatorReplay::verify`] re-runs `tick` over the recorded inputs
/// from the initial state and asserts a byte-identical per-round state trajectory.
#[derive(Clone, Debug)]
pub struct CoordinatorReplay {
    initial: CoordinatorState,
    inputs: Vec<Input>,
    states_by_round: BTreeMap<RoundId, Vec<u8>>,
}

impl CoordinatorReplay {
    /// The number of rounds whose post-record state was snapshotted.
    #[must_use]
    pub fn recorded_rounds(&self) -> usize {
        self.states_by_round.len()
    }

    /// Re-run `tick` over the recorded inputs from the initial state and confirm the per-round
    /// canonical-CBOR state trajectory is byte-identical to the live run (I1 / PROTO-20 spirit).
    #[must_use]
    pub fn verify(&self) -> bool {
        let mut state = self.initial.clone();
        let mut replayed: BTreeMap<RoundId, Vec<u8>> = BTreeMap::new();
        for input in &self.inputs {
            let (next, outputs) = tick(state, input.clone());
            state = next;
            for o in &outputs {
                if let Output::Publish(msg) = o {
                    if let SwarmMessage::RoundRecord(rr) = msg.as_ref() {
                        match to_canonical_vec(&state) {
                            Ok(bytes) => {
                                replayed.insert(rr.round, bytes);
                            }
                            Err(_) => return false,
                        }
                    }
                }
            }
        }
        replayed == self.states_by_round
    }
}

/// The collected outcome of a swarm run: every `(peer, event)` the engines emitted.
#[derive(Clone, Debug)]
pub struct SwarmRun {
    /// The sorted roster (node identities).
    pub roster: Vec<PeerId>,
    /// Every emitted event, tagged with the peer that emitted it (collection order).
    pub events: Vec<(PeerId, EngineEvent)>,
    /// The coordinator's recorded `tick` trajectory for the replay assertion, if captured.
    pub replay: Option<CoordinatorReplay>,
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

/// A deterministic per-peer node identity key (index `i`).
#[must_use]
pub fn peer_key(i: usize) -> SigningKey {
    SigningKey::from_bytes(&[0x11 + i as u8; 32])
}

/// The coordinator's node identity key (signs the emitted RoundOpen/RoundRecord + the shell's
/// StorageReceipt evidence).
#[must_use]
pub fn coordinator_key() -> SigningKey {
    SigningKey::from_bytes(&[0xC0; 32])
}

/// The experiment config bytes every peer builds its backend from (identical across peers, so the
/// consensus round base coincides — the reconvergence precondition, §5.6).
pub const EXPERIMENT_CONFIG: &[u8] = b"e2e-experiment-config";

/// The impure shell around the pure [`tick`]: signs + publishes the coordinator's outputs, supplies
/// `StorageReceipt` evidence, and drives round finalization deterministically (`// MERGE-2` resolved
/// — this replaced R2's `ScriptedCoordinator`).
struct TickCoordinator<C> {
    control: Arc<C>,
    store: Arc<FsPayloadStore>,
    key: SigningKey,
    version: SwarmProtoVersion,
    run: RunId,
    num_peers: usize,
    /// The pure coordinator state (kept in an `Option` so `tick` can take it by value + return it).
    state: Option<CoordinatorState>,
    /// Peers whose commitment for a round has been fed **and** receipted (evidenced).
    committed: BTreeMap<RoundId, BTreeSet<PeerId>>,
    /// Peers that reported a `Straggle(Stalled)` (skipped training) for a round.
    stalled: BTreeMap<RoundId, BTreeSet<PeerId>>,
    /// The ordered input log + per-round state snapshots for the replay assertion.
    inputs: Vec<Input>,
    states_by_round: BTreeMap<RoundId, Vec<u8>>,
    initial: CoordinatorState,
}

impl<C: ControlPlane> TickCoordinator<C> {
    fn new(
        control: Arc<C>,
        store: Arc<FsPayloadStore>,
        key: SigningKey,
        run: RunId,
        cfg: &SwarmConfig,
    ) -> Self {
        // Global batch = peers × steps × micro-batch, so the per-peer (class-equal) partition is
        // steps × micro-batch — divisible by `steps_per_round` for `slice_interval` (RUN-3).
        let g =
            (cfg.num_peers as u64) * u64::from(cfg.steps_per_round) * u64::from(cfg.micro_batch);
        let n = cfg.num_peers as u32;
        let params = CoordinatorParams {
            seq_len: 1,
            witness_target: 0, // every peer witnesses (matches the engine's witness set)
            overlap_bps: 0,
            k_absences: 1, // drop a genuinely-absent peer promptly (leave path)
            verification_percent: 0,
            authorized: Vec::new(),
        };
        let config = RunConfig {
            run_id: run.as_str().to_string(),
            proto_version: SWARM_PROTO_VERSION,
            envelope_hash: Hash([0u8; 32]),
            required_capabilities: CapabilitySet::new(),
            min_peers: n,
            max_peers: n.max(1),
            warmup_s: 1,
            // The happy path finalizes event-driven (no clocks), so these timeouts only bound the
            // shell's *forced* progress clocks; keep them small.
            round_train_max_s: 1,
            round_witness_s: 1,
            cooldown_s: 1,
            epoch_rounds: 0, // one epoch for the whole run (no mid-run roster freeze boundary)
            stall_rounds_max: cfg.stall_rounds_max,
            global_batch: GlobalBatch {
                start: g as u32,
                end: g as u32,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(cfg.num_rounds),
            steps_per_round: cfg.steps_per_round,
            seq_len: params.seq_len,
            witness_target: params.witness_target,
            overlap_bps: params.overlap_bps,
            k_absences: params.k_absences,
            verification_percent: params.verification_percent,
            authorized: params.authorized,
        };
        let state = CoordinatorState::new(config, Seed([0xAB; 32]), 0);
        Self {
            control,
            store,
            key,
            version: SWARM_PROTO_VERSION,
            run,
            num_peers: cfg.num_peers,
            state: Some(state.clone()),
            committed: BTreeMap::new(),
            stalled: BTreeMap::new(),
            inputs: Vec::new(),
            states_by_round: BTreeMap::new(),
            initial: state,
        }
    }

    fn state(&self) -> &CoordinatorState {
        self.state.as_ref().expect("coordinator state present")
    }

    /// Feed one input to the pure `tick`, record it, and sign + publish any emitted messages.
    async fn apply(&mut self, input: Input) -> Result<(), SwarmRunError> {
        self.inputs.push(input.clone());
        let state = self.state.take().expect("coordinator state present");
        let (state, outputs) = tick(state, input);
        self.state = Some(state);
        for o in outputs {
            if let Output::Publish(msg) = o {
                let payload = *msg;
                let record_round = match &payload {
                    SwarmMessage::RoundRecord(rr) => Some(rr.round),
                    _ => None,
                };
                let signed = SignedMessage::sign(&self.key, self.version, payload)
                    .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator sign: {e}")))?;
                let bytes = to_canonical_vec(&signed)
                    .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator encode: {e}")))?;
                self.control.publish(&bytes).await?;
                if let Some(r) = record_round {
                    if let Ok(sbytes) = to_canonical_vec(self.state()) {
                        self.states_by_round.insert(r, sbytes);
                    }
                }
            }
        }
        Ok(())
    }

    /// Bootstrap: synthesize each peer's signed `Join`, then clock past warmup so round 0 opens.
    async fn bootstrap(&mut self) -> Result<(), SwarmRunError> {
        for i in 0..self.num_peers {
            let k = peer_key(i);
            let join = Join {
                run_id: self.run.as_str().to_string(),
                iroh_id: IrohId([0x22; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
            };
            let signed = SignedMessage::sign(&k, self.version, SwarmMessage::Join(join))
                .map_err(|e| SwarmRunError::Lifecycle(format!("join sign: {e}")))?;
            self.apply(Input::Message(signed)).await?;
        }
        // WaitingForMembers → Warmup, then Warmup → RoundTrain (warmup_s = 1) → RoundOpen(0).
        self.apply(Input::Clock(1)).await?;
        self.apply(Input::Clock(3)).await?;
        Ok(())
    }

    /// Force the current round to finalize via the round/witness timeouts (a straggler will not
    /// commit, so the event-driven fast path cannot fire). Loops a few times to walk
    /// `RoundTrain → RoundWitness → commit → open(next)` in case one clock is not enough.
    async fn force_current_round(&mut self) -> Result<(), SwarmRunError> {
        for _ in 0..4 {
            if !self.state().phase.is_round_active() {
                break;
            }
            let before = self.state().round;
            let now = self.state().now_s
                + self.state().config.round_train_max_s
                + self.state().config.round_witness_s
                + 1;
            self.apply(Input::Clock(now)).await?;
            if self.state().round != before {
                break;
            }
        }
        Ok(())
    }

    /// After new round evidence, force the current round iff every still-healthy peer is accounted
    /// (committed **and** receipted, or has sent a `Straggle(Stalled)` this round) yet the round has
    /// not auto-finalized — the deterministic, content-driven finalization trigger.
    async fn maybe_force_accounted(&mut self) -> Result<(), SwarmRunError> {
        if !self.state().phase.is_round_active() {
            return Ok(());
        }
        let round = self.state().round;
        let healthy = self.state().healthy_peer_ids();
        let committed = self.committed.get(&round);
        let stalled = self.stalled.get(&round);
        let all_accounted = healthy.iter().all(|p| {
            committed.is_some_and(|c| c.contains(p)) || stalled.is_some_and(|s| s.contains(p))
        });
        if all_accounted && !healthy.is_empty() {
            self.force_current_round().await?;
        }
        Ok(())
    }

    /// Produce + feed a signed `StorageReceipt` for `(round, peer)` if the object is in the store
    /// (the coordinator-as-storage-client evidence path). Records the peer as evidenced.
    async fn receipt_for(&mut self, round: RoundId, peer: PeerId) -> Result<(), SwarmRunError> {
        let key = PayloadKey::new(self.run.clone(), round, peer);
        let Ok(stat) = self.store.head(&key).await else {
            return Ok(());
        };
        let sr = StorageReceipt {
            round,
            verified: vec![RecordEntry {
                peer,
                hash: stat.hash,
                size: stat.size,
            }],
        };
        let signed = SignedMessage::sign(&self.key, self.version, SwarmMessage::StorageReceipt(sr))
            .map_err(|e| SwarmRunError::Lifecycle(format!("receipt sign: {e}")))?;
        self.apply(Input::Message(signed)).await?;
        self.committed.entry(round).or_default().insert(peer);
        Ok(())
    }

    /// The run has finalized every round (or stopped) — nothing more to drive.
    fn finished(&self) -> bool {
        matches!(
            self.state().phase,
            Phase::Cooldown | Phase::Finished | Phase::Uninitialized
        )
    }

    /// Drive the run to completion: bootstrap, then feed inbound signed peer messages (+ receipts +
    /// accounted/quiescence-forced clocks) until the run reaches cooldown after the final round.
    async fn drive(mut self) -> Result<CoordinatorReplay, SwarmRunError> {
        let mut sub = self.control.subscribe();
        self.bootstrap().await?;

        // A generous quiescence guard: it never fires on the happy path (the coordinator's own
        // republished RoundOpen/RoundRecord echoes keep the subscription lively), only when a peer
        // has gone fully silent (left) so no `Straggle` will ever account it.
        let quiescence = Duration::from_millis(1500);
        while !self.finished() {
            match tokio::time::timeout(quiescence, sub.recv()).await {
                Ok(Some(bytes)) => {
                    let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                        continue;
                    };
                    if msg.verify_for_run(self.version).is_err() {
                        continue;
                    }
                    let signer = msg.signer;
                    // Skip the coordinator's own republished outputs (fed to gossip, echoed back).
                    match &msg.payload {
                        SwarmMessage::RoundOpen(_) | SwarmMessage::RoundRecord(_) => continue,
                        SwarmMessage::Commitment(Commitment { round, .. }) => {
                            let round = *round;
                            self.apply(Input::Message(msg)).await?;
                            self.receipt_for(round, signer).await?;
                        }
                        SwarmMessage::Straggle(Straggle { round, status }) => {
                            let round = *round;
                            let stalled = *status == StraggleStatus::Stalled;
                            self.apply(Input::Message(msg)).await?;
                            if stalled {
                                self.stalled.entry(round).or_default().insert(signer);
                            }
                        }
                        _ => {
                            self.apply(Input::Message(msg)).await?;
                        }
                    }
                    self.maybe_force_accounted().await?;
                }
                Ok(None) => break, // control plane closed
                Err(_) => {
                    // Quiescence: a still-expected peer has gone silent — force the current round so
                    // the run can make progress (and eventually drop the silent peer).
                    self.force_current_round().await?;
                }
            }
        }

        Ok(CoordinatorReplay {
            initial: self.initial,
            inputs: self.inputs,
            states_by_round: self.states_by_round,
        })
    }
}

/// Run an in-process swarm (N peers + the real coordinator `tick`) for `cfg.num_rounds` rounds over
/// the deterministic [`StubBackend`]. Deterministic: same `cfg` → byte-identical digest transcript.
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
    let peer_ids: Vec<PeerId> = keys.iter().map(peer_id).collect::<Vec<_>>();
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

    // The real coordinator tick loop (subscribes at construction, before it opens round 0).
    let coordinator =
        TickCoordinator::new(gossip.clone(), fs.clone(), coordinator_key(), run, &cfg);
    let coord_handle: JoinHandle<Result<CoordinatorReplay, SwarmRunError>> =
        tokio::spawn(async move { coordinator.drive().await });

    // Collect events until every peer has completed the final round (or left), with a safety
    // timeout so a silent/left peer cannot hang the harness.
    let last_round = cfg.num_rounds.saturating_sub(1);
    let mut events: Vec<(PeerId, EngineEvent)> = Vec::new();
    let mut done: BTreeSet<PeerId> = BTreeSet::new();
    loop {
        if roster.iter().all(|p| done.contains(p)) {
            break;
        }
        match tokio::time::timeout(Duration::from_secs(15), col_rx.recv()).await {
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

    // The coordinator has published every round by now; capture its replay trajectory, then tear
    // the peers down and drain any tail events.
    let replay = coord_handle.await.ok().and_then(Result::ok);
    for h in &peer_handles {
        h.abort();
    }
    for h in fwd_handles {
        h.abort();
    }
    while let Ok((peer, ev)) = col_rx.try_recv() {
        events.push((peer, ev));
    }

    Ok(SwarmRun {
        roster,
        events,
        replay,
    })
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
