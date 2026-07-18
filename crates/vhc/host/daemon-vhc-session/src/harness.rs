// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! In-process multi-peer harness driven by the production **wasm coordinator** (spec §6.2/§6.4;
//! architecture §4.1: consensus is a sandboxed, content-addressed module).
//!
//! The harness spins up N in-process [`RoundEngine`] peers over a shared [`LoopbackGossip`] control
//! plane + a shared [`FsPayloadStore`], plus the [`crate::wasm_coordinator_shell`] recording drive
//! around the `coordinator-quorum` module. It collects every peer's [`EngineEvent`] and the
//! coordinator's driving trace, and drives the **churn/failure drills** over the same machinery via
//! the extended [`VhcConfig`]:
//!
//! - [`StallFault`] — a single peer misses one object for N gets (the §6.4 stall ladder);
//! - [`StoreOutage`] — a peer's whole-round `get`s are denied for a window (store outage);
//! - [`SilentDeath`] — a peer goes silent mid-run (no `Straggle`) → dropped after K absences;
//! - [`LateJoin`] — a peer joins at the next epoch boundary, resyncs from a checkpoint, contributes;
//! - `restart_after_round` — the coordinator module is re-instantiated from its EXPORTED state
//!   mid-run (quiesce → snapshot → `da_migrate`, ABI §10.2).
//!
//! The coordinator-as-storage-client `StorageReceipt` evidence path, the event-count clock
//! discipline (one tick per delivered frame; filler-frame deadline forcing gated on the
//! accounted-set state predicate), and the recording drive live in
//! [`crate::wasm_coordinator_shell`].

#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;

use daemon_vhc_net::{
    ControlPlane, FsPayloadStore, LoopbackGossip, PayloadStat, PayloadStore, VhcNetError,
};
use daemon_vhc_observe::{
    digest_tally, logged_round_records, replay_capture, DesyncVerdict, MessageLog, ObserveError,
    RunCapture, RunHealth,
};
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::messages::{RecordEntry, RoundRecord};
use daemon_vhc_proto::{
    from_canonical_slice, peer_id, CapabilitySet, Hash, PeerId, Seed, SignedMessage, SigningKey,
    VhcMessage, VhcProtoVersion, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorParams, CoordinatorState, RunConfig};

use crate::backend::{StateDigest, StubBackend, TrainerBackend};
use crate::checkpoint::CheckpointManifest;
use crate::engine::{EngineConfig, EngineEvent, RoundEngine};
use crate::seam::{PayloadKey, RoundId, RunId};
use crate::VhcRunError;

pub use crate::wasm_coordinator_shell::CoordinatorReplay;

/// The injected-fault mode a [`FaultyStore`] plays (TEST-ONLY).
enum Fault {
    /// Withhold one object (`key`) for its first `n` `get`s (the single-object stall, §6.4).
    WithholdKey { key: PayloadKey, n: u32 },
    /// Deny every `get` for `round` for the first `n` calls (a payload-store outage window).
    RoundOutage { round: RoundId, n: u32 },
}

/// A [`PayloadStore`] wrapper that injects deterministic, wall-clock-independent fetch faults to
/// drive the §6.4 stall ladder / store-outage drills (TEST-ONLY).
///
/// The faulted `get`s return a typed [`VhcNetError::PayloadMiss`] for their first `n` calls (the
/// stalling peer's prefetch + barrier attempts), then delegate — so a subsequent cross-round
/// catch-up attempt succeeds. All other keys (and `put`/`head`) delegate unchanged.
pub struct FaultyStore {
    inner: Arc<FsPayloadStore>,
    fault: Option<Fault>,
    gets: AtomicU32,
}

impl FaultyStore {
    /// A transparent wrapper (no fault).
    #[must_use]
    pub fn transparent(inner: Arc<FsPayloadStore>) -> Self {
        Self {
            inner,
            fault: None,
            gets: AtomicU32::new(0),
        }
    }

    /// Withhold `key` for its first `first_n_gets` `get` calls, then serve it.
    #[must_use]
    pub fn withholding(inner: Arc<FsPayloadStore>, key: PayloadKey, first_n_gets: u32) -> Self {
        Self {
            inner,
            fault: Some(Fault::WithholdKey {
                key,
                n: first_n_gets,
            }),
            gets: AtomicU32::new(0),
        }
    }

    /// Deny every `get` for `round` for the first `first_n_gets` calls (a store outage window).
    #[must_use]
    pub fn outage(inner: Arc<FsPayloadStore>, round: RoundId, first_n_gets: u32) -> Self {
        Self {
            inner,
            fault: Some(Fault::RoundOutage {
                round,
                n: first_n_gets,
            }),
            gets: AtomicU32::new(0),
        }
    }

    /// Whether `key` is faulted right now; increments the shared miss counter if it fires.
    fn should_miss(&self, key: &PayloadKey) -> bool {
        let (matches, n) = match &self.fault {
            Some(Fault::WithholdKey { key: k, n }) => (key == k, *n),
            Some(Fault::RoundOutage { round, n }) => (key.round == *round, *n),
            None => return false,
        };
        if !matches {
            return false;
        }
        let seen = self.gets.fetch_add(1, Ordering::SeqCst);
        seen < n
    }
}

#[async_trait]
impl PayloadStore for FaultyStore {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<Hash, VhcNetError> {
        self.inner.put(key, bytes).await
    }

    async fn get(&self, key: &PayloadKey, expected: &Hash) -> Result<Vec<u8>, VhcNetError> {
        if self.should_miss(key) {
            return Err(VhcNetError::PayloadMiss(format!(
                "injected miss for {}@r{}/{}",
                key.run.as_str(),
                key.round,
                key.peer.to_hex()
            )));
        }
        self.inner.get(key, expected).await
    }

    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, VhcNetError> {
        self.inner.head(key).await
    }
}

/// A single-object payload miss injected at a peer/round to drive the stall ladder (TEST-ONLY).
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

/// A payload-store outage window at one peer: every `get` for `round` is denied for `first_n_gets`
/// calls, then recovers (the stall ladder absorbs it).
#[derive(Clone, Copy, Debug)]
pub struct StoreOutage {
    /// The roster index of the peer whose store is out.
    pub peer_index: usize,
    /// The round whose objects are unavailable during the outage.
    pub round: RoundId,
    /// How many `get`s are denied before the store recovers.
    pub first_n_gets: u32,
}

/// A silent peer death: after this peer reports `after_round`, its engine task is aborted (no
/// `Straggle`, no further commitments) → the coordinator drops it after K record-absences (§6.4).
#[derive(Clone, Copy, Debug)]
pub struct SilentDeath {
    /// The roster index of the peer that dies.
    pub peer_index: usize,
    /// The last round the peer completes before going silent.
    pub after_round: RoundId,
}

/// A late peer that joins at the first epoch boundary, resyncs from the previous epoch's checkpoint,
/// and contributes from there (§6.4 admission + §9 rejoin). Requires `epoch_rounds > 0` and
/// `checkpoint_every_rounds` covering `epoch_rounds - 1`.
#[derive(Clone, Copy, Debug)]
pub struct LateJoin {
    /// The checkpoint round the late peer resyncs from (the last round of epoch 0).
    pub resume_round: RoundId,
}

/// Configuration for an in-process vhc run.
#[derive(Clone, Debug)]
pub struct VhcConfig {
    /// Number of peers admitted at run start.
    pub num_peers: usize,
    /// Total number of rounds the coordinator drives (`[data].stop = Rounds(num_rounds)`).
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
    /// Minimum healthy peers to open/keep a round (`None` = `num_peers`).
    pub min_peers: Option<u32>,
    /// Rounds per epoch (0 = a single epoch for the whole run — no mid-run roster refreeze).
    pub epoch_rounds: u64,
    /// K record-absences before a genuinely-absent peer is dropped (§6.4).
    pub k_absences: u32,
    /// Optional single-object stall.
    pub fault: Option<StallFault>,
    /// Optional payload-store outage window.
    pub outage: Option<StoreOutage>,
    /// Optional silent peer death.
    pub silent_death: Option<SilentDeath>,
    /// Optional late peer that joins at the first epoch boundary.
    pub late_join: Option<LateJoin>,
    /// If set, reload the coordinator state from canonical CBOR after finalizing this round.
    pub restart_after_round: Option<RoundId>,
    /// Enable the E1 typed checkpoint lane (Phase E): each cadence checkpoint also publishes the
    /// typed sectioned manifest and emits `EngineEvent::TypedCheckpointed`. Default off — every
    /// pre-E harness run is behavior-identical.
    pub typed_checkpoints: bool,
}

impl Default for VhcConfig {
    fn default() -> Self {
        Self {
            num_peers: 3,
            num_rounds: 20,
            steps_per_round: 2,
            micro_batch: 2,
            stall_rounds_max: 2,
            checkpoint_every_rounds: 0,
            corpus_seed: 0xDAE0_7E57,
            min_peers: None,
            epoch_rounds: 0,
            k_absences: 1,
            fault: None,
            outage: None,
            silent_death: None,
            late_join: None,
            restart_after_round: None,
            typed_checkpoints: false,
        }
    }
}

impl VhcConfig {
    /// A small, fault-free default (3 peers, `num_rounds` rounds).
    #[must_use]
    pub fn small(num_rounds: u64) -> Self {
        Self {
            num_rounds,
            ..Self::default()
        }
    }

    /// The effective `min_peers` floor.
    #[must_use]
    fn min_peers(&self) -> u32 {
        self.min_peers.unwrap_or(self.num_peers as u32)
    }

    /// The maximum roster size across the run (bootstrap + any late joiner) — sizes the global batch.
    #[must_use]
    fn max_roster(&self) -> u32 {
        self.num_peers as u32 + u32::from(self.late_join.is_some())
    }
}

/// The collected outcome of a vhc run: every `(peer, event)` the engines emitted, the coordinator
/// replay, and (for the drills) the shared store + captured round records.
#[derive(Clone)]
pub struct VhcRun {
    /// The sorted bootstrap roster (node identities).
    pub roster: Vec<PeerId>,
    /// Every emitted event, tagged with the peer that emitted it (collection order).
    pub events: Vec<(PeerId, EngineEvent)>,
    /// The coordinator's recorded `tick` trajectory for the replay assertion, if captured.
    pub replay: Option<CoordinatorReplay>,
    /// The shared payload store (checkpoints + payloads), for offline resync in the desync drill.
    pub store: Arc<FsPayloadStore>,
    /// The run id (payload-key component).
    pub run: RunId,
    /// The committed set per round, captured from the coordinator's `RoundRecord`s.
    pub records: BTreeMap<RoundId, Vec<RecordEntry>>,
    /// The node-visible signed-message log (every verified `SignedMessage` seen on the control
    /// plane, arrival order) — the `daemon-vhc-observe` audit artifact `--observe` writes.
    pub message_log: MessageLog,
}

impl VhcRun {
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
    /// [`VhcRun::all_agree`] to assert every peer matched it.
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

    /// The observe-driven [`DesyncVerdict`] for `round`: folds the peers' reported digests through
    /// [`daemon_vhc_observe::digest_tally`] (§9), the authoritative desync trigger. `quorum` is the
    /// number of agreeing peers a digest needs to count as the quorum digest (e.g.
    /// [`daemon_vhc_sdk_consensus::witness_quorum`] of the roster).
    ///
    /// This replaces R3's Wave-3 local quorum-digest fold stand-in with observe's shared verdict, so
    /// the harness, the drills, and the (future) live coordinator path all consume one detector.
    #[must_use]
    pub fn desync_verdict(&self, round: RoundId, quorum: u32) -> DesyncVerdict {
        let reports = self
            .digests_by_round()
            .get(&round)
            .map(|peers| {
                peers
                    .iter()
                    // Bridge the run-local `StateDigest` to proto's (both `[u8; 16]`), the type
                    // observe's tally speaks.
                    .map(|(p, d)| (*p, daemon_vhc_proto::StateDigest(*d.as_bytes())))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        digest_tally(round, reports, quorum)
    }

    /// The majority (quorum) digest per round, via observe's [`digest_tally`] (`quorum = 1` ⇒ the
    /// plurality digest). Returned in the run-local [`StateDigest`] newtype for the drill assertions.
    #[must_use]
    pub fn quorum_digests(&self) -> BTreeMap<RoundId, StateDigest> {
        let mut out = BTreeMap::new();
        for round in self.digests_by_round().keys().copied() {
            if let Some(q) = self.desync_verdict(round, 1).quorum_digest {
                out.insert(round, StateDigest(q.0));
            }
        }
        out
    }

    /// The peers whose reported digest for `round` disagrees with the quorum (desync outliers), from
    /// observe's [`DesyncVerdict`] (`quorum = 1` ⇒ any minority peer is an outlier).
    #[must_use]
    pub fn desync_outliers(&self, round: RoundId) -> BTreeSet<PeerId> {
        self.desync_verdict(round, 1).outliers.into_iter().collect()
    }

    /// The set of peers that left the run (stall budget exhausted).
    #[must_use]
    pub fn left_peers(&self) -> BTreeSet<PeerId> {
        self.events
            .iter()
            .filter_map(|(peer, ev)| matches!(ev, EngineEvent::Left { .. }).then_some(*peer))
            .collect()
    }

    /// The peers the coordinator dropped after K record-absences (§6.4), from the replay.
    #[must_use]
    pub fn dropped_peers(&self) -> BTreeSet<PeerId> {
        self.replay
            .as_ref()
            .map(|r| r.dropped().clone())
            .unwrap_or_default()
    }

    /// The observe [`RunCapture`] for this run (the coordinator's reproducible `tick` driver trace),
    /// if a coordinator replay was captured. Feeds [`daemon_vhc_observe::replay_capture`].
    #[must_use]
    pub fn run_capture(&self) -> Option<RunCapture> {
        self.replay
            .as_ref()
            .map(|r| RunCapture::new(r.initial_state().clone(), r.inputs().to_vec()))
    }

    /// Write the `daemon-vhc-observe` artifacts for this run into `dir` (the `--observe <dir>`
    /// gate-ceremony instrumentation): `<run_id>.dsmlog` (the node-visible [`MessageLog`]) and, if a
    /// coordinator replay was captured, `<run_id>.dsmcap` (the [`RunCapture`] the replay oracle verifies).
    ///
    /// # Errors
    ///
    /// [`ObserveError`] on a directory-create / file-write / codec failure.
    // Plain local-fs writes to an **operator-supplied** gate-ceremony directory (not an
    // attacker-influenced path); the `harness` feature makes this dev/gate-only. The clippy fs ban
    // targets attacker-influenced paths (routed through `ContainedRoot`); network stays separately
    // locked (disallowed-TYPES), so this narrow fs escape cannot re-open anything else.
    #[allow(clippy::disallowed_methods)]
    pub fn write_observe(&self, dir: &std::path::Path) -> Result<(), ObserveError> {
        std::fs::create_dir_all(dir).map_err(|e| ObserveError::Store(e.to_string()))?;
        let base = self.run.as_str();
        let mut log_file = std::fs::File::create(dir.join(format!("{base}.dsmlog")))
            .map_err(|e| ObserveError::Store(e.to_string()))?;
        self.message_log.write_to(&mut log_file)?;
        if let Some(capture) = self.run_capture() {
            let mut cap_file = std::fs::File::create(dir.join(format!("{base}.dsmcap")))
                .map_err(|e| ObserveError::Store(e.to_string()))?;
            capture.write_to(&mut cap_file)?;
        }
        Ok(())
    }
}

/// The verdict of a [`verify_observe_dir`] replay: the run re-derived from its recorded artifacts.
#[derive(Clone, Debug)]
pub struct ObserveVerify {
    /// The run id read from the recorded log.
    pub run_id: String,
    /// How many recorded `RoundRecord`s re-derived byte-identically (the digest-equality count).
    pub rounds_verified: u64,
    /// How many `RoundRecord`s the wire log carried (the oracle count `rounds_verified` must match).
    pub logged_records: usize,
    /// The run-health projection distilled from the recorded message log.
    pub health: RunHealth,
}

impl ObserveVerify {
    /// Whether every logged round record re-derived (full digest-equality over the run).
    #[must_use]
    pub fn all_verified(&self) -> bool {
        self.rounds_verified as usize == self.logged_records && self.logged_records > 0
    }
}

/// Replay + verify the `daemon-vhc-observe` artifacts written by [`VhcRun::write_observe`] into
/// `dir` (the replay entry point): re-derive the recorded [`RunCapture`] inside the
/// sandboxed `coordinator-quorum` module (never a native tick) and assert its `RoundRecord`s
/// re-derive byte-identically against the independent wire [`MessageLog`] (§6.4 I1 / PROTO-20). Also
/// projects [`RunHealth`] from the log.
///
/// # Errors
///
/// [`VhcRunError::Lifecycle`] if the artifacts are missing/corrupt, or if the replay diverges
/// (the first divergent round is included in the message).
// Reads back the artifacts `write_observe` wrote to an operator-supplied gate directory (same
// dev/gate-only, not-attacker-influenced rationale as `write_observe`).
#[allow(clippy::disallowed_methods)]
pub fn verify_observe_dir(dir: &std::path::Path) -> Result<ObserveVerify, VhcRunError> {
    let mut cap_path = None;
    for entry in std::fs::read_dir(dir)
        .map_err(|e| VhcRunError::Lifecycle(format!("observe dir {}: {e}", dir.display())))?
    {
        let path = entry
            .map_err(|e| VhcRunError::Lifecycle(format!("observe dir entry: {e}")))?
            .path();
        if path.extension().is_some_and(|e| e == "dsmcap") {
            cap_path = Some(path);
            break;
        }
    }
    let cap_path = cap_path.ok_or_else(|| {
        VhcRunError::Lifecycle(format!("no .dsmcap capture in {}", dir.display()))
    })?;
    let log_path = cap_path.with_extension("dsmlog");

    let mut cap_file = std::fs::File::open(&cap_path)
        .map_err(|e| VhcRunError::Lifecycle(format!("open {}: {e}", cap_path.display())))?;
    let capture = RunCapture::read_from(&mut cap_file)
        .map_err(|e| VhcRunError::Lifecycle(format!("read capture: {e}")))?;
    let mut log_file = std::fs::File::open(&log_path)
        .map_err(|e| VhcRunError::Lifecycle(format!("open {}: {e}", log_path.display())))?;
    let log = MessageLog::read_from(&mut log_file)
        .map_err(|e| VhcRunError::Lifecycle(format!("read log: {e}")))?;

    let logged_records = logged_round_records(&log);
    let health = RunHealth::from_log(&log);
    // Re-derive inside the real coordinator module : consensus never runs natively, even here.
    let sandbox = crate::replay_sandbox::WasmCoordinatorSandbox::from_built_guest()
        .map_err(|e| VhcRunError::Lifecycle(format!("coordinator sandbox: {e}")))?;
    let report = replay_capture(&sandbox, capture, &log)
        .map_err(|e| VhcRunError::Lifecycle(format!("replay diverged: {e}")))?;
    Ok(ObserveVerify {
        run_id: log.run_id().to_string(),
        rounds_verified: report.rounds_verified,
        logged_records,
        health,
    })
}

/// A collector task plus the handle that stops it **losslessly**: on [`CollectorStop::stop`] the
/// task drains everything already queued on its subscription before exiting, so messages published
/// before the stop are never dropped. (A bare `JoinHandle::abort` loses whatever the task had not
/// yet been scheduled to pull off the channel — under parallel-suite CPU contention that was
/// routinely the final round's `RoundRecord`, the `observe_record_and_replay_green` flake.)
pub(crate) struct CollectorStop {
    handle: JoinHandle<()>,
    stop: tokio::sync::oneshot::Sender<()>,
}

impl CollectorStop {
    /// Signal the collector to drain its queue and exit, then wait for it.
    pub(crate) async fn stop(self) {
        // A closed receiver (task already gone) is fine — awaiting the handle is the real barrier.
        let _ = self.stop.send(());
        let _ = self.handle.await;
    }
}

/// Spawn the observe message-log collector: subscribe to `control` and append every verified
/// `SignedMessage` to the shared [`MessageLog`] in arrival order (the `--observe` capture). Generic
/// over the control plane so both the loopback and live (iroh) harnesses reuse it.
///
/// Stop it via [`CollectorStop::stop`] once every publisher has quiesced: because publishes enqueue
/// synchronously, the drain-on-stop then deterministically captures the complete wire history.
pub(crate) fn spawn_message_log<C: ControlPlane>(
    control: &Arc<C>,
    version: VhcProtoVersion,
    log: Arc<std::sync::Mutex<MessageLog>>,
) -> CollectorStop {
    let mut sub = control.subscribe();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let append = move |bytes: Vec<u8>| {
            let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                return;
            };
            if msg.verify_for_run(version).is_err() {
                return;
            }
            log.lock().expect("observe log lock").append(msg);
        };
        loop {
            tokio::select! {
                m = sub.recv() => match m {
                    Some(bytes) => append(bytes),
                    None => return, // plane closed: nothing more can arrive
                },
                _ = &mut stop_rx => break,
            }
        }
        // Stop requested: drain what was already enqueued so no published message is lost.
        while let Some(bytes) = sub.try_recv() {
            append(bytes);
        }
    });
    CollectorStop {
        handle,
        stop: stop_tx,
    }
}

/// A deterministic per-peer node identity key (index `i`).
#[must_use]
pub fn peer_key(i: usize) -> SigningKey {
    SigningKey::from_bytes(&[0x11 + i as u8; 32])
}

/// The experiment config bytes every peer builds its backend from (identical across peers, so the
/// consensus round base coincides — the reconvergence precondition, §5.6).
pub const EXPERIMENT_CONFIG: &[u8] = b"e2e-experiment-config";

/// Build the coordinator's resolved [`RunConfig`] for an in-process run of `cfg`.
pub(crate) fn build_run_config(run: &RunId, cfg: &VhcConfig) -> RunConfig {
    // Global batch = max-roster × steps × micro-batch, so the per-peer (class-equal) partition is
    // non-empty for every roster size the run will see (bootstrap and post-late-join).
    let g =
        u64::from(cfg.max_roster()) * u64::from(cfg.steps_per_round) * u64::from(cfg.micro_batch);
    let params = CoordinatorParams {
        seq_len: 1,
        witness_target: 0, // every peer witnesses (matches the engine's witness set)
        overlap_bps: 0,
        k_absences: cfg.k_absences,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    RunConfig {
        run_id: run.as_str().to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: cfg.min_peers(),
        max_peers: cfg.max_roster(),
        warmup_s: 1,
        round_train_max_s: 1,
        round_witness_s: 1,
        cooldown_s: 1,
        epoch_rounds: cfg.epoch_rounds,
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
    }
}

/// Build a peer's payload store, wrapping the shared FS store with any injected fault for peer `i`.
pub(crate) fn peer_store(
    fs: &Arc<FsPayloadStore>,
    run: &RunId,
    peer_ids: &[PeerId],
    cfg: &VhcConfig,
    i: usize,
) -> Arc<FaultyStore> {
    if let Some(f) = cfg.fault {
        if f.peer_index == i {
            let missing = peer_ids[f.missing_peer_index];
            let withheld = PayloadKey::new(run.clone(), f.round, missing);
            return Arc::new(FaultyStore::withholding(
                fs.clone(),
                withheld,
                f.first_n_gets,
            ));
        }
    }
    if let Some(o) = cfg.outage {
        if o.peer_index == i {
            return Arc::new(FaultyStore::outage(fs.clone(), o.round, o.first_n_gets));
        }
    }
    Arc::new(FaultyStore::transparent(fs.clone()))
}

/// Run an in-process vhc (N peers + the real coordinator `tick`) for `cfg.num_rounds` rounds over
/// the deterministic [`StubBackend`]. Deterministic: same `cfg` → byte-identical digest transcript.
pub async fn run_vhc(cfg: VhcConfig) -> Result<VhcRun, VhcRunError> {
    run_vhc_with(cfg, |_i| {
        let mut backend = StubBackend::new();
        backend
            .build(EXPERIMENT_CONFIG)
            .expect("build stub backend");
        backend
    })
    .await
}

/// Like [`run_vhc`], but each peer's [`TrainerBackend`] is produced by `make_backend(index)` — so
/// tests can inject an instrumented backend (barrier assertion, desync injection). The factory MUST
/// return an already-`build`-ed backend (base snapshot set from the shared config).
pub async fn run_vhc_with<B, F>(cfg: VhcConfig, make_backend: F) -> Result<VhcRun, VhcRunError>
where
    B: TrainerBackend + Send + Sync + 'static,
    F: Fn(usize) -> B,
{
    let run = RunId::new("e2e-run");
    let version = VHC_PROTO_VERSION;

    // Bootstrap peer identities + the (larger) full roster including any late joiner.
    let boot_keys: Vec<SigningKey> = (0..cfg.num_peers).map(peer_key).collect();
    let boot_ids: Vec<PeerId> = boot_keys.iter().map(peer_id).collect();
    let mut boot_roster = boot_ids.clone();
    boot_roster.sort_unstable();

    let late_key = cfg.late_join.map(|_| peer_key(cfg.num_peers));
    let mut full_roster = boot_roster.clone();
    if let Some(k) = &late_key {
        full_roster.push(peer_id(k));
    }
    full_roster.sort_unstable();

    let corpus = Arc::new(crate::data::Corpus::synthetic(cfg.corpus_seed, 4, 256, 8)?);
    let gossip = Arc::new(LoopbackGossip::new());

    let root = std::env::temp_dir().join(format!(
        "daemon-vhc-e2e-{}-{}",
        std::process::id(),
        fastcounter()
    ));
    let fs = Arc::new(FsPayloadStore::open(&root, cfg.num_rounds + 8)?);

    let (col_tx, mut col_rx) = unbounded_channel::<(PeerId, EngineEvent)>();
    let mut peer_handles: BTreeMap<
        PeerId,
        JoinHandle<Result<crate::engine::RunOutcome, VhcRunError>>,
    > = BTreeMap::new();
    let mut fwd_handles: Vec<JoinHandle<()>> = Vec::new();

    // Spawn the bootstrap peers.
    for (i, key) in boot_keys.iter().enumerate() {
        let store = peer_store(&fs, &run, &boot_ids, &cfg, i);
        let backend = make_backend(i);
        let (peer, engine_h, fwd_h) = launch_engine(
            &gossip,
            store,
            backend,
            key.clone(),
            &corpus,
            &run,
            version,
            &cfg,
            boot_roster.clone(),
            None,
            &col_tx,
        );
        peer_handles.insert(peer, engine_h);
        fwd_handles.push(fwd_h);
    }

    // A background collector for the coordinator's `RoundRecord`s (for the desync-drill resync).
    let records: Arc<std::sync::Mutex<BTreeMap<RoundId, Vec<RecordEntry>>>> =
        Arc::new(std::sync::Mutex::new(BTreeMap::new()));
    let rec_handle = spawn_record_collector(&gossip, version, records.clone());

    // The observe message-log collector: capture every verified signed message (arrival order) into
    // a shared `MessageLog` — the `--observe` audit artifact + the replay oracle source.
    let obs_log = Arc::new(std::sync::Mutex::new(MessageLog::new(run.as_str())));
    let obs_handle = spawn_message_log(&gossip, version, obs_log.clone());

    // The coordinator drive (subscribes at construction, before it opens round 0): the production
    // coordinator-quorum module (event-driven, one tick per delivered frame), so the recorded run
    // and its replay oracle share one substrate. Any run that can leave a round without a full
    // commitment set (a silent peer to drop, a store outage, a late-join epoch boundary, or a
    // payload fault whose victim skips training while stalled) needs finite phase deadlines the
    // shell can reach with filler frames (the event-count clock discipline); the forcing itself
    // stays gated on the accounted-set state predicate, so an all-committed round always closes
    // on the fast path untouched. Fault-free runs keep the deadlines effectively infinite so the
    // module is driven purely by its event-driven fast paths.
    let force_deadlines = cfg.silent_death.is_some()
        || cfg.outage.is_some()
        || cfg.late_join.is_some()
        || cfg.fault.is_some();
    // The drill's fault plan, in the shell's accounted-set vocabulary: the silent-death peer acts
    // through `after_round` and is deliberately absent thereafter — accounted immediately, so its
    // rounds are deadline-forceable as soon as every healthy peer is accounted.
    let planned_absent: Vec<(PeerId, u64)> = cfg
        .silent_death
        .map(|d| (boot_ids[d.peer_index], d.after_round))
        .into_iter()
        .collect();
    // The frame-ordered late-join barrier: set once the late engine is CONSTRUCTED (its control
    // subscription exists), which is when forcing the epoch boundary becomes safe — the new
    // epoch's RoundOpen is then guaranteed to be queued to it.
    let late_engines_live = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // A phase-deadline value comfortably above the per-round frame count (commitments + authored
    // receipts + straggles) so a healthy round always closes on the fast path first, yet small
    // enough that a bounded filler-frame batch expires it.
    const FORCED_PHASE_S: u64 = 40;
    let mut coord_run_config = build_run_config(&run, &cfg);
    let phase = if force_deadlines {
        FORCED_PHASE_S
    } else {
        1_000_000
    };
    coord_run_config.warmup_s = phase;
    coord_run_config.round_train_max_s = phase;
    coord_run_config.round_witness_s = phase;
    coord_run_config.cooldown_s = phase;
    let coord_state = CoordinatorState::new(coord_run_config, Seed([0xAB; 32]), 0);
    let coord_handle: JoinHandle<Result<CoordinatorReplay, VhcRunError>> = {
        let shell_cfg = crate::wasm_coordinator_shell::WasmCoordinatorShellConfig {
            run: run.clone(),
            version,
            state: coord_state,
            bootstrap_keys: boot_keys.clone(),
            late_keys: late_key.iter().cloned().collect(),
            num_rounds: cfg.num_rounds,
            poll: Duration::from_millis(50),
            deadline: Duration::from_secs(180),
            late_engines_live: late_engines_live.clone(),
            restart_after_round: cfg.restart_after_round,
            force_deadlines,
            planned_absent,
        };
        let shell = crate::wasm_coordinator_shell::WasmCoordinatorShell::new(
            gossip.clone(),
            fs.clone(),
            shell_cfg,
        );
        tokio::spawn(async move { shell.drive().await })
    };

    // Collect events until every expected peer finishes (or leaves / is killed). The timeout is a
    // pure OUTER FAILSAFE (a wedged run cannot hang the harness forever) and feeds no protocol
    // decision — every in-run decision (deadline forcing, late-join release, drops) is event- or
    // state-driven. It is sized generously: under heavy parallel-suite CPU starvation a healthy
    // run can legitimately go quiet for tens of seconds, and giving up early here would cancel the
    // late-engine spawn (an event-driven step this collector performs).
    let last_round = cfg.num_rounds.saturating_sub(1);
    let mut events: Vec<(PeerId, EngineEvent)> = Vec::new();
    let mut done: BTreeSet<PeerId> = BTreeSet::new();
    let mut expected: BTreeSet<PeerId> = boot_ids.iter().copied().collect();
    let mut awaiting_late = cfg.late_join.is_some();
    let mut killed_death = false;

    let death_target = cfg.silent_death.map(|d| boot_ids[d.peer_index]);

    loop {
        if !awaiting_late && expected.iter().all(|p| done.contains(p)) {
            break;
        }
        match tokio::time::timeout(Duration::from_secs(90), col_rx.recv()).await {
            Ok(Some((peer, ev))) => {
                // Late-join: when a bootstrap peer checkpoints the resume round, spawn the late peer.
                if awaiting_late {
                    if let (Some(lj), Some(lk)) = (cfg.late_join, &late_key) {
                        if let EngineEvent::Checkpointed { round, manifest } = &ev {
                            if *round == lj.resume_round {
                                let store = Arc::new(FaultyStore::transparent(fs.clone()));
                                let backend = make_backend(cfg.num_peers);
                                let (lpeer, engine_h, fwd_h) = launch_engine(
                                    &gossip,
                                    store,
                                    backend,
                                    lk.clone(),
                                    &corpus,
                                    &run,
                                    version,
                                    &cfg,
                                    full_roster.clone(),
                                    Some(*manifest),
                                    &col_tx,
                                );
                                peer_handles.insert(lpeer, engine_h);
                                fwd_handles.push(fwd_h);
                                expected.insert(lpeer);
                                awaiting_late = false;
                                // The engine subscribed to the control plane in its constructor
                                // (inside `launch_engine`), so the wasm shell's epoch-boundary
                                // force is now safe to release: the new epoch's RoundOpen will be
                                // queued to the late engine (the frame-ordered roster barrier).
                                late_engines_live.store(true, std::sync::atomic::Ordering::Release);
                            }
                        }
                    }
                }

                // Silent death: once the target reports its last live round, abort its engine.
                if let (Some(d), Some(target)) = (cfg.silent_death, death_target) {
                    if !killed_death && peer == target {
                        let reached = matches!(&ev,
                            EngineEvent::RoundComplete { round, .. } | EngineEvent::CaughtUp { round, .. }
                            if *round >= d.after_round);
                        if reached {
                            if let Some(h) = peer_handles.remove(&target) {
                                h.abort();
                            }
                            expected.remove(&target);
                            killed_death = true;
                        }
                    }
                }

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

    let replay = coord_handle.await.ok().and_then(Result::ok);
    for h in peer_handles.values() {
        h.abort();
    }
    for h in fwd_handles {
        h.abort();
    }
    // Every publisher has quiesced (coordinator awaited, peers done/aborted), so a lossless
    // drain-on-stop leaves the collectors with the complete wire history — an `abort()` here
    // raced the collectors' append loops and intermittently dropped the tail of the log (the
    // final round's `RoundRecord`), the `observe_record_and_replay_green` parallel-load flake.
    rec_handle.stop().await;
    obs_handle.stop().await;
    while let Ok((peer, ev)) = col_rx.try_recv() {
        events.push((peer, ev));
    }

    let records = records.lock().expect("records lock").clone();
    let message_log = obs_log.lock().expect("observe log lock").clone();

    Ok(VhcRun {
        roster: boot_roster,
        events,
        replay,
        store: fs,
        run,
        records,
        message_log,
    })
}

/// Construct + spawn one [`RoundEngine`] (optionally resuming from `resume`) plus its event
/// forwarder. Returns the peer id and both task handles.
#[allow(clippy::too_many_arguments)]
fn launch_engine<B>(
    gossip: &Arc<LoopbackGossip>,
    store: Arc<FaultyStore>,
    backend: B,
    key: SigningKey,
    corpus: &Arc<crate::data::Corpus>,
    run: &RunId,
    version: VhcProtoVersion,
    cfg: &VhcConfig,
    roster: Vec<PeerId>,
    resume: Option<CheckpointManifest>,
    col_tx: &UnboundedSender<(PeerId, EngineEvent)>,
) -> (
    PeerId,
    JoinHandle<Result<crate::engine::RunOutcome, VhcRunError>>,
    JoinHandle<()>,
)
where
    B: TrainerBackend + Send + Sync + 'static,
{
    let peer = peer_id(&key);
    let engine_cfg = EngineConfig {
        run: run.clone(),
        roster: roster.clone(),
        witnesses: roster,
        steps_per_round: cfg.steps_per_round,
        micro_batch: cfg.micro_batch,
        stall_rounds_max: cfg.stall_rounds_max,
        checkpoint_every_rounds: cfg.checkpoint_every_rounds,
        version,
    };

    let (ev_tx, mut ev_rx) = unbounded_channel::<EngineEvent>();
    let col = col_tx.clone();
    let fwd = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            if col.send((peer, ev)).is_err() {
                break;
            }
        }
    });

    let mut engine = RoundEngine::new(
        gossip.clone(),
        store,
        backend,
        key,
        corpus.clone(),
        engine_cfg,
        ev_tx,
    );
    if cfg.typed_checkpoints {
        // The harness's pre-D0 execution identity: run_id = blake3(label), epoch 0, and a
        // domain-separated placeholder module hash (the StubBackend is not a wasm blob).
        engine = engine.with_typed_checkpoints(crate::checkpoint::CheckpointIdent {
            run_id: daemon_vhc_proto::blake3_hash(run.as_str().as_bytes()),
            epoch: 0,
            module: daemon_vhc_proto::blake3_hash(b"daemon-vhc-session/harness/stub-backend"),
        });
    }
    let engine_h = tokio::spawn(async move {
        if let Some(manifest) = resume {
            engine.resume_from_checkpoint(&manifest).await?;
        }
        engine.run().await
    });
    (peer, engine_h, fwd)
}

/// A background task that decodes the coordinator's published `RoundRecord`s off the gossip plane
/// and records their committed sets (for the desync drill's offline resync). Stopped losslessly via
/// [`CollectorStop::stop`] (same drain-on-stop rationale as [`spawn_message_log`]).
fn spawn_record_collector(
    gossip: &Arc<LoopbackGossip>,
    version: VhcProtoVersion,
    records: Arc<std::sync::Mutex<BTreeMap<RoundId, Vec<RecordEntry>>>>,
) -> CollectorStop {
    let mut sub = gossip.subscribe();
    let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let record = move |bytes: Vec<u8>| {
            let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                return;
            };
            if msg.verify_for_run(version).is_err() {
                return;
            }
            if let VhcMessage::RoundRecord(RoundRecord {
                round,
                inline: Some(entries),
                ..
            }) = &msg.payload
            {
                records
                    .lock()
                    .expect("records lock")
                    .entry(*round)
                    .or_insert_with(|| entries.clone());
            }
        };
        loop {
            tokio::select! {
                m = sub.recv() => match m {
                    Some(bytes) => record(bytes),
                    None => return,
                },
                _ = &mut stop_rx => break,
            }
        }
        while let Some(bytes) = sub.try_recv() {
            record(bytes);
        }
    });
    CollectorStop {
        handle,
        stop: stop_tx,
    }
}

/// A tiny process-lifetime counter for unique temp-dir names (no external `tempfile` dep).
pub(crate) fn fastcounter() -> u64 {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos ^ N.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9)
}
