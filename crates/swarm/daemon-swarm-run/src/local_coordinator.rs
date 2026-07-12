// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`LocalCoordinator`] — the runnable local-mode coordinator shell (spec §6.2, §10.4, §11.2).
//!
//! Wave 3 graduates the Merge-2 test-only `TickCoordinator` fixture into this public library module:
//! the **impure shell** around lane P2's pure [`tick`](daemon_swarm_coordinator::tick). The pure
//! state machine stays in `daemon-swarm-coordinator`; this shell owns the impure edges the pure
//! function refuses to touch —
//!
//! - **clock**: it feeds `Input::Clock` to drive the timeout-based phase transitions (warmup, the
//!   forced-round path, cooldown), event-driven on the happy path so the transcript is deterministic;
//! - **signing**: `tick` emits *unsigned* `RoundOpen`/`RoundRecord` [`Output::Publish`] values; the
//!   shell signs them with the coordinator identity and broadcasts them over a [`ControlPlane`];
//! - **receipt production**: on each `Commitment` it `HEAD`s the shared [`FsPayloadStore`] and feeds
//!   a signed `StorageReceipt` — the coordinator-as-storage-client availability evidence path
//!   (§6.4 I6), the load-bearing evidence source the Merge-2 integration proved out (witness-quorum
//!   coverage alone under-covers small / stalled rosters).
//!
//! It also drives **admission**: it synthesizes each peer's signed `Join` (the frozen `RoundEngine`
//! never joins), so the roster forms through the real admission path, and it re-admits late peers at
//! the next epoch boundary (the late-join drill).
//!
//! ## Determinism + replay (PROTO-20)
//!
//! `tick` is pure, so the same input sequence yields the same `(state, outputs)` and the digest
//! transcript is arrival-order-independent. [`CoordinatorReplay`] records the exact `Input` sequence
//! + a canonical-CBOR `CoordinatorState` snapshot after each `RoundRecord`; [`CoordinatorReplay::verify`]
//! re-runs `tick` offline and proves a byte-identical per-round state trajectory.
//!
//! ## Restart (practical PROTO-20)
//!
//! Because `CoordinatorState` round-trips through canonical CBOR byte-identically, a coordinator
//! process restart is transparent: [`LocalCoordinator::snapshot`] serializes the live state (as a
//! node would persist it each round), and [`LocalCoordinator::reload_state`] rebuilds the in-memory
//! state from those bytes — exactly what a fresh shell reloads from disk. The restart drill exercises
//! this mid-run via [`LocalCoordinatorConfig::restart_after_round`].

#![allow(clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use daemon_swarm_coordinator::{tick, CoordinatorState, Input, Notice, Output, Phase};
use daemon_swarm_net::{ControlPlane, FsPayloadStore, PayloadStore};
use daemon_swarm_proto::messages::{
    Commitment, Join, RecordEntry, StorageReceipt, Straggle, StraggleStatus, ThroughputClass,
};
use daemon_swarm_proto::{
    from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, IrohId, PeerId, SignedMessage,
    SigningKey, SwarmMessage, SwarmProtoVersion,
};

use crate::seam::{PayloadKey, RoundId, RunId};
use crate::SwarmRunError;

/// A recorded coordinator run trajectory for the offline replay assertion (PROTO-20 spirit).
///
/// Holds the exact ordered [`Input`] sequence the [`LocalCoordinator`] fed its pure `tick`, the
/// initial [`CoordinatorState`], and a canonical-CBOR snapshot of the state taken right after each
/// `RoundRecord` was emitted. [`CoordinatorReplay::verify`] re-runs `tick` over the recorded inputs
/// from the initial state and asserts a byte-identical per-round state trajectory.
#[derive(Clone, Debug)]
pub struct CoordinatorReplay {
    initial: CoordinatorState,
    inputs: Vec<Input>,
    states_by_round: BTreeMap<RoundId, Vec<u8>>,
    reloads: u32,
    dropped: BTreeSet<PeerId>,
}

impl CoordinatorReplay {
    /// The number of rounds whose post-record state was snapshotted.
    #[must_use]
    pub fn recorded_rounds(&self) -> usize {
        self.states_by_round.len()
    }

    /// How many times the shell reloaded its state from canonical CBOR mid-run (the restart drill).
    #[must_use]
    pub fn reloads(&self) -> u32 {
        self.reloads
    }

    /// The peers the coordinator dropped after K record-absences (the silent-death drill, §6.4).
    #[must_use]
    pub fn dropped(&self) -> &BTreeSet<PeerId> {
        &self.dropped
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

/// Construction inputs for a [`LocalCoordinator`].
///
/// The caller builds the initial [`CoordinatorState`] (from a `RunConfig` — the harness — or from a
/// frozen `Envelope` via `RunConfig::from_envelope` — the `swarm-local` runner) and supplies the
/// coordinator identity + the deterministic per-peer bootstrap keys the shell synthesizes `Join`s
/// from.
#[derive(Clone)]
pub struct LocalCoordinatorConfig {
    /// The run this coordinator drives.
    pub run: RunId,
    /// The coordinator's node identity (signs `RoundOpen`/`RoundRecord` + `StorageReceipt` evidence).
    pub key: SigningKey,
    /// The pinned swarm proto version.
    pub version: SwarmProtoVersion,
    /// The initial coordinator state (phase `WaitingForMembers`).
    pub state: CoordinatorState,
    /// Signing keys of the peers admitted at run start (one synthesized `Join` each).
    pub bootstrap_keys: Vec<SigningKey>,
    /// Signing keys of peers admitted at the first epoch boundary (the late-join drill).
    pub late_keys: Vec<SigningKey>,
    /// The quiescence guard: how long to wait for control traffic before force-progressing a round
    /// (covers a peer gone fully silent — it never fires on the happy path).
    pub quiescence: Duration,
    /// If set, reload the state from canonical CBOR right after finalizing this round (restart drill).
    pub restart_after_round: Option<RoundId>,
}

/// The iroh id + class every synthesized `Join` carries (the in-process peers are class-equal).
const JOIN_IROH_ID: IrohId = IrohId([0x22; 32]);

/// The impure shell around the pure [`tick`]: signs + publishes coordinator outputs, supplies
/// `StorageReceipt` availability evidence, synthesizes joins, and drives finalization deterministically.
pub struct LocalCoordinator<C> {
    control: Arc<C>,
    store: Arc<FsPayloadStore>,
    key: SigningKey,
    version: SwarmProtoVersion,
    run: RunId,
    bootstrap_keys: Vec<SigningKey>,
    late_keys: Vec<SigningKey>,
    quiescence: Duration,
    restart_after_round: Option<RoundId>,
    /// The pure coordinator state (in an `Option` so `tick` can take it by value + return it).
    state: Option<CoordinatorState>,
    /// Peers whose commitment for a round has been fed **and** receipted (evidenced).
    committed: BTreeMap<RoundId, BTreeSet<PeerId>>,
    /// Peers that reported a `Straggle(Stalled)` (skipped training) for a round.
    stalled: BTreeMap<RoundId, BTreeSet<PeerId>>,
    /// Peers already admitted (so an epoch re-entry does not double-join them).
    joined: BTreeSet<PeerId>,
    /// The epoch the shell last drove through `WaitingForMembers` (stuck-loop guard).
    entered_epoch: Option<u64>,
    /// Whether the late keys have been admitted yet.
    late_admitted: bool,
    reloads: u32,
    /// Peers the coordinator dropped after K record-absences (§6.4).
    dropped: BTreeSet<PeerId>,
    /// The ordered input log + per-round state snapshots for the replay assertion.
    inputs: Vec<Input>,
    states_by_round: BTreeMap<RoundId, Vec<u8>>,
    initial: CoordinatorState,
}

impl<C: ControlPlane> LocalCoordinator<C> {
    /// Build a shell over `control` + the shared `store`, ready to [`LocalCoordinator::drive`].
    #[must_use]
    pub fn new(control: Arc<C>, store: Arc<FsPayloadStore>, cfg: LocalCoordinatorConfig) -> Self {
        Self {
            control,
            store,
            key: cfg.key,
            version: cfg.version,
            run: cfg.run,
            bootstrap_keys: cfg.bootstrap_keys,
            late_keys: cfg.late_keys,
            quiescence: cfg.quiescence,
            restart_after_round: cfg.restart_after_round,
            state: Some(cfg.state.clone()),
            committed: BTreeMap::new(),
            stalled: BTreeMap::new(),
            joined: BTreeSet::new(),
            entered_epoch: None,
            late_admitted: false,
            reloads: 0,
            dropped: BTreeSet::new(),
            inputs: Vec::new(),
            states_by_round: BTreeMap::new(),
            initial: cfg.state,
        }
    }

    fn state(&self) -> &CoordinatorState {
        self.state.as_ref().expect("coordinator state present")
    }

    /// The current lifecycle phase (observability / drill hooks).
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.state().phase
    }

    /// Serialize the live coordinator state to canonical CBOR — what a node persists each round so a
    /// restart can reload it (spec §11.2 / PROTO-20).
    pub fn snapshot(&self) -> Result<Vec<u8>, SwarmRunError> {
        to_canonical_vec(self.state())
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator snapshot: {e}")))
    }

    /// Reload the coordinator state from a [`LocalCoordinator::snapshot`] blob, discarding the
    /// in-memory round scratch — exactly what a freshly-restarted shell reloads from disk. The
    /// control-plane subscription is unaffected (transport reconnect is orthogonal).
    pub fn reload_state(&mut self, bytes: &[u8]) -> Result<(), SwarmRunError> {
        let state: CoordinatorState = from_canonical_slice(bytes)
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator reload: {e}")))?;
        self.state = Some(state);
        self.committed.clear();
        self.stalled.clear();
        self.reloads += 1;
        Ok(())
    }

    /// Feed one input to the pure `tick`, record it, and sign + publish any emitted messages.
    async fn apply(&mut self, input: Input) -> Result<(), SwarmRunError> {
        self.inputs.push(input.clone());
        let state = self.state.take().expect("coordinator state present");
        let (state, outputs) = tick(state, input);
        self.state = Some(state);
        for o in outputs {
            match o {
                Output::Publish(msg) => {
                    let payload = *msg;
                    let record_round = match &payload {
                        SwarmMessage::RoundRecord(rr) => Some(rr.round),
                        _ => None,
                    };
                    let signed = SignedMessage::sign(&self.key, self.version, payload)
                        .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator sign: {e}")))?;
                    let bytes = to_canonical_vec(&signed).map_err(|e| {
                        SwarmRunError::Lifecycle(format!("coordinator encode: {e}"))
                    })?;
                    self.control.publish(&bytes).await?;
                    if let Some(r) = record_round {
                        if let Ok(sbytes) = to_canonical_vec(self.state()) {
                            self.states_by_round.insert(r, sbytes);
                        }
                    }
                }
                Output::Note(Notice::Dropped(p)) => {
                    self.dropped.insert(p);
                }
                Output::Note(_) | Output::Reject(_) => {}
            }
        }
        Ok(())
    }

    /// Synthesize + feed a signed `Join` for `key` (the frozen `RoundEngine` never joins).
    async fn feed_join(&mut self, key: &SigningKey) -> Result<(), SwarmRunError> {
        // Assert the run's frozen-envelope hash so the coordinator's envelope-hash admission check
        // (§6.5, Wave-3 carrier) is exercised end-to-end: a matching hash is admitted, a peer that
        // assessed a different envelope would be rejected.
        let envelope_hash = self.state().config.envelope_hash;
        let join = Join {
            run_id: self.run.as_str().to_string(),
            iroh_id: JOIN_IROH_ID,
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: Some(envelope_hash),
        };
        let signed = SignedMessage::sign(key, self.version, SwarmMessage::Join(join))
            .map_err(|e| SwarmRunError::Lifecycle(format!("join sign: {e}")))?;
        self.apply(Input::Message(signed)).await?;
        self.joined.insert(peer_id(key));
        Ok(())
    }

    /// In `WaitingForMembers`: admit any not-yet-joined peers (bootstrap at epoch 0, late peers at
    /// the first epoch boundary), then clock past warmup so the next round opens.
    async fn enter_members(&mut self) -> Result<(), SwarmRunError> {
        // Bootstrap keys always; late keys once the run has crossed into a new epoch.
        let mut to_join: Vec<SigningKey> = self
            .bootstrap_keys
            .iter()
            .filter(|k| !self.joined.contains(&peer_id(k)))
            .cloned()
            .collect();
        if self.state().epoch >= 1 && !self.late_admitted {
            for k in &self.late_keys {
                if !self.joined.contains(&peer_id(k)) {
                    to_join.push(k.clone());
                }
            }
            self.late_admitted = true;
        }
        for k in to_join {
            self.feed_join(&k).await?;
        }
        self.entered_epoch = Some(self.state().epoch);
        // WaitingForMembers → Warmup (needs healthy >= min_peers), then Warmup → RoundTrain.
        let now = self.state().now_s;
        let warmup = self.state().config.warmup_s;
        self.apply(Input::Clock(now + 1)).await?;
        self.apply(Input::Clock(now + warmup + 2)).await?;
        Ok(())
    }

    /// Force the current round to finalize via the round/witness timeouts (a straggler or a silent
    /// peer will not commit, so the event-driven fast path cannot fire).
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
    /// (committed + receipted, or `Straggle(Stalled)` this round) yet the round has not auto-finalized.
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

    /// Produce + feed a signed `StorageReceipt` for `(round, peer)` if the object is in the store.
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

    /// Reload the state from canonical CBOR if the restart drill targets `round` (once).
    async fn maybe_restart(&mut self, round: RoundId) -> Result<(), SwarmRunError> {
        if self.restart_after_round == Some(round) {
            let bytes = self.snapshot()?;
            self.reload_state(&bytes)?;
            self.restart_after_round = None; // restart once
        }
        Ok(())
    }

    /// Whether the run has reached a terminal phase.
    fn finished(&self) -> bool {
        matches!(
            self.state().phase,
            Phase::Finished | Phase::Uninitialized | Phase::Paused
        )
    }

    /// Drive the run to completion: admit + warmup at each `WaitingForMembers`, feed inbound signed
    /// peer messages (+ receipts + accounted/quiescence-forced clocks), clock cooldown boundaries,
    /// until the run reaches `Finished`.
    pub async fn drive(mut self) -> Result<CoordinatorReplay, SwarmRunError> {
        let mut sub = self.control.subscribe();

        loop {
            if self.finished() {
                break;
            }
            match self.state().phase {
                Phase::WaitingForMembers => {
                    // Stuck guard: if we already tried to admit this epoch and still cannot reach
                    // min_peers, the run cannot continue — stop driving.
                    if self.entered_epoch == Some(self.state().epoch)
                        && self.state().healthy_count() < self.state().config.min_peers
                    {
                        break;
                    }
                    self.enter_members().await?;
                    continue;
                }
                Phase::Cooldown => {
                    let now = self.state().now_s + self.state().config.cooldown_s + 1;
                    self.apply(Input::Clock(now)).await?;
                    continue;
                }
                _ => {}
            }

            match tokio::time::timeout(self.quiescence, sub.recv()).await {
                Ok(Some(bytes)) => {
                    let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                        continue;
                    };
                    if msg.verify_for_run(self.version).is_err() {
                        continue;
                    }
                    let signer = msg.signer;
                    match &msg.payload {
                        // Skip the coordinator's own republished outputs (echoed back over gossip).
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
                    let round = self.state().round.saturating_sub(1);
                    self.maybe_force_accounted().await?;
                    self.maybe_restart(round).await?;
                }
                Ok(None) => break, // control plane closed
                Err(_) => {
                    // Quiescence: a still-expected peer has gone silent — force the current round so
                    // the run can make progress (and eventually drop the silent peer).
                    let before = self.state().round;
                    self.force_current_round().await?;
                    if self.state().round != before {
                        self.maybe_restart(before).await?;
                    }
                }
            }
        }

        Ok(CoordinatorReplay {
            initial: self.initial,
            inputs: self.inputs,
            states_by_round: self.states_by_round,
            reloads: self.reloads,
            dropped: self.dropped,
        })
    }
}
