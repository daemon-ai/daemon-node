// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The wasm-coordinator recording drive for the in-process whole-run harness (architecture §4.1,
//! §6.2; spec §6.4 I1).
//!
//! Consensus is a wasm module, not a native host service — so the harness records a run by driving
//! the production `coordinator-quorum` module (the same content-addressed blob the replay oracle
//! re-derives through), never a native `tick`. The module owns a deterministic logical clock (one
//! tick per delivered frame), so the run is driven **event-driven**: members are admitted with
//! synthesized joins, warmup exits on readiness heartbeats, and each round closes on the
//! all-committed + all-evidenced fast path. The captured driving trace is therefore reproducible
//! from the frames alone, so a recorded run and its `replay` re-derivation share one
//! coordinator substrate and one clock discipline (this is what un-gates
//! `observe_record_and_replay_green`).
//!
//! ## Event-count deadline forcing (the churn drills)
//!
//! A churn drill needs phases to *expire* (a silent peer's round, the epoch-boundary cooldown +
//! warmup). Expiry is expressed in the module's own clock discipline: the shell delivers **filler
//! frames** that each advance the one-tick-per-frame synthetic clock until the phase deadline
//! passes. Crucially, the *decision to force* is a **state predicate over the shell's evidence
//! accounting** (mirroring the native shell's accounted-set rule), never a wall-clock timer:
//!
//! - an open round may be forced only when every peer expected to act in it is **accounted** —
//!   committed (receipted), evidenced-stalled (`Straggle(Stalled)` for that round), or
//!   **fault-planned** absent (the drill deliberately killed it) — and at least one expected peer
//!   has not committed (all-committed rounds close on the fast path, no forcing);
//! - the epoch-boundary cooldown/warmup may be forced only when the last relayed record is
//!   arithmetically an epoch boundary, and any late-joining peers' engines are already live (a
//!   frame-ordered barrier: the new epoch's `RoundOpen` is published only after their control
//!   subscriptions exist).
//!
//! A healthy peer that has neither committed nor stalled therefore *blocks* the force — correct
//! under arbitrary scheduling delay, which is what makes the drive deterministic under parallel
//! lane load. Wall clock survives only as the outer [`WasmCoordinatorShellConfig::deadline`]
//! failsafe (a wedged run errors out) and as polling *mechanism* (recv timeouts / sleeps that feed
//! no protocol decision).
//!
//! This shell plays the network + storage seats around the module exactly as the testkit's cell-8
//! whole-run harness does: it signs + relays the module's published `RoundOpen`/`RoundRecord`s onto
//! the control plane so the `RoundEngine` peers hear them, and authors the coordinator-as-storage
//! `StorageReceipt` availability evidence over the shared payload store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_vhc_host::wasm_coordinator::{WasmCoordinator, WasmCoordinatorSpec};
use daemon_vhc_net::{ControlPlane, FsPayloadStore, PayloadStore};
use daemon_vhc_proto::messages::{Commitment, Join, RecordEntry, StorageReceipt, ThroughputClass};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId,
    PeerId, SignedMessage, SigningKey, VhcMessage, VhcProtoVersion,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

use crate::seam::{PayloadKey, RunId};
use crate::VhcRunError;

/// The iroh id + class every synthesized `Join` carries (the in-process peers are class-equal).
const JOIN_IROH_ID: IrohId = IrohId([0x22; 32]);

/// A recorded coordinator run trajectory: the module's genesis-derived initial state + the exact
/// driving-frame trace it consumed (the reproducible driver trace `daemon-vhc-observe`'s
/// `RunCapture` records for offline replay), plus how many times the module was re-instantiated
/// from its exported state mid-run (the restart drill) and the peers it dropped. The module owns
/// its state, so there is no host-side per-round state trajectory to snapshot — a recorded run is
/// re-derived through the sandboxed module by [`crate::harness::verify_observe_dir`].
#[derive(Clone, Debug)]
pub struct CoordinatorReplay {
    initial: CoordinatorState,
    inputs: Vec<Input>,
    reloads: u32,
    dropped: std::collections::BTreeSet<PeerId>,
}

impl CoordinatorReplay {
    /// Assemble the capture from the recording drive's collected trace.
    #[must_use]
    pub(crate) fn from_wasm_capture(
        initial: CoordinatorState,
        inputs: Vec<Input>,
        dropped: std::collections::BTreeSet<PeerId>,
        reloads: u32,
    ) -> Self {
        Self {
            initial,
            inputs,
            reloads,
            dropped,
        }
    }

    /// How many times the module was re-instantiated from its exported state mid-run (the
    /// restart drill).
    #[must_use]
    pub fn reloads(&self) -> u32 {
        self.reloads
    }

    /// The peers the coordinator dropped after K record-absences (the silent-death drill, §6.4).
    #[must_use]
    pub fn dropped(&self) -> &std::collections::BTreeSet<PeerId> {
        &self.dropped
    }

    /// The `CoordinatorState` the run started from (the replay genesis for the observe capture).
    #[must_use]
    pub fn initial_state(&self) -> &CoordinatorState {
        &self.initial
    }

    /// The exact ordered driving inputs the module consumed — the reproducible driver trace
    /// `daemon-vhc-observe`'s `RunCapture` records for offline replay.
    #[must_use]
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }
}

/// Construction inputs for a [`WasmCoordinatorShell`].
pub struct WasmCoordinatorShellConfig {
    /// The run this coordinator drives.
    pub run: RunId,
    /// The pinned vhc proto version.
    pub version: VhcProtoVersion,
    /// The genesis-derived initial coordinator state (the module's opaque `{state}` `da_init`).
    pub state: CoordinatorState,
    /// Signing keys of the peers admitted at run start (one synthesized `Join` + ready heartbeat).
    pub bootstrap_keys: Vec<SigningKey>,
    /// Total rounds the run drives (finalize + stop when this many records are published).
    pub num_rounds: u64,
    /// How long to wait for control traffic before re-polling the module for new decisions.
    pub poll: Duration,
    /// Hard wall on the whole drive (a wedged run cannot hang the harness).
    pub deadline: Duration,
    /// Signing keys of peers admitted at the first epoch boundary (the late-join drill). Their
    /// synthesized `Join` is staged into `pending` during epoch 0 and applied at the epoch boundary
    /// (the frozen `RoundEngine` never joins — the coordinator drives admission, §6.5).
    pub late_keys: Vec<SigningKey>,
    /// The frame-ordered barrier for the epoch-boundary roster transition (the late-join drill):
    /// the harness sets this once every late-joining peer's engine has been **constructed** — the
    /// engine subscribes to the control plane in its constructor, so once the flag is observed, the
    /// next epoch's `RoundOpen` is guaranteed to be queued to the late engine. The shell refuses to
    /// force the epoch boundary until then: whether the late peer hears its first round is a
    /// wait-on-observed-state, never a thread-interleaving race. Ignored when `late_keys` is empty.
    pub late_engines_live: Arc<AtomicBool>,
    /// If set, re-instantiate the coordinator module from its exported state right after the record
    /// for this round is relayed (the mid-run restart drill): quiesce→snapshot→`da_migrate` a fresh
    /// incarnation, which resumes the same logical timeline (ABI §10.2/§10.3).
    pub restart_after_round: Option<u64>,
    /// Drive clock-forced phases (round/cooldown/warmup timeouts) by delivering **filler frames**
    /// that advance the module's one-tick-per-frame synthetic clock past the phase deadline (the
    /// event-count clock discipline for the churn drills). `false` keeps the pure event-driven
    /// fast-path drive (the fault-free whole-run / observe-replay lane — no deadline ever fires).
    /// When to force is decided by the accounted-set state predicate (module docs), never a timer.
    pub force_deadlines: bool,
    /// The drill's fault plan, as the accounted-set predicate consumes it: `(peer, last round the
    /// peer acts in)` — for later rounds the peer is deliberately dead/silent and counts as
    /// accounted immediately (no waiting), so its round can be deadline-forced as soon as every
    /// healthy peer is accounted. Empty when no peer is fault-planned.
    pub planned_absent: Vec<(PeerId, u64)>,
}

/// The wasm-backed impure shell: it drives the production `coordinator-quorum` module over the
/// control plane, authoring joins + readiness + availability receipts, and captures the exact
/// driving-frame trace for the replay oracle.
pub struct WasmCoordinatorShell<C> {
    control: Arc<C>,
    store: Arc<FsPayloadStore>,
    run: RunId,
    version: VhcProtoVersion,
    initial: CoordinatorState,
    bootstrap_keys: Vec<SigningKey>,
    late_keys: Vec<SigningKey>,
    late_engines_live: Arc<AtomicBool>,
    num_rounds: u64,
    poll: Duration,
    deadline: Duration,
    restart_after_round: Option<u64>,
    force_deadlines: bool,
    planned_absent: BTreeMap<PeerId, u64>,
    /// The coordinator's §12.1 frame-signing identity (the envelope-named SingleKey authority).
    coord_key: SigningKey,
    /// Peers whose commitment for a round has been evidenced (drives one receipt per commitment).
    committed: BTreeMap<u64, BTreeSet<PeerId>>,
    /// Peers that reported `Straggle(Stalled)` for a round (evidenced-stalled — accounted, §6.4).
    stalled: BTreeMap<u64, BTreeSet<PeerId>>,
    /// The peers the module currently counts as roster members the shell expects action from:
    /// bootstrap at start, + the late joiners once the epoch boundary applies their staged join,
    /// − every peer a published record dropped. The accounted-set predicate ranges over this.
    expected: BTreeSet<PeerId>,
    /// The round the module currently has open (`RoundOpen(r)` relayed, its record not yet) — the
    /// shell's frame-ordered view of "a round is active".
    open_round: Option<u64>,
    /// The last round whose `RoundRecord` was relayed.
    last_record: Option<u64>,
    /// Whether the late joiners have been added to `expected` (the boundary force ran).
    late_expected: bool,
    /// The captured driving-frame trace (worker messages the module consumed), for the replay
    /// oracle — never the module's own published decisions (those are the wire-log oracle).
    inputs: Vec<Input>,
    /// Peers the module dropped after K record-absences (from the published record `drops`).
    dropped: BTreeSet<PeerId>,
}

impl<C: ControlPlane> WasmCoordinatorShell<C> {
    /// Build a shell over `control` + the shared `store`.
    #[must_use]
    pub fn new(
        control: Arc<C>,
        store: Arc<FsPayloadStore>,
        cfg: WasmCoordinatorShellConfig,
    ) -> Self {
        let coord_key = coordinator_key();
        let expected: BTreeSet<PeerId> = cfg.bootstrap_keys.iter().map(peer_id).collect();
        Self {
            control,
            store,
            run: cfg.run,
            version: cfg.version,
            initial: cfg.state,
            bootstrap_keys: cfg.bootstrap_keys,
            late_keys: cfg.late_keys,
            late_engines_live: cfg.late_engines_live,
            num_rounds: cfg.num_rounds,
            poll: cfg.poll,
            deadline: cfg.deadline,
            restart_after_round: cfg.restart_after_round,
            force_deadlines: cfg.force_deadlines,
            planned_absent: cfg.planned_absent.into_iter().collect(),
            coord_key,
            committed: BTreeMap::new(),
            stalled: BTreeMap::new(),
            expected,
            open_round: None,
            last_record: None,
            late_expected: false,
            inputs: Vec::new(),
            dropped: BTreeSet::new(),
        }
    }

    /// The genesis-derived coordinator spec: the module hash pinned to the built blob, the opaque
    /// `{state}` config bytes, the envelope-named `SingleKey` authority, and the run identity.
    fn spec(&self, wasm: &[u8]) -> Result<WasmCoordinatorSpec, VhcRunError> {
        let config_bytes = {
            let v = ciborium::value::Value::Map(vec![(
                ciborium::value::Value::Text("state".into()),
                ciborium::value::Value::serialized(&self.initial)
                    .map_err(|e| VhcRunError::Lifecycle(format!("state value: {e}")))?,
            )]);
            to_canonical_vec(&v).map_err(|e| VhcRunError::Lifecycle(format!("config cbor: {e}")))?
        };
        Ok(WasmCoordinatorSpec {
            module_hash: Hash(*blake3::hash(wasm).as_bytes()),
            config_bytes,
            authority: AuthorityConfig {
                topology: Topology::SingleKey(SingleKey::new(peer_id(&self.coord_key))),
                records_channel: DEFAULT_RECORDS_CHANNEL,
            },
            run_id: Hash(*blake3_hash(self.run.as_str().as_bytes()).as_bytes()),
        })
    }

    /// Deliver one driving frame to the module and capture it in the replay trace. The frame is
    /// signed under `key` (the peer/coordinator whose evidence it is); the recorded `SignedMessage`
    /// is byte-identical to the frame the module ingested (ed25519 is deterministic), so the replay
    /// oracle re-derives from the exact same trace.
    fn deliver(
        &mut self,
        coord: &mut WasmCoordinator,
        key: &SigningKey,
        msg: &VhcMessage,
    ) -> Result<(), VhcRunError> {
        let signed = SignedMessage::sign(key, self.version, msg.clone())
            .map_err(|e| VhcRunError::Lifecycle(format!("frame sign: {e}")))?;
        coord
            .deliver(key, msg)
            .map_err(|e| VhcRunError::Lifecycle(format!("coordinator deliver: {e}")))?;
        self.inputs.push(Input::Message(signed));
        Ok(())
    }

    /// Author + deliver the coordinator-as-storage `StorageReceipt` for `(round, peer)` if the
    /// object is in the shared store (the §6.4 I6 availability-evidence path).
    async fn receipt_for(
        &mut self,
        coord: &mut WasmCoordinator,
        round: u64,
        peer: PeerId,
    ) -> Result<(), VhcRunError> {
        if self
            .committed
            .get(&round)
            .is_some_and(|c| c.contains(&peer))
        {
            return Ok(());
        }
        let key = PayloadKey::new(self.run.clone(), round, peer);
        let Ok(stat) = self.store.head(&key).await else {
            return Ok(());
        };
        let receipt = VhcMessage::StorageReceipt(StorageReceipt {
            round,
            verified: vec![RecordEntry {
                peer,
                hash: stat.hash,
                size: stat.size,
            }],
        });
        let coord_key = self.coord_key.clone();
        self.deliver(coord, &coord_key, &receipt)?;
        self.committed.entry(round).or_default().insert(peer);
        Ok(())
    }

    /// Relay every module decision published since `consumed`: sign it under the coordinator key
    /// and broadcast it on the control plane (the peers verify + consume it). Maintains the shell's
    /// frame-ordered view of the module's round lifecycle (`open_round`/`last_record`), accounts
    /// drops out of the expected set, and returns the number of **new** `RoundRecord`s relayed this
    /// call (a delta — the caller keeps a running total, which survives a mid-run re-instantiation
    /// where the fresh instance's `published` cursor restarts at 0).
    async fn relay_new(
        &mut self,
        coord: &WasmCoordinator,
        consumed: &mut usize,
    ) -> Result<u64, VhcRunError> {
        let published = coord.published();
        let start = (*consumed).min(published.len());
        let mut new_records = 0u64;
        for (_, _, _, msg) in published.iter().skip(start) {
            let signed = SignedMessage::sign(&self.coord_key, self.version, msg.clone())
                .map_err(|e| VhcRunError::Lifecycle(format!("relay sign: {e}")))?;
            let bytes = to_canonical_vec(&signed)
                .map_err(|e| VhcRunError::Lifecycle(format!("relay encode: {e}")))?;
            self.control.publish(&bytes).await?;
            match msg {
                VhcMessage::RoundOpen(ro) => {
                    self.open_round = Some(ro.round);
                }
                VhcMessage::RoundRecord(rr) => {
                    new_records += 1;
                    self.last_record = Some(rr.round);
                    if self.open_round == Some(rr.round) {
                        self.open_round = None;
                    }
                    for p in &rr.drops {
                        self.dropped.insert(*p);
                        self.expected.remove(p);
                    }
                }
                _ => {}
            }
        }
        *consumed = published.len();
        Ok(new_records)
    }

    /// The accounted-set force predicate (the event-count analogue of the native shell's
    /// `maybe_force_accounted`, a STATE predicate — never a timer). Returns whether the module is
    /// sitting in a clock-gated phase that the shell should now expire with filler frames:
    ///
    /// - **An open round**: forceable iff every expected peer is accounted — committed (receipted),
    ///   evidenced-stalled (`Straggle(Stalled)` for this round), or fault-planned absent — AND some
    ///   expected peer has not committed (all-committed rounds close on the fast path untouched).
    ///   A healthy peer that has neither committed nor stalled BLOCKS the force, which makes the
    ///   decision correct under arbitrary scheduling delay.
    /// - **Between rounds at an epoch boundary** (the record relayed, the next round's open
    ///   requires cooldown + warmup expiry): forceable once the late-join barrier is satisfied —
    ///   the staged joins were delivered frame-ordered long before, and the late engines' control
    ///   subscriptions exist (`late_engines_live`), so the new epoch's `RoundOpen` cannot be
    ///   published into the void. Adds the late joiners to the expected set at that point.
    ///
    /// Within-epoch record→open transitions are module-internal fast paths (`finalize_round` opens
    /// the next round in the same tick) and are never forced — this also makes the shell immune to
    /// the torn read where the record is relayed while its sibling open is still in flight.
    fn should_force(&mut self) -> bool {
        if let Some(r) = self.open_round {
            let committed = self.committed.get(&r);
            let stalled = self.stalled.get(&r);
            let is_committed = |p: &PeerId| committed.is_some_and(|c| c.contains(p));
            let accounted = |p: &PeerId| {
                is_committed(p)
                    || stalled.is_some_and(|s| s.contains(p))
                    || self.planned_absent.get(p).is_some_and(|last| r > *last)
            };
            return !self.expected.is_empty()
                && self.expected.iter().all(accounted)
                && self.expected.iter().any(|p| !is_committed(p));
        }
        // No round open: force only across a genuine epoch boundary (cooldown + warmup expiry).
        let Some(r) = self.last_record else {
            return false;
        };
        let epoch_rounds = self.initial.config.epoch_rounds;
        let boundary = epoch_rounds > 0 && (r + 1) % epoch_rounds == 0;
        if !boundary || r + 1 >= self.num_rounds {
            return false;
        }
        // The frame-ordered roster barrier: hold the boundary until every late engine is live.
        if !self.late_keys.is_empty() {
            if !self.late_engines_live.load(Ordering::Acquire) {
                return false;
            }
            if !self.late_expected {
                for k in &self.late_keys {
                    self.expected.insert(peer_id(k));
                }
                self.late_expected = true;
            }
        }
        true
    }

    /// Whether the module has published a `RoundOpen` yet (round 0 has opened) — the cue to stage
    /// the late-join `Join` into `pending` so it lands in the roster at the first epoch boundary
    /// (delivering it before an open would upsert it into epoch 0's roster instead).
    fn round_opened(coord: &WasmCoordinator) -> bool {
        coord
            .published()
            .iter()
            .any(|(_, _, _, m)| matches!(m, VhcMessage::RoundOpen(_)))
    }

    /// Force the module past a clock-gated phase deadline (round train/witness, cooldown, warmup)
    /// by delivering **filler frames** — coordinator-signed no-op heartbeats — that each advance the
    /// module's one-tick-per-frame synthetic clock (architecture §4.1). A filler is a non-member
    /// frame: the tick ignores it (the coordinator is not a roster member) but still ticks the clock
    /// once, so a bounded batch expires the deadline and the module publishes the next decision. The
    /// event-count analogue of the native shell's `Input::Clock` jump — no wall-clock forcing (the
    /// sleep below is a polling mechanism for the guest thread to drain, feeding no decision).
    fn force_step(&mut self, coord: &mut WasmCoordinator) -> Result<(), VhcRunError> {
        const FORCE_CAP: usize = 512;
        const BATCH: usize = 8;
        // The filler loop blocks (deliver + poll sleeps): hand the worker's task queue off so the
        // same runtime's engine/collector tasks keep running while the phase is expired.
        tokio::task::block_in_place(|| {
            let before = coord.published().len();
            let coord_key = self.coord_key.clone();
            let filler = VhcMessage::Heartbeat(daemon_vhc_proto::messages::Heartbeat {
                round: 0,
                ready: None,
            });
            let mut delivered = 0usize;
            while delivered < FORCE_CAP {
                for _ in 0..BATCH {
                    self.deliver(coord, &coord_key, &filler)?;
                    delivered += 1;
                }
                std::thread::sleep(Duration::from_millis(5));
                if coord.published().len() > before {
                    break;
                }
            }
            Ok(())
        })
    }

    /// Drive the run to completion over the wasm coordinator, returning the reproducible driving
    /// trace ([`CoordinatorReplay`]) for the observe capture.
    ///
    /// On the fault-free lane (`force_deadlines == false`) this is the pure event-driven drive:
    /// joins + readiness heartbeats open round 0 and each round closes on the all-committed +
    /// all-evidenced fast path. The churn drills set `force_deadlines`, which adds the event-count
    /// clock discipline: a stalled/silent round (or the cooldown+warmup at an epoch boundary) is
    /// force-closed with **filler frames** ([`WasmCoordinatorShell::force_step`]), the late joiner is
    /// admitted at the epoch boundary, and the restart drill re-instantiates the module from its
    /// exported state ([`WasmCoordinator::quiesce_snapshot`] → [`WasmCoordinator::start_migrating`]).
    ///
    /// # Errors
    /// [`VhcRunError::Lifecycle`] on a build/start/deliver failure or the run's hard deadline.
    #[allow(clippy::too_many_lines)]
    pub async fn drive(mut self) -> Result<CoordinatorReplay, VhcRunError> {
        let wasm = crate::replay_sandbox::coordinator_quorum_wasm()
            .map_err(|e| VhcRunError::Lifecycle(format!("coordinator blob: {e}")))?;
        let spec = self.spec(&wasm)?;
        let coord_seed = *blake3_hash(b"daemon-vhc/harness/wasm-coordinator/frame-key").as_bytes();
        // The module start compiles the blob (CPU-seconds): hand this worker's task queue off
        // first so the drive cannot starve the same runtime's engine/collector tasks.
        let mut coord = tokio::task::block_in_place(|| {
            WasmCoordinator::start(&wasm, &spec, Vec::new(), 0, coord_seed)
        })
        .map_err(|e| VhcRunError::Lifecycle(format!("coordinator start: {e}")))?;

        let mut sub = self.control.subscribe();

        // Admit the roster and exit warmup event-driven: joins bring the roster to `min_peers`,
        // then readiness heartbeats open round 0 (no warmup clock — the module ticks per frame).
        let boot = self.bootstrap_keys.clone();
        let late = self.late_keys.clone();
        let envelope_hash = self.initial.config.envelope_hash;
        for key in &boot {
            let join = VhcMessage::Join(Join {
                run_id: self.run.as_str().to_string(),
                iroh_id: JOIN_IROH_ID,
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: Some(envelope_hash),
            });
            self.deliver(&mut coord, key, &join)?;
        }
        for key in &boot {
            let hb = VhcMessage::Heartbeat(daemon_vhc_proto::messages::Heartbeat {
                round: 0,
                ready: Some(true),
            });
            self.deliver(&mut coord, key, &hb)?;
        }

        // The full keyed roster (bootstrap + any late joiner) the inbound-frame router keys on.
        let peer_keys: BTreeMap<PeerId, SigningKey> = boot
            .iter()
            .chain(late.iter())
            .map(|k| (peer_id(k), k.clone()))
            .collect();

        let mut consumed = 0usize;
        let mut total_records = 0u64;
        let mut late_joined = late.is_empty();
        let mut reloads = 0u32;
        // Wall clock appears ONLY as the outer failsafe below (a wedged run errors the harness
        // out); every force decision is the accounted-set state predicate (`should_force`).
        let start = Instant::now();
        loop {
            let delta = self.relay_new(&coord, &mut consumed).await?;
            total_records += delta;

            // Stage the late joiner once round 0 has opened: delivered now it lands in `pending`
            // and is applied to the roster at the first epoch boundary (§6.2). Frame-ordered: this
            // join precedes every boundary-forcing filler on the module's delivery stream, so
            // "staged before the boundary drain" is a certainty, not a race.
            if !late_joined && Self::round_opened(&coord) {
                for key in &late {
                    let join = VhcMessage::Join(Join {
                        run_id: self.run.as_str().to_string(),
                        iroh_id: JOIN_IROH_ID,
                        class: ThroughputClass::C1,
                        capabilities: CapabilitySet::new(),
                        envelope_hash: Some(envelope_hash),
                    });
                    self.deliver(&mut coord, key, &join)?;
                }
                late_joined = true;
            }

            // Mid-run restart: after the record for `restart_after_round` is relayed, re-instantiate
            // the module from its exported state (the fresh incarnation resumes the same timeline).
            if let Some(r) = self.restart_after_round {
                if total_records > r {
                    // Blocking span (guest-thread join + module recompile): hand the worker off.
                    coord = tokio::task::block_in_place(|| {
                        let capture = coord.quiesce_snapshot(60_000).map_err(|e| {
                            VhcRunError::Lifecycle(format!("coordinator restart quiesce: {e}"))
                        })?;
                        WasmCoordinator::start_migrating(
                            &wasm,
                            &spec,
                            Vec::new(),
                            1,
                            coord_seed,
                            capture,
                        )
                        .map_err(|e| {
                            VhcRunError::Lifecycle(format!(
                                "coordinator restart re-instantiate: {e}"
                            ))
                        })
                    })?;
                    consumed = 0;
                    reloads += 1;
                    self.restart_after_round = None;
                    continue;
                }
            }

            if total_records >= self.num_rounds {
                break;
            }
            if start.elapsed() >= self.deadline {
                return Err(VhcRunError::Lifecycle(format!(
                    "wasm coordinator drive deadline: {total_records}/{} records",
                    self.num_rounds
                )));
            }
            // Event-count deadline-close, gated on the accounted-set STATE predicate: expire the
            // stuck phase (a round whose only missing actors are evidenced-stalled or
            // fault-planned, or the epoch-boundary cooldown + warmup) with filler frames. A healthy
            // peer that has neither committed nor stalled blocks this, so the decision is
            // deterministic under arbitrary scheduling delay — no timer is consulted.
            if self.force_deadlines && self.should_force() {
                self.force_step(&mut coord)?;
                continue;
            }
            match tokio::time::timeout(self.poll, sub.recv()).await {
                Ok(Some(bytes)) => {
                    let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                        continue;
                    };
                    if msg.verify_for_run(self.version).is_err() {
                        continue;
                    }
                    let signer = msg.signer;
                    match &msg.payload {
                        // Skip the coordinator's own relayed outputs echoed back over gossip.
                        VhcMessage::RoundOpen(_) | VhcMessage::RoundRecord(_) => continue,
                        // Peer heartbeats are advisory post-warmup (readiness is synthesized at
                        // admission), and every delivered frame ticks the module's synthetic
                        // clock — relaying a steady heartbeat stream would burn a round's deadline
                        // budget before its evidence lands. Not delivered; carries no accounting.
                        VhcMessage::Heartbeat(_) => continue,
                        VhcMessage::Commitment(Commitment { round, .. }) => {
                            let round = *round;
                            if let Some(key) = peer_keys.get(&signer) {
                                let key = key.clone();
                                self.deliver(&mut coord, &key, &msg.payload)?;
                            }
                            self.receipt_for(&mut coord, round, signer).await?;
                        }
                        VhcMessage::Straggle(st) => {
                            let (round, is_stalled) = (
                                st.round,
                                st.status == daemon_vhc_proto::messages::StraggleStatus::Stalled,
                            );
                            if let Some(key) = peer_keys.get(&signer) {
                                let key = key.clone();
                                self.deliver(&mut coord, &key, &msg.payload)?;
                            }
                            // Evidenced-stalled: the peer declared it skips this round — it is
                            // accounted, so the round becomes deadline-forceable (§6.4).
                            if is_stalled {
                                self.stalled.entry(round).or_default().insert(signer);
                            }
                        }
                        _ => {
                            if let Some(key) = peer_keys.get(&signer) {
                                let key = key.clone();
                                self.deliver(&mut coord, &key, &msg.payload)?;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => { /* poll timeout — re-drain + re-evaluate forcing at the loop top */ }
            }
        }

        // Final relay of any trailing decisions, then a clean stop.
        self.relay_new(&coord, &mut consumed).await?;
        coord
            .stop()
            .map_err(|e| VhcRunError::Lifecycle(format!("coordinator stop: {e}")))?;

        Ok(CoordinatorReplay::from_wasm_capture(
            self.initial,
            self.inputs,
            self.dropped,
            reloads,
        ))
    }
}

/// The coordinator's node identity key (signs the relayed RoundOpen/RoundRecord + the receipts).
#[must_use]
fn coordinator_key() -> SigningKey {
    SigningKey::from_bytes(&[0xC0; 32])
}
