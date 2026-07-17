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
//! all-committed + all-evidenced fast path — no wall-clock phase deadlines, no clock-timeout
//! forcing. The captured driving trace is therefore reproducible from the frames alone, so a
//! recorded run and its `swarm-replay` re-derivation share one coordinator substrate and one clock
//! discipline (this is what un-gates `observe_record_and_replay_green`).
//!
//! This shell plays the network + storage seats around the module exactly as the testkit's cell-8
//! whole-run harness does: it signs + relays the module's published `RoundOpen`/`RoundRecord`s onto
//! the control plane so the `RoundEngine` peers hear them, and authors the coordinator-as-storage
//! `StorageReceipt` availability evidence over the shared payload store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_vhc_host::wasm_coordinator::{WasmCoordinator, WasmCoordinatorSpec};
use daemon_vhc_net::{ControlPlane, FsPayloadStore, PayloadStore};
use daemon_vhc_proto::messages::{Commitment, Join, RecordEntry, StorageReceipt, ThroughputClass};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId,
    PeerId, SignedMessage, SigningKey, SwarmMessage, SwarmProtoVersion,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

use crate::local_coordinator::CoordinatorReplay;
use crate::seam::{PayloadKey, RunId};
use crate::SwarmRunError;

/// The iroh id + class every synthesized `Join` carries (the in-process peers are class-equal).
const JOIN_IROH_ID: IrohId = IrohId([0x22; 32]);

/// Construction inputs for a [`WasmCoordinatorShell`] (mirrors `LocalCoordinatorConfig`, minus the
/// clock knobs the module owns).
pub struct WasmCoordinatorShellConfig {
    /// The run this coordinator drives.
    pub run: RunId,
    /// The pinned swarm proto version.
    pub version: SwarmProtoVersion,
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
}

/// The wasm-backed impure shell: it drives the production `coordinator-quorum` module over the
/// control plane, authoring joins + readiness + availability receipts, and captures the exact
/// driving-frame trace for the replay oracle.
pub struct WasmCoordinatorShell<C> {
    control: Arc<C>,
    store: Arc<FsPayloadStore>,
    run: RunId,
    version: SwarmProtoVersion,
    initial: CoordinatorState,
    bootstrap_keys: Vec<SigningKey>,
    num_rounds: u64,
    poll: Duration,
    deadline: Duration,
    /// The coordinator's §12.1 frame-signing identity (the envelope-named SingleKey authority).
    coord_key: SigningKey,
    /// Peers whose commitment for a round has been evidenced (drives one receipt per commitment).
    committed: BTreeMap<u64, BTreeSet<PeerId>>,
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
        Self {
            control,
            store,
            run: cfg.run,
            version: cfg.version,
            initial: cfg.state,
            bootstrap_keys: cfg.bootstrap_keys,
            num_rounds: cfg.num_rounds,
            poll: cfg.poll,
            deadline: cfg.deadline,
            coord_key,
            committed: BTreeMap::new(),
            inputs: Vec::new(),
            dropped: BTreeSet::new(),
        }
    }

    /// The genesis-derived coordinator spec: the module hash pinned to the built blob, the opaque
    /// `{state}` config bytes, the envelope-named `SingleKey` authority, and the run identity.
    fn spec(&self, wasm: &[u8]) -> Result<WasmCoordinatorSpec, SwarmRunError> {
        let config_bytes = {
            let v = ciborium::value::Value::Map(vec![(
                ciborium::value::Value::Text("state".into()),
                ciborium::value::Value::serialized(&self.initial)
                    .map_err(|e| SwarmRunError::Lifecycle(format!("state value: {e}")))?,
            )]);
            to_canonical_vec(&v)
                .map_err(|e| SwarmRunError::Lifecycle(format!("config cbor: {e}")))?
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
        msg: &SwarmMessage,
    ) -> Result<(), SwarmRunError> {
        let signed = SignedMessage::sign(key, self.version, msg.clone())
            .map_err(|e| SwarmRunError::Lifecycle(format!("frame sign: {e}")))?;
        coord
            .deliver(key, msg)
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator deliver: {e}")))?;
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
    ) -> Result<(), SwarmRunError> {
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
        let receipt = SwarmMessage::StorageReceipt(StorageReceipt {
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
    /// and broadcast it on the control plane (the peers verify + consume it), accounting drops and
    /// counting `RoundRecord`s. Returns the running record count.
    async fn relay_new(
        &mut self,
        coord: &WasmCoordinator,
        consumed: &mut usize,
    ) -> Result<u64, SwarmRunError> {
        let published = coord.published();
        // Relay only the decisions published since last time (the peers consume them once).
        for (_, _, _, msg) in published.iter().skip(*consumed) {
            let signed = SignedMessage::sign(&self.coord_key, self.version, msg.clone())
                .map_err(|e| SwarmRunError::Lifecycle(format!("relay sign: {e}")))?;
            let bytes = to_canonical_vec(&signed)
                .map_err(|e| SwarmRunError::Lifecycle(format!("relay encode: {e}")))?;
            self.control.publish(&bytes).await?;
        }
        *consumed = published.len();
        // Account drops (idempotent set-insert) + count the records the module has published.
        let mut records = 0u64;
        for (_, _, _, msg) in &published {
            if let SwarmMessage::RoundRecord(rr) = msg {
                records += 1;
                for p in &rr.drops {
                    self.dropped.insert(*p);
                }
            }
        }
        Ok(records)
    }

    /// Drive the run to completion over the wasm coordinator, returning the reproducible driving
    /// trace ([`CoordinatorReplay`]) for the observe capture.
    ///
    /// # Errors
    /// [`SwarmRunError::Lifecycle`] on a build/start/deliver failure or the run's hard deadline.
    pub async fn drive(mut self) -> Result<CoordinatorReplay, SwarmRunError> {
        let wasm = crate::replay_sandbox::coordinator_quorum_wasm()
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator blob: {e}")))?;
        let spec = self.spec(&wasm)?;
        let coord_seed =
            *blake3_hash(b"daemon-swarm/harness/wasm-coordinator/frame-key").as_bytes();
        let mut coord = WasmCoordinator::start(&wasm, &spec, Vec::new(), 0, coord_seed)
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator start: {e}")))?;

        let mut sub = self.control.subscribe();

        // Admit the roster and exit warmup event-driven: joins bring the roster to `min_peers`,
        // then readiness heartbeats open round 0 (no warmup clock — the module ticks per frame).
        let boot = self.bootstrap_keys.clone();
        let envelope_hash = self.initial.config.envelope_hash;
        for key in &boot {
            let join = SwarmMessage::Join(Join {
                run_id: self.run.as_str().to_string(),
                iroh_id: JOIN_IROH_ID,
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: Some(envelope_hash),
            });
            self.deliver(&mut coord, key, &join)?;
        }
        for key in &boot {
            let hb = SwarmMessage::Heartbeat(daemon_vhc_proto::messages::Heartbeat {
                round: 0,
                ready: Some(true),
            });
            self.deliver(&mut coord, key, &hb)?;
        }

        let peer_keys: BTreeMap<PeerId, SigningKey> =
            boot.iter().map(|k| (peer_id(k), k.clone())).collect();

        let mut consumed = 0usize;
        let start = Instant::now();
        loop {
            let records = self.relay_new(&coord, &mut consumed).await?;
            if records >= self.num_rounds {
                break;
            }
            if start.elapsed() >= self.deadline {
                return Err(SwarmRunError::Lifecycle(format!(
                    "wasm coordinator drive deadline: {records}/{} records",
                    self.num_rounds
                )));
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
                        SwarmMessage::RoundOpen(_) | SwarmMessage::RoundRecord(_) => continue,
                        SwarmMessage::Commitment(Commitment { round, .. }) => {
                            let round = *round;
                            if let Some(key) = peer_keys.get(&signer) {
                                let key = key.clone();
                                self.deliver(&mut coord, &key, &msg.payload)?;
                            }
                            self.receipt_for(&mut coord, round, signer).await?;
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
                Err(_) => { /* poll timeout — re-drain the module's decisions at the loop top */ }
            }
        }

        // Final relay of any trailing decisions, then a clean stop.
        self.relay_new(&coord, &mut consumed).await?;
        coord
            .stop()
            .map_err(|e| SwarmRunError::Lifecycle(format!("coordinator stop: {e}")))?;

        Ok(CoordinatorReplay::from_wasm_capture(
            self.initial,
            self.inputs,
            self.dropped,
        ))
    }
}

/// The coordinator's node identity key (signs the relayed RoundOpen/RoundRecord + the receipts).
#[must_use]
fn coordinator_key() -> SigningKey {
    SigningKey::from_bytes(&[0xC0; 32])
}
