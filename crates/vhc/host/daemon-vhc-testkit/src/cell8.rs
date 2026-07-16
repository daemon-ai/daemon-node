// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **Cell 8** — the mixed-fleet matrix's target end-state (decisions D3): wasm coordinator ×
//! v2 workers × envelope v2, as a whole-run harness (refactor §8/D2 acceptance: "the matrix
//! cells for {wasm coordinator × v1/v2 workers} pass").
//!
//! The wasm twin of [`crate::barrier`]: N production `tiny_llama_v2.wasm` workers train real
//! barrier rounds under the production `coordinator_quorum.wasm` blob — **both** sides running
//! under the real major-2 event-loop driver, journaled, §8.7 replay-verified. The harness plays
//! the network + async-runtime seats only:
//!
//! - **coordinator → workers**: the coordinator's §12.1-signed publishes are decoded, checked
//!   against the envelope-named `SingleKey` identity (the D2 thin `Authority` seam at the network
//!   seat — D1's `Authority::accept` replaces the judgment), and relayed to each worker pump with
//!   the original signed frame as tag-12 evidence;
//! - **workers → coordinator**: each worker's control publishes (`Commitment`/`Digest`) are
//!   re-signed as wire `SignedMessage`s under the worker's node key and delivered to the
//!   coordinator pump as host-verified frames;
//! - **payload plane**: the harness services `payload_put`, verifies the guest's commitment hash
//!   covers exactly the serviced bytes, authors the availability `StorageReceipt`, and stages the
//!   record-listed payload set to every worker (record order, §5.11).
//!
//! The run is configured **from a genesis envelope v2** ([`crate::configure_wasm_coordinator`]) —
//! the coordinator module hash pinned in the role set, its opaque config carried verbatim, the
//! `SingleKey` identity in `[identities]` — which is exactly what makes cells 3/7 (envelope v1)
//! typed refusals.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_host::v2::{
    replay_v2, start_run, DeliverVerdict, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd,
    ReplayScript, RunEnd, RunIdentity, SinkEntry, V2RunConfig,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::messages::{
    BatchWindow, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_proto::{
    blake3_hash, digest_state, from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet,
    ControlTransport, GenesisEnvelope, Hash, Identities, IrohId, PeerId, RoleEntry, RoleGrants,
    RunSectionV2, Seed, SigningKey, SnapshotArtifact, SwarmMessage, TransportSelection,
    GENESIS_SCHEMA_MAJOR, SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig};
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

use crate::barrier::{phase_a_grants, tiny_llama_config, SEQ_LEN};
use crate::run::Decision;
use crate::wasm_coordinator::{
    authorize_coordinator_frame, configure_wasm_coordinator, WasmCoordinator,
};

/// How a cell-8 whole run is set up.
pub struct Cell8Spec {
    /// A stable run label (the genesis `run_label`; the `RunId` is the genesis hash).
    pub run_label: String,
    /// Wasm workers to run (2+ is the multi-worker shape). Worker `i` gets instance `i + 1`.
    pub workers: usize,
    /// Rounds to drive to a record.
    pub rounds: u64,
    /// Inner steps per round (must divide each worker's assigned window).
    pub steps_per_round: u32,
    /// Sequences per round across the roster.
    pub global_batch: u32,
    /// Hard wall per wait step.
    pub timeout: Duration,
}

impl Cell8Spec {
    /// A spec with the t2 per-worker geometry (`steps_per_round = 2`, one sequence per inner step
    /// per worker).
    #[must_use]
    pub fn new(run_label: &str, workers: usize, rounds: u64) -> Self {
        Self {
            run_label: run_label.to_string(),
            workers,
            rounds,
            steps_per_round: 2,
            global_batch: 2 * workers as u32,
            timeout: Duration::from_secs(180),
        }
    }
}

/// One worker's observable product (mirrors [`crate::WorkerReport`]).
pub struct Cell8WorkerReport {
    /// The worker's peer identity.
    pub peer: PeerId,
    /// How its run ended.
    pub end: RunEnd,
    /// Its final det-lane state digest.
    pub digest: [u8; 16],
    /// Its §8.7 replay verdict.
    pub replay_matched: bool,
    /// How many decisions the replay re-derived.
    pub replay_decisions: usize,
}

/// The whole run's observable product.
pub struct Cell8Report {
    /// Per-worker reports, worker-index order.
    pub workers: Vec<Cell8WorkerReport>,
    /// Rounds driven to a record.
    pub rounds_done: u64,
    /// `RoundRecord`s the wasm coordinator published, in order.
    pub coordinator_records: u64,
    /// How the coordinator's run ended.
    pub coordinator_end: RunEnd,
    /// The run's cryptographic `RunId` (the genesis hash).
    pub run_id: Hash,
}

impl Cell8Report {
    /// True iff every worker ended cleanly with a matching §8.7 replay, all workers agree on the
    /// det-lane digest, the coordinator recorded every driven round, and it ended cleanly.
    #[must_use]
    pub fn is_green(&self) -> bool {
        let ends_clean = self
            .workers
            .iter()
            .all(|w| matches!(w.end, RunEnd::Outcome(0)) && w.replay_matched);
        let first = self.workers.first().map(|w| w.digest);
        ends_clean
            && self.workers.iter().all(|w| Some(w.digest) == first)
            && self.coordinator_records == self.rounds_done
            && matches!(self.coordinator_end, RunEnd::Outcome(0))
    }
}

/// Author the cell-8 **genesis envelope v2** for a barrier run: the coordinator role pinning the
/// `coordinator_quorum.wasm` blob with its opaque `{state: CoordinatorState}` config, one trainer
/// role pinning the worker blob, and the envelope-named `SingleKey` coordinator identity.
///
/// The coordinator's `RunConfig` is authored directly (it is the coordinator module's OPAQUE
/// config under v2 — the v1 `[data]`/`[phases]` sections it used to be projected from left the
/// envelope at D0). Phase deadlines are effectively infinite: the run is driven entirely by
/// event-driven fast paths so the guest's synthetic clock (one tick per event) suffices.
#[must_use]
pub fn cell8_genesis(
    run_label: &str,
    coordinator_wasm_blake3: Hash,
    worker_wasm_blake3: Hash,
    coordinator_identity: PeerId,
    workers: u32,
    steps_per_round: u32,
    global_batch: u32,
) -> GenesisEnvelope {
    // The coordinator's opaque config: an authored RunConfig + genesis CoordinatorState, exactly
    // the guest's `da_init` shape (`{state: …}`; event-driven synthetic clock defaults).
    let run_config = RunConfig {
        run_id: run_label.to_string(),
        proto_version: SWARM_PROTO_VERSION,
        // The envelope anchor a worker Join asserts under v2 is the genesis hash; the harness
        // passes `envelope_hash: None` on joins (the v1-era anchor is superseded — the genesis
        // hash IS the run identity, carried in the execution identity / §12.1 scope).
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: workers,
        max_peers: workers.max(4),
        warmup_s: 1_000_000,
        round_train_max_s: 1_000_000,
        round_witness_s: 1_000_000,
        cooldown_s: 1_000_000,
        epoch_rounds: 0,
        stall_rounds_max: 2,
        global_batch: daemon_vhc_proto::envelope::GlobalBatch {
            start: global_batch,
            end: global_batch,
            ramp_rounds: 1,
        },
        stop: daemon_vhc_proto::envelope::StopCondition::Rounds(1_000_000),
        steps_per_round,
        seq_len: u64::from(SEQ_LEN),
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, Seed([0x33; 32]), 0);
    let coord_config = Value::Map(vec![(
        Value::Text("state".into()),
        Value::serialized(&state).expect("state to cbor value"),
    )]);

    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "coordinator.wasm".to_string(),
        SnapshotArtifact {
            url: "r2://mods/coordinator_quorum.wasm".into(),
            blake3: coordinator_wasm_blake3,
            size: None,
        },
    );
    artifacts.insert(
        "worker.wasm".to_string(),
        SnapshotArtifact {
            url: "r2://mods/tiny_llama_v2.wasm".into(),
            blake3: worker_wasm_blake3,
            size: None,
        },
    );

    let mut roles = BTreeMap::new();
    roles.insert(
        "coordinator".to_string(),
        RoleEntry {
            lane: "coordinator".into(),
            module: "coordinator.wasm".into(),
            abi: "vhc@2".into(),
            config: coord_config,
            grants: RoleGrants::default(),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );
    roles.insert(
        "trainer".to_string(),
        RoleEntry {
            lane: "trainer".into(),
            module: "worker.wasm".into(),
            abi: "vhc@2".into(),
            // Per-worker config (peer identity, roster) is join-time wiring the node authors —
            // the shared role config stays empty here (the D1 genesis join flow owns threading).
            config: Value::Map(vec![]),
            grants: RoleGrants::default(),
            device_min: daemon_vhc_proto::DeviceMinimums::default(),
        },
    );

    GenesisEnvelope {
        run: RunSectionV2 {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: run_label.to_string(),
            min_peers: workers,
            max_peers: workers.max(4),
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        // The run's declared trust topology (D1's typed AuthorityConfig, encoded into the opaque
        // section the host never interprets): launch SingleKey over the coordinator identity,
        // records on the default authoritative channel.
        authority: AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(coordinator_identity)),
            records_channel: DEFAULT_RECORDS_CHANNEL,
        }
        .encode(),
        transport: TransportSelection {
            control: vec![ControlTransport::Mem],
            payload_store: "fs".into(),
        },
        identities: Identities {
            coordinator: Some(coordinator_identity),
            coordinator_set: Vec::new(),
            upgrade_authority: Vec::new(),
        },
    }
}

/// One live worker under the harness (the [`crate::barrier`] `LiveWorker` shape).
struct LiveWorker {
    key: SigningKey,
    peer: PeerId,
    identity: RunIdentity,
    config: Vec<u8>,
    pump: PumpHandle,
    sink: Arc<Mutex<MemorySink>>,
    run: Option<daemon_vhc_host::v2::V2Run>,
    engine: Worker,
    /// Per-worker coordinator-plane delivery seq (§12.2 dense-seq discipline, channel 0).
    coord_seq: u64,
    /// The guest's `payload_put` bytes, serviced by the harness.
    puts: Vec<Vec<u8>>,
    /// Round → the round's sealed payload set (stashed at commit, staged at the record).
    stash: BTreeMap<u64, BTreeMap<PeerId, Vec<u8>>>,
}

impl LiveWorker {
    /// Relay a coordinator decision: the coordinator's ORIGINAL §12.1 signed frame is the tag-12
    /// evidence; the payload is the decoded control message's canonical bytes; the sender is the
    /// coordinator's §12.1 frame identity (checked against the envelope-named authority upstream).
    fn deliver_from_coordinator(
        &mut self,
        sender: [u8; 32],
        msg: &SwarmMessage,
        evidence: Vec<u8>,
    ) -> Result<(), String> {
        let payload = to_canonical_vec(msg).map_err(|e| format!("payload encode: {e}"))?;
        let seq = self.coord_seq;
        self.coord_seq += 1;
        match self
            .pump
            .deliver_frame(0, seq, sender, payload, evidence)
            .map_err(|e| format!("deliver: {e}"))?
        {
            DeliverVerdict::Accepted => Ok(()),
            other => Err(format!(
                "coordinator frame back-pressured/refused ({other:?}) in the cell-8 drive"
            )),
        }
    }

    fn messages(&self) -> Vec<SwarmMessage> {
        self.pump
            .published()
            .into_iter()
            .filter_map(|(_, _, frame)| {
                let v: Value = ciborium::de::from_reader(frame.as_slice()).ok()?;
                let Value::Array(parts) = v else { return None };
                let Value::Bytes(payload) = parts.get(1)? else {
                    return None;
                };
                from_canonical_slice::<SwarmMessage>(payload).ok()
            })
            .collect()
    }

    fn service_ops(&mut self) -> Result<(), String> {
        for (op, request) in self.pump.take_op_requests() {
            match request {
                OpRequest::PayloadPut { bytes } => {
                    self.puts.push(bytes.to_vec());
                    self.pump
                        .complete_op(op, OpOutcome::PutDone)
                        .map_err(|e| format!("put completion: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "unexpected op request from the cell-8 worker: {other:?}"
                    ))
                }
            }
        }
        Ok(())
    }

    fn wait_for(
        &mut self,
        timeout: Duration,
        what: &str,
        pred: impl Fn(&[SwarmMessage]) -> bool,
    ) -> Result<Vec<SwarmMessage>, String> {
        let deadline = Instant::now() + timeout;
        loop {
            self.service_ops()?;
            let msgs = self.messages();
            if pred(&msgs) {
                return Ok(msgs);
            }
            if Instant::now() >= deadline {
                return Err(format!("timed out waiting for {what}"));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

fn commitments_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Commitment(c) if c.round == round))
        .count()
}

fn digests_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Digest(d) if d.round == round))
        .count()
}

fn assigned_len(window: BatchWindow, seed: Seed, roster: &[PeerId], peer: &PeerId) -> u64 {
    let weighted: Vec<(PeerId, ThroughputClass)> =
        roster.iter().map(|p| (*p, ThroughputClass::C1)).collect();
    daemon_vhc_sdk_consensus::assign_batches(&weighted, &seed, window, 0)
        .into_iter()
        .find(|(p, _)| p == peer)
        .map_or(0, |(_, w)| w.end.saturating_sub(w.start))
}

/// Drive the cell-8 whole run: N production tiny-llama-v2 workers through `spec.rounds` barrier
/// rounds under the production wasm coordinator, configured from a genesis envelope v2; then stop
/// everything cleanly and §8.7 replay-verify each worker.
///
/// # Errors
/// A `String` on any harness-level failure.
#[allow(clippy::too_many_lines)]
pub fn cell8_whole_run(
    coordinator_wasm: &[u8],
    worker_wasm: &[u8],
    spec: &Cell8Spec,
) -> Result<Cell8Report, String> {
    let coord_hash = Hash(*blake3::hash(coordinator_wasm).as_bytes());
    let worker_hash = *blake3::hash(worker_wasm).as_bytes();
    let grants = phase_a_grants();

    // Identities: the coordinator's §12.1 frame key IS the envelope-named SingleKey identity
    // (launch topology, architecture §4.4); worker node keys are index-derived.
    let coord_key_seed =
        *blake3::hash(format!("cell8-coordinator/{}", spec.run_label).as_bytes()).as_bytes();
    let coord_identity = peer_id(&SigningKey::from_bytes(&coord_key_seed));
    let worker_keys: Vec<SigningKey> = (0..spec.workers)
        .map(|i| {
            SigningKey::from_bytes(
                blake3::hash(format!("cell8-worker/{}/{i}", spec.run_label).as_bytes()).as_bytes(),
            )
        })
        .collect();
    let roster: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    // The genesis envelope v2 + the wasm-coordinator configuration derived from it (cell 8's
    // configuration half — the exact seat that REFUSES under envelope v1, cells 3/7).
    let genesis = cell8_genesis(
        &spec.run_label,
        coord_hash,
        Hash(worker_hash),
        coord_identity,
        spec.workers as u32,
        spec.steps_per_round,
        spec.global_batch,
    );
    let author = SigningKey::from_bytes(
        blake3::hash(format!("cell8-author/{}", spec.run_label).as_bytes()).as_bytes(),
    );
    let frozen = genesis
        .freeze(&author)
        .map_err(|e| format!("genesis freeze: {e}"))?;
    let coord_spec =
        configure_wasm_coordinator(&frozen).map_err(|e| format!("coordinator config: {e}"))?;
    let run_id = coord_spec.run_id;

    let mut coord = WasmCoordinator::start(
        coordinator_wasm,
        &coord_spec,
        grants.clone(),
        0,
        coord_key_seed,
    )?;

    // Start every worker under the real v2 event-loop driver, keyed by the cryptographic RunId.
    let mut workers: Vec<LiveWorker> = Vec::with_capacity(spec.workers);
    for (i, key) in worker_keys.iter().enumerate() {
        let peer = roster[i];
        let config = tiny_llama_config(&peer, &roster, spec.steps_per_round, 1, 2);
        let engine = Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
        let sel = select_driver(&engine, worker_wasm, Some(&worker_hash))
            .map_err(|e| format!("worker {i} selection: {e}"))?;
        if sel.driver != daemon_vhc_abi::CandidateDriver::V2 {
            return Err(format!(
                "worker {i}: not a major-2 module: {:?}",
                sel.driver
            ));
        }
        let identity = RunIdentity {
            run_id: run_id.0,
            epoch: 0,
            role: "trainer".to_string(),
            instance: i as u64 + 1,
            module: worker_hash,
        };
        let key_seed =
            *blake3::hash(format!("cell8-frame-key/{}/{i}", spec.run_label).as_bytes()).as_bytes();
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let run_cfg = V2RunConfig::new(identity.clone(), key_seed, config.clone(), grants.clone());
        let run = start_run(&engine, worker_wasm, run_cfg, Box::new(sink.clone()))
            .map_err(|e| format!("worker {i} start_run: {e}"))?;
        workers.push(LiveWorker {
            key: key.clone(),
            peer,
            identity,
            config,
            pump: run.pump.clone(),
            sink,
            run: Some(run),
            engine,
            coord_seq: 0,
            puts: Vec::new(),
            stash: BTreeMap::new(),
        });
    }

    // Admit the roster + exit warmup: joins then ready-heartbeats (the event-driven fast path;
    // the guest's synthetic clock advances one tick per frame).
    for w in &workers {
        coord.deliver(
            &w.key,
            &SwarmMessage::Join(Join {
                run_id: spec.run_label.clone(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: None,
            }),
        )?;
    }
    for w in &workers {
        coord.deliver(
            &w.key,
            &SwarmMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        )?;
    }

    // The drive: consume coordinator decisions in publish order.
    let mut rounds_done = 0u64;
    let mut coordinator_records = 0u64;
    while rounds_done < spec.rounds {
        let (sender, evidence, msg) = coord.next_decision(spec.timeout)?;
        // The network seat's record-authority judgment (the reconciled D1 seam): the §12.1
        // frame's signature is authorized through the run's declared AuthorityConfig — only
        // authoritative frames are relayed. An identity comparison no longer lives here.
        let (auth_sender, _token) =
            authorize_coordinator_frame(&coord_spec.authority, &evidence)
                .map_err(|e| format!("coordinator frame not authoritative: {e}"))?;
        debug_assert_eq!(auth_sender, sender);

        match &msg {
            SwarmMessage::RoundOpen(ro) => {
                let round = ro.round;
                // Stage each worker's assigned batches, then deliver the open.
                for (i, worker) in workers.iter_mut().enumerate() {
                    let mine = assigned_len(ro.batch, ro.seed, &roster, &worker.peer);
                    if mine == 0 || !mine.is_multiple_of(u64::from(spec.steps_per_round)) {
                        return Err(format!(
                            "worker {i} assigned {mine} seqs — not divisible by steps_per_round"
                        ));
                    }
                    for _ in 0..mine {
                        worker
                            .pump
                            .stage_batch(&vec![0u32; SEQ_LEN as usize], 1, SEQ_LEN, None)
                            .map_err(|e| format!("worker {i} stage batch: {e}"))?;
                    }
                    worker.deliver_from_coordinator(sender, &msg, evidence.clone())?;
                }

                // Each worker trains, seals + puts its own container, voices its commitment; the
                // harness relays the commitment (worker-signed) to the coordinator.
                let mut sealed_by_peer: BTreeMap<PeerId, Vec<u8>> = BTreeMap::new();
                for (i, w) in workers.iter_mut().enumerate() {
                    let msgs = w.wait_for(spec.timeout, "commitment", |m| {
                        commitments_for(m, round) >= 1
                    })?;
                    let sealed = w
                        .puts
                        .last()
                        .cloned()
                        .ok_or_else(|| format!("worker {i}: no payload_put after commit"))?;
                    let Some(SwarmMessage::Commitment(commitment)) = msgs
                        .iter()
                        .find(|m| matches!(m, SwarmMessage::Commitment(c) if c.round == round))
                        .cloned()
                    else {
                        return Err(format!("worker {i}: commitment frame missing"));
                    };
                    if commitment.payload != blake3_hash(&sealed) {
                        return Err(format!(
                            "worker {i}: guest commitment hash != serviced put bytes"
                        ));
                    }
                    let key = w.key.clone();
                    coord.deliver(&key, &SwarmMessage::Commitment(commitment))?;
                    sealed_by_peer.insert(w.peer, sealed);
                }

                // Availability evidence (§6.4): the harness (storage seat) authors the receipt;
                // any authenticated sender carries it (`on_receipt` consumes content, not signer).
                let receipt = SwarmMessage::StorageReceipt(StorageReceipt {
                    round,
                    verified: sealed_by_peer
                        .iter()
                        .map(|(peer, sealed)| RecordEntry {
                            peer: *peer,
                            hash: blake3_hash(sealed),
                            size: sealed.len() as u64,
                        })
                        .collect(),
                });
                let key0 = workers[0].key.clone();
                coord.deliver(&key0, &receipt)?;
                // The all-committed + all-evidenced fast paths finalize the round with no clock
                // manipulation; the record arrives as the coordinator's next publish.
                round_payloads_stash(&mut workers, round, sealed_by_peer);
            }
            SwarmMessage::RoundRecord(rr) => {
                coordinator_records += 1;
                let round = rr.round;
                let entries = rr.inline.clone().unwrap_or_default();
                if entries.is_empty() {
                    return Err(format!("round {round} record carries no inline entries"));
                }
                for (i, worker) in workers.iter_mut().enumerate() {
                    // Record-listed staging order (§5.11).
                    let sealed = worker.stash.remove(&round).unwrap_or_default();
                    let ordered: Vec<Vec<u8>> = entries
                        .iter()
                        .map(|e| {
                            sealed
                                .get(&e.peer)
                                .cloned()
                                .ok_or_else(|| "record entry for unknown peer".to_string())
                        })
                        .collect::<Result<_, String>>()?;
                    for p in &ordered {
                        worker
                            .pump
                            .stage_update(p.clone(), None)
                            .map_err(|e| format!("worker {i} stage update: {e}"))?;
                    }
                    worker.deliver_from_coordinator(sender, &msg, evidence.clone())?;
                }
                for w in workers.iter_mut() {
                    let msgs =
                        w.wait_for(spec.timeout, "digest", |m| digests_for(m, round) >= 1)?;
                    // Relay the digest to the coordinator (roster liveness/desync accounting).
                    if let Some(SwarmMessage::Digest(d)) = msgs
                        .iter()
                        .find(|m| matches!(m, SwarmMessage::Digest(d) if d.round == round))
                        .cloned()
                    {
                        let key = w.key.clone();
                        coord.deliver(&key, &SwarmMessage::Digest(d))?;
                    }
                }
                rounds_done += 1;
            }
            _ => { /* notes/chatter — not part of the closed drive */ }
        }
    }

    // Clean stop; §8.7 replay-verify each worker (bit-for-bit decisions).
    let mut reports = Vec::with_capacity(workers.len());
    for (i, mut w) in workers.into_iter().enumerate() {
        w.pump
            .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .map_err(|e| format!("worker {i} stop: {e}"))?;
        let end = w
            .run
            .take()
            .expect("run present")
            .wait()
            .map_err(|e| format!("worker {i} guest thread: {e}"))?;
        let final_state = w
            .pump
            .bridge_final_state()
            .ok_or_else(|| format!("worker {i}: no final bridge state"))?;
        let mut dseed = [0u8; 32];
        dseed[..8].copy_from_slice(&rounds_done.to_le_bytes());
        let digest = *digest_state(&Seed(dseed), 64, u32::MAX, &final_state).as_bytes();

        let entries: Vec<SinkEntry> = w.sink.lock().expect("sink").entries.clone();
        let recorded: Vec<Decision> = entries
            .iter()
            .filter_map(|e| match e {
                SinkEntry::Publish {
                    channel,
                    seq,
                    payload_hash,
                    ..
                } => Some((*channel, *seq, *payload_hash)),
                _ => None,
            })
            .collect();
        let mut script = ReplayScript::from_entries(&entries);
        script.identity = Some(w.identity.clone());
        let replayed = replay_v2(&w.engine, worker_wasm, &w.config, &phase_a_grants(), script)
            .map_err(|e| format!("worker {i} replay harness: {e}"))?;
        let redriven: Vec<Decision> = replayed
            .decisions
            .iter()
            .map(|d| (d.channel, d.seq, d.payload_hash))
            .collect();
        let replay_matched = redriven == recorded && matches!(replayed.end, ReplayEnd::Outcome(_));

        reports.push(Cell8WorkerReport {
            peer: w.peer,
            end,
            digest,
            replay_matched,
            replay_decisions: redriven.len(),
        });
    }

    let coordinator_end = coord.stop()?;

    Ok(Cell8Report {
        workers: reports,
        rounds_done,
        coordinator_records,
        coordinator_end,
        run_id,
    })
}

/// Stash a round's sealed payload set per worker (consumed at the record's staging).
fn round_payloads_stash(
    workers: &mut [LiveWorker],
    round: u64,
    sealed_by_peer: BTreeMap<PeerId, Vec<u8>>,
) {
    for w in workers.iter_mut() {
        w.stash.insert(round, sealed_by_peer.clone());
    }
}
