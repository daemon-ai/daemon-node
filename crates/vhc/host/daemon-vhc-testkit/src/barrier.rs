// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The barrier-round whole-run harness: N production wasm workers under the in-process
//! [`crate::NativeCoordinator`], journaled and §8.7 replay-verified — the "native coordinator +
//! wasm workers" whole-run shape of refactor §6, generalizing `daemon-vhc-worker`'s A2 t2
//! `v2_session` drive into reusable testkit infrastructure.
//!
//! ## SDK-free authoring (the dependency wall)
//!
//! The testkit links `host/*` + `contracts/*` only — never `sdk/*` — so the tiny-llama guest
//! config is authored here as **raw canonical CBOR** against the guest's documented schema
//! (`guests/tiny-llama-v2`: `{"model": TinyLlamaCfg, "peer": bstr32, "roster": [bstr32…],
//! "steps_per_round": uint, "micro_batch": uint, "stall_rounds_max": uint}`), with every
//! non-defaultable `TinyLlamaCfg` field written out explicitly. The values mirror the A2 t2
//! parity configuration (1-layer tiny model, seq_len 9, `sparse_loco` h=3/chunk=64/topk=8).
//! If the SDK schema ever moves, this fixture fails loud at `da_init` (guest status 16), not
//! silently.
//!
//! ## The fault-injection rig (adversarial-suite seeds, architecture §4.2)
//!
//! Coordinator→worker deliveries pass through a deterministic [`FaultPlan`]: authoritative
//! frames can be **dropped** or **duplicated** per `(worker, round, kind)` rule, and a round's
//! committed-payload staging can be **delayed** past its record (the straggle trigger at the
//! round-driver layer: a record whose payloads cannot be minted stalls, voices
//! `Straggle{fetching}`, and catches up when the payloads arrive — delaying the *record frame*
//! alone produces no straggle signal, because the driver only stalls on an unmintable pending
//! record). The rig itself lives in [`crate::cell8`] (the wasm-coordinator whole-run harness —
//! the adversarial drills' home); this native drive shares it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::v2::{
    replay_v2, start_run, DeliverVerdict, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd,
    ReplayScript, RunEnd, RunIdentity, SinkEntry, V2RunConfig,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, ExperimentSection, GlobalBatch, Phases, Requirements, RoundMode,
    RunSection, StopCondition,
};
use daemon_vhc_proto::messages::{
    BatchWindow, RecordEntry, StorageReceipt, SwarmMessage, ThroughputClass,
};
use daemon_vhc_proto::{
    blake3_hash, digest_state, from_canonical_slice, peer_id, to_canonical_vec, Envelope, Hash,
    PeerId, Seed, SignedMessage, SigningKey,
};

use crate::cell8::{phase_a_grants, FaultAction, FrameKind, SEQ_LEN};
use crate::coordinator::NativeCoordinator;
use crate::run::Decision;

pub use crate::cell8::{FaultPlan, FaultRule};

// -- SDK-free fixture authoring ---------------------------------------------------------------------

fn text(s: &str) -> Value {
    Value::Text(s.into())
}

fn uint(v: u64) -> Value {
    Value::Integer(v.into())
}

/// The tiny-llama-v2 guest config, authored SDK-free as raw canonical CBOR (module docs). The
/// model block mirrors the A2 t2 parity configuration: the 1-layer tiny model whose parameter
/// sizes are divisible by the `sparse_loco` chunking.
#[must_use]
pub fn tiny_llama_config(
    peer: &PeerId,
    roster: &[PeerId],
    steps_per_round: u32,
    micro_batch: u32,
    stall_rounds_max: u32,
) -> Vec<u8> {
    // TinyLlamaCfg, every non-`#[serde(default)]` field explicit (diloco/demo default cleanly;
    // sparse_loco must be explicit because the tiny preset overrides the paper defaults).
    let adamw = Value::Map(vec![
        (text("lr"), Value::Float(4.0e-4)),
        (text("beta1"), Value::Float(0.9)),
        (text("beta2"), Value::Float(0.95)),
        (text("eps"), Value::Float(1.0e-8)),
        (text("wd"), Value::Float(0.1)),
    ]);
    let sparse_loco = Value::Map(vec![
        (text("h"), uint(3)),
        (text("ef_decay"), Value::Float(0.95)),
        (text("chunk"), uint(64)),
        (text("topk"), uint(8)),
        (text("bits"), uint(2)),
        (text("outer_alpha"), Value::Float(1.0)),
        (text("clip"), Value::Bool(false)),
    ]);
    let model = Value::Map(vec![
        (text("d_model"), uint(64)),
        (text("n_layers"), uint(1)),
        (text("n_heads"), uint(4)),
        (text("n_kv_heads"), uint(4)),
        (text("head_dim"), uint(16)),
        (text("vocab"), uint(64)),
        (text("seq_len"), uint(u64::from(SEQ_LEN))),
        (text("ffn_mult"), uint(2)),
        (text("rope_theta"), Value::Float(10_000.0)),
        (text("rmsnorm_eps"), Value::Float(1.0e-5)),
        (text("inner"), adamw),
        (text("profile"), text("sparse_loco")),
        (text("sparse_loco"), sparse_loco),
    ]);
    let cfg = Value::Map(vec![
        (text("model"), model),
        (text("peer"), Value::Bytes(peer.0.to_vec())),
        (
            text("roster"),
            Value::Array(roster.iter().map(|p| Value::Bytes(p.0.to_vec())).collect()),
        ),
        (text("steps_per_round"), uint(u64::from(steps_per_round))),
        (text("micro_batch"), uint(u64::from(micro_batch))),
        (text("stall_rounds_max"), uint(u64::from(stall_rounds_max))),
    ]);
    to_canonical_vec(&cfg).expect("guest config cbor")
}

/// A schema-major-1 envelope for a barrier run: `global_batch` sequences per round over
/// `steps_per_round` inner steps (the v1 envelope shape the native coordinator consumes).
#[must_use]
pub fn barrier_envelope(
    run_id: &str,
    min_peers: u32,
    max_peers: u32,
    steps_per_round: u32,
    global_batch: u32,
) -> Envelope {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
        },
    );
    Envelope {
        run: RunSection {
            schema: 1,
            run_id: run_id.into(),
            min_peers,
            max_peers,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config: Value::Null,
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round,
            global_batch: GlobalBatch {
                start: global_batch,
                end: global_batch,
                ramp_rounds: 1,
            },
            stop: StopCondition::Tokens(1_000_000),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 1,
            uplink_mbps_min: 1,
            downlink_mbps_min: 1,
            disk_gb_min: 1,
            throughput_floor: "c1".into(),
            update_mb_max: 8,
            capabilities: vec![],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 60,
            round_witness: 1,
            cooldown: 1,
            epoch_rounds: 10,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    }
}

// -- the whole-run harness -----------------------------------------------------------------------------

/// How a barrier whole-run is set up.
pub struct BarrierSpec {
    /// A stable run label (hashed into the 32-byte run id).
    pub run_id: String,
    /// Wasm workers to run (2+ is the multi-worker shape). Worker `i` gets instance id `i + 1`.
    pub workers: usize,
    /// Rounds to drive to completion.
    pub rounds: u64,
    /// Inner steps per round (must divide each worker's assigned window).
    pub steps_per_round: u32,
    /// Sequences per round across the roster (the envelope `global_batch`).
    pub global_batch: u32,
    /// The deterministic fault plan (`Default` = fault-free).
    pub faults: FaultPlan,
    /// Hard wall per wait step so a wedged run cannot hang the gate.
    pub timeout: Duration,
}

impl BarrierSpec {
    /// A fault-free spec with the A2 t2 per-worker geometry (`steps_per_round = 2`, one assigned
    /// sequence per inner step per worker).
    #[must_use]
    pub fn new(run_id: &str, workers: usize, rounds: u64) -> Self {
        Self {
            run_id: run_id.to_string(),
            workers,
            rounds,
            steps_per_round: 2,
            global_batch: 2 * workers as u32,
            faults: FaultPlan::default(),
            timeout: Duration::from_secs(180),
        }
    }
}

/// One worker's observable product.
pub struct WorkerReport {
    /// The worker's peer identity.
    pub peer: PeerId,
    /// How its run ended.
    pub end: RunEnd,
    /// Its recorded publishes as decoded control messages (in publish order).
    pub messages: Vec<SwarmMessage>,
    /// Its final det-lane state digest (16 bytes, the round agreement digest input).
    pub digest: [u8; 16],
    /// Its §8.7 replay verdict: recorded decisions reproduced bit-for-bit and a clean outcome.
    pub replay_matched: bool,
    /// How many decisions (publishes) the replay re-derived.
    pub replay_decisions: usize,
}

/// The whole run's observable product.
pub struct BarrierRunReport {
    /// Per-worker reports, worker-index order.
    pub workers: Vec<WorkerReport>,
    /// Rounds driven to a record.
    pub rounds_done: u64,
}

impl BarrierRunReport {
    /// True iff every worker ended cleanly, every §8.7 replay matched, and — the det-lane
    /// agreement claim — every worker holds the identical final state digest.
    #[must_use]
    pub fn is_green(&self) -> bool {
        let ends_clean = self
            .workers
            .iter()
            .all(|w| matches!(w.end, RunEnd::Outcome(0)) && w.replay_matched);
        let first = self.workers.first().map(|w| w.digest);
        ends_clean && self.workers.iter().all(|w| Some(w.digest) == first)
    }
}

/// One live worker under the harness.
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
    /// Sealed containers awaiting delayed staging: round → record-ordered payloads.
    held_payloads: BTreeMap<u64, Vec<Vec<u8>>>,
    /// The guest's `payload_put` bytes, serviced by the harness (the async-runtime seat): the
    /// B1 sealing-gap retirement means the GUEST seals + puts its own container, and the
    /// embedder captures the bytes here (commitment evidence + barrier staging input).
    puts: Vec<Vec<u8>>,
}

impl LiveWorker {
    fn deliver_signed(
        &mut self,
        coord: &NativeCoordinator,
        msg: &SwarmMessage,
        seq: u64,
    ) -> Result<(), String> {
        let signed = coord.sign(msg.clone())?;
        signed
            .verify()
            .map_err(|e| format!("coordinator frame REFUSED above the pump: {e}"))?;
        let payload = to_canonical_vec(msg).map_err(|e| format!("payload encode: {e}"))?;
        let evidence = to_canonical_vec(&signed).map_err(|e| format!("evidence encode: {e}"))?;
        match self
            .pump
            .deliver_frame(0, seq, coord.sender(), payload, evidence)
            .map_err(|e| format!("deliver: {e}"))?
        {
            DeliverVerdict::Accepted => Ok(()),
            other => Err(format!(
                "coordinator frame back-pressured/refused ({other:?}) — the barrier drive \
                 never fills the spool (a SpoolFull adversarial case sets its own expectations)"
            )),
        }
    }

    /// Deliver a coordinator message: sign, verify above the pump (the seam under test), deliver
    /// with the original signed bytes as tag-12 evidence.
    fn deliver(&mut self, coord: &NativeCoordinator, msg: &SwarmMessage) -> Result<(), String> {
        let seq = self.coord_seq;
        self.coord_seq += 1;
        self.deliver_signed(coord, msg, seq)
    }

    /// Re-deliver the same message with the SAME seq — a true duplicate (the round driver's
    /// watermark is the dedup under test; B1's per-sender quota admits the re-delivery because
    /// the first copy has already drained by the time the case asserts).
    fn deliver_duplicate(
        &mut self,
        coord: &NativeCoordinator,
        msg: &SwarmMessage,
    ) -> Result<(), String> {
        let seq = self.coord_seq.saturating_sub(1);
        self.deliver_signed(coord, msg, seq)
    }

    /// The worker's publishes decoded as control messages (signed frame → payload → SwarmMessage).
    fn messages(&self) -> Vec<SwarmMessage> {
        decode_published(&self.pump.published())
    }

    /// Service the guest's outstanding op requests (the async-runtime seat): `payload_put`
    /// bytes are captured into [`Self::puts`]; anything else is unexpected in the barrier drive
    /// (tiny-llama-v2 stages batches through the kind-1 path, not `data.fetch`).
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
                        "unexpected op request from the barrier guest: {other:?}"
                    ))
                }
            }
        }
        Ok(())
    }

    /// Wait (bounded) until the worker's decoded publish stream satisfies `pred`, servicing the
    /// guest's ops meanwhile (a parked put would otherwise deadlock the commit seam).
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
                return Err(format!(
                    "timed out waiting for {what}; published so far: {:?}",
                    msgs.iter().map(kind_of).collect::<Vec<_>>()
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// Decode `(channel, seq, signed frame)` publishes into their control-message payloads.
fn decode_published(published: &[(u64, u64, Vec<u8>)]) -> Vec<SwarmMessage> {
    published
        .iter()
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

/// A short label for a control message (diagnostics).
fn kind_of(m: &SwarmMessage) -> &'static str {
    match m {
        SwarmMessage::Commitment(_) => "Commitment",
        SwarmMessage::Digest(_) => "Digest",
        SwarmMessage::Straggle(_) => "Straggle",
        SwarmMessage::RoundOpen(_) => "RoundOpen",
        SwarmMessage::RoundRecord(_) => "RoundRecord",
        _ => "Other",
    }
}

/// Count a worker's `Commitment` publishes for `round`.
fn commitments_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Commitment(c) if c.round == round))
        .count()
}

/// Count a worker's `Digest` publishes for `round`.
fn digests_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Digest(d) if d.round == round))
        .count()
}

/// Whether the worker has voiced a `Straggle` for `round`.
fn straggled_for(msgs: &[SwarmMessage], round: u64) -> bool {
    msgs.iter()
        .any(|m| matches!(m, SwarmMessage::Straggle(s) if s.round == round))
}

/// Each worker's assigned sequence window for a round — the same `assign_batches` math the guest
/// runs (assignment lives in `daemon-vhc-proto` until D0, so the harness shares one definition).
fn assigned_len(window: BatchWindow, seed: Seed, roster: &[PeerId], peer: &PeerId) -> u64 {
    let weighted: Vec<(PeerId, ThroughputClass)> =
        roster.iter().map(|p| (*p, ThroughputClass::C1)).collect();
    daemon_vhc_sdk_consensus::assign_batches(&weighted, &seed, window, 0)
        .into_iter()
        .find(|(p, _)| p == peer)
        .map_or(0, |(_, w)| w.end.saturating_sub(w.start))
}

/// Drive N production tiny-llama-v2 workers through `spec.rounds` barrier rounds under the
/// in-process native coordinator, under the spec's fault plan; then stop each cleanly and §8.7
/// replay-verify each worker's journal.
///
/// # Errors
/// A `String` on any harness-level failure (selection, start, a coordinator rejection, a wait
/// timeout, or a worker ending un-cleanly where the drive needed it live).
#[allow(clippy::too_many_lines)]
pub fn barrier_whole_run(wasm: &[u8], spec: &BarrierSpec) -> Result<BarrierRunReport, String> {
    let module_hash = *blake3::hash(wasm).as_bytes();
    let grants = phase_a_grants();

    // Identities: worker keys are index-derived; the coordinator key is run-derived.
    let coord_key = SigningKey::from_bytes(
        blake3::hash(format!("vhc-testkit-coordinator/{}", spec.run_id).as_bytes()).as_bytes(),
    );
    let worker_keys: Vec<SigningKey> = (0..spec.workers)
        .map(|i| {
            SigningKey::from_bytes(
                blake3::hash(format!("vhc-testkit-worker/{}/{i}", spec.run_id).as_bytes())
                    .as_bytes(),
            )
        })
        .collect();
    let roster: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    // The run envelope + the native coordinator over it.
    let envelope = barrier_envelope(
        &spec.run_id,
        spec.workers as u32,
        (spec.workers as u32).max(4),
        spec.steps_per_round,
        spec.global_batch,
    );
    let mut coord = NativeCoordinator::from_envelope(&envelope, coord_key, u64::from(SEQ_LEN))?;

    // Start every worker under the real v2 event-loop driver, journaled into a MemorySink.
    let mut workers: Vec<LiveWorker> = Vec::with_capacity(spec.workers);
    for (i, key) in worker_keys.iter().enumerate() {
        let peer = roster[i];
        let config = tiny_llama_config(
            &peer,
            &roster,
            spec.steps_per_round,
            1, // micro_batch 1: one staged batch per micro window (the t2 geometry)
            envelope.phases.stall_rounds_max,
        );
        let engine = Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
        let sel = select_driver(&engine, wasm, Some(&module_hash))
            .map_err(|e| format!("worker {i} selection: {e}"))?;
        if sel.driver != daemon_vhc_abi::CandidateDriver::V2 {
            return Err(format!(
                "worker {i}: not a major-2 module: {:?}",
                sel.driver
            ));
        }
        let identity = RunIdentity {
            run_id: *blake3::hash(spec.run_id.as_bytes()).as_bytes(),
            epoch: 0,
            role: "trainer".to_string(),
            instance: i as u64 + 1,
            module: module_hash,
        };
        let key_seed =
            *blake3::hash(format!("frame-key/{}/{i}", spec.run_id).as_bytes()).as_bytes();
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let run_cfg = V2RunConfig::new(identity.clone(), key_seed, config.clone(), grants.clone());
        let run = start_run(&engine, wasm, run_cfg, Box::new(sink.clone()))
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
            held_payloads: BTreeMap::new(),
            puts: Vec::new(),
        });
    }

    // Admit the roster.
    for w in &workers {
        coord.join(&w.key, &spec.run_id)?;
    }

    // The drive: pop coordinator messages; advance the clock when the coordinator is quiet.
    // `round_payloads` stashes each round's sealed set (peer → sealed container) between the
    // commit seam and the record's record-listed staging.
    let mut round_payloads: BTreeMap<u64, BTreeMap<PeerId, Vec<u8>>> = BTreeMap::new();
    let mut rounds_done = 0u64;
    let window = spec.global_batch;
    while rounds_done < spec.rounds {
        let Some(msg) = coord.next_message() else {
            coord.advance_bounded(10_000)?;
            continue;
        };
        match &msg {
            SwarmMessage::RoundOpen(ro) => {
                let round = ro.round;
                for (i, worker) in workers.iter_mut().enumerate() {
                    // Delayed payloads from the PREVIOUS round stage now, BEFORE this open is
                    // delivered — the catch-up input for the straggle path (module docs).
                    let held: Vec<(u64, Vec<Vec<u8>>)> = worker
                        .held_payloads
                        .iter()
                        .map(|(r, p)| (*r, p.clone()))
                        .collect();
                    for (_, payloads) in &held {
                        for p in payloads {
                            worker
                                .pump
                                .stage_update(p.clone(), None)
                                .map_err(|e| format!("worker {i} stage held update: {e}"))?;
                        }
                    }
                    worker.held_payloads.clear();

                    // Stage this round's batches (training order): the worker's assigned window,
                    // one zero-token batch per micro window (micro_batch = 1 → per sequence).
                    let mine = assigned_len(ro.batch, ro.seed, &roster, &worker.peer);
                    if mine == 0 || !mine.is_multiple_of(u64::from(spec.steps_per_round)) {
                        return Err(format!(
                            "worker {i} assigned {mine} of {window} seqs — not divisible by \
                             steps_per_round {}; fix the spec geometry",
                            spec.steps_per_round
                        ));
                    }
                    for _ in 0..mine {
                        worker
                            .pump
                            .stage_batch(&vec![0u32; SEQ_LEN as usize], 1, SEQ_LEN, None)
                            .map_err(|e| format!("worker {i} stage batch: {e}"))?;
                    }

                    // Deliver the open, under the fault plan.
                    match spec.faults.action(i, round, FrameKind::Open) {
                        Some(FaultAction::Drop) => continue,
                        Some(FaultAction::Duplicate) => {
                            worker.deliver(&coord, &msg)?;
                            worker.deliver_duplicate(&coord, &msg)?;
                        }
                        None => worker.deliver(&coord, &msg)?,
                    }
                }

                // Each live worker trains, seals + `payload_put`s its OWN container (the B1
                // sealing-gap retirement), and voices its commitment over the put's hash; the
                // harness services the put (async-runtime seat) and verifies the guest's
                // evidence hashes exactly the serviced bytes before relaying it.
                let mut sealed_by_peer: BTreeMap<PeerId, Vec<u8>> = BTreeMap::new();
                for (i, w) in workers.iter_mut().enumerate() {
                    if spec.faults.action(i, round, FrameKind::Open) == Some(FaultAction::Drop) {
                        continue; // the open never arrived; this worker sits the round out
                    }
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
                            "worker {i}: guest commitment hash != serviced put bytes (evidence \
                             must be authored over the guest's own sealed bytes)"
                        ));
                    }
                    let signed = SignedMessage::sign(
                        &w.key,
                        daemon_vhc_proto::SWARM_PROTO_VERSION,
                        SwarmMessage::Commitment(commitment),
                    )
                    .map_err(|e| format!("commitment sign: {e}"))?;
                    coord.feed_message(signed)?;
                    sealed_by_peer.insert(w.peer, sealed);
                }

                // Availability evidence (§6.4): one coordinator-signed receipt over the set.
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
                let signed = coord.sign(receipt)?;
                coord.feed_message(signed)?;

                // Clock past the training window so the round closes into a record; the sealed
                // set stashes harness-side until the record's inline order says how to stage it.
                coord.advance_clock(u64::from(envelope.phases.round_train_max) + 1)?;
                round_payloads.insert(round, sealed_by_peer);
            }
            SwarmMessage::RoundRecord(rr) => {
                let round = rr.round;
                let entries = rr.inline.clone().unwrap_or_default();
                if entries.is_empty() {
                    return Err(format!("round {round} record carries no inline entries"));
                }
                let sealed = round_payloads.remove(&round).unwrap_or_default();
                // Record-listed staging order (§5.11): resolve each entry to its sealed bytes.
                let ordered: Vec<Vec<u8>> = entries
                    .iter()
                    .map(|e| {
                        sealed
                            .get(&e.peer)
                            .cloned()
                            .ok_or_else(|| format!("record entry for unknown peer {:?}", e.peer))
                    })
                    .collect::<Result<_, String>>()?;
                // Harness sanity: the record's hashes are the sealed bytes' hashes.
                for (e, p) in entries.iter().zip(&ordered) {
                    if e.hash != blake3_hash(p) {
                        return Err(format!("record hash mismatch for peer {:?}", e.peer));
                    }
                }

                for (i, worker) in workers.iter_mut().enumerate() {
                    if spec.faults.payloads_delayed(i, round) {
                        // The straggle trigger: the record arrives, its payloads do not.
                        worker.held_payloads.insert(round, ordered.clone());
                    } else {
                        for p in &ordered {
                            worker
                                .pump
                                .stage_update(p.clone(), None)
                                .map_err(|e| format!("worker {i} stage update: {e}"))?;
                        }
                        worker.held_payloads.remove(&round);
                    }
                    match spec.faults.action(i, round, FrameKind::Record) {
                        Some(FaultAction::Drop) => continue,
                        Some(FaultAction::Duplicate) => {
                            worker.deliver(&coord, &msg)?;
                            worker.deliver_duplicate(&coord, &msg)?;
                        }
                        None => worker.deliver(&coord, &msg)?,
                    }
                }
                for (i, w) in workers.iter_mut().enumerate() {
                    if spec.faults.action(i, round, FrameKind::Record) == Some(FaultAction::Drop) {
                        continue;
                    }
                    if spec.faults.payloads_delayed(i, round) {
                        // The straggle path: the driver voices Straggle{fetching}, no digest yet.
                        w.wait_for(spec.timeout, "straggle", |m| straggled_for(m, round))?;
                    } else {
                        w.wait_for(spec.timeout, "digest", |m| digests_for(m, round) >= 1)?;
                    }
                }
                rounds_done += 1;
            }
            _ => { /* witness/digest chatter — not part of the closed Phase-A drive */ }
        }
    }

    // Delayed payloads from the FINAL round have no next open; drive one more open so the
    // stragglers catch up (the coordinator keeps the run alive past `spec.rounds`).
    if workers.iter().any(|w| !w.held_payloads.is_empty()) {
        let mut opened = false;
        while !opened {
            let Some(msg) = coord.next_message() else {
                coord.advance_bounded(10_000)?;
                continue;
            };
            if let SwarmMessage::RoundOpen(ro) = &msg {
                let round = ro.round;
                for (i, worker) in workers.iter_mut().enumerate() {
                    let held: Vec<Vec<Vec<u8>>> = worker.held_payloads.values().cloned().collect();
                    for payloads in held {
                        for p in payloads {
                            worker
                                .pump
                                .stage_update(p, None)
                                .map_err(|e| format!("worker {i} stage held update: {e}"))?;
                        }
                    }
                    let caught_up: Vec<u64> = worker.held_payloads.keys().copied().collect();
                    worker.held_payloads.clear();
                    // Stage this open's batches too — the driver trains the new round after
                    // catching up.
                    let mine = assigned_len(ro.batch, ro.seed, &roster, &worker.peer);
                    for _ in 0..mine {
                        worker
                            .pump
                            .stage_batch(&vec![0u32; SEQ_LEN as usize], 1, SEQ_LEN, None)
                            .map_err(|e| format!("worker {i} stage batch: {e}"))?;
                    }
                    worker.deliver(&coord, &msg)?;
                    for r in caught_up {
                        worker.wait_for(spec.timeout, "catch-up digest", |m| {
                            digests_for(m, r) >= 1
                        })?;
                    }
                    let _ = round;
                }
                opened = true;
            }
        }
    }

    // Clean stop; each guest returns Outcome Ok and exports its final canonical state.
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

        // §8.7: re-drive the recorded journal; every decision must reproduce bit-for-bit.
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
        // The identity behind the recorded run (the tag-0 run header in a real journal): what
        // `sys@2::rng_seed` re-derives from at replay.
        script.identity = Some(w.identity.clone());
        let replayed = replay_v2(&w.engine, wasm, &w.config, &phase_a_grants(), script)
            .map_err(|e| format!("worker {i} replay harness: {e}"))?;
        let redriven: Vec<Decision> = replayed
            .decisions
            .iter()
            .map(|d| (d.channel, d.seq, d.payload_hash))
            .collect();
        let replay_matched = redriven == recorded && matches!(replayed.end, ReplayEnd::Outcome(_));

        reports.push(WorkerReport {
            peer: w.peer,
            end,
            messages: decode_published(&w.pump.published()),
            digest,
            replay_matched,
            replay_decisions: redriven.len(),
        });
    }

    Ok(BarrierRunReport {
        workers: reports,
        rounds_done,
    })
}
