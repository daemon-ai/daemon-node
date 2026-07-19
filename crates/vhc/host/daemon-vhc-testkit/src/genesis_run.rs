// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **End-state** — the mixed-fleet matrix's target end-state (decisions D3): coordinator ×
//! v2 workers × envelope v2, as a whole-run harness (refactor §8/D2 acceptance: "the matrix
//! cells for {coordinator × v1/v2 workers} pass").
//!
//! N production `tiny_llama.wasm` trainers (the compute@2 reference worker: a real Burn
//! LLaMA, det-lane ingest in-guest, kind-0 byte staging) train real barrier rounds under the
//! production `coordinator_quorum.wasm` blob — **both** sides running under the real major-2
//! event-loop driver, journaled, §8.7 replay-verified. The harness plays the network +
//! async-runtime seats only:
//!
//! - **coordinator → workers**: the coordinator's §12.1-signed publishes are decoded, checked
//!   against the envelope-named `SingleKey` identity (the D2 thin `Authority` seam at the network
//!   seat — D1's `Authority::accept` replaces the judgment), and relayed to each worker pump with
//!   the original signed frame as tag-12 evidence;
//! - **workers → coordinator**: each worker's module-tagged control voices — the tag-3 commitment
//!   hash and the tag-4 post-ingest det digest — are re-signed as wire `SignedMessage`s under the
//!   worker's node key and delivered to the coordinator pump as host-verified frames (the same
//!   relay shape as the worker binary's self-driven join);
//! - **payload plane**: the harness services `payload_put`, verifies the guest's tag-3 voice
//!   covers exactly the serviced bytes, authors the availability `StorageReceipt`, and stages the
//!   record-listed payload set to every worker (record order, §5.11) as the trainer's
//!   `[1, round, peer32, payload]` kind-0 wrapper.
//!
//! **Round digests are the guest's own det-lane voice** (the tag-4 frame, computed in-guest over
//! the post-ingest canonical state) — never a host-side re-derivation. Cross-worker digest
//! agreement is asserted over those voices.
//!
//! The run is configured **from a genesis envelope v2** ([`crate::configure_coordinator`]) —
//! the coordinator module hash pinned in the role set, its opaque config carried verbatim, the
//! `SingleKey` identity in `[identities]` — which is exactly what makes the envelope-v1 matrix
//! cells typed refusals.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_host::run::{
    replay, start_run, DeliverVerdict, MemorySink, OpOutcome, OpRequest, PumpHandle, ReplayEnd,
    ReplayScript, RunConfig, RunEnd, RunIdentity, SinkEntry,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, ControlTransport, GenesisEnvelope, Hash,
    Identities, IrohId, PeerId, RoleEntry, RoleGrants, RunSection, Seed, SigningKey,
    SnapshotArtifact, StateDigest, TransportSelection, GENESIS_SCHEMA_MAJOR, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig as CoordinatorRunConfig};
use daemon_vhc_sdk_consensus::messages::{
    BatchWindow, Commitment, Digest, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_sdk_consensus::VhcMessage;
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};

use crate::coordinator::{authorize_coordinator_frame, configure_coordinator, Coordinator};
use crate::run::Decision;

/// Tokens per sequence for the staged batches (the t2 geometry; real corpora are `data@2`).
pub const SEQ_LEN: u32 = 9;

/// The trainer model's vocabulary (tokens are embedding indices strictly below it).
const VOCAB: u32 = 64;

/// The Phase-A derived grants document (§2.6 stand-in, byte-identical to the worker's
/// `derive_grants`): the admitted channel table + the worlds the major-2 driver links.
#[must_use]
pub fn phase_a_grants() -> Vec<u8> {
    let channels: Vec<Value> = daemon_vhc_abi::PHASE_A_DEFAULT_CHANNEL_TABLE
        .iter()
        .map(|c| uint(u64::from(c.id)))
        .collect();
    let worlds = ["vhc@2", "net@2", "sys@2"]
        .iter()
        .map(|w| text(w))
        .collect();
    let doc = Value::Map(vec![
        (text("channels"), Value::Array(channels)),
        (text("worlds"), Value::Array(worlds)),
    ]);
    to_canonical_vec(&doc).expect("grants cbor")
}

fn text(s: &str) -> Value {
    Value::Text(s.into())
}

fn uint(v: u64) -> Value {
    Value::Integer(v.into())
}

// -- the fault-injection rig (adversarial-suite seeds, architecture §4.2) -------------------------
//
// Coordinator→worker deliveries pass through a deterministic [`FaultPlan`]: authoritative frames
// can be **dropped** or **duplicated** per `(worker, round, kind)` rule, and a round's
// committed-payload staging can be **delayed** past its record (the straggle trigger at the
// round-driver layer: a record whose payloads cannot be minted stalls; the guest catches up when
// the payloads arrive at the next open).

/// Which coordinator→worker authoritative frame a [`FaultRule`] targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// A `RoundOpen` frame.
    Open,
    /// A `RoundRecord` frame.
    Record,
}

/// What the rig does to a matched delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultAction {
    /// The frame is not delivered at all (loss / withholding).
    Drop,
    /// The frame is delivered twice, byte-identical, same seq — a true duplicate.
    Duplicate,
}

/// One deterministic fault rule: `(worker, round, kind) → action`.
#[derive(Debug, Clone, Copy)]
pub struct FaultRule {
    /// The worker index the fault applies to.
    pub worker: usize,
    /// The round whose frame is targeted.
    pub round: u64,
    /// Which frame of that round.
    pub kind: FrameKind,
    /// What happens to it.
    pub action: FaultAction,
}

/// A deterministic fault plan over a barrier run (module docs). `Default` is fault-free.
#[derive(Debug, Clone, Default)]
pub struct FaultPlan {
    /// Frame-plane faults (drop / duplicate).
    pub rules: Vec<FaultRule>,
    /// Payload-plane faults: `(worker, round)` pairs whose committed payloads are staged only at
    /// the NEXT round's open instead of before the record — the straggle trigger (module docs).
    pub delay_payload_staging: Vec<(usize, u64)>,
}

impl FaultPlan {
    /// The action (if any) for `(worker, round, kind)`.
    #[must_use]
    pub fn action(&self, worker: usize, round: u64, kind: FrameKind) -> Option<FaultAction> {
        self.rules
            .iter()
            .find(|r| r.worker == worker && r.round == round && r.kind == kind)
            .map(|r| r.action)
    }

    /// Whether `(worker, round)`'s committed payloads are delayed past the record.
    #[must_use]
    pub fn payloads_delayed(&self, worker: usize, round: u64) -> bool {
        self.delay_payload_staging.contains(&(worker, round))
    }
}

// -- SDK-free trainer-config authoring -------------------------------------------------------------
//
// The testkit links `host/*` + `contracts/*` only — never `sdk/*` — so the compute@2 trainer's
// config is authored as **raw canonical CBOR** against the guest's documented schema
// (`guests/tiny-llama`: `{"model": ModelCfg, "peer": bstr32, "roster": [bstr32…],
// "steps_per_round": uint, "micro_batch": uint, "stall_rounds_max": uint,
// "profile": SparseLocoCfg, "init": [f32…]}`). The literals are the t2 parity shape (the
// worker-binary genesis join authors the same tiny model). If the guest schema ever moves, this
// fixture fails loud at `da_init` (guest status 16), not silently.

/// The trainer model's canonical parameter element counts for the tiny t2 parity shape
/// (`ModelCfg::param_numels` — tok, per-block 9 params, final norm; d_model 64, 1 layer,
/// 4 heads × head_dim 16, vocab 64, ffn_mult 2).
#[must_use]
fn param_numels() -> Vec<usize> {
    let (d, qdim, hidden, vocab) = (64usize, 64usize, 128usize, 64usize);
    let mut out = vec![vocab * d];
    out.extend([
        d,
        d * qdim,
        d * qdim,
        d * qdim,
        qdim * d,
        d,
        d * hidden,
        d * hidden,
        hidden * d,
    ]);
    out.push(d);
    out
}

/// A deterministic matched init (identical across the roster — the cross-peer digest-agreement
/// precondition; the guest asserts the flat length against its layout).
#[must_use]
fn matched_init(total: usize) -> Vec<f32> {
    let mut s = 0x5EED_C0DEu64;
    (0..total)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            #[allow(clippy::cast_precision_loss)]
            let v = ((s >> 33) % 2001) as f32;
            (v - 1000.0) / 20000.0 // [-0.05, 0.05]
        })
        .collect()
}

/// The compute@2 trainer's guest config, authored SDK-free as raw canonical CBOR (module docs):
/// the tiny t2 parity model + the `sparse_loco` profile + the deterministic matched init.
#[must_use]
pub fn trainer_config(
    peer: &PeerId,
    roster: &[PeerId],
    steps_per_round: u32,
    micro_batch: u32,
    stall_rounds_max: u32,
) -> Vec<u8> {
    let model = Value::Map(vec![
        (text("d_model"), uint(64)),
        (text("n_layers"), uint(1)),
        (text("n_heads"), uint(4)),
        (text("head_dim"), uint(16)),
        (text("vocab"), uint(u64::from(VOCAB))),
        (text("seq_len"), uint(u64::from(SEQ_LEN))),
        (text("ffn_mult"), uint(2)),
        (text("rope_theta"), Value::Float(10_000.0)),
        (text("rmsnorm_eps"), Value::Float(1.0e-5)),
        (text("lr"), Value::Float(4.0e-4)),
        (text("beta1"), Value::Float(0.9)),
        (text("beta2"), Value::Float(0.95)),
        (text("adam_eps"), Value::Float(1.0e-8)),
        (text("wd"), Value::Float(0.1)),
    ]);
    let profile = Value::Map(vec![
        (text("h"), uint(3)),
        (text("ef_decay"), Value::Float(0.95)),
        (text("chunk"), uint(64)),
        (text("topk"), uint(8)),
        (text("bits"), uint(2)),
        (text("outer_alpha"), Value::Float(1.0)),
        (text("clip"), Value::Bool(false)),
    ]);
    let total: usize = param_numels().iter().sum();
    let init = matched_init(total);
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
        (text("profile"), profile),
        (text("init"), Value::serialized(&init).expect("init value")),
    ]);
    to_canonical_vec(&cfg).expect("guest config cbor")
}

// -- the trainer's module wire (kind-0 staged wrappers + tagged publishes) --------------------------

/// Deterministic varied tokens for one staged batch: a pure mixer over `(worker, round, step)` so
/// no two wrappers are byte-identical (the pump's advisory dedup-by-hash never coalesces them).
fn tokens_for(worker: usize, round: u64, step: u32, sequences: u32) -> Vec<u32> {
    let n = u64::from(sequences) * u64::from(SEQ_LEN);
    (0..n)
        .map(|i| {
            let x = i
                + 1
                + 1_000 * u64::from(step)
                + 100_000 * round
                + 10_000_000 * (worker as u64 + 1);
            (x.wrapping_mul(2_654_435_761) % u64::from(VOCAB)) as u32
        })
        .collect()
}

/// The trainer's staged-batch wrapper: `[0, round, step, sequences, seq_len, tokens_le]`
/// (module-wire kind-0 bytes — the compute@2 trainer's contract).
fn batch_wrapper(round: u64, step: u32, sequences: u32, tokens: &[u32]) -> Vec<u8> {
    let mut le = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        le.extend_from_slice(&t.to_le_bytes());
    }
    let v = Value::Array(vec![
        uint(0),
        uint(round),
        uint(u64::from(step)),
        uint(u64::from(sequences)),
        uint(u64::from(SEQ_LEN)),
        Value::Bytes(le),
    ]);
    to_canonical_vec(&v).expect("batch wrapper")
}

/// The trainer's staged committed-payload wrapper: `[1, round, peer32, payload]`.
fn update_wrapper(round: u64, peer: &PeerId, payload: &[u8]) -> Vec<u8> {
    let v = Value::Array(vec![
        uint(1),
        uint(round),
        Value::Bytes(peer.0.to_vec()),
        Value::Bytes(payload.to_vec()),
    ]);
    to_canonical_vec(&v).expect("update wrapper")
}

/// Decode one published frame's module-authored `[tag, round, bytes]` payload.
fn decode_tagged(frame: &[u8]) -> Option<(u64, u64, Vec<u8>)> {
    let v: Value = ciborium::de::from_reader(frame).ok()?;
    let Value::Array(parts) = v else { return None };
    let Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    let inner: Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    let Value::Array(items) = inner else {
        return None;
    };
    let get_uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(Value::as_integer)
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
    };
    let bytes = match items.get(2) {
        Some(Value::Bytes(b)) => b.clone(),
        _ => Vec::new(),
    };
    Some((get_uint(0)?, get_uint(1)?, bytes))
}

/// Count a worker's tag-`tag` voices for `round`.
fn voices_for(voices: &[(u64, u64, Vec<u8>)], tag: u64, round: u64) -> usize {
    voices
        .iter()
        .filter(|(t, r, _)| *t == tag && *r == round)
        .count()
}

// -- the whole-run harness --------------------------------------------------------------------------

/// How a end-state whole run is set up.
pub struct GenesisRunSpec {
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
    /// The deterministic fault plan applied to coordinator→worker deliveries + committed-payload
    /// staging (`Default` = fault-free).
    pub faults: FaultPlan,
    /// Hard wall per wait step.
    pub timeout: Duration,
}

impl GenesisRunSpec {
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
            faults: FaultPlan::default(),
            timeout: Duration::from_secs(180),
        }
    }
}

/// One worker's observable product.
pub struct GenesisWorkerReport {
    /// The worker's peer identity.
    pub peer: PeerId,
    /// How its run ended.
    pub end: RunEnd,
    /// Its module-tagged voices decoded from its publishes, in publish order:
    /// `(tag, round, bytes)` — tag 2 = trained θ, tag 3 = commitment hash, tag 4 = det digest.
    pub voices: Vec<(u64, u64, Vec<u8>)>,
    /// The guest's FINAL tag-4 det-lane digest (its own in-guest det voice — the round-agreement
    /// digest; never a host-side re-derivation).
    pub digest: [u8; 16],
    /// Its §8.7 replay verdict.
    pub replay_matched: bool,
    /// How many decisions the replay re-derived.
    pub replay_decisions: usize,
}

impl GenesisWorkerReport {
    /// Count this worker's tag-4 digest voices for `round`.
    #[must_use]
    pub fn digests_for(&self, round: u64) -> usize {
        voices_for(&self.voices, 4, round)
    }
}

/// The whole run's observable product.
pub struct GenesisRunReport {
    /// Per-worker reports, worker-index order.
    pub workers: Vec<GenesisWorkerReport>,
    /// Rounds driven to a record.
    pub rounds_done: u64,
    /// `RoundRecord`s the coordinator published, in order.
    pub coordinator_records: u64,
    /// How the coordinator's run ended.
    pub coordinator_end: RunEnd,
    /// The run's cryptographic `RunId` (the genesis hash).
    pub run_id: Hash,
}

impl GenesisRunReport {
    /// True iff every worker ended cleanly with a matching §8.7 replay, all workers agree on the
    /// guest-voiced det-lane digest, the coordinator recorded every driven round, and it ended
    /// cleanly.
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

/// Author the end-state **genesis envelope v2** for a barrier run: the coordinator role pinning the
/// `coordinator_quorum.wasm` blob with its opaque `{state: CoordinatorState}` config, one trainer
/// role pinning the worker blob, and the envelope-named `SingleKey` coordinator identity.
///
/// The coordinator's `RunConfig` is authored directly (it is the coordinator module's OPAQUE
/// config under v2 — the v1 `[data]`/`[phases]` sections it used to be projected from left the
/// envelope at D0). Phase deadlines are effectively infinite: the run is driven entirely by
/// event-driven fast paths so the guest's synthetic clock (one tick per event) suffices.
#[must_use]
pub fn genesis_envelope(
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
    let run_config = CoordinatorRunConfig {
        run_id: run_label.to_string(),
        proto_version: VHC_PROTO_VERSION,
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
            url: "r2://mods/tiny_llama.wasm".into(),
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
        run: RunSection {
            schema: GENESIS_SCHEMA_MAJOR,
            run_label: run_label.to_string(),
            min_peers: workers,
            max_peers: workers.max(4),
            access: daemon_vhc_proto::envelope::Access::Org,
        },
        roles,
        artifacts,
        corpus_manifest: None,
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

/// One live worker under the harness.
struct LiveWorker {
    key: SigningKey,
    peer: PeerId,
    identity: RunIdentity,
    config: Vec<u8>,
    pump: PumpHandle,
    sink: Arc<Mutex<MemorySink>>,
    run: Option<daemon_vhc_host::run::Run>,
    engine: Worker,
    /// Per-worker coordinator-plane delivery seq (§12.2 dense-seq discipline, channel 0).
    coord_seq: u64,
    /// The guest's `payload_put` bytes, serviced by the harness.
    puts: Vec<Vec<u8>>,
    /// Round → the round's sealed payload set (stashed at commit, staged at the record).
    stash: BTreeMap<u64, BTreeMap<PeerId, Vec<u8>>>,
    /// Pre-wrapped committed-payload staging bytes whose ingest was delayed past their record
    /// (the straggle trigger): round → record-ordered wrappers, staged at the NEXT open (the
    /// catch-up input).
    held_payloads: BTreeMap<u64, Vec<Vec<u8>>>,
}

impl LiveWorker {
    /// Relay a coordinator decision: the coordinator's ORIGINAL §12.1 signed frame is the tag-12
    /// evidence; the payload is the decoded control message's canonical bytes; the sender is the
    /// coordinator's §12.1 frame identity (checked against the envelope-named authority upstream).
    fn deliver_from_coordinator(
        &mut self,
        sender: [u8; 32],
        msg: &VhcMessage,
        evidence: Vec<u8>,
    ) -> Result<(), String> {
        let seq = self.coord_seq;
        self.coord_seq += 1;
        self.deliver_from_coordinator_seq(sender, msg, evidence, seq)
    }

    /// Re-deliver the same coordinator frame with the SAME seq — a true duplicate (byte-identical
    /// frame, same §12.2 seq): the round driver's ingest watermark is the dedup under test.
    fn deliver_duplicate_from_coordinator(
        &mut self,
        sender: [u8; 32],
        msg: &VhcMessage,
        evidence: Vec<u8>,
    ) -> Result<(), String> {
        let seq = self.coord_seq.saturating_sub(1);
        self.deliver_from_coordinator_seq(sender, msg, evidence, seq)
    }

    fn deliver_from_coordinator_seq(
        &mut self,
        sender: [u8; 32],
        msg: &VhcMessage,
        evidence: Vec<u8>,
        seq: u64,
    ) -> Result<(), String> {
        let payload = to_canonical_vec(msg).map_err(|e| format!("payload encode: {e}"))?;
        match self
            .pump
            .deliver_frame(0, seq, sender, payload, evidence)
            .map_err(|e| format!("deliver: {e}"))?
        {
            DeliverVerdict::Accepted => Ok(()),
            other => Err(format!(
                "coordinator frame back-pressured/refused ({other:?}) in the whole-run drive"
            )),
        }
    }

    /// The worker's module-tagged voices, decoded from its publishes in publish order.
    fn voices(&self) -> Vec<(u64, u64, Vec<u8>)> {
        self.pump
            .published()
            .into_iter()
            .filter_map(|(_, _, frame)| decode_tagged(&frame))
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
                        "unexpected op request from the trainer guest: {other:?}"
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
        pred: impl Fn(&[(u64, u64, Vec<u8>)]) -> bool,
    ) -> Result<Vec<(u64, u64, Vec<u8>)>, String> {
        let deadline = Instant::now() + timeout;
        loop {
            self.service_ops()?;
            let voices = self.voices();
            if pred(&voices) {
                return Ok(voices);
            }
            if Instant::now() >= deadline {
                return Err(format!("timed out waiting for {what}"));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Stage one round's assigned batches in training order: the module's own slicing math
    /// (assignment interval → `steps_per_round` inner steps → `micro_batch`-sized micro-windows),
    /// one kind-0 wrapper per micro-window.
    fn stage_batches(
        &mut self,
        index: usize,
        round: u64,
        mine: u64,
        steps_per_round: u32,
        micro_batch: u32,
    ) -> Result<(), String> {
        let per_step = mine / u64::from(steps_per_round);
        for h in 0..steps_per_round {
            let mut cursor = 0u64;
            while cursor < per_step {
                let seqs = (per_step - cursor).min(u64::from(micro_batch)) as u32;
                self.pump
                    .stage_payload(
                        batch_wrapper(round, h, seqs, &tokens_for(index, round, h, seqs)),
                        None,
                    )
                    .map_err(|e| format!("worker {index} stage batch: {e}"))?;
                cursor += u64::from(seqs);
            }
        }
        Ok(())
    }
}

fn assigned_len(window: BatchWindow, seed: Seed, roster: &[PeerId], peer: &PeerId) -> u64 {
    let weighted: Vec<(PeerId, ThroughputClass)> =
        roster.iter().map(|p| (*p, ThroughputClass::C1)).collect();
    daemon_vhc_sdk_consensus::assign_batches(&weighted, &seed, window, 0)
        .into_iter()
        .find(|(p, _)| p == peer)
        .map_or(0, |(_, w)| w.end.saturating_sub(w.start))
}

/// Drive the whole run: N production compute@2 trainers through `spec.rounds` barrier rounds
/// under the production coordinator, configured from a genesis envelope v2; then stop
/// everything cleanly and §8.7 replay-verify each worker.
///
/// # Errors
/// A `String` on any harness-level failure.
#[allow(clippy::too_many_lines)]
pub fn genesis_whole_run(
    coordinator_wasm: &[u8],
    worker_wasm: &[u8],
    spec: &GenesisRunSpec,
) -> Result<GenesisRunReport, String> {
    let coord_hash = Hash(*blake3::hash(coordinator_wasm).as_bytes());
    let worker_hash = *blake3::hash(worker_wasm).as_bytes();
    let grants = phase_a_grants();

    // Identities: the coordinator's §12.1 frame key IS the envelope-named SingleKey identity
    // (launch topology, architecture §4.4); worker node keys are index-derived.
    let coord_key_seed =
        *blake3::hash(format!("genesis_run-coordinator/{}", spec.run_label).as_bytes()).as_bytes();
    let coord_identity = peer_id(&SigningKey::from_bytes(&coord_key_seed));
    let worker_keys: Vec<SigningKey> = (0..spec.workers)
        .map(|i| {
            SigningKey::from_bytes(
                blake3::hash(format!("genesis_run-worker/{}/{i}", spec.run_label).as_bytes())
                    .as_bytes(),
            )
        })
        .collect();
    let roster: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    // The genesis envelope v2 + the coordinator configuration derived from it (the
    // configuration half — the exact seat that REFUSES under envelope v1).
    let genesis = genesis_envelope(
        &spec.run_label,
        coord_hash,
        Hash(worker_hash),
        coord_identity,
        spec.workers as u32,
        spec.steps_per_round,
        spec.global_batch,
    );
    let author = SigningKey::from_bytes(
        blake3::hash(format!("genesis_run-author/{}", spec.run_label).as_bytes()).as_bytes(),
    );
    let frozen = genesis
        .freeze(&author)
        .map_err(|e| format!("genesis freeze: {e}"))?;
    let coord_spec =
        configure_coordinator(&frozen).map_err(|e| format!("coordinator config: {e}"))?;
    let run_id = coord_spec.run_id;

    let mut coord = Coordinator::start(
        coordinator_wasm,
        &coord_spec,
        grants.clone(),
        0,
        coord_key_seed,
    )?;

    // Start every worker under the real event-loop driver, keyed by the cryptographic RunId.
    let mut workers: Vec<LiveWorker> = Vec::with_capacity(spec.workers);
    for (i, key) in worker_keys.iter().enumerate() {
        let peer = roster[i];
        let config = trainer_config(&peer, &roster, spec.steps_per_round, 1, 2);
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
            *blake3::hash(format!("genesis_run-frame-key/{}/{i}", spec.run_label).as_bytes())
                .as_bytes();
        let sink = Arc::new(Mutex::new(MemorySink::new()));
        let mut run_cfg =
            RunConfig::new(identity.clone(), key_seed, config.clone(), grants.clone());
        // A real transformer's per-round op stream exceeds the tiny default queue depth (the
        // guest also fences per inner step to reclaim depth).
        run_cfg.compute_queue_depth = 1 << 20;
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
            held_payloads: BTreeMap::new(),
        });
    }

    // Admit the roster + exit warmup: joins then ready-heartbeats (the event-driven fast path;
    // the guest's synthetic clock advances one tick per frame).
    for w in &workers {
        coord.deliver(
            &w.key,
            &VhcMessage::Join(Join {
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
            &VhcMessage::Heartbeat(Heartbeat {
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
            VhcMessage::RoundOpen(ro) => {
                let round = ro.round;
                // Stage each worker's assigned batches, then deliver the open — first flushing any
                // payload wrappers held from an earlier record (the straggle catch-up input,
                // staged BEFORE this open so the driver ingests the stalled round then trains the
                // new one; the guest folds the caught-up round's digest into its state — the
                // compute@2 trainer voices tag-4 only from the record handler).
                for (i, worker) in workers.iter_mut().enumerate() {
                    let held: Vec<Vec<Vec<u8>>> = worker.held_payloads.values().cloned().collect();
                    for wrappers in held {
                        for wb in wrappers {
                            worker
                                .pump
                                .stage_payload(wb, None)
                                .map_err(|e| format!("worker {i} stage held update: {e}"))?;
                        }
                    }
                    worker.held_payloads.clear();

                    let mine = assigned_len(ro.batch, ro.seed, &roster, &worker.peer);
                    if mine == 0 || !mine.is_multiple_of(u64::from(spec.steps_per_round)) {
                        return Err(format!(
                            "worker {i} assigned {mine} seqs — not divisible by steps_per_round"
                        ));
                    }
                    worker.stage_batches(i, round, mine, spec.steps_per_round, 1)?;
                    deliver_open_under_faults(
                        worker,
                        &spec.faults,
                        i,
                        round,
                        sender,
                        &msg,
                        &evidence,
                    )?;
                }

                // Each worker trains, exports θ, seals + puts its own container, and voices its
                // tag-3 commitment hash; the harness verifies the voice covers exactly the
                // serviced put bytes and relays a worker-signed Commitment to the coordinator.
                let mut sealed_by_peer: BTreeMap<PeerId, Vec<u8>> = BTreeMap::new();
                for (i, w) in workers.iter_mut().enumerate() {
                    if spec.faults.action(i, round, FrameKind::Open) == Some(FaultAction::Drop) {
                        continue; // the open never arrived; this worker sits the round out
                    }
                    let voices =
                        w.wait_for(spec.timeout, "commitment", |v| voices_for(v, 3, round) >= 1)?;
                    let sealed = w
                        .puts
                        .last()
                        .cloned()
                        .ok_or_else(|| format!("worker {i}: no payload_put after commit"))?;
                    let guest_hash: [u8; 32] = voices
                        .iter()
                        .rev()
                        .find_map(|(t, r, bytes)| (*t == 3 && *r == round).then(|| bytes.clone()))
                        .ok_or_else(|| format!("worker {i}: tag-3 commitment voice missing"))?
                        .as_slice()
                        .try_into()
                        .map_err(|_| format!("worker {i}: tag-3 voice is not 32 bytes"))?;
                    if guest_hash != blake3_hash(&sealed).0 {
                        return Err(format!(
                            "worker {i}: guest commitment hash != serviced put bytes (evidence \
                             must be authored over the guest's own sealed bytes)"
                        ));
                    }
                    let key = w.key.clone();
                    coord.deliver(
                        &key,
                        &VhcMessage::Commitment(Commitment {
                            round,
                            payload: Hash(guest_hash),
                            size: sealed.len() as u64,
                            locators: Vec::new(),
                        }),
                    )?;
                    sealed_by_peer.insert(w.peer, sealed);
                }

                // Availability evidence (§6.4): the harness (storage seat) authors the receipt;
                // any authenticated sender carries it (`on_receipt` consumes content, not signer).
                let receipt = VhcMessage::StorageReceipt(StorageReceipt {
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
            VhcMessage::RoundRecord(rr) => {
                coordinator_records += 1;
                let round = rr.round;
                let entries = rr.inline.clone().unwrap_or_default();
                if entries.is_empty() {
                    return Err(format!("round {round} record carries no inline entries"));
                }
                for (i, worker) in workers.iter_mut().enumerate() {
                    // Record-listed staging order (§5.11): resolve each entry to its sealed
                    // bytes and pre-wrap it as the trainer's `[1, round, peer, payload]` kind-0
                    // staging wrapper.
                    let sealed = worker.stash.remove(&round).unwrap_or_default();
                    let ordered: Vec<Vec<u8>> = entries
                        .iter()
                        .map(|e| {
                            sealed
                                .get(&e.peer)
                                .map(|p| update_wrapper(round, &e.peer, p))
                                .ok_or_else(|| "record entry for unknown peer".to_string())
                        })
                        .collect::<Result<_, String>>()?;
                    if spec.faults.payloads_delayed(i, round) {
                        // The straggle trigger: the record arrives, its payloads do not (staged at
                        // the next open instead) — the driver stalls; the compute@2 trainer voices
                        // nothing for the stalled round (the straggle detour is observable as the
                        // absent tag-4 + the preserved final-state agreement).
                        worker.held_payloads.insert(round, ordered);
                    } else {
                        for wb in &ordered {
                            worker
                                .pump
                                .stage_payload(wb.clone(), None)
                                .map_err(|e| format!("worker {i} stage update: {e}"))?;
                        }
                    }
                    deliver_record_under_faults(
                        worker,
                        &spec.faults,
                        i,
                        round,
                        sender,
                        &msg,
                        &evidence,
                    )?;
                }
                for (i, w) in workers.iter_mut().enumerate() {
                    if spec.faults.payloads_delayed(i, round)
                        || spec.faults.action(i, round, FrameKind::Record)
                            == Some(FaultAction::Drop)
                    {
                        // Straggling (or record-less) workers voice no digest for this round.
                        continue;
                    }
                    let voices =
                        w.wait_for(spec.timeout, "digest", |v| voices_for(v, 4, round) >= 1)?;
                    // Relay the digest to the coordinator (roster liveness/desync accounting).
                    if let Some(digest16) = voices.iter().rev().find_map(|(t, r, bytes)| {
                        (*t == 4 && *r == round)
                            .then(|| <[u8; 16]>::try_from(bytes.as_slice()).ok())
                            .flatten()
                    }) {
                        let key = w.key.clone();
                        coord.deliver(
                            &key,
                            &VhcMessage::Digest(Digest {
                                round,
                                digest: StateDigest(digest16),
                            }),
                        )?;
                    }
                }
                rounds_done += 1;
            }
            _ => { /* notes/chatter — not part of the closed drive */ }
        }
    }

    // Payloads delayed in the FINAL driven round have no next open in the closed drive; pull one
    // more coordinator open so the stragglers catch up (the coordinator keeps opening rounds).
    // The catch-up ingest happens inside the open's event slice; the new round's tag-3 commitment
    // voice is the completion barrier (the caught-up digest folds into state, unvoiced).
    if workers.iter().any(|w| !w.held_payloads.is_empty()) {
        let mut opened = false;
        while !opened {
            let (sender, evidence, msg) = coord.next_decision(spec.timeout)?;
            if let VhcMessage::RoundOpen(ro) = &msg {
                for (i, worker) in workers.iter_mut().enumerate() {
                    let held: Vec<Vec<Vec<u8>>> = worker.held_payloads.values().cloned().collect();
                    for wrappers in held {
                        for wb in wrappers {
                            worker
                                .pump
                                .stage_payload(wb, None)
                                .map_err(|e| format!("worker {i} stage held update: {e}"))?;
                        }
                    }
                    worker.held_payloads.clear();
                    let mine = assigned_len(ro.batch, ro.seed, &roster, &worker.peer);
                    worker.stage_batches(i, ro.round, mine, spec.steps_per_round, 1)?;
                    worker.deliver_from_coordinator(sender, &msg, evidence.clone())?;
                    let round = ro.round;
                    worker.wait_for(spec.timeout, "catch-up commitment", |v| {
                        voices_for(v, 3, round) >= 1
                    })?;
                }
                opened = true;
            }
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
        let voices = w.voices();
        // The guest's FINAL tag-4 det digest — its own det-lane voice (the round-agreement
        // digest); the harness never re-derives state host-side.
        let digest: [u8; 16] = voices
            .iter()
            .rev()
            .find_map(|(t, _, bytes)| {
                (*t == 4)
                    .then(|| <[u8; 16]>::try_from(bytes.as_slice()).ok())
                    .flatten()
            })
            .ok_or_else(|| format!("worker {i}: no tag-4 digest voice in the run"))?;

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
        let replayed = replay(&w.engine, worker_wasm, &w.config, &phase_a_grants(), script)
            .map_err(|e| format!("worker {i} replay harness: {e}"))?;
        let redriven: Vec<Decision> = replayed
            .decisions
            .iter()
            .map(|d| (d.channel, d.seq, d.payload_hash))
            .collect();
        let replay_matched = redriven == recorded && matches!(replayed.end, ReplayEnd::Outcome(_));

        reports.push(GenesisWorkerReport {
            peer: w.peer,
            end,
            voices,
            digest,
            replay_matched,
            replay_decisions: redriven.len(),
        });
    }

    let coordinator_end = coord.stop()?;

    Ok(GenesisRunReport {
        workers: reports,
        rounds_done,
        coordinator_records,
        coordinator_end,
        run_id,
    })
}

/// Deliver a `RoundOpen` to one worker under the fault plan: drop it (the worker sits the round
/// out), duplicate it byte-identically (same §12.2 seq — the ingest-watermark dedup), or deliver
/// it once.
fn deliver_open_under_faults(
    worker: &mut LiveWorker,
    faults: &FaultPlan,
    i: usize,
    round: u64,
    sender: [u8; 32],
    msg: &VhcMessage,
    evidence: &[u8],
) -> Result<(), String> {
    match faults.action(i, round, FrameKind::Open) {
        Some(FaultAction::Drop) => Ok(()),
        Some(FaultAction::Duplicate) => {
            worker.deliver_from_coordinator(sender, msg, evidence.to_vec())?;
            worker.deliver_duplicate_from_coordinator(sender, msg, evidence.to_vec())
        }
        None => worker.deliver_from_coordinator(sender, msg, evidence.to_vec()),
    }
}

/// Deliver a `RoundRecord` to one worker under the fault plan (same actions as the open).
fn deliver_record_under_faults(
    worker: &mut LiveWorker,
    faults: &FaultPlan,
    i: usize,
    round: u64,
    sender: [u8; 32],
    msg: &VhcMessage,
    evidence: &[u8],
) -> Result<(), String> {
    match faults.action(i, round, FrameKind::Record) {
        Some(FaultAction::Drop) => Ok(()),
        Some(FaultAction::Duplicate) => {
            worker.deliver_from_coordinator(sender, msg, evidence.to_vec())?;
            worker.deliver_duplicate_from_coordinator(sender, msg, evidence.to_vec())
        }
        None => worker.deliver_from_coordinator(sender, msg, evidence.to_vec()),
    }
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
