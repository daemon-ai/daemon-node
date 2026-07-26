// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The MODULE-DRIVEN LIVE CORPUS lane: the production trainer guest feeds itself.
//
// Why this suite exists (the gap it closes). The trainer guest has two data contracts, selected by
// whether its config carries a `live` section:
//
//   * absent  — the HOST stages each batch through `pump.stage_payload` in the wrapper shape
//               `[0, round, step, sequences, seq_len, tokens_le]`. Every gate in the battery used
//               this form.
//   * present — the GUEST feeds itself: fetch the genesis-pinned corpus manifest, register every
//               shard's chunk map, plan the round's assigned sequences into per-shard byte ranges,
//               issue one `data@2` fetch per segment, and stage the decoded tokens itself when the
//               last segment lands.
//
// Only the fleet genesis carries `live`. So the fleet was the only place the second contract had
// ever run, and a defect on it could not be caught before a box ran it — which is exactly what
// happened: the trainer pulled all 30 of its verified corpus ranges and then trapped, on both
// fleet platforms, at the seam between the last segment landing and the round's first step.
//
// What this lane drives, end to end, with nothing host-staged:
//
//   1. the manifest fetch (`ArtifactFetch` over the pinned hash) and per-shard `register_chunks`;
//   2. the round plan — the guest's own `interval_for`/`slice_interval` over the window a
//      `RoundOpen` names — and one `data@2` fetch per planned segment;
//   3. the CHUNK-COLLIDING shape the ceremony corpus actually has: one chunk hash per 2 MiB shard
//      and 512 sequences inside it, so all 30 of a single-peer round's sequences resolve to the
//      SAME covering chunk of the SAME shard. Thirty fetches, thirty distinct sub-ranges, one
//      chunk — the case a manifest with many small chunks would never produce;
//   4. `stage_fetched_batches` decoding those thirty verified slices into thirty staged batches;
//   5. the round's inner loop over them, its commitment, and its θ export.
//
// THE RED LINE is assertion 4 below: the MEASURED peak guest linear memory against the module's
// OWN admitted claim, with a real training step running at the frozen `(seq_len, vocab)`. That is
// the assertion the fleet failed. The forward pass built two guest-resident, geometry-scaled
// images in linear memory — the `s × s` causal mask and the `rows × vocab` one-hot target matrix
// (256 MiB at the frozen values, against a whole-module claim of ~57 MiB) — so the guest's
// allocator hit the sandbox's memory cap and Rust aborted into a wasm `unreachable`. Every prior
// gate ran at 64 tokens, where the same two images are 16 KiB and 8 MiB and fit comfortably.
//
// The bound: the parameter layout is reduced (`ceremony_trainer_config_live_staging` owns the
// argument) because 30 real steps over 786_507_264 parameters at 2048 tokens is ~290 TFLOP. The
// frozen `seq_len` and `vocab` — the two dimensions BOTH guest-resident images are functions of —
// are the fleet's, and the staging path is parameter-independent code. `ceremony_training_step`
// gates the complementary axis (the frozen parameter count at a shortened sequence).
//
// Cost: one round, minutes, single, CPU. Host RAM: the `[2047, 32768]` activation tensors are
// DEVICE-side (host burn-ndarray here) and several are live across the backward pass — a few GiB.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::{
    admit, start_run, DeviceProfile, Dropped, JournalSink, MemoryClaim, OpOutcome, OpRequest,
    OwnerPolicy, ParticipationLane, RunConfig, RunEnd, RunIdentity, SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::corpus::{
    CorpusManifest, Endianness, SequenceBoundary, TokenWidth, TokenizerId, CORPUS_MANIFEST_FORMAT,
};
use daemon_vhc_proto::{to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{BatchWindow, RoundOpen, VhcMessage};
use daemon_vhc_testkit::ceremony::{
    ceremony_trainer_config_live_staging, staging_gate_param_numels, CEREMONY_MICRO_BATCH,
    CEREMONY_SEQ_LEN, CEREMONY_STEPS_PER_ROUND, CEREMONY_VOCAB,
};

/// The sole trainer identity: `peer` and `roster[0]`, so the round's whole window is this peer's.
const PEER: [u8; 32] = [0x5c; 32];

/// The ceremony corpus's chunk size: 2 MiB, which at the frozen 2048-token u16 sequence is 512
/// whole sequences per chunk — and, since the ceremony shards ARE 2 MiB, exactly one chunk hash
/// per shard. This is the number that makes every sequence of a single-peer round collide on one
/// covering chunk.
const CORPUS_CHUNK_SIZE: u64 = 2 << 20;

/// Shards in the gate's corpus. Two, not the ceremony's thirty-two: one shard proves the
/// collision, the second proves a multi-shard manifest still registers every chunk map.
const CORPUS_SHARDS: usize = 2;

/// The run label the `live` section names.
const RUN_LABEL: &str = "vhc-live-staging-gate";

/// The per-slice readback allowance / live-buffer grants of the production TRAINER lane.
const TRAINER_LANE_READBACK_BYTES: u64 = 64 << 20;
const TRAINER_LANE_BUFFER_BYTES: u64 = 1 << 30;
const TRAINER_LANE_BUFFER_HANDLES: u64 = 1024;

/// The concurrent-operation ceiling the production TRAINER lane admits. The ceremony genesis
/// leaves the role's `max_outstanding` grant unbounded, so this lane ceiling is what a fleet
/// trainer actually runs under — and it has to exceed the round's planned segment count, because
/// the guest issues every one of them before returning to its event loop.
const TRAINER_LANE_OUTSTANDING_OPS: u64 = 256;

/// The wall the whole round is given: init, thirty corpus fetches, thirty real optimizer steps at
/// the frozen sequence, and the streamed θ export + commit.
const ROUND_DEADLINE: Duration = Duration::from_secs(2700);

// -- the corpus fixture: the ceremony's chunk-addressed shape ------------------------------------

/// A synthetic corpus in the CEREMONY manifest shape: `CORPUS_SHARDS` shards of exactly
/// `CORPUS_CHUNK_SIZE` bytes each, u16 little-endian tokens, the frozen `seq_len` — so each shard
/// carries one chunk hash and 512 whole sequences.
struct CorpusFixture {
    manifest_bytes: Vec<u8>,
    manifest_hash: Hash,
    /// Shard fold identity → its raw bytes (what the embedder serves ranges out of).
    shards: BTreeMap<[u8; 32], Vec<u8>>,
    /// Fold identities in manifest order.
    order: Vec<[u8; 32]>,
}

fn corpus_fixture() -> CorpusFixture {
    let tokens_per_shard = CORPUS_CHUNK_SIZE / TokenWidth::U16.bytes();
    let mut entries = Vec::with_capacity(CORPUS_SHARDS);
    let mut shards = BTreeMap::new();
    let mut order = Vec::with_capacity(CORPUS_SHARDS);
    for s in 0..CORPUS_SHARDS {
        // Deterministic, distinguishable token ids: the round's staged batches are asserted
        // against these exact bytes, so the decode has to be reproducible from the shard index.
        let bytes: Vec<u8> = (0..tokens_per_shard)
            .flat_map(|t| {
                let id = ((t + 1).wrapping_mul(2_654_435_761) ^ (s as u64) << 32) as u16;
                id.to_le_bytes()
            })
            .collect();
        let entry = CorpusManifest::author_shard(&bytes, tokens_per_shard, CORPUS_CHUNK_SIZE)
            .expect("author the shard");
        assert_eq!(
            entry.chunk_hashes.len(),
            1,
            "the ceremony shape registers exactly ONE chunk hash per shard"
        );
        order.push(entry.shard_hash.0);
        shards.insert(entry.shard_hash.0, bytes);
        entries.push(entry);
    }
    let manifest = CorpusManifest {
        format_version: CORPUS_MANIFEST_FORMAT,
        token_width: TokenWidth::U16,
        endianness: Endianness::Little,
        seq_len: CEREMONY_SEQ_LEN,
        sequence_boundary: SequenceBoundary::WholeSequencesPerShard,
        eos_id: None,
        pad_id: None,
        chunk_size: CORPUS_CHUNK_SIZE,
        tokenizer: TokenizerId {
            hash: Hash([0x7a; 32]),
            name: "live-staging-gate".to_string(),
            revision: "0".to_string(),
        },
        total_tokens: tokens_per_shard * CORPUS_SHARDS as u64,
        shards: entries,
    };
    let manifest_bytes = manifest
        .to_canonical_bytes()
        .expect("the fixture manifest is well formed");
    let manifest_hash = manifest.manifest_hash().expect("manifest hash");
    CorpusFixture {
        manifest_bytes,
        manifest_hash,
        shards,
        order,
    }
}

// -- the journal sink: measure, do not retain ----------------------------------------------------

#[derive(Default)]
struct MeasuringSink {
    publishes: Vec<(u64, u64, usize)>,
    next_seq: u64,
    terminal: Option<String>,
}

impl MeasuringSink {
    fn shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }
}

impl JournalSink for MeasuringSink {
    fn run_header(
        &mut self,
        _abi: u64,
        _worlds: &[(String, u64)],
        _bridge: bool,
        _manifest: &[u8],
        _config: &[u8],
        _grants: &[u8],
        _claim: &[u8],
        _channels: &[u8],
        _device: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn instantiation(&mut self, _counter: u64, _reason: u64, _at: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn init(&mut self, _c: [u8; 32], _g: [u8; 32], _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn execution_grant(&mut self, _hash: [u8; 32], _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn event(&mut self, _at: u64, _frame: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn signed_frame(
        &mut self,
        _channel: u64,
        _seq: u64,
        _sender: [u8; 32],
        _frame: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn next_seq(&mut self, _channel: u64) -> u64 {
        self.next_seq
    }
    fn publish(
        &mut self,
        _channel: u64,
        seq: u64,
        payload: &[u8],
        _frame: &[u8],
    ) -> Result<(), SinkError> {
        self.next_seq = seq + 1;
        // The guest's det-lane voice is `[tag, round, bytes]`; tag 3 is the commitment.
        if let Ok(Value::Array(items)) = ciborium::de::from_reader::<Value, _>(payload) {
            let uint = |i: usize| -> Option<u64> {
                items
                    .get(i)
                    .and_then(Value::as_integer)
                    .and_then(|n| u64::try_from(i128::from(n)).ok())
            };
            if let (Some(tag), Some(round)) = (uint(0), uint(1)) {
                self.publishes.push((tag, round, payload.len()));
            }
        }
        Ok(())
    }
    fn clock(&mut self, _now: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn timer_arm(&mut self, _id: u64, _delay: u64, _armed_at: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn timer_cancel(&mut self, _id: u64, _status: u64) -> Result<(), SinkError> {
        Ok(())
    }
    fn read_back(
        &mut self,
        _src: u64,
        _kind: u64,
        _status: u64,
        _value: &[u8],
    ) -> Result<(), SinkError> {
        Ok(())
    }
    fn device_profile(&mut self, _profile: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn drop_coalesced(&mut self, _c: u64, _r: u64, _d: Dropped) -> Result<(), SinkError> {
        Ok(())
    }
    fn condition(&mut self, _code: &str, _detail: &str) -> Result<(), SinkError> {
        Ok(())
    }
    fn completion(&mut self, _op: u64, _result: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn snapshot(&mut self, _manifest: &[u8]) -> Result<(), SinkError> {
        Ok(())
    }
    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.terminal = Some(format!("kind {kind}, outcome {outcome:?}, trap {trap:?}"));
        Ok(())
    }
}

/// The module's own admitted claim for this config, through the REAL admission funnel — the number
/// the production join engine enforces as the sandbox's linear-memory cap.
fn admitted_claim(wasm: &[u8], cfg_bytes: &[u8]) -> MemoryClaim {
    let mut lane = ParticipationLane::trainer_launch_defaults();
    lane.gpu = 1;
    lane.vram_bytes = 0;
    lane.ram_bytes = 0;
    lane.disk_bytes = 0;
    let device = DeviceProfile {
        gpu: false,
        vram_bytes: 0,
        ram_bytes: 64 << 30,
        disk_bytes: 512 << 30,
    };
    let owner = OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    let assessment =
        Worker::new(EngineConfig::real_model(BackendKind::Cpu, None)).expect("assessment engine");
    admit(
        &assessment,
        wasm,
        Some(blake3::hash(wasm).as_bytes()),
        cfg_bytes,
        &[],
        &lane,
        &device,
        &owner,
        None,
        None,
        // The trainer this staging test drives declares a tiered claim, so no composition authority
        // is involved and the declared claim is what comes back.
        None,
    )
    .expect("the trainer's own claim admits under the production trainer lane")
    .claim
    .expect("a lower-minor trainer declares its own claim")
}

/// What the driven round produced.
struct StagedRound {
    /// The committed container the guest PUT.
    container: Vec<u8>,
    /// Peak guest linear memory over the whole run.
    peak: u64,
    /// The claim the module derived for this config (the enforced cap).
    claim_host: u64,
    /// Every corpus range the guest asked for: `(shard fold, range_off, range_len)`.
    ranges: Vec<([u8; 32], u64, u64)>,
    /// Every COVERING span the pump asked the embedder for: `(shard fold, span_off, span_len)`.
    spans: Vec<([u8; 32], u64, u64)>,
    /// Wall to the commitment.
    wall: Duration,
}

/// Drive one live-staging round to its committed-container PUT.
fn drive_live_round(wasm: &[u8], corpus: &CorpusFixture) -> StagedRound {
    let roster = [PeerId(PEER)];
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_live_staging(
        RUN_LABEL,
        corpus.manifest_hash,
        &roster,
    ))
    .expect("live-staging trainer config");

    let claim = admitted_claim(wasm, &cfg_bytes);
    let engine = EngineConfig::real_model(BackendKind::Cpu, None)
        .with_claimed_memory(claim.hard_accountable.host);
    let worker = Worker::new(engine).expect("engine");

    let identity = RunIdentity {
        run_id: [0x5d; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = corpus_state_chunk_size();
    run_cfg.compute_queue_depth = 1 << 20;
    run_cfg.max_readback_bytes_per_slice = TRAINER_LANE_READBACK_BYTES;
    run_cfg.max_live_buffer_bytes = TRAINER_LANE_BUFFER_BYTES;
    run_cfg.max_live_buffer_handles = TRAINER_LANE_BUFFER_HANDLES;
    run_cfg.max_outstanding_ops = TRAINER_LANE_OUTSTANDING_OPS;
    run_cfg.hard_accountable_host_bytes = claim.declared_peak.host;
    // The genesis grants: the pinned manifest, and every shard by its FOLD identity (a
    // chunk-addressed shard has no whole-object hash — the fold is the grant, the registration
    // key and the fetch key alike).
    let mut granted: BTreeSet<[u8; 32]> = BTreeSet::new();
    granted.insert(corpus.manifest_hash.0);
    granted.extend(corpus.order.iter().copied());
    run_cfg.granted_artifacts = granted;

    let sink = MeasuringSink::shared();
    let t0 = Instant::now();
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    // Round 0's window: the fleet's shape — `steps_per_round × micro_batch` sequences for the sole
    // roster peer, so the guest plans 30 inner steps of one 2048-token sequence each.
    let window = u64::from(CEREMONY_STEPS_PER_ROUND) * u64::from(CEREMONY_MICRO_BATCH);
    let open = to_canonical_vec(&VhcMessage::RoundOpen(RoundOpen {
        round: 0,
        seed: Seed([0; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow {
            start: 0,
            end: window,
        },
        deadline_unix_s: 0,
    }))
    .expect("round open");

    let mut opened = false;
    let mut container: Option<Vec<u8>> = None;
    let mut ranges = Vec::new();
    let mut spans = Vec::new();
    let deadline = Instant::now() + ROUND_DEADLINE;
    loop {
        for (op, request) in pump.take_op_requests() {
            match request {
                // The pinned corpus manifest, by content hash.
                OpRequest::ArtifactFetch { hash, .. } => {
                    assert_eq!(
                        hash, corpus.manifest_hash.0,
                        "the only whole-object fetch is the pinned manifest"
                    );
                    pump.complete_op(
                        op,
                        OpOutcome::FetchDone {
                            artifact: corpus.manifest_bytes.clone(),
                        },
                    )
                    .map(|_| ())
                    .expect("manifest fetch");
                }
                // One planned corpus segment. The embedder is asked for the COVERING SPAN (the
                // whole chunk the range falls in) and serves exactly that; the pump verifies it
                // against the registered chunk hashes and slices the guest's own range out.
                OpRequest::ArtifactRange {
                    hash,
                    range_off,
                    range_len,
                    span_off,
                    span_len,
                } => {
                    let bytes = corpus
                        .shards
                        .get(&hash)
                        .expect("a granted shard fold")
                        .clone();
                    ranges.push((hash, range_off, range_len));
                    spans.push((hash, span_off, span_len));
                    let lo = usize::try_from(span_off).expect("span offset fits usize");
                    let hi = lo + usize::try_from(span_len).expect("span length fits usize");
                    pump.complete_op(
                        op,
                        OpOutcome::RangeDone {
                            bytes: bytes[lo..hi].to_vec(),
                        },
                    )
                    .map(|_| ())
                    .expect("corpus range");
                }
                OpRequest::PayloadPut { bytes } => {
                    container = Some(bytes.to_vec());
                    pump.complete_op(op, OpOutcome::PutDone)
                        .map(|_| ())
                        .expect("put done");
                }
                other => panic!("unexpected op request from the live round: {other:?}"),
            }
        }

        // The guest announces itself (Join + a ready Heartbeat) once its manifest is registered;
        // the round opens after that, exactly as a coordinator would open it.
        if !opened && pump.published().len() >= 2 {
            assert_eq!(
                pump.deliver_frame(0, 0, [9u8; 32], open.clone(), open.clone())
                    .expect("deliver round open"),
                daemon_vhc_host::run::DeliverVerdict::Accepted
            );
            opened = true;
        }

        let committed = sink
            .lock()
            .expect("sink")
            .publishes
            .iter()
            .any(|(tag, round, _)| *tag == 3 && *round == 0);
        if committed {
            break;
        }
        let terminal = sink.lock().expect("sink").terminal.clone();
        if let Some(terminal) = terminal {
            let peak = pump.guest_memory_high_water();
            panic!(
                "the live-staging round ended before its commitment: {terminal}\n  Peak guest \
                 linear memory {peak} B against the admitted claim {} B; {} corpus ranges served.\n\
                 \x20 A GuestPanic here is the module-driven data path failing at the frozen \
                 geometry: read the trap detail — the SDK panic hook forwards the guest's own \
                 message and `file:line:col` through `sys@2::log` — and note that an allocation \
                 abort names the byte count it could not get.",
                claim.hard_accountable.host,
                ranges.len(),
            );
        }
        assert!(
            Instant::now() < deadline,
            "the live-staging round did not commit within {ROUND_DEADLINE:?}; published {:?}, {} \
             corpus ranges served, peak guest linear memory {} B",
            sink.lock().expect("sink").publishes,
            ranges.len(),
            pump.guest_memory_high_water(),
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let wall = t0.elapsed();
    let peak = pump.guest_memory_high_water();

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("the live-staging round must end cleanly, got {other:?}"),
    }

    StagedRound {
        container: container.expect("the round PUTs its committed container"),
        peak,
        claim_host: claim.hard_accountable.host,
        ranges,
        spans,
        wall,
    }
}

/// The run-pinned `state_chunk_size` of the reduced staging layout (the state contract's own).
fn corpus_state_chunk_size() -> u64 {
    daemon_vhc_testkit::ceremony::staging_gate_state_contract().chunk_size
}

/// The gate. **Release-only, and therefore `#[ignore]`d for a bare `cargo test`** — the merge lane
/// runs it explicitly (`xtask vhc-ci-t2`). Like `ceremony_training_step`, its inner loop is real
/// host fp32 arithmetic, which the unoptimized test profile runs an order of magnitude slower.
#[test]
#[ignore = "release-only: thirty real optimizer steps at the frozen 2048-token sequence; \
            `xtask vhc-ci-t2` runs it with --release --ignored"]
fn the_trainer_feeds_itself_from_the_pinned_corpus_and_trains_the_round_it_plans() {
    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let corpus = corpus_fixture();

    // The fixture IS the ceremony's colliding shape: one chunk per shard, and far more sequences
    // inside that chunk than a single-peer round asks for.
    let seqs_per_shard =
        CORPUS_CHUNK_SIZE / (u64::from(CEREMONY_SEQ_LEN) * TokenWidth::U16.bytes());
    let window = u64::from(CEREMONY_STEPS_PER_ROUND) * u64::from(CEREMONY_MICRO_BATCH);
    assert!(
        window <= seqs_per_shard,
        "the gate's premise is that all {window} of the round's sequences fall in ONE shard \
         ({seqs_per_shard} per shard)"
    );

    let round = drive_live_round(&wasm, &corpus);

    // 1. THE PLAN. The guest issued one fetch per planned sequence — 30 inner steps × 1
    //    micro-batch — with no host staging anywhere.
    assert_eq!(
        round.ranges.len() as u64,
        window,
        "the guest plans and fetches one segment per assigned sequence"
    );
    let token_bytes = u64::from(CEREMONY_SEQ_LEN) * TokenWidth::U16.bytes();
    for (i, (shard, off, len)) in round.ranges.iter().enumerate() {
        assert_eq!(
            *shard, corpus.order[0],
            "sequence {i} resolves to the first shard"
        );
        assert_eq!(*len, token_bytes, "each segment is one whole sequence");
        assert_eq!(
            *off,
            i as u64 * token_bytes,
            "the segments tile the assigned interval in training order"
        );
    }

    // 2. THE COLLISION. Every one of those distinct sub-ranges resolved to the SAME covering
    //    chunk — the ceremony corpus's real shape, and the one a many-small-chunks manifest
    //    would never produce. Thirty fetches, one chunk.
    let distinct: BTreeSet<([u8; 32], u64, u64)> = round.spans.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        1,
        "all {window} verified ranges must cover ONE chunk of ONE shard, got {distinct:?}"
    );
    let (_, span_off, span_len) = round.spans[0];
    assert_eq!(
        (span_off, span_len),
        (0, CORPUS_CHUNK_SIZE),
        "the covering span is the whole 2 MiB chunk the manifest registers"
    );

    // 3. THE ROUND COMMITTED. Reaching the tag-3 commitment means the manifest fetched and
    //    registered, the plan resolved, all thirty slices decoded into staged batches, thirty real
    //    optimizer steps ran over them at the frozen sequence, the fence passed, θ exported
    //    window-by-window and the committed container was built and PUT.
    assert!(
        !round.container.is_empty(),
        "the committed container carries this peer's compressed progress"
    );

    // 4. THE RED LINE. The measured peak fits the module's OWN admitted claim, with a real
    //    training step running at the FROZEN `(seq_len, vocab)`. This is the assertion the fleet
    //    failed: the forward pass materialized an `s × s` causal mask and a `rows × vocab` one-hot
    //    in linear memory — 16 MiB and 256 MiB at these values — against a ~57 MiB cap, and the
    //    allocator's failure aborted the guest into a wasm `unreachable`. Both images are built on
    //    DEVICE now; the guest holds only the O(tokens) index rows.
    assert!(
        round.peak <= round.claim_host,
        "measured peak guest linear memory {} B exceeds the admitted claim {} B while training at \
         the frozen seq_len {CEREMONY_SEQ_LEN} / vocab {CEREMONY_VOCAB}. The forward pass is \
         holding a geometry-scaled image in linear memory — build it on device, do not raise the \
         cap",
        round.peak,
        round.claim_host
    );

    let params: usize = staging_gate_param_numels().iter().sum();
    eprintln!(
        "ceremony_live_staging: {window} module-driven corpus ranges ({} B each) over ONE 2 MiB \
         covering chunk, decoded into {window} staged batches and trained through \
         {CEREMONY_STEPS_PER_ROUND} optimizer steps at seq {CEREMONY_SEQ_LEN} / vocab \
         {CEREMONY_VOCAB} over {params} parameters; committed {} B in {:.1} s. MEASURED peak guest \
         linear memory {} B ({:.1} % of the admitted claim {} B)",
        token_bytes,
        round.container.len(),
        round.wall.as_secs_f64(),
        round.peak,
        100.0 * round.peak as f64 / round.claim_host as f64,
        round.claim_host,
    );
}
