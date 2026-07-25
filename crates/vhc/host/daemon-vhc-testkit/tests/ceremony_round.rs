// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The REAL-GEOMETRY guest ROUND-PATH gate: the production trainer guest walks one whole round —
// θ export -> `make_update` (commit) -> ingest -> quiesce — at the FROZEN fleet-ceremony geometry
// (`daemon_vhc_testkit::ceremony` — 786_507_264 parameters) under the production sandbox budgets,
// on the CPU lane.
//
// Why this suite exists (the class it locks down). Its sibling `ceremony_geometry` proves the
// guest's INIT streams at this geometry; it stops there, and the round path carried the identical
// residency class unfixed: the round's θ arrived as whole-parameter device readbacks collected
// into a whole-family `Vec<Vec<f32>>`, ingest assembled the post-ingest master into a second whole
// family before uploading it, and a drain/checkpoint read both AdamW moment families back whole to
// seal them. At this geometry each of those is ~2.93 GiB of wasm32 linear memory (which the 64 MiB
// cap cannot hold) and the tied embedding alone is a single 192 MiB readback (which the per-slice
// readback allowance cannot pass). Every OTHER guest lane in the battery runs a toy geometry, so
// all of it passed everywhere and would have died on the fleet at round 0 — exactly how the init
// defect reached a fleet smoke.
//
// What it drives, and what it deliberately does not. The round is opened with `steps_per_round = 0`
// (`ceremony_trainer_config_round_walk`): the barrier still opens, commits, fences and exports, so
// every STATE walk runs at full ceremony size, but the 30-step inner loop — hours of ndarray CPU at
// seq 2048 over 24 layers — does not. The trainer goldens own the training math (bit-exact, toy
// geometry); this owns the geometry. The committed payload rides a low `topk` for the reason
// documented on that config helper: the profile's payload wire is a whole-blob container, so its
// size is a payload-plane residency class of its own, separate from the state families under test.
//
// The two assertions that carry the class:
//
//  1. COMPLETION under the production budgets. `EngineConfig::real_model` keeps the toy-tier 64 MiB
//     linear-memory cap, so reaching `QuiesceReady` at all is the bounded-guest-memory proof — a
//     whole-family θ/master/moment buffer traps `GuestPanic` (alloc abort) long before here.
//  2. WINDOWED READBACKS. Every device export is journaled verbatim (§8.3 tag 2, kind
//     `READBACK_KIND_TENSOR_EXPORT`), so the journal IS the record of how much crossed the boundary
//     per readback. The gate asserts the LARGEST one is a window, not a parameter — which fails on
//     the old whole-tensor export by three orders of magnitude (192 MiB vs ~4 MiB) even on a host
//     with enough memory to have survived assertion 1.
//
// Cost: this and `ceremony_geometry` are the battery's two real-geometry lanes. It expands + folds
// the ~2.93 GiB init, then walks ~3 GiB of θ off the device, folds a full master family, and seals
// both moment families — the real fp32 device working set (master + both moments ≈ 8.8 GiB of
// ndarray tensors) plus the sealed families in the state store. Minutes, single, no training math.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::{
    start_run, Dropped, JournalSink, OpOutcome, OpRequest, RunConfig, RunEnd, RunIdentity,
    SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::det_state::family_byte_len;
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, VhcMessage,
};
use daemon_vhc_testkit::ceremony::{
    ceremony_param_numels, ceremony_state_chunk_size, ceremony_trainer_config_round_walk,
    CEREMONY_PARAM_COUNT,
};

/// The sole trainer identity the harness form pins as `peer`/`roster`.
const PEER: [u8; 32] = [0x3b; 32];

/// The compression density this gate drives (see `ceremony_trainer_config_round_walk`): one value
/// per 1536-wide profile chunk keeps the whole-blob payload container ~10 MB instead of ~210 MB.
const ROUND_WALK_TOPK: u64 = 1;

/// The per-slice readback allowance the production TRAINER lane grants
/// (`ParticipationLane::trainer_launch_defaults`). The gate runs the real grant, not the driver
/// default, because the windowed readbacks under test are exactly what it bounds.
const TRAINER_LANE_READBACK_BYTES: u64 = 64 << 20;

/// A journal sink that MEASURES instead of retaining. The run reads ~3 GiB of θ and ~6 GiB of
/// moments back off the device and every byte is journaled verbatim (§8.3 tag 2) — a `MemorySink`
/// would hold all of it in host RAM beside the 8.8 GiB device working set. This keeps the one
/// statistic the gate asserts on (the largest single tensor-export readback) and the publish
/// stream, and drops the bulk.
#[derive(Default)]
struct MeasuringSink {
    /// The largest single `READBACK_KIND_TENSOR_EXPORT` value, in bytes.
    max_export_readback: u64,
    /// How many tensor exports were read back.
    export_readbacks: u64,
    /// Published `[tag, round, …]` frames, as `(tag, round, payload len)`.
    publishes: Vec<(u64, u64, usize)>,
    /// The accepted §10.2 snapshot manifests (the drain's), verbatim — small, so kept.
    snapshots: Vec<Vec<u8>>,
    /// Per-channel durable publish sequence.
    next_seq: u64,
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
        if let Some((tag, round)) = decode_tagged(payload) {
            self.publishes.push((tag, round, payload.len()));
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
        kind: u64,
        _status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        if kind == u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT) {
            self.export_readbacks += 1;
            self.max_export_readback = self.max_export_readback.max(value.len() as u64);
        }
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
    fn snapshot(&mut self, manifest: &[u8]) -> Result<(), SinkError> {
        self.snapshots.push(manifest.to_vec());
        Ok(())
    }
    fn terminal(
        &mut self,
        _kind: u64,
        _outcome: Option<u64>,
        _trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        Ok(())
    }
}

/// The `[tag, round, bytes]` head of one of the trainer's control publishes.
fn decode_tagged(payload: &[u8]) -> Option<(u64, u64)> {
    let Value::Array(items) = ciborium::de::from_reader(payload).ok()? else {
        return None;
    };
    let uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(Value::as_integer)
            .and_then(|n| u64::try_from(i128::from(n)).ok())
    };
    Some((uint(0)?, uint(1)?))
}

/// The section names of an accepted §10.2 state manifest, in declaration order (decoded at the
/// value level — the gate reads the manifest the guest authored, not a host-side type).
fn manifest_section_names(manifest: &[u8]) -> Vec<String> {
    let Value::Map(fields) = ciborium::de::from_reader(manifest).expect("manifest cbor") else {
        panic!("a state manifest is a map");
    };
    let sections = fields
        .iter()
        .find_map(|(k, v)| matches!(k, Value::Text(t) if t == "sections").then_some(v))
        .expect("the manifest declares sections");
    let Value::Array(sections) = sections else {
        panic!("sections is an array");
    };
    sections
        .iter()
        .map(|s| {
            let Value::Map(fields) = s else {
                panic!("a section is a map")
            };
            fields
                .iter()
                .find_map(|(k, v)| match (k, v) {
                    (Value::Text(k), Value::Text(name)) if k == "name" => Some(name.clone()),
                    _ => None,
                })
                .expect("a section declares a name")
        })
        .collect()
}

fn round_open(round: u64) -> VhcMessage {
    VhcMessage::RoundOpen(RoundOpen {
        round,
        seed: Seed([round as u8; 32]),
        roster_digest: Hash([0; 32]),
        // Zero-width: `steps_per_round = 0` slices no inner steps either way (the gate drives the
        // round path, not the training math).
        batch: BatchWindow { start: 0, end: 0 },
        deadline_unix_s: 0,
    })
}

fn round_record(round: u64, payload: &[u8]) -> VhcMessage {
    let entry = RecordEntry {
        peer: PeerId(PEER),
        hash: blake3_hash(payload),
        size: payload.len() as u64,
    };
    VhcMessage::RoundRecord(RoundRecord {
        round,
        set: commit_set(&[(PeerId(PEER), entry.hash)]).commitment(),
        drops: Vec::new(),
        next_seed: Seed([0; 32]),
        set_locator: Locator::StoreKey(String::new()),
        inline: Some(vec![entry]),
    })
}

/// The staged committed-payload wrapper the harness contract takes: `[1, round, peer32, payload]`.
fn update_wrapper(round: u64, payload: &[u8]) -> Vec<u8> {
    let v = Value::Array(vec![
        Value::from(1u8),
        Value::from(round),
        Value::Bytes(PEER.to_vec()),
        Value::Bytes(payload.to_vec()),
    ]);
    to_canonical_vec(&v).expect("update wrapper")
}

/// Service the guest's op requests, capturing the committed container it PUTs (the gate feeds the
/// trainer its own payload back at the barrier — the single-peer committed set).
fn service_ops(pump: &daemon_vhc_host::run::PumpHandle, put: &mut Option<Vec<u8>>) {
    for (op, request) in pump.take_op_requests() {
        match request {
            OpRequest::PayloadPut { bytes } => {
                *put = Some(bytes.to_vec());
                pump.complete_op(op, OpOutcome::PutDone).expect("put done");
            }
            other => panic!("unexpected op request from the trainer guest: {other:?}"),
        }
    }
}

/// Pump until `cond` holds, servicing ops (the guest's own puts) as they arrive.
fn wait_for(
    pump: &daemon_vhc_host::run::PumpHandle,
    put: &mut Option<Vec<u8>>,
    what: &str,
    timeout: Duration,
    cond: impl Fn(&daemon_vhc_host::run::PumpHandle) -> bool,
) {
    let deadline = Instant::now() + timeout;
    loop {
        service_ops(pump, put);
        if cond(pump) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}; published {} frame(s), logs: {:?}",
            pump.published().len(),
            pump.logs()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn ceremony_geometry_trainer_round_streams_under_the_production_budgets() {
    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let numels = ceremony_param_numels();
    let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
    let family_bytes = family_byte_len(&numels_u64);
    let window_size = ceremony_state_chunk_size();
    assert_eq!(
        numels_u64.iter().sum::<u64>(),
        CEREMONY_PARAM_COUNT,
        "the gate drives the frozen ceremony geometry"
    );

    // The production join-lane profile — notably NOT a raised linear-memory cap: a conforming
    // guest streams its families, so the toy-tier 64 MiB cap must suffice at any geometry.
    let engine = EngineConfig::real_model(BackendKind::Cpu, None);
    assert_eq!(
        engine.max_memory_bytes,
        EngineConfig::default().max_memory_bytes,
        "the real-model profile must not buy its way past the bounded-guest-memory invariant"
    );
    let worker = Worker::new(engine).expect("engine");

    let identity = RunIdentity {
        run_id: [0xce; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_round_walk(
        &[PeerId(PEER)],
        ROUND_WALK_TOPK,
    ))
    .expect("ceremony trainer config (round-walk form)");
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = window_size;
    run_cfg.compute_queue_depth = 1 << 20;
    // The production trainer lane's grants, not the driver defaults: the windowed readbacks under
    // test are charged against exactly this allowance.
    run_cfg.max_readback_bytes_per_slice = TRAINER_LANE_READBACK_BYTES;

    let sink = MeasuringSink::shared();
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();
    let mut put: Option<Vec<u8>> = None;
    let mut seq = 0u64;
    let sender = [9u8; 32];
    let deliver = |msg: &VhcMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::run::DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    // -- round 0: open -> commit -> fence -> the streamed θ export + make_update walk -------------
    // The open is delivered while the guest is still streaming its init (~100 s at this geometry);
    // the guest observes it at its first `next_event`, after init completes.
    deliver(&round_open(0), &mut seq);
    wait_for(
        &pump,
        &mut put,
        "the round-0 commitment (tag 3) — the θ export + make_update walk",
        Duration::from_secs(1800),
        |pump| !pump.published().is_empty(),
    );
    let payload = put.take().unwrap_or_else(|| {
        // Copy the diagnostic out and RELEASE the lock before panicking: the guest thread journals
        // through this sink, and unwinding while holding it would poison the mutex and bury the
        // real failure under a lock panic.
        let (publishes, readbacks, largest) = {
            let m = sink.lock().expect("sink");
            (
                m.publishes.clone(),
                m.export_readbacks,
                m.max_export_readback,
            )
        };
        panic!(
            "the round-0 θ export + make_update walk must reach the committed-container PUT; \
             published {:?} (tag, round, bytes) after {} device export readback(s), largest {} B. \
             A tag-9 `export-failed` with one oversize readback is the whole-parameter export this \
             gate exists to catch: at this geometry the tied embedding is a single 192 MiB export, \
             past both the live-buffer grant and the per-slice readback allowance.",
            publishes, readbacks, largest,
        )
    });

    // -- the barrier: feed the trainer its own committed payload, then the record ------------------
    pump.stage_payload(update_wrapper(0, &payload), None)
        .expect("stage the committed payload");
    deliver(&round_record(0, &payload), &mut seq);
    wait_for(
        &pump,
        &mut put,
        "the round-0 post-ingest digest (tag 4) — the streamed ingest walk",
        Duration::from_secs(1800),
        |pump| pump.published().len() >= 2,
    );

    // -- the drain: the §10.2 producing protocol seals both moment families window-by-window -------
    pump.quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, 1_800_000)
        .expect("quiesce delivery");
    wait_for(
        &pump,
        &mut put,
        "the quiesce snapshot",
        Duration::from_secs(1800),
        |_| !sink.lock().expect("sink").snapshots.is_empty(),
    );

    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(2) => {} // OUTCOME_QUIESCE_READY
        other => panic!(
            "the ceremony-geometry ROUND path must complete inside the production sandbox budgets, \
             got {other:?} (a GuestPanic here is a guest-resident family — a whole-θ export \
             collection, a whole master assembly, or a whole moment family at seal — that does not \
             scale to the fleet geometry; OutOfFuel is the real-model per-slice fuel budget)"
        ),
    }

    let stats = pump.state_store_stats();
    let measured = sink.lock().expect("sink");

    // The round voiced its commitment and its post-ingest digest, and NOT the trained-θ frame: at
    // this geometry the θ image is ~2.93 GiB, which is neither a publishable control frame nor a
    // bounded guest buffer, so the harness-tier parity voice is skipped by construction.
    let tags: Vec<u64> = measured.publishes.iter().map(|(t, _, _)| *t).collect();
    assert_eq!(
        tags,
        vec![3, 4],
        "the round path voices the commitment then the digest (and no gigabyte θ frame)"
    );

    // THE readback assertion: every device export crossed the boundary as a WINDOW. The largest
    // tensor-export readback is one `state_chunk_size` window plus its `TensorData` CBOR framing —
    // not one parameter (the tied embedding is 192 MiB, 48× the window and 3× the per-slice
    // allowance) and not one family.
    let readback_ceiling = window_size + 4096;
    assert!(
        measured.max_export_readback <= readback_ceiling,
        "device exports must be WINDOWED: largest readback {} B over a {window_size} B window \
         (+framing) across {} exports — a whole-parameter export is the residency class this gate \
         exists to catch",
        measured.max_export_readback,
        measured.export_readbacks,
    );
    // A round's worth of windowed exports: θ once, then both moment families at the drain.
    let windows_per_family =
        daemon_vhc_proto::det_state::family_chunk_count(&numels_u64, window_size);
    assert_eq!(
        measured.export_readbacks,
        3 * windows_per_family,
        "one windowed export per θ / adamw_m / adamw_v window"
    );

    // The drain's accepted manifest is the proof that BOTH moment families sealed off the device
    // window-by-window (a `state_seal` whose emitted bytes fall short of the declared family length
    // traps host-side, so a declared section IS a fully-streamed family).
    assert_eq!(measured.snapshots.len(), 1, "one accepted drain manifest");
    let sections = manifest_section_names(&measured.snapshots[0]);
    assert_eq!(
        sections,
        vec!["master", "ef", "adamw_m", "adamw_v"],
        "the drain declares all four families by fold"
    );

    // The state plane the round left behind. The FOLD count is deliberately not asserted: with no
    // training math θ equals the round base, so the round's master re-seals to the init's fold and
    // every all-zero family (ef and both moments) folds to one shared identity — the store dedups
    // them content-addressed, which is correct behavior, not missing work (the four seals are
    // evidenced by the tag-3/tag-4 voices and the manifest above). What must hold is that a full
    // family image is retained.
    assert!(
        stats.retained_bytes >= family_bytes,
        "the sealed master family must retain the full {family_bytes} B image, got {stats:?}"
    );

    eprintln!(
        "ceremony_round: {} windowed device exports, largest readback {} B (window {} B); \
         {} folds sealed, {} retained bytes at {} parameters; payload {} B (topk {})",
        measured.export_readbacks,
        measured.max_export_readback,
        window_size,
        stats.sealed_folds,
        stats.retained_bytes,
        CEREMONY_PARAM_COUNT,
        payload.len(),
        ROUND_WALK_TOPK,
    );
}
