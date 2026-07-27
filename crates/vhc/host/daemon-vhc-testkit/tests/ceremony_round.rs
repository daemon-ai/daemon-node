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
// geometry); this owns the geometry. The compression profile is the FROZEN one (`topk = 64`), so
// the committed payload is the fleet's real ~54 MB container and its ingest is the real one: the
// gate ingests a THREE-peer record whose entries are real-size containers, range-read a fold window
// at a time out of the host buffers `payload_get` delivered ([SF-R3]).
//
// The cap is the CLAIM's, not a host constant. The engine's linear-memory ceiling comes from the
// admitted claim's hard-accountable host tier (`EngineConfig::with_claimed_memory`, architecture
// §3.5) — the module's own `decl_for_config` figure at this exact geometry, run through the real
// admission funnel here. The gate additionally pins that the claim stays UNDER the 64 MiB the host
// used to hardcode: the fix was enforcement wiring, not a raised ceiling, and an honest claim that
// crept past the old constant would be a regression this suite must catch.
//
// The four assertions that carry the class:
//
//  1. COMPLETION under the admitted claim. Reaching `QuiesceReady` at all is the bounded-guest-
//     memory proof — a whole-family θ/master/moment buffer, or a whole decoded peer payload, traps
//     `GuestPanic` (alloc abort) long before here.
//  2. THE MEASURED PEAK. The pump samples the guest's linear memory at every event slice (wasm
//     memory never shrinks, so the sample IS the high-water). The gate asserts the MEASUREMENT
//     against the claim, so "it did not trap" is not the evidence — the number is.
//  3. WINDOWED READBACKS. Every device export is journaled verbatim (§8.3 tag 2, kind
//     `READBACK_KIND_TENSOR_EXPORT`), so the journal IS the record of how much crossed the boundary
//     per readback. The gate asserts the LARGEST one is a window, not a parameter — which fails on
//     the old whole-tensor export by three orders of magnitude (192 MiB vs ~4 MiB) even on a host
//     with enough memory to have survived assertion 1.
//  4. THE RED LINE (`ceremony_geometry_whole_payload_ingest_cannot_fit_the_admitted_claim`): the
//     retired whole-blob ingest's per-peer residency, computed from the run's OWN pinned geometry,
//     against the same claim — and a live proof that this engine really refuses an over-cap guest,
//     so assertion 1 is not vacuous.
//
// Cost: this and `ceremony_geometry` are the battery's two real-geometry lanes. It expands + folds
// the ~2.93 GiB init, then walks ~3 GiB of θ off the device, folds a full master family against a
// three-peer committed set, and seals both moment families — the real fp32 device working set
// (master + both moments ≈ 8.8 GiB of ndarray tensors) plus the sealed families in the state store.
// Minutes, single, no training math.

// Dev/test harness: the guest builder shells `cargo` for the guests workspace.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::admission::{Admission, ResourceAuthority};
use daemon_vhc_host::run::{
    admit, start_run, DeviceProfile, Dropped, JournalSink, OpOutcome, OpRequest, OwnerPolicy,
    ParticipationLane, RunConfig, RunEnd, RunIdentity, SinkError,
};
use daemon_vhc_host::{BackendKind, EngineConfig, Worker};
use daemon_vhc_proto::det_state::family_byte_len;
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_resource::revision::BackendClass;
use daemon_vhc_resource::{test_support, ReservationIdentity};
use daemon_vhc_sdk_consensus::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, VhcMessage,
};
use daemon_vhc_testkit::ceremony::{
    ceremony_param_numels, ceremony_profile_chunk, ceremony_state_chunk_size,
    ceremony_trainer_config_round_walk, CEREMONY_PARAM_COUNT,
};

/// The trainer identities the record lists. The first is the run's own `peer` (the harness form
/// pins `roster[0]`); the frozen ceremony roster is three trainers, so the gate folds a three-peer
/// committed set — the fleet's shape, at the fleet's payload size.
const PEERS: [[u8; 32]; 3] = [[0x3b; 32], [0x3c; 32], [0x3d; 32]];

/// The FROZEN compression density (`ceremony_profile_value`): 64 selected values per 1536-wide
/// profile chunk. Restated here only so the gate can compute the retired path's residency and the
/// container's own size from the same numbers the config carries.
const CEREMONY_TOPK: u64 = 64;

/// The absmax value width the frozen profile packs at (`ceremony_profile_value`).
const CEREMONY_BITS: u32 = 2;

/// The committed container's fixed header (the module-policy layout in
/// `daemon-vhc-sdk-profiles::payload`). The testkit links host + contracts only, never the SDK, so
/// the one number it needs from that layout is restated here; the gate's own container measurement
/// below cross-checks it against the bytes the guest actually produced.
const CONTAINER_HEADER_BYTES: u64 = 40;

/// The per-slice readback allowance the production TRAINER lane grants
/// (`ParticipationLane::trainer_launch_defaults`). The gate runs the real grant, not the driver
/// default, because the windowed readbacks under test are exactly what it bounds.
const TRAINER_LANE_READBACK_BYTES: u64 = 64 << 20;

/// The live-buffer grants the production TRAINER lane carries (`max_live_bytes` /
/// `max_live_handles`). Three real-size committed payloads plus the guest's own sealed container
/// are live at once at the barrier, which is what this grant is sized for.
const TRAINER_LANE_BUFFER_BYTES: u64 = 1 << 30;
const TRAINER_LANE_BUFFER_HANDLES: u64 = 1024;

/// The linear-memory ceiling the host used to HARDCODE for every run (`EngineConfig::default`).
/// It is no longer a cap — the admitted claim is — but it stays the gate's yardstick: the fix was
/// enforcement wiring, so an honest claim at the frozen ceremony geometry must still come in under
/// the figure the constant used to impose.
const UNRAISED_MEMORY_CAP: u64 = 64 << 20;

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
    /// The run's terminal record, once it has one. A trapped guest stops producing publishes, so
    /// without this the gate would sit out its whole (necessarily generous, real-geometry) deadline
    /// and then report a timeout instead of the trap that caused it.
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
        _resources: daemon_vhc_host::run::RunHeaderResources<'_>,
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
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.terminal = Some(format!("kind {kind}, outcome {outcome:?}, trap {trap:?}"));
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

/// The round's record: one entry per roster peer, every one naming the SAME container. The gate
/// has one real trainer, so the other two peers' committed updates are that trainer's own container
/// under their identities — which makes the fold a genuine three-peer ingest at the fleet's payload
/// size (three separate `payload_get`s, three host buffers, three ranged reads per fold window)
/// without a second real trainer's hours of ndarray CPU.
fn round_record(round: u64, payload: &[u8]) -> VhcMessage {
    let hash = blake3_hash(payload);
    let entries: Vec<RecordEntry> = PEERS
        .iter()
        .map(|p| RecordEntry {
            peer: PeerId(*p),
            hash,
            size: payload.len() as u64,
        })
        .collect();
    let set: Vec<(PeerId, Hash)> = PEERS.iter().map(|p| (PeerId(*p), hash)).collect();
    VhcMessage::RoundRecord(RoundRecord {
        round,
        set: commit_set(&set).commitment(),
        drops: Vec::new(),
        next_seed: Seed([0; 32]),
        set_locator: Locator::StoreKey(String::new()),
        inline: Some(entries),
    })
}

/// The embedder seat: the guest's own committed container is captured at its `payload_put`, and
/// every record-listed `payload_get` is answered content-addressed from what the seat has
/// published. [SF-R3] — the payload reaches the guest as a host BUFFER it range-reads, so this gate
/// drives the same ingest the fleet does, at the same size.
#[derive(Default)]
struct Seat {
    /// The container the guest PUT this round.
    put: Option<Vec<u8>>,
    /// The fetchable committed containers, by content address.
    store: std::collections::HashMap<[u8; 32], Vec<u8>>,
    /// `payload_get` ops the store could not answer yet (held, never failed).
    deferred: Vec<(u64, [u8; 32])>,
}

impl Seat {
    fn publish_committed(&mut self, payload: &[u8]) {
        self.store.insert(blake3_hash(payload).0, payload.to_vec());
    }
}

fn service_ops(pump: &daemon_vhc_host::run::PumpHandle, seat: &Mutex<Seat>) {
    for (op, request) in pump.take_op_requests() {
        let mut s = seat.lock().expect("seat");
        match request {
            OpRequest::PayloadPut { bytes } => {
                s.put = Some(bytes.to_vec());
                drop(s);
                pump.complete_op(op, OpOutcome::PutDone).expect("put done");
            }
            OpRequest::PayloadGet { hash } => s.deferred.push((op, hash)),
            other => panic!("unexpected op request from the trainer guest: {other:?}"),
        }
    }
    let ready: Vec<(u64, Vec<u8>)> = {
        let mut s = seat.lock().expect("seat");
        let mut pending = Vec::new();
        let mut ready = Vec::new();
        for (op, hash) in std::mem::take(&mut s.deferred) {
            match s.store.get(&hash) {
                Some(bytes) => ready.push((op, bytes.clone())),
                None => pending.push((op, hash)),
            }
        }
        s.deferred = pending;
        ready
    };
    for (op, bytes) in ready {
        pump.complete_op(op, OpOutcome::GetDone { bytes })
            .expect("committed payload get done");
    }
}

/// Pump until `cond` holds, servicing the guest's ops as they arrive. A run that reaches its
/// terminal record without satisfying `cond` fails HERE with the trap, not after the deadline: at
/// this geometry the deadline is half an hour, and a half-hour timeout report hides the one line
/// that says what went wrong.
fn wait_for(
    pump: &daemon_vhc_host::run::PumpHandle,
    seat: &Mutex<Seat>,
    sink: &Arc<Mutex<MeasuringSink>>,
    what: &str,
    timeout: Duration,
    cond: impl Fn(&daemon_vhc_host::run::PumpHandle) -> bool,
) {
    let deadline = Instant::now() + timeout;
    loop {
        service_ops(pump, seat);
        if cond(pump) {
            return;
        }
        // One lock at a time, sink first, guard released before the pump lock is taken: the guest
        // thread journals (sink) from inside imports (pump lock), so holding both here in the other
        // order deadlocks the run.
        let terminal = sink.lock().expect("sink").terminal.clone();
        let peak = pump.guest_memory_high_water();
        if let Some(terminal) = terminal {
            panic!(
                "the run ended before {what}: {terminal}. Peak guest linear memory {peak} B \
                 (a GuestPanic at this seam is an allocation the admitted claim does not cover)"
            );
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}; published {} frame(s), peak guest linear memory \
             {peak} B, logs: {:?}",
            pump.published().len(),
            pump.logs()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run the REAL admission funnel over the ceremony trainer config and return the admission whose
/// composed figures the production join engine enforces as the sandbox's memory cap and this gate
/// measures against — obtained the same way the worker obtains them, not restated.
///
/// The trainer declares the certification minor, so the funnel composes its physical estimate from
/// the Logical Resource Plan the module emitted plus an authenticated Backend Execution Profile —
/// assembled here from the resource crate's fixture assemblers, the one input a test cannot
/// fabricate by hand.
///
/// The lane is the production trainer lane with its DEVICE floor relaxed: this gate runs the CPU
/// backend, and the floor is about which hardware may join, not about what the module may use. Its
/// ceilings — the claim bounds the funnel checks against, the buffer and readback quotas the run
/// then runs under — are the production ones, unmodified.
fn admitted(wasm: &[u8], cfg_bytes: &[u8]) -> Admission {
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
    let (store, running) = test_support::stocked_profile_store(BackendClass::Cpu);
    let policy = test_support::accepting_policy();
    let profile = store
        .select(&test_support::authentication_context(&running, &policy))
        .expect("the fixture profile authenticates under the fixture policy");
    // Supply raised above the ceremony-geometry conservative estimate: this gate's subject is
    // streaming under the production linear-memory budgets, not device supply, and the fixture's
    // stock figure would refuse the geometry before the budgets were ever exercised.
    let report = test_support::capability_report_with_supply(BackendClass::Cpu, 128 << 30);
    let lane_bounds = test_support::generous_lane_bounds();
    let authority = ResourceAuthority {
        profile: &profile,
        report: &report,
        lane_bounds: &lane_bounds,
        co_resident_roles: 1,
        reservation_identity: ReservationIdentity {
            role: "trainer".into(),
            incarnation: 1,
            device_identity: "ceremony-round-device".into(),
            sequence: 1,
        },
        frozen_binding: None,
    };
    // The worker's assessment seat: manifest/claim evaluation never runs compute, so it stays on
    // the roomy CPU profile (`backend::assess_engine_config`) regardless of the measured selection.
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
        Some(&authority),
    )
    .expect("the ceremony trainer's own claim admits under the production trainer lane")
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

    let roster: Vec<PeerId> = PEERS.iter().map(|p| PeerId(*p)).collect();
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_round_walk(&roster))
        .expect("ceremony trainer config (round-walk form)");

    // The cap is the module's OWN composed figure at this geometry, through the real funnel. It is
    // a conservative budget — the plan's walk peaks plus a named fragmentation allowance — so it is
    // allowed to sit above the historical 64 MiB constant; what must NOT exceed that constant is
    // the MEASURED peak, asserted below where the measurement exists. Bounding the estimate by a
    // measurement-era constant would compare a budget to a footprint.
    let admission = admitted(&wasm, &cfg_bytes);
    let engine = EngineConfig::real_model(BackendKind::Cpu, None)
        .with_claimed_memory(admission.admitted_host_bytes());
    assert_eq!(
        engine.max_memory_bytes as u64,
        admission.admitted_host_bytes(),
        "the sandbox's memory cap IS the admitted claim (architecture §3.5), not a host constant"
    );
    let worker = Worker::new(engine).expect("engine");

    let identity = RunIdentity {
        run_id: [0xce; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = window_size;
    run_cfg.compute_queue_depth = 1 << 20;
    // The production trainer lane's grants, not the driver defaults: the windowed readbacks and
    // the three real-size committed payloads under test are charged against exactly these.
    run_cfg.max_readback_bytes_per_slice = TRAINER_LANE_READBACK_BYTES;
    run_cfg.max_live_buffer_bytes = TRAINER_LANE_BUFFER_BYTES;
    run_cfg.max_live_buffer_handles = TRAINER_LANE_BUFFER_HANDLES;
    // The other exactly-metered tier of the same declaration: the host-side bytes the module stages
    // above its linear-memory floor — for this round, the committed container it builds through
    // `buffer_append` and never holds itself (`role_binding` wires exactly this on the live path).
    run_cfg.hard_accountable_host_bytes = admission.hard_accountable_host_bytes();

    let sink = MeasuringSink::shared();
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();
    let seat = Mutex::new(Seat::default());
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
        &seat,
        &sink,
        "the round-0 commitment (tag 3) — the θ export + make_update walk",
        Duration::from_secs(1800),
        |pump| !pump.published().is_empty(),
    );
    eprintln!(
        "ceremony_round[stage]: peak after the θ export + make_update walk = {} B",
        pump.guest_memory_high_water()
    );
    let payload = seat.lock().expect("seat").put.take().unwrap_or_else(|| {
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

    // The container the guest built append-by-append is exactly the geometry's size — the layout is
    // arithmetic from the run's own pinned config, which is what lets a consumer range-address it.
    assert_eq!(
        payload.len() as u64,
        container_bytes(&numels_u64),
        "the committed container's length must be the layout's, computed from the pinned geometry"
    );

    // -- the barrier: publish the committed set, then the three-peer record -----------------------
    // The guest fetches each record entry itself and range-reads the rows one fold window needs
    // ([SF-R3]) — no committed payload ever enters linear memory whole, at the fleet's real size.
    seat.lock().expect("seat").publish_committed(&payload);
    deliver(&round_record(0, &payload), &mut seq);
    wait_for(
        &pump,
        &seat,
        &sink,
        "the round-0 post-ingest digest (tag 4) — the streamed three-peer ingest walk",
        Duration::from_secs(1800),
        |pump| pump.published().len() >= 2,
    );
    eprintln!(
        "ceremony_round[stage]: peak after the three-peer ingest walk = {} B",
        pump.guest_memory_high_water()
    );

    // -- the drain: the §10.2 producing protocol seals both moment families window-by-window -------
    pump.quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, 1_800_000)
        .expect("quiesce delivery");
    wait_for(
        &pump,
        &seat,
        &sink,
        "the quiesce snapshot",
        Duration::from_secs(1800),
        |_| !sink.lock().expect("sink").snapshots.is_empty(),
    );

    // Read the guest's peak linear memory BEFORE the run is torn down.
    let peak = pump.guest_memory_high_water();

    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(2) => {} // OUTCOME_QUIESCE_READY
        other => panic!(
            "the ceremony-geometry ROUND path must complete inside the admitted claim ({} B), got \
             {other:?} (a GuestPanic here is a guest-resident object that does not scale to the \
             fleet geometry — a whole-θ export collection, a whole master assembly, a whole moment \
             family at seal, or a whole decoded peer payload; OutOfFuel is the per-slice fuel \
             budget). Peak guest linear memory reached {peak} B.",
            admission.admitted_host_bytes()
        ),
    }

    // THE MEASURED-PEAK assertion: the run's real footprint, sampled at every event slice, against
    // the claim the module made for this exact geometry. Completion alone would only say "nothing
    // trapped"; this says how much was actually resident — and it is what makes the claim honest
    // evidence rather than a number nobody checks.
    assert!(
        peak <= admission.admitted_host_bytes(),
        "measured peak guest linear memory {peak} B exceeds the admitted claim {} B — the module \
         under-claimed at this geometry (the claim is derived in `decl_for_config`; fix the \
         derivation, do not raise the cap)",
        admission.admitted_host_bytes()
    );
    // The anti-regression tripwire, on the measurement rather than the budget: the residency the
    // enforcement-wiring defect era proved possible at this geometry. A measured peak past this
    // constant is a real residency regression regardless of what the conservative budget permits.
    assert!(
        peak <= UNRAISED_MEMORY_CAP,
        "measured peak guest linear memory {peak} B exceeds the {UNRAISED_MEMORY_CAP} B the host \
         used to hardcode — the defect back then was enforcement wiring, not a low ceiling, so \
         residency above the old constant is a regression, not a bigger appetite"
    );

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
         {} folds sealed, {} retained bytes at {} parameters; committed container {} B per peer \
         (topk {CEREMONY_TOPK}) x {} peers ingested; claim hard/peak host {} / {} B; MEASURED peak \
         guest linear memory {} B ({:.1} % of the claim, {:.1} % of the retired {} B constant)",
        measured.export_readbacks,
        measured.max_export_readback,
        window_size,
        stats.sealed_folds,
        stats.retained_bytes,
        CEREMONY_PARAM_COUNT,
        payload.len(),
        PEERS.len(),
        admission.admitted_host_bytes(),
        admission.hard_accountable_host_bytes(),
        peak,
        100.0 * peak as f64 / admission.admitted_host_bytes() as f64,
        100.0 * peak as f64 / UNRAISED_MEMORY_CAP as f64,
        UNRAISED_MEMORY_CAP,
    );
}

/// THE RED LINE: the retired whole-blob ingest cannot fit the admitted claim, and the claim is
/// really enforced.
///
/// Two halves, both real:
///
///  1. **The arithmetic, from the run's own pinned geometry.** The retired ingest decoded each
///     peer's committed payload up front and held its index section as a `Vec<u32>` — one machine
///     word per selected value, `n_chunks × topk` of them — plus the unpacked values and the
///     container itself. That figure is computed here from the frozen ceremony config (parameter
///     numels, profile chunk, topk, value width), not restated, and compared against the SAME
///     claim the streaming gate above runs under. It is multiples of it, per peer, before the fold
///     allocates anything: a fleet round on the retired path could not start, which is the defect.
///  2. **The enforcement, live.** A cap nobody enforces would make assertion 1 of the streaming
///     gate vacuous ("it completed" proves nothing if nothing bounds it). So the same trainer, the
///     same geometry, is started under a cap BELOW its honest floor and must fail to bring its
///     state plane up. This is the failure MODE the retired path produced — a guest that asks
///     linear memory for more than the sandbox admits — reproduced through the real engine.
#[test]
fn ceremony_geometry_whole_payload_ingest_cannot_fit_the_admitted_claim() {
    let wasm = daemon_vhc_guest_build::guest_wasm("tiny_llama");
    let numels = ceremony_param_numels();
    let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
    let roster: Vec<PeerId> = PEERS.iter().map(|p| PeerId(*p)).collect();
    let cfg_bytes = to_canonical_vec(&ceremony_trainer_config_round_walk(&roster))
        .expect("ceremony trainer config (round-walk form)");
    let admission = admitted(&wasm, &cfg_bytes);

    // -- half 1: the retired residency, derived from the frozen geometry -------------------------
    let chunk = ceremony_profile_chunk();
    let rows: u64 = numels_u64.iter().map(|n| n / chunk).sum();
    let selected = rows * CEREMONY_TOPK;
    // What the retired path materialized per peer, before folding anything:
    //   - the indices as one `Vec<u32>` (the dominant term),
    //   - the unpacked values as one `Vec<f32>`,
    //   - the container itself, read whole out of the payload buffer.
    let resident_indices = selected * 4;
    let resident_values = selected * 4;
    let container = container_bytes(&numels_u64);
    let whole_blob_per_peer = resident_indices + resident_values + container;

    assert!(
        whole_blob_per_peer > admission.admitted_host_bytes(),
        "the red line is not red: one peer's whole-blob residency {whole_blob_per_peer} B fits the \
         admitted claim {} B",
        admission.admitted_host_bytes()
    );
    // The comfort margin is taken against the MEASURED-residency bar (the streaming gate's
    // constant), not against the composed budget: the budget is conservative by design — it
    // carries workspace and fragmentation allowances — so a margin computed against it shrinks
    // whenever the estimate gets more honest, which is the opposite of what this guard watches.
    let over = whole_blob_per_peer as f64 / UNRAISED_MEMORY_CAP as f64;
    assert!(
        over >= 4.0,
        "one peer's whole-blob residency is only {over:.1}x the measured-residency bar \
         ({UNRAISED_MEMORY_CAP} B) — the frozen geometry's margin collapsed; re-derive before \
         trusting either gate"
    );
    eprintln!(
        "ceremony_round[red line]: {rows} compression rows x topk {CEREMONY_TOPK} = {selected} \
         selected values per peer -> retired resident form {whole_blob_per_peer} B ({} B indices \
         + {} B values + {} B container) = {over:.1}x the measured-residency bar, against the \
         admitted claim {} B, PER PEER, x {} peers; the streamed form reads {} B of container \
         rows per fold window per peer",
        resident_indices,
        resident_values,
        container,
        admission.admitted_host_bytes(),
        PEERS.len(),
        window_section_bytes(),
    );

    // -- half 2: the cap is really enforced ------------------------------------------------------
    // A cap of TWO fold windows: derived from the run's own geometry, comfortably above the wasm
    // image's static data (so the instance starts and the guest really runs) and below what any
    // streaming walk here needs — the init expansion alone holds the generated window, its f32-le
    // image and the zeroed `ef` window. So the guest must fail to bring its state plane up.
    let starved = 2 * ceremony_state_chunk_size();
    assert!(
        starved < admission.admitted_host_bytes(),
        "the starved cap must be below the honest claim to prove anything"
    );
    let engine = EngineConfig::real_model(BackendKind::Cpu, None).with_claimed_memory(starved);
    let worker = Worker::new(engine).expect("engine");
    let identity = RunIdentity {
        run_id: [0xcf; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(identity, [0x9d; 32], cfg_bytes, Vec::new());
    run_cfg.state_chunk_size = ceremony_state_chunk_size();
    run_cfg.compute_queue_depth = 1 << 20;
    run_cfg.max_readback_bytes_per_slice = TRAINER_LANE_READBACK_BYTES;

    let sink = MeasuringSink::shared();
    match start_run(&worker, &wasm, run_cfg, Box::new(sink)) {
        // The instance never came up: the pooling allocator refused the module's own memory
        // demand outright. The cap is enforced at the earliest possible point.
        Err(_) => {}
        Ok(run) => {
            let pump = run.pump.clone();
            pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
                .expect("stop");
            let end = run.wait().expect("guest thread clean");
            assert!(
                !matches!(end, RunEnd::Outcome(0)),
                "a guest starved to {starved} B — two fold windows, against an honest claim of \
                 {} B — brought its ceremony-geometry state plane up anyway: the claim-derived cap \
                 is NOT being enforced, which would make the streaming gate's completion assertion \
                 vacuous",
                admission.admitted_host_bytes()
            );
            eprintln!("ceremony_round[red line]: starved to {starved} B, run ended {end:?}");
        }
    }
}

/// The committed container's total length at the frozen ceremony geometry, from the geometry
/// alone: the fixed header plus, per compression row, one absmax-packed value row and its
/// `topk` chunk-local indices at their own bit width.
fn container_bytes(numels: &[u64]) -> u64 {
    let chunk = ceremony_profile_chunk();
    let rows: u64 = numels.iter().map(|n| n / chunk).sum();
    let topk = usize::try_from(CEREMONY_TOPK).expect("topk fits usize");
    CONTAINER_HEADER_BYTES
        + rows * daemon_vhc_det::absmax_row_bytes(topk, CEREMONY_BITS) as u64
        + daemon_vhc_det::packed_index_len(
            usize::try_from(rows * CEREMONY_TOPK).expect("count fits usize"),
            usize::try_from(chunk).expect("chunk fits usize"),
        )
        .expect("index geometry") as u64
}

/// One fold window's committed-payload section rows per peer at the frozen geometry — the working
/// set the streamed ingest actually holds, for the contrast the red line is measured against.
fn window_section_bytes() -> u64 {
    let chunk = ceremony_profile_chunk();
    let window_rows = ceremony_state_chunk_size() / (chunk * 4);
    let topk = usize::try_from(CEREMONY_TOPK).expect("topk fits usize");
    let values = window_rows * daemon_vhc_det::absmax_row_bytes(topk, CEREMONY_BITS) as u64;
    let indices = daemon_vhc_det::packed_index_len(
        usize::try_from(window_rows * CEREMONY_TOPK).expect("count fits usize"),
        usize::try_from(chunk).expect("chunk fits usize"),
    )
    .expect("index geometry") as u64;
    values + indices
}
