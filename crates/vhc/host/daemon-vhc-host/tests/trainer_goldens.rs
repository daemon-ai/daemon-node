// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The v2-native **trainer goldens** reproduction lanes (retirement plan §3): the compute@2
// trainer guest (`tiny-llama-c3`) reproduces a RECORDED, content-addressed golden bundle rather
// than the v1 parity oracle. This is the successor drift oracle that lets the v1 recording +
// `v2_parity.rs` retire later.
//
// The bundle (`tests/fixtures/trainer-goldens/`) was captured from the trainer's OWN single-peer
// barrier whole-run (the guest commits its update and ingests that same committed set): per-round
// det digests, the trainer's own committed payload bytes, the matched-init trained-theta
// trajectory, exact config literals, and the schedule. Its provenance chain (v1 oracle -> the
// c3_parity C3c equality proof at the capture commit -> these goldens) is written into the bundle
// README; the recorded digests coincide bit-for-bit with the v1 oracle's, captured from the
// autonomous v2-native lane.
//
// Two comparison classes, exactly as the C3c lane splits them:
//   * the DET LANE is an EQUALITY class — the guest's post-ingest digests (tag 4) must equal the
//     recorded golden digests bit-for-bit. The digest is a pure function of (init, ingested
//     committed payloads) via `daemon-vhc-det`, so it is backend-independent by construction; a
//     drift here is a stop-and-escalate, never a tolerance to widen.
//   * the NATIVE LANE is a TOLERANCE class — the per-round trained theta (tag 2) agrees with the
//     recorded golden theta within the `OpClass::Optimizer` band.
//
// Tiers: cpu + burn-ndarray run here; wgpu + cuda are hardware-gated feature variants (the
// v2_parity skip-if-no-hardware convention). NOTE (this commit): the compute@2 host execution
// backend is the ndarray `ComputeRunner` regardless of `EngineConfig.backend` (the driver wires
// no GPU compute runner yet — that lands with the GPU/CUDA workstream). The det digest is
// backend-independent, so the cpu and burn-ndarray tiers reproduce it identically; the wgpu/cuda
// tiers exercise the compute@2 kernels on the device through the op-journal replay seam (the same
// mechanism as `compute_replay.rs`) and check the trained theta within the tolerance class.
//
// Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

mod tolerance;

// The c3 model source, dual-compiled here (the C3b `#[path]`-include pattern) so the wgpu/cuda
// tiers record the SAME kernels the guest submits over compute@2. Only the device tiers use it.
#[cfg(any(feature = "wgpu", feature = "cuda"))]
#[path = "../../../guests/tiny-llama-c3/src/model.rs"]
#[allow(dead_code)]
mod c3_model;

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::v2::{
    start_run, start_run_migrating, MemorySink, MigrationInput, OpOutcome, OpRequest, RunEnd,
    RunIdentity, V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, VhcMessage,
};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed};

use tolerance::{tol_for, OpClass};

const ROUNDS: u64 = 2;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2;
const SEQ_LEN: u32 = 9;
const VOCAB: u32 = 64;
const PEER: [u8; 32] = [7u8; 32];

// -- guest build (the shared c3_parity pattern) --------------------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn guest(name: &str) -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// -- the recorded golden bundle (tests/fixtures/trainer-goldens) ---------------------------------

const GOLDENS_EXPECTED: &str = include_str!("fixtures/trainer-goldens/expected.json");

fn goldens_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/trainer-goldens")
}

fn goldens_expected() -> serde_json::Value {
    serde_json::from_str(GOLDENS_EXPECTED).expect("trainer-goldens expected.json")
}

fn hex_bytes(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "even hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// Load + content-verify one recorded file against its `expected.json` entry.
fn golden_file(entry: &serde_json::Value) -> Vec<u8> {
    let rel = entry["file"].as_str().expect("file");
    let bytes = std::fs::read(goldens_root().join(rel))
        .unwrap_or_else(|e| panic!("read recorded golden file {rel}: {e}"));
    assert_eq!(
        blake3_hash(&bytes).to_hex().to_string(),
        entry["blake3"].as_str().expect("blake3"),
        "recorded golden file {rel} is content-addressed — bytes must match the pin"
    );
    assert_eq!(bytes.len() as u64, entry["bytes"].as_u64().expect("bytes"));
    bytes
}

fn split_params(flat_le: &[u8], numels: &[usize]) -> Vec<Vec<f32>> {
    let flat: Vec<f32> = flat_le
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::with_capacity(numels.len());
    let mut off = 0;
    for &n in numels {
        out.push(flat[off..off + n].to_vec());
        off += n;
    }
    assert_eq!(off, flat.len(), "flat buffer matches the recorded numels");
    out
}

/// The recorded golden bundle, content-verified on load.
struct Goldens {
    numels: Vec<usize>,
    names: Vec<String>,
    /// Per-round trained theta (the tolerance comparison surface).
    trained: Vec<Vec<Vec<f32>>>,
    /// Per-round committed payload bytes (the trainer's own — fed at the barrier).
    payloads: Vec<Vec<u8>>,
    /// Per-round post-ingest det digests (the equality-class oracle).
    digests: Vec<[u8; 16]>,
    /// The guest config bytes the goldens were captured with (model + profile + init).
    cfg_bytes: Vec<u8>,
}

fn load_goldens() -> Goldens {
    let j = goldens_expected();
    let sched = &j["schedule"];
    assert_eq!(
        (
            sched["rounds"].as_u64().unwrap(),
            sched["steps_per_round"].as_u64().unwrap(),
            sched["micro_batch"].as_u64().unwrap(),
            sched["seq_len"].as_u64().unwrap(),
        ),
        (
            ROUNDS,
            u64::from(STEPS_PER_ROUND),
            u64::from(MICRO_BATCH),
            u64::from(SEQ_LEN)
        ),
        "the recorded golden schedule must match the harness constants"
    );
    let numels: Vec<usize> = j["param_numels"]
        .as_array()
        .expect("numels")
        .iter()
        .map(|n| usize::try_from(n.as_u64().expect("numel")).expect("usize"))
        .collect();
    let names: Vec<String> = j["param_names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_string())
        .collect();
    let init = split_params(&golden_file(&j["init"]), &numels);
    let trained: Vec<Vec<Vec<f32>>> = j["trained"]
        .as_array()
        .expect("trained")
        .iter()
        .map(|e| split_params(&golden_file(e), &numels))
        .collect();
    let payloads: Vec<Vec<u8>> = j["payloads"]
        .as_array()
        .expect("payloads")
        .iter()
        .map(golden_file)
        .collect();
    let digests: Vec<[u8; 16]> = j["digests"]
        .as_array()
        .expect("digests")
        .iter()
        .map(|d| {
            hex_bytes(d.as_str().expect("hex"))
                .try_into()
                .expect("digest16")
        })
        .collect();
    assert_eq!(trained.len() as u64, ROUNDS);
    assert_eq!(payloads.len() as u64, ROUNDS);
    assert_eq!(digests.len() as u64, ROUNDS);
    let cfg_bytes = guest_cfg_bytes(&j, &init);
    Goldens {
        numels,
        names,
        trained,
        payloads,
        digests,
        cfg_bytes,
    }
}

/// Rebuild the exact guest config (canonical CBOR) from the recorded literals + matched init — the
/// c3 `GuestCfg` map. The model/profile sub-maps are handed through verbatim from `expected.json`.
fn guest_cfg_bytes(j: &serde_json::Value, init: &[Vec<f32>]) -> Vec<u8> {
    let flat: Vec<f32> = init.iter().flatten().copied().collect();
    let map = Value::Map(vec![
        (Value::Text("model".into()), json_to_cbor(&j["model_cfg"])),
        (Value::Text("peer".into()), Value::Bytes(PEER.to_vec())),
        (
            Value::Text("roster".into()),
            Value::Array(vec![Value::Bytes(PEER.to_vec())]),
        ),
        (
            Value::Text("steps_per_round".into()),
            Value::from(STEPS_PER_ROUND),
        ),
        (Value::Text("micro_batch".into()), Value::from(MICRO_BATCH)),
        (Value::Text("stall_rounds_max".into()), Value::from(2u32)),
        (
            Value::Text("profile".into()),
            json_to_cbor(&j["profile_cfg"]),
        ),
        (
            Value::Text("init".into()),
            Value::serialized(&flat).expect("init"),
        ),
    ]);
    to_canonical_vec(&map).expect("guest cfg")
}

/// Convert a recorded JSON config sub-object to the CBOR value the guest deserializes. Integer
/// JSON numbers become CBOR integers (the guest's `u32` fields); non-integers become floats (its
/// `f64` fields) — the same split serde applied when the capture wrote the literals.
fn json_to_cbor(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Object(map) => Value::Map(
            map.iter()
                .map(|(k, val)| (Value::Text(k.clone()), json_to_cbor(val)))
                .collect(),
        ),
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Value::from(u)
            } else if let Some(i) = n.as_i64() {
                Value::from(i)
            } else {
                Value::from(n.as_f64().expect("finite number"))
            }
        }
        serde_json::Value::String(s) => Value::Text(s.clone()),
        other => panic!("unexpected config json node: {other:?}"),
    }
}

// -- schedule + wire shapes (the c3 module contract) ---------------------------------------------

/// Deterministic varied tokens for `(round, step)` — the recorded schedule.
fn tokens_for(round: u64, step: u32) -> Vec<u32> {
    let n = u64::from(MICRO_BATCH * SEQ_LEN);
    (0..n)
        .map(|i| {
            let x = i + 1_000 * u64::from(step) + 100_000 * round + 1;
            (x.wrapping_mul(2_654_435_761) % u64::from(VOCAB)) as u32
        })
        .collect()
}

fn batch_wrapper(round: u64, step: u32, tokens: &[u32]) -> Vec<u8> {
    let mut le = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        le.extend_from_slice(&t.to_le_bytes());
    }
    let v = Value::Array(vec![
        Value::from(0u8),
        Value::from(round),
        Value::from(step),
        Value::from(MICRO_BATCH),
        Value::from(SEQ_LEN),
        Value::Bytes(le),
    ]);
    to_canonical_vec(&v).expect("batch wrapper")
}

fn update_wrapper(round: u64, payload: &[u8]) -> Vec<u8> {
    let v = Value::Array(vec![
        Value::from(1u8),
        Value::from(round),
        Value::Bytes(PEER.to_vec()),
        Value::Bytes(payload.to_vec()),
    ]);
    to_canonical_vec(&v).expect("update wrapper")
}

fn decode_publish(frame: &[u8]) -> Option<(u64, u64, Vec<u8>)> {
    let v: Value = ciborium::de::from_reader(frame).ok()?;
    let Value::Array(parts) = v else { return None };
    let Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    let inner: Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    let Value::Array(items) = inner else {
        return None;
    };
    let uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(Value::as_integer)
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
    };
    let bytes = match items.get(2) {
        Some(Value::Bytes(b)) => b.clone(),
        _ => Vec::new(),
    };
    Some((uint(0)?, uint(1)?, bytes))
}

fn wait_published(pump: &daemon_vhc_host::v2::PumpHandle, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while pump.published().len() < n {
        service_puts(pump);
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {n} publishes (have {}); logs: {:?}",
            pump.published().len(),
            pump.logs()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    service_puts(pump);
}

/// The async-runtime seat's minimal duty here: the trainer `payload_put`s its sealed committed
/// container each round (the B1 discipline); this harness compares against RECORDED payloads, so
/// the put is acknowledged and its bytes dropped.
fn service_puts(pump: &daemon_vhc_host::v2::PumpHandle) {
    for (op, request) in pump.take_op_requests() {
        match request {
            OpRequest::PayloadPut { .. } => {
                pump.complete_op(op, OpOutcome::PutDone).expect("put done");
            }
            other => panic!("unexpected op request from the trainer guest: {other:?}"),
        }
    }
}

// -- driving the trainer -------------------------------------------------------------------------

struct Reproduced {
    /// Per-round trained theta (tag-2), split by the canonical layout.
    trained: Vec<Vec<Vec<f32>>>,
    /// Per-round post-ingest digests (tag-4).
    digests: Vec<[u8; 16]>,
}

fn theta_from_le(bytes: &[u8], numels: &[usize]) -> Vec<Vec<f32>> {
    split_params(bytes, numels)
}

fn round_open(round: u64) -> VhcMessage {
    VhcMessage::RoundOpen(RoundOpen {
        round,
        seed: Seed([round as u8; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow {
            start: 0,
            end: u64::from(STEPS_PER_ROUND * MICRO_BATCH),
        },
        deadline_unix_s: 0,
    })
}

fn round_record(round: u64, payload: &[u8]) -> VhcMessage {
    let entry = RecordEntry {
        peer: PeerId(PEER),
        hash: blake3_hash(payload),
        size: payload.len() as u64,
    };
    let set: Vec<(PeerId, Hash)> = vec![(PeerId(PEER), entry.hash)];
    VhcMessage::RoundRecord(RoundRecord {
        round,
        set: commit_set(&set).commitment(),
        drops: Vec::new(),
        next_seed: Seed([0; 32]),
        set_locator: Locator::StoreKey(String::new()),
        inline: Some(vec![entry]),
    })
}

/// Drive the trainer through the recorded schedule, feeding the recorded golden payloads at the
/// barrier (the successor of the c3_parity C3c drive). Returns the per-round trained theta + the
/// per-round post-ingest digests.
fn drive_reproduce(engine: EngineConfig, g: &Goldens, wasm: &[u8]) -> Reproduced {
    let worker = Worker::new(engine).expect("engine");
    let sel = daemon_vhc_host::select_driver(&worker, wasm, Some(blake3::hash(wasm).as_bytes()))
        .expect("trainer guest admitted");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::COMPUTE_MINOR_V2),
        "the trainer guest is a compute@2 module"
    );

    let identity = RunIdentity {
        run_id: [0x67; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(wasm).as_bytes(),
    };
    let mut run_cfg = V2RunConfig::new(identity, [0x9d; 32], g.cfg_bytes.clone(), Vec::new());
    run_cfg.compute_queue_depth = 1 << 20;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let mut seq = 0u64;
    let sender = [9u8; 32];
    let deliver = |msg: &VhcMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    for round in 0..ROUNDS {
        for h in 0..STEPS_PER_ROUND {
            pump.stage_payload(batch_wrapper(round, h, &tokens_for(round, h)), None)
                .expect("stage batch");
        }
        deliver(&round_open(round), &mut seq);
        wait_published(&pump, (round as usize) * 3 + 2); // + theta + commitment

        // Barrier: feed the RECORDED golden payload (the trainer's own committed set), then the
        // single-peer record.
        let bytes = g.payloads[round as usize].clone();
        pump.stage_payload(update_wrapper(round, &bytes), None)
            .expect("stage update");
        deliver(&round_record(round, &bytes), &mut seq);
        wait_published(&pump, (round as usize) * 3 + 3); // + digest
    }

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }

    collect(&pump, &g.numels)
}

fn collect(pump: &daemon_vhc_host::v2::PumpHandle, numels: &[usize]) -> Reproduced {
    let mut trained: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut digests: Vec<[u8; 16]> = Vec::new();
    for (_, _, frame) in pump.published() {
        let Some((tag, _round, bytes)) = decode_publish(&frame) else {
            continue;
        };
        match tag {
            2 => trained.push(theta_from_le(&bytes, numels)),
            4 => digests.push(bytes.as_slice().try_into().expect("digest16")),
            _ => {}
        }
    }
    Reproduced { trained, digests }
}

// -- assertions ----------------------------------------------------------------------------------

/// The det lane is an EQUALITY class: reproduced digests must equal the golden digests bit-for-bit.
fn assert_digests_bit_exact(g: &Goldens, r: &Reproduced, tier: &str) {
    assert_eq!(
        r.digests.len(),
        g.digests.len(),
        "{tier}: one digest per round"
    );
    for round in 0..g.digests.len() {
        assert_eq!(
            r.digests[round], g.digests[round],
            "{tier} round {round}: the det-lane digest must reproduce the golden bit-exactly \
             (stop-and-escalate on drift)"
        );
    }
    eprintln!(
        "trainer_goldens[{tier}]: det-lane digests bit-exact across {} rounds",
        g.digests.len()
    );
}

/// The native lane is a TOLERANCE class: reproduced theta within the Optimizer band of the golden.
fn assert_theta_within_band(g: &Goldens, r: &Reproduced, tier: &str) {
    assert_eq!(
        r.trained.len(),
        g.trained.len(),
        "{tier}: one theta publish per round"
    );
    let tol = tol_for(OpClass::Optimizer);
    for round in 0..g.trained.len() {
        let mut max_delta = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (want, got)) in g.trained[round]
            .iter()
            .zip(r.trained[round].iter())
            .enumerate()
        {
            assert_eq!(
                want.len(),
                got.len(),
                "{tier} round {round} {} numel",
                g.names[i]
            );
            for (j, (&wv, &gv)) in want.iter().zip(got.iter()).enumerate() {
                let diff = (wv - gv).abs();
                let bound = tol.atol + tol.rtol * wv.abs();
                max_delta = max_delta.max(diff);
                if wv != 0.0 {
                    max_rel = max_rel.max(diff / wv.abs());
                }
                assert!(
                    diff <= bound,
                    "{tier} round {round} {}[{j}]: |{wv} - {gv}| = {diff} > {bound} (Optimizer \
                     band rtol {} / atol {}) — tolerance-class breach is a stop-and-escalate",
                    g.names[i],
                    tol.rtol,
                    tol.atol
                );
            }
        }
        eprintln!(
            "trainer_goldens[{tier}] round {round}: trained-theta max |Δ| = {max_delta:.3e}, \
             max rel = {max_rel:.3e} (Optimizer band rtol {:.0e}/atol {:.0e})",
            tol.rtol, tol.atol
        );
    }
}

// -- the tiers -----------------------------------------------------------------------------------

/// The cpu tier — the tier-1 lane (default engine). Reproduces the golden det digests bit-exactly
/// and the trained theta within the Optimizer band.
#[test]
fn trainer_goldens_reproduce_cpu() {
    let g = load_goldens();
    let wasm = guest("tiny_llama_c3");
    let r = drive_reproduce(EngineConfig::default(), &g, &wasm);
    assert_digests_bit_exact(&g, &r, "cpu");
    assert_theta_within_band(&g, &r, "cpu");
}

/// The burn-ndarray tier (runs under the host suite's `--features burn-ndarray` lane). The
/// compute@2 host execution backend is the ndarray `ComputeRunner` regardless of the selected
/// `BackendKind` at this commit, so this reproduces the same trajectory as the cpu tier — the
/// named tier documents the lane and stays wired for when a GPU compute runner lands.
#[cfg(feature = "burn-ndarray")]
#[test]
fn trainer_goldens_reproduce_burn_ndarray() {
    let g = load_goldens();
    let wasm = guest("tiny_llama_c3");
    let engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::BurnNdarray,
        ..EngineConfig::default()
    };
    let r = drive_reproduce(engine, &g, &wasm);
    assert_digests_bit_exact(&g, &r, "burn-ndarray");
    assert_theta_within_band(&g, &r, "burn-ndarray");
}

// -- the straggle -> catch-up leg (ported from v2_parity's catch_up_after_straggle lane) ---------

/// Drive a STRAGGLE -> CATCH-UP schedule against the goldens: round 0 trains + commits, but its
/// record arrives while the committed payload is not yet fetchable (straggle); the payload lands;
/// `RoundOpen(1)` then makes the guest ingest round 0 (catch-up) AND train round 1 in one event
/// slice — the §5.9 ingest epilogue must run at the ingest->training boundary, so round 1 trains
/// against the post-ingest-0 base and the run ends in EXACTLY the clean run's det state.
///
/// The compute@2 trainer publishes the catch-up digest from the record handler, not the
/// `RoundOpen` handler (the guest defers only the training/export from `RoundOpen`), so the
/// caught-up round-0 digest is folded into the final state and the observable is the final
/// (round 1) digest — exactly what v2_parity's lane asserts against the clean run.
fn drive_straggle(g: &Goldens, wasm: &[u8]) -> Reproduced {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let identity = RunIdentity {
        run_id: [0x68; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 2,
        module: *blake3::hash(wasm).as_bytes(),
    };
    let mut run_cfg = V2RunConfig::new(identity, [0x9e; 32], g.cfg_bytes.clone(), Vec::new());
    run_cfg.compute_queue_depth = 1 << 20;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let mut seq = 0u64;
    let sender = [9u8; 32];
    let deliver = |msg: &VhcMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    // Round 0: train + guest-authored commit (theta + commitment publish).
    for h in 0..STEPS_PER_ROUND {
        pump.stage_payload(batch_wrapper(0, h, &tokens_for(0, h)), None)
            .expect("stage batch");
    }
    deliver(&round_open(0), &mut seq);
    wait_published(&pump, 2); // theta(0) + commitment(0)

    // The record arrives while the committed payload is NOT yet fetchable → straggle (the c3
    // guest publishes nothing on a straggle heartbeat, so there is no publish to await here).
    deliver(&round_record(0, &g.payloads[0]), &mut seq);

    // The payload lands (the archive/store caught up).
    pump.stage_payload(update_wrapper(0, &g.payloads[0]), None)
        .expect("stage update 0");

    // RoundOpen(1): ONE slice ingests round 0 (catch-up, folded into state) and trains round 1.
    for h in 0..STEPS_PER_ROUND {
        pump.stage_payload(batch_wrapper(1, h, &tokens_for(1, h)), None)
            .expect("stage batch");
    }
    deliver(&round_open(1), &mut seq);
    wait_published(&pump, 4); // theta(1) + commitment(1)

    // Barrier 1, normal path → the final (round 1) digest.
    pump.stage_payload(update_wrapper(1, &g.payloads[1]), None)
        .expect("stage update 1");
    deliver(&round_record(1, &g.payloads[1]), &mut seq);
    wait_published(&pump, 5); // digest(1)

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }
    collect(&pump, &g.numels)
}

/// The straggle -> catch-up parity pin: a run whose round-0 record straggles and catches up must
/// end in EXACTLY the clean run's final det state — the recorded golden round-1 digest — with
/// round 1's trained theta within the Optimizer band.
#[test]
fn trainer_goldens_catch_up_after_straggle_cpu() {
    let g = load_goldens();
    let wasm = guest("tiny_llama_c3");
    let r = drive_straggle(&g, &wasm);

    let final_round = (ROUNDS - 1) as usize;
    // The compute@2 trainer voices the round-0 catch-up digest from the record handler only; here
    // the catch-up happens inside `RoundOpen(1)` (whose outbound the guest defers), so round 0's
    // digest is folded into the round-1 state and the single observable digest is the FINAL one —
    // which must equal the clean run's recorded final digest.
    assert_eq!(
        r.digests.len(),
        1,
        "the catch-up run publishes exactly the folded final digest"
    );
    assert_eq!(
        r.digests[0], g.digests[final_round],
        "catch-up (ingest round 0 + train round 1 in ONE slice) must reproduce the clean run's \
         final det digest — the §5.9 epilogue runs at the ingest->training boundary"
    );

    // Both rounds' trained theta are voiced (round 0 trained normally, round 1 from the
    // post-ingest-0 base); each must sit within the Optimizer band of the clean run's golden theta.
    assert_eq!(
        r.trained.len(),
        ROUNDS as usize,
        "both rounds' theta are voiced in the straggle run"
    );
    let tol = tol_for(OpClass::Optimizer);
    for round in 0..ROUNDS as usize {
        for (i, (want, got)) in g.trained[round]
            .iter()
            .zip(r.trained[round].iter())
            .enumerate()
        {
            assert_eq!(
                want.len(),
                got.len(),
                "catch-up round {round} {} numel",
                g.names[i]
            );
            for (j, (&wv, &gv)) in want.iter().zip(got.iter()).enumerate() {
                let bound = tol.atol + tol.rtol * wv.abs();
                assert!(
                    (wv - gv).abs() <= bound,
                    "catch-up round {round} {}[{j}]: |{wv} - {gv}| > {bound} (Optimizer band) — \
                     stop-and-escalate",
                    g.names[i]
                );
            }
        }
    }
    eprintln!("trainer_goldens[straggle]: catch-up reproduced the clean final digest + theta band");
}

// -- the checkpoint/migration continuity pin (ABI §10.2 over the trainer at LLaMA scale) ----------

/// The round's tag-3 commitment hash from the published frames, if voiced yet.
fn commitment_hash(pump: &daemon_vhc_host::v2::PumpHandle, round: u64) -> Option<[u8; 32]> {
    for (_, _, frame) in pump.published() {
        if let Some((3, r, bytes)) = decode_publish(&frame) {
            if r == round {
                return Some(bytes.as_slice().try_into().expect("hash32"));
            }
        }
    }
    None
}

/// Typed checkpoint/restore at LLaMA scale, pinned against the frozen goldens: round 0 runs on
/// instance 1, a `Quiesce{Upgrade}` drain snapshots the typed state-manifest (`master` + `ef` +
/// `adamw_m`/`adamw_v` sections — the moments walk device→host through the async export seam),
/// `da_migrate` restores it into a FRESH instance, and round 1 on the new instance must produce
/// (a) the round-1 commitment hash of the recorded golden payload — which pins the replica-local
/// restore (error feedback + moments), because the committed payload is a function of them and
/// the digest alone would not see their loss — and (b) the recorded golden round-1 det digest
/// bit-exactly.
#[test]
fn trainer_checkpoint_restores_across_migration_with_digest_continuity() {
    let g = load_goldens();
    let wasm = guest("tiny_llama_c3");
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let module_hash = *blake3::hash(&wasm).as_bytes();

    // -- instance 1: round 0 exactly as the reproduce lane drives it ------------------------------
    let identity = RunIdentity {
        run_id: [0x69; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: module_hash,
    };
    let mut run_cfg = V2RunConfig::new(
        identity.clone(),
        [0x9f; 32],
        g.cfg_bytes.clone(),
        Vec::new(),
    );
    run_cfg.compute_queue_depth = 1 << 20;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let mut seq = 0u64;
    let sender = [9u8; 32];
    let deliver = |pump: &daemon_vhc_host::v2::PumpHandle, msg: &VhcMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    for h in 0..STEPS_PER_ROUND {
        pump.stage_payload(batch_wrapper(0, h, &tokens_for(0, h)), None)
            .expect("stage batch");
    }
    deliver(&pump, &round_open(0), &mut seq);
    wait_published(&pump, 2); // theta(0) + commitment(0)
    pump.stage_payload(update_wrapper(0, &g.payloads[0]), None)
        .expect("stage update 0");
    deliver(&pump, &round_record(0, &g.payloads[0]), &mut seq);
    wait_published(&pump, 3); // digest(0)

    // -- quiesce: the §10.2 producing protocol snapshots the typed manifest -----------------------
    pump.quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, 60_000)
        .expect("quiesce delivery");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(code) if code == daemon_vhc_abi::OUTCOME_QUIESCE_READY => {}
        other => panic!("expected QuiesceReady, got {other:?}"),
    }
    let capture = pump
        .snapshot_capture()
        .expect("the drain accepted a snapshot");
    assert_eq!(
        capture
            .sections
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["master", "ef", "adamw_m", "adamw_v"],
        "the typed manifest declares the canonical masters + the replica-local continuity state \
         (error feedback and AdamW moments)"
    );
    let total: usize = g.numels.iter().sum();
    for (name, bytes) in &capture.sections {
        assert_eq!(
            bytes.len(),
            total * 4,
            "section `{name}` is the flat f32-le canonical layout"
        );
    }
    let round0_digest: [u8; 16] = {
        let published = pump.published();
        published
            .iter()
            .find_map(|(_, _, frame)| match decode_publish(frame) {
                Some((4, 0, bytes)) => Some(bytes.as_slice().try_into().expect("digest16")),
                _ => None,
            })
            .expect("round-0 digest voiced before the drain")
    };
    assert_eq!(
        round0_digest, g.digests[0],
        "pre-snapshot round-0 digest matches the golden (the baseline for continuity)"
    );

    // -- instance 2: da_init -> da_migrate(restore) -> round 1 ------------------------------------
    let identity2 = RunIdentity {
        instance: 2, // never-reused (§8.1): the migrated incarnation
        ..identity
    };
    let mut run_cfg2 = V2RunConfig::new(identity2, [0xa0; 32], g.cfg_bytes.clone(), Vec::new());
    run_cfg2.compute_queue_depth = 1 << 20;
    let sink2 = Arc::new(Mutex::new(MemorySink::new()));
    let run2 = start_run_migrating(
        &worker,
        &wasm,
        run_cfg2,
        Box::new(sink2),
        Some(MigrationInput {
            capture,
            restore: true,
            migrate_fuel: None,
        }),
    )
    .expect("start migrating");
    let pump2 = run2.pump.clone();
    let mut seq2 = 0u64;

    for h in 0..STEPS_PER_ROUND {
        pump2
            .stage_payload(batch_wrapper(1, h, &tokens_for(1, h)), None)
            .expect("stage batch");
    }
    deliver(&pump2, &round_open(1), &mut seq2);
    wait_published(&pump2, 2); // theta(1) + commitment(1)

    // The ef-restore pin: round 1's committed payload is a function of (theta, round_base, ef);
    // master/round_base continuity alone would still digest-match after ingesting the RECORDED
    // payload, so the commitment hash is the assertion that catches a dropped/zeroed ef.
    assert_eq!(
        commitment_hash(&pump2, 1).expect("round-1 commitment voiced"),
        blake3_hash(&g.payloads[1]).0,
        "the restored instance's round-1 committed payload must hash-match the recorded golden \
         (error-feedback continuity across the migration)"
    );

    pump2
        .stage_payload(update_wrapper(1, &g.payloads[1]), None)
        .expect("stage update 1");
    deliver(&pump2, &round_record(1, &g.payloads[1]), &mut seq2);
    wait_published(&pump2, 3); // digest(1)

    pump2
        .stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run2.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }

    let final_digest: [u8; 16] = pump2
        .published()
        .iter()
        .find_map(|(_, _, frame)| match decode_publish(frame) {
            Some((4, 1, bytes)) => Some(bytes.as_slice().try_into().expect("digest16")),
            _ => None,
        })
        .expect("round-1 digest voiced");
    assert_eq!(
        final_digest, g.digests[1],
        "round 1 on the migrated instance must reproduce the recorded golden digest bit-exactly \
         — checkpoint/restore continuity at LLaMA scale"
    );
    eprintln!(
        "trainer_goldens[checkpoint]: quiesce -> typed manifest (master + ef + adamw moments) -> \
         da_migrate -> round 1 reproduced the golden commitment hash + det digest"
    );
}

// -- wgpu + cuda tiers (genuine compute@2 device coverage via op-journal replay) ------------------
//
// The compute@2 host execution backend is the ndarray `ComputeRunner` regardless of
// `EngineConfig.backend` at this commit — driving the guest under a GPU `BackendKind` would still
// execute on ndarray, so it would NOT exercise the device. The det digest is backend-independent
// (it is a pure function of the ingested committed payloads via `daemon-vhc-det`), so the ONLY
// backend-sensitive output is the trained theta. These tiers exercise the compute@2 KERNELS on the
// real device through the op-journal replay seam (the `compute_replay.rs` mechanism): the c3
// model's round-0 forward+backward+AdamW op stream is recorded over a `burn-router` recording
// client, then re-executed against the production `ComputeRunner<Device>` on the GPU, and the
// exported theta is checked against the recorded golden within the native (tolerance) class. This
// is the lane that retires the plan's "first GPU run may expose a compute@2 det-lane divergence"
// risk — it runs the trainer's actual kernels on the device.
//
// Feature-gated + self-skipping without hardware (the v2_parity convention). wgpu is attempted on
// this host's Vulkan/RADV; cuda is gated for the remote CUDA lane (never attempted locally).
#[cfg(any(feature = "wgpu", feature = "cuda"))]
mod gpu {
    use std::cell::RefCell;

    use burn::backend::Autodiff;
    use burn_backend::{
        DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape, TensorData as BackendTensorData,
    };
    use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
    use burn_ndarray::NdArrayDevice;
    use burn_router::{
        BackendRouter, MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient,
    };
    use burn_std::future::DynFut;
    use daemon_vhc_host::{ComputeRunner, HostReal};

    use super::c3_model::{C3Llama, ModelCfg};
    use super::{
        goldens_expected, load_goldens, tokens_for, tolerance, Goldens, MICRO_BATCH, SEQ_LEN,
        STEPS_PER_ROUND,
    };
    use tolerance::{tol_for, OpClass};

    // -- the recording burn-router client (captures the op stream, never executes) ----------------
    // Verbatim shape from tests/compute_replay.rs: a thread-local recorder + a router client that
    // enqueues `OperationIr`/imports and never reads back.

    #[derive(Clone)]
    enum ComputeStep {
        Import { id: u64, data: Vec<u8> },
        Op(Vec<u8>),
    }

    thread_local! {
        static REC: RefCell<(u64, Vec<ComputeStep>)> = const { RefCell::new((1, Vec::new())) };
    }

    fn rec_reset() {
        REC.with(|r| *r.borrow_mut() = (1, Vec::new()));
    }
    fn rec_mint() -> u64 {
        REC.with(|r| {
            let mut b = r.borrow_mut();
            let id = b.0;
            b.0 += 1;
            id
        })
    }
    fn rec_push(step: ComputeStep) {
        REC.with(|r| r.borrow_mut().1.push(step));
    }
    fn rec_take() -> Vec<ComputeStep> {
        REC.with(|r| std::mem::take(&mut r.borrow_mut().1))
    }

    fn ser<T: serde::Serialize>(v: &T) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(v, &mut buf).expect("IR/TensorData encodes");
        buf
    }

    #[derive(Clone)]
    struct RecClient;

    impl RunnerClient for RecClient {
        type Device = NdArrayDevice;

        fn register_op(&self, op: OperationIr) {
            rec_push(ComputeStep::Op(ser(&op)));
        }
        fn read_tensor_async(
            &self,
            _tensor: TensorIr,
        ) -> DynFut<Result<BackendTensorData, ExecutionError>> {
            panic!("the recording client never reads back — capture the TensorIr and replay");
        }
        fn sync(&self) -> Result<(), ExecutionError> {
            Ok(())
        }
        fn create_empty_handle(&self) -> TensorId {
            TensorId::new(rec_mint())
        }
        fn register_tensor_data(&self, data: BackendTensorData) -> RouterTensor<Self> {
            let id = rec_mint();
            let shape = data.shape.clone();
            let dtype = data.dtype;
            rec_push(ComputeStep::Import {
                id,
                data: ser(&data),
            });
            RouterTensor::new(TensorId::new(id), shape, dtype, self.clone())
        }
        fn device(&self) -> Self::Device {
            NdArrayDevice::Cpu
        }
        fn seed(&self, _seed: u64) {}
        fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
            DTypeUsage::general()
        }
    }

    struct NoBridge;
    impl MultiBackendBridge for NoBridge {
        type TensorHandle = ();
        type Device = NdArrayDevice;
        fn change_backend_float(_: (), _: Shape, _: &NdArrayDevice) {
            unreachable!("single-device recording never transfers backends")
        }
        fn change_backend_int(_: (), _: Shape, _: &NdArrayDevice) {
            unreachable!("single-device recording never transfers backends")
        }
        fn change_backend_bool(_: (), _: Shape, _: &NdArrayDevice) {
            unreachable!("single-device recording never transfers backends")
        }
    }

    #[derive(Clone)]
    struct RecChannel;
    impl RunnerChannel for RecChannel {
        type Device = NdArrayDevice;
        type Bridge = NoBridge;
        type Client = RecClient;
        type FloatElem = f32;
        type IntElem = i64;
        type BoolElem = bool;

        fn name(_device: &NdArrayDevice) -> String {
            "compute@2-recording".to_string()
        }
        fn init_client(_device: &NdArrayDevice) -> RecClient {
            RecClient
        }
        fn get_tensor_handle(_tensor: &TensorIr, _client: &RecClient) {
            unreachable!("single-device recording never extracts a cross-backend handle")
        }
        fn register_tensor(
            _client: &RecClient,
            _handle: (),
            _shape: Shape,
            _dtype: DType,
        ) -> RouterTensor<RecClient> {
            unreachable!("single-device recording never registers a cross-backend handle")
        }
    }

    type RecBackend = BackendRouter<RecChannel>;

    fn model_cfg() -> ModelCfg {
        serde_json::from_value(goldens_expected()["model_cfg"].clone()).expect("model cfg")
    }

    /// Record the c3 model's round-0 forward+backward+AdamW op stream from the matched init, and
    /// the exported per-param `TensorIr` handles — the exact `compute@2` op stream the guest
    /// submits (C3b: the guest lowers this source bit-exactly).
    fn record_round0(cfg: &ModelCfg, init: &[Vec<f32>]) -> (Vec<ComputeStep>, Vec<Vec<u8>>) {
        rec_reset();
        let device = NdArrayDevice::Cpu;
        let mut model = C3Llama::<Autodiff<RecBackend>>::from_flat(cfg.clone(), device, init);
        for h in 0..STEPS_PER_ROUND {
            let grads = model.forward_backward(
                &tokens_for(0, h),
                MICRO_BATCH as usize,
                SEQ_LEN as usize,
                1.0,
            );
            model.adamw_apply(&grads, h);
        }
        // Capture the export handles, then drain the journal BEFORE the model drops — so no `Drop`
        // ops enter the journal and every export id stays live at replay.
        let export_irs: Vec<Vec<u8>> = model
            .export_tensors()
            .into_iter()
            .map(|t| ser(&t.into_primitive().tensor().into_ir()))
            .collect();
        let journal = rec_take();
        std::mem::forget(model);
        (journal, export_irs)
    }

    /// Replay a recorded op journal against the production `ComputeRunner<B>` on `device`, reading
    /// back each exported param's theta (kernels re-executed on `B`).
    fn replay_theta_on<B: BackendIr>(
        journal: &[ComputeStep],
        export_irs: &[Vec<u8>],
        device: B::Device,
    ) -> Vec<Vec<f32>> {
        let mut runner = ComputeRunner::<B>::new(device);
        for step in journal {
            match step {
                ComputeStep::Import { id, data } => {
                    runner.import_tensor(*id, data).expect("import replays");
                }
                ComputeStep::Op(op) => runner.submit_op(op).expect("op replays"),
            }
        }
        runner.fence().expect("fence drains clean");
        export_irs
            .iter()
            .map(|ir| {
                let cbor = runner.read_tensor(ir).expect("export reads back");
                let data: BackendTensorData =
                    ciborium::from_reader(&cbor[..]).expect("TensorData decodes");
                data.convert::<f32>().to_vec::<f32>().expect("f32 values")
            })
            .collect()
    }

    fn golden_init(g: &Goldens) -> Vec<Vec<f32>> {
        super::split_params(&super::golden_file(&goldens_expected()["init"]), &g.numels)
    }

    /// Assert the replayed round-0 theta reproduces the recorded golden within the given class.
    fn assert_theta(
        golden: &[Vec<f32>],
        got: &[Vec<f32>],
        names: &[String],
        class: OpClass,
        tier: &str,
    ) {
        let tol = tol_for(class);
        let exact = tol.rtol == 0.0 && tol.atol == 0.0;
        let mut max_delta = 0.0f32;
        for (i, (want, g)) in golden.iter().zip(got.iter()).enumerate() {
            assert_eq!(want.len(), g.len(), "{tier} {} numel", names[i]);
            for (j, (&wv, &gv)) in want.iter().zip(g.iter()).enumerate() {
                let diff = (wv - gv).abs();
                max_delta = max_delta.max(diff);
                if exact {
                    assert!(
                        wv.to_bits() == gv.to_bits(),
                        "{tier} {}[{j}]: recording must replay bit-exactly on ndarray, got {gv} \
                         want {wv}",
                        names[i]
                    );
                } else {
                    let bound = tol.atol + tol.rtol * wv.abs();
                    assert!(
                        diff <= bound,
                        "{tier} {}[{j}]: |{wv} - {gv}| = {diff} > {bound} ({class:?} band) — a \
                         compute@2 device divergence is a stop-and-escalate finding",
                        names[i]
                    );
                }
            }
        }
        eprintln!("trainer_goldens[{tier}]: round-0 theta reproduced ({class:?}, max |Δ| = {max_delta:.3e})");
    }

    /// Record once, then: prove the journal replays bit-exactly on ndarray (the recording is
    /// faithful to the golden), then reproduce round-0 theta on the device within tolerance.
    fn run_device_tier<B: BackendIr>(device: B::Device, tier: &str) {
        let g = load_goldens();
        let cfg = model_cfg();
        let init = golden_init(&g);
        let (journal, export_irs) = record_round0(&cfg, &init);
        assert!(
            journal
                .iter()
                .any(|s| matches!(s, ComputeStep::Import { .. }))
                && journal.iter().any(|s| matches!(s, ComputeStep::Op(_))),
            "the journal captured imports + ops"
        );
        // Faithfulness: ndarray replay must equal the recorded golden round-0 theta bit-for-bit.
        let nd = replay_theta_on::<HostReal>(&journal, &export_irs, NdArrayDevice::Cpu);
        assert_theta(
            &g.trained[0],
            &nd,
            &g.names,
            OpClass::Exact,
            "ndarray-selfcheck",
        );
        // The device tier: reproduce round-0 theta within the native (Optimizer) tolerance class.
        let dev = replay_theta_on::<B>(&journal, &export_irs, device);
        assert_theta(&g.trained[0], &dev, &g.names, OpClass::Optimizer, tier);
    }

    /// The wgpu tier — hardware-gated (the v2_parity skip convention; `.#vulkan` / a GPU runner
    /// exercises it). The compute@2 kernels re-execute on wgpu and reproduce the recorded golden
    /// round-0 theta within the native tolerance class.
    #[cfg(feature = "wgpu")]
    #[test]
    fn trainer_goldens_reproduce_wgpu() {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        if !daemon_vhc_host::wgpu_adapter_available() {
            eprintln!("SKIP trainer_goldens(wgpu): no usable wgpu adapter on this runner");
            return;
        }
        run_device_tier::<Wgpu<f32, i32>>(WgpuDevice::default(), "wgpu");
    }

    /// The cuda tier — hardware-gated like the wgpu tier (the remote `.#cuda-train` lane runs it;
    /// this host has no NVIDIA device, so it self-skips — never attempted locally).
    #[cfg(feature = "cuda")]
    #[test]
    fn trainer_goldens_reproduce_cuda() {
        if !daemon_vhc_host::cuda_adapter_available() {
            eprintln!("SKIP trainer_goldens(cuda): no usable CUDA device on this runner");
            return;
        }
        run_device_tier::<burn::backend::Cuda>(Default::default(), "cuda");
    }
}
