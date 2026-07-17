// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The C3 models-exodus acceptance (refactor §7): the **re-authored** `tiny-llama-c3` guest — a
// real Burn model over `Autodiff<HostBackend>` + the C3a `sdk-profiles` det-lane — driven through
// a barrier whole-run against the **v1 digest oracle**. Since the Phase-E sunset the oracle is
// the RECORDED `tests/fixtures/v1-parity-oracle/` bundle (captured pre-sunset from the live v1
// driver over the frozen `tiny_llama.wasm`; decisions D5). Two legs, deliberately split:
//
// - **C3b — bit-exact lowering** (`c3_guest_training_lowers_bit_exact_vs_native`): the guest's
//   trained θ (published per round) equals a native `Autodiff<NdArray>` run of the SAME
//   dual-compiled model source (`#[path]`-included from the guest crate) on the same schedule —
//   isolating lowering correctness from model-vs-model numerics. Bit-exactness holds because
//   both sides execute identical burn-ndarray kernels; only the wasm32 + CBOR + driver path
//   differs (the toy-mlp proof, at LLaMA scale).
//
// - **C3c — tolerance-class parity vs v1** (`c3_reauthored_tiny_llama_parity_vs_v1_oracle_cpu`):
//   the v1 oracle and the c3 guest run the same 2-round barrier schedule from matched init over
//   identical batches, the guest ingesting the v1 committed payloads (single-peer roster, the
//   v2_parity harness shape). The **det lane must stay bit-exact**: the guest's in-guest digests
//   (daemon-vhc-det over guest-held state) must EQUAL the v1 digests. The **native lane is the
//   tolerance class**: per-round trained θ agrees within the existing `OpClass::Optimizer` band
//   (`tests/tolerance`), per-band numbers reported. A band breach here is a stop-and-escalate,
//   never a tolerance to widen.
//
// Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed file-wide.
#![cfg(feature = "burn-ndarray")]
#![allow(clippy::disallowed_methods)]

mod tolerance;

#[path = "../../../guests/tiny-llama-c3/src/model.rs"]
#[allow(dead_code)]
mod c3_model;

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::v2::{
    start_run, MemorySink, OpOutcome, OpRequest, RunEnd, RunIdentity, V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, SwarmMessage,
};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_profiles::{IngestParam, SparseLoco, SparseLocoCfg};

use c3_model::{C3Llama, ModelCfg};
use tolerance::{tol_for, OpClass};

const ROUNDS: u64 = 2;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2;
const SEQ_LEN: u32 = 9;
const VOCAB: u32 = 64;
const PEER: [u8; 32] = [7u8; 32];

type NativeAd = burn::backend::Autodiff<burn::backend::NdArray>;

// -- guest build (the shared Once pattern) -------------------------------------------------------

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

// -- the shared schedule --------------------------------------------------------------------------

/// The model/profile config literals the run reconstructs — recorded in the frozen oracle bundle
/// (`expected.json` `c3_parity.model_cfg` / `.profile_cfg`, written from the live v1
/// `TinyLlamaCfg` at capture, so nothing here was transcribed by hand). The frozen-pin parity
/// shape: 1 layer, seq 9, `sparse_loco` `chunk 64 / topk 8 / clip false`.
fn oracle_expected() -> serde_json::Value {
    serde_json::from_str(ORACLE_EXPECTED).expect("expected.json")
}

/// The c3 model config, reconstructed from the recorded oracle's config literals.
fn c3_cfg() -> ModelCfg {
    let j = oracle_expected();
    let m = &j["c3_parity"]["model_cfg"];
    let u = |k: &str| u32::try_from(m[k].as_u64().unwrap_or_else(|| panic!("{k}"))).expect("u32");
    let f = |k: &str| m[k].as_f64().unwrap_or_else(|| panic!("{k}"));
    ModelCfg {
        d_model: u("d_model"),
        n_layers: u("n_layers"),
        n_heads: u("n_heads"),
        head_dim: u("head_dim"),
        vocab: u("vocab"),
        seq_len: u("seq_len"),
        ffn_mult: u("ffn_mult"),
        rope_theta: f("rope_theta"),
        rmsnorm_eps: f("rmsnorm_eps"),
        lr: f("lr"),
        beta1: f("beta1"),
        beta2: f("beta2"),
        adam_eps: f("adam_eps"),
        wd: f("wd"),
    }
}

fn profile_cfg() -> SparseLocoCfg {
    let j = oracle_expected();
    let p = &j["c3_parity"]["profile_cfg"];
    let u = |k: &str| u32::try_from(p[k].as_u64().unwrap_or_else(|| panic!("{k}"))).expect("u32");
    SparseLocoCfg {
        h: u("h"),
        ef_decay: p["ef_decay"].as_f64().expect("ef_decay"),
        chunk: u("chunk"),
        topk: u("topk"),
        bits: u("bits"),
        outer_alpha: p["outer_alpha"].as_f64().expect("outer_alpha"),
        clip: p["clip"].as_bool().expect("clip"),
    }
}

/// Deterministic varied tokens for `(round, step)` — identical on every path.
fn tokens_for(round: u64, step: u32) -> Vec<u32> {
    let n = (MICRO_BATCH * SEQ_LEN) as u64;
    (0..n)
        .map(|i| {
            let x = i + 1_000 * u64::from(step) + 100_000 * round + 1;
            (x.wrapping_mul(2_654_435_761) % u64::from(VOCAB)) as u32
        })
        .collect()
}

// -- the v1 digest oracle (RECORDED — tests/fixtures/v1-parity-oracle; decisions D5) --------------
//
// Before the Phase-E sunset this oracle ran the frozen v1 module LIVE via `Instance` on every
// test execution; the sunset deleted the v1 driver, so the oracle values were frozen FIRST as
// the content-addressed `v1-parity-oracle` bundle (captured at pre-sunset commit `1390f0b7` by
// the bundled capture crate). "The v1 digests remain the oracle" — now as a recording.

const ORACLE_EXPECTED: &str = include_str!("fixtures/v1-parity-oracle/expected.json");

struct V1Run {
    /// The matched init (canonical registration order).
    init: Vec<Vec<f32>>,
    /// Per-round trained θ (post-inner-steps, pre-ingest) — the tolerance comparison surface.
    trained: Vec<Vec<Vec<f32>>>,
    /// Per-round committed payload bytes (the wire containers the guest ingests).
    payloads: Vec<Vec<u8>>,
    /// Per-round post-ingest digests (the v1 digest oracle).
    digests: Vec<[u8; 16]>,
}

fn oracle_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1-parity-oracle")
}

fn hex_bytes(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "even hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// Load + content-verify one recorded fixture file against its `expected.json` entry.
fn oracle_file(entry: &serde_json::Value) -> Vec<u8> {
    let rel = entry["file"].as_str().expect("file");
    let bytes = std::fs::read(oracle_root().join(rel))
        .unwrap_or_else(|e| panic!("read recorded oracle file {rel}: {e}"));
    assert_eq!(
        blake3_hash(&bytes).to_hex().to_string(),
        entry["blake3"].as_str().expect("blake3"),
        "recorded oracle file {rel} is content-addressed — bytes must match the pin"
    );
    assert_eq!(bytes.len() as u64, entry["bytes"].as_u64().expect("bytes"));
    bytes
}

/// Split a flat little-endian f32 buffer into the recorded per-param layout.
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

fn v1_oracle() -> V1Run {
    let j = oracle_expected();
    let c3 = &j["c3_parity"];
    let sched = &c3["schedule"];
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
        "the recorded oracle's schedule must match the harness constants"
    );
    let numels: Vec<usize> = c3["param_numels"]
        .as_array()
        .expect("numels")
        .iter()
        .map(|n| usize::try_from(n.as_u64().expect("numel")).expect("usize"))
        .collect();
    // The dual-compiled model source must still agree with the recorded layout — a drifted
    // model config would silently misalign every comparison below.
    assert_eq!(
        numels,
        c3_cfg().param_numels(),
        "recorded param layout ≡ the dual-compiled model's layout"
    );
    let init = split_params(&oracle_file(&c3["init"]), &numels);
    let trained: Vec<Vec<Vec<f32>>> = c3["trained"]
        .as_array()
        .expect("trained")
        .iter()
        .map(|e| split_params(&oracle_file(e), &numels))
        .collect();
    let payloads: Vec<Vec<u8>> = c3["payloads"]
        .as_array()
        .expect("payloads")
        .iter()
        .map(oracle_file)
        .collect();
    let digests: Vec<[u8; 16]> = c3["digests"]
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
    V1Run {
        init,
        trained,
        payloads,
        digests,
    }
}

// -- the c3 guest run ------------------------------------------------------------------------------

/// The c3 guest's config map (canonical CBOR — the guest's `GuestCfg`).
fn c3_config(init: &[Vec<f32>]) -> Vec<u8> {
    let flat: Vec<f32> = init.iter().flatten().copied().collect();
    let map = Value::Map(vec![
        (
            Value::Text("model".into()),
            Value::serialized(&c3_cfg()).expect("model cfg"),
        ),
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
            Value::serialized(&profile_cfg()).expect("profile cfg"),
        ),
        (
            Value::Text("init".into()),
            Value::serialized(&flat).expect("init"),
        ),
    ]);
    to_canonical_vec(&map).expect("guest cfg")
}

/// A staged batch wrapper: `[0, round, step, sequences, seq_len, tokens_le]`.
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

/// A staged committed-payload wrapper: `[1, round, peer32, payload]`.
fn update_wrapper(round: u64, payload: &[u8]) -> Vec<u8> {
    let v = Value::Array(vec![
        Value::from(1u8),
        Value::from(round),
        Value::Bytes(PEER.to_vec()),
        Value::Bytes(payload.to_vec()),
    ]);
    to_canonical_vec(&v).expect("update wrapper")
}

/// One published frame's `[tag, round, bytes]` decoded.
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

struct C3Run {
    /// Per-round trained θ (from the tag-2 publishes), flat LE bytes decoded to per-param vecs.
    trained: Vec<Vec<Vec<f32>>>,
    /// Per-round in-guest digests (tag-4 publishes).
    digests: Vec<[u8; 16]>,
}

/// Drive the c3 guest through the barrier schedule, feeding the v1 oracle's committed payloads.
fn c3_run(init: &[Vec<f32>], v1_payloads: &[Vec<u8>]) -> C3Run {
    let wasm = guest("tiny_llama_c3");
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = daemon_vhc_host::select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("c3 guest admitted");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::COMPUTE_MINOR_V2),
        "the re-authored model is a compute@2 module"
    );

    let identity = RunIdentity {
        run_id: [0xC3; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let mut run_cfg = V2RunConfig::new(identity, [0x83; 32], c3_config(init), Vec::new());
    // A real transformer's per-round op stream exceeds the tiny default queue depth; the guest
    // also fences per inner step (§3.3 depth reclaim) — belt and braces.
    run_cfg.compute_queue_depth = 1 << 20;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let mut seq = 0u64;
    let sender = [9u8; 32];

    let deliver = |msg: &SwarmMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
        *seq += 1;
    };
    // The async-runtime seat's minimal duty: the trainer `payload_put`s its sealed committed
    // container each round (B1 discipline); this lane feeds the v1 oracle's payloads, so the put
    // is acknowledged and its bytes dropped.
    let service_puts = || {
        for (op, request) in pump.take_op_requests() {
            match request {
                OpRequest::PayloadPut { .. } => {
                    pump.complete_op(op, OpOutcome::PutDone).expect("put done");
                }
                other => panic!("unexpected op request from the c3 guest: {other:?}"),
            }
        }
    };
    let wait_published = |n: usize| {
        let deadline = Instant::now() + Duration::from_secs(120);
        while pump.published().len() < n {
            service_puts();
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {n} publishes (have {}); logs: {:?}",
                pump.published().len(),
                pump.logs()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        service_puts();
    };

    for round in 0..ROUNDS {
        // The round's batches, staged in training order (kind-0 bytes).
        for h in 0..STEPS_PER_ROUND {
            pump.stage_payload(batch_wrapper(round, h, &tokens_for(round, h)), None)
                .expect("stage batch");
        }
        // RoundOpen → train; the guest then walks fence → export → publishes θ (tag 2) and its
        // commitment voice (tag 3).
        let ro = SwarmMessage::RoundOpen(RoundOpen {
            round,
            seed: Seed([round as u8; 32]),
            roster_digest: Hash([0; 32]),
            batch: BatchWindow {
                start: 0,
                end: u64::from(STEPS_PER_ROUND * MICRO_BATCH),
            },
            deadline_unix_s: 0,
        });
        deliver(&ro, &mut seq);
        wait_published((round as usize) * 3 + 2); // + θ + commitment

        // Barrier: stage the v1-committed payload, then the record (single-peer set).
        let bytes = v1_payloads[round as usize].clone();
        let entry = RecordEntry {
            peer: PeerId(PEER),
            hash: blake3_hash(&bytes),
            size: bytes.len() as u64,
        };
        pump.stage_payload(update_wrapper(round, &bytes), None)
            .expect("stage update");
        let set: Vec<(PeerId, Hash)> = vec![(PeerId(PEER), entry.hash)];
        let rr = SwarmMessage::RoundRecord(RoundRecord {
            round,
            set: commit_set(&set).commitment(),
            drops: Vec::new(),
            next_seed: Seed([0; 32]),
            set_locator: Locator::StoreKey(String::new()),
            inline: Some(vec![entry]),
        });
        deliver(&rr, &mut seq);
        wait_published((round as usize) * 3 + 3); // + digest
    }

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }

    // Decode the publishes: tag 2 = trained θ (flat LE), tag 4 = digest.
    let numels = c3_cfg().param_numels();
    let mut trained: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut digests: Vec<[u8; 16]> = Vec::new();
    for (_, _, frame) in pump.published() {
        let Some((tag, _round, bytes)) = decode_publish(&frame) else {
            continue;
        };
        match tag {
            2 => {
                let flat: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut per = Vec::new();
                let mut off = 0;
                for &n in &numels {
                    per.push(flat[off..off + n].to_vec());
                    off += n;
                }
                assert_eq!(off, flat.len(), "θ publish matches the canonical layout");
                trained.push(per);
            }
            4 => {
                let d: [u8; 16] = bytes.as_slice().try_into().expect("digest16");
                digests.push(d);
            }
            _ => {}
        }
    }
    assert_eq!(trained.len(), ROUNDS as usize, "one θ publish per round");
    assert_eq!(digests.len(), ROUNDS as usize, "one digest per round");
    C3Run { trained, digests }
}

// -- C3b: bit-exact lowering vs the native run of the SAME model source ---------------------------

#[test]
fn c3_guest_training_lowers_bit_exact_vs_native() {
    let v1 = v1_oracle();
    let c3 = c3_run(&v1.init, &v1.payloads);

    // The native oracle: the SAME dual-compiled model source over Autodiff<NdArray>, the same
    // schedule, the same det-lane transitions (the C3a profile natively).
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let mut model = C3Llama::<NativeAd>::from_flat(c3_cfg(), device, &v1.init);
    let numels = c3_cfg().param_numels();
    let mut profile = SparseLoco::new(profile_cfg(), &numels);
    let mut master = v1.init.clone();
    let mut round_base = v1.init.clone();

    for round in 0..ROUNDS {
        for h in 0..STEPS_PER_ROUND {
            let grads = model.forward_backward(
                &tokens_for(round, h),
                MICRO_BATCH as usize,
                SEQ_LEN as usize,
                1.0,
            );
            model.adamw_apply(&grads, h);
        }
        let native_theta: Vec<Vec<f32>> = model
            .export_tensors()
            .into_iter()
            .map(|t| t.into_data().to_vec::<f32>().expect("f32"))
            .collect();
        let guest_theta = &c3.trained[round as usize];
        for (i, (n, g)) in native_theta.iter().zip(guest_theta.iter()).enumerate() {
            assert_eq!(n.len(), g.len(), "round {round} param {i} numel");
            for (j, (nv, gv)) in n.iter().zip(g.iter()).enumerate() {
                assert_eq!(
                    nv.to_bits(),
                    gv.to_bits(),
                    "round {round} param {i}[{j}]: native {nv} vs guest {gv} — the compute@2 \
                     lowering must be bit-exact (same kernels, same order)"
                );
            }
        }

        // The det-lane transition (the same C3a profile, natively) to continue the trajectory.
        let payloads = vec![
            daemon_vhc_sdk_profiles::decode_payload(&v1.payloads[round as usize])
                .expect("v1 payload decodes under the C3 Section wire"),
        ];
        let mut params: Vec<IngestParam<'_>> = master
            .iter_mut()
            .zip(round_base.iter())
            .map(|(m, b)| IngestParam {
                master: m,
                round_base: b,
            })
            .collect();
        profile.ingest(&mut params, &payloads).expect("ingest");
        model.set_params_from_flat(&master);
        round_base = master.clone();
    }
}

// -- C3c: the tolerance-class parity lane vs the v1 digest oracle ---------------------------------

#[test]
fn c3_reauthored_tiny_llama_parity_vs_v1_oracle_cpu() {
    let v1 = v1_oracle();
    let c3 = c3_run(&v1.init, &v1.payloads);

    // Leg 1 — the det lane is an EQUALITY class (architecture §3.6): the guest's in-guest digests
    // (daemon-vhc-det over guest-held canonical state) equal the v1 oracle's digests bit-for-bit
    // — "the v1 digests remain the oracle", literally.
    for round in 0..ROUNDS as usize {
        assert_eq!(
            c3.digests[round], v1.digests[round],
            "round {round}: the det-lane digest must stay bit-exact (stop-and-escalate on drift)"
        );
    }

    // Leg 2 — the native lane is a TOLERANCE class: per-round trained θ within the existing
    // Optimizer band (the outer bound the reference-parity lanes pin), numbers reported.
    let tol = tol_for(OpClass::Optimizer);
    let layout: Vec<String> = (0..v1.init.len()).map(|i| format!("p{i}")).collect();
    for round in 0..ROUNDS as usize {
        let mut max_delta = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (v, c)) in v1.trained[round]
            .iter()
            .zip(c3.trained[round].iter())
            .enumerate()
        {
            assert_eq!(v.len(), c.len(), "round {round} {} numel", layout[i]);
            for (j, (&vv, &cv)) in v.iter().zip(c.iter()).enumerate() {
                let diff = (vv - cv).abs();
                let bound = tol.atol + tol.rtol * vv.abs();
                max_delta = max_delta.max(diff);
                if vv != 0.0 {
                    max_rel = max_rel.max(diff / vv.abs());
                }
                assert!(
                    diff <= bound,
                    "round {round} {}[{j}]: |{vv} - {cv}| = {diff} > {bound} (Optimizer band \
                     rtol {} / atol {}) — tolerance-class breach is a stop-and-escalate",
                    layout[i],
                    tol.rtol,
                    tol.atol
                );
            }
        }
        eprintln!(
            "c3_parity round {round}: trained-θ max |Δ| = {max_delta:.3e}, max rel = \
             {max_rel:.3e} (Optimizer band rtol {:.0e}/atol {:.0e})",
            tol.rtol, tol.atol
        );
    }
    eprintln!(
        "c3_parity: det-lane digests bit-exact across {ROUNDS} rounds (v1 oracle reproduced)"
    );
}
