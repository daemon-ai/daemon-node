// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// THE Phase-A acceptance (refactor §5): TinyLlama on `BarrierRound` under the v2 event-loop
// driver + the §2.5 tabi bridge reproduces the v1 det-lane state digests — the evidence that the
// control inversion changed the choreography's HOME, not the math.
//
// Harness (the recorded sitting-7 design): single-peer roster; the v1 `WasmBackend` oracle runs
// first on identical inputs (same model config, same zero-token batches, same step/micro
// schedule); its per-round sealed update bytes feed the v2 run's kind-2 staged ingest (the
// OUTBOUND SEALING GAP: a bridge guest cannot seal its own container — Phase B's payload_put
// closes it; the state digest covers the gap transitively, since the container is a function of
// the same params the digest hashes); the v2 run's final canonical state is exported through the
// pump hook and digested exactly as `WasmBackend::digest_of` does. Equal digests ⇒ equal math.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::v2::{start_run, MemorySink, RunEnd, RunIdentity, V2RunConfig};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, SwarmMessage,
};
use daemon_vhc_proto::{blake3_hash, digest_state, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk::models::TinyLlamaCfg;
use daemon_vhc_session::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use daemon_vhc_session::{WasmBackend, WasmBackendConfig};

const ROUNDS: u64 = 2;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2; // == sequences per step ⇒ one micro-window per inner step
const SEQ_LEN: u32 = 9;
const PEER: [u8; 32] = [7u8; 32];

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

fn model_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: SEQ_LEN,
        ..TinyLlamaCfg::default()
    }
}

fn model_cfg_bytes() -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(&model_cfg(), &mut b).expect("cfg cbor");
    b
}

fn zero_tokens() -> Vec<u32> {
    vec![0u32; (MICRO_BATCH * SEQ_LEN) as usize]
}

/// The digest exactly as `WasmBackend::digest_of` computes it (round-seeded, full sampling).
fn digest_of_state(state: &[u8], round: u64) -> [u8; 16] {
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&round.to_le_bytes());
    let d = digest_state(&Seed(seed), 64, u32::MAX, state);
    *d.as_bytes()
}

/// The v1 oracle: drive `WasmBackend` through the identical schedule; return the per-round sealed
/// update bytes and the final round's digest.
fn v1_oracle(engine: EngineConfig) -> (Vec<Vec<u8>>, StateDigest) {
    v1_oracle_n(engine, ROUNDS)
}

fn v1_oracle_n(engine: EngineConfig, rounds: u64) -> (Vec<Vec<u8>>, StateDigest) {
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: guest("tiny_llama"),
        engine,
    })
    .expect("v1 backend");
    b.build(&model_cfg_bytes()).expect("build");
    let peer = PeerId(PEER);
    let mut updates = Vec::new();
    let mut last = None;
    for round in 0..rounds {
        for h in 0..STEPS_PER_ROUND {
            b.train_step(
                &BatchRef {
                    tokens: zero_tokens(),
                    seq_len: SEQ_LEN,
                },
                StepCtx {
                    inner_step: h,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: MICRO_BATCH,
                },
            )
            .expect("step");
            b.inner_update(h).expect("inner");
        }
        let bytes = b.make_update(round).expect("make_update");
        let digest = b
            .ingest(
                round,
                &[StagedPayload {
                    peer,
                    hash: blake3_hash(&bytes),
                    bytes: bytes.clone(),
                }],
            )
            .expect("ingest");
        updates.push(bytes);
        last = Some(digest);
    }
    (updates, last.expect("rounds ran"))
}

fn guest_cfg_bytes() -> Vec<u8> {
    let cfg = Value::Map(vec![
        (
            Value::Text("model".into()),
            Value::serialized(&model_cfg()).expect("model value"),
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
    ]);
    to_canonical_vec(&cfg).expect("guest cfg")
}

fn wait_published(pump: &daemon_vhc_host::v2::PumpHandle, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while pump.published().len() < n {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {n} publishes (have {})",
            pump.published().len()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Drive the v2 guest through the identical schedule, feeding the v1 oracle's update bytes at the
/// barrier; return the final canonical state digest.
fn v2_run(engine: EngineConfig, v1_updates: &[Vec<u8>]) -> StateDigest {
    v2_run_n(engine, v1_updates, ROUNDS)
}

fn v2_run_n(engine: EngineConfig, v1_updates: &[Vec<u8>], rounds: u64) -> StateDigest {
    let wasm = guest("tiny_llama_v2");
    let worker = Worker::new(engine).expect("engine");
    let sel = daemon_vhc_host::select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("v2 guest admitted");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);

    let identity = RunIdentity {
        run_id: [0xAB; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let run_cfg = V2RunConfig::new(identity, [0x81; 32], guest_cfg_bytes(), Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let peer = PeerId(PEER);
    let mut seq = 0u64;
    let sender = [9u8; 32];

    for round in 0..rounds {
        // Stage the round's batches in training order (one micro-window per inner step).
        for _h in 0..STEPS_PER_ROUND {
            pump.stage_batch(&zero_tokens(), MICRO_BATCH, SEQ_LEN, None)
                .expect("stage batch");
        }
        // RoundOpen → the guest trains + voices its Commitment.
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
        let payload = to_canonical_vec(&ro).expect("ro");
        pump.deliver_frame(0, seq, sender, payload.clone(), payload)
            .expect("deliver RoundOpen");
        seq += 1;
        wait_published(&pump, (round as usize) * 2 + 1); // + this round's Commitment

        // Barrier: stage the (v1-sealed) committed update, then the record.
        let bytes = v1_updates[round as usize].clone();
        let entry = RecordEntry {
            peer,
            hash: blake3_hash(&bytes),
            size: bytes.len() as u64,
        };
        pump.stage_update(bytes, None).expect("stage update");
        let set: Vec<(PeerId, Hash)> = vec![(peer, entry.hash)];
        let rr = SwarmMessage::RoundRecord(RoundRecord {
            round,
            set: commit_set(&set).commitment(),
            drops: Vec::new(),
            next_seed: Seed([0; 32]),
            set_locator: Locator::StoreKey(String::new()),
            inline: Some(vec![entry]),
        });
        let payload = to_canonical_vec(&rr).expect("rr");
        pump.deliver_frame(0, seq, sender, payload.clone(), payload)
            .expect("deliver RoundRecord");
        seq += 1;
        wait_published(&pump, (round as usize) * 2 + 2); // + this round's Digest
    }

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }
    let state = pump
        .bridge_final_state()
        .expect("the digest-extraction hook exported the bridge state");
    StateDigest(digest_of_state(&state, rounds - 1))
}

/// The named Phase-A acceptance lane, CPU tier: v1 driver ≡ v2 BarrierRound det digests.
#[test]
fn tiny_llama_on_barrier_round_reproduces_v1_digests_cpu() {
    let (updates, v1_digest) = v1_oracle(EngineConfig::default());
    let v2_digest = v2_run(EngineConfig::default(), &updates);
    assert_eq!(
        v1_digest, v2_digest,
        "the control inversion must not change the det-lane math (refactor §5 A2)"
    );
}

/// The burn-ndarray tier (runs under the host suite's `--features burn-ndarray` lane).
#[cfg(feature = "burn-ndarray")]
#[test]
fn tiny_llama_on_barrier_round_reproduces_v1_digests_burn_ndarray() {
    let engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::BurnNdarray,
        ..EngineConfig::default()
    };
    let (updates, v1_digest) = v1_oracle(engine.clone());
    let v2_digest = v2_run(engine, &updates);
    assert_eq!(v1_digest, v2_digest);
}

/// The wgpu tier — hardware-gated (TDD §8.1 tier-2 GPU-skip convention; the `.#vulkan` shell /
/// a GPU runner exercises it). The det lane is bit-exact on every backend (§5.9 residency).
#[cfg(feature = "wgpu")]
#[test]
fn tiny_llama_on_barrier_round_reproduces_v1_digests_wgpu() {
    if !daemon_vhc_host::wgpu_adapter_available() {
        eprintln!("SKIP v2_parity(wgpu): no usable wgpu adapter on this runner");
        return;
    }
    let engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::Wgpu,
        ..EngineConfig::default()
    };
    let (updates, v1_digest) = v1_oracle(engine.clone());
    let v2_digest = v2_run(engine, &updates);
    assert_eq!(v1_digest, v2_digest);
}

/// The cuda tier — hardware-gated like the wgpu tier.
#[cfg(feature = "cuda")]
#[test]
fn tiny_llama_on_barrier_round_reproduces_v1_digests_cuda() {
    if !daemon_vhc_host::cuda_adapter_available() {
        eprintln!("SKIP v2_parity(cuda): no usable CUDA device on this runner");
        return;
    }
    let engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::Cuda,
        ..EngineConfig::default()
    };
    let (updates, v1_digest) = v1_oracle(engine.clone());
    let v2_digest = v2_run(engine, &updates);
    assert_eq!(v1_digest, v2_digest);
}
