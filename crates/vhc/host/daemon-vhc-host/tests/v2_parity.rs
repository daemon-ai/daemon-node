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
        // Service the guest's payload_put (the B1 guest-authored sealing path): the harness is
        // the async-runtime seat; the pump computes the commitment hash itself.
        for (op, request) in pump.take_op_requests() {
            match request {
                daemon_vhc_host::v2::OpRequest::PayloadPut { .. } => {
                    pump.complete_op(op, daemon_vhc_host::v2::OpOutcome::PutDone)
                        .expect("put completion");
                }
                other => panic!("unexpected op request from the parity guest: {other:?}"),
            }
        }
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
///
/// Only the GPU/burn tier tests below call this; the default (no-feature) tier drives `v2_run_n`
/// directly, so gate it to keep a plain `--all-targets` clippy pass dead-code-clean.
#[cfg(any(feature = "burn-ndarray", feature = "wgpu", feature = "cuda"))]
fn v2_run(engine: EngineConfig, v1_updates: &[Vec<u8>]) -> StateDigest {
    v2_run_n(engine, v1_updates, ROUNDS).0
}

fn v2_run_n(
    engine: EngineConfig,
    v1_updates: &[Vec<u8>],
    rounds: u64,
) -> (StateDigest, Vec<daemon_vhc_host::v2::SinkEntry>) {
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
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
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
        assert_eq!(
            pump.deliver_frame(0, seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
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
        assert_eq!(
            pump.deliver_frame(0, seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
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
    let entries = sink.lock().expect("sink").entries.clone();
    (StateDigest(digest_of_state(&state, rounds - 1)), entries)
}

/// Drive the v2 guest through a STRAGGLE → CATCH-UP schedule (the B3-found §5.9 defect shape):
/// round 0 trains + commits normally, but its record arrives while the committed payload is not
/// yet fetchable (straggle); the payload lands; RoundOpen(1) then makes the guest ingest round 0
/// AND train round 1 **in one event slice** — the catch-up path where the ingest epilogue
/// (post-ingest master → round base) must run at the ingest→training boundary, not at slice
/// close, or round 1 trains against a pre-ingest base and diverges from v1.
fn v2_run_catchup(
    engine: EngineConfig,
    v1_updates: &[Vec<u8>],
) -> (StateDigest, Vec<daemon_vhc_host::v2::SinkEntry>) {
    let wasm = guest("tiny_llama_v2");
    let worker = Worker::new(engine).expect("engine");
    let identity = RunIdentity {
        run_id: [0xAC; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 2,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let run_cfg = V2RunConfig::new(identity, [0x82; 32], guest_cfg_bytes(), Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();
    let peer = PeerId(PEER);
    let mut seq = 0u64;
    let sender = [9u8; 32];

    let ro = |round: u64| {
        SwarmMessage::RoundOpen(RoundOpen {
            round,
            seed: Seed([round as u8; 32]),
            roster_digest: Hash([0; 32]),
            batch: BatchWindow {
                start: 0,
                end: u64::from(STEPS_PER_ROUND * MICRO_BATCH),
            },
            deadline_unix_s: 0,
        })
    };
    let rr = |round: u64, bytes: &[u8]| {
        let entry = RecordEntry {
            peer,
            hash: blake3_hash(bytes),
            size: bytes.len() as u64,
        };
        let set: Vec<(PeerId, Hash)> = vec![(peer, entry.hash)];
        SwarmMessage::RoundRecord(RoundRecord {
            round,
            set: commit_set(&set).commitment(),
            drops: Vec::new(),
            next_seed: Seed([0; 32]),
            set_locator: Locator::StoreKey(String::new()),
            inline: Some(vec![entry]),
        })
    };
    let deliver = |msg: &SwarmMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            daemon_vhc_host::v2::DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    // Round 0: normal train + guest-authored commit.
    for _h in 0..STEPS_PER_ROUND {
        pump.stage_batch(&zero_tokens(), MICRO_BATCH, SEQ_LEN, None)
            .expect("stage batch");
    }
    deliver(&ro(0), &mut seq);
    wait_published(&pump, 1); // Commitment(0)

    // The record arrives while the committed payload is NOT yet fetchable → straggle.
    deliver(&rr(0, &v1_updates[0]), &mut seq);
    wait_published(&pump, 2); // Straggle(0)

    // The payload lands (the archive/store caught up).
    pump.stage_update(v1_updates[0].clone(), None)
        .expect("stage update 0");

    // RoundOpen(1): ONE slice ingests round 0 (catch-up) and trains round 1.
    for _h in 0..STEPS_PER_ROUND {
        pump.stage_batch(&zero_tokens(), MICRO_BATCH, SEQ_LEN, None)
            .expect("stage batch");
    }
    deliver(&ro(1), &mut seq);
    wait_published(&pump, 3); // Digest(0) — the CaughtUp voice
    wait_published(&pump, 4); // Commitment(1) — after the guest's put completes

    // Barrier 1, normal path.
    pump.stage_update(v1_updates[1].clone(), None)
        .expect("stage update 1");
    deliver(&rr(1, &v1_updates[1]), &mut seq);
    wait_published(&pump, 5); // Digest(1)

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }
    let state = pump.bridge_final_state().expect("final state exported");
    let entries = sink.lock().expect("sink").entries.clone();
    (StateDigest(digest_of_state(&state, ROUNDS - 1)), entries)
}

/// The B1 catch-up parity pin (the driver-side twin of B3's adversarial `assert_ne!`): a
/// straggle → catch-up run must end in EXACTLY the clean run's det state — the §5.9 ingest
/// epilogue fires at the ingest→training boundary, so round 1 trains against the post-ingest-0
/// base even when ingest(0) and train(1) share one slice — and the recorded run replays
/// bit-for-bit (drops/straggles are advisory; the journal records what was delivered).
#[test]
fn catch_up_after_straggle_reproduces_v1_digests_cpu() {
    let (updates, v1_digest) = v1_oracle(EngineConfig::default());
    let (v2_digest, entries) = v2_run_catchup(EngineConfig::default(), &updates);
    assert_eq!(
        v1_digest, v2_digest,
        "catch-up (ingest r + train r+1 in ONE slice) must reproduce v1 semantics — the §5.9 \
         epilogue must run at the ingest→training boundary, not at slice close"
    );

    // Replay green on the catch-up journal too.
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let script = daemon_vhc_host::v2::ReplayScript::from_entries(&entries);
    let replayed = daemon_vhc_host::v2::replay_v2(
        &worker,
        &guest("tiny_llama_v2"),
        &guest_cfg_bytes(),
        &[],
        script,
    )
    .expect("replay harness");
    assert_eq!(replayed.end, daemon_vhc_host::v2::ReplayEnd::Outcome(0));
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            daemon_vhc_host::v2::SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let replayed_decisions: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    assert_eq!(
        recorded, replayed_decisions,
        "catch-up decisions replay bit-for-bit"
    );
}

/// The named Phase-A acceptance lane, CPU tier: v1 driver ≡ v2 BarrierRound det digests — and
/// the recorded run REPLAYS bit-for-bit through the §8.7 input-replay engine (the richest
/// record mix: bridge compute, nr readouts, staged kinds 1/2, control frames, publishes).
#[test]
fn tiny_llama_on_barrier_round_reproduces_v1_digests_cpu() {
    let (updates, v1_digest) = v1_oracle(EngineConfig::default());
    let (v2_digest, entries) = v2_run_n(EngineConfig::default(), &updates, ROUNDS);
    assert_eq!(
        v1_digest, v2_digest,
        "the control inversion must not change the det-lane math (refactor §5 A2)"
    );

    // Input replay (§8.7): re-drive the SAME module from the journal alone; every publish and
    // the outcome must reproduce exactly (kernels not re-executed — nr readouts answered from
    // the record). The full verifier-contract wiring lives in tests/v2_replay.rs; this asserts
    // the invariant holds on the Phase-A acceptance run itself.
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let script = daemon_vhc_host::v2::ReplayScript::from_entries(&entries);
    let replayed = daemon_vhc_host::v2::replay_v2(
        &worker,
        &guest("tiny_llama_v2"),
        &guest_cfg_bytes(),
        &[],
        script,
    )
    .expect("replay harness");
    assert_eq!(replayed.end, daemon_vhc_host::v2::ReplayEnd::Outcome(0));
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            daemon_vhc_host::v2::SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let replayed_decisions: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    assert_eq!(
        recorded, replayed_decisions,
        "every recorded decision reproduces bit-for-bit (refactor §12.6 journal soak, v2)"
    );
    assert!(!recorded.is_empty(), "the run decided things");
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
