// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capture the A0 frozen v1 compatibility fixture (refactor §5 A0) from the PRE-REFACTOR tree.
//!
//! Links the pre-Phase-0 crates (`daemon-train` / `daemon-swarm-run` / `daemon-swarm-proto` /
//! `daemon-train-sdk`) by path and drives the **pre-refactor v1 five-phase driver** over the
//! **immutable pre-refactor `tiny_llama.wasm` bytes** (read from the pre-refactor guests target,
//! asserted against its committed `guests.blake3`; never recompiled), on the deterministic CPU
//! backend, with byte-reproducible corpus-derived inputs. Outputs (into `../`, the fixture
//! bundle dir in the post-rename worktree):
//!
//! - `tiny_llama.pre-refactor.wasm` — the frozen module bytes;
//! - `envelope.signed.cbor`         — the exact schema-major-1 `SignedEnvelope` wire bytes
//!   (canonical CBOR; deterministic seed key), pinning the module + corpus by blake3;
//! - `expected.json`                — every pin (hashes, window, batch derivation, run shape) plus
//!   the expected transcript: per-round payload blake3 + post-ingest det-lane state digest.
//!
//! The named tier-1 replay test (`daemon-vhc-host/tests/a0_frozen_fixture.rs`) reloads this bundle
//! on the renamed tree and must reproduce the transcript bit-for-bit under the v1 driver.
//!
//! Input derivation (the pinned, byte-reproducible rule — duplicated verbatim in the replay test):
//! shard-0000.bin (pinned by blake3) decodes as little-endian u16 into `raw[0..262144]`; for batch
//! index `b = round * H + step`, token `i` of the 2×8 micro-batch is
//! `raw[(WINDOW_START + b*16 + i) % 262144] % VOCAB`.

use std::path::{Path, PathBuf};

use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition,
};
use daemon_swarm_proto::{blake3_hash, to_canonical_vec, SigningKey};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StepCtx, TrainerBackend};
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig};
use daemon_train_sdk::models::TinyLlamaCfg;
use serde::Serialize;

/// Sequences per micro-batch (mirrors the pre-refactor determinism harness).
const SEQS: u32 = 2;
/// Tokens per sequence (must be < the config's `seq_len`).
const SEQ: u32 = 8;
/// Rounds captured.
const ROUNDS: u64 = 4;
/// Token-index window start into the decoded shard-0 stream.
const WINDOW_START: usize = 4096;
/// Token ids are folded into the tiny model's vocabulary (`id % VOCAB`).
const VOCAB: u32 = 64;
/// Deterministic fixture author key (a test key; the fixture's signer identity).
const AUTHOR_SEED: [u8; 32] = [7; 32];

fn pre_tree() -> PathBuf {
    // capture/ -> a0-frozen-v1 -> fixtures -> tests -> daemon-vhc-host -> host -> vhc -> crates
    // -> vhc-a0 -> daemon-worktree; then into the pre-refactor worktree.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../../../../../swarm-p3-integration")
        .canonicalize()
        .expect("pre-refactor worktree at ../swarm-p3-integration next to this checkout")
}

fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("fixture bundle dir")
}

/// The pinned tiny config (mirrors the pre-refactor determinism harness's `tiny_cfg`).
fn tiny_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        profile: "sparse_loco".to_string(),
        ..TinyLlamaCfg::default()
    }
}

/// Decode a `u16`-width shard into token ids (little-endian pairs — the tokenize-corpus layout).
fn decode_u16_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(2)
        .map(|p| u32::from(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

/// The pinned batch derivation (see the module docs; duplicated in the replay test).
fn batch_tokens(raw: &[u32], batch_index: u64) -> Vec<u32> {
    let n = (SEQS * SEQ) as usize;
    let base = WINDOW_START + (batch_index as usize) * n;
    (0..n)
        .map(|i| raw[(base + i) % raw.len()] % VOCAB)
        .collect()
}

#[derive(Serialize)]
struct ExpectedRound {
    round: u64,
    payload_blake3: String,
    digest: String,
}

fn main() {
    let pre = pre_tree();
    let out = out_dir();

    // 1. The immutable pre-refactor module bytes, asserted against the committed guests.blake3.
    let module_path = pre.join("guests/target/wasm32-unknown-unknown/release/tiny_llama.wasm");
    let module = std::fs::read(&module_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", module_path.display()));
    let module_hash = blake3_hash(&module);
    let manifest_text = std::fs::read_to_string(pre.join("guests/guests.blake3"))
        .expect("pre-refactor guests.blake3");
    let committed = manifest_text
        .lines()
        .find_map(|l| l.strip_suffix("  tiny_llama.wasm"))
        .expect("tiny_llama.wasm entry in guests.blake3");
    assert_eq!(
        module_hash.to_hex(),
        committed,
        "the on-disk tiny_llama.wasm must be the committed pre-refactor build — rebuild it in the \
         pre-refactor tree (cargo run -p xtask -- build-guests) if stale"
    );

    // 2. The pinned corpus shard (the vendored TinyStories fixture, pinned by blake3).
    let corpus_dir = pre.join("crates/swarm/daemon-swarm-run/tests/fixtures/tinystories");
    let corpus_manifest =
        std::fs::read(corpus_dir.join("manifest.json")).expect("corpus manifest.json");
    let shard0 = std::fs::read(corpus_dir.join("shard-0000.bin")).expect("shard-0000.bin");
    let corpus_manifest_hash = blake3_hash(&corpus_manifest);
    let shard0_hash = blake3_hash(&shard0);
    assert_eq!(
        shard0_hash.to_hex().as_str(),
        "96da080176dabf76a9321ec4df2332f1089b7c77dfbdea39d1a3894186393ae8",
        "shard-0000.bin must be the vendored fixture the manifest pins"
    );
    let raw = decode_u16_le(&shard0);
    assert_eq!(raw.len(), 262_144);

    // 3. The exact schema-major-1 envelope, frozen + signed with the deterministic author key.
    let cfg = tiny_cfg();
    let mut cfg_cbor = Vec::new();
    ciborium::into_writer(&cfg, &mut cfg_cbor).expect("cfg cbor");
    let cfg_value: ciborium::value::Value =
        ciborium::from_reader(cfg_cbor.as_slice()).expect("cfg value");

    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "tiny-llama".to_string(),
        Artifact {
            url: "file://fixtures/a0-frozen-v1/tiny_llama.pre-refactor.wasm".to_string(),
            blake3: module_hash,
        },
    );
    artifacts.insert(
        "tinystories".to_string(),
        Artifact {
            url: "file://fixtures/tinystories/manifest.json".to_string(),
            blake3: corpus_manifest_hash,
        },
    );
    let envelope = Envelope {
        run: RunSection {
            schema: 1,
            run_id: "a0-frozen-v1-fixture".to_string(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "tiny-llama".to_string(),
            abi: "tensor-abi@1".to_string(),
            config: cfg_value,
        },
        artifacts,
        data: DataSection {
            manifest: "tinystories".to_string(),
            steps_per_round: 3,
            global_batch: GlobalBatch {
                start: SEQS,
                end: SEQS,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(ROUNDS),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 1,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 1,
            throughput_floor: "c1".to_string(),
            update_mb_max: 4,
            capabilities: vec!["tensor-abi@1".to_string()],
            payload_store: "fs".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 60,
            round_train_max: 600,
            round_witness: 60,
            cooldown: 30,
            epoch_rounds: 4,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    };
    let key = SigningKey::from_bytes(&AUTHOR_SEED);
    let frozen = envelope.freeze(&key).expect("freeze fixture envelope");
    let wire = frozen.to_wire();
    let wire_bytes = to_canonical_vec(&wire).expect("wire cbor");

    // 4. Drive the pre-refactor v1 driver: build → (step × H → inner_update) → make_update →
    //    self-ingest, on the deterministic CPU backend, recording the transcript.
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.clone(),
        engine: EngineConfig::default(),
    })
    .expect("construct pre-refactor WasmBackend");
    backend.build(frozen.config_bytes()).expect("da_build");
    let steps = backend.steps_per_round().expect("steps_per_round");
    assert_eq!(steps, 3, "the sparse_loco tiny config runs H=3");

    let mut transcript = Vec::new();
    for round in 0..ROUNDS {
        for step in 0..steps {
            let b = BatchRef {
                tokens: batch_tokens(&raw, round * u64::from(steps) + u64::from(step)),
                seq_len: SEQ,
            };
            let ctx = StepCtx {
                inner_step: step,
                mb_index: 0,
                mb_count: 1,
                step_seqs: SEQS,
            };
            backend.train_step(&b, ctx).expect("train_step");
            backend.inner_update(step).expect("inner_update");
        }
        let payload = backend.make_update(round).expect("make_update");
        let staged = StagedPayload {
            peer: daemon_swarm_proto::PeerId([1; 32]),
            hash: blake3_hash(&payload),
            bytes: payload.clone(),
        };
        let digest = backend.ingest(round, &[staged]).expect("ingest");
        transcript.push(ExpectedRound {
            round,
            payload_blake3: blake3_hash(&payload).to_hex(),
            digest: digest.to_hex(),
        });
    }

    // 5. Write the bundle.
    let expected = serde_json::json!({
        "captured_from": {
            "tree": "swarm-p3-integration (pre-Phase-0 rename)",
            "commit": "6706fda",
            "driver": "v1 five-phase (daemon-train WasmBackend, CpuBackend det lane)",
        },
        "module": {
            "file": "tiny_llama.pre-refactor.wasm",
            "bytes": module.len(),
            "blake3": module_hash.to_hex(),
        },
        "envelope": {
            "file": "envelope.signed.cbor",
            "schema_major": 1,
            "wire_blake3": blake3_hash(&wire_bytes).to_hex(),
            "envelope_hash": frozen.hash().to_hex(),
            "config_blake3": blake3_hash(frozen.config_bytes()).to_hex(),
            "signer_seed": "SigningKey::from_bytes([7; 32]) (deterministic test key)",
        },
        "corpus": {
            "manifest_blake3": corpus_manifest_hash.to_hex(),
            "shard0_blake3": shard0_hash.to_hex(),
            "shard0_tokens": raw.len(),
            "token_width": "u16-le",
            "window_start": WINDOW_START,
            "vocab_mod": VOCAB,
            "derivation": "token[i] of batch b = raw[(window_start + b*16 + i) % 262144] % 64",
        },
        "run": {
            "rounds": ROUNDS,
            "steps_per_round": steps,
            "seqs_per_batch": SEQS,
            "seq_len": SEQ,
            "backend": "cpu (BackendKind::Cpu, EngineConfig::default())",
            "staged_peer": "PeerId([1; 32]) (self-ingest, one payload per round)",
        },
        "transcript": transcript,
    });

    std::fs::write(out.join("tiny_llama.pre-refactor.wasm"), &module).expect("write module");
    std::fs::write(out.join("envelope.signed.cbor"), &wire_bytes).expect("write envelope");
    std::fs::write(
        out.join("expected.json"),
        format!("{}\n", serde_json::to_string_pretty(&expected).unwrap()),
    )
    .expect("write expected.json");

    println!("A0 frozen fixture captured into {}", out.display());
    println!("  module  blake3 {}", module_hash.to_hex());
    println!(
        "  envelope wire blake3 {}",
        blake3_hash(&wire_bytes).to_hex()
    );
    println!("  envelope hash {}", frozen.hash().to_hex());
    for r in expected["transcript"].as_array().unwrap() {
        println!(
            "  round {} payload {} digest {}",
            r["round"], r["payload_blake3"], r["digest"]
        );
    }
}
