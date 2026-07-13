// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Envelope freeze / validation conformance (TDD PROTO-11, spec §6.1).

use std::collections::BTreeMap;

use ciborium::value::Value;
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition,
};
use daemon_swarm_proto::{blake3_hash, to_canonical_vec, Hash, SigningKey};

fn text(s: &str) -> Value {
    Value::Text(s.into())
}

fn int(n: i64) -> Value {
    Value::Integer(n.into())
}

/// Build the `[experiment.config]` map. `reversed` presents the same logical config with its keys
/// (and a nested map's keys) in the opposite insertion order — the analogue of two TOML files that
/// differ only in field ordering / formatting.
fn sample_config(reversed: bool) -> Value {
    let inner = {
        let mut entries = vec![
            (text("rule"), text("adamw")),
            (text("lr"), Value::Float(4e-4)),
            (
                text("betas"),
                Value::Array(vec![Value::Float(0.9), Value::Float(0.95)]),
            ),
            (text("wd"), Value::Float(0.1)),
        ];
        if reversed {
            entries.reverse();
        }
        Value::Map(entries)
    };
    let mut entries = vec![
        (text("d_model"), int(1024)),
        (text("n_layers"), int(24)),
        (text("profile"), text("sparse-loco")),
        (text("h"), int(30)),
        (text("ef_decay"), Value::Float(0.95)),
        (text("inner"), inner),
    ];
    if reversed {
        entries.reverse();
    }
    Value::Map(entries)
}

fn artifact(seed: u8) -> Artifact {
    Artifact {
        url: format!("r2://runs/smollm/{seed}.bin"),
        blake3: Hash([seed; 32]),
    }
}

fn sample_envelope(config: Value) -> Envelope {
    let mut artifacts = BTreeMap::new();
    artifacts.insert("experiment.wasm".to_string(), artifact(1));
    artifacts.insert("data.manifest".to_string(), artifact(2));
    artifacts.insert("tokenizer.json".to_string(), artifact(3));
    Envelope {
        run: RunSection {
            schema: 1,
            run_id: "smollm-500m-01".into(),
            min_peers: 4,
            max_peers: 64,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config,
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round: 30,
            global_batch: GlobalBatch {
                start: 256,
                end: 512,
                ramp_rounds: 2000,
            },
            stop: StopCondition::Tokens(10_000_000_000),
        },
        requirements: Requirements {
            vram_mb_min: 11000,
            ram_gb_min: 16,
            uplink_mbps_min: 15,
            downlink_mbps_min: 100,
            disk_gb_min: 60,
            throughput_floor: "c2".into(),
            update_mb_max: 40,
            capabilities: vec!["tensor-abi@1".into(), "adamw_step@1".into()],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 300,
            round_train_max: 900,
            round_witness: 60,
            cooldown: 120,
            epoch_rounds: 400,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 8,
        },
    }
}

fn author() -> SigningKey {
    SigningKey::from_bytes(&[0x42; 32])
}

#[test]
fn freeze_idempotent() {
    let env = sample_envelope(sample_config(false));
    let k = author();
    let a = env.freeze(&k).unwrap();
    let b = env.freeze(&k).unwrap();
    assert_eq!(a.bytes(), b.bytes());
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.signature(), b.signature());
    assert_eq!(a.config_bytes(), b.config_bytes());
}

#[test]
fn frozen_bytes_stable_across_toml_formatting() {
    let k = author();
    let a = sample_envelope(sample_config(false)).freeze(&k).unwrap();
    let b = sample_envelope(sample_config(true)).freeze(&k).unwrap();
    // Same logical content, different authoring key order → identical frozen bytes and signature.
    assert_eq!(a.bytes(), b.bytes());
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.signature(), b.signature());
}

#[test]
fn config_subslice_is_da_build_input() {
    let env = sample_envelope(sample_config(false));
    let frozen = env.freeze(&author()).unwrap();
    // The config bytes equal the standalone canonical encoding of `[experiment.config]` …
    let standalone = to_canonical_vec(&env.experiment.config).unwrap();
    assert_eq!(frozen.config_bytes(), standalone.as_slice());
    // … and are a contiguous subslice of the whole frozen envelope (one byte chain to da_build).
    let cfg = frozen.config_bytes();
    assert!(
        frozen.bytes().windows(cfg.len()).any(|w| w == cfg),
        "config canonical bytes must appear contiguously in the frozen envelope"
    );
    assert!(!cfg.is_empty());
}

#[test]
fn verify_signature_rejects_tamper() {
    let frozen = sample_envelope(sample_config(false))
        .freeze(&author())
        .unwrap();
    assert!(frozen.verify().is_ok());

    // Re-open with a flipped byte: the recomputed hash no longer matches the signed hash.
    let mut bytes = frozen.bytes().to_vec();
    bytes[10] ^= 0x01;
    let reopened =
        daemon_swarm_proto::FrozenEnvelope::open(bytes, *frozen.signature(), *frozen.signer());
    assert!(reopened.is_err(), "tampered envelope must fail to open");

    // A wrong signer is also rejected on a pristine body.
    let other = SigningKey::from_bytes(&[0x11; 32]);
    let bad = daemon_swarm_proto::FrozenEnvelope::open(
        frozen.bytes().to_vec(),
        *frozen.signature(),
        daemon_swarm_proto::peer_id(&other),
    );
    assert!(bad.is_err());
}

#[test]
fn rejects_unknown_schema_major() {
    let mut env = sample_envelope(sample_config(false));
    env.run.schema = 2;
    assert!(env.validate().is_err());
    assert!(env.freeze(&author()).is_err());
}

#[test]
fn rejects_missing_artifact() {
    let mut env = sample_envelope(sample_config(false));
    env.artifacts.remove("experiment.wasm");
    assert!(env.validate().is_err());

    let mut env2 = sample_envelope(sample_config(false));
    env2.data.manifest = "nope.manifest".into();
    assert!(env2.validate().is_err());
}

#[test]
fn frozen_hash_is_blake3_of_bytes() {
    let frozen = sample_envelope(sample_config(false))
        .freeze(&author())
        .unwrap();
    assert_eq!(&blake3_hash(frozen.bytes()), frozen.hash());
}
