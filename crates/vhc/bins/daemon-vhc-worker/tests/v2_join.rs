// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The D3 cell-5/6 live t2 whole-run** (decisions D3; ABI §9.3): a REAL v2 worker module
// (tiny-llama-v2, the macro-emitted BarrierRound guest) under the modified **v1 envelope**
// carrying `device_min`, driven by the **native coordinator** (the pure `tick`, in-process in
// the worker binary), through the real worker protocol: probe → assess (claim funnel + stage-3
// pre-screen over the envelope section) → join → two full rounds (train → commit → record →
// barrier ingest) → digest — upgrading cell 5 from fixture-supported to whole-run-proven and
// exercising cell 6's v2-worker × native-coordinator path (its envelope-v2 form is D0's).
//
// The run also proves the inline replay soak (refactor §12.6): the worker re-drives its own
// recorded journal through the §8.7 engine before reporting the round outcome — a diverging
// run is a join FAILURE (the `replay_decisions` metric is the green receipt).

// Dev/test harness: shells cargo for the guest build (same pattern as worker_protocol.rs); the
// env/spawn bans target the shipped node, so they are allowed file-wide here.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, ExperimentSection, GlobalBatch, Phases, Requirements, RoundMode,
    RunSection, StopCondition,
};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Envelope, Hash, SignedEnvelope, SigningKey};
use daemon_vhc_sdk::models::TinyLlamaCfg;
use daemon_vhc_session::protocol::{Event, JoinPolicy, PolicyMode};
use daemon_vhc_supervisor::{TrainClientConfig, TrainSupervisor};

const RUN_ID: &str = "cell56-t2";

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

fn module_path() -> PathBuf {
    BUILD.call_once(|| {
        let status = StdCommand::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    guests_root().join("target/wasm32-unknown-unknown/release/tiny_llama_v2.wasm")
}

/// The guest config (`GuestCfg`): the tiny parity model on a 2-step round, one sequence per
/// micro window (matching the worker's Phase-A zero-token staging shape).
fn guest_config() -> Value {
    let model = Value::serialized(&TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    })
    .expect("model value");
    Value::Map(vec![
        (Value::from("model"), model),
        (Value::from("peer"), Value::Bytes(vec![7u8; 32])),
        (
            Value::from("roster"),
            Value::Array(vec![Value::Bytes(vec![7u8; 32])]),
        ),
        (Value::from("steps_per_round"), Value::from(2u32)),
        (Value::from("micro_batch"), Value::from(1u32)),
        (Value::from("stall_rounds_max"), Value::from(2u32)),
    ])
}

/// The v1 envelope for the run: schema 1, single peer, a 2-sequence window per round — plus the
/// additive `device_min` section injected at the raw-CBOR level and re-signed (the cell-5 form).
fn envelope_wire() -> Vec<u8> {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
        },
    );
    let env = Envelope {
        run: RunSection {
            schema: 1,
            run_id: RUN_ID.into(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config: guest_config(),
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round: 2,
            global_batch: GlobalBatch {
                start: 2,
                end: 2,
                ramp_rounds: 1,
            },
            stop: StopCondition::Tokens(1_000_000),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 1,
            uplink_mbps_min: 1,
            downlink_mbps_min: 1,
            disk_gb_min: 1,
            throughput_floor: "c1".into(),
            update_mb_max: 8,
            capabilities: vec![],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 60,
            round_witness: 1,
            cooldown: 1,
            epoch_rounds: 10,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    };
    let key = SigningKey::from_bytes(&[0x42; 32]);
    let frozen = env.freeze(&key).expect("freeze");
    // Inject `device_min` additively (the raw-CBOR author-side operation) and re-sign.
    let v: Value = ciborium::de::from_reader(frozen.bytes()).expect("decode");
    let Value::Map(mut entries) = v else { panic!() };
    entries.push((
        Value::from("device_min"),
        Value::Map(vec![
            (Value::from("gpu"), Value::from(1u64)),
            (Value::from("ram_bytes"), Value::from(1u64 << 20)),
        ]),
    ));
    let bytes = to_canonical_vec(&Value::Map(entries)).expect("re-encode");
    let hash = blake3_hash(&bytes);
    let signature = daemon_vhc_proto::sign::sign_canonical(&key, &hash).expect("sign");
    let wire = SignedEnvelope {
        bytes,
        signature,
        signer: daemon_vhc_proto::sign::peer_id(&key),
    };
    to_canonical_vec(&wire).expect("wire")
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// probe → assess → join → two native-coordinator-driven rounds → replay-soaked digest.
#[tokio::test]
async fn v2_worker_joins_and_runs_rounds_under_the_native_coordinator() {
    let module = module_path();
    assert!(module.exists(), "tiny_llama_v2.wasm missing");
    let mut cfg = TrainClientConfig::new(env!("CARGO_BIN_EXE_daemon-vhc-worker").to_string());
    cfg.env = vec![
        (
            "DAEMON_TRAIN_MODULE".to_string(),
            module.to_string_lossy().into_owned(),
        ),
        // The owner's node-side lane choice (§9.6 numbers-are-config): CPU-admitting t2 lane.
        ("DAEMON_VHC_LANE_GPU_OPTIONAL".to_string(), "1".to_string()),
    ];
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(180);
    let sup = TrainSupervisor::new(cfg);

    // Assess through the REAL funnel: the envelope's device_min feeds stage 3; the claim is the
    // SDK-derived one the main! macro emits for tiny-llama-v2.
    let elig = sup.assess(envelope_wire()).await.expect("assess");
    assert!(
        elig.eligible,
        "v2 module admits under the t2 lane: {:?}",
        elig.reasons
    );

    // Join: the worker's v2 session — pump attach + the in-process native coordinator.
    let mut events = sup
        .join_streaming(RUN_ID, "local://native-tick", vec![], policy())
        .await
        .expect("join");

    // The event stream: RunPhase(train) → Metric(replay_decisions) → RoundOutcome(final digest).
    let mut replay_decisions = None;
    let mut outcome = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    while outcome.is_none() {
        let ev = tokio::time::timeout_at(deadline, events.recv())
            .await
            .expect("worker event stream stalled")
            .expect("worker event stream closed early");
        match ev {
            Event::Metric { name, value } if name == "replay_decisions" => {
                replay_decisions = Some(value);
            }
            Event::RoundOutcome {
                round,
                committed,
                ingested,
                stalled,
                digest,
            } => {
                assert_eq!(round, 1, "two rounds ran (0 and 1)");
                assert_eq!((committed, ingested, stalled), (1, 1, false));
                assert_ne!(digest, [0u8; 16], "a real det-lane digest");
                outcome = Some(digest);
            }
            Event::Error { detail, .. } => panic!("worker error: {detail}"),
            _ => {}
        }
    }

    // The inline §12.6 replay soak ran and reproduced every decision (2 rounds × 2 publishes).
    assert_eq!(
        replay_decisions,
        Some(4.0),
        "the recorded journal re-drove bit-for-bit before the outcome was reported"
    );
}
