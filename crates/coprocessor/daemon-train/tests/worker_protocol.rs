// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The `daemon-train-worker` binary speaks the frozen `daemon_swarm_run::protocol` over the
// length-framed stdio cut. Two integration paths:
//   * through `daemon-train-client::TrainSupervisor` (the node-side supervisor) — probe/assess/join;
//   * a lower-level direct subprocess drive — observe the self-driven one-round `RoundOutcome`.
// Both spawn the real binary against a real tiny-llama `.wasm`.
//
// Dev/test harness: it shells `cargo build` for the guests and reads the `.wasm`, so the fs/process
// bans (which target the shipped node) are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use daemon_common::SessionId;
use daemon_provision::{Placement, PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_swarm_proto::{to_canonical_vec, Hash, SigningKey};
use daemon_swarm_run::protocol::{self, Command as WCmd, Event, JoinPolicy, PolicyMode};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};
use daemon_train_sdk::models::TinyLlamaCfg;

// -- guest module loading (mirrors tests/guest_lifecycle.rs) ------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
}

fn module_path() -> PathBuf {
    let path = guest_dir().join("tiny_llama.wasm");
    if !path.exists() {
        ensure_built();
    }
    assert!(
        path.exists(),
        "tiny_llama.wasm missing at {}",
        path.display()
    );
    path
}

fn tiny_cfg_cbor() -> Vec<u8> {
    let cfg = TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    };
    let mut b = Vec::new();
    ciborium::into_writer(&cfg, &mut b).expect("cbor");
    b
}

fn worker_bin() -> String {
    env!("CARGO_BIN_EXE_daemon-train-worker").to_string()
}

// -- through TrainSupervisor (the node-side supervisor) ------------------------------------------

/// CLI-1 / RUN-9 worker side: the supervisor spawns the real worker, and probe → assess → join all
/// succeed against a real tiny-llama module.
#[tokio::test]
async fn supervisor_probe_assess_join() {
    let module = module_path();
    let mut cfg = TrainClientConfig::new(worker_bin());
    cfg.env = vec![(
        "DAEMON_TRAIN_MODULE".to_string(),
        module.to_string_lossy().into_owned(),
    )];
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(60);
    let sup = TrainSupervisor::new(cfg);

    // Probe: a real host capability report — CPU-only, the full tabi@1 vocabulary.
    let hw = sup.probe().await.expect("probe");
    assert_eq!(hw.gpus, 0, "this build has no GPU lane");
    assert_eq!(hw.capabilities.abi_version, 1);
    assert_eq!(
        hw.capabilities.ops.len(),
        66,
        "the host reports the full frozen tabi@1 vocabulary"
    );
    assert!(hw.capabilities.ops.iter().any(|o| o == "flash_attn@1"));

    // Assess: the static import scan + meta pass over the tiny config → eligible.
    let elig = sup.assess(tiny_cfg_cbor()).await.expect("assess");
    assert!(
        elig.eligible,
        "tiny-llama must be eligible: {:?}",
        elig.reasons
    );

    // Join: the worker emits RunPhase{train} (the supervisor's join resolves here) then self-drives.
    sup.join("run-e3", "wss://coord.example/swarm", vec![], policy())
        .await
        .expect("join");

    sup.shutdown().await;
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// A real signed run envelope whose `tiny-llama.wasm` artifact points at the built module by
/// `file://` + real blake3, with the tiny-llama preset as `[experiment.config]`. Reopened + verified
/// by the worker, which resolves the module through `ArtifactResolver` (no `DAEMON_TRAIN_MODULE`).
fn signed_envelope_wire() -> Vec<u8> {
    let module = module_path();
    let bytes = std::fs::read(&module).expect("read module");
    let blake3 = Hash(*blake3::hash(&bytes).as_bytes());
    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "tiny-llama.wasm".to_string(),
        Artifact {
            url: format!("file://{}", module.display()),
            blake3,
        },
    );
    artifacts.insert(
        "manifest.json".to_string(),
        Artifact {
            url: "file://manifest.json".to_string(),
            blake3: Hash([0u8; 32]),
        },
    );
    let cfg = TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    };
    let env = Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: "worker-seam".to_string(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "tiny-llama.wasm".to_string(),
            abi: "tensor-abi@1".to_string(),
            config: ciborium::value::Value::serialized(&cfg).expect("cfg value"),
        },
        artifacts,
        data: DataSection {
            manifest: "manifest.json".to_string(),
            steps_per_round: 3,
            global_batch: GlobalBatch {
                start: 12,
                end: 12,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(4),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 0,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 0,
            throughput_floor: "c1".to_string(),
            update_mb_max: 64,
            capabilities: Vec::new(),
            payload_store: "r2".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 1_000,
            round_witness: 1_000,
            cooldown: 1,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 2,
            payload_retention_rounds: 8,
        },
    };
    let author = SigningKey::from_bytes(&[0xA1u8; 32]);
    let frozen = env.freeze(&author).expect("freeze envelope");
    to_canonical_vec(&frozen.to_wire()).expect("encode signed envelope")
}

/// The Merge-3 envelope seam: the worker receives the **real** signed envelope, verifies it, extracts
/// `[experiment.config]`, and resolves the module from the artifact map via `ArtifactResolver`
/// (`file://`, blake3-verified) — **no `DAEMON_TRAIN_MODULE` override**. Assess is eligible and join
/// drives a real round.
#[tokio::test]
async fn worker_resolves_module_from_signed_envelope() {
    // Ensure the module exists on disk for the artifact resolver (no env override is set).
    let _ = module_path();
    let mut cfg = TrainClientConfig::new(worker_bin());
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(60);
    let sup = TrainSupervisor::new(cfg);

    let elig = sup
        .assess(signed_envelope_wire())
        .await
        .expect("assess over the signed envelope");
    assert!(
        elig.eligible,
        "the worker resolved its module from the envelope + assessed eligible: {:?}",
        elig.reasons
    );
    sup.join("worker-seam", "wss://coord.example/swarm", vec![], policy())
        .await
        .expect("join after envelope-resolved assess");
    sup.shutdown().await;
}

/// RUN-9 (§10.5) over the **real** worker: preemption-as-churn. After a join, `Throttle{paused}`
/// pauses the `WasmBackend` (checkpoint + drop the wasm instance) and resume re-instantiates; a
/// subsequent join re-enters — all over the **same** worker (pause/resume is churn, never a respawn).
#[tokio::test]
async fn real_worker_preemption_pause_resume_rejoins_without_respawn() {
    let module = module_path();
    let mut cfg = TrainClientConfig::new(worker_bin());
    cfg.env = vec![(
        "DAEMON_TRAIN_MODULE".to_string(),
        module.to_string_lossy().into_owned(),
    )];
    cfg.spawn_timeout = Duration::from_secs(30);
    cfg.op_timeout = Duration::from_secs(60);
    let sup = TrainSupervisor::new(cfg);

    sup.assess(tiny_cfg_cbor()).await.expect("assess");
    sup.join("run-9", "wss://coord", vec![], policy())
        .await
        .expect("initial join");

    // Inference preempts training: pause frees the wasm instance, resume re-instantiates.
    sup.throttle(None, None, true).await.expect("pause");
    sup.throttle(None, None, false).await.expect("resume");
    // Rejoin at the next boundary — a fresh backend over the same worker process.
    sup.join("run-9", "wss://coord", vec![], policy())
        .await
        .expect("rejoin after resume");

    assert_eq!(
        sup.restarts().await,
        0,
        "pause/resume is churn over the same worker — never a respawn"
    );
    sup.shutdown().await;
}

// -- direct subprocess drive (observe the one-round RoundOutcome) --------------------------------

/// Drive the worker subprocess directly over the protocol and observe the full self-driven round:
/// `RunPhase{train}` → `Metric{loss}` → `RoundOutcome{round:0, digest}` (a 16-byte state digest).
#[tokio::test]
async fn worker_drives_one_round() {
    let module = module_path();
    let spec = PlacementSpec {
        program: PathBuf::from(worker_bin()),
        args: Vec::new(),
        env: vec![(
            "DAEMON_TRAIN_MODULE".to_string(),
            module.to_string_lossy().into_owned(),
        )],
    };
    let Placement { channel, mut child } = ProcessProvisioner::new()
        .place(&SessionId::new("daemon-train-worker-e3"), spec)
        .await
        .expect("spawn worker");
    let (writer, mut reader) = channel.split();

    // Ready handshake.
    assert!(
        matches!(read_event(&mut reader).await, Some(Event::Ready { .. })),
        "worker announces Ready first"
    );

    // Assess (caches the config for the join), then join and drive the round.
    send(
        &writer,
        &WCmd::AssessRun {
            envelope: tiny_cfg_cbor(),
        },
    )
    .await;
    assert!(
        matches!(read_event(&mut reader).await, Some(Event::Assessed(e)) if e.eligible),
        "assess is eligible"
    );

    send(
        &writer,
        &WCmd::JoinRun {
            run_id: "run-e3".to_string(),
            coordinator: "wss://coord.example/swarm".to_string(),
            credentials: Vec::new(),
            policy: policy(),
        },
    )
    .await;

    // Collect the round's event stream until RoundOutcome (or the worker dies).
    let mut saw_run_phase = false;
    let mut saw_metric = false;
    let mut outcome = None;
    for _ in 0..16 {
        match read_event(&mut reader).await {
            Some(Event::RunPhase { phase, .. }) => {
                assert_eq!(phase, "train");
                saw_run_phase = true;
            }
            Some(Event::Metric { name, .. }) if name == "loss" => saw_metric = true,
            Some(Event::RoundOutcome {
                round,
                digest,
                committed,
                ingested,
                ..
            }) => {
                assert_eq!(round, 0);
                assert_eq!((committed, ingested), (1, 1));
                outcome = Some(digest);
                break;
            }
            Some(Event::Error { class, detail }) => panic!("worker error ({class:?}): {detail}"),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    assert!(saw_run_phase, "worker emitted RunPhase{{train}}");
    assert!(saw_metric, "worker emitted a loss metric");
    let digest = outcome.expect("worker emitted a RoundOutcome");
    assert_eq!(
        digest.len(),
        16,
        "the round outcome carries a 16-byte state digest"
    );

    send(&writer, &WCmd::Shutdown).await;
    child.shutdown().await;
}

async fn send(writer: &daemon_provision::CutWriter, cmd: &WCmd) {
    let bytes = protocol::encode(cmd).expect("encode command");
    writer.send(&bytes).await.expect("send command");
}

async fn read_event(reader: &mut daemon_provision::CutReader) -> Option<Event> {
    let bytes = reader.recv().await?;
    Some(protocol::decode::<Event>(&bytes).expect("decode event"))
}
