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
