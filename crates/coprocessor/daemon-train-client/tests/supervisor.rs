// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Integration harness: a per-test scratch state file drives the fake worker's spawn-index
// scenarios. That scratch fs is test-only and daemon-controlled (mirrors `fake-infer-worker`).
#![allow(clippy::disallowed_methods)]

//! `TrainSupervisor` supervision tests over the scripted `fake-train-worker` binary (CLI-2).
//!
//! CLI-3 (crash-loop meltdown over a bogus binary) is a unit test in the library; here we exercise
//! the real spawn → handshake → command path and the respawn-after-crash flow against a fixture
//! worker that speaks the true `daemon_swarm_run::protocol`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use daemon_swarm_run::protocol::{JoinPolicy, PolicyMode};
use daemon_train_client::{TrainClientConfig, TrainClientError, TrainSupervisor};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique scratch path for the fake worker's spawn counter (`DAEMON_FAKE_STATE`).
fn state_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "daemon-train-client-{tag}-{pid}-{n}-{nanos}.state",
        pid = std::process::id()
    ))
}

fn cfg(scenario: &str, state: &Path) -> TrainClientConfig {
    let mut c = TrainClientConfig::new(env!("CARGO_BIN_EXE_fake-train-worker"));
    c.env = vec![
        ("DAEMON_FAKE_SCENARIO".into(), scenario.into()),
        (
            "DAEMON_FAKE_STATE".into(),
            state.to_string_lossy().into_owned(),
        ),
    ];
    c.spawn_timeout = Duration::from_secs(10);
    c.op_timeout = Duration::from_secs(10);
    c.respawn_backoff = Duration::from_millis(10);
    c
}

fn policy() -> JoinPolicy {
    JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// The full spawn → Ready → command round trip against a healthy fixture worker.
#[tokio::test]
async fn supervisor_probe_assess_ping() {
    let state = state_path("happy");
    let sup = TrainSupervisor::new(cfg("ready", &state));

    let hw = sup.probe().await.expect("probe");
    assert_eq!(hw.gpus, 1);
    assert_eq!(hw.throughput_class, "c3");

    let elig = sup.assess(vec![1, 2, 3]).await.expect("assess");
    assert!(elig.eligible);

    sup.ping().await.expect("ping");
    sup.shutdown().await;
    let _ = std::fs::remove_file(&state);
}

/// A worker that crashes on its first `JoinRun` (spawn index 0) then behaves on the respawn: the
/// first join surfaces a transient fault, and a subsequent join respawns a fresh worker and
/// succeeds (CLI-2).
#[tokio::test]
async fn supervisor_respawn() {
    let state = state_path("respawn");
    let sup = TrainSupervisor::new(cfg("crash-once", &state));

    let first = sup.join("run-1", "wss://coord", vec![], policy()).await;
    assert!(
        matches!(first, Err(TrainClientError::Transient(_))),
        "first join should see the worker crash: {first:?}"
    );

    sup.join("run-1", "wss://coord", vec![], policy())
        .await
        .expect("respawn join should succeed");
    assert!(sup.restarts().await >= 1, "the worker was respawned");

    sup.shutdown().await;
    let _ = std::fs::remove_file(&state);
}

/// RUN-9 (§10.5): preemption-as-churn. `Throttle{paused}` makes the peer leave the round cleanly;
/// resume + rejoin re-enter at a boundary — over the **same** worker (pause/resume is churn, not a
/// crash, so there is no respawn). This fixture-worker test pins the supervision semantics (no
/// respawn on pause/resume); the **real** `daemon-train-worker` preemption (wasm VRAM-free
/// pause/resume via `WasmBackend`) is exercised in `daemon-train/tests/worker_protocol.rs`
/// (`daemon-train-client` cannot depend on `daemon-train` — that would be a dependency cycle).
#[tokio::test]
async fn preemption_as_churn_pauses_and_rejoins_without_respawn() {
    let state = state_path("preempt");
    let sup = TrainSupervisor::new(cfg("ready", &state));

    sup.join("run-9", "wss://coord", vec![], policy())
        .await
        .expect("initial join");

    // Inference preempts training: pause (the worker leaves the round + frees VRAM on the real side).
    sup.throttle(None, None, true).await.expect("pause");
    // Resume per policy, then rejoin at the next boundary.
    sup.throttle(None, None, false).await.expect("resume");
    sup.join("run-9", "wss://coord", vec![], policy())
        .await
        .expect("rejoin after resume");

    assert_eq!(
        sup.restarts().await,
        0,
        "pause/resume is churn over the same worker — never a respawn"
    );

    sup.shutdown().await;
    let _ = std::fs::remove_file(&state);
}

/// CLI-4 (§10.5): `Throttle{paused}` — the supervisor half. Pausing while joined delivers the
/// governor lever as a fire-and-forget oneway and does **not** tear the worker down (the abort of
/// the in-flight guest call is graceful, worker-side), so the same worker keeps serving afterwards.
/// The real VRAM-free + CPU-master-retention on the wasm backend is `daemon-train`'s
/// `worker_protocol.rs`; here we pin that pause is churn over one worker, never a respawn.
#[tokio::test]
async fn throttle_aborts_in_flight_call() {
    let state = state_path("throttle-abort");
    let sup = TrainSupervisor::new(cfg("ready", &state));

    sup.join("run-4", "wss://coord", vec![], policy())
        .await
        .expect("join");
    // Pause aborts the in-flight round on the worker side; the supervisor's oneway must succeed and
    // must not classify it as a fault (no worker teardown).
    sup.throttle(None, None, true).await.expect("pause abort");
    // The same worker still answers — the abort did not crash or replace it.
    sup.ping().await.expect("worker survives the abort");
    assert_eq!(
        sup.restarts().await,
        0,
        "pause is a graceful abort, not a respawn"
    );

    sup.shutdown().await;
    let _ = std::fs::remove_file(&state);
}

/// CLI-4 (§10.5): `Throttle{paused}` frees VRAM on the worker but retains the CPU masters — the
/// supervisor half is that resume + a subsequent command land on the **same** worker (never a fresh
/// spawn), which is what "masters retained" means at this layer.
#[tokio::test]
async fn throttle_frees_vram_keeps_masters() {
    let state = state_path("throttle-masters");
    let sup = TrainSupervisor::new(cfg("ready", &state));

    sup.join("run-4b", "wss://coord", vec![], policy())
        .await
        .expect("join");
    sup.throttle(None, None, true)
        .await
        .expect("pause (free VRAM)");
    sup.throttle(None, None, false)
        .await
        .expect("resume (rebuild)");
    // Resume + rejoin reuse the same worker process → the CPU masters were never discarded.
    sup.join("run-4b", "wss://coord", vec![], policy())
        .await
        .expect("rejoin at boundary");
    assert_eq!(
        sup.restarts().await,
        0,
        "pause/resume frees VRAM but keeps the worker (and its masters) — no respawn"
    );

    sup.shutdown().await;
    let _ = std::fs::remove_file(&state);
}

/// RUN-10 (§6.5): assess staging. `AssessRun` against an envelope stages an eligibility verdict over
/// the worker protocol — both the eligible and the pre-screen-rejected paths.
#[tokio::test]
async fn assess_staging_returns_eligibility() {
    let ok_state = state_path("assess-ok");
    let sup = TrainSupervisor::new(cfg("ready", &ok_state));
    let elig = sup
        .assess(b"frozen-envelope-bytes".to_vec())
        .await
        .expect("assess (eligible)");
    assert!(elig.eligible, "the fits-fake reports eligible");
    assert!(!elig.headroom.is_empty(), "eligibility carries headroom");
    sup.shutdown().await;
    let _ = std::fs::remove_file(&ok_state);

    let no_state = state_path("assess-no");
    let sup = TrainSupervisor::new(cfg("ineligible", &no_state));
    let elig = sup
        .assess(b"frozen-envelope-bytes".to_vec())
        .await
        .expect("assess (ineligible)");
    assert!(!elig.eligible, "the ineligible fake declines the run");
    assert!(
        !elig.reasons.is_empty(),
        "an ineligible assessment carries why-not reasons"
    );
    sup.shutdown().await;
    let _ = std::fs::remove_file(&no_state);
}
