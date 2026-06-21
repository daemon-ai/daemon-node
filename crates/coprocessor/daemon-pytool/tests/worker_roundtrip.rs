//! Integration test: drive the hermetic `fake-pytool-worker` over the supervised client.
//!
//! Spawns the compiled fake worker (via `CARGO_BIN_EXE_fake-pytool-worker`) through the same
//! `PyToolHost` the daemon uses, then exercises discovery, a `py_echo` round-trip, crash-respawn,
//! and the op-timeout watchdog + crash-loop meltdown — the Python-free mirror of the metta worker's
//! protocol test, so CI covers the client/proxy path without a system Python.

use std::time::Duration;

use daemon_pytool_client::{discover, PyToolConfig, PyToolError, PyToolHost};

fn worker_config(extra_args: &[&str]) -> PyToolConfig {
    let mut args: Vec<String> = Vec::new();
    args.extend(extra_args.iter().map(|s| s.to_string()));
    let mut cfg = PyToolConfig::new(env!("CARGO_BIN_EXE_fake-pytool-worker"), args);
    cfg.op_timeout = Duration::from_secs(5);
    cfg.spawn_timeout = Duration::from_secs(5);
    cfg
}

#[tokio::test]
async fn discovers_and_calls_py_echo() {
    let tools = discover(worker_config(&[])).await.expect("discover tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "py_echo");
    assert!(tools[0].schema().contains("text"));

    // Drive a call through the host (the proxy's run() needs a full TurnCx; the host call is the
    // same round-trip the proxy issues).
    let host = PyToolHost::new(worker_config(&[]));
    let reply = host
        .call_tool("c-1", "py_echo", r#"{"text":"hello"}"#, "s-1", 0)
        .await
        .expect("call py_echo");
    assert!(reply.ok);
    assert_eq!(reply.content, "hello");
    let detail = reply.detail.expect("echo detail");
    assert_eq!(detail.kind, "py_echo");
    assert_eq!(detail.body["echoed"], "hello");

    host.ping().await.expect("ping");
    host.shutdown().await;
}

#[tokio::test]
async fn respawns_after_a_worker_crash() {
    // This worker exits the process on the first CallTool; the call must fail transiently, and the
    // next op must transparently respawn a fresh worker.
    let host = PyToolHost::new(worker_config(&["--crash-on-call"]));
    host.discover().await.expect("initial discover");

    let err = host
        .call_tool("c-1", "py_echo", r#"{"text":"boom"}"#, "s-1", 0)
        .await
        .expect_err("call must fail when the worker exits");
    assert!(matches!(err, PyToolError::Transient(_)), "got {err:?}");

    // The worker is dead; ping respawns it (the crash only triggers on CallTool, so the fresh worker
    // answers the probe).
    host.ping().await.expect("ping after respawn");
    host.shutdown().await;
}

#[tokio::test]
async fn op_timeout_when_the_worker_hangs() {
    let mut cfg = worker_config(&["--hang-on-call"]);
    cfg.op_timeout = Duration::from_millis(300);
    let host = PyToolHost::new(cfg);
    host.discover().await.expect("discover");

    let err = host
        .call_tool("c-1", "py_echo", r#"{"text":"hi"}"#, "s-1", 0)
        .await
        .expect_err("a hung call must trip the watchdog");
    assert!(matches!(err, PyToolError::Transient(_)), "got {err:?}");
    host.shutdown().await;
}

#[tokio::test]
async fn crash_loop_trips_meltdown() {
    // A bogus worker binary makes every spawn fail; the supervisor must trip the crash-loop
    // meltdown to `Fatal` after `max_restarts` attempts within the window.
    let mut cfg = PyToolConfig::new("/nonexistent/fake-pytool-worker-binary", Vec::new());
    cfg.max_restarts = 2;
    cfg.restart_window = Duration::from_secs(60);
    let host = PyToolHost::new(cfg);

    for _ in 0..2 {
        let err = host
            .ping()
            .await
            .expect_err("spawn of a bogus binary must fail");
        assert!(matches!(err, PyToolError::Transient(_)), "got {err:?}");
    }
    let err = host.ping().await.expect_err("meltdown");
    assert!(matches!(err, PyToolError::Fatal(_)), "got {err:?}");
}
