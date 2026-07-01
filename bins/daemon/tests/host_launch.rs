// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE HOST-LAUNCH BOOT GATE: the `daemon` binary in its host role now BOOTS with no model provider
//! configured (no silent mock — the node installs `UnconfiguredProvider`, and a turn against an
//! unconfigured profile fails clearly at turn time). This is the process-level counterpart to the
//! pure `resolve_provider` unit tests in `bins/daemon/src/config.rs`: it proves the real
//! launch-boundary call (`cfg.resolve_for_host()?` at the top of `run_as_host`) no longer aborts a
//! bare launch — the node comes up and serves its socket. An *explicitly-set* networked provider
//! with no model is still a deliberate misconfiguration and fails fast. A bounded wait guards every
//! assertion.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Spawn the `daemon` binary in its **host** role (no placed-child / transport-server env, no
/// subcommand) with a clean, controlled environment plus `extra` overrides. Returns
/// `(exited_successfully, stderr)`. Bounded: kills the child if it does not exit within `timeout`
/// (validation is the first thing `run_as_host` does, so a real launch exits well under it).
fn run_host_launch(extra: &[(&str, &str)], timeout: Duration) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_daemon"));
    // A minimal, deterministic environment so a stray `DAEMON_*` from the test host cannot mask the
    // fail-fast. Keep `PATH` for any loader lookups; pin an ephemeral store + throwaway data dir so
    // nothing touches a real home even if validation were (wrongly) skipped.
    cmd.env_clear();
    if let Ok(p) = std::env::var("PATH") {
        cmd.env("PATH", p);
    }
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp =
        std::env::temp_dir().join(format!("daemon-host-launch-{}-{uniq}", std::process::id()));
    cmd.env("DAEMON_STORE", "memory");
    cmd.env("DAEMON_DATA_DIR", &tmp);
    cmd.env("DAEMON_SOCKET_PATH", tmp.join("api.sock"));
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn daemon binary");
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "daemon host launch did not exit within {timeout:?} — it must fail fast at \
                 validate_for_host"
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    // The fail-fast launch writes only a few log lines + the anyhow error to stderr (far under the
    // pipe buffer), so reading after exit cannot deadlock.
    let mut stderr = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut stderr);
    }
    (status.success(), stderr)
}

/// Spawn the host-role binary with `extra` overrides and a dedicated socket path; return
/// `(booted_and_serving, stderr_tail, child)`. "Booted and serving" = the process is still alive and
/// the Unix socket has appeared within `timeout` (it did not fail-fast exit). The caller kills the
/// child. Bounded: a fail-fast launch exits quickly and reports `booted = false`.
fn spawn_host_launch(
    extra: &[(&str, &str)],
    timeout: Duration,
) -> (bool, String, std::process::Child, std::path::PathBuf) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_daemon"));
    cmd.env_clear();
    if let Ok(p) = std::env::var("PATH") {
        cmd.env("PATH", p);
    }
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("daemon-host-boot-{}-{uniq}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let socket = tmp.join("api.sock");
    cmd.env("DAEMON_STORE", "memory");
    cmd.env("DAEMON_DATA_DIR", &tmp);
    cmd.env("DAEMON_SOCKET_PATH", &socket);
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn daemon binary");
    let deadline = Instant::now() + timeout;
    let mut booted = false;
    loop {
        if let Some(_status) = child.try_wait().expect("try_wait") {
            // The process exited before serving — a fail-fast (not a boot).
            break;
        }
        if socket.exists() {
            booted = true;
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // Read whatever is on stderr so far without blocking the still-running child: on boot we skip the
    // read (the pipe stays open) and rely on `booted`; on exit we drain it below.
    let stderr = if booted {
        String::new()
    } else {
        let mut s = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut s);
        }
        s
    };
    (booted, stderr, child, socket)
}

/// No `DAEMON_MODEL_PROVIDER`: the host launch now BOOTS and serves its socket (unconfigured — a turn
/// then fails clearly at turn time; never a silent mock at boot).
#[test]
fn unconfigured_provider_host_launch_boots_and_serves() {
    let (booted, stderr, mut child, _socket) = spawn_host_launch(&[], Duration::from_secs(20));
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        booted,
        "a bare (unconfigured) host launch must boot and serve; stderr:\n{stderr}"
    );
}

/// A networked provider (`daemon_api`) with a model but no credential also BOOTS now — credentials
/// are provisioned per-profile over the API (`CredentialSet`), not required at boot.
#[test]
fn daemon_api_without_credential_host_launch_boots() {
    let (booted, stderr, mut child, _socket) = spawn_host_launch(
        &[
            ("DAEMON_MODEL_PROVIDER", "daemon_api"),
            ("DAEMON_MODEL", "anthropic/claude-sonnet-4-5"),
        ],
        Duration::from_secs(20),
    );
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        booted,
        "a keyless networked host launch must boot (credential deferred to CredentialSet); stderr:\n{stderr}"
    );
}

/// An *explicitly-set* networked provider with no model is a deliberate misconfiguration and still
/// fails fast, naming the model env key.
#[test]
fn daemon_api_without_model_host_launch_fails_fast() {
    let (ok, stderr) = run_host_launch(
        &[("DAEMON_MODEL_PROVIDER", "daemon_api")],
        Duration::from_secs(20),
    );
    assert!(
        !ok,
        "an explicit networked provider with no model must exit non-zero; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("DAEMON_MODEL"),
        "the failure must name the model env key; stderr:\n{stderr}"
    );
}
