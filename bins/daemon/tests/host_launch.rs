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
//!
//! It also gates the container/microvm launch contract: SIGTERM and SIGINT both trip the graceful
//! shutdown (exit 0, shutdown log line, socket removed).

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn configure_base_env(cmd: &mut Command) {
    cmd.env_clear();
    if let Ok(p) = std::env::var("PATH") {
        cmd.env("PATH", p);
    }
    forward_tls_trust_store(cmd);
}

/// Forward system TLS roots into the controlled child environment, but never preserve Nix's
/// sandbox placeholder (`/no-cert-file.crt`) or any other stale path.
fn forward_tls_trust_store(cmd: &mut Command) {
    for var in ["SSL_CERT_FILE", "NIX_SSL_CERT_FILE"] {
        forward_existing_file_env(cmd, var);
    }
    forward_existing_dir_env(cmd, "SSL_CERT_DIR");
}

fn forward_existing_file_env(cmd: &mut Command, var: &str) {
    if let Ok(value) = std::env::var(var) {
        if Path::new(&value).is_file() {
            cmd.env(var, value);
        }
    }
}

fn forward_existing_dir_env(cmd: &mut Command, var: &str) {
    if let Ok(value) = std::env::var(var) {
        if Path::new(&value).is_dir() {
            cmd.env(var, value);
        }
    }
}

/// A unique throwaway path under the system temp dir (never a real home).
fn unique_tmp(prefix: &str) -> std::path::PathBuf {
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{prefix}-{}-{uniq}", std::process::id()))
}

/// Build the host-role `Command`: a minimal, deterministic environment (so a stray `DAEMON_*` from
/// the test host cannot mask a fail-fast), an ephemeral store + throwaway data dir, and `socket`
/// as the api socket path, plus `extra` overrides.
fn host_command(tmp: &Path, socket: &Path, extra: &[(&str, &str)]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_daemon"));
    configure_base_env(&mut cmd);
    cmd.env("DAEMON_STORE", "memory");
    cmd.env("DAEMON_DATA_DIR", tmp);
    cmd.env("DAEMON_SOCKET_PATH", socket);
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd
}

/// Spawn the `daemon` binary in its **host** role (no placed-child / transport-server env, no
/// subcommand) with a clean, controlled environment plus `extra` overrides. Returns
/// `(exited_successfully, stderr)`. Bounded: kills the child if it does not exit within `timeout`
/// (validation is the first thing `run_as_host` does, so a real launch exits well under it).
fn run_host_launch(extra: &[(&str, &str)], timeout: Duration) -> (bool, String) {
    let tmp = unique_tmp("daemon-host-launch");
    let mut child = host_command(&tmp, &tmp.join("api.sock"), extra)
        .spawn()
        .expect("spawn daemon binary");
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
    let tmp = unique_tmp("daemon-host-boot");
    let _ = std::fs::create_dir_all(&tmp);
    let socket = tmp.join("api.sock");
    let mut child = host_command(&tmp, &socket, extra)
        .spawn()
        .expect("spawn daemon binary");
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

/// Wait (bounded) for `child` to exit; panics — after killing it — if `timeout` elapses first.
fn wait_bounded(child: &mut std::process::Child, timeout: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("daemon did not exit within {timeout:?} after the shutdown signal");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Boot the host role, deliver `signal` (raw kill(2), exactly like a container runtime or an
/// interactive ^C), and assert the graceful-shutdown contract: a prompt exit 0, the shutdown log
/// line naming the signal, and the api socket removed on the way out.
fn assert_graceful_shutdown_on(signal: nix::sys::signal::Signal, name: &str) {
    let (booted, stderr, mut child, socket) = spawn_host_launch(&[], Duration::from_secs(20));
    assert!(
        booted,
        "the host launch must boot before the signal; stderr:\n{stderr}"
    );

    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(child.id().try_into().expect("pid fits i32")),
        signal,
    )
    .expect("deliver the shutdown signal");
    let status = wait_bounded(&mut child, Duration::from_secs(20));

    // The child has exited, so draining the (small, far under the pipe buffer) stderr is safe.
    let mut stderr = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut stderr);
    }
    assert!(
        status.success(),
        "{name} must trip a graceful (exit 0) shutdown, got {status:?}; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("shutdown signal received") && stderr.contains(name),
        "the shutdown log line must name {name}; stderr:\n{stderr}"
    );
    assert!(
        !socket.exists(),
        "a graceful shutdown must remove the api socket"
    );
}

/// SIGTERM — what container runtimes (`docker stop`, Fly Machines, systemd) send first — must trip
/// the same graceful shutdown as SIGINT instead of running into the stop timeout + SIGKILL.
#[test]
fn sigterm_host_launch_shuts_down_gracefully() {
    assert_graceful_shutdown_on(nix::sys::signal::Signal::SIGTERM, "SIGTERM");
}

/// SIGINT (`ctrl_c`) — the other arm of the shutdown select — keeps the prior behavior.
#[test]
fn sigint_host_launch_shuts_down_gracefully() {
    assert_graceful_shutdown_on(nix::sys::signal::Signal::SIGINT, "SIGINT");
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

/// A LIVE listener already bound on the target socket: the launch must fail fast with the
/// bind-safety error instead of unlinking the socket out from under the serving daemon (the
/// orphaned-daemon incident: 7 leaked processes from 4 builds).
#[test]
fn live_socket_occupied_host_launch_fails_fast() {
    let tmp = unique_tmp("daemon-host-occupied");
    std::fs::create_dir_all(&tmp).expect("create socket dir");
    let socket = tmp.join("api.sock");
    let _live = std::os::unix::net::UnixListener::bind(&socket).expect("bind live listener");

    let (ok, stderr) = run_host_launch(
        &[("DAEMON_SOCKET_PATH", socket.to_str().expect("utf8 path"))],
        Duration::from_secs(20),
    );
    assert!(
        !ok,
        "a launch against a live socket must exit non-zero; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("already bound by a live daemon"),
        "the failure must name the bind-safety error; stderr:\n{stderr}"
    );
    assert!(socket.exists(), "the live socket must be left in place");
}

/// A stale socket FILE with no listener (a previous daemon that died without cleanup) must not
/// block a fresh launch: the node clears it and boots. Boot signal is a successful connect — the
/// file exists from the start, so file-existence polling would prove nothing.
#[test]
fn stale_socket_file_host_launch_boots_and_serves() {
    let tmp = unique_tmp("daemon-host-stale");
    std::fs::create_dir_all(&tmp).expect("create socket dir");
    let socket = tmp.join("api.sock");
    drop(std::os::unix::net::UnixListener::bind(&socket).expect("bind then abandon"));
    assert!(socket.exists(), "the stale socket file must pre-exist");

    let mut child = host_command(&tmp, &socket, &[])
        .spawn()
        .expect("spawn daemon binary");
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut serving = false;
    let mut exited = false;
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            exited = true;
            break;
        }
        if std::os::unix::net::UnixStream::connect(&socket).is_ok() {
            serving = true;
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut stderr = String::new();
    if exited {
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut stderr);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        serving,
        "a launch over a stale socket file must clear it and serve; stderr:\n{stderr}"
    );
}
