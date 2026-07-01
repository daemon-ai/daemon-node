// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE TRACK D HOST-LAUNCH FAIL-FAST GATE: the `daemon` binary in its host role refuses to start
//! when no model provider is configured (or a networked provider has no credential), exiting
//! non-zero with a message that names the missing env key. This is the process-level counterpart to
//! the pure `validate_provider` unit tests in `bins/daemon/src/config.rs`: it proves the real
//! launch-boundary call (`cfg.validate_for_host()?` at the very top of `run_as_host`) rejects a
//! misconfigured launch before any store/socket/service setup — so it cannot hang. A bounded wait
//! guards the assertion regardless (a fail-fast launch exits at once).

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
    cmd.env("DAEMON_API_SOCKET", tmp.join("api.sock"));
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

/// No `DAEMON_MODEL_PROVIDER`: the host launch fails fast, naming the provider env key.
#[test]
fn unconfigured_provider_host_launch_fails_fast() {
    let (ok, stderr) = run_host_launch(&[], Duration::from_secs(20));
    assert!(
        !ok,
        "an unconfigured host launch must exit non-zero; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("DAEMON_MODEL_PROVIDER"),
        "the failure must name the provider env key; stderr:\n{stderr}"
    );
}

/// A networked provider (`daemon_api`) with a model but no credential also fails fast, naming the
/// credential env key.
#[test]
fn daemon_api_without_credential_host_launch_fails_fast() {
    let (ok, stderr) = run_host_launch(
        &[
            ("DAEMON_MODEL_PROVIDER", "daemon_api"),
            ("DAEMON_MODEL", "anthropic/claude-sonnet-4-5"),
        ],
        Duration::from_secs(20),
    );
    assert!(
        !ok,
        "a keyless networked host launch must exit non-zero; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("DAEMON_CREDENTIAL_KEY"),
        "the failure must name the credential env key; stderr:\n{stderr}"
    );
}
