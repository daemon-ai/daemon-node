// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Phase 4: integration test crate; raw fs/reqwest/Command are expected in tests.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

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
//! shutdown (exit 0, shutdown log line, socket removed), a HOME-less environment still boots, and
//! a nonexistent `DAEMON_DATA_DIR` is created (private) rather than aborting the sqlite store open.

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

/// Poll until `child` serves `socket` (true) or it exits / `timeout` elapses first (false).
fn wait_until_serving(child: &mut std::process::Child, socket: &Path, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            return false; // exited before serving — a fail-fast, not a boot
        }
        if socket.exists() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Drain whatever the (exited or killed) child wrote to its piped stderr.
fn drain_stderr(child: &mut std::process::Child) -> String {
    let mut s = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut s);
    }
    s
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

/// A HOME-less environment (a container/microvm: no `HOME`, no `HF_*`/XDG variables) must still
/// boot to ready — the model subsystem used to panic here inside hf-hub's eager home-directory
/// probe (`ApiBuilder::new` → `Cache::default`), even with local inference unconfigured. The
/// controlled base env (`env_clear`) already omits `HOME`; the explicit `env_remove` locks that in
/// if the base env ever grows. (On dev machines the panic also needed a uid without a passwd
/// entry — hf-hub falls back to getpwuid_r — so this gate guards the env-probe path; the fix
/// removes the probe from the boot path entirely.)
#[test]
fn homeless_env_host_launch_boots_and_serves() {
    let tmp = unique_tmp("daemon-host-homeless");
    let _ = std::fs::create_dir_all(&tmp);
    let socket = tmp.join("api.sock");
    let mut cmd = host_command(&tmp, &socket, &[]);
    cmd.env_remove("HOME");
    let mut child = cmd.spawn().expect("spawn daemon binary");

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut booted = false;
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            break; // exited before serving — a fail-fast (or the old panic), not a boot
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
    let mut stderr = String::new();
    if !booted {
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut stderr);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        booted,
        "a HOME-less host launch must boot and serve; stderr:\n{stderr}"
    );
}

/// `DAEMON_DATA_DIR` pointing at a nonexistent nested path must boot: the host role creates the
/// data directory (recursively) before anything under it is opened. `DAEMON_STORE=sqlite`
/// (overriding the harness memory default) reproduces the reported failure shape — the store file
/// is the data dir's first boot-time consumer ("opening sqlite store …: unable to open database
/// file"). On unix the created leaf is private (0700): auth.sqlite + journal seeds live inside.
#[test]
fn nonexistent_data_dir_host_launch_creates_it_and_boots() {
    let root = unique_tmp("daemon-host-datadir");
    let data_dir = root.join("a").join("b").join("data");
    assert!(!data_dir.exists(), "the data dir must start nonexistent");
    let socket = root.join("api.sock");
    let mut child = host_command(&data_dir, &socket, &[("DAEMON_STORE", "sqlite")])
        .spawn()
        .expect("spawn daemon binary");
    let booted = wait_until_serving(&mut child, &socket, Duration::from_secs(20));
    let _ = child.kill();
    let _ = child.wait();
    let stderr = drain_stderr(&mut child);
    assert!(
        booted,
        "a nonexistent data dir must be created at boot, not abort it; stderr:\n{stderr}"
    );
    assert!(data_dir.is_dir(), "the data dir must exist after boot");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&data_dir)
            .expect("stat the created data dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700, "a daemon-created data dir must be private");
    }
}

/// A data dir that ALREADY exists is left completely untouched: the creation path never chmods an
/// existing directory, so operator-managed permissions (here: a deliberately non-default 0755)
/// survive a boot. (Every other test in this file already boots over a pre-created data dir and
/// would catch an existing-dir regression; this one pins the permission bits explicitly.)
#[cfg(unix)]
#[test]
fn preexisting_data_dir_permissions_survive_boot() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = unique_tmp("daemon-host-datadir-keep");
    std::fs::create_dir_all(&tmp).expect("pre-create the data dir");
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755)).expect("set 0755");
    let socket = tmp.join("api.sock");
    let mut child = host_command(&tmp, &socket, &[("DAEMON_STORE", "sqlite")])
        .spawn()
        .expect("spawn daemon binary");
    let booted = wait_until_serving(&mut child, &socket, Duration::from_secs(20));
    let _ = child.kill();
    let _ = child.wait();
    let stderr = drain_stderr(&mut child);
    assert!(
        booted,
        "an existing data dir must keep booting; stderr:\n{stderr}"
    );
    let mode = std::fs::metadata(&tmp)
        .expect("stat the pre-existing data dir")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o755,
        "boot must not clobber an existing data dir's permissions"
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

// --- D2: Daemon Cloud attach-credential + default-profile seeding --------------------------------

/// Boot the host role over an explicit `data_dir` (so the test can inspect the persisted
/// `credentials.json` / `profiles/` afterwards), with `DAEMON_STORE=sqlite` forced so the durable
/// file-backed stores are used. Returns `(booted_and_serving, child, socket)`; the caller stops it.
fn spawn_over_data_dir(
    data_dir: &Path,
    extra: &[(&str, &str)],
    timeout: Duration,
) -> (bool, std::process::Child, std::path::PathBuf) {
    let _ = std::fs::create_dir_all(data_dir);
    let socket = data_dir.join("api.sock");
    // `DAEMON_STORE=sqlite` (via `extra`, applied after the harness memory default) makes the node
    // durable, so `credentials.json` + `profiles/` are written under the data dir. The seed runs
    // before the socket is bound, so "serving" is a safe signal that seeding completed.
    let mut merged: Vec<(&str, &str)> = vec![("DAEMON_STORE", "sqlite")];
    merged.extend_from_slice(extra);
    let mut child = host_command(data_dir, &socket, &merged)
        .spawn()
        .expect("spawn daemon binary");
    let booted = wait_until_serving(&mut child, &socket, timeout);
    (booted, child, socket)
}

/// SIGTERM the child and wait (bounded) for its graceful exit — the clean way to release the sqlite
/// stores between two boots over the same data dir.
fn graceful_stop(child: &mut std::process::Child) {
    let pid = nix::unistd::Pid::from_raw(child.id().try_into().expect("pid fits i32"));
    let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
    let _ = wait_bounded(child, Duration::from_secs(20));
}

/// `DAEMON_CLOUD_API_KEY` on a fresh (durable) node seeds the credential-store entry for the node's
/// profile AND lands a default profile that selects the `daemon_api` ("Daemon Cloud") provider — the
/// zero-GUI attach wiring (D2). The key is never written to stderr.
#[test]
fn cloud_api_key_seeds_credential_and_daemon_api_profile() {
    let data_dir = unique_tmp("daemon-d2-seed");
    let key = "sk-daemon-cloud-seed-test";
    let (booted, mut child, _socket) = spawn_over_data_dir(
        &data_dir,
        &[("DAEMON_PROFILE", "hosted"), ("DAEMON_CLOUD_API_KEY", key)],
        Duration::from_secs(20),
    );
    assert!(
        booted,
        "the node must boot and serve with the cloud key set"
    );
    graceful_stop(&mut child);
    let stderr = drain_stderr(&mut child);

    // The credential store maps the node's profile id to the injected key.
    let creds = std::fs::read_to_string(data_dir.join("credentials.json"))
        .expect("credentials.json must exist on a durable node");
    assert!(
        creds.contains("hosted") && creds.contains(key),
        "credentials.json must hold the attach key under the profile; got: {creds}"
    );
    // The seeded default profile selects the daemon_api provider.
    let profile = std::fs::read_to_string(data_dir.join("profiles").join("hosted.json"))
        .expect("the seeded profile must be persisted");
    assert!(
        profile.contains("\"daemon_api\""),
        "the seeded default profile must select daemon_api; got: {profile}"
    );
    // The secret must NEVER reach the log/stderr surface.
    assert!(
        !stderr.contains(key),
        "the attach key must never be logged; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

/// A rotated secret + restart re-seeds in place (create-or-update): the new key overwrites the old,
/// never a duplicate. This is the §9.4 attach-key rotation flow (control plane updates the secret,
/// restarts the node).
#[test]
fn cloud_api_key_rotation_overwrites_in_place() {
    let data_dir = unique_tmp("daemon-d2-rotate");
    let key_a = "sk-attach-rotation-AAA";
    let key_b = "sk-attach-rotation-BBB";

    let (booted, mut child, _s) = spawn_over_data_dir(
        &data_dir,
        &[
            ("DAEMON_PROFILE", "hosted"),
            ("DAEMON_CLOUD_API_KEY", key_a),
        ],
        Duration::from_secs(20),
    );
    assert!(booted, "first boot (key A) must serve");
    graceful_stop(&mut child);

    let (booted, mut child, _s) = spawn_over_data_dir(
        &data_dir,
        &[
            ("DAEMON_PROFILE", "hosted"),
            ("DAEMON_CLOUD_API_KEY", key_b),
        ],
        Duration::from_secs(20),
    );
    assert!(booted, "second boot (rotated key B) must serve");
    graceful_stop(&mut child);

    let creds = std::fs::read_to_string(data_dir.join("credentials.json"))
        .expect("credentials.json must exist");
    assert!(
        creds.contains(key_b),
        "the rotated key must be seeded; got: {creds}"
    );
    assert!(
        !creds.contains(key_a),
        "the old key must be overwritten in place (no duplicate); got: {creds}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

/// The `DAEMON_CLOUD_API_KEY_FILE` variant seeds from the (trimmed) file contents — the secret-file
/// injection shape (mirrors `DAEMON_ADMIN_PASSWORD_FILE`).
#[test]
fn cloud_api_key_file_variant_seeds() {
    let data_dir = unique_tmp("daemon-d2-file");
    let _ = std::fs::create_dir_all(&data_dir);
    let key = "sk-attach-from-file";
    let key_path = data_dir.join("attach-key.txt");
    std::fs::write(&key_path, format!("{key}\n")).expect("write key file");

    let (booted, mut child, _s) = spawn_over_data_dir(
        &data_dir,
        &[
            ("DAEMON_PROFILE", "hosted"),
            (
                "DAEMON_CLOUD_API_KEY_FILE",
                key_path.to_str().expect("utf8 path"),
            ),
        ],
        Duration::from_secs(20),
    );
    assert!(booted, "boot with the _FILE variant must serve");
    graceful_stop(&mut child);

    let creds = std::fs::read_to_string(data_dir.join("credentials.json"))
        .expect("credentials.json must exist");
    assert!(
        creds.contains(key),
        "the _FILE key must be seeded (trimmed); got: {creds}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
}

/// Unsetting the secret must NEVER scrub an already-seeded credential (seed-if-present, mirroring
/// the first-admin seed): a reboot with no cloud env leaves the prior credential in place.
#[test]
fn cloud_api_key_unset_does_not_scrub() {
    let data_dir = unique_tmp("daemon-d2-unset");
    let key = "sk-attach-persist-test";

    let (booted, mut child, _s) = spawn_over_data_dir(
        &data_dir,
        &[("DAEMON_PROFILE", "hosted"), ("DAEMON_CLOUD_API_KEY", key)],
        Duration::from_secs(20),
    );
    assert!(booted, "first boot (key set) must serve");
    graceful_stop(&mut child);

    // Reboot with the cloud env removed entirely (host_command's env_clear omits it already).
    let (booted, mut child, _s) = spawn_over_data_dir(
        &data_dir,
        &[("DAEMON_PROFILE", "hosted")],
        Duration::from_secs(20),
    );
    assert!(booted, "second boot (key unset) must serve");
    graceful_stop(&mut child);

    let creds = std::fs::read_to_string(data_dir.join("credentials.json"))
        .expect("credentials.json must still exist");
    assert!(
        creds.contains(key),
        "unset must not scrub the seeded credential; got: {creds}"
    );

    let _ = std::fs::remove_dir_all(&data_dir);
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
