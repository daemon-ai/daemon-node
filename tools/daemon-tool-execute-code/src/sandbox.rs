// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! OS sandbox selection + argv construction for execute_code.
//!
//! The child can run under a per-platform kernel sandbox behind the one [`SandboxPolicy`]:
//! - **Linux:** [bubblewrap](https://github.com/containers/bubblewrap) namespaces when usable, else
//!   an in-process **Landlock + seccomp** confinement (via [`daemon_sandbox`]) — the case that
//!   previously degraded straight to an unconfined `Plain` subprocess when user namespaces were off.
//! - **macOS:** a Seatbelt profile via `sandbox-exec` (an argv wrapper).
//!
//! [`resolve`] maps the posture + probed host capabilities to a concrete [`SandboxKind`]; `Require`
//! fails closed when no backend is usable, `Auto` degrades to `Plain` (with a warning), and `Plain`
//! is the explicit unconfined choice. [`argv`] builds the program+args for the argv-wrapper backends;
//! the Landlock backend is applied in-process at spawn (see `exec::run_subprocess`).

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use daemon_sandbox::{Backend, Capabilities, Posture};

use crate::{NetworkPolicy, SandboxPolicy};

/// The chosen execution backend for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SandboxKind {
    /// Run under a bubblewrap namespace sandbox (Linux; argv wrapper).
    Bwrap,
    /// Run under an in-process Landlock (fs) + seccomp (network) sandbox (Linux; applied at spawn).
    Landlock,
    /// Run under a macOS Seatbelt profile via `sandbox-exec` (argv wrapper).
    SandboxExec,
    /// Run as a plain subprocess (no OS jail).
    Plain,
}

impl SandboxKind {
    /// The operator-visible backend label for the result detail envelope.
    pub(crate) fn label(self) -> &'static str {
        match self {
            SandboxKind::Bwrap => "bwrap",
            SandboxKind::Landlock => "landlock",
            SandboxKind::SandboxExec => "sandbox-exec",
            SandboxKind::Plain => "plain",
        }
    }

    /// Whether this backend applies any OS confinement (everything but [`SandboxKind::Plain`]).
    pub(crate) fn is_confined(self) -> bool {
        !matches!(self, SandboxKind::Plain)
    }
}

/// Read-only system paths a child needs to run an interpreter (shared by the bwrap ro-binds and the
/// Landlock ro scope). `-try`/existence-filtered so a non-Nix or minimal host still works.
const SYSTEM_RO_PATHS: &[&str] = &[
    "/nix/store",
    "/usr",
    "/bin",
    "/lib",
    "/lib64",
    "/run/current-system/sw",
    "/etc/resolv.conf",
    "/etc/ssl",
    "/etc/pki",
    "/etc/nsswitch.conf",
    "/etc/static",
];

/// The bwrap capability-probe timeout.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolve the effective backend for `policy` from the host's probed capabilities.
///
/// `Auto` picks the strongest usable backend, else `Plain`; `Require` errors when none is usable
/// (fail-closed); `Plain` is always `Plain`. The bwrap/`sandbox-exec` argv-wrapper availability is
/// probed here; the in-process Landlock backend is probed by [`daemon_sandbox`].
pub(crate) async fn resolve(policy: SandboxPolicy) -> std::io::Result<SandboxKind> {
    let posture = match policy {
        SandboxPolicy::Auto => Posture::Auto,
        SandboxPolicy::Require => Posture::Require,
        SandboxPolicy::Plain => Posture::Plain,
    };
    let caps = Capabilities {
        bwrap: bwrap_usable().await,
        landlock_seccomp: daemon_sandbox::landlock_seccomp_available(),
        sandbox_exec: cfg!(target_os = "macos"),
    };
    let kind = match daemon_sandbox::decide(posture, &caps)? {
        Backend::Bwrap => SandboxKind::Bwrap,
        Backend::LandlockSeccomp => SandboxKind::Landlock,
        Backend::SandboxExec => SandboxKind::SandboxExec,
        Backend::Plain => SandboxKind::Plain,
    };
    // Surface the degraded posture: `Auto` chose no confinement because none was usable.
    if policy == SandboxPolicy::Auto && kind == SandboxKind::Plain {
        warn_unconfined_once();
    }
    Ok(kind)
}

/// The in-process Landlock/seccomp scope for a run: read/write the working dir, read+execute the
/// interpreter + system library paths, and deny INET/INET6 sockets unless network is `Shared`.
pub(crate) fn landlock_spec(
    cwd: &Path,
    interpreter: &Path,
    network: NetworkPolicy,
) -> daemon_sandbox::SandboxSpec {
    let mut ro: Vec<PathBuf> = SYSTEM_RO_PATHS.iter().map(PathBuf::from).collect();
    // Landlock cannot synthesize a private /proc or /dev (that is bwrap's advantage); grant read on
    // the real ones so the interpreter can read /dev/urandom, /proc/self, etc.
    ro.push(PathBuf::from("/proc"));
    ro.push(PathBuf::from("/dev"));
    if let Some(dir) = interpreter.parent() {
        ro.push(dir.to_path_buf());
    }
    daemon_sandbox::SandboxSpec {
        rw_paths: vec![cwd.to_path_buf()],
        ro_paths: ro,
        allow_network: network == NetworkPolicy::Shared,
    }
}

/// Build the full argv (program + args) for a run. `Plain`/`Landlock` are `[interpreter, script]`
/// (Landlock confinement is applied in-process at spawn, not via a wrapper); `Bwrap` wraps that in a
/// namespace sandbox and `SandboxExec` in a macOS Seatbelt profile.
pub(crate) fn argv(
    kind: SandboxKind,
    network: NetworkPolicy,
    cwd: &Path,
    interpreter: &Path,
    script: &Path,
    path_env: &OsStr,
    tz: Option<&str>,
) -> Vec<OsString> {
    let interp = interpreter.as_os_str().to_os_string();
    let script = script.as_os_str().to_os_string();
    match kind {
        SandboxKind::Plain | SandboxKind::Landlock => vec![interp, script],
        SandboxKind::Bwrap => bwrap_argv(network, cwd, interp, script, path_env, tz),
        SandboxKind::SandboxExec => sandbox_exec_argv(network, cwd, interp, script),
    }
}

/// Build the bwrap namespace-sandbox argv rooted at `cwd`.
fn bwrap_argv(
    network: NetworkPolicy,
    cwd: &Path,
    interp: OsString,
    script: OsString,
    path_env: &OsStr,
    tz: Option<&str>,
) -> Vec<OsString> {
    let cwd_os = cwd.as_os_str().to_os_string();
    let mut a: Vec<OsString> = Vec::new();
    a.push(OsString::from("bwrap"));
    for flag in [
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
    ] {
        a.push(OsString::from(flag));
    }
    if network == NetworkPolicy::Off {
        a.push(OsString::from("--unshare-net"));
    }
    // Read-only system paths (`-try` tolerates absence so a non-Nix or minimal host still works).
    for path in SYSTEM_RO_PATHS {
        a.push(OsString::from("--ro-bind-try"));
        a.push(OsString::from(*path));
        a.push(OsString::from(*path));
    }
    a.push(OsString::from("--proc"));
    a.push(OsString::from("/proc"));
    a.push(OsString::from("--dev"));
    a.push(OsString::from("/dev"));
    a.push(OsString::from("--tmpfs"));
    a.push(OsString::from("/tmp"));
    // The only writable bind: the working directory (workspace root in project mode, staging in
    // strict mode). Its parent is not bound, so a write to `../x` cannot escape.
    a.push(OsString::from("--bind"));
    a.push(cwd_os.clone());
    a.push(cwd_os.clone());
    a.push(OsString::from("--chdir"));
    a.push(cwd_os);
    // Child env: cleared, then only PATH + PYTHONDONTWRITEBYTECODE (+ TZ) — mirrors the plain path.
    a.push(OsString::from("--clearenv"));
    a.push(OsString::from("--setenv"));
    a.push(OsString::from("PATH"));
    a.push(path_env.to_os_string());
    a.push(OsString::from("--setenv"));
    a.push(OsString::from("PYTHONDONTWRITEBYTECODE"));
    a.push(OsString::from("1"));
    if let Some(tz) = tz {
        a.push(OsString::from("--setenv"));
        a.push(OsString::from("TZ"));
        a.push(OsString::from(tz));
    }
    a.push(OsString::from("--"));
    a.push(interp);
    a.push(script);
    a
}

/// Build the macOS `sandbox-exec` argv: a deny-default Seatbelt profile allowing read of system
/// paths, read+write of the working dir, and (unless network is `Shared`) denying outbound network.
fn sandbox_exec_argv(
    network: NetworkPolicy,
    cwd: &Path,
    interp: OsString,
    script: OsString,
) -> Vec<OsString> {
    let net = if network == NetworkPolicy::Off {
        "(deny network*)"
    } else {
        "(allow network*)"
    };
    // SBPL: deny by default, allow process exec + system reads, read anywhere, write only under the
    // workdir (default-deny means no explicit deny of `/` is needed).
    let profile = format!(
        "(version 1)\n(deny default)\n(allow process-exec)(allow process-fork)(allow sysctl-read)\n\
         (allow file-read*)\n(allow file-write* (subpath \"{cwd}\"))\n{net}\n",
        cwd = cwd.display(),
    );
    vec![
        OsString::from("sandbox-exec"),
        OsString::from("-p"),
        OsString::from(profile),
        interp,
        script,
    ]
}

/// Log once per process that `Auto` produced an unconfined run (no kernel backend was usable).
fn warn_unconfined_once() {
    use std::sync::Once;
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "execute_code sandbox=auto: no OS sandbox backend usable on this host — running \
             UNCONFINED. Install bubblewrap or a Landlock-capable kernel, or set sandbox=require \
             to fail closed."
        );
    });
}

/// Whether bwrap is present *and* can create a user namespace on this host (cached once per process).
async fn bwrap_usable() -> bool {
    static CACHE: tokio::sync::OnceCell<bool> = tokio::sync::OnceCell::const_new();
    *CACHE.get_or_init(probe_bwrap).await
}

/// Probe bwrap by running a trivial sandbox that unshares user + net; a non-zero exit (or a missing
/// binary) means bwrap is unusable here and we must fall back to a plain subprocess.
async fn probe_bwrap() -> bool {
    let fut = tokio::process::Command::new("bwrap")
        .args([
            "--unshare-user",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--",
            "/bin/sh",
            "-c",
            ":",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(
        tokio::time::timeout(PROBE_TIMEOUT, fut).await,
        Ok(Ok(status)) if status.success()
    )
}
