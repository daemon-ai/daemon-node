// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! OS sandbox selection + argv construction for execute_code.
//!
//! On Linux the child can run inside [bubblewrap](https://github.com/containers/bubblewrap): a
//! read-only view of `/nix/store` + common system paths, a read-write bind of *only* the working
//! directory, a private `/proc`/`/dev`/`/tmp`, and (by default) no network. This is additive
//! hardening over hermes, whose `execute_code` runs unsandboxed — so when bwrap is missing or user
//! namespaces are unavailable, [`resolve`] falls back to a plain subprocess (unless the policy
//! *requires* bwrap). The workspace-containment of the tool's own staging + CWD holds either way.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use crate::{NetworkPolicy, SandboxPolicy};

/// The chosen execution backend for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SandboxKind {
    /// Run under a bubblewrap namespace sandbox.
    Bwrap,
    /// Run as a plain subprocess (no OS jail).
    Plain,
}

/// The bwrap capability-probe timeout.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolve the effective backend for `policy`: `Auto` uses bwrap when usable else plain; `Bwrap`
/// errors if bwrap is unavailable; `None` is always plain.
pub(crate) async fn resolve(policy: SandboxPolicy) -> std::io::Result<SandboxKind> {
    match policy {
        SandboxPolicy::None => Ok(SandboxKind::Plain),
        SandboxPolicy::Auto => Ok(if bwrap_usable().await {
            SandboxKind::Bwrap
        } else {
            SandboxKind::Plain
        }),
        SandboxPolicy::Bwrap => {
            if bwrap_usable().await {
                Ok(SandboxKind::Bwrap)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "bubblewrap sandbox required but unavailable \
                     (bwrap missing or user namespaces disabled)",
                ))
            }
        }
    }
}

/// Build the full argv (program + args) for a run. `Plain` is `[interpreter, script]`; `Bwrap`
/// wraps that in a namespace sandbox rooted at `cwd`.
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
    if kind == SandboxKind::Plain {
        return vec![interp, script];
    }

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
    for path in [
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
    ] {
        a.push(OsString::from("--ro-bind-try"));
        a.push(OsString::from(path));
        a.push(OsString::from(path));
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
