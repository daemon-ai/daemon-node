// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-sandbox` — kernel-enforced, in-process confinement for a spawned child (Cluster B/C).
//!
//! This crate exists so a caller that runs arbitrary code (today `daemon-tool-execute-code`) can
//! confine the child *without* user namespaces and *without* itself carrying `unsafe`: it owns the
//! only `unsafe` in the exec-sandbox path — the `pre_exec` install of Landlock + seccomp on Linux.
//!
//! ## Posture
//!
//! [`decide`] maps a caller [`Posture`] plus probed [`Capabilities`] to a concrete [`Backend`]:
//! `Require` fails closed when no OS backend is usable (never a silent unconfined run), `Auto`
//! degrades to [`Backend::Plain`] (the caller warns), and `Plain` is always the explicit
//! unconfined choice.
//!
//! ## Backends
//!
//! - **Linux:** [`confine_command`] installs Landlock (filesystem scope) + seccomp (deny
//!   `AF_INET`/`AF_INET6` sockets) on the child via `pre_exec`. The ruleset fds and the seccomp BPF
//!   are built in the *parent*; the post-`fork` child closure performs syscalls only (no allocation
//!   or locking), which is the async-signal-safe discipline a `pre_exec` closure requires.
//! - **macOS / Windows:** [`confine_command`] is a no-op. macOS confinement is expressed as a
//!   `sandbox-exec` argv wrapper by the caller (an argv prefix, not an in-process install), and the
//!   Windows v1 lane is documented fail-closed (no kernel backend → `Require` errors in [`decide`]).

// Phase 4: the sandbox test harness spawns `/bin/sh -c` and interpreters; those are test-only. The
// --lib pass still guards production (which spawns nothing raw). No process spawns in production here.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

use std::io;

#[cfg(target_os = "linux")]
mod linux;

/// The caller's requested confinement posture (a mirror of the tool-level policy, kept here so this
/// crate stays independent of any tool crate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Posture {
    /// Use the strongest available backend; if none is usable, run unconfined ([`Backend::Plain`]).
    Auto,
    /// Require a kernel backend; if none is usable, fail (never a silent unconfined run).
    Require,
    /// Never confine — an explicit, unconfined subprocess.
    Plain,
}

/// The concrete backend chosen for one run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// A bubblewrap namespace sandbox (an argv wrapper; the caller builds its argv).
    Bwrap,
    /// In-process Landlock (fs) + seccomp (network) applied via [`confine_command`] (Linux).
    LandlockSeccomp,
    /// A macOS Seatbelt profile via `sandbox-exec` (an argv wrapper; the caller builds its argv).
    SandboxExec,
    /// No OS confinement.
    Plain,
}

/// Which backends are usable on this host. The caller probes `bwrap`/`sandbox_exec` (backend-argv
/// wrappers it owns); [`landlock_seccomp_available`] fills `landlock_seccomp`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// Bubblewrap is present and can create a user namespace.
    pub bwrap: bool,
    /// The Linux in-process Landlock+seccomp backend is usable.
    pub landlock_seccomp: bool,
    /// The macOS `sandbox-exec` backend is usable (macOS only).
    pub sandbox_exec: bool,
}

impl Capabilities {
    /// The strongest usable backend, preferring the more complete isolation:
    /// bwrap (namespaces) > Landlock+seccomp (in-process fs/net) > `sandbox-exec` (macOS).
    fn strongest(&self) -> Option<Backend> {
        if self.bwrap {
            Some(Backend::Bwrap)
        } else if self.landlock_seccomp {
            Some(Backend::LandlockSeccomp)
        } else if self.sandbox_exec {
            Some(Backend::SandboxExec)
        } else {
            None
        }
    }
}

/// Resolve `posture` against probed `caps` to a concrete [`Backend`].
///
/// `Plain` is always [`Backend::Plain`]. Otherwise the strongest usable backend is chosen; when none
/// is usable, `Require` errors ([`io::ErrorKind::Unsupported`], the fail-closed guarantee) and `Auto`
/// falls back to [`Backend::Plain`] (the caller is expected to warn).
pub fn decide(posture: Posture, caps: &Capabilities) -> io::Result<Backend> {
    if posture == Posture::Plain {
        return Ok(Backend::Plain);
    }
    match caps.strongest() {
        Some(backend) => Ok(backend),
        None => match posture {
            Posture::Require => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "kernel exec sandbox required but no backend is usable on this host \
                 (install bubblewrap or a Landlock-capable kernel on Linux); \
                 set the sandbox policy to `plain` to run unconfined",
            )),
            // Auto: explicit, caller-warned unconfined fallback.
            Posture::Auto => Ok(Backend::Plain),
            Posture::Plain => unreachable!("handled above"),
        },
    }
}

/// Whether the in-process Landlock+seccomp backend is usable on this host (cached once per process).
///
/// Always `false` off Linux. On Linux it probes Landlock support with a hard-requirement ruleset
/// create (a kernel below the required Landlock ABI reports `false`, so `Require` fails closed and
/// `Auto` falls through — no partial/misleading enforcement is silently accepted).
pub fn landlock_seccomp_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        linux::available()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Install the in-process child sandbox on a not-yet-spawned `cmd` for the Linux backend.
///
/// On Linux this attaches a `pre_exec` hook that installs Landlock (scoping the child to
/// [`SandboxSpec::rw_paths`]/[`SandboxSpec::ro_paths`]) and, when
/// [`SandboxSpec::allow_network`] is false, a seccomp filter denying `AF_INET`/`AF_INET6` sockets.
/// The hook is **fail-closed**: if the install fails in the child, the closure returns an error and
/// the spawn fails rather than running unconfined.
///
/// Off Linux this is a no-op (macOS/Windows confinement is expressed elsewhere — see the crate
/// docs), so callers can invoke it unconditionally.
pub fn confine_command(cmd: &mut tokio::process::Command, spec: &SandboxSpec) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::install(cmd, spec)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (cmd, spec);
        Ok(())
    }
}

/// The filesystem/network scope for the in-process (Linux) backend.
#[derive(Clone, Debug, Default)]
pub struct SandboxSpec {
    /// Roots the child may read, write, and create under (the run's working directory).
    pub rw_paths: Vec<std::path::PathBuf>,
    /// Roots the child may read and execute (the interpreter plus system library paths).
    pub ro_paths: Vec<std::path::PathBuf>,
    /// When false, `AF_INET`/`AF_INET6` sockets are denied via seccomp.
    pub allow_network: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(bwrap: bool, landlock_seccomp: bool, sandbox_exec: bool) -> Capabilities {
        Capabilities {
            bwrap,
            landlock_seccomp,
            sandbox_exec,
        }
    }

    #[test]
    fn plain_posture_is_always_plain() {
        assert_eq!(
            decide(Posture::Plain, &caps(true, true, true)).unwrap(),
            Backend::Plain
        );
        assert_eq!(
            decide(Posture::Plain, &caps(false, false, false)).unwrap(),
            Backend::Plain
        );
    }

    #[test]
    fn require_fails_closed_without_any_backend() {
        let err = decide(Posture::Require, &caps(false, false, false)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn auto_degrades_to_plain_without_any_backend() {
        assert_eq!(
            decide(Posture::Auto, &caps(false, false, false)).unwrap(),
            Backend::Plain
        );
    }

    #[test]
    fn strongest_backend_is_preferred() {
        // bwrap wins over the in-process backend.
        assert_eq!(
            decide(Posture::Require, &caps(true, true, false)).unwrap(),
            Backend::Bwrap
        );
        // Landlock+seccomp is the fallback when bwrap is unusable.
        assert_eq!(
            decide(Posture::Require, &caps(false, true, false)).unwrap(),
            Backend::LandlockSeccomp
        );
        assert_eq!(
            decide(Posture::Auto, &caps(false, true, false)).unwrap(),
            Backend::LandlockSeccomp
        );
        // sandbox-exec is last (macOS).
        assert_eq!(
            decide(Posture::Require, &caps(false, false, true)).unwrap(),
            Backend::SandboxExec
        );
    }
}
