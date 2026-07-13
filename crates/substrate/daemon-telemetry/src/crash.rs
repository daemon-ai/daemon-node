// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Consent-gated crash reporting for the node process tree (Sentry SaaS).
//!
//! Two independent failure modes are covered, mirroring the C++ app side (`crash_reporter`):
//! - **Rust panics** — captured by Sentry's default panic integration (works under `panic =
//!   "unwind"`, which the workspace profile pins and this module must NOT change).
//! - **native crashes** (SIGSEGV / abort / stack corruption / FFI faults in `unsafe`, `llama.cpp`,
//!   `burn`, …) — captured by [`sentry_rust_minidump`], which spawns an *out-of-process* monitor
//!   (a re-exec of the current binary) that writes a minidump and uploads it as a Sentry event.
//!
//! Every reporting process reports into the ONE Sentry project, correlated by tags: `component`
//! (the process role), plus `session_id` / `parent_pid` propagated from the spawning node through
//! the environment (see the worker-spawn env injection in `daemon-providers` / the train client).
//!
//! ## Gates (both required to arm upload)
//! 1. **DSN present** — `DAEMON_SENTRY_DSN` is set + non-empty (packaging / the spawning node
//!    injects it). Absent ⇒ crash reporting is a compiled-in no-op (nothing initializes).
//! 2. **Consent on** — `DAEMON_CRASH_CONSENT=1`. The app owns the consent toggle and threads the
//!    current value to every worker it spawns; for node processes, consent-off simply means Sentry
//!    is never initialized (there is no local dump retention on the node side — that is the app's
//!    job). This is the deliberate node/app asymmetry: the app arms capture-without-upload via
//!    `require_user_consent`, the node just stays dark until consent is granted.
//!
//! `send_default_pii` is never enabled.
//!
//! ## The minidump monitor re-exec (why init runs where it does)
//! [`sentry_rust_minidump::init`] → `minidumper_child` spawns the monitor by re-exec'ing the
//! current executable with an extra `--crash-reporter-server=<socket>` argument; in that re-exec'd
//! process `spawn()` runs the monitor server loop and then `std::process::exit(0)` — it never
//! returns to the caller. Two consequences the call sites must respect:
//! - the binary's argument parser must **accept** (and ignore) `--crash-reporter-server=…` so the
//!   monitor copy does not die in `clap` before it reaches this init; and
//! - anything the binary does *before* calling [`init_crash_reporting`] also runs in the monitor
//!   copy, so keep pre-init work cheap and side-effect-free (subscriber init is fine).
//!
//! The monitor is a plain child of the spawning process (no `kill_on_drop`, no process group): it
//! is not a placed unit and is invisible to `ProcessProvisioner` accounting, and it self-exits on a
//! stale-socket timeout once its parent is gone.

use std::time::Duration;

/// How long [`CrashGuard`] drop waits for the Sentry transport to flush buffered events. Kept short
/// so a clean shutdown is not stalled by an unreachable ingest endpoint.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);

/// Process-lifetime crash-reporting resources. Holds the Sentry client-init guard (flushes buffered
/// events and disposes the client on drop) and the out-of-process minidump monitor handle (must
/// outlive the process to catch native crashes). Bind it for the whole run — `let _crash =
/// init_crash_reporting(role);` in `main`, next to the telemetry guard. A default (disabled) guard
/// is returned whenever a gate is unmet, so callers never branch on whether reporting is on.
#[derive(Default)]
#[must_use = "hold the CrashGuard for the whole process; dropping it disarms crash reporting"]
pub struct CrashGuard {
    sentry: Option<sentry::ClientInitGuard>,
    // The minidump monitor client handle; dropping it tears the IPC link to the monitor. Held for
    // the process lifetime. `None` when the native monitor could not start (panic capture still
    // works through the Sentry guard alone).
    _minidump: Option<sentry_rust_minidump::Handle>,
}

impl CrashGuard {
    /// Whether Sentry crash reporting actually armed (both gates met and the client is enabled).
    pub fn is_armed(&self) -> bool {
        self.sentry.as_ref().is_some_and(|g| g.is_enabled())
    }
}

impl Drop for CrashGuard {
    fn drop(&mut self) {
        if let Some(guard) = self.sentry.take() {
            // Give the transport a bounded moment to ship anything still queued.
            guard.flush(Some(FLUSH_TIMEOUT));
        }
    }
}

/// The env var carrying the Sentry DSN (packaging / a parent process sets it; a child forwards it).
pub const ENV_DSN: &str = "DAEMON_SENTRY_DSN";
/// The env var carrying the current crash-reporting consent (`"1"` = granted; anything else = off).
pub const ENV_CONSENT: &str = "DAEMON_CRASH_CONSENT";
/// The env var carrying the placement session id, for cross-process crash correlation.
pub const ENV_SESSION_ID: &str = "DAEMON_SESSION_ID";
/// The env var carrying the spawning process's pid, for cross-process crash correlation.
pub const ENV_PARENT_PID: &str = "DAEMON_PARENT_PID";

/// Build the crash-reporting correlation environment a node process injects into a worker it
/// spawns, to be appended to the child `PlacementSpec.env`:
/// - `DAEMON_SENTRY_DSN` / `DAEMON_CRASH_CONSENT` — forwarded from *this* process's environment so
///   the whole tree shares one DSN and one consent decision (env is the live propagation channel;
///   `crash_consent_set` updates this process's `DAEMON_CRASH_CONSENT` so future spawns see the new
///   value). Absent DSN ⇒ not forwarded (the child then no-ops, same as this process).
/// - `DAEMON_SESSION_ID` — the placement `session_id` (`session` argument).
/// - `DAEMON_PARENT_PID` — this process's pid.
///
/// The child's [`init_crash_reporting`] reads these to gate + tag its own reporter.
pub fn correlation_env(session_id: &str) -> Vec<(String, String)> {
    let mut env = Vec::with_capacity(4);
    if let Ok(dsn) = std::env::var(ENV_DSN) {
        if !dsn.trim().is_empty() {
            env.push((ENV_DSN.to_string(), dsn));
        }
    }
    // Forward the current consent decision (default off when unset).
    let consent = std::env::var(ENV_CONSENT).unwrap_or_default();
    env.push((ENV_CONSENT.to_string(), consent));
    env.push((ENV_SESSION_ID.to_string(), session_id.to_string()));
    env.push((ENV_PARENT_PID.to_string(), std::process::id().to_string()));
    env
}

/// The two gates + `sentry::init` + scope tagging, shared by [`init_crash_reporting`] (which adds
/// the native minidump monitor) and [`init_panic_reporting`] (panic capture only). Returns `None`
/// (disabled no-op) unless a DSN is present and `DAEMON_CRASH_CONSENT=1`.
fn init_sentry(component: &str) -> Option<sentry::ClientInitGuard> {
    // Gate 1: DSN. Runtime env override wins; absent/empty ⇒ disabled no-op.
    let dsn = match std::env::var("DAEMON_SENTRY_DSN") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return None,
    };

    // Gate 2: consent. The app propagates the current toggle to every worker it spawns; for node
    // processes consent-off means "stay dark" (no local retention on this side).
    let consent = std::env::var("DAEMON_CRASH_CONSENT").is_ok_and(|v| v == "1");
    if !consent {
        tracing::debug!(
            component,
            "crash reporting: consent off — Sentry not initialized"
        );
        return None;
    }

    let release = format!("daemon-node@{}", daemon_common::VERSION);

    let guard = sentry::init((
        dsn,
        sentry::ClientOptions {
            release: Some(release.clone().into()),
            // A minidump is memory; consent already gates its upload. Never opt into PII on top.
            send_default_pii: false,
            // Attach a Rust backtrace to captured panics/errors for symbolication.
            attach_stacktrace: true,
            ..Default::default()
        },
    ));

    // Correlation tags: the process role plus the session id / parent pid the spawning node
    // injected into the environment (absent for the top-level host, present in every worker).
    sentry::configure_scope(|scope| {
        scope.set_tag("component", component);
        if let Ok(session_id) = std::env::var("DAEMON_SESSION_ID") {
            if !session_id.is_empty() {
                scope.set_tag("session_id", session_id);
            }
        }
        if let Ok(parent_pid) = std::env::var("DAEMON_PARENT_PID") {
            if !parent_pid.is_empty() {
                scope.set_tag("parent_pid", parent_pid);
            }
        }
    });

    tracing::info!(
        component,
        release,
        "crash reporting armed (consent granted)"
    );
    Some(guard)
}

/// Initialize consent-gated crash reporting for a node process `component` role (e.g. `"host"`,
/// `"placed-child"`, `"transport-server"`, `"infer-worker"`, `"train-worker"`).
///
/// Returns a disabled [`CrashGuard`] (a no-op) when the DSN is absent or consent is off. When both
/// gates are met it installs the Sentry panic integration, tags the scope for cross-process
/// correlation, and spawns the out-of-process minidump monitor for native crashes.
///
/// See the [module docs](self) for the re-exec contract the call site must honour: because this may
/// re-exec the binary, call it as early as feasible and ensure the argument parser tolerates a
/// `--crash-reporter-server=…` flag.
pub fn init_crash_reporting(component: &str) -> CrashGuard {
    let Some(guard) = init_sentry(component) else {
        return CrashGuard::default();
    };

    // Native crash capture. In the re-exec'd monitor process this call runs the server loop and
    // never returns (the process exits inside it — see module docs).
    let minidump = match sentry_rust_minidump::init(&guard) {
        Ok(handle) => Some(handle),
        Err(error) => {
            tracing::warn!(%error, component, "crash reporting: minidump monitor failed to start");
            None
        }
    };

    CrashGuard {
        sentry: Some(guard),
        _minidump: minidump,
    }
}

/// Initialize consent-gated **panic-only** crash reporting (no native minidump monitor, so no
/// re-exec). For short-lived CLI processes where a Rust panic is the only failure mode worth
/// reporting and the out-of-process monitor's cost/latency is not warranted. Same DSN + consent
/// gates as [`init_crash_reporting`]; returns a disabled [`CrashGuard`] when either is unmet.
pub fn init_panic_reporting(component: &str) -> CrashGuard {
    CrashGuard {
        sentry: init_sentry(component),
        _minidump: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test drives all env-sensitive cases sequentially: cargo runs tests within a crate in
    // parallel threads that share the process environment, so splitting these across `#[test]`s
    // would race on `DAEMON_SENTRY_DSN` / `DAEMON_CRASH_CONSENT`.
    #[test]
    fn gates_and_correlation_env() {
        // Clean slate.
        std::env::remove_var("DAEMON_SENTRY_DSN");
        std::env::remove_var("DAEMON_CRASH_CONSENT");
        std::env::remove_var("DAEMON_SESSION_ID");
        std::env::remove_var("DAEMON_PARENT_PID");

        // Gate 1: no DSN ⇒ disabled no-op (both entry points), never panics / spawns a monitor.
        assert!(!init_crash_reporting("test-host").is_armed());
        assert!(!init_panic_reporting("test-cli").is_armed());

        // Gate 2: DSN present but consent off ⇒ still disabled.
        std::env::set_var("DAEMON_SENTRY_DSN", "https://public@example.invalid/1");
        assert!(!init_crash_reporting("test-host").is_armed());
        std::env::set_var("DAEMON_CRASH_CONSENT", "0");
        assert!(!init_panic_reporting("test-cli").is_armed());

        // correlation_env forwards the DSN + consent and always tags session id + parent pid.
        std::env::set_var("DAEMON_CRASH_CONSENT", "1");
        let env = correlation_env("sess-123");
        let get = |k: &str| env.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.as_str());
        assert_eq!(get(ENV_DSN), Some("https://public@example.invalid/1"));
        assert_eq!(get(ENV_CONSENT), Some("1"));
        assert_eq!(get(ENV_SESSION_ID), Some("sess-123"));
        assert_eq!(
            get(ENV_PARENT_PID),
            Some(std::process::id().to_string().as_str())
        );

        // With no DSN in env, correlation_env omits the DSN entry but still tags the rest.
        std::env::remove_var("DAEMON_SENTRY_DSN");
        let env = correlation_env("sess-9");
        assert!(env.iter().all(|(k, _)| k != ENV_DSN));
        assert!(env
            .iter()
            .any(|(k, v)| k == ENV_SESSION_ID && v == "sess-9"));

        // Cleanup so a sibling crate's tests don't observe these.
        std::env::remove_var("DAEMON_CRASH_CONSENT");
        std::env::remove_var("DAEMON_SESSION_ID");
        std::env::remove_var("DAEMON_PARENT_PID");
    }
}
