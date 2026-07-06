// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Declared child-process environment policy (OpenClaw Cluster E hardening).
//!
//! Every child-process spawn site states an [`EnvPolicy`] explicitly, so environment inheritance
//! is an audited, greppable, lintable choice instead of an implicit default. Trusted node workers
//! (provisioner cuts, MCP stdio servers, ACP agents) legitimately inherit the full daemon env —
//! the point is not to change their behavior but to make that inheritance *declared* at the spawn
//! site, so a future clippy `disallowed-methods` gate can ban raw `Command` env mutation outside
//! [`EnvPolicy::apply`] and an undeclared spawn fails the lint. Agent-facing subprocesses use the
//! scrubbed shape (`Clean`) already proven by `daemon-core`'s `LocalEnvironment::run`.
//!
//! The enum is pure data and always compiled; the tokio application helper is gated behind the
//! `process` feature so non-spawning consumers of `daemon-common` stay runtime-free.

/// The declared environment-inheritance policy for a child-process spawn. Every spawn site states
/// one explicitly — inheritance is an audited choice, never a default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvPolicy {
    /// Inherit the parent (daemon) environment as-is. For **trusted node workers** that
    /// legitimately need the daemon's ambient env (provider keys, `PATH`, locale). Every use
    /// carries a call-site comment justifying the trust; this is the audited choice, not a
    /// default.
    InheritFull,
    /// Start from an empty environment and carry through only the variables named in `allowlist`
    /// (values read from the parent env; unset names are skipped). For **agent-facing /
    /// less-trusted** children, so no host secret leaks into the subprocess. Mirrors the scrubbed
    /// env in `daemon-core`'s `LocalEnvironment::run` (allowlist `["PATH"]`).
    Clean {
        /// Variable names carried through from the parent environment.
        allowlist: Vec<String>,
    },
}

#[cfg(feature = "process")]
mod sealed {
    pub trait Sealed {}
    impl Sealed for tokio::process::Command {}
    impl Sealed for std::process::Command {}
}

/// The child-command types that can receive a declared [`EnvPolicy`] via [`EnvPolicy::apply`] — the
/// async `tokio::process::Command` and the blocking `std::process::Command`. Sealed: only those two
/// flavors implement it, so [`EnvPolicy::apply`] is the single sanctioned env-mutation site for both
/// (the Phase 4 clippy `disallowed-methods` env ban lives on the two impls below and nowhere else).
#[cfg(feature = "process")]
pub trait EnvSink: sealed::Sealed {
    /// Clear the child's inherited environment.
    #[doc(hidden)]
    fn clear_env(&mut self);
    /// Set one child environment variable.
    #[doc(hidden)]
    fn set_env(&mut self, key: &std::ffi::OsStr, value: &std::ffi::OsStr);
}

#[cfg(feature = "process")]
#[allow(clippy::disallowed_methods)] // the one sanctioned env-mutation site (Phase 4 lint anchor)
impl EnvSink for tokio::process::Command {
    fn clear_env(&mut self) {
        self.env_clear();
    }
    fn set_env(&mut self, key: &std::ffi::OsStr, value: &std::ffi::OsStr) {
        self.env(key, value);
    }
}

#[cfg(feature = "process")]
#[allow(clippy::disallowed_methods)] // the one sanctioned env-mutation site (Phase 4 lint anchor)
impl EnvSink for std::process::Command {
    fn clear_env(&mut self) {
        self.env_clear();
    }
    fn set_env(&mut self, key: &std::ffi::OsStr, value: &std::ffi::OsStr) {
        self.env(key, value);
    }
}

#[cfg(feature = "process")]
impl EnvPolicy {
    /// Apply this policy to a child command, then layer the caller's explicit `extra` vars on top
    /// (in order, overriding any inherited/allowlisted value of the same name — the same
    /// precedence as the per-site `.env` loops this replaces). Works for both the `tokio` and `std`
    /// [`Command`](std::process::Command) flavors, and accepts `OsStr`-valued keys/values so a
    /// non-UTF-8 path (e.g. `TMPDIR`) is carried losslessly.
    ///
    /// This is the **only sanctioned way** to set a child's environment: routing both the base
    /// inheritance choice and the declared extras through one function (backed by the two anchored
    /// [`EnvSink`] impls) lets the clippy `disallowed-methods` gate ban raw `env`/`env_clear`/`envs`
    /// calls everywhere else, making an *undeclared* policy unrepresentable.
    pub fn apply<'c, C, K, V>(&self, cmd: &'c mut C, extra: &[(K, V)]) -> &'c mut C
    where
        C: EnvSink,
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
    {
        match self {
            EnvPolicy::InheritFull => { /* keep the parent env exactly as-is */ }
            EnvPolicy::Clean { allowlist } => {
                cmd.clear_env();
                for name in allowlist {
                    if let Some(value) = std::env::var_os(name) {
                        cmd.set_env(name.as_ref(), &value);
                    }
                }
            }
        }
        for (key, value) in extra {
            cmd.set_env(key.as_ref(), value.as_ref());
        }
        cmd
    }
}

#[cfg(all(test, feature = "process"))]
mod tests {
    use super::EnvPolicy;
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::OsString;

    /// Parse `env` output (one `KEY=VALUE` per line) into a map. Single-line values only are asserted
    /// on (multi-line continuation lines parse as noise entries, never colliding with the specific
    /// keys the test checks).
    fn parse_env(stdout: &[u8]) -> BTreeMap<String, String> {
        String::from_utf8_lossy(stdout)
            .lines()
            .filter_map(|l| {
                l.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect()
    }

    /// Spawn `env` under a `tokio` command and parse the child's environment.
    async fn child_env(cmd: &mut tokio::process::Command) -> BTreeMap<String, String> {
        let out = cmd.output().await.expect("spawn `env`");
        assert!(out.status.success(), "`env` exited nonzero");
        parse_env(&out.stdout)
    }

    fn is_simple_marker(key: &str, value: &str) -> bool {
        key != "PATH" && !key.is_empty() && !value.is_empty() && !value.contains('\n')
    }

    /// A pre-existing parent env var (never `PATH`) used as the inheritance marker — chosen from
    /// the ambient environment so the test never mutates process-global env (no `set_var`, no
    /// cross-test races under the parallel runner). `HOME` when available, else any simple var
    /// (the cargo test runner always exports several, e.g. `CARGO_MANIFEST_DIR`).
    fn parent_marker() -> (String, String) {
        std::env::var("HOME")
            .ok()
            .filter(|home| is_simple_marker("HOME", home))
            .map(|home| ("HOME".to_string(), home))
            .or_else(|| {
                std::env::vars_os()
                    .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
                    .find(|(k, v)| is_simple_marker(k, v))
            })
            .expect("test process exports at least one simple env var besides PATH")
    }

    #[tokio::test]
    async fn env_policy_variants_apply_expected_child_env() {
        let (marker_key, marker_value) = parent_marker();

        // InheritFull → the child env is a superset of the parent's: the pre-existing marker
        // passes through untouched, the declared extra is layered on top, ambient PATH survives.
        // (`&str` keys/values exercise the `AsRef<OsStr>` extras path.)
        let mut inherit = tokio::process::Command::new("env");
        EnvPolicy::InheritFull.apply(&mut inherit, &[("DAEMON_ENV_POLICY_EXTRA", "inherit")]);
        let child = child_env(&mut inherit).await;
        assert_eq!(
            child.get(&marker_key),
            Some(&marker_value),
            "InheritFull passes the pre-existing {marker_key} through"
        );
        assert_eq!(
            child.get("DAEMON_ENV_POLICY_EXTRA").map(String::as_str),
            Some("inherit"),
            "declared extras are applied on top of the inherited env"
        );
        assert!(child.contains_key("PATH"), "ambient PATH survives");

        // Clean { allowlist: ["PATH"] } → the child env is exactly the allowlist plus the declared
        // extras: PATH carries the parent's value, the marker is dropped. Extra VALUES here are
        // `OsString` (the lossless path execute-code's `TMPDIR` relies on), with `&str` keys.
        let mut clean = tokio::process::Command::new("env");
        EnvPolicy::Clean {
            allowlist: vec!["PATH".into()],
        }
        .apply(
            &mut clean,
            &[
                ("DAEMON_ENV_POLICY_EXTRA", OsString::from("clean")),
                ("DAEMON_ENV_POLICY_OS", OsString::from("os-value")),
            ],
        );
        let child = child_env(&mut clean).await;
        assert_eq!(
            child.get("PATH"),
            std::env::var("PATH").ok().as_ref(),
            "allowlisted PATH carries the parent's value"
        );
        assert_eq!(
            child.get("DAEMON_ENV_POLICY_EXTRA").map(String::as_str),
            Some("clean"),
            "declared extras are applied on top of the scrubbed env"
        );
        assert_eq!(
            child.get("DAEMON_ENV_POLICY_OS").map(String::as_str),
            Some("os-value"),
            "an OsString-valued extra is carried through unchanged"
        );
        assert!(
            !child.contains_key(&marker_key),
            "Clean drops the non-allowlisted {marker_key}"
        );
        let keys: BTreeSet<&str> = child.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["DAEMON_ENV_POLICY_EXTRA", "DAEMON_ENV_POLICY_OS", "PATH"]
                .into_iter()
                .collect(),
            "Clean yields exactly the allowlist + declared extras"
        );
    }

    /// The same `Clean` invariant on a **blocking `std::process::Command`** — proves the `EnvSink`
    /// std path used by the `daemon-processes` `sh -c` gate: the child env is exactly the allowlist
    /// (`PATH`) plus the declared extra, and the non-allowlisted parent marker is dropped.
    #[test]
    fn clean_scrubs_std_command_env() {
        let (marker_key, _marker_value) = parent_marker();

        let mut cmd = std::process::Command::new("env");
        EnvPolicy::Clean {
            allowlist: vec!["PATH".into()],
        }
        .apply(&mut cmd, &[("DAEMON_ENV_POLICY_EXTRA", "std-clean")]);
        let out = cmd.output().expect("spawn `env`");
        assert!(out.status.success(), "`env` exited nonzero");
        let child = parse_env(&out.stdout);

        assert_eq!(
            child.get("PATH"),
            std::env::var("PATH").ok().as_ref(),
            "allowlisted PATH carries the parent's value (std Command)"
        );
        assert_eq!(
            child.get("DAEMON_ENV_POLICY_EXTRA").map(String::as_str),
            Some("std-clean"),
            "declared extra applied on the scrubbed std-Command env"
        );
        assert!(
            !child.contains_key(&marker_key),
            "Clean drops the non-allowlisted {marker_key} (std Command)"
        );
        let keys: BTreeSet<&str> = child.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["DAEMON_ENV_POLICY_EXTRA", "PATH"].into_iter().collect(),
            "Clean on std Command yields exactly the allowlist + declared extra"
        );
    }
}
