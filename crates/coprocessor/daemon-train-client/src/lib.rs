// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train-client` — the node-side training-worker supervisor.
//!
//! What `daemon-metta-client` is to the MeTTa worker (and `LocalProvider` to the inference worker):
//! it spawns the `daemon-train` child over a length-framed [`CutChannel`], speaks the worker
//! protocol (swarm-training-spec.md §10.2), respawns after a crash / transport fault, and trips a
//! crash-loop "meltdown" when restarts exceed a budget within a sliding window. It links only the
//! light node-side crates — never wasmtime / Burn — so the daemon stays out of the worker fault
//! domain (§10.1, §10.5).
//!
//! Wave-0 scaffold: only the config + error types are present; the supervisor client lands with
//! lane **R** (mirror the shape of `daemon-metta-client`).
//!
//! [`CutChannel`]: daemon_provision::CutChannel

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Duration;

/// Construction + tuning for a training-worker supervisor (mirrors `MettaConfig`).
#[derive(Clone, Debug)]
pub struct TrainClientConfig {
    /// Path to the `daemon-train` worker binary.
    pub worker_bin: PathBuf,
    /// How long to wait for worker readiness after spawning.
    pub spawn_timeout: Duration,
    /// Crash-loop meltdown: max restarts allowed within [`TrainClientConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which [`TrainClientConfig::max_restarts`] is counted.
    pub restart_window: Duration,
}

impl TrainClientConfig {
    /// A config with sensible supervision defaults for `worker_bin`.
    #[must_use]
    pub fn new(worker_bin: impl Into<PathBuf>) -> Self {
        Self {
            worker_bin: worker_bin.into(),
            spawn_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// Errors surfaced by the training-worker supervisor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrainClientError {
    /// Spawning or handshaking with the worker child failed.
    #[error("worker spawn error: {0}")]
    Spawn(String),
    /// The worker crash-looped past its meltdown budget.
    #[error("worker meltdown: restarts exceeded budget")]
    Fatal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_has_defaults() {
        let cfg = TrainClientConfig::new("/usr/bin/daemon-train");
        assert_eq!(cfg.max_restarts, 3);
        assert_eq!(cfg.spawn_timeout, Duration::from_secs(30));
    }
}
