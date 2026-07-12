// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-run` — the participant runtime.
//!
//! The join / warmup / round loops, artifact + data pipeline, checkpoint manager, and digest
//! checks (swarm-training-spec.md §10.1). It is **engine-agnostic**: it drives an abstract
//! `TrainerBackend`, so the same runtime hosts the stub backend and the real Burn/wasmtime worker.
//!
//! Wave-0 scaffold: only the error type is present; the round loops land with lane **R**.

#![forbid(unsafe_code)]

/// Errors surfaced by the participant runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SwarmRunError {
    /// The transport (control or payload plane) failed.
    #[error(transparent)]
    Net(#[from] daemon_swarm_net::SwarmNetError),
    /// A round-lifecycle invariant was violated (warmup, digest, or checkpoint step).
    #[error("swarm run lifecycle error: {0}")]
    Lifecycle(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_net_errors() {
        let err: SwarmRunError = daemon_swarm_net::SwarmNetError::Transport("gossip".into()).into();
        assert!(err.to_string().contains("gossip"));
    }
}
