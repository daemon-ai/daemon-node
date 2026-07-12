// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-coordinator` — the coordinator harness.
//!
//! The local / private coordinator server (axum/WS) plus the `wasm32` export for the cloud Durable
//! Object (swarm-training-spec.md §10.1, §11.2). It drives the `tick` state machine defined in
//! [`daemon_swarm_proto`] — the harness is I/O; the decisions live in the wasm-clean proto crate.
//!
//! Wave-0 scaffold: only the error type is present; the harness lands with lane **P**.

#![forbid(unsafe_code)]

/// Errors surfaced by the coordinator harness.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CoordinatorError {
    /// The coordinator server (bind / accept / upgrade) failed.
    #[error("coordinator server error: {0}")]
    Server(String),
    /// A protocol-envelope step failed while advancing the state machine.
    #[error(transparent)]
    Proto(#[from] daemon_swarm_proto::SwarmProtoError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_proto_errors() {
        let err: CoordinatorError =
            daemon_swarm_proto::SwarmProtoError::Validation("bad tick".into()).into();
        assert!(err.to_string().contains("validation failed"));
    }
}
