// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-observe` — the observer / replay-oracle for a swarm run.
//!
//! A node-side tool that consumes **only signed messages + published objects** (never privileged
//! coordinator state) and turns them into audit + recovery primitives (swarm-training-spec.md §14,
//! §10.1; TDD §3.9 + PROTO-20; the Psyche event-sourcing lesson, Appendix A.5):
//!
//! * [`MessageLog`] — an append-only, replayable log of [`SignedMessage`](daemon_swarm_proto::messages::SignedMessage)
//!   in arrival order, canonical-CBOR framed, indexed by `(round, kind)`. Writer + reader.
//! * [`replay`] — the replay oracle (PROTO-20 as a library): re-run
//!   [`daemon_swarm_coordinator::tick`] from genesis over a recorded `Input` trace and verify the
//!   recorded [`RoundRecord`](daemon_swarm_proto::messages::RoundRecord)s match the pure function's —
//!   the "anyone can re-derive the coordinator" property, with the first divergence pinpointed.
//! * [`digest_tally`] / [`DesyncVerdict`] — fold `Digest` messages per round into a quorum digest +
//!   outlier set (the observe-driven desync trigger the runtime lane consumes — this crate does not
//!   wire it into `daemon-swarm-run`).
//! * [`RunHealth`] / [`RoundHealth`] — per-round facts (committed count, attested coverage,
//!   stragglers, drops, round span) as plain serializable types — the base for CLI/UX.
//!
//! **std, not wasm.** Unlike `daemon-swarm-proto`/`-coordinator` (wasm32-clean substrate types), the
//! observer is a node-side log tool: it uses `std::io` framing and `thiserror`, and is never linked
//! into the coordinator DO.

#![forbid(unsafe_code)]

pub mod desync;
pub mod health;
pub mod log;
pub mod replay;

pub use desync::{digest_tally, DesyncVerdict};
pub use health::{RoundHealth, RunHealth};
pub use log::{MessageKind, MessageLog};
pub use replay::{genesis_seed, replay, ReplayDivergence, ReplayError, ReplayReport};

/// Errors surfaced by the run-log store and its projections.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObserveError {
    /// Appending to or reading the ordered event store failed (framing / I/O).
    #[error("run-log store error: {0}")]
    Store(String),
    /// Rebuilding a projection from the event log failed.
    #[error("projection error: {0}")]
    Projection(String),
    /// A canonical-CBOR (de)serialization step failed.
    #[error("codec error: {0}")]
    Codec(String),
    /// Replay setup failed (envelope config / genesis derivation).
    #[error("replay setup error: {0}")]
    Replay(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_renders() {
        assert!(ObserveError::Store("append".into())
            .to_string()
            .contains("run-log store"));
        assert!(ObserveError::Codec("x".into())
            .to_string()
            .contains("codec"));
    }
}
