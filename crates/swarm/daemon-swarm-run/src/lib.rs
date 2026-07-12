// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-run` — the participant runtime.
//!
//! The join / warmup / round loops, artifact + data pipeline, checkpoint manager, and digest
//! checks (swarm-training-spec.md §10.1). It is **engine-agnostic**: it drives an abstract
//! [`TrainerBackend`](backend::TrainerBackend), so the same runtime hosts the [`StubBackend`] and
//! the real Burn/wasmtime worker.
//!
//! Seams:
//! - [`data`] — the pre-tokenized shard [`Manifest`], `BatchId → (shard, offset)` mapping, interval
//!   slicing into `steps_per_round` × micro-batches, a deterministic [`SyntheticCorpus`], and the
//!   in-memory [`Corpus`] the engine reads batches from (§8, §6.3).
//! - [`backend`] — the [`TrainerBackend`] trait (**the R↔E seam**) and the deterministic
//!   [`StubBackend`] (§5.1, §10.2, ABI §2.3).
//! - [`engine`] — Wave-2's [`RoundEngine`]: the peer-side round state machine over the frozen seams
//!   (round protocol, barrier I2, record-order staging I3, stall ladder — §6.4).
//! - [`protocol`] — the worker `Command`/`Event` wire types + CBOR codec (§10.2), which lane E's
//!   `daemon-train` worker implements against in Wave 3.
//!
//! Identity/hash types are re-exported from `daemon-swarm-net`'s [`seam`], which (as of Merge 1)
//! resolves them to the canonical `daemon-swarm-proto` types (blake3 `Hash`, `PeerId`).

#![forbid(unsafe_code)]

pub mod backend;
pub mod checkpoint;
pub mod data;
pub mod engine;
pub mod protocol;
pub mod seam;

/// In-process multi-peer test harness + the TEST-ONLY scripted coordinator. Available to external
/// crates behind the `harness` feature, and to this crate's own tests via `cfg(test)`.
#[cfg(any(test, feature = "harness"))]
pub mod harness;

pub use backend::{
    AssessMeta, Assessment, BatchRef, StateDigest, StepCtx, StepStats, StubBackend, TrainerBackend,
};
pub use checkpoint::{CheckpointManifest, ReplayStep};
pub use data::{
    BatchInterval, BatchLocation, Corpus, DataError, InnerStep, Manifest, MicroBatch, ShardDesc,
    SyntheticCorpus, TokenWidth,
};
pub use engine::{EngineConfig, EngineEvent, RoundEngine, RunOutcome};
pub use seam::BatchId;

/// Errors surfaced by the participant runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SwarmRunError {
    /// The transport (control or payload plane) failed.
    #[error(transparent)]
    Net(#[from] daemon_swarm_net::SwarmNetError),
    /// The data pipeline (manifest / batch mapping) failed.
    #[error(transparent)]
    Data(#[from] data::DataError),
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

    #[test]
    fn wraps_data_errors() {
        let err: SwarmRunError = data::DataError::EmptyManifest.into();
        assert!(err.to_string().contains("no shards"));
    }
}
