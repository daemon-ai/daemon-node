// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-run` — the participant runtime.
//!
//! The join / warmup / round loops, artifact + data pipeline, checkpoint manager, and digest
//! checks (swarm-training-spec.md §10.1). It is **engine-agnostic**: it drives an abstract
//! [`TrainerBackend`](backend::TrainerBackend), so the same runtime hosts the [`StubBackend`] and
//! the real Burn/wasmtime worker.
//!
//! Wave-1 seams (the round loop itself lands in Wave 2):
//! - [`data`] — the pre-tokenized shard [`Manifest`], `BatchId → (shard, offset)` mapping, interval
//!   slicing into `steps_per_round` × micro-batches, and a deterministic [`SyntheticCorpus`] (§8, §6.3).
//! - [`backend`] — the [`TrainerBackend`] trait (**the R↔E seam**) and the deterministic
//!   [`StubBackend`] (§5.1, §10.2, ABI §2.3).
//!
//! Identity/hash types are re-exported from `daemon-swarm-net`'s [`seam`] (MERGE-1 placeholders for
//! `daemon-swarm-proto`, lane P).

#![forbid(unsafe_code)]

pub mod backend;
pub mod data;
pub mod seam;

pub use backend::{
    AssessMeta, Assessment, BatchRef, StateDigest, StepCtx, StepStats, StubBackend, TrainerBackend,
};
pub use data::{
    BatchInterval, BatchLocation, DataError, InnerStep, Manifest, MicroBatch, ShardDesc,
    SyntheticCorpus, TokenWidth,
};
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
