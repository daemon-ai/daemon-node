// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-session` — the participant runtime.
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
//! - [`engine`] — the [`RoundEngine`]: the peer-side round state machine over the frozen seams
//!   (round protocol, barrier I2, record-order staging I3, stall ladder — §6.4).
//! - [`protocol`] — the worker `Command`/`Event` wire types + CBOR codec (§10.2), which lane E's
//!   `daemon-vhc-host` worker implements later.
//!
//! Identity/hash types are re-exported from `daemon-vhc-net`'s [`seam`], which (as of Merge 1)
//! resolves them to the canonical `daemon-vhc-proto` types (blake3 `Hash`, `PeerId`).

#![forbid(unsafe_code)]

pub mod assess;
pub mod backend;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod engine;
pub mod protocol;
pub mod seam;
// The host-enforced upgrade transaction (Phase E; architecture §5.4, ABI §10.3): the LOCAL half of
// the two-key model — quiesce → snapshot → owner-law re-check (grant-expanding fails closed) →
// migrate → validate → activate locally → rollback-and-retry-or-leave. Composes the committed
// transition chain (`daemon_vhc_proto::TransitionChain`, deliverable 1); the wasm-guest step
// adapters and drills live in the host testkit.
pub mod attach;
pub mod upgrade;
// `wasm_backend` (the v1 five-phase driver's TrainerBackend binding, moved here at the A2
// inversion) RETIRED at the Phase-E v1 sunset (decisions D5) together with the host's Instance
// lifecycle: the trait seam + `StubBackend` remain (the harness/checkpoint/cold-join substrate);
// no wasm TrainerBackend exists — major-2 modules run under the event-loop driver.

/// In-process multi-peer harness + the churn/failure drill machinery, driven by the production
/// coordinator through the [`coordinator_shell`] recording drive. Available to external
/// crates behind the `harness` feature, and to this crate's own tests via `cfg(test)`.
#[cfg(any(test, feature = "harness"))]
pub mod harness;

/// The coordinator replay sandbox: the concrete [`daemon_vhc_observe::CoordinatorSandbox`] the
/// replay oracle drives consensus through (consensus re-derives inside the sandboxed
/// `coordinator-quorum` module, never a native tick). Needs the host runtime + observe, so it lives
/// behind the `harness` feature (and this crate's own tests).
#[cfg(any(test, feature = "harness"))]
pub mod replay_sandbox;

/// The coordinator recording drive for the in-process whole-run harness: drives the
/// production `coordinator-quorum` module (event-driven, one tick per frame) instead of a native
/// tick, so a recorded run and its `replay` re-derivation share one coordinator substrate.
/// Behind the `harness` feature (needs the host runtime), and this crate's own tests.
#[cfg(any(test, feature = "harness"))]
pub mod coordinator_shell;

// `live_harness` (the iroh live-transport harness driving the v1 WasmBackend over the RoundEngine)
// RETIRED with the v1 driver at the Phase-E sunset; the loopback `harness` (StubBackend) remains
// the deterministic multi-peer substrate, and the live v2 lanes are the testkit's.

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
pub enum VhcRunError {
    /// The transport (control or payload plane) failed.
    #[error(transparent)]
    Net(#[from] daemon_vhc_net::VhcNetError),
    /// The data pipeline (manifest / batch mapping) failed.
    #[error(transparent)]
    Data(#[from] data::DataError),
    /// A round-lifecycle invariant was violated (warmup, digest, or checkpoint step).
    #[error("vhc run lifecycle error: {0}")]
    Lifecycle(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_net_errors() {
        let err: VhcRunError = daemon_vhc_net::VhcNetError::Transport("gossip".into()).into();
        assert!(err.to_string().contains("gossip"));
    }

    #[test]
    fn wraps_data_errors() {
        let err: VhcRunError = data::DataError::EmptyManifest.into();
        assert!(err.to_string().contains("no shards"));
    }
}
