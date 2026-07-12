// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! MERGE-1 placeholder types for the participant runtime.
//!
//! The shared identity/hash types are re-exported from [`daemon_swarm_net::seam`] (lane R owns them
//! this wave); [`BatchId`] is defined here because §6.3 assignment produces `BatchId` intervals and
//! the runtime only maps `BatchId → (shard, offset)`. All are swapped for `daemon-swarm-proto`
//! (lane P) at Merge 1.

pub use daemon_swarm_net::seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};

/// The index of one training sequence within an epoch's data window (spec §6.3).
///
/// Deterministic assignment (§6.3) splits the round's global batch into contiguous `BatchId`
/// intervals across peers; the runtime maps each `BatchId` to a `(shard, offset)` locally, with no
/// per-batch RPC.
///
// MERGE-1: replace with daemon_swarm_proto::BatchId if the proto crate owns the assignment type.
pub type BatchId = u64;
