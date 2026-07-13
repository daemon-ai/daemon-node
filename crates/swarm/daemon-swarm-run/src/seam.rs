// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared identity/hash vocabulary for the participant runtime.
//!
//! The shared types are re-exported from [`daemon_swarm_net::seam`], which (as of Merge 1) resolves
//! them to the canonical [`daemon_swarm_proto`] types — the content hash is proto's blake3 `Hash`
//! and the peer identity is proto's `PeerId`. [`BatchId`] is defined here because §6.3 assignment
//! produces `BatchId` intervals and the runtime only maps `BatchId → (shard, offset)`; proto models
//! the interval (`messages::BatchWindow`) but not the id, which stays a bare `u64` primitive.

pub use daemon_swarm_net::seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};

/// The index of one training sequence within an epoch's data window (spec §6.3).
///
/// Deterministic assignment (§6.3) splits the round's global batch into contiguous `BatchId`
/// intervals across peers; the runtime maps each `BatchId` to a `(shard, offset)` locally, with no
/// per-batch RPC. Proto carries the *interval* as `messages::BatchWindow { start, end }` over the
/// same `u64` id space; there is no distinct proto `BatchId` newtype to swap for.
pub type BatchId = u64;
