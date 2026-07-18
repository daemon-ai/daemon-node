// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator's resolved run configuration (spec §6.1/§6.2).
//!
//! [`RunConfig`] is the coordination-consumed run schedule/membership shape, authored directly by
//! the run author as the coordinator module's opaque config (a genesis envelope v2 carries it
//! verbatim inside the coordinator role's `{state: CoordinatorState}` — the host never interprets
//! it, §4.3 seam rule), plus the coordinator-only knobs ([`CoordinatorParams`]). It is part of
//! [`crate::CoordinatorState`] and therefore canonical-CBOR-serializable (the replay foundation,
//! PROTO-20).

use crate::assignment::WITNESS_TARGET_DEFAULT;
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::{CapabilitySet, Hash, PeerId, VhcProtoVersion};
use serde::{Deserialize, Serialize};

/// Default K record-absences before a peer is dropped (§6.4 daemon Delta; TDD PROTO-7).
pub const K_ABSENCES_DEFAULT: u32 = 3;

/// Coordinator-only run parameters that the frozen envelope does not carry (ledger-P2 note).
///
/// Supplied at run creation (Wave-3 authoring), never read from `[experiment.config]` at runtime, so
/// the seam rule (§4.3) is preserved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorParams {
    /// Tokens per sequence — converts the `[data].global_batch` (sequences/round) into tokens for a
    /// `[data].stop = { tokens }` termination (§6.1). `1` means "count sequences as tokens".
    pub seq_len: u64,
    /// Target witness-committee size (§6.3). `0` means "every peer witnesses".
    pub witness_target: u32,
    /// Deliberate batch overlap in basis points (0–10000; §6.3), 0 = exact partition.
    pub overlap_bps: u32,
    /// K record-absences before a peer is dropped (§6.4).
    pub k_absences: u32,
    /// Verifier-committee sampling percent (§12) — `0` keeps the seam a no-op (TDD PROTO-15).
    pub verification_percent: u32,
    /// Principals (node identities) authorized to pause/resume (§11.1; TDD PROTO-14).
    pub authorized: Vec<PeerId>,
}

impl Default for CoordinatorParams {
    fn default() -> Self {
        Self {
            seq_len: 1,
            witness_target: WITNESS_TARGET_DEFAULT,
            overlap_bps: 0,
            k_absences: K_ABSENCES_DEFAULT,
            verification_percent: 0,
            authorized: Vec::new(),
        }
    }
}

/// The resolved, coordination-consumed run configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Run identity (`[run].run_id`).
    pub run_id: String,
    /// The vhc proto version this run is pinned to (exact-match join gate, §16).
    pub proto_version: VhcProtoVersion,
    /// blake3 hash of the frozen envelope (§6.1) — the admission envelope-hash anchor.
    pub envelope_hash: Hash,
    /// The run's required capability set (`[requirements].capabilities`, §6.5).
    pub required_capabilities: CapabilitySet,
    /// `min_peers` floor to leave `WaitingForMembers` (§6.2).
    pub min_peers: u32,
    /// `max_peers` roster ceiling.
    pub max_peers: u32,
    /// Warmup timeout (seconds).
    pub warmup_s: u64,
    /// Max training time per round (seconds).
    pub round_train_max_s: u64,
    /// Witness grace window (seconds).
    pub round_witness_s: u64,
    /// Cooldown duration (seconds).
    pub cooldown_s: u64,
    /// Rounds per epoch (roster-stable span, §6.2).
    pub epoch_rounds: u64,
    /// Fetch-recovery budget before a stalled peer must leave (§6.4).
    pub stall_rounds_max: u32,
    /// Sequences-per-round schedule (`[data].global_batch`, §6.1).
    pub global_batch: GlobalBatch,
    /// Termination condition (`[data].stop`, §6.2).
    pub stop: StopCondition,
    /// Inner steps per round (H) — carried for peers, not consumed by `tick` (§6.1).
    pub steps_per_round: u32,
    /// Tokens per sequence (coordinator-only, [`CoordinatorParams`]).
    pub seq_len: u64,
    /// Target witness-committee size (coordinator-only).
    pub witness_target: u32,
    /// Deliberate batch overlap in basis points (coordinator-only).
    pub overlap_bps: u32,
    /// K record-absences drop threshold (coordinator-only).
    pub k_absences: u32,
    /// Verifier-committee sampling percent (coordinator-only).
    pub verification_percent: u32,
    /// Principals authorized to pause/resume (coordinator-only).
    pub authorized: Vec<PeerId>,
}
