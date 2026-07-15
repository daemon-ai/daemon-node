// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Run-health summary — plain, serializable per-round facts (spec §6.4, §14; TDD §3.9).
//!
//! The base for `daemon-cli swarm observe` / the app's run view: each round's committed count,
//! attested coverage, stragglers, drops, digest agreement, and activity span, derived **only** from
//! the signed [`MessageLog`]. No privileged coordinator state — the app renders the node's answer
//! (architecture invariant), and this is the node-side projection it renders.

use std::collections::BTreeSet;

use daemon_swarm_proto::messages::SwarmMessage;
use daemon_swarm_proto::PeerId;
use serde::{Deserialize, Serialize};

use crate::log::{round_of, MessageKind, MessageLog};

/// Per-round facts distilled from the message log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundHealth {
    /// The round.
    pub round: u64,
    /// Committed peers this round (the `RoundRecord`'s set count; `0` if no record was published).
    pub committed: u32,
    /// Distinct witnesses that attested this round (attestation coverage).
    pub attested_coverage: u32,
    /// Peers that reported straggling this round, sorted by peer.
    pub stragglers: Vec<PeerId>,
    /// Peers dropped by the round record, sorted by peer.
    pub drops: Vec<PeerId>,
    /// Distinct peers that reported a post-ingest digest this round.
    pub digest_reporters: u32,
    /// Whether every digest reporter agreed (one distinct digest, ≥ 1 reporter).
    pub digest_agreed: bool,
    /// Activity span: arrival-order distance between the round's first and last log record. A coarse
    /// duration proxy (the log carries no wall clock); `0` for a single-record round.
    pub duration_ticks: u64,
    /// Whether a `RoundRecord` was published for this round (finalized).
    pub finalized: bool,
}

/// The whole-run health projection: per-round facts in ascending round order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunHealth {
    /// The run this summary describes.
    pub run_id: String,
    /// Per-round facts, ascending by round.
    pub rounds: Vec<RoundHealth>,
}

impl RunHealth {
    /// Project a [`MessageLog`] into per-round health facts.
    #[must_use]
    pub fn from_log(log: &MessageLog) -> Self {
        let rounds = log
            .rounds()
            .into_iter()
            .map(|round| round_health(log, round))
            .collect();
        Self {
            run_id: log.run_id().to_string(),
            rounds,
        }
    }
}

fn round_health(log: &MessageLog, round: u64) -> RoundHealth {
    // Committed count + drops from the round record (if finalized).
    let record = log
        .by_round_kind(round, MessageKind::RoundRecord)
        .filter_map(|m| match &m.payload {
            SwarmMessage::RoundRecord(r) => Some(r.clone()),
            _ => None,
        })
        .last();
    let finalized = record.is_some();
    let committed = record.as_ref().map_or(0, |r| r.set.count);
    let mut drops = record.map_or_else(Vec::new, |r| r.drops);
    drops.sort_unstable();
    drops.dedup();

    // Attestation coverage = distinct witnesses that attested.
    let attesters: BTreeSet<PeerId> = log
        .by_round_kind(round, MessageKind::Attestation)
        .map(|m| m.signer)
        .collect();

    // Stragglers = distinct signers of Straggle messages.
    let stragglers: Vec<PeerId> = log
        .by_round_kind(round, MessageKind::Straggle)
        .map(|m| m.signer)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    // Digest agreement across distinct reporters.
    let mut reporters: BTreeSet<PeerId> = BTreeSet::new();
    let mut digests: BTreeSet<_> = BTreeSet::new();
    for m in log.by_round_kind(round, MessageKind::Digest) {
        if let SwarmMessage::Digest(d) = &m.payload {
            reporters.insert(m.signer);
            digests.insert(d.digest);
        }
    }
    let digest_reporters = reporters.len() as u32;
    let digest_agreed = digest_reporters > 0 && digests.len() == 1;

    RoundHealth {
        round,
        committed,
        attested_coverage: attesters.len() as u32,
        stragglers,
        drops,
        digest_reporters,
        digest_agreed,
        duration_ticks: round_span(log, round),
        finalized,
    }
}

/// Arrival-order distance between the first and last log record pertaining to `round`.
fn round_span(log: &MessageLog, round: u64) -> u64 {
    let positions: Vec<usize> = log
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, m)| round_of(&m.payload) == Some(round))
        .map(|(i, _)| i)
        .collect();
    match (positions.first(), positions.last()) {
        (Some(&first), Some(&last)) => (last - first) as u64,
        _ => 0,
    }
}
