// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Digest tally / desync detection (spec §5.6, §6.4, §9; TDD §3.9).
//!
//! Every peer emits a post-ingest [`Digest`](daemon_swarm_proto::messages::Digest) per round (§5.6).
//! Folding a round's digests yields a **quorum digest** (the value a quorum of peers agree on) and
//! the **outlier** set (peers that diverged) — the observe-driven desync trigger the runtime lane's
//! resync path consumes (§9). This crate produces the [`DesyncVerdict`]; wiring it into
//! `daemon-swarm-run`'s `checkpoint.rs` is lane R3's side (the marker it awaits).

use std::collections::BTreeMap;

use daemon_swarm_proto::{PeerId, StateDigest};
use serde::{Deserialize, Serialize};

use crate::log::{MessageKind, MessageLog};

/// The per-round digest agreement outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesyncVerdict {
    /// The round tallied.
    pub round: u64,
    /// The digest a quorum agreed on (the majority when it meets `quorum`), if any.
    pub quorum_digest: Option<StateDigest>,
    /// Peers whose digest differs from `quorum_digest` (empty ⇒ full agreement). Sorted by peer.
    pub outliers: Vec<PeerId>,
    /// How many distinct peers reported a digest for the round.
    pub reporters: u32,
    /// Whether every reporter agreed (no outliers and at least one reporter).
    pub agreed: bool,
}

impl DesyncVerdict {
    /// Whether the round is in desync (a quorum digest exists and at least one peer diverged).
    #[must_use]
    pub fn is_desync(&self) -> bool {
        self.quorum_digest.is_some() && !self.outliers.is_empty()
    }
}

/// Fold a round's `(peer, digest)` reports into a [`DesyncVerdict`]. `quorum` is the number of
/// agreeing peers required for a digest to count as the quorum digest (e.g.
/// [`daemon_swarm_proto::assignment::witness_quorum`] of the roster). The last report from a peer
/// wins (a peer only has one true post-ingest digest per round).
#[must_use]
pub fn digest_tally(
    round: u64,
    reports: impl IntoIterator<Item = (PeerId, StateDigest)>,
    quorum: u32,
) -> DesyncVerdict {
    // Last-write-wins per peer (dedupe), keeping a deterministic peer order.
    let mut by_peer: BTreeMap<PeerId, StateDigest> = BTreeMap::new();
    for (peer, digest) in reports {
        by_peer.insert(peer, digest);
    }
    let reporters = by_peer.len() as u32;

    // Tally digests; pick the most-agreed (ties broken by digest bytes for determinism).
    let mut counts: BTreeMap<StateDigest, u32> = BTreeMap::new();
    for digest in by_peer.values() {
        *counts.entry(*digest).or_insert(0) += 1;
    }
    let top = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(d, n)| (*d, *n));

    let quorum_digest = top.and_then(|(d, n)| (n >= quorum.max(1)).then_some(d));
    let outliers: Vec<PeerId> = match quorum_digest {
        Some(d) => by_peer
            .iter()
            .filter(|(_, dg)| **dg != d)
            .map(|(p, _)| *p)
            .collect(),
        None => Vec::new(),
    };
    let agreed = reporters > 0 && quorum_digest.is_some() && outliers.is_empty();

    DesyncVerdict {
        round,
        quorum_digest,
        outliers,
        reporters,
        agreed,
    }
}

/// Tally a round directly from a [`MessageLog`], extracting its `Digest` messages (§6.4).
#[must_use]
pub fn digest_tally_from_log(log: &MessageLog, round: u64, quorum: u32) -> DesyncVerdict {
    let reports = log
        .by_round_kind(round, MessageKind::Digest)
        .filter_map(|m| match &m.payload {
            daemon_swarm_proto::messages::SwarmMessage::Digest(d) => Some((m.signer, d.digest)),
            _ => None,
        });
    digest_tally(round, reports, quorum)
}
