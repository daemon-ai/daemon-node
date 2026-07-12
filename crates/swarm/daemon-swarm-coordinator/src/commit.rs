// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The round-record commit rule (spec §6.4 I6; TDD PROTO-5).
//!
//! A **pure function of signed evidence**: a peer's payload enters the round record iff its
//! `Commitment` arrived **and** availability evidence exists — a `StorageReceipt` covering
//! `(peer, hash)` **or** a witness-quorum of `Attestation`s covering it. No I/O happens inside the
//! rule (the coordinator's `HEAD`s already arrived as signed `StorageReceipt` inputs), so the whole
//! round is re-executable and provable on a substrate that cannot perform I/O (§11.2, §18 q.14).
//!
//! Attestation coverage reads the `inline` witness set (the small-roster transport path, §6.4) —
//! the signed field is still the merkle root; the root-only + membership-proof path is Wave 3
//! (ledger-P2 decision 4).

use daemon_swarm_proto::assignment::witness_quorum;
use daemon_swarm_proto::messages::RecordEntry;
use daemon_swarm_proto::{Hash, PeerId};

use crate::state::{Member, RoundState};

/// Whether availability evidence exists for `(peer, hash)` in this round (§6.4 I6).
#[must_use]
pub fn has_evidence(rs: &RoundState, peer: &PeerId, hash: &Hash) -> bool {
    // Storage-receipt path: the coordinator-as-storage-client HEAD-verified the object.
    let by_receipt = rs
        .receipts
        .iter()
        .any(|e| &e.peer == peer && &e.hash == hash);
    if by_receipt {
        return true;
    }
    // Witness-quorum path: count round witnesses whose inline set covers (peer, hash).
    let quorum = witness_quorum(rs.witnesses.len() as u32);
    if quorum == 0 {
        return false;
    }
    let covering = rs
        .witnesses
        .iter()
        .filter(|w| {
            rs.attestations.get(w).is_some_and(|a| {
                a.inline
                    .as_ref()
                    .is_some_and(|set| set.iter().any(|ae| &ae.peer == peer && &ae.hash == hash))
            })
        })
        .count();
    covering as u32 >= quorum
}

/// The committed set for this round: every healthy member whose commitment is evidenced, as sorted
/// `(peer, hash, size)` entries ordered by node public-key bytes (§6.4 I3).
#[must_use]
pub fn committed_entries(rs: &RoundState, roster: &[Member]) -> Vec<RecordEntry> {
    let mut entries: Vec<RecordEntry> = Vec::new();
    for m in roster.iter().filter(|m| m.is_healthy()) {
        if let Some(c) = rs.commitments.get(&m.peer) {
            if has_evidence(rs, &m.peer, &c.payload) {
                entries.push(RecordEntry {
                    peer: m.peer,
                    hash: c.payload,
                    size: c.size,
                });
            }
        }
    }
    entries.sort_by_key(|a| a.peer);
    entries
}

/// Whether every healthy member has submitted a commitment (the early `RoundTrain → RoundWitness`
/// advance condition, §6.2/§6.4).
#[must_use]
pub fn all_committed(rs: &RoundState, roster: &[Member]) -> bool {
    let healthy: Vec<&Member> = roster.iter().filter(|m| m.is_healthy()).collect();
    if healthy.is_empty() {
        return false;
    }
    healthy.iter().all(|m| rs.commitments.contains_key(&m.peer))
}

/// Whether every submitted commitment now has availability evidence (the early `RoundWitness →
/// commit` advance condition, §6.4).
#[must_use]
pub fn all_evidenced(rs: &RoundState, roster: &[Member]) -> bool {
    let mut any = false;
    for m in roster.iter().filter(|m| m.is_healthy()) {
        if let Some(c) = rs.commitments.get(&m.peer) {
            any = true;
            if !has_evidence(rs, &m.peer, &c.payload) {
                return false;
            }
        }
    }
    any
}
