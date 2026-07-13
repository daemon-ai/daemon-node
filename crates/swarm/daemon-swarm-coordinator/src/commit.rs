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
//! Attestation coverage has two transport shapes, both grounded in the merkle root the witness
//! signs (§6.4):
//!
//! * **Inline-quorum** (small rosters) — a witness-quorum whose *inline* sets each cover
//!   `(peer, hash)`. This was the Merge-2 path.
//! * **Root-only** (Wave 3, scale-invariant) — witnesses may attest a **bare root** (inline
//!   omitted). When a quorum agree on the same root `R`, membership of `(peer, hash)` in `R` is
//!   pinned by a `StorageReceipt` **or** a single inline opening that reconstructs `R`
//!   (a full opening whose recomputed root equals `R` *is* the committed set, so membership stays
//!   exact). This is "StorageReceipts + root equality across the witness quorum" (§6.4). A future
//!   coordinator that *holds* membership proofs would use [`SetCommitment::verify_membership`] on the
//!   agreed root instead; that proof path is frozen in proto since Merge 1.

use std::collections::BTreeMap;

use daemon_swarm_proto::assignment::witness_quorum;
use daemon_swarm_proto::messages::RecordEntry;
use daemon_swarm_proto::{commit_set, Hash, PeerId, Root};

use crate::state::{Member, RoundState};

/// Whether availability evidence exists for `(peer, hash)` in this round (§6.4 I6).
#[must_use]
pub fn has_evidence(rs: &RoundState, peer: &PeerId, hash: &Hash) -> bool {
    // 1. Storage-receipt path: the coordinator-as-storage-client HEAD-verified the object.
    let by_receipt = rs
        .receipts
        .iter()
        .any(|e| &e.peer == peer && &e.hash == hash);
    if by_receipt {
        return true;
    }
    let quorum = witness_quorum(rs.witnesses.len() as u32);
    if quorum == 0 {
        return false;
    }
    // 2. Inline-quorum path: a quorum of round witnesses whose inline set covers (peer, hash).
    let covering = rs
        .witnesses
        .iter()
        .filter(|w| witness_inline_covers(rs, w, peer, hash))
        .count();
    if covering as u32 >= quorum {
        return true;
    }
    // 3. Root-only path: a quorum agree on a root R, and one inline opening reconstructs R and holds
    //    (peer, hash) — so most witnesses can attest a bare root without under-covering (§6.4).
    if let Some(root) = quorum_root(rs) {
        if opening_pins_membership(rs, &root, peer, hash) {
            return true;
        }
    }
    false
}

/// The set root attested by at least a witness-quorum of the round's witnesses, if any — the
/// consensus a bare-root (inline-omitted) attestation still evidences (§6.4 root-only path).
#[must_use]
pub fn quorum_root(rs: &RoundState) -> Option<Root> {
    let quorum = witness_quorum(rs.witnesses.len() as u32);
    if quorum == 0 {
        return None;
    }
    let mut counts: BTreeMap<Root, u32> = BTreeMap::new();
    for w in &rs.witnesses {
        if let Some(a) = rs.attestations.get(w) {
            *counts.entry(a.set.root).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .find_map(|(root, n)| (n >= quorum).then_some(root))
}

/// Whether witness `w`'s inline attestation set covers `(peer, hash)`.
fn witness_inline_covers(rs: &RoundState, w: &PeerId, peer: &PeerId, hash: &Hash) -> bool {
    rs.attestations.get(w).is_some_and(|a| {
        a.inline
            .as_ref()
            .is_some_and(|set| set.iter().any(|ae| &ae.peer == peer && &ae.hash == hash))
    })
}

/// Whether some witness supplied an inline opening that *reconstructs* the agreed `root` and contains
/// `(peer, hash)` — the exact (non-probabilistic) membership pin for the root-only path. The declared
/// `set.root` is not trusted: the opening's root is recomputed with [`commit_set`].
fn opening_pins_membership(rs: &RoundState, root: &Root, peer: &PeerId, hash: &Hash) -> bool {
    rs.attestations.values().any(|a| {
        a.inline.as_ref().is_some_and(|set| {
            let pairs: Vec<(PeerId, Hash)> = set.iter().map(|ae| (ae.peer, ae.hash)).collect();
            commit_set(&pairs).commitment().root == *root
                && set.iter().any(|ae| &ae.peer == peer && &ae.hash == hash)
        })
    })
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
