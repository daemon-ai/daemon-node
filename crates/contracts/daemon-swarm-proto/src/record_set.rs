// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `record-set.cbor` object codec (spec §6.4, §11.3; TDD PROTO-5 / RUN-2).
//!
//! A [`RoundRecord`](crate::messages::RoundRecord) signs a **set commitment** (root + count), not the
//! set itself, so the consensus message stays scale-invariant (§6.4). The full committed set — the
//! ordered `[(peer, hash, size)]` list — is published alongside as a content-addressed object,
//! `runs/<run>/rounds/<r>/record-set.cbor` (§11.3). A replaying peer or the observer fetches it,
//! verifies it against the record's root, and stages it in record order (§6.4 barrier, invariant I3).
//!
//! This type is that object. It is **canonical CBOR of the sorted entries**; its blake3 content hash
//! is the locator hash, and [`RecordSet::commitment`] reproduces exactly the
//! [`SetCommitment`](crate::merkle::SetCommitment) the `RoundRecord` signs, so
//! [`RecordSet::verify_against`] is an exact (non-probabilistic) membership guarantee. Entries are
//! ordered by node public-key bytes then payload hash (invariant I3), matching `commit_set`.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId};
use crate::canonical::{from_canonical_slice, to_canonical_vec};
use crate::error::SwarmProtoError;
use crate::hash::blake3_hash;
use crate::merkle::{commit_set, SetCommitment};
use crate::messages::RecordEntry;

/// The committed-set object a `RoundRecord`'s root signs (`record-set.cbor`, §11.3).
///
/// Serializes as a CBOR map `{ "entries": [* record-entry] }`. Build via [`RecordSet::new`], which
/// imposes the canonical order (by peer bytes, then hash, then size) + de-duplication so the content
/// hash is stable regardless of input order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordSet {
    /// The committed `(peer, hash, size)` entries, sorted by node public-key bytes (I3).
    pub entries: Vec<RecordEntry>,
}

impl RecordSet {
    /// Build a record-set object from entries, imposing the canonical `(peer, hash, size)` order and
    /// de-duplicating — so the same committed set always encodes to the same bytes (content address).
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = RecordEntry>) -> Self {
        let mut entries: Vec<RecordEntry> = entries.into_iter().collect();
        entries.sort_by(|a, b| {
            a.peer
                .cmp(&b.peer)
                .then_with(|| a.hash.cmp(&b.hash))
                .then_with(|| a.size.cmp(&b.size))
        });
        entries.dedup();
        Self { entries }
    }

    /// The canonical CBOR encoding of the object (RFC 8949 §4.2, the content-address preimage).
    pub fn to_canonical_vec(&self) -> Result<Vec<u8>, SwarmProtoError> {
        to_canonical_vec(self)
    }

    /// Decode a record-set object from CBOR bytes. Order is not trusted from the wire — callers that
    /// need the object's content address must have produced it via [`RecordSet::new`]; the exact
    /// (order-independent) consensus check is [`RecordSet::verify_against`].
    pub fn from_canonical_slice(bytes: &[u8]) -> Result<Self, SwarmProtoError> {
        from_canonical_slice(bytes)
    }

    /// The object's content address: blake3 of its canonical CBOR (the locator hash, §11.3).
    pub fn content_hash(&self) -> Result<Hash, SwarmProtoError> {
        Ok(blake3_hash(&self.to_canonical_vec()?))
    }

    /// The set commitment (merkle root + count) over the object's `(peer, hash)` pairs — byte-for-byte
    /// the [`SetCommitment`] a [`RoundRecord`](crate::messages::RoundRecord) signs for this set.
    #[must_use]
    pub fn commitment(&self) -> SetCommitment {
        let pairs: Vec<(PeerId, Hash)> = self.entries.iter().map(|e| (e.peer, e.hash)).collect();
        commit_set(&pairs).commitment()
    }

    /// Verify the object reconstructs `commitment` (the record's signed root + count). Order- and
    /// duplicate-independent: [`commit_set`] re-sorts + de-duplicates, so a witness that received the
    /// object in any order still gets an exact membership guarantee (§6.4 I3).
    pub fn verify_against(&self, commitment: &SetCommitment) -> Result<(), SwarmProtoError> {
        if self.commitment() == *commitment {
            Ok(())
        } else {
            Err(SwarmProtoError::Merkle(
                "record-set object does not reconstruct the record's set commitment".into(),
            ))
        }
    }

    /// The sorted `(peer, hash, size)` entries.
    #[must_use]
    pub fn entries(&self) -> &[RecordEntry] {
        &self.entries
    }

    /// The number of committed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(n: u8) -> PeerId {
        PeerId([n; 32])
    }

    fn hash(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn entry(p: u8, h: u8, size: u64) -> RecordEntry {
        RecordEntry {
            peer: peer(p),
            hash: hash(h),
            size,
        }
    }

    #[test]
    fn new_sorts_and_dedups() {
        let rs = RecordSet::new([
            entry(3, 9, 1),
            entry(1, 2, 3),
            entry(1, 2, 3),
            entry(2, 5, 7),
        ]);
        assert_eq!(rs.len(), 3, "duplicate collapsed");
        assert_eq!(rs.entries()[0].peer, peer(1));
        assert_eq!(rs.entries()[1].peer, peer(2));
        assert_eq!(rs.entries()[2].peer, peer(3));
    }

    #[test]
    fn empty_set_hash_is_stable() {
        let a = RecordSet::new([]);
        let b = RecordSet::default();
        assert!(a.is_empty());
        assert_eq!(a.content_hash().unwrap(), b.content_hash().unwrap());
    }
}
