// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Merkle set commitments (spec §6.4; TDD PROTO-5 root/proof half).
//!
//! The signed, consensus-critical field of an [`Attestation`](crate::messages::Attestation) and a
//! [`RoundRecord`](crate::messages::RoundRecord) is a **set commitment**: a blake3 merkle root over
//! the set's `(peer, payload-hash)` pairs, **sorted by node public-key bytes** (invariant I3), plus
//! the element count. The root is constant-size at any roster (≈32 B at n = 4 and at n = 4000), so
//! the consensus messages are scale-invariant while membership stays exactly provable via O(log n)
//! paths — no probabilistic structure is ever a consensus input (spec §18 open q. 12).
//!
//! Leaves and interior nodes are domain-separated (a `0x00`/`0x01` prefix byte) so no interior hash
//! can be reinterpreted as a leaf (second-preimage hardening). Odd levels duplicate the final node.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Root};
use crate::error::SwarmProtoError;
use crate::hash::blake3_hash;

const LEAF_DOMAIN: u8 = 0x00;
const NODE_DOMAIN: u8 = 0x01;
const EMPTY_ROOT_LABEL: &[u8] = b"daemon-swarm/merkle/empty/v1";

fn leaf_hash(peer: &PeerId, hash: &Hash) -> Hash {
    let mut buf = [0u8; 1 + PeerId::LEN + Hash::LEN];
    buf[0] = LEAF_DOMAIN;
    buf[1..1 + PeerId::LEN].copy_from_slice(peer.as_bytes());
    buf[1 + PeerId::LEN..].copy_from_slice(hash.as_bytes());
    blake3_hash(&buf)
}

fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut buf = [0u8; 1 + Hash::LEN + Hash::LEN];
    buf[0] = NODE_DOMAIN;
    buf[1..1 + Hash::LEN].copy_from_slice(left.as_bytes());
    buf[1 + Hash::LEN..].copy_from_slice(right.as_bytes());
    blake3_hash(&buf)
}

fn empty_root() -> Hash {
    blake3_hash(EMPTY_ROOT_LABEL)
}

/// The signed commitment to a set: its merkle root + element count (spec §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetCommitment {
    /// blake3 merkle root over the sorted `(peer, hash)` leaves.
    pub root: Root,
    /// The number of elements committed.
    pub count: u32,
}

/// An O(log n) membership proof against a [`SetCommitment`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MembershipProof {
    /// The leaf's index in the sorted set.
    pub leaf_index: u32,
    /// Sibling hashes from leaf level up to (but excluding) the root.
    pub siblings: Vec<Hash>,
}

/// A built merkle tree over a set, retaining the levels so membership proofs can be produced.
#[derive(Clone, Debug)]
pub struct SetCommitmentTree {
    entries: Vec<(PeerId, Hash)>,
    levels: Vec<Vec<Hash>>,
}

/// Build a set commitment tree from `(peer, payload-hash)` pairs. Input order is irrelevant: the
/// pairs are sorted by peer public-key bytes (then hash) and de-duplicated before the tree is built.
#[must_use]
pub fn commit_set(entries: &[(PeerId, Hash)]) -> SetCommitmentTree {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    sorted.dedup();

    let leaves: Vec<Hash> = sorted.iter().map(|(p, h)| leaf_hash(p, h)).collect();
    let mut levels: Vec<Vec<Hash>> = vec![leaves];

    if levels[0].is_empty() {
        levels.push(vec![empty_root()]);
    } else {
        while levels.last().map_or(0, Vec::len) > 1 {
            let cur = levels.last().expect("non-empty");
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                let left = cur[i];
                let right = if i + 1 < cur.len() {
                    cur[i + 1]
                } else {
                    cur[i]
                };
                next.push(node_hash(&left, &right));
                i += 2;
            }
            levels.push(next);
        }
    }

    SetCommitmentTree {
        entries: sorted,
        levels,
    }
}

impl SetCommitmentTree {
    /// The merkle root.
    #[must_use]
    pub fn root(&self) -> Root {
        Root(*self.levels.last().expect("at least one level")[0].as_bytes())
    }

    /// The signed commitment (root + count).
    #[must_use]
    pub fn commitment(&self) -> SetCommitment {
        SetCommitment {
            root: self.root(),
            count: self.entries.len() as u32,
        }
    }

    /// The sorted, de-duplicated `(peer, hash)` entries.
    #[must_use]
    pub fn entries(&self) -> &[(PeerId, Hash)] {
        &self.entries
    }

    /// Produce a membership proof for the element at sorted `index`, or `None` if out of range.
    #[must_use]
    pub fn prove(&self, index: usize) -> Option<MembershipProof> {
        if index >= self.entries.len() {
            return None;
        }
        let mut siblings = Vec::with_capacity(self.levels.len().saturating_sub(1));
        let mut idx = index;
        // Walk every level except the root.
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = idx ^ 1;
            let sibling = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                level[idx] // odd level: the node was duplicated with itself.
            };
            siblings.push(sibling);
            idx >>= 1;
        }
        Some(MembershipProof {
            leaf_index: index as u32,
            siblings,
        })
    }
}

impl SetCommitment {
    /// Verify that `(peer, hash)` is the element at `proof.leaf_index` of the committed set.
    pub fn verify_membership(
        &self,
        peer: &PeerId,
        hash: &Hash,
        proof: &MembershipProof,
    ) -> Result<(), SwarmProtoError> {
        if self.count == 0 {
            return Err(SwarmProtoError::Merkle("set is empty".into()));
        }
        if proof.leaf_index >= self.count {
            return Err(SwarmProtoError::Merkle("leaf index out of range".into()));
        }
        let mut idx = proof.leaf_index as usize;
        let mut cur = leaf_hash(peer, hash);
        for sibling in &proof.siblings {
            cur = if idx & 1 == 0 {
                node_hash(&cur, sibling)
            } else {
                node_hash(sibling, &cur)
            };
            idx >>= 1;
        }
        if Root(*cur.as_bytes()) == self.root {
            Ok(())
        } else {
            Err(SwarmProtoError::Merkle(
                "membership proof does not reconstruct the committed root".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(n: u32) -> PeerId {
        let mut b = [0u8; 32];
        b[..4].copy_from_slice(&n.to_be_bytes());
        PeerId(b)
    }

    fn hash(n: u32) -> Hash {
        let mut b = [0u8; 32];
        b[28..].copy_from_slice(&n.to_be_bytes());
        Hash(b)
    }

    #[test]
    fn empty_set_has_stable_root() {
        let t = commit_set(&[]);
        assert_eq!(t.commitment().count, 0);
        assert_eq!(t.root(), Root(*empty_root().as_bytes()));
    }

    #[test]
    fn single_element_root_is_leaf() {
        let t = commit_set(&[(peer(1), hash(1))]);
        assert_eq!(t.commitment().count, 1);
        assert_eq!(t.root(), Root(*leaf_hash(&peer(1), &hash(1)).as_bytes()));
    }
}
