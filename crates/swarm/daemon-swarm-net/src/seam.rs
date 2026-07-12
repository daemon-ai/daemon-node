// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared swarm identity / hash vocabulary.
//!
//! Merge 1 swapped the Wave-1 placeholders for the canonical [`daemon_swarm_proto`] types: the
//! content hash is proto's blake3 [`Hash`](daemon_swarm_proto::Hash) (re-exported here as
//! [`ContentHash`]) and the peer identity is proto's [`PeerId`](daemon_swarm_proto::PeerId). The
//! run-scoped locator types proto does not model — run ids stay opaque strings (proto carries
//! `run_id: String`), and the payload-store key is a transport concern — remain local, but are
//! re-expressed over the proto primitives (`PayloadKey` keys on the proto `PeerId`).

use serde::{Deserialize, Serialize};

/// A content-addressed identity over opaque bytes: proto's canonical blake3 [`Hash`].
///
/// The whole swarm integrity model — artifacts, round payloads, checkpoints — is blake3-addressed
/// (spec §6.4, the delta from Psyche's sha256). Compute one with
/// [`daemon_swarm_proto::blake3_hash`].
///
/// [`Hash`]: daemon_swarm_proto::Hash
pub use daemon_swarm_proto::Hash as ContentHash;

/// A peer's ed25519 **node** identity public key (spec §7.2 — never the iroh `NodeId`).
///
/// Proto's [`PeerId`](daemon_swarm_proto::PeerId): 32 raw bytes, ordered lexicographically by those
/// bytes, which is the total order the `RoundRecord`'s committed set and proto's `commit_set` use
/// (§6.4 I3).
pub use daemon_swarm_proto::PeerId;

/// A round number within a run.
///
/// Proto keeps round ids as bare `u64` throughout its message set (`RoundOpen::round`,
/// `Commitment::round`, …), so the runtime uses the same primitive rather than a newtype.
pub type RoundId = u64;

/// An opaque run identifier.
///
/// Proto has no dedicated run-id type — it carries run ids as `String` (`Join::run_id`) — so this
/// stays a local newtype over the same primitive.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub String);

impl RunId {
    /// Wrap a string as a run id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// The run id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The address of one payload object in a payload plane: `(run, round, peer)`.
///
/// A trainer publishes exactly one update object per round, so `(run, round, peer)` is the natural
/// key; the content hash is carried separately (verified on `get`). This is a transport-plane
/// locator (not a signed wire field), so it stays local, keyed over the proto [`PeerId`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PayloadKey {
    /// The run this payload belongs to.
    pub run: RunId,
    /// The round this payload was produced in.
    pub round: RoundId,
    /// The peer that produced it (node pubkey).
    pub peer: PeerId,
}

impl PayloadKey {
    /// Construct a payload key.
    pub fn new(run: RunId, round: RoundId, peer: PeerId) -> Self {
        Self { run, round, peer }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_swarm_proto::blake3_hash;

    #[test]
    fn content_hash_is_proto_blake3() {
        // ContentHash is proto's blake3 Hash; `blake3_hash` computes it, `to_hex` renders 64 chars.
        let h = blake3_hash(b"daemon-swarm");
        assert_eq!(h.to_hex().len(), 64);
        assert_eq!(h, blake3_hash(b"daemon-swarm"));
    }

    #[test]
    fn peer_id_orders_by_bytes() {
        let a = PeerId([0u8; 32]);
        let mut b_bytes = [0u8; 32];
        b_bytes[31] = 1;
        let b = PeerId(b_bytes);
        assert!(a < b);
    }

    #[test]
    fn payload_key_round_trips_fields() {
        let k = PayloadKey::new(RunId::new("run-x"), 7, PeerId([0x11; 32]));
        assert_eq!(k.run.as_str(), "run-x");
        assert_eq!(k.round, 7);
        assert_eq!(k.peer, PeerId([0x11; 32]));
    }
}
