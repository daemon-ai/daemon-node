// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`Deduper`] — reusable content-hash dedupe for the control plane (spec §7.1; TDD NET-6).
//!
//! Gossip is *dissemination, never arbitration* (§7.1): the same signed message may arrive over
//! more than one path (WS push and gossip fanout), so every control plane de-duplicates by the
//! blake3 of the raw message bytes before delivering it. Wave 1 baked that set into
//! [`LoopbackGossip`](crate::gossip::LoopbackGossip); Wave 2 factors it into this small type so any
//! future carrier (a real iroh gossip mesh, a WS bridge) reuses the identical rule instead of
//! re-implementing it — the NET-6 property "the same message via WS and gossip dedupes" is one
//! implementation, shared.
//!
//! The dedupe key is proto's canonical [`blake3_hash`](daemon_swarm_proto::blake3_hash) over the
//! opaque message bytes — the same content address the payload plane and the signed envelope use,
//! so a message's identity is stable across re-encodings that preserve its bytes.

use std::collections::HashSet;

use daemon_swarm_proto::{blake3_hash, Hash};

/// A content-hash dedupe set over opaque control-message bytes.
///
/// Not internally synchronized — a carrier that shares it across tasks wraps it in its own lock
/// (as [`LoopbackGossip`](crate::gossip::LoopbackGossip) does), keeping this type allocation-cheap
/// and lock-free for single-owner uses.
#[derive(Debug, Default, Clone)]
pub struct Deduper {
    seen: HashSet<[u8; Hash::LEN]>,
}

impl Deduper {
    /// A fresh, empty deduper.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The content id used for dedupe: proto's blake3 over the raw message bytes.
    #[must_use]
    pub fn id(message: &[u8]) -> Hash {
        blake3_hash(message)
    }

    /// Record `message` as seen, returning `true` if it is **new** (should be disseminated) or
    /// `false` if it is a duplicate (already delivered — drop it).
    pub fn observe(&mut self, message: &[u8]) -> bool {
        self.seen.insert(*blake3_hash(message).as_bytes())
    }

    /// Whether `message` has already been observed, without recording it.
    #[must_use]
    pub fn contains(&self, message: &[u8]) -> bool {
        self.seen.contains(blake3_hash(message).as_bytes())
    }

    /// The number of distinct messages observed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether nothing has been observed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sight_is_new_repeat_is_duplicate() {
        let mut d = Deduper::new();
        assert!(d.observe(b"round-open"), "first sighting is new");
        assert!(!d.observe(b"round-open"), "second sighting is a duplicate");
        // A distinct message is still new.
        assert!(d.observe(b"commitment"));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn contains_does_not_record() {
        let mut d = Deduper::new();
        assert!(!d.contains(b"m"));
        assert!(d.observe(b"m"));
        assert!(d.contains(b"m"));
    }

    #[test]
    fn id_is_proto_blake3() {
        assert_eq!(Deduper::id(b"x"), blake3_hash(b"x"));
    }
}
