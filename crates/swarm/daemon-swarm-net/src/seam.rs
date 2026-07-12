// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! MERGE-1 placeholder identity + hash types.
//!
//! Lane P (`daemon-swarm-proto`) owns the canonical wire types, but its API is not available to
//! lane R this wave beyond the Wave-0 scaffold (which exports only [`SWARM_PROTO_VERSION`] and
//! [`SwarmProtoError`]). Everything here is a minimal LOCAL stand-in, sized to be mechanically
//! swappable at Merge 1: newtypes / plain structs with construction + hex/round-trip helpers only,
//! no behavior the proto crate would not also provide.
//!
//! Each item carries a `// MERGE-1: replace with daemon_swarm_proto::…` marker.
//!
//! [`SWARM_PROTO_VERSION`]: daemon_swarm_proto::SWARM_PROTO_VERSION
//! [`SwarmProtoError`]: daemon_swarm_proto::SwarmProtoError

use serde::{Deserialize, Serialize};

/// A content-addressed identity over opaque bytes (blake3, 32 bytes).
///
/// The whole swarm integrity model — artifacts, round payloads, checkpoints — is blake3-addressed
/// (spec §6.4, the delta from Psyche's sha256). Merge 1 replaces this with the proto crate's
/// canonical hash type; keep it blake3.
///
// MERGE-1: replace with daemon_swarm_proto::ContentHash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// The blake3 hash of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// The raw 32-byte digest.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex rendering (64 chars).
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
            s.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
        }
        s
    }

    /// Parse a 64-char lowercase/uppercase hex digest.
    pub fn from_hex(s: &str) -> Result<Self, HashParseError> {
        if s.len() != 64 {
            return Err(HashParseError::Length(s.len()));
        }
        let mut out = [0u8; 32];
        let bytes = s.as_bytes();
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let hi = (chunk[0] as char)
                .to_digit(16)
                .ok_or(HashParseError::Digit)?;
            let lo = (chunk[1] as char)
                .to_digit(16)
                .ok_or(HashParseError::Digit)?;
            out[i] = ((hi << 4) | lo) as u8;
        }
        Ok(Self(out))
    }
}

impl core::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ContentHash({})", self.to_hex())
    }
}

/// A `ContentHash::from_hex` parse failure.
#[derive(Debug, thiserror::Error)]
pub enum HashParseError {
    /// The hex string was not exactly 64 characters.
    #[error("content hash hex must be 64 chars, got {0}")]
    Length(usize),
    /// A character was not a valid hex digit.
    #[error("content hash hex contains a non-hex digit")]
    Digit,
}

/// An opaque run identifier.
///
// MERGE-1: replace with daemon_swarm_proto::RunId.
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

/// A round number within a run.
///
// MERGE-1: replace with daemon_swarm_proto::RoundId (watch for a proto newtype).
pub type RoundId = u64;

/// A peer's ed25519 **node** identity public key bytes (spec §7.2 — never the iroh `NodeId`).
///
/// The `RoundRecord`'s committed set is totally ordered by these bytes (§6.4 I3), so Merge 1 must
/// preserve the byte ordering when it swaps in the proto type.
///
// MERGE-1: replace with daemon_swarm_proto::PeerId.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PeerId(pub [u8; 32]);

impl PeerId {
    /// The raw 32-byte node public key.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex rendering (64 chars) — used as a filesystem-safe key segment.
    #[must_use]
    pub fn to_hex(&self) -> String {
        ContentHash(self.0).to_hex()
    }
}

impl core::fmt::Debug for PeerId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PeerId({})", self.to_hex())
    }
}

/// The address of one payload object in a payload plane: `(run, round, peer)`.
///
/// A trainer publishes exactly one update object per round, so `(run, round, peer)` is the natural
/// key; the content hash is carried separately (verified on `get`).
///
// MERGE-1: proto may fold this into the `Commitment` locator type.
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

    #[test]
    fn content_hash_hex_round_trips() {
        let h = ContentHash::of(b"daemon-swarm");
        let hex = h.to_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(ContentHash::from_hex(&hex).unwrap(), h);
    }

    #[test]
    fn content_hash_rejects_bad_hex() {
        assert!(matches!(
            ContentHash::from_hex("zz"),
            Err(HashParseError::Length(2))
        ));
        let mut bad = "a".repeat(64);
        bad.replace_range(0..1, "z");
        assert!(matches!(
            ContentHash::from_hex(&bad),
            Err(HashParseError::Digit)
        ));
    }

    #[test]
    fn peer_id_orders_by_bytes() {
        let a = PeerId([0u8; 32]);
        let mut b_bytes = [0u8; 32];
        b_bytes[31] = 1;
        let b = PeerId(b_bytes);
        assert!(a < b);
    }
}
