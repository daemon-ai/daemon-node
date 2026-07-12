// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! blake3 content hashing.
//!
//! blake3 is the content-address function for every artifact, payload, and checkpoint in the swarm
//! (spec §5.6, §7.3 — replacing Psyche's sha256). The per-round *comparison* digest is xxh3-128
//! (see [`crate::digest`]); everything content-addressed here is full blake3.

use crate::bytes::Hash;

/// blake3 hash of `data`.
#[must_use]
pub fn blake3_hash(data: &[u8]) -> Hash {
    Hash(*blake3::hash(data).as_bytes())
}

/// blake3 keyed hash of `data` (domain separation for merkle leaves / nodes; see [`crate::merkle`]).
#[must_use]
pub fn blake3_keyed(key: &[u8; 32], data: &[u8]) -> Hash {
    Hash(*blake3::keyed_hash(key, data).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_matches_known_vector() {
        // Official blake3 test vector for the empty input.
        assert_eq!(
            blake3_hash(b"").to_hex(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    #[test]
    fn blake3_is_deterministic_and_sensitive() {
        assert_eq!(blake3_hash(b"round-42"), blake3_hash(b"round-42"));
        assert_ne!(blake3_hash(b"round-42"), blake3_hash(b"round-43"));
    }
}
