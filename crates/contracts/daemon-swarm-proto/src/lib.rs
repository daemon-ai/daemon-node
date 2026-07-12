// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-proto` — the swarm-training consensus / wire contract.
//!
//! Canonical CBOR codec, run-envelope schema + freeze/verify, capability-set admission, merkle set
//! commitments, the seven round messages + their CDDL, the round state-digest schedule, and the
//! [`SwarmProtoVersion`]. This crate is the single authority for the swarm wire shapes shared by
//! the host, the participant runtime, and the (wasm32) coordinator DO — see
//! `docs/specs/swarm-training-spec.md` §6, §7.3, §10.1, §16.
//!
//! **wasm32-clean by construction:** the only dependencies are `serde`, `ciborium`, `blake3`,
//! `xxhash-rust`, and `ed25519-dalek` — no `tokio`, Burn, or wasmtime — so it builds for the
//! `wasm32-unknown-unknown` coordinator target (§11.2). Signing uses only deterministic
//! ed25519 operations (no RNG on the crate's non-test paths).

#![forbid(unsafe_code)]

pub mod bytes;
pub mod canonical;
pub mod envelope;
pub mod error;
pub mod hash;
pub mod sign;

pub use bytes::{Hash, PeerId, Root, Seed, Signature};
pub use canonical::{from_canonical_slice, to_canonical_vec};
pub use envelope::{Envelope, FrozenEnvelope, ENVELOPE_SCHEMA_MAJOR};
pub use error::SwarmProtoError;
pub use hash::blake3_hash;
pub use sign::{peer_id, sign_canonical, verify_canonical, Signed, SigningKey, VerifyingKey};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_renders() {
        let err = SwarmProtoError::Validation("round out of range".into());
        assert!(err.to_string().contains("validation failed"));
    }
}
