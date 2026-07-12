// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-proto` — the swarm-training wire contract.
//!
//! Envelope schema + validation, the coordinator state machine (`tick`), committee / assignment
//! math, capability-set types, the swarm CDDL types, and [`SWARM_PROTO_VERSION`]. This crate is
//! the single authority for the swarm wire shapes shared by the host, the participant runtime, and
//! the (wasm32) coordinator DO — see `docs/specs/swarm-training-spec.md` §10.1.
//!
//! **wasm32-clean by construction:** the only dependencies are `serde` (schema) and `ciborium`
//! (CBOR codec). No `tokio`, no Burn, no wasmtime — nothing that would fail to build for the
//! `wasm32-unknown-unknown` coordinator target (§11.2).
//!
//! Wave-0 scaffold: the concrete envelope/state-machine types land with lane **P**.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// The swarm protocol version negotiated between coordinator and participants.
///
/// Distinct from the app↔node `WireVersion`: this governs the swarm control-plane envelope only
/// (swarm-training-spec.md §10.1). Bump on any incompatible envelope change.
pub const SWARM_PROTO_VERSION: u32 = 0;

/// Errors surfaced when validating or decoding a swarm protocol envelope.
///
/// Hand-rolled (rather than via `thiserror`) to keep this crate's dependency surface to `serde` +
/// `ciborium`, so it stays trivially `wasm32-unknown-unknown`-clean.
#[derive(Debug)]
#[non_exhaustive]
pub enum SwarmProtoError {
    /// An envelope failed schema validation (a field was out of range or inconsistent).
    Validation(String),
    /// A CBOR (de)serialization step failed.
    Codec(String),
}

impl fmt::Display for SwarmProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(detail) => write!(f, "swarm envelope validation failed: {detail}"),
            Self::Codec(detail) => write!(f, "swarm envelope codec error: {detail}"),
        }
    }
}

impl Error for SwarmProtoError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proto_version_is_stable() {
        assert_eq!(SWARM_PROTO_VERSION, 0);
    }

    #[test]
    fn error_renders() {
        let err = SwarmProtoError::Validation("round out of range".into());
        assert!(err.to_string().contains("validation failed"));
    }
}
