// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The crate's hand-rolled error type.
//!
//! Deliberately dependency-free (no `thiserror`) so the crate keeps a `serde` + `ciborium` +
//! `blake3` + `xxhash-rust` + `ed25519-dalek` surface and stays trivially
//! `wasm32-unknown-unknown`-clean (Wave-0 lean-contract note; swarm-training-spec.md §10.1).

use std::error::Error;
use std::fmt;

/// Errors surfaced across the swarm proto contract.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SwarmProtoError {
    /// A schema value was out of range, missing, or inconsistent (e.g. unknown envelope major).
    Validation(String),
    /// A CBOR (de)serialization / canonicalization step failed.
    Codec(String),
    /// An ed25519 signature failed to verify, or a key/signature was malformed.
    Signature(String),
    /// A merkle commitment or membership proof did not check out.
    Merkle(String),
    /// A capability requirement was not satisfied (missing `name@version`).
    Capability(String),
    /// A `SwarmProtoVersion` did not exactly match the run's pinned version.
    Version(String),
}

impl fmt::Display for SwarmProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(d) => write!(f, "swarm envelope validation failed: {d}"),
            Self::Codec(d) => write!(f, "swarm envelope codec error: {d}"),
            Self::Signature(d) => write!(f, "swarm signature error: {d}"),
            Self::Merkle(d) => write!(f, "swarm set-commitment error: {d}"),
            Self::Capability(d) => write!(f, "swarm capability error: {d}"),
            Self::Version(d) => write!(f, "swarm proto version error: {d}"),
        }
    }
}

impl Error for SwarmProtoError {}
