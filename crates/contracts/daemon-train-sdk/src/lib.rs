// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train-sdk` — the guest experiment SDK.
//!
//! The guest-side SDK an experiment module links against: a Burn-like composition API over the
//! tensor ABI plus the optimization profiles (swarm-training-spec.md §5.3, §10.1). The first-party
//! preset experiments are its reference consumers.
//!
//! It targets `wasm32-unknown-unknown` (it runs inside the sandboxed module, ABI spec §5.1), so its
//! entire dependency surface is `serde` + `ciborium` — no host runtime ever leaks into a guest.
//!
//! Wave-0 scaffold: only the ABI-version handshake constant + the CBOR error type are present; the
//! composition API lands with lane **E**.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// The tensor-ABI major version this SDK is built against.
pub const DA_ABI_MAJOR: u32 = 1;
/// The tensor-ABI minor version this SDK is built against.
pub const DA_ABI_MINOR: u32 = 0;

/// The tensor-ABI version this SDK is built against, packed as `(major << 16) | minor`.
///
/// The guest advertises it via the `da_abi` export so the host can reject an incompatible module
/// before instantiation (swarm-tensor-abi-spec.md).
pub const DA_ABI_VERSION: u32 = (DA_ABI_MAJOR << 16) | DA_ABI_MINOR;

/// Errors surfaced by the SDK's CBOR (de)serialization helpers.
///
/// Hand-rolled to keep the guest dependency surface to `serde` + `ciborium` only.
#[derive(Debug)]
#[non_exhaustive]
pub enum SdkError {
    /// A CBOR encode/decode step failed.
    Codec(String),
}

impl fmt::Display for SdkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "train-sdk codec error: {detail}"),
        }
    }
}

impl Error for SdkError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_packs_major_minor() {
        assert_eq!(DA_ABI_VERSION >> 16, 1);
        assert_eq!(DA_ABI_VERSION & 0xffff, 0);
    }

    #[test]
    fn error_renders() {
        assert!(SdkError::Codec("bad frame".into())
            .to_string()
            .contains("codec error"));
    }
}
