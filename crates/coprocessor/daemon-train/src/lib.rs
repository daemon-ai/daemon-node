// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train` — the training worker library + host runtime.
//!
//! The worker binary's engine: the tensor ABI + module sandbox (meta / trace / execute, tensor-ABI
//! spec §5.1), the lifecycle driver, param / persistent storage, the deterministic ops + kernels,
//! and the first-party preset experiments (swarm-training-spec.md §10.1). It links the heavy trees
//! — wasmtime (guest sandbox) and Burn (engine) — because it *is* the isolated worker fault domain;
//! the node process never links them.
//!
//! Wave-0 scaffold: the sandbox / engine wiring lands with lane **E**. Present now: the ABI-version
//! constant and a content-digest helper (blake3 + xxh3) used for artifact / tensor identity.

#![forbid(unsafe_code)]

/// The tensor-ABI major version this worker implements.
pub const TENSOR_ABI_MAJOR: u32 = 1;
/// The tensor-ABI minor version this worker implements.
pub const TENSOR_ABI_MINOR: u32 = 0;

/// The tensor-ABI version this worker implements, packed as `(major << 16) | minor`.
///
/// Must match the guest's `da_abi` export for a module to be instantiated (tensor-ABI spec).
pub const TENSOR_ABI_VERSION: u32 = (TENSOR_ABI_MAJOR << 16) | TENSOR_ABI_MINOR;

/// Errors surfaced by the worker host runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrainError {
    /// The wasm module sandbox (instantiate / meta / trace / execute) failed.
    #[error("module sandbox error: {0}")]
    Sandbox(String),
    /// The training engine (build / step / optimize) failed.
    #[error("engine error: {0}")]
    Engine(String),
}

/// A stable content digest over `bytes`: the 256-bit blake3 hash plus a fast xxh3-64 checksum.
///
/// blake3 is the canonical artifact / tensor identity; the xxh3 checksum is the cheap in-memory
/// change probe (swarm-training-spec.md §5.1 host-RAM planning).
#[must_use]
pub fn content_digest(bytes: &[u8]) -> ([u8; 32], u64) {
    let blake = *blake3::hash(bytes).as_bytes();
    let xxh = xxhash_rust::xxh3::xxh3_64(bytes);
    (blake, xxh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_packs_major_minor() {
        assert_eq!(TENSOR_ABI_VERSION >> 16, 1);
    }

    #[test]
    fn digest_is_deterministic() {
        assert_eq!(content_digest(b"round-0"), content_digest(b"round-0"));
        assert_ne!(content_digest(b"a").0, content_digest(b"b").0);
    }
}
