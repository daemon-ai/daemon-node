// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-host` — the training worker library + host runtime.
//!
//! The worker binary's engine: the module sandbox (wasmtime, ABI §2.2), the major-2 event-loop
//! driver ([`v2`]), the ABI §1.3 driver-selection front door ([`select`]), the `compute@2`
//! runner ([`compute`]), and the permanent device probe ([`probe`]). It links the heavy trees —
//! wasmtime (guest sandbox) and Burn (engine) — because it *is* the isolated worker fault
//! domain; the node process never links them.
//!
//! No v1 machinery exists: a major-1 module meets a typed `AbiUnsupportedMajor` admission
//! refusal, and a module importing the retired `tabi@1` compute bridge meets a typed
//! `BridgeRetired` refusal — both at the §1.3 front door, both pinned over synthetic in-test
//! inputs. Compute crosses the boundary exclusively through the `compute@2` world.

// `deny` (not `forbid`) so the two cfg-gated platform-probe FFI modules in `probe` can carry a
// scoped `#[allow(unsafe_code)]` (DXGI/D3D12 on Windows; the Objective-C runtime + `sysctlbyname` on
// macOS). Every other line of the crate still errors on stray `unsafe`; the worker bin keeps its
// own `#![forbid(unsafe_code)]` and only calls the safe probe wrappers. See swarm-ledger-p2-c2 D1.
#![deny(unsafe_code)]

// The `compute@2` host runner: `CBOR(burn_ir::OperationIr)` wire + burn-router dispatch
// (decisions D8, ABI §15). Unconditional — the tier-1 ndarray backend is always compiled (the root
// `burn` dep pins `ndarray`+`autodiff`); wgpu/cuda ride the same generic seam behind their features.
pub mod compute;
// The PERMANENT device probe (decisions D5: "the device probe stays forever").
pub mod coordinator;
pub mod probe;
pub mod run;
pub mod runtime;
pub mod select;
pub mod trap;

pub use compute::{unservable_op, ComputeError, ComputeRunner, HostReal};
#[cfg(feature = "cuda")]
pub use probe::cuda_adapter_available;
#[cfg(feature = "wgpu")]
pub use probe::wgpu_adapter_available;
pub use probe::DeviceLimits;
pub use runtime::BackendKind;
pub use runtime::{EngineConfig, Worker};
pub use select::{select_driver, Selection};
pub use trap::{Trap, TrapCode};

/// Errors surfaced by the worker host runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrainError {
    /// A typed trap raised by a host call or mapped from a wasmtime trap (ABI §3.6).
    #[error("{0}")]
    Trap(#[from] Trap),
    /// The wasm engine failed to compile / link / instantiate a module.
    #[error("module sandbox error: {0}")]
    Sandbox(String),
    /// The training engine (build / step / optimize) failed.
    #[error("engine error: {0}")]
    Engine(String),
}

impl TrainError {
    /// The trap code, if this error is (or wraps) a typed trap.
    #[must_use]
    pub fn trap_code(&self) -> Option<TrapCode> {
        match self {
            Self::Trap(t) => Some(t.code),
            _ => None,
        }
    }
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
    fn digest_is_deterministic() {
        assert_eq!(content_digest(b"round-0"), content_digest(b"round-0"));
        assert_ne!(content_digest(b"a").0, content_digest(b"b").0);
    }
}
