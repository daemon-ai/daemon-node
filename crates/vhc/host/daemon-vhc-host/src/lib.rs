// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-host` — the training worker library + host runtime.
//!
//! The worker binary's engine: the module sandbox (wasmtime, ABI §2.2), the major-2 event-loop
//! driver ([`v2`]) with the §2.5 `tabi@1` compute bridge, the ABI §1.3 driver-selection front
//! door ([`select`]), the `compute@2` runner ([`compute`]), the permanent device probe
//! ([`probe`]), and the deterministic ops + kernels. It links the heavy trees — wasmtime (guest
//! sandbox) and Burn (engine) — because it *is* the isolated worker fault domain; the node
//! process never links them.
//!
//! **The v1 five-phase driver retired at the Phase-E sunset** (decisions D5): the `Instance`
//! lifecycle dispatch, the phase-legality table, and the autotune admission were removed in one
//! auditable step; a major-1 module now meets a typed `AbiUnsupportedMajor` admission refusal
//! (the flipped A0 fixture is the standing regression). The frozen 66-import vocabulary
//! (`daemon-vhc-abi`) and the [`TENSOR_ABI_MINOR`] constant (served by the §2.5 bridge's
//! `abi_minor@1` import) remain as the live bridge surface; the retired-major packing constants
//! (`TENSOR_ABI_MAJOR`/`TENSOR_ABI_VERSION`) were dropped with the Wave-0 `daemon-vhc` scaffold
//! bin that was their only reader.

// `deny` (not `forbid`) so the two cfg-gated platform-probe FFI modules in `probe` can carry a
// scoped `#[allow(unsafe_code)]` (DXGI/D3D12 on Windows; the Objective-C runtime + `sysctlbyname` on
// macOS). Every other line of the crate still errors on stray `unsafe`; the worker bin keeps its
// own `#![forbid(unsafe_code)]` and only calls the safe probe wrappers. See swarm-ledger-p2-c2 D1.
#![deny(unsafe_code)]

pub mod backend;
// The burn autodiff engine backs the G1 `burn-ndarray` (CPU), G2 `wgpu` (Vulkan), and P3 Lane-G
// `cuda` (NVIDIA) lanes; the generic `BurnBackend<B>` impl needs only burn-tensor (always on), so the
// module compiles when any backend feature is enabled, and each concrete alias is feature-gated inside.
#[cfg(any(feature = "burn-ndarray", feature = "wgpu", feature = "cuda"))]
pub mod burn_backend;
// The Phase-C `compute@2` host runner: `CBOR(burn_ir::OperationIr)` wire + burn-router dispatch
// (decisions D8, ABI §15). Unconditional — the tier-1 ndarray backend is always compiled (the root
// `burn` dep pins `ndarray`+`autodiff`); wgpu/cuda ride the same generic seam behind their features.
pub mod compute;
pub mod handle;
// The PERMANENT device probe (decisions D5: "the device probe stays forever") — split out of the
// retired autotune-admission module at the Phase-E sunset.
pub mod probe;
pub mod runtime;
pub mod select;
pub mod trap;
pub mod v2;
// (A2 inversion): `wasm_backend` moved to `daemon-vhc-session` — the host no longer links the
// session, so the TrainerBackend seam impl lives with the trait (refactor §5 A2 item 3).

pub use backend::{AdamwHp, CpuBackend, OpBackend, TensorId};
#[cfg(any(feature = "burn-ndarray", feature = "wgpu", feature = "cuda"))]
pub use burn_backend::BurnBackend;
#[cfg(feature = "burn-ndarray")]
pub use burn_backend::BurnNdarrayBackend;
#[cfg(feature = "cuda")]
pub use burn_backend::{cuda_adapter_available, BurnCudaBackend};
#[cfg(feature = "wgpu")]
pub use burn_backend::{wgpu_adapter_available, BurnWgpuBackend};
pub use compute::{unservable_op, ComputeError, ComputeRunner, HostReal};
pub use handle::{HandleClass, Lane};
pub use probe::DeviceLimits;
pub use runtime::BackendKind;
pub use runtime::{EngineConfig, Worker};
pub use select::{select_driver, Selection};
pub use trap::{Trap, TrapCode};

/// The frozen `tabi@1` minor served by the §2.5 compute bridge's `abi_minor@1` import (decisions
/// D5: the sunset removed the v1 DRIVER, not the vocabulary; `tabi@1` lives on as the bridge under
/// major 2). A major-1 module is still refused `AbiUnsupportedMajor` at the §1.3 front door.
pub const TENSOR_ABI_MINOR: u32 = 0;

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
