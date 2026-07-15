// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The reference CPU backend (architecture §6): the native compute world.
//!
//! The **det lane** is the shared [`daemon_vhc_det`] fixed-order fp32 kernels — the SAME
//! implementation the host worker links, so "sim ≡ host" on the det path is one codebase and
//! bit-identical by construction (ABI §10.4). The **native lane** is the v1 SDK's `sim` reference
//! tape (a semantics reference, not a performance one): re-exported so native policy code — round
//! experiments, profiles — runs against exactly the surface the wasm guest sees.

/// The bit-exact fixed-order fp32 consensus kernels (the det lane; normative, ABI §5.6/§10.4).
pub use daemon_vhc_det as det;

/// The v1 SDK's native reference CPU backend (`sim` feature): the fp32 tape + shared det kernels
/// that make experiments and profiles unit-testable natively (no GPU, no wasm host).
pub use daemon_vhc_sdk::sim as cpu;

/// The first-party model + profile presets native policy code composes over the reference backend.
pub use daemon_vhc_sdk::{models, profiles};
