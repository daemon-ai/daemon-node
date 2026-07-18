// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The reference CPU backend (architecture §6): the native det-lane world.
//!
//! The **det lane** is the shared [`daemon_vhc_det`] fixed-order fp32 kernels — the SAME
//! implementation the host worker links, so "sim ≡ host" on the det path is one codebase and
//! bit-identical by construction (ABI §10.4).

/// The bit-exact fixed-order fp32 consensus kernels (the det lane; normative, ABI §5.6/§10.4).
pub use daemon_vhc_det as det;
