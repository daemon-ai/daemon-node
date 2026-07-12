// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tiny-llama` — the reference guest experiment module.
//!
//! A `cdylib` compiled to `wasm32-unknown-unknown` and instantiated by the `daemon-train` host in
//! the wasm sandbox (tensor-ABI spec §5.1). Wave-0 scaffold: it exports only the ABI-version
//! handshake (`da_abi`); the `da_build` / `da_step` / `da_ingest_updates` surface lands with lane
//! **E**.

/// The tensor-ABI version this module targets, packed as `(major << 16) | minor`.
///
/// The host calls this before instantiation to reject an incompatible module. Sourced from the
/// guest SDK so there is a single source of truth for the ABI version.
#[no_mangle]
pub extern "C" fn da_abi() -> u32 {
    daemon_train_sdk::DA_ABI_VERSION
}
