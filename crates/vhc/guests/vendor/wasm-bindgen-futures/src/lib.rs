// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! JS-free `wasm-bindgen-futures` stub (see Cargo.toml): the one symbol the Burn tree consumes
//! (`cubecl-common::future::spawn` → [`spawn_local`]) panics **if called** — a daemon-vhc guest
//! has no in-guest executor (all actual waiting lives in the host's async runtime, architecture
//! §3.3), so a conforming module never reaches it.

use core::future::Future;

/// Browser-executor spawn, stubbed for wasmtime guests: panics if ever reached.
pub fn spawn_local<F>(_future: F)
where
    F: Future<Output = ()> + 'static,
{
    panic!(
        "wasm-bindgen-futures::spawn_local has no meaning in a daemon-vhc guest: there is no \
         in-guest executor (all waiting lives in the host's async runtime, architecture §3.3)"
    );
}
