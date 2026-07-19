// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The v2 linker — the Phase-A closed capability subset, one module per imported world
//! (`vhc@2`, `net@2`, `data@2`, `sys@2`, `compute@2`). Registration is name-keyed (the wasmtime
//! `Linker` resolves imports by `(module, name)`), so the per-world grouping carries no ordering
//! semantics; every import body enters through the §6.6 temporal-legality gate
//! (`Host::enter`) and stashes its typed trap through `stash`.

mod compute;
mod data;
mod net;
mod sys;
mod vhc;

use wasmtime::Linker;

use crate::run::driver::host::Host;

/// Link every `*@2` world the Phase-A driver serves.
pub(crate) fn link_v2(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    vhc::link(linker)?;
    net::link(linker)?;
    data::link(linker)?;
    sys::link(linker)?;
    compute::link(linker)?;
    Ok(())
}
