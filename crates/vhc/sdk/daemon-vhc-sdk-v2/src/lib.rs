// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 guest SDK tier: raw `vhc@2`/`net@2`/`sys@2` bindings ([`abi`]), the module layer
//! with SDK-side claim generation ([`module`]), migration scaffolding ([`migrate`], ABI §10.2),
//! and the [`main!`] macro emitting the v2 exports (ABI §2.1/§10.1) — the v2 analogue of the
//! v1 SDK's `experiment!`.
//!
//! ## Why this is a SIBLING of `daemon-vhc-sdk`, permanently — not a module inside it
//!
//! The v1 SDK's source feeds the frozen v1 guest bytes (`guests.blake3` / the A0 fixture pins).
//! Measured twice in this tree (sittings 8–9): ANY edit to `daemon-vhc-sdk` — even an unused,
//! cfg-gated module — perturbs cargo's crate metadata and reorders `tiny_llama.wasm`'s emitted
//! sections (pin `57ae15…` → different bytes), while growing THIS crate leaves the v1 pin
//! byte-identical. The plan to "fold into the SDK when the macro layer lands" is therefore
//! **rescinded**: the fold would trade the A0 byte-identity invariant (refactor invariant 2)
//! for cosmetics. The two crates merge only when the v1 SDK itself is next deliberately
//! re-pinned (a v1-sunsetting change, Phase E at the earliest).

#[cfg(target_arch = "wasm32")]
pub mod abi;
#[cfg(target_arch = "wasm32")]
pub use abi::{
    buffer_len, buffer_release, cancel, create_from, next_event, payload_get, payload_put, publish,
    read_back_bytes, read_back_uint, read_buffer, set_timer, Event,
};

pub mod migrate;
pub mod module;

pub use migrate::{
    build_manifest, MigrateState, MigrationDescriptor, MigrationSection, OwnedSection, SectionDecl,
    SectionReader, SimSections, StateManifest,
};
pub use module::{derive_claim, manifest_bytes, ModuleDecl, V2Module};

/// A [`crate::migrate::SectionReader`] over the live `read_back(kind = 3)` restore capability —
/// what `main!`'s `da_migrate` hands the module (§6.6: kind 3 is legal exactly there).
#[cfg(target_arch = "wasm32")]
pub struct AbiSections;

#[cfg(target_arch = "wasm32")]
impl migrate::SectionReader for AbiSections {
    fn read(&mut self, staging_id: u64) -> Vec<u8> {
        abi::read_back_bytes(staging_id, 3)
    }
}

/// Emit the required major-2 exports (ABI §2.1) for a [`module::V2Module`] type: `da_abi`,
/// `da_alloc`/`da_free`, `da_manifest`, `da_claim` (both SDK-derived from the declaration),
/// `da_init`, `da_run`, and `da_migrate` (ABI §10.1 — always exported; the manifest's
/// `migratable: true` echo is therefore truthful, and a module that does not override
/// [`module::V2Module::migrate`] answers `Incompatible` honestly at runtime).
///
/// Expands to nothing on non-wasm targets, exactly like the v1 `experiment!` — sim tests call
/// the trait methods directly.
#[macro_export]
macro_rules! main {
    ($module:ty) => {
        #[cfg(target_arch = "wasm32")]
        const _: () = {
            use ::core::cell::RefCell;

            ::std::thread_local! {
                static MODULE: RefCell<Option<$module>> = const { RefCell::new(None) };
            }

            #[no_mangle]
            pub extern "C" fn da_abi() -> u32 {
                // The declaration is cross-checked against the import shape at selection
                // (ABI §1.3 step 5): the declared minor must cover every imported symbol's
                // introducing minor.
                (2 << 16) | <$module as $crate::module::V2Module>::decl().abi_minor
            }

            #[no_mangle]
            pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
                $crate::module::rt::da_alloc(size, align)
            }

            #[no_mangle]
            pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
                $crate::module::rt::da_free(ptr, size, align)
            }

            #[no_mangle]
            pub extern "C" fn da_manifest(_cfg: u32, _cfg_len: u32) -> u64 {
                let decl = <$module as $crate::module::V2Module>::decl();
                $crate::module::rt::emit_cbor(&$crate::module::manifest_bytes(&decl))
            }

            #[no_mangle]
            pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
                let decl = <$module as $crate::module::V2Module>::decl();
                $crate::module::rt::emit_cbor(&$crate::module::derive_claim(&decl))
            }

            /// # Safety
            /// The host writes both spans before the call (ABI §2.3/§9.4 step 11).
            #[no_mangle]
            pub unsafe extern "C" fn da_init(
                cfg_ptr: u32,
                cfg_len: u32,
                grants_ptr: u32,
                grants_len: u32,
            ) -> u32 {
                let read = |ptr: u32, len: u32| -> Vec<u8> {
                    if len == 0 {
                        Vec::new()
                    } else {
                        ::std::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec()
                    }
                };
                let config = read(cfg_ptr, cfg_len);
                let grants = read(grants_ptr, grants_len);
                match <$module as $crate::module::V2Module>::init(&config, &grants) {
                    Ok(m) => {
                        MODULE.with(|s| *s.borrow_mut() = Some(m));
                        0
                    }
                    Err(code) => code,
                }
            }

            #[no_mangle]
            pub extern "C" fn da_run() -> u32 {
                MODULE.with(|s| {
                    let mut m = s.borrow_mut();
                    let m = m.as_mut().expect("da_init ran (host contract, §9.4)");
                    <$module as $crate::module::V2Module>::run(m)
                })
            }

            /// # Safety
            /// The host writes the descriptor span before the call (ABI §10.2).
            #[no_mangle]
            pub unsafe extern "C" fn da_migrate(descriptor_ptr: u32, descriptor_len: u32) -> u32 {
                let bytes = ::std::slice::from_raw_parts(
                    descriptor_ptr as *const u8,
                    descriptor_len as usize,
                )
                .to_vec();
                let Ok(descriptor) = $crate::migrate::MigrationDescriptor::from_wire(&bytes) else {
                    return 1; // Incompatible: undecodable descriptor (§10.2)
                };
                MODULE.with(|s| {
                    let mut m = s.borrow_mut();
                    let m = m
                        .as_mut()
                        .expect("da_init ran before da_migrate (§10.3 step 4)");
                    let mut reader = $crate::AbiSections;
                    <$module as $crate::module::V2Module>::migrate(m, &descriptor, &mut reader)
                })
            }
        };
    };
}
