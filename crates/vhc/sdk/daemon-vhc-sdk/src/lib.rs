// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 guest SDK tier: raw `vhc@2`/`net@2`/`sys@2` bindings ([`abi`]), the module layer
//! with SDK-side claim generation ([`module`]), migration scaffolding ([`migrate`], ABI §10.2),
//! and the [`main!`] macro emitting the v2 exports (ABI §2.1/§10.1). This is the base guest SDK
//! tier — the retired v1 SDK it once grew beside is deleted.

#[cfg(target_arch = "wasm32")]
pub mod abi;
#[cfg(target_arch = "wasm32")]
pub use abi::{
    buffer_len, buffer_release, cancel, compute_export, compute_fence, compute_import,
    compute_submit_op, create_from, data_fetch, device_profile, emit_metric, hash_accel,
    next_event, payload_get, payload_put, publish, read_back_bytes, read_back_uint, read_buffer,
    rng_seed, set_timer, snapshot_state, stage_state, stream_accept, stream_open, stream_read,
    stream_write, verify_sig_accel, Event,
};

pub mod corpus;
pub mod migrate;
pub mod module;

#[cfg(target_arch = "wasm32")]
pub use corpus::register_shard_chunks;
pub use corpus::{
    chunk_descriptor, decode_sequence_tokens, derive_assignment, plan_window, sequence_byte_range,
    Assignment, AssignmentParams, BatchLocation, CorpusError, CorpusManifest, CorpusWindow,
    Manifest, RangeFetch, ShardDesc, TokenWidth,
};
pub use migrate::{
    build_manifest, MigrateState, MigrationDescriptor, MigrationSection, OwnedSection, SectionDecl,
    SectionReader, SimSections, StateManifest,
};
pub use module::{derive_claim, manifest_bytes, GuestModule, ModuleDecl};

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

/// Emit the required major-2 exports (ABI §2.1) for a [`module::GuestModule`] type: `da_abi`,
/// `da_alloc`/`da_free`, `da_manifest`, `da_claim` (both SDK-derived from the declaration),
/// `da_init`, `da_run`, and `da_migrate` (ABI §10.1 — always exported; the manifest's
/// `migratable: true` echo is therefore truthful, and a module that does not override
/// [`module::GuestModule::migrate`] answers `Incompatible` honestly at runtime).
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
                (2 << 16) | <$module as $crate::module::GuestModule>::decl().abi_minor
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
                let decl = <$module as $crate::module::GuestModule>::decl();
                $crate::module::rt::emit_cbor(&$crate::module::manifest_bytes(&decl))
            }

            #[no_mangle]
            pub extern "C" fn da_claim(_c: u32, _cl: u32, _g: u32, _gl: u32) -> u64 {
                let decl = <$module as $crate::module::GuestModule>::decl();
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
                match <$module as $crate::module::GuestModule>::init(&config, &grants) {
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
                    <$module as $crate::module::GuestModule>::run(m)
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
                    <$module as $crate::module::GuestModule>::migrate(m, &descriptor, &mut reader)
                })
            }
        };
    };
}
