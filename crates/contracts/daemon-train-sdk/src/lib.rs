// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train-sdk` — the guest experiment SDK.
//!
//! The guest-side SDK an experiment module links against: safe wrapper types over the tensor ABI
//! (`tabi@1`, ABI §5), the [`Experiment`] trait + [`experiment!`] macro that wire the `da_*` exports
//! (ABI §4/§10.2), and — under the `sim` feature — an in-crate CPU reference backend so experiments
//! are unit-testable natively without a GPU or the wasm host (ABI §10.4). It targets
//! `wasm32-unknown-unknown` (it runs inside the sandboxed module, ABI §5.1); its entire dependency
//! surface is `serde` + `ciborium` (+ `det-core` under `sim`).
//!
//! ## Build shapes (why the `cfg` gating looks like this)
//!
//! - **wasm guest** (`target_arch = "wasm32"`, no `sim`): the real `tabi@1` extern block + `da_*`
//!   export trampolines. This is the shipped module.
//! - **sim** (`feature = "sim"`, native): the extern block is replaced by an in-crate CPU backend
//!   (`det-core`-backed det lane + a tiny reverse-mode tape) so `cargo test --features sim` runs a
//!   toy experiment end to end.
//! - **native, no `sim`** (the default workspace gate): the crate compiles down to its constants +
//!   error type only. The tensor surface needs a backend, so it is `cfg`-gated to the two shapes
//!   above — the default gate still type-checks the crate skeleton cheaply.
//!
//! `unsafe` is forbidden everywhere except the wasm extern/`da_alloc` glue (the ABI boundary):
//! the sim path is fully safe.

#![cfg_attr(not(target_arch = "wasm32"), forbid(unsafe_code))]

use std::error::Error;
use std::fmt;

/// The tensor-ABI major version this SDK is built against.
pub const DA_ABI_MAJOR: u32 = 1;
/// The tensor-ABI minor version this SDK is built against.
pub const DA_ABI_MINOR: u32 = 0;

/// The tensor-ABI version this SDK is built against, packed as `(major << 16) | minor`.
///
/// The guest advertises it via the `da_abi` export so the host can reject an incompatible module
/// before instantiation (swarm-tensor-abi-spec.md §4).
pub const DA_ABI_VERSION: u32 = (DA_ABI_MAJOR << 16) | DA_ABI_MINOR;

/// Errors surfaced by the SDK's CBOR (de)serialization helpers.
///
/// Hand-rolled to keep the guest dependency surface to `serde` + `ciborium` only.
#[derive(Debug)]
#[non_exhaustive]
pub enum SdkError {
    /// A CBOR encode/decode step failed.
    Codec(String),
}

impl fmt::Display for SdkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(detail) => write!(f, "train-sdk codec error: {detail}"),
        }
    }
}

impl Error for SdkError {}

// The tensor surface + sim backend only exist where there is a backend to call.
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
mod abi;
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
mod api;
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub use api::{
    abi_minor, det_sum, det_zeros, log, zero_grads, Batch, Config, DetPersistent, DetTensor, Dtype,
    Experiment, Init, Manifest, Param, Persistent, SectionKind, StepCtx, Tensor, UpdateBuilder,
    UpdateRef, UpdatesView,
};

/// `det_chunk_scatter@1` — allocating dense-from-sparse (re-export of [`api::det_chunk_scatter`]).
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub use api::det_chunk_scatter;

/// The in-crate CPU reference backend (feature `sim`): drives the tensor surface natively so
/// experiments and profiles are unit-testable with `cargo test --features sim` (ABI §10.4).
#[cfg(feature = "sim")]
pub mod sim;

/// The `da_*` export runtime glue (allocator + CBOR emit + config unmarshal), called by the code
/// [`experiment!`] generates. wasm-only: it is the ABI boundary.
#[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
#[doc(hidden)]
pub mod rt;

/// A convenient glob for experiment authors.
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub mod prelude {
    pub use crate::{det_sum, det_zeros, zero_grads};
    pub use crate::{
        experiment, Batch, Config, DetPersistent, DetTensor, Dtype, Experiment, Init, Manifest,
        Param, Persistent, StepCtx, Tensor, UpdateBuilder, UpdatesView,
    };
}

/// Generate the `da_*` export trampolines for an [`Experiment`] implementor.
///
/// Emits `da_abi` / `da_manifest` / `da_defaults` / `da_alloc` / `da_free` and the
/// `da_build` / `da_step` / `da_inner_update` / `da_make_update` / `da_ingest_updates` lifecycle
/// entry points (ABI §4), holding the experiment singleton in a guest-static — legitimate under T3
/// because it holds only handles + config, both re-derived by `da_build` after any re-instantiation
/// (ABI §10.2). The exports only exist on the wasm guest target; under `sim`/native the macro
/// expands to nothing (tests call the [`Experiment`] methods directly).
///
/// Note (frozen at Merge 1): `da_manifest` / `da_defaults` return their `(ptr, len)` CBOR result as
/// a single packed `u64` (`ptr << 32 | len`) rather than a wasm multi-value pair — the host reads it
/// back and calls `da_free`. The logical ABI signature is unchanged; this is the Rust-ABI wire form.
#[macro_export]
macro_rules! experiment {
    ($exp:ty) => {
        #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
        const _: () = {
            use ::core::cell::RefCell;

            thread_local! {
                static EXP: RefCell<Option<$exp>> = const { RefCell::new(None) };
            }

            #[no_mangle]
            pub extern "C" fn da_abi() -> u32 {
                $crate::DA_ABI_VERSION
            }

            #[no_mangle]
            pub extern "C" fn da_alloc(size: u32, align: u32) -> u32 {
                $crate::rt::da_alloc(size, align)
            }

            #[no_mangle]
            pub extern "C" fn da_free(ptr: u32, size: u32, align: u32) {
                $crate::rt::da_free(ptr, size, align)
            }

            #[no_mangle]
            pub extern "C" fn da_manifest(cfg_ptr: u32, cfg_len: u32) -> u64 {
                let cfg = $crate::rt::config_from_raw(cfg_ptr, cfg_len);
                let manifest = <$exp as $crate::Experiment>::manifest(&cfg);
                $crate::rt::emit_cbor(&manifest.to_cbor())
            }

            #[no_mangle]
            pub extern "C" fn da_defaults() -> u64 {
                $crate::rt::emit_cbor(&<$exp as $crate::Experiment>::defaults())
            }

            #[no_mangle]
            pub extern "C" fn da_build(cfg_ptr: u32, cfg_len: u32) {
                let cfg = $crate::rt::config_from_raw(cfg_ptr, cfg_len);
                let exp = <$exp as $crate::Experiment>::build(&cfg);
                EXP.with(|slot| *slot.borrow_mut() = Some(exp));
            }

            #[no_mangle]
            pub extern "C" fn da_step(
                batch: u64,
                inner_step: u32,
                mb_index: u32,
                mb_count: u32,
                step_seqs: u32,
            ) {
                let batch = $crate::Batch::from_handle(batch);
                let ctx = $crate::StepCtx {
                    inner_step,
                    mb_index,
                    mb_count,
                    step_seqs,
                };
                EXP.with(|slot| {
                    slot.borrow_mut()
                        .as_mut()
                        .expect("da_build must run before da_step")
                        .step(&batch, &ctx)
                });
            }

            #[no_mangle]
            pub extern "C" fn da_inner_update(inner_step: u32) {
                EXP.with(|slot| {
                    slot.borrow_mut()
                        .as_mut()
                        .expect("da_build must run before da_inner_update")
                        .inner_update(inner_step)
                });
            }

            #[no_mangle]
            pub extern "C" fn da_make_update(round: u64) -> u64 {
                EXP.with(|slot| {
                    slot.borrow_mut()
                        .as_mut()
                        .expect("da_build must run before da_make_update")
                        .make_update(round)
                        .into_handle()
                })
            }

            #[no_mangle]
            pub extern "C" fn da_ingest_updates(round: u64, count: u32) {
                let updates = $crate::UpdatesView::with_count(count);
                EXP.with(|slot| {
                    slot.borrow_mut()
                        .as_mut()
                        .expect("da_build must run before da_ingest_updates")
                        .ingest(round, &updates)
                });
            }
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_packs_major_minor() {
        assert_eq!(DA_ABI_VERSION >> 16, 1);
        assert_eq!(DA_ABI_VERSION & 0xffff, 0);
    }

    #[test]
    fn error_renders() {
        assert!(SdkError::Codec("bad frame".into())
            .to_string()
            .contains("codec error"));
    }
}
