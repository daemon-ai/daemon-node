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

/// The complete `tabi@1` import vocabulary this SDK binds (the extern block in `abi.rs`), in
/// registration order: the Merge-1 frozen 50-import subset followed by the Wave-2 additions.
///
/// This is the **frozen surface**: the host `Linker` (`daemon-train`) and the phase-legality table
/// must agree with it name-for-name (asserted by `daemon-train/tests/abi_surface.rs`). Growth is
/// additive only (ABI §9) — append here, never reorder or remove.
pub const TABI_IMPORTS: &[&str] = &[
    // --- Merge-1 frozen subset (50) ---
    "param@1",
    "persistent@1",
    "det_persistent@1",
    "drop@1",
    "param_round_base@1",
    "backward@1",
    "grad@1",
    "zero_grads@1",
    "assign@1",
    "zeros@1",
    "ones@1",
    "full@1",
    "add@1",
    "sub@1",
    "mul@1",
    "mul_s@1",
    "matmul@1",
    "relu@1",
    "cross_entropy@1",
    "scalar@1",
    "metric@1",
    "log@1",
    "abi_minor@1",
    "adamw_step@1",
    "batch_tokens@1",
    "batch_size@1",
    "batch_seq_len@1",
    "upd_new@1",
    "upd_push_bytes@1",
    "upd_push_tensor@1",
    "upd_sections@1",
    "upd_kind@1",
    "upd_bytes_len@1",
    "upd_read_bytes@1",
    "upd_tensor@1",
    "det_zeros@1",
    "det_sum@1",
    "det_scale@1",
    "det_l2norm@1",
    "det_sign@1",
    "det_add@1",
    "det_sub@1",
    "det_mul@1",
    "det_absmax_unpack@1",
    "det_chunk_scatter_add@1",
    "det_chunk_scatter@1",
    "det_assign@1",
    "det_param@1",
    "det_reset_param_to_base@1",
    "det_axpy_param@1",
    // --- Wave-2 additions (16) ---
    "embedding@1",
    "rmsnorm@1",
    "softmax@1",
    "silu@1",
    "rope@1",
    "flash_attn@1",
    "reshape@1",
    "transpose@1",
    "slice@1",
    "topk_chunk@1",
    "chunk_scatter@1",
    "absmax_pack@1",
    "absmax_unpack@1",
    "dct2@1",
    "idct2@1",
    "det_idct2@1",
];

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
    abi_minor, det_sum, det_zeros, embedding, log, zero_grads, Batch, Config, DetPersistent,
    DetTensor, Dtype, Experiment, Init, Manifest, Param, Persistent, SectionKind, StepCtx, Tensor,
    UpdateBuilder, UpdateRef, UpdatesView,
};

/// `det_chunk_scatter@1` — allocating dense-from-sparse (re-export of [`api::det_chunk_scatter`]).
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub use api::det_chunk_scatter;

/// First-party optimization profiles (`SparseLoco`/`DiLoCo`/`Demo`) as composable library code
/// (architecture §5.3, ABI §10.3).
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub mod profiles;

/// First-party preset experiments (the `TinyLlama` reference decoder, architecture §10.5).
#[cfg(any(target_arch = "wasm32", feature = "sim"))]
pub mod models;

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
    pub use crate::profiles;
    pub use crate::{det_sum, det_zeros, embedding, zero_grads};
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
        // Exports exist only on the wasm guest target; under `sim`/native the macro expands to
        // nothing (tests call the `Experiment` methods directly). Guest crates are always wasm and
        // never carry a `sim` feature, so gating on the target arch alone keeps `check-cfg` quiet.
        #[cfg(target_arch = "wasm32")]
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

    #[test]
    fn tabi_imports_are_unique_and_complete() {
        // 50 Merge-1 frozen imports + 16 Wave-2 additions = the frozen v1 vocabulary.
        assert_eq!(TABI_IMPORTS.len(), 66);
        let mut names: Vec<&str> = TABI_IMPORTS.to_vec();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "tabi import names must be unique");
        // Every name carries an explicit @version (additive growth is by version, ABI §9).
        assert!(TABI_IMPORTS.iter().all(|n| n.contains('@')));
    }
}
