// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `tabi@1` import seam.
//!
//! One safe Rust interface (`&str` / `&[u32]` / `&[u8]` / handles) the wrapper types in
//! [`crate::api`] call. Its body is selected by build shape:
//!
//! - wasm guest: marshal into the real `extern "C"` `tabi@1` import block (ABI §5).
//! - `sim`: forward to the in-crate CPU store ([`crate::sim`]).
//!
//! The subset wired here is Merge-1's frozen `tabi@1` vocabulary (see `swarm-ledger-e1.md`); it maps
//! name-for-name onto the host Linker in `daemon-train`. Growth is additive (ABI §9).

/// An opaque `tabi@1` handle (nonzero when valid; `0` is never a live handle, ABI §3.3).
pub(crate) type RawHandle = u64;

// -- the raw wasm import block (wasm guest only) -------------------------------------------------

#[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
mod raw {
    #[link(wasm_import_module = "tabi@1")]
    extern "C" {
        #[link_name = "param@1"]
        pub(super) fn param(
            np: u32,
            nl: u32,
            dp: u32,
            dr: u32,
            dt: u32,
            init: u32,
            p0: f64,
            p1: f64,
        ) -> u64;
        #[link_name = "persistent@1"]
        pub(super) fn persistent(np: u32, nl: u32, dp: u32, dr: u32, dt: u32, class: u32) -> u64;
        #[link_name = "det_persistent@1"]
        pub(super) fn det_persistent(np: u32, nl: u32, dp: u32, dr: u32, class: u32) -> u64;
        #[link_name = "drop@1"]
        pub(super) fn drop(h: u64);
        #[link_name = "param_round_base@1"]
        pub(super) fn param_round_base(p: u64) -> u64;
        #[link_name = "backward@1"]
        pub(super) fn backward(loss: u64);
        #[link_name = "grad@1"]
        pub(super) fn grad(p: u64) -> u64;
        #[link_name = "zero_grads@1"]
        pub(super) fn zero_grads();
        #[link_name = "assign@1"]
        pub(super) fn assign(dst: u64, src: u64);
        #[link_name = "zeros@1"]
        pub(super) fn zeros(dp: u32, dr: u32, dt: u32) -> u64;
        #[link_name = "ones@1"]
        pub(super) fn ones(dp: u32, dr: u32, dt: u32) -> u64;
        #[link_name = "full@1"]
        pub(super) fn full(dp: u32, dr: u32, dt: u32, value: f64) -> u64;
        #[link_name = "add@1"]
        pub(super) fn add(a: u64, b: u64) -> u64;
        #[link_name = "sub@1"]
        pub(super) fn sub(a: u64, b: u64) -> u64;
        #[link_name = "mul@1"]
        pub(super) fn mul(a: u64, b: u64) -> u64;
        #[link_name = "mul_s@1"]
        pub(super) fn mul_s(x: u64, v: f64) -> u64;
        #[link_name = "matmul@1"]
        pub(super) fn matmul(a: u64, b: u64) -> u64;
        #[link_name = "relu@1"]
        pub(super) fn relu(x: u64) -> u64;
        #[link_name = "cross_entropy@1"]
        pub(super) fn cross_entropy(logits: u64, targets: u64, ignore_index: i64) -> u64;
        #[link_name = "adamw_step@1"]
        pub(super) fn adamw_step(
            p: u64,
            g: u64,
            m: u64,
            v: u64,
            step: u32,
            lr: f64,
            beta1: f64,
            beta2: f64,
            eps: f64,
            wd: f64,
        );
        #[link_name = "batch_tokens@1"]
        pub(super) fn batch_tokens(b: u64) -> u64;
        #[link_name = "batch_size@1"]
        pub(super) fn batch_size(b: u64) -> u32;
        #[link_name = "batch_seq_len@1"]
        pub(super) fn batch_seq_len(b: u64) -> u32;
        #[link_name = "scalar@1"]
        pub(super) fn scalar(x: u64) -> f64;
        #[link_name = "metric@1"]
        pub(super) fn metric(np: u32, nl: u32, x: u64);
        #[link_name = "log@1"]
        pub(super) fn log(level: u32, mp: u32, ml: u32);
        #[link_name = "abi_minor@1"]
        pub(super) fn abi_minor() -> u32;
        #[link_name = "upd_new@1"]
        pub(super) fn upd_new() -> u64;
        #[link_name = "upd_push_bytes@1"]
        pub(super) fn upd_push_bytes(u: u64, dp: u32, dl: u32);
        #[link_name = "upd_push_tensor@1"]
        pub(super) fn upd_push_tensor(u: u64, x: u64);
        #[link_name = "upd_sections@1"]
        pub(super) fn upd_sections(i: u32) -> u32;
        #[link_name = "upd_kind@1"]
        pub(super) fn upd_kind(i: u32, s: u32) -> u32;
        #[link_name = "upd_bytes_len@1"]
        pub(super) fn upd_bytes_len(i: u32, s: u32) -> u32;
        #[link_name = "upd_read_bytes@1"]
        pub(super) fn upd_read_bytes(i: u32, s: u32, dp: u32, dl: u32) -> u32;
        #[link_name = "upd_tensor@1"]
        pub(super) fn upd_tensor(i: u32, s: u32) -> u64;
        #[link_name = "det_zeros@1"]
        pub(super) fn det_zeros(dp: u32, dr: u32) -> u64;
        #[link_name = "det_sum@1"]
        pub(super) fn det_sum(hp: u32, hc: u32) -> u64;
        #[link_name = "det_scale@1"]
        pub(super) fn det_scale(x: u64, alpha: f64) -> u64;
        #[link_name = "det_l2norm@1"]
        pub(super) fn det_l2norm(x: u64) -> f64;
        #[link_name = "det_sign@1"]
        pub(super) fn det_sign(x: u64) -> u64;
        #[link_name = "det_add@1"]
        pub(super) fn det_add(a: u64, b: u64) -> u64;
        #[link_name = "det_sub@1"]
        pub(super) fn det_sub(a: u64, b: u64) -> u64;
        #[link_name = "det_mul@1"]
        pub(super) fn det_mul(a: u64, b: u64) -> u64;
        #[link_name = "det_absmax_unpack@1"]
        pub(super) fn det_absmax_unpack(packed: u64, chunk: u32, bits: u32) -> u64;
        #[link_name = "det_chunk_scatter_add@1"]
        pub(super) fn det_chunk_scatter_add(acc: u64, vals: u64, idx: u64, chunk: u32);
        #[link_name = "det_chunk_scatter@1"]
        pub(super) fn det_chunk_scatter(vals: u64, idx: u64, chunk: u32, dp: u32, dr: u32) -> u64;
        #[link_name = "det_assign@1"]
        pub(super) fn det_assign(dst: u64, src: u64);
        #[link_name = "det_param@1"]
        pub(super) fn det_param(p: u64) -> u64;
        #[link_name = "det_reset_param_to_base@1"]
        pub(super) fn det_reset_param_to_base(p: u64);
        #[link_name = "det_axpy_param@1"]
        pub(super) fn det_axpy_param(p: u64, x: u64, alpha: f64);
        // -- Wave-2 additions (additive; §9) --------------------------------------------------
        #[link_name = "embedding@1"]
        pub(super) fn embedding(w: u64, ids: u64) -> u64;
        #[link_name = "rmsnorm@1"]
        pub(super) fn rmsnorm(x: u64, w: u64, eps: f64) -> u64;
        #[link_name = "softmax@1"]
        pub(super) fn softmax(x: u64, dim: u32) -> u64;
        #[link_name = "silu@1"]
        pub(super) fn silu(x: u64) -> u64;
        #[link_name = "rope@1"]
        pub(super) fn rope(x: u64, pos_start: u32, theta: f64, interleaved: u32) -> u64;
        #[link_name = "flash_attn@1"]
        pub(super) fn flash_attn(q: u64, k: u64, v: u64, causal: u32, scale: f64) -> u64;
        #[link_name = "reshape@1"]
        pub(super) fn reshape(x: u64, dp: u32, dr: u32) -> u64;
        #[link_name = "transpose@1"]
        pub(super) fn transpose(x: u64, d0: u32, d1: u32) -> u64;
        #[link_name = "slice@1"]
        pub(super) fn slice(x: u64, dim: u32, start: u32, end: u32) -> u64;
        // `topk_chunk@1` logically returns (values, indices); Rust's wasm C-ABI cannot emit a clean
        // multi-value (i64,i64), so — mirroring the da_manifest packing (E1) — the indices handle is
        // written to a guest `*mut u64` out-param and the values handle is returned (E2 deviation).
        #[link_name = "topk_chunk@1"]
        pub(super) fn topk_chunk(x: u64, chunk: u32, k: u32, out_idx: u32) -> u64;
        #[link_name = "chunk_scatter@1"]
        pub(super) fn chunk_scatter(vals: u64, idx: u64, chunk: u32, dp: u32, dr: u32) -> u64;
        #[link_name = "absmax_pack@1"]
        pub(super) fn absmax_pack(x: u64, chunk: u32, bits: u32) -> u64;
        #[link_name = "absmax_unpack@1"]
        pub(super) fn absmax_unpack(packed: u64, chunk: u32, bits: u32, dtype: u32) -> u64;
        #[link_name = "dct2@1"]
        pub(super) fn dct2(x: u64, tile: u32) -> u64;
        #[link_name = "idct2@1"]
        pub(super) fn idct2(x: u64, tile: u32) -> u64;
        #[link_name = "det_idct2@1"]
        pub(super) fn det_idct2(x: u64, tile: u32) -> u64;
    }
}

// -- the safe dispatch surface ------------------------------------------------------------------
//
// Each function is defined once; exactly one `cfg` body compiles (wasm-non-sim → extern marshal;
// sim → the CPU store). Native-non-sim never reaches here (the module is `cfg`-gated out).

// On the wasm guest target `unsafe` is not forbidden (the crate-level forbid is gated to non-wasm),
// so no per-site allow is needed. The `tabi@1` host guarantees these imports; misuse traps
// host-side with a typed code (ABI §3.6, T4).
#[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
macro_rules! wasm_call {
    ($($tt:tt)*) => {
        unsafe { raw::$($tt)* }
    };
}

pub(crate) fn param(
    name: &str,
    dims: &[u32],
    dtype: u32,
    init: u32,
    p0: f64,
    p1: f64,
) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(param(
        name.as_ptr() as u32,
        name.len() as u32,
        dims.as_ptr() as u32,
        dims.len() as u32,
        dtype,
        init,
        p0,
        p1
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.param(name, dims, dtype, init, p0, p1));
}

pub(crate) fn persistent(name: &str, dims: &[u32], dtype: u32, class: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(persistent(
        name.as_ptr() as u32,
        name.len() as u32,
        dims.as_ptr() as u32,
        dims.len() as u32,
        dtype,
        class
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.persistent(name, dims, dtype, class));
}

pub(crate) fn det_persistent(name: &str, dims: &[u32], class: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_persistent(
        name.as_ptr() as u32,
        name.len() as u32,
        dims.as_ptr() as u32,
        dims.len() as u32,
        class
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_persistent(name, dims, class));
}

pub(crate) fn drop_handle(h: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(drop(h));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.drop_handle(h));
}

pub(crate) fn param_round_base(p: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(param_round_base(p));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.param_round_base(p));
}

pub(crate) fn backward(loss: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(backward(loss));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.backward(loss));
}

pub(crate) fn grad(p: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(grad(p));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.grad(p));
}

pub(crate) fn zero_grads() {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(zero_grads());
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.zero_grads());
}

pub(crate) fn assign(dst: RawHandle, src: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(assign(dst, src));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.assign(dst, src));
}

pub(crate) fn zeros(dims: &[u32], dtype: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(zeros(dims.as_ptr() as u32, dims.len() as u32, dtype));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.zeros(dims, dtype));
}

pub(crate) fn ones(dims: &[u32], dtype: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(ones(dims.as_ptr() as u32, dims.len() as u32, dtype));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.full(dims, dtype, 1.0));
}

pub(crate) fn full(dims: &[u32], dtype: u32, value: f64) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(full(dims.as_ptr() as u32, dims.len() as u32, dtype, value));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.full(dims, dtype, value));
}

pub(crate) fn add(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(add(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.add(a, b));
}

pub(crate) fn sub(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(sub(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.sub(a, b));
}

pub(crate) fn mul(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(mul(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.mul(a, b));
}

pub(crate) fn mul_s(x: RawHandle, v: f64) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(mul_s(x, v));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.mul_s(x, v));
}

pub(crate) fn matmul(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(matmul(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.matmul(a, b));
}

pub(crate) fn relu(x: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(relu(x));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.relu(x));
}

pub(crate) fn cross_entropy(logits: RawHandle, targets: RawHandle, ignore_index: i64) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(cross_entropy(logits, targets, ignore_index));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.cross_entropy(logits, targets, ignore_index));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adamw_step(
    p: RawHandle,
    g: RawHandle,
    m: RawHandle,
    v: RawHandle,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(adamw_step(p, g, m, v, step, lr, beta1, beta2, eps, wd));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.adamw_step(p, g, m, v, step, lr, beta1, beta2, eps, wd));
}

pub(crate) fn batch_tokens(b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(batch_tokens(b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.batch_tokens(b));
}

pub(crate) fn batch_size(b: RawHandle) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(batch_size(b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.batch_size(b));
}

pub(crate) fn batch_seq_len(b: RawHandle) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(batch_seq_len(b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.batch_seq_len(b));
}

pub(crate) fn scalar(x: RawHandle) -> f64 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(scalar(x));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.scalar(x));
}

pub(crate) fn metric(name: &str, x: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(metric(name.as_ptr() as u32, name.len() as u32, x));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.metric(name, x));
}

pub(crate) fn log(level: u32, msg: &str) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(log(level, msg.as_ptr() as u32, msg.len() as u32));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.log(level, msg));
}

pub(crate) fn abi_minor() -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(abi_minor());
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.abi_minor());
}

pub(crate) fn upd_new() -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_new());
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.upd_new());
}

pub(crate) fn upd_push_bytes(u: RawHandle, data: &[u8]) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(upd_push_bytes(u, data.as_ptr() as u32, data.len() as u32));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.upd_push_bytes(u, data));
}

pub(crate) fn upd_push_tensor(u: RawHandle, x: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(upd_push_tensor(u, x));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.upd_push_tensor(u, x));
}

pub(crate) fn upd_sections(i: u32) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_sections(i));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.upd_sections(i));
}

pub(crate) fn upd_kind(i: u32, s: u32) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_kind(i, s));
    #[cfg(feature = "sim")]
    return crate::sim::with(|st| st.upd_kind(i, s));
}

pub(crate) fn upd_bytes_len(i: u32, s: u32) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_bytes_len(i, s));
    #[cfg(feature = "sim")]
    return crate::sim::with(|st| st.upd_bytes_len(i, s));
}

pub(crate) fn upd_read_bytes(i: u32, s: u32, dst: &mut [u8]) -> u32 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_read_bytes(
        i,
        s,
        dst.as_mut_ptr() as u32,
        dst.len() as u32
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|st| st.upd_read_bytes(i, s, dst));
}

pub(crate) fn upd_tensor(i: u32, s: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(upd_tensor(i, s));
    #[cfg(feature = "sim")]
    return crate::sim::with(|st| st.upd_tensor(i, s));
}

pub(crate) fn det_zeros(dims: &[u32]) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_zeros(dims.as_ptr() as u32, dims.len() as u32));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_zeros(dims));
}

pub(crate) fn det_sum(handles: &[RawHandle]) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_sum(handles.as_ptr() as u32, handles.len() as u32));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_sum(handles));
}

pub(crate) fn det_scale(x: RawHandle, alpha: f64) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_scale(x, alpha));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_scale(x, alpha));
}

pub(crate) fn det_l2norm(x: RawHandle) -> f64 {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_l2norm(x));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_l2norm(x));
}

pub(crate) fn det_sign(x: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_sign(x));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_sign(x));
}

pub(crate) fn det_add(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_add(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_add(a, b));
}

pub(crate) fn det_sub(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_sub(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_sub(a, b));
}

pub(crate) fn det_mul(a: RawHandle, b: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_mul(a, b));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_mul(a, b));
}

pub(crate) fn det_absmax_unpack(packed: RawHandle, chunk: u32, bits: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_absmax_unpack(packed, chunk, bits));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_absmax_unpack(packed, chunk, bits));
}

pub(crate) fn det_chunk_scatter_add(acc: RawHandle, vals: RawHandle, idx: RawHandle, chunk: u32) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(det_chunk_scatter_add(acc, vals, idx, chunk));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.det_chunk_scatter_add(acc, vals, idx, chunk));
}

pub(crate) fn det_chunk_scatter(
    vals: RawHandle,
    idx: RawHandle,
    chunk: u32,
    dims: &[u32],
) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_chunk_scatter(
        vals,
        idx,
        chunk,
        dims.as_ptr() as u32,
        dims.len() as u32
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_chunk_scatter(vals, idx, chunk, dims));
}

pub(crate) fn det_assign(dst: RawHandle, src: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(det_assign(dst, src));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.det_assign(dst, src));
}

pub(crate) fn det_param(p: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_param(p));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_param(p));
}

pub(crate) fn det_reset_param_to_base(p: RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(det_reset_param_to_base(p));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.det_reset_param_to_base(p));
}

pub(crate) fn det_axpy_param(p: RawHandle, x: RawHandle, alpha: f64) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    wasm_call!(det_axpy_param(p, x, alpha));
    #[cfg(feature = "sim")]
    crate::sim::with(|s| s.det_axpy_param(p, x, alpha));
}

// -- Wave-2 additions -----------------------------------------------------------------------------

pub(crate) fn embedding(w: RawHandle, ids: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(embedding(w, ids));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.embedding(w, ids));
}

pub(crate) fn rmsnorm(x: RawHandle, w: RawHandle, eps: f64) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(rmsnorm(x, w, eps));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.rmsnorm(x, w, eps));
}

pub(crate) fn softmax(x: RawHandle, dim: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(softmax(x, dim));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.softmax(x, dim));
}

pub(crate) fn silu(x: RawHandle) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(silu(x));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.silu(x));
}

pub(crate) fn rope(x: RawHandle, pos_start: u32, theta: f64, interleaved: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(rope(x, pos_start, theta, interleaved));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.rope(x, pos_start, theta, interleaved));
}

pub(crate) fn flash_attn(
    q: RawHandle,
    k: RawHandle,
    v: RawHandle,
    causal: u32,
    scale: f64,
) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(flash_attn(q, k, v, causal, scale));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.flash_attn(q, k, v, causal, scale));
}

pub(crate) fn reshape(x: RawHandle, dims: &[u32]) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(reshape(x, dims.as_ptr() as u32, dims.len() as u32));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.reshape(x, dims));
}

pub(crate) fn transpose(x: RawHandle, d0: u32, d1: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(transpose(x, d0, d1));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.transpose(x, d0, d1));
}

pub(crate) fn slice(x: RawHandle, dim: u32, start: u32, end: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(slice(x, dim, start, end));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.slice(x, dim, start, end));
}

/// `topk_chunk@1` → `(values, indices)`. On the guest the indices handle is returned via an
/// out-param (see the extern block note); the sim returns the pair directly.
pub(crate) fn topk_chunk(x: RawHandle, chunk: u32, k: u32) -> (RawHandle, RawHandle) {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    {
        let mut idx: u64 = 0;
        let values = wasm_call!(topk_chunk(x, chunk, k, &mut idx as *mut u64 as u32));
        return (values, idx);
    }
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.topk_chunk(x, chunk, k));
}

pub(crate) fn chunk_scatter(
    vals: RawHandle,
    idx: RawHandle,
    chunk: u32,
    dims: &[u32],
) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(chunk_scatter(
        vals,
        idx,
        chunk,
        dims.as_ptr() as u32,
        dims.len() as u32
    ));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.chunk_scatter(vals, idx, chunk, dims));
}

pub(crate) fn absmax_pack(x: RawHandle, chunk: u32, bits: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(absmax_pack(x, chunk, bits));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.absmax_pack(x, chunk, bits));
}

pub(crate) fn absmax_unpack(packed: RawHandle, chunk: u32, bits: u32, dtype: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(absmax_unpack(packed, chunk, bits, dtype));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.absmax_unpack(packed, chunk, bits, dtype));
}

pub(crate) fn dct2(x: RawHandle, tile: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(dct2(x, tile));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.dct2(x, tile));
}

pub(crate) fn idct2(x: RawHandle, tile: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(idct2(x, tile));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.idct2(x, tile));
}

pub(crate) fn det_idct2(x: RawHandle, tile: u32) -> RawHandle {
    #[cfg(all(target_arch = "wasm32", not(feature = "sim")))]
    return wasm_call!(det_idct2(x, tile));
    #[cfg(feature = "sim")]
    return crate::sim::with(|s| s.det_idct2(x, tile));
}
