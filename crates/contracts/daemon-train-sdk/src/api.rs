// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Safe wrapper types over `tabi@1` (ABI §10.1) + the [`Experiment`] trait (ABI §10.2).
//!
//! The wrappers track shape/dtype guest-side (ABI §5.3) and map 1:1 onto imports — nothing here
//! computes tensor math itself. [`Tensor`] / [`DetTensor`] are separate types so a native/det lane
//! mix is a **compile** error, not a runtime trap (ABI §3.4). Step-scoped handles free at scope
//! exit via `Drop` (RAII over `drop@1`, ABI §3.3); stable handles (params/persistents) skip it.

use crate::abi::{self, RawHandle};

/// Tensor element type (ABI §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Dtype {
    /// 32-bit float.
    F32 = 0,
    /// bfloat16.
    Bf16 = 1,
    /// IEEE half.
    F16 = 2,
    /// 64-bit signed int.
    I64 = 3,
    /// 32-bit signed int.
    I32 = 4,
    /// 32-bit unsigned int.
    U32 = 5,
    /// 8-bit unsigned int (also the carrier for packed/quantized data, ABI §3.2).
    U8 = 6,
    /// Boolean.
    Bool = 7,
}

impl Dtype {
    fn code(self) -> u32 {
        self as u32
    }
}

/// Param / det-persistent init strategy (ABI `param@1`). The seed is host-derived from
/// `(run_id, name)` — author code carries no seeds (ABI §5.1/§7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Init {
    /// All zeros.
    Zeros = 0,
    /// All ones.
    Ones = 1,
    /// Uniform in `(p0, p1)`.
    Uniform = 2,
    /// Normal `mean = p0, std = p1`.
    Normal = 3,
    /// Truncated normal.
    TruncNormal = 4,
}

// -- native lane tensor -------------------------------------------------------------------------

/// A native-lane tensor handle (GPU/CPU, vendor-variant numerics, ABI §3.4). Shape/dtype tracked
/// guest-side.
#[derive(Debug)]
pub struct Tensor {
    h: RawHandle,
    shape: Vec<u32>,
    dtype: Dtype,
    stable: bool,
}

impl Tensor {
    fn step(h: RawHandle, shape: Vec<u32>, dtype: Dtype) -> Self {
        Self {
            h,
            shape,
            dtype,
            stable: false,
        }
    }

    /// The raw handle (for the `experiment!` glue + container ops).
    #[must_use]
    pub fn handle(&self) -> RawHandle {
        self.h
    }

    /// The guest-tracked shape.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// The element dtype.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// `zeros@1`.
    #[must_use]
    pub fn zeros(dims: &[u32], dtype: Dtype) -> Self {
        Self::step(abi::zeros(dims, dtype.code()), dims.to_vec(), dtype)
    }

    /// `ones@1`.
    #[must_use]
    pub fn ones(dims: &[u32], dtype: Dtype) -> Self {
        Self::step(abi::ones(dims, dtype.code()), dims.to_vec(), dtype)
    }

    /// `full@1`.
    #[must_use]
    pub fn full(dims: &[u32], dtype: Dtype, value: f64) -> Self {
        Self::step(abi::full(dims, dtype.code(), value), dims.to_vec(), dtype)
    }

    /// `matmul@1` — trailing-2-dim contraction (ABI §5.6).
    #[must_use]
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let mut shape = self.shape.clone();
        let k = shape.pop().expect("matmul lhs needs rank >= 1");
        debug_assert_eq!(
            k,
            rhs.shape[rhs.shape.len() - 2],
            "matmul inner dims must agree"
        );
        shape.push(rhs.shape[rhs.shape.len() - 1]);
        Tensor::step(abi::matmul(self.h, rhs.h), shape, self.dtype)
    }

    /// `add@1` (NumPy-style trailing broadcast, ABI §5.4).
    #[must_use]
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        Tensor::step(abi::add(self.h, rhs.h), self.shape.clone(), self.dtype)
    }

    /// `sub@1`.
    #[must_use]
    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        Tensor::step(abi::sub(self.h, rhs.h), self.shape.clone(), self.dtype)
    }

    /// `mul@1`.
    #[must_use]
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        Tensor::step(abi::mul(self.h, rhs.h), self.shape.clone(), self.dtype)
    }

    /// `mul_s@1` (scalar right-hand).
    #[must_use]
    pub fn mul_s(&self, v: f64) -> Tensor {
        Tensor::step(abi::mul_s(self.h, v), self.shape.clone(), self.dtype)
    }

    /// `relu@1`.
    #[must_use]
    pub fn relu(&self) -> Tensor {
        Tensor::step(abi::relu(self.h), self.shape.clone(), self.dtype)
    }

    /// `cross_entropy@1` — mean over non-ignored rows; rank-0 (ABI §5.6).
    #[must_use]
    pub fn cross_entropy(&self, targets: &Tensor, ignore_index: i64) -> Tensor {
        Tensor::step(
            abi::cross_entropy(self.h, targets.h, ignore_index),
            Vec::new(),
            self.dtype,
        )
    }

    /// `backward@1` — reverse pass; `self` must be scalar-shaped (ABI §5.1).
    pub fn backward(&self) {
        abi::backward(self.h);
    }

    /// `scalar@1` — numel-1 readout (GPU sync in `execute`).
    #[must_use]
    pub fn scalar(&self) -> f64 {
        abi::scalar(self.h)
    }

    /// `metric@1` — non-blocking telemetry readback.
    pub fn metric(&self, name: &str) {
        abi::metric(name, self.h);
    }

    // -- Wave-2 elementwise / NN / shape (ABI §5.3–5.6) -----------------------------------------

    /// `silu@1` — `x · sigmoid(x)` (SwiGLU gate activation).
    #[must_use]
    pub fn silu(&self) -> Tensor {
        Tensor::step(abi::silu(self.h), self.shape.clone(), self.dtype)
    }

    /// `rmsnorm@1` — RMS layer norm scaled by `w` (`[d]`), `eps` for stability (ABI §5.6).
    #[must_use]
    pub fn rmsnorm(&self, w: &Tensor, eps: f64) -> Tensor {
        Tensor::step(
            abi::rmsnorm(self.h, w.h, eps),
            self.shape.clone(),
            self.dtype,
        )
    }

    /// `softmax@1` over `dim`.
    #[must_use]
    pub fn softmax(&self, dim: u32) -> Tensor {
        Tensor::step(abi::softmax(self.h, dim), self.shape.clone(), self.dtype)
    }

    /// `rope@1` — rotary position embedding applied per `[.., s, d]` (ABI §5.6).
    #[must_use]
    pub fn rope(&self, pos_start: u32, theta: f64, interleaved: bool) -> Tensor {
        Tensor::step(
            abi::rope(self.h, pos_start, theta, u32::from(interleaved)),
            self.shape.clone(),
            self.dtype,
        )
    }

    /// `flash_attn@1` — fused scaled-dot-product attention over `[b, h, s, d]` (ABI §5.6). `self`
    /// is the query; `k`/`v` the key/value. Returns `[b, h, s, d]` shaped like the query.
    #[must_use]
    pub fn flash_attn(&self, k: &Tensor, v: &Tensor, causal: bool, scale: f64) -> Tensor {
        Tensor::step(
            abi::flash_attn(self.h, k.h, v.h, u32::from(causal), scale),
            self.shape.clone(),
            self.dtype,
        )
    }

    /// `reshape@1` — same data, new `dims` (numel must match).
    #[must_use]
    pub fn reshape(&self, dims: &[u32]) -> Tensor {
        Tensor::step(abi::reshape(self.h, dims), dims.to_vec(), self.dtype)
    }

    /// `transpose@1` — swap axes `d0` and `d1`.
    #[must_use]
    pub fn transpose(&self, d0: u32, d1: u32) -> Tensor {
        let mut shape = self.shape.clone();
        shape.swap(d0 as usize, d1 as usize);
        Tensor::step(abi::transpose(self.h, d0, d1), shape, self.dtype)
    }

    /// `slice@1` — `x[.., start..end, ..]` along `dim`.
    #[must_use]
    pub fn slice(&self, dim: u32, start: u32, end: u32) -> Tensor {
        let mut shape = self.shape.clone();
        shape[dim as usize] = end - start;
        Tensor::step(abi::slice(self.h, dim, start, end), shape, self.dtype)
    }

    // -- Wave-2 compression natives (ABI §5.8) --------------------------------------------------

    /// `topk_chunk@1` — per-chunk top-k by magnitude → `(values, indices)`, each `[n_chunks, k]`.
    #[must_use]
    pub fn topk_chunk(&self, chunk: u32, k: u32) -> (Tensor, Tensor) {
        let numel: u32 = self.shape.iter().product();
        let n_chunks = numel.checked_div(chunk).unwrap_or(0);
        let (vh, ih) = abi::topk_chunk(self.h, chunk, k);
        (
            Tensor::step(vh, vec![n_chunks, k], Dtype::F32),
            Tensor::step(ih, vec![n_chunks, k], Dtype::U32),
        )
    }

    /// `chunk_scatter@1` — dense-from-sparse inverse of [`Tensor::topk_chunk`] (ABI §5.8).
    #[must_use]
    pub fn chunk_scatter(&self, idx: &Tensor, chunk: u32, dims: &[u32]) -> Tensor {
        Tensor::step(
            abi::chunk_scatter(self.h, idx.h, chunk, dims),
            dims.to_vec(),
            self.dtype,
        )
    }

    /// `absmax_pack@1` — per-chunk absmax codebook quantization to a packed `U8` tensor (§6.6).
    #[must_use]
    pub fn absmax_pack(&self, chunk: u32, bits: u32) -> Tensor {
        let numel: u32 = self.shape.iter().product();
        let n_chunks = numel.checked_div(chunk).unwrap_or(0);
        let code_bytes = (chunk * bits).div_ceil(8);
        let total = n_chunks * (2 + code_bytes);
        Tensor::step(
            abi::absmax_pack(self.h, chunk, bits),
            vec![total],
            Dtype::U8,
        )
    }

    /// `absmax_unpack@1` — decode a packed `U8` tensor back to `dtype` (native lane).
    #[must_use]
    pub fn absmax_unpack(&self, chunk: u32, bits: u32, dtype: Dtype) -> Tensor {
        Tensor::step(
            abi::absmax_unpack(self.h, chunk, bits, dtype.code()),
            Vec::new(),
            dtype,
        )
    }

    /// `dct2@1` — orthonormal 2-D DCT over `tile × tile` blocks (ABI §5.8).
    #[must_use]
    pub fn dct2(&self, tile: u32) -> Tensor {
        Tensor::step(abi::dct2(self.h, tile), self.shape.clone(), self.dtype)
    }

    /// `idct2@1` — the inverse 2-D DCT.
    #[must_use]
    pub fn idct2(&self, tile: u32) -> Tensor {
        Tensor::step(abi::idct2(self.h, tile), self.shape.clone(), self.dtype)
    }
}

/// `embedding@1` — gather rows of the weight `w` (`[vocab, d]`) by token `ids` (ABI §5.6). Output
/// shape is `ids.shape ++ [d]`.
#[must_use]
pub fn embedding(w: &Param, ids: &Tensor) -> Tensor {
    let d = *w.0.shape.last().expect("embedding weight is [vocab, d]");
    let mut shape = ids.shape.clone();
    shape.push(d);
    Tensor::step(abi::embedding(w.0.h, ids.h), shape, w.0.dtype)
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.stable {
            abi::drop_handle(self.h);
        }
    }
}

// -- det lane tensor ----------------------------------------------------------------------------

/// A det-lane tensor handle (CPU fp32, bit-exact everywhere, ABI §3.4/§5.9). Ingest-only; a
/// separate type from [`Tensor`] so lane mixing cannot compile.
#[derive(Debug)]
pub struct DetTensor {
    h: RawHandle,
    shape: Vec<u32>,
    stable: bool,
}

impl DetTensor {
    fn step(h: RawHandle, shape: Vec<u32>) -> Self {
        Self {
            h,
            shape,
            stable: false,
        }
    }

    /// The raw handle.
    #[must_use]
    pub fn handle(&self) -> RawHandle {
        self.h
    }

    /// The guest-tracked shape.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// `det_scale@1` (`alpha` cast `f64 → f32`, ABI §5.9).
    #[must_use]
    pub fn scale(&self, alpha: f64) -> DetTensor {
        DetTensor::step(abi::det_scale(self.h, alpha), self.shape.clone())
    }

    /// `det_l2norm@1` — fixed-order accumulation; safe to branch on (ABI §7).
    #[must_use]
    pub fn l2norm(&self) -> f64 {
        abi::det_l2norm(self.h)
    }

    /// `det_sign@1`.
    #[must_use]
    pub fn sign(&self) -> DetTensor {
        DetTensor::step(abi::det_sign(self.h), self.shape.clone())
    }

    /// `det_add@1`.
    #[must_use]
    pub fn add(&self, rhs: &DetTensor) -> DetTensor {
        DetTensor::step(abi::det_add(self.h, rhs.h), self.shape.clone())
    }

    /// `det_sub@1`.
    #[must_use]
    pub fn sub(&self, rhs: &DetTensor) -> DetTensor {
        DetTensor::step(abi::det_sub(self.h, rhs.h), self.shape.clone())
    }

    /// `det_mul@1`.
    #[must_use]
    pub fn mul(&self, rhs: &DetTensor) -> DetTensor {
        DetTensor::step(abi::det_mul(self.h, rhs.h), self.shape.clone())
    }

    /// `det_absmax_unpack@1` — decode a staged packed payload section (ABI §6.6).
    #[must_use]
    pub fn absmax_unpack(&self, chunk: u32, bits: u32) -> DetTensor {
        // Output shape is derived by the host from the decode; tracked loosely guest-side.
        DetTensor::step(abi::det_absmax_unpack(self.h, chunk, bits), Vec::new())
    }

    /// `det_chunk_scatter_add@1` — in-place `self[c·chunk + idx] += vals` (streaming ingest hot
    /// path, ABI §5.9). `self` is the accumulator.
    pub fn chunk_scatter_add(&mut self, vals: &DetTensor, idx: &DetTensor, chunk: u32) {
        abi::det_chunk_scatter_add(self.h, vals.h, idx.h, chunk);
    }

    /// `det_idct2@1` — det-lane inverse 2-D DCT (the demo-profile decode, ABI §5.9).
    #[must_use]
    pub fn idct2(&self, tile: u32) -> DetTensor {
        DetTensor::step(abi::det_idct2(self.h, tile), self.shape.clone())
    }
}

impl Drop for DetTensor {
    fn drop(&mut self) {
        if !self.stable {
            abi::drop_handle(self.h);
        }
    }
}

/// `det_zeros@1` — a fresh det-lane accumulator.
#[must_use]
pub fn det_zeros(dims: &[u32]) -> DetTensor {
    DetTensor::step(abi::det_zeros(dims), dims.to_vec())
}

/// `det_sum@1` — elementwise sum in **array order** (the post-clip reduce, ABI §5.9).
#[must_use]
pub fn det_sum(xs: &[&DetTensor]) -> DetTensor {
    let handles: Vec<RawHandle> = xs.iter().map(|d| d.h).collect();
    let shape = xs.first().map(|d| d.shape.clone()).unwrap_or_default();
    DetTensor::step(abi::det_sum(&handles), shape)
}

/// `det_chunk_scatter@1` — allocating dense-from-sparse (ABI §5.9).
#[must_use]
pub fn det_chunk_scatter(vals: &DetTensor, idx: &DetTensor, chunk: u32, dims: &[u32]) -> DetTensor {
    DetTensor::step(
        abi::det_chunk_scatter(vals.h, idx.h, chunk, dims),
        dims.to_vec(),
    )
}

// -- registered state ---------------------------------------------------------------------------

/// A trainable weight (ABI `param@1`) — a stable handle equal to its 1-based registration index
/// (ABI §3.3). Registration order **is** the canonical state dict (ABI §6.3).
#[derive(Debug)]
pub struct Param(Tensor);

impl Param {
    /// Register a param in `da_build`.
    #[must_use]
    pub fn new(name: &str, dims: &[u32], dtype: Dtype, init: Init, p0: f64, p1: f64) -> Self {
        let h = abi::param(name, dims, dtype.code(), init as u32, p0, p1);
        Param(Tensor {
            h,
            shape: dims.to_vec(),
            dtype,
            stable: true,
        })
    }

    /// The underlying native tensor view.
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    /// `grad@1` — a read-only view of the accumulated fp32 gradient.
    #[must_use]
    pub fn grad(&self) -> Tensor {
        Tensor::step(abi::grad(self.0.h), self.0.shape.clone(), Dtype::F32)
    }

    /// `param_round_base@1` — the native-lane view of θ at the start of the round (ABI §5.1).
    #[must_use]
    pub fn round_base(&self) -> Tensor {
        Tensor::step(
            abi::param_round_base(self.0.h),
            self.0.shape.clone(),
            self.0.dtype,
        )
    }

    /// `det_param@1` — the det-lane view of the round-base master snapshot (ABI §5.9). The only
    /// state-carrying read into the det lane.
    #[must_use]
    pub fn det_base(&self) -> DetTensor {
        DetTensor::step(abi::det_param(self.0.h), self.0.shape.clone())
    }

    /// `det_reset_param_to_base@1` — `master ← round-base snapshot` (the DiLoCo-family rebase).
    pub fn det_reset_to_base(&self) {
        abi::det_reset_param_to_base(self.0.h);
    }

    /// `det_axpy_param@1` — `master += alpha · x`, then requantize to storage (the outer step, and
    /// the only det→param doorway, ABI §5.9).
    pub fn det_axpy(&self, x: &DetTensor, alpha: f64) {
        abi::det_axpy_param(self.0.h, x.h, alpha);
    }

    /// `assign@1` — overwrite this param's storage with `src` (the only native-lane mutation path
    /// besides fused optimizer steps, ABI §5.1).
    pub fn assign(&self, src: &Tensor) {
        abi::assign(self.0.h, src.h);
    }

    /// `adamw_step@1` — fused inner optimizer step (legal only in `da_inner_update`, ABI §5.7).
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        &self,
        grad: &Tensor,
        m: &Persistent,
        v: &Persistent,
        step: u32,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
    ) {
        abi::adamw_step(
            self.0.h, grad.h, m.0.h, v.0.h, step, lr, beta1, beta2, eps, wd,
        );
    }
}

impl core::ops::Deref for Param {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.0
    }
}

/// Auxiliary native-lane state surviving across rounds (ABI `persistent@1`).
#[derive(Debug)]
pub struct Persistent(Tensor);

impl Persistent {
    /// `class = 0` (local): inner moments / error feedback; never digested, peers rebuild it.
    #[must_use]
    pub fn local(name: &str, dims: &[u32], dtype: Dtype) -> Self {
        Self(Self::reg(name, dims, dtype, 0))
    }

    /// `class = 1` (replicated): consensus state (digested + fp32-exact in checkpoints, ABI §5.1).
    #[must_use]
    pub fn replicated(name: &str, dims: &[u32], dtype: Dtype) -> Self {
        Self(Self::reg(name, dims, dtype, 1))
    }

    fn reg(name: &str, dims: &[u32], dtype: Dtype, class: u32) -> Tensor {
        let h = abi::persistent(name, dims, dtype.code(), class);
        Tensor {
            h,
            shape: dims.to_vec(),
            dtype,
            stable: true,
        }
    }

    /// The underlying native tensor view.
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }
}

/// Auxiliary det-lane fp32 state surviving across rounds (ABI `det_persistent@1`).
#[derive(Debug)]
pub struct DetPersistent(DetTensor);

impl DetPersistent {
    /// `class = 0` (local): det-side scratch that is legitimately peer-divergent.
    #[must_use]
    pub fn local(name: &str, dims: &[u32]) -> Self {
        Self(Self::reg(name, dims, 0))
    }

    /// `class = 1` (replicated): outer momentum / EMA feeding the outer step — MUST be replicated
    /// or rejoiners permanently desync (ABI §5.1).
    #[must_use]
    pub fn replicated(name: &str, dims: &[u32]) -> Self {
        Self(Self::reg(name, dims, 1))
    }

    fn reg(name: &str, dims: &[u32], class: u32) -> DetTensor {
        let h = abi::det_persistent(name, dims, class);
        DetTensor {
            h,
            shape: dims.to_vec(),
            stable: true,
        }
    }

    /// `det_assign@1` — overwrite the persistent (outer-momentum / EMA update, ABI §5.9).
    pub fn assign(&self, src: &DetTensor) {
        abi::det_assign(self.0.h, src.h);
    }

    /// The underlying det tensor view.
    #[must_use]
    pub fn tensor(&self) -> &DetTensor {
        &self.0
    }
}

// -- batch / step context -----------------------------------------------------------------------

/// A training micro-batch (ABI §5.10). Handle-only; the host owns the tokens.
#[derive(Debug, Clone, Copy)]
pub struct Batch {
    h: RawHandle,
}

impl Batch {
    /// Wrap a batch handle (used by the `experiment!` `da_step` glue).
    #[must_use]
    pub fn from_handle(h: RawHandle) -> Self {
        Self { h }
    }

    /// `batch_tokens@1` — `U32 [batch, seq_len]` token ids.
    #[must_use]
    pub fn tokens(&self) -> Tensor {
        let b = self.size();
        let s = self.seq_len();
        Tensor::step(abi::batch_tokens(self.h), vec![b, s], Dtype::U32)
    }

    /// `batch_size@1`.
    #[must_use]
    pub fn size(&self) -> u32 {
        abi::batch_size(self.h)
    }

    /// `batch_seq_len@1`.
    #[must_use]
    pub fn seq_len(&self) -> u32 {
        abi::batch_seq_len(self.h)
    }
}

/// `da_step` accumulation context (ABI §10.1).
#[derive(Debug, Clone, Copy)]
pub struct StepCtx {
    /// Run-monotonic inner step (never resets per round, ABI §2.3).
    pub inner_step: u32,
    /// This micro-batch's index within the inner step, `[0, mb_count)`.
    pub mb_index: u32,
    /// Micro-batch count for this inner step.
    pub mb_count: u32,
    /// Σ sequences across this inner step's micro-batches.
    pub step_seqs: u32,
}

impl StepCtx {
    /// `size(batch) / step_seqs` — scale each micro-batch loss so accumulated grads equal the exact
    /// step mean, independent of the host's OOM-probed slicing (ABI §4).
    #[must_use]
    pub fn loss_scale(&self, batch: &Batch) -> f64 {
        f64::from(batch.size()) / f64::from(self.step_seqs.max(1))
    }
}

// -- config / manifest --------------------------------------------------------------------------

/// A serde-deserializable view of `[experiment.config]` (canonical CBOR, ABI §6.1).
#[derive(Debug, Clone)]
pub struct Config {
    bytes: Vec<u8>,
}

impl Config {
    /// Wrap raw config CBOR bytes.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Serialize a value into a `Config` (test/authoring convenience).
    #[must_use]
    pub fn from_value<T: serde::Serialize>(value: &T) -> Self {
        let mut bytes = Vec::new();
        ciborium::into_writer(value, &mut bytes).expect("config value must be CBOR-serializable");
        Self { bytes }
    }

    /// The raw config bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Deserialize the config into `T` (ABI §6.1). Panics (→ `GuestPanic`, ABI §3.6) on a schema
    /// mismatch — the module is misconfigured.
    #[must_use]
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> T {
        ciborium::from_reader(self.bytes.as_slice()).expect("config CBOR must match the schema")
    }
}

/// The module cadence + round-mode block (ABI §6.2), returned by `da_manifest`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// Module name.
    pub name: String,
    /// Module version.
    pub version: String,
    /// SDK version string.
    pub sdk: String,
    /// H — inner-step cadence the host paces (ABI §2.3).
    pub steps_per_round: u32,
    /// Apply orderings this experiment's math tolerates (`"barrier"` / `"pipelined"`).
    pub round_modes: Vec<String>,
    /// Minimum viable round interval (ms); `0 = any`.
    pub min_round_interval_ms: u32,
}

impl Manifest {
    /// A barrier-mode manifest with the given cadence.
    #[must_use]
    pub fn new(name: &str, version: &str, steps_per_round: u32) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            sdk: env!("CARGO_PKG_VERSION").to_string(),
            steps_per_round,
            round_modes: vec!["barrier".to_string()],
            min_round_interval_ms: 0,
        }
    }

    /// Encode to CBOR (the bytes `da_manifest` returns).
    #[must_use]
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).expect("manifest is always CBOR-serializable");
        bytes
    }
}

// -- update container ---------------------------------------------------------------------------

/// Build side of the update container (`da_make_update`, ABI §5.11).
#[derive(Debug)]
pub struct UpdateBuilder {
    u: RawHandle,
}

impl UpdateBuilder {
    /// `upd_new@1`.
    #[must_use]
    pub fn new() -> Self {
        Self { u: abi::upd_new() }
    }

    /// `upd_push_tensor@1` — serialize a native tensor section.
    pub fn push_tensor(&mut self, x: &Tensor) {
        abi::upd_push_tensor(self.u, x.h);
    }

    /// `upd_push_bytes@1` — an experiment-defined opaque section.
    pub fn push_bytes(&mut self, data: &[u8]) {
        abi::upd_push_bytes(self.u, data);
    }

    /// The container handle (non-consuming; used by the `sim` staging driver).
    #[must_use]
    pub fn handle(&self) -> RawHandle {
        self.u
    }

    /// The container handle the host seals (returned from `da_make_update`).
    #[must_use]
    pub fn into_handle(self) -> RawHandle {
        self.u
    }
}

impl Default for UpdateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Section kind of a staged update section (`upd_kind@1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// An opaque byte section.
    Bytes,
    /// A tensor section (staged as det lane).
    Tensor,
}

/// Ingest side of the update container (`da_ingest_updates`, ABI §5.11): the host has staged
/// exactly the round record's committed set, in record order.
#[derive(Debug, Clone, Copy)]
pub struct UpdatesView {
    count: u32,
}

impl UpdatesView {
    /// Construct from the `count` the host passed to `da_ingest_updates`.
    #[must_use]
    pub fn with_count(count: u32) -> Self {
        Self { count }
    }

    /// The number of staged updates.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.count
    }

    /// Whether the staged set is empty (should never be at ingest — the host stalls instead).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// A reader over staged update `i` (record order).
    #[must_use]
    pub fn get(&self, i: u32) -> UpdateRef {
        UpdateRef { i }
    }
}

/// A reader over one staged update.
#[derive(Debug, Clone, Copy)]
pub struct UpdateRef {
    i: u32,
}

impl UpdateRef {
    /// `upd_sections@1`.
    #[must_use]
    pub fn sections(&self) -> u32 {
        abi::upd_sections(self.i)
    }

    /// `upd_kind@1`.
    #[must_use]
    pub fn kind(&self, section: u32) -> SectionKind {
        match abi::upd_kind(self.i, section) {
            0 => SectionKind::Bytes,
            _ => SectionKind::Tensor,
        }
    }

    /// `upd_read_bytes@1` — copy a byte section out.
    #[must_use]
    pub fn read_bytes(&self, section: u32) -> Vec<u8> {
        let len = abi::upd_bytes_len(self.i, section) as usize;
        let mut dst = vec![0u8; len];
        abi::upd_read_bytes(self.i, section, &mut dst);
        dst
    }

    /// `upd_tensor@1` — a tensor section, staged as det lane (ABI §3.4).
    #[must_use]
    pub fn tensor(&self, section: u32) -> DetTensor {
        DetTensor::step(abi::upd_tensor(self.i, section), Vec::new())
    }
}

// -- the Experiment trait -----------------------------------------------------------------------

/// An experiment module (ABI §10.2). The host drives the lifecycle; the implementor fills in the
/// math. Wire the `da_*` exports with [`crate::experiment!`].
pub trait Experiment: Sized {
    /// Cadence + round modes (ABI §6.2). Pure function of the config.
    fn manifest(cfg: &Config) -> Manifest;

    /// Register params/persistents from the config (`da_build`, ABI §6.3).
    fn build(cfg: &Config) -> Self;

    /// One micro-batch: forward + backward (accumulate) (`da_step`).
    fn step(&mut self, batch: &Batch, ctx: &StepCtx);

    /// Apply the inner optimizer at the accumulation boundary (`da_inner_update`).
    fn inner_update(&mut self, inner_step: u32);

    /// Compress local progress into an update container (`da_make_update`).
    fn make_update(&mut self, round: u64) -> UpdateBuilder;

    /// Decode + aggregate + outer step over the staged set (`da_ingest_updates`, det lane).
    fn ingest(&mut self, round: u64, updates: &UpdatesView);

    /// The `[experiment.config]` defaults layer (ABI §6.2). Defaults to an empty CBOR map.
    #[must_use]
    fn defaults() -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(&ciborium::value::Value::Map(Vec::new()), &mut bytes)
            .expect("empty map is always CBOR-serializable");
        bytes
    }
}

/// Read the host's implemented `tabi@1` minor via `abi_minor@1` (SDK diagnostics only, ABI §9).
#[must_use]
pub fn abi_minor() -> u32 {
    abi::abi_minor()
}

/// `log@1` — host-rate-limited tracing.
pub fn log(level: u32, msg: &str) {
    abi::log(level, msg);
}

/// `zero_grads@1` — clear all grad accumulators (typically in `da_inner_update` after applying).
pub fn zero_grads() {
    abi::zero_grads();
}
