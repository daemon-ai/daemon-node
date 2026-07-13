// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`BurnBackend`] — the [`OpBackend`] engine over a real Burn autodiff backend (G1, ABI §5).
//!
//! This is the Wave-2 engine seam the [`crate::backend`] module described as "burn slots in behind
//! this same trait". [`BurnBackend`] is generic over any [`AutodiffBackend`]; G1 proves it on
//! **burn-ndarray** (CPU) via the [`BurnNdarrayBackend`] alias, and G2 swaps the type parameter to
//! `Autodiff<Wgpu>` with no other change.
//!
//! ## Native lane — burn's own autodiff (HOST-9)
//!
//! Unlike [`crate::backend::CpuBackend`], which carries a hand-written reverse-mode tape, this
//! backend maps every native op onto burn tensor ops and lets **burn's `Autodiff` decorator** build
//! and differentiate the graph. `backward(loss)` calls `Tensor::backward`; `grad_of(param)` reads
//! `Tensor::grad`. burn's graph holds its own references to the intermediates, so the guest can drop
//! step tensors before `backward` (ABI §3.3) without losing them — the deferred-free discipline is
//! kept only to mirror [`CpuBackend`]'s handle-id recycling exactly.
//!
//! The native lane is therefore **not** bit-identical to [`CpuBackend`]: it is a *tolerance class*
//! (program ledger "Determinism story", spec §7.2). The `tests/tolerance.rs` harness pins the
//! per-op rtol/atol.
//!
//! ## Det lane — det-core CPU fp32, bit-exact (ABI §5.9 residency contract)
//!
//! Every `det_*` op and every compression native materializes the tensor **host-side** (`to_data`),
//! runs the same `det_core` kernel [`CpuBackend`] runs, and re-inserts the result. This is the
//! §5.9 "materialized CPU-side at the ingest boundary" residency contract: burn tensors on the
//! native side, fp32 masters host-side, requantize on `det_axpy`/writes. The consensus digest
//! (`digest_state` over the post-ingest masters, all written by det ops) is thus **backend-
//! independent** and bit-identical to [`CpuBackend`] — the cross-backend digest test is the guard.
//!
//! ## Tensor representation
//!
//! Each tensor is a flat rank-1 `Tensor<B, 1>` plus a cached host `Vec<f32>` so [`OpBackend::view`]
//! stays a cheap `&[f32]`. Shape-carrying ops (`matmul`, `transpose`, `slice`, `rmsnorm`, …) reshape
//! to the needed const rank internally and flatten back; `reshape@1` is an autodiff identity (a
//! tensor clone that keeps the graph edge). Trade: a 2× host copy, chosen for a clean `view` and
//! numerical fidelity over speed this wave.

#![cfg(feature = "burn-ndarray")]

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation, Int, Tensor, TensorData};

use crate::backend::{AdamwHp, OpBackend, TensorId};
use crate::trap::TrapCode;

/// [`BurnBackend`] on the ndarray CPU backend + autodiff — the G1 native lane.
pub type BurnNdarrayBackend = BurnBackend<burn::backend::Autodiff<burn::backend::NdArray>>;

/// One live tensor: the burn tensor + a cached host copy backing [`OpBackend::view`].
struct Slot<B: AutodiffBackend> {
    t: Tensor<B, 1>,
    host: Vec<f32>,
}

/// A [`OpBackend`] driven by a Burn [`AutodiffBackend`] (native lane) + `det_core` (det lane).
pub struct BurnBackend<B: AutodiffBackend> {
    device: B::Device,
    tensors: Vec<Option<Slot<B>>>,
    free: Vec<u32>,
    /// Gradients from the last `backward`, read by `grad_of` before `end_pass`.
    grads: Option<B::Gradients>,
    /// Whether a differentiable pass is recording (mirrors [`CpuBackend`]'s deferred-free discipline
    /// so freed step-handle ids recycle at `end_pass`, not mid-pass).
    recording: bool,
    /// Step-tensor ids freed while recording — actually recycled at `end_pass`.
    deferred: Vec<TensorId>,
}

impl<B: AutodiffBackend> Default for BurnBackend<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Read a rank-1 tensor's data as `Vec<f32>` (generic over any backend so it also serves the inner
/// grad tensor `Tensor<B::InnerBackend, 1>`).
fn to_vec_f32<BK: Backend>(t: &Tensor<BK, 1>) -> Vec<f32> {
    t.to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("f32 tensor data")
}

impl<B: AutodiffBackend> BurnBackend<B> {
    /// A fresh backend on the backend's default device.
    #[must_use]
    pub fn new() -> Self {
        Self::with_device(B::Device::default())
    }

    /// A fresh backend on `device`.
    #[must_use]
    pub fn with_device(device: B::Device) -> Self {
        Self {
            device,
            tensors: Vec::new(),
            free: Vec::new(),
            grads: None,
            recording: false,
            deferred: Vec::new(),
        }
    }

    fn mk(&self, data: Vec<f32>) -> Tensor<B, 1> {
        let n = data.len();
        Tensor::<B, 1>::from_data(TensorData::new(data, [n]), &self.device)
    }

    /// A fresh **leaf** tensor that can carry a gradient (params / persistents / det results / any
    /// value re-materialized from host data). Only `from_data` leaves may `require_grad`.
    fn leaf(&self, data: Vec<f32>) -> Tensor<B, 1> {
        self.mk(data).require_grad()
    }

    fn insert(&mut self, slot: Slot<B>) -> TensorId {
        if let Some(id) = self.free.pop() {
            self.tensors[id as usize] = Some(slot);
            id
        } else {
            self.tensors.push(Some(slot));
            (self.tensors.len() - 1) as TensorId
        }
    }

    fn insert_leaf(&mut self, data: Vec<f32>) -> TensorId {
        let host = data.clone();
        let t = self.leaf(data);
        self.insert(Slot { t, host })
    }

    fn insert_result(&mut self, t: Tensor<B, 1>) -> TensorId {
        let host = to_vec_f32(&t);
        self.insert(Slot { t, host })
    }

    fn slot(&self, id: TensorId) -> &Slot<B> {
        self.tensors[id as usize]
            .as_ref()
            .expect("live backend tensor")
    }

    fn t1(&self, id: TensorId) -> Tensor<B, 1> {
        self.slot(id).t.clone()
    }

    fn host(&self, id: TensorId) -> &[f32] {
        &self.slot(id).host
    }

    fn u32s(&self, id: TensorId) -> Vec<u32> {
        self.host(id).iter().map(|&f| f as u32).collect()
    }
}

/// Map a `det_core` error to the ABI trap taxonomy (same mapping as [`CpuBackend`]).
fn det_trap(e: det_core::DetError) -> TrapCode {
    match e {
        det_core::DetError::UnsupportedBits { .. } => TrapCode::BadEnum,
        _ => TrapCode::ShapeMismatch,
    }
}

/// Host-side axis permutation preserving values (the transpose/slice fallback for ranks the burn
/// const-generic arms below do not cover — ranks ≥5, unused by the tiny-llama model + G1 harness;
/// forward-correct, grad is cut for those ranks only).
fn permute_axes(data: &[f32], shape_in: &[usize], d0: usize, d1: usize) -> Vec<f32> {
    let mut shape_out = shape_in.to_vec();
    shape_out.swap(d0, d1);
    let sin = row_major_strides(shape_in);
    let sout = row_major_strides(&shape_out);
    let rank = shape_in.len();
    let mut out = vec![0.0_f32; data.len()];
    let mut coord = vec![0usize; rank];
    for (flat, &val) in data.iter().enumerate() {
        let mut rem = flat;
        for r in 0..rank {
            coord[r] = rem / sin[r];
            rem %= sin[r];
        }
        coord.swap(d0, d1);
        let mut out_flat = 0usize;
        for r in 0..rank {
            out_flat += coord[r] * sout[r];
        }
        out[out_flat] = val;
        coord.swap(d0, d1);
    }
    out
}

fn slice_dim(data: &[f32], shape_in: &[usize], dim: usize, start: usize, end: usize) -> Vec<f32> {
    let mut shape_out = shape_in.to_vec();
    shape_out[dim] = end - start;
    let sin = row_major_strides(shape_in);
    let sout = row_major_strides(&shape_out);
    let rank = shape_in.len();
    let n: usize = shape_out.iter().product();
    let mut out = vec![0.0_f32; n];
    let mut coord = vec![0usize; rank];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut rem = flat;
        for r in 0..rank {
            coord[r] = rem / sout[r];
            rem %= sout[r];
        }
        let mut in_flat = 0usize;
        for r in 0..rank {
            let c = if r == dim { coord[r] + start } else { coord[r] };
            in_flat += c * sin[r];
        }
        *o = data[in_flat];
    }
    out
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<B: AutodiffBackend> OpBackend for BurnBackend<B> {
    fn create(&mut self, data: Vec<f32>) -> TensorId {
        self.insert_leaf(data)
    }
    fn zeros(&mut self, n: usize) -> TensorId {
        self.insert_leaf(vec![0.0; n])
    }
    fn clone_tensor(&mut self, id: TensorId) -> TensorId {
        // A detached copy (a fresh leaf) — used for read-only grad / round-base views and `detach@1`.
        let data = self.host(id).to_vec();
        self.insert_leaf(data)
    }
    fn view(&self, id: TensorId) -> &[f32] {
        self.host(id)
    }
    fn write(&mut self, id: TensorId, data: &[f32]) {
        // A write installs a fresh leaf: the next differentiable pass differentiates through it
        // (params re-synced after an optimizer step, det doorway masters — ABI §5.9).
        let host = data.to_vec();
        let t = self.leaf(host.clone());
        self.tensors[id as usize] = Some(Slot { t, host });
    }
    fn free(&mut self, id: TensorId) {
        if (id as usize) < self.tensors.len() && self.tensors[id as usize].is_some() {
            if self.recording {
                self.deferred.push(id);
                return;
            }
            self.tensors[id as usize] = None;
            self.free.push(id);
        }
    }

    fn begin_pass(&mut self) {
        self.recording = true;
        self.grads = None;
    }

    fn end_pass(&mut self) {
        self.recording = false;
        for id in std::mem::take(&mut self.deferred) {
            if (id as usize) < self.tensors.len() && self.tensors[id as usize].is_some() {
                self.tensors[id as usize] = None;
                self.free.push(id);
            }
        }
        self.grads = None;
    }

    fn grad_of(&self, id: TensorId) -> Option<Vec<f32>> {
        let grads = self.grads.as_ref()?;
        let slot = self.tensors.get(id as usize)?.as_ref()?;
        slot.t.grad(grads).map(|g| to_vec_f32(&g))
    }

    fn reshape(&mut self, x: TensorId) -> TensorId {
        // Identity data + shape change (runtime tracks the shape); the tensor clone keeps the
        // autodiff edge so gradients flow through (HOST-9).
        let t = self.t1(x);
        self.insert_result(t)
    }

    fn matmul(&mut self, a: TensorId, m: usize, k: usize, b: TensorId, n: usize) -> TensorId {
        let a2 = self.t1(a).reshape([m, k]);
        let b2 = self.t1(b).reshape([k, n]);
        let out = a2.matmul(b2).reshape([m * n]);
        self.insert_result(out)
    }
    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = self.t1(a).add(self.t1(b));
        self.insert_result(out)
    }
    fn add_bias(&mut self, a: TensorId, b: TensorId, rows: usize, cols: usize) -> TensorId {
        let a2 = self.t1(a).reshape([rows, cols]);
        let b2 = self.t1(b).reshape([1, cols]);
        let out = a2.add(b2).reshape([rows * cols]);
        self.insert_result(out)
    }
    fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = self.t1(a).sub(self.t1(b));
        self.insert_result(out)
    }
    fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out = self.t1(a).mul(self.t1(b));
        self.insert_result(out)
    }
    fn mul_s(&mut self, x: TensorId, s: f64) -> TensorId {
        let out = self.t1(x).mul_scalar(s as f32);
        self.insert_result(out)
    }
    fn relu(&mut self, x: TensorId) -> TensorId {
        let out = activation::relu(self.t1(x));
        self.insert_result(out)
    }
    fn cross_entropy(
        &mut self,
        logits: TensorId,
        rows: usize,
        cols: usize,
        targets: &[i64],
        ignore: i64,
    ) -> TensorId {
        let l2 = self.t1(logits).reshape([rows, cols]);
        // Subtract the per-row max as a *constant* (detached) — the shift-invariant, numerically
        // stable log-softmax, matching CpuBackend's `row - max` (max treated as constant).
        let max = l2.clone().max_dim(1).detach();
        let shifted = l2.sub(max);
        let logsm = shifted.clone().sub(shifted.exp().sum_dim(1).log());
        // One-hot targets + counted-row normalizer (constants): loss = -Σ onehot·logsoftmax / counted.
        let mut onehot = vec![0.0_f32; rows * cols];
        let mut counted = 0.0_f32;
        for i in 0..rows {
            let t = targets.get(i).copied().unwrap_or(ignore);
            if t != ignore {
                onehot[i * cols + t as usize] = 1.0;
                counted += 1.0;
            }
        }
        let denom = counted.max(1.0);
        let oh = Tensor::<B, 2>::from_data(TensorData::new(onehot, [rows, cols]), &self.device);
        let loss = logsm.mul(oh).sum().mul_scalar(-1.0 / denom);
        self.insert_result(loss)
    }

    fn embedding(&mut self, w: TensorId, ids: &[usize], d: usize) -> TensorId {
        let vocab = self.host(w).len() / d;
        let w2 = self.t1(w).reshape([vocab, d]);
        let idx_data: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(idx_data, [ids.len()]), &self.device);
        let out = w2.select(0, idx).reshape([ids.len() * d]);
        self.insert_result(out)
    }
    fn rmsnorm(&mut self, x: TensorId, w: TensorId, rows: usize, d: usize, eps: f64) -> TensorId {
        let x2 = self.t1(x).reshape([rows, d]);
        let w2 = self.t1(w).reshape([1, d]);
        let ms = x2.clone().powf_scalar(2.0).mean_dim(1); // [rows,1]
        let inv = ms.add_scalar(eps).sqrt().recip(); // [rows,1]
        let out = x2.mul(inv).mul(w2).reshape([rows * d]);
        self.insert_result(out)
    }
    fn silu(&mut self, x: TensorId) -> TensorId {
        let out = activation::silu(self.t1(x));
        self.insert_result(out)
    }
    fn softmax(&mut self, x: TensorId, outer: usize, dimlen: usize, inner: usize) -> TensorId {
        let x3 = self.t1(x).reshape([outer, dimlen, inner]);
        let out = activation::softmax(x3, 1).reshape([outer * dimlen * inner]);
        self.insert_result(out)
    }
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &mut self,
        x: TensorId,
        rows: usize,
        seq: usize,
        hd: usize,
        pos_start: usize,
        theta: f64,
        interleaved: bool,
    ) -> TensorId {
        let half = hd / 2;
        let thetaf = theta as f32;
        let mut cosv = vec![0.0_f32; rows * half];
        let mut sinv = vec![0.0_f32; rows * half];
        for r in 0..rows {
            let pos = (pos_start + (r % seq)) as f32;
            for j in 0..half {
                let freq = 1.0 / thetaf.powf(2.0 * j as f32 / hd as f32);
                let angle = pos * freq;
                cosv[r * half + j] = angle.cos();
                sinv[r * half + j] = angle.sin();
            }
        }
        let cos = Tensor::<B, 2>::from_data(TensorData::new(cosv, [rows, half]), &self.device);
        let sin = Tensor::<B, 2>::from_data(TensorData::new(sinv, [rows, half]), &self.device);
        let out = if interleaved {
            let x3 = self.t1(x).reshape([rows, half, 2]);
            let a = x3.clone().narrow(2, 0, 1);
            let b = x3.narrow(2, 1, 1);
            let cos3 = cos.reshape([rows, half, 1]);
            let sin3 = sin.reshape([rows, half, 1]);
            let oa = a.clone().mul(cos3.clone()).sub(b.clone().mul(sin3.clone()));
            let ob = a.mul(sin3).add(b.mul(cos3));
            Tensor::cat(vec![oa, ob], 2).reshape([rows * hd])
        } else {
            let x2 = self.t1(x).reshape([rows, hd]);
            let x1 = x2.clone().narrow(1, 0, half);
            let x2b = x2.narrow(1, half, half);
            let out1 = x1
                .clone()
                .mul(cos.clone())
                .sub(x2b.clone().mul(sin.clone()));
            let out2 = x1.mul(sin).add(x2b.mul(cos));
            Tensor::cat(vec![out1, out2], 1).reshape([rows * hd])
        };
        self.insert_result(out)
    }
    #[allow(clippy::too_many_arguments)]
    fn flash_attn(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        bh: usize,
        s: usize,
        d: usize,
        causal: bool,
        scale: f64,
    ) -> TensorId {
        let q3 = self.t1(q).reshape([bh, s, d]);
        let k3 = self.t1(k).reshape([bh, s, d]);
        let v3 = self.t1(v).reshape([bh, s, d]);
        let scores = q3.matmul(k3.swap_dims(1, 2)).mul_scalar(scale as f32); // [bh,s,s]
        let scores = if causal {
            let mut m = vec![0.0_f32; s * s];
            for i in 0..s {
                for j in (i + 1)..s {
                    m[i * s + j] = -1.0e30;
                }
            }
            let mask = Tensor::<B, 3>::from_data(TensorData::new(m, [1, s, s]), &self.device);
            scores.add(mask)
        } else {
            scores
        };
        let probs = activation::softmax(scores, 2);
        let out = probs.matmul(v3).reshape([bh * s * d]);
        self.insert_result(out)
    }
    fn transpose(&mut self, x: TensorId, shape_in: &[usize], d0: usize, d1: usize) -> TensorId {
        let n = self.host(x).len();
        let out = match shape_in.len() {
            0 | 1 => self.t1(x),
            2 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1]])
                .swap_dims(d0, d1)
                .reshape([n]),
            3 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1], shape_in[2]])
                .swap_dims(d0, d1)
                .reshape([n]),
            4 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1], shape_in[2], shape_in[3]])
                .swap_dims(d0, d1)
                .reshape([n]),
            _ => {
                // rank ≥5: forward-correct host permutation (unused by the model/harness).
                let data = permute_axes(self.host(x), shape_in, d0, d1);
                self.mk(data)
            }
        };
        self.insert_result(out)
    }
    fn slice(
        &mut self,
        x: TensorId,
        shape_in: &[usize],
        dim: usize,
        start: usize,
        end: usize,
    ) -> TensorId {
        let len = end - start;
        let mut out_shape = shape_in.to_vec();
        out_shape[dim] = len;
        let on: usize = out_shape.iter().product();
        let out = match shape_in.len() {
            1 => self
                .t1(x)
                .reshape([shape_in[0]])
                .narrow(dim, start, len)
                .reshape([on]),
            2 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1]])
                .narrow(dim, start, len)
                .reshape([on]),
            3 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1], shape_in[2]])
                .narrow(dim, start, len)
                .reshape([on]),
            4 => self
                .t1(x)
                .reshape([shape_in[0], shape_in[1], shape_in[2], shape_in[3]])
                .narrow(dim, start, len)
                .reshape([on]),
            _ => {
                let data = slice_dim(self.host(x), shape_in, dim, start, end);
                self.mk(data)
            }
        };
        self.insert_result(out)
    }

    fn backward(&mut self, loss: TensorId) {
        let t = self.t1(loss);
        self.grads = Some(t.backward());
    }

    fn adamw_step(
        &mut self,
        master: TensorId,
        grad: TensorId,
        m: TensorId,
        v: TensorId,
        hp: AdamwHp,
    ) {
        // Fused AdamW on burn tensor ops (native lane, detached from any graph). f32 arithmetic vs
        // CpuBackend's f64 accumulation is a genuine tolerance-class divergence (Optimizer class);
        // the det-lane rebase discards it at ingest, so it never reaches the consensus digest.
        let t = i32::try_from(hp.step.max(1)).unwrap_or(i32::MAX);
        let bc1 = 1.0 - hp.beta1.powi(t);
        let bc2 = 1.0 - hp.beta2.powi(t);
        let g = self.t1(grad).detach();
        let m0 = self.t1(m).detach();
        let v0 = self.t1(v).detach();
        let w0 = self.t1(master).detach();

        let m1 = m0
            .mul_scalar(hp.beta1)
            .add(g.clone().mul_scalar(1.0 - hp.beta1));
        let v1 = v0
            .mul_scalar(hp.beta2)
            .add(g.clone().mul(g).mul_scalar(1.0 - hp.beta2));
        let mhat = m1.clone().div_scalar(bc1);
        let vhat = v1.clone().div_scalar(bc2);
        let denom = vhat.sqrt().add_scalar(hp.eps);
        let w1 = w0
            .mul_scalar(1.0 - hp.lr * hp.wd)
            .sub(mhat.div(denom).mul_scalar(hp.lr));

        let mh = to_vec_f32(&m1);
        let vh = to_vec_f32(&v1);
        let wh = to_vec_f32(&w1);
        self.tensors[m as usize] = Some(Slot { t: m1, host: mh });
        self.tensors[v as usize] = Some(Slot { t: v1, host: vh });
        self.tensors[master as usize] = Some(Slot { t: w1, host: wh });
    }

    // -- det lane (real det-core kernels; host-side materialization, §5.9) ----------------------

    fn det_sum(&mut self, xs: &[TensorId]) -> Result<TensorId, TrapCode> {
        let vecs: Vec<Vec<f32>> = xs.iter().map(|&id| self.host(id).to_vec()).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(Vec::as_slice).collect();
        let out = det_core::det_sum(&refs).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_scale(&mut self, x: TensorId, alpha: f64) -> TensorId {
        let out = det_core::det_scale(self.host(x), alpha);
        self.insert_leaf(out)
    }
    fn det_l2norm(&self, x: TensorId) -> f32 {
        det_core::det_l2norm(self.host(x))
    }
    fn det_sign(&mut self, x: TensorId) -> TensorId {
        let out = det_core::det_sign(self.host(x));
        self.insert_leaf(out)
    }
    fn det_add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_add(self.host(a), self.host(b)).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_sub(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_sub(self.host(a), self.host(b)).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_mul(self.host(a), self.host(b)).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_absmax_unpack(
        &mut self,
        packed: TensorId,
        chunk: usize,
        bits: u32,
    ) -> Result<TensorId, TrapCode> {
        let bytes: Vec<u8> = self.host(packed).iter().map(|&f| f as u8).collect();
        let out = det_core::det_absmax_unpack(&bytes, chunk, bits).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_chunk_scatter_add(
        &mut self,
        acc: TensorId,
        vals: TensorId,
        idx: TensorId,
        chunk: usize,
    ) -> Result<(), TrapCode> {
        let valsv = self.host(vals).to_vec();
        let idxv = self.u32s(idx);
        let mut accv = self.host(acc).to_vec();
        det_core::det_chunk_scatter_add(&mut accv, &valsv, &idxv, chunk).map_err(det_trap)?;
        self.write(acc, &accv);
        Ok(())
    }
    fn det_chunk_scatter(
        &mut self,
        vals: TensorId,
        idx: TensorId,
        chunk: usize,
        out_len: usize,
    ) -> Result<TensorId, TrapCode> {
        let out = det_core::det_chunk_scatter(self.host(vals), &self.u32s(idx), chunk, out_len)
            .map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn det_axpy(&mut self, y: TensorId, alpha: f64, x: TensorId) -> Result<(), TrapCode> {
        let xv = self.host(x).to_vec();
        let mut yv = self.host(y).to_vec();
        det_core::det_axpy(&mut yv, alpha, &xv).map_err(det_trap)?;
        self.write(y, &yv);
        Ok(())
    }

    fn topk_chunk(
        &mut self,
        x: TensorId,
        chunk: usize,
        k: usize,
    ) -> Result<(TensorId, TensorId), TrapCode> {
        let (vals, idx) = det_core::topk_chunk(self.host(x), chunk, k).map_err(det_trap)?;
        let ivals: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
        let vh = self.insert_leaf(vals);
        let ih = self.insert_leaf(ivals);
        Ok((vh, ih))
    }
    fn absmax_pack(&mut self, x: TensorId, chunk: usize, bits: u32) -> Result<TensorId, TrapCode> {
        let packed = det_core::absmax_pack(self.host(x), chunk, bits).map_err(det_trap)?;
        let vals: Vec<f32> = packed.iter().map(|&b| f32::from(b)).collect();
        Ok(self.insert_leaf(vals))
    }
    fn dct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode> {
        let out = det_core::dct2(self.host(x), tile).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
    fn idct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode> {
        let out = det_core::idct2(self.host(x), tile).map_err(det_trap)?;
        Ok(self.insert_leaf(out))
    }
}
