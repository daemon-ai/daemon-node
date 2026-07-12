// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The op backend — the Wave-2 engine seam (ABI §5, architecture §5.1).
//!
//! [`OpBackend`] is the numeric engine behind the host dispatch layer. Wave 1 ships [`CpuBackend`],
//! a plain `Vec<f32>` fake: the **det lane** is the real `det-core` (so HOST-5/6 determinism holds),
//! the native lane is a functional fp32 forward (no autodiff yet — `backward` is a no-op, HOST-9
//! parity is Wave 2). Wave 2 slots burn/CubeCL in behind this same trait; the arena, trap taxonomy,
//! phase table, and budgets above it stay lane-E stable.
//!
//! Tensors are addressed by [`TensorId`]. Shape validation for native ops is the caller's
//! ([`crate::runtime`]) responsibility (so host functions never panic across the ABI boundary); the
//! det ops return typed traps directly, since they are the consensus-critical path.

use crate::trap::TrapCode;

/// A backend tensor identity (opaque to the ABI layer).
pub type TensorId = u32;

/// AdamW hyperparameters for a fused step (ABI §5.7).
#[derive(Debug, Clone, Copy)]
pub struct AdamwHp {
    /// The 1-based optimizer timestep.
    pub step: u32,
    /// Learning rate.
    pub lr: f64,
    /// β₁.
    pub beta1: f64,
    /// β₂.
    pub beta2: f64,
    /// ε.
    pub eps: f64,
    /// Decoupled weight decay.
    pub wd: f64,
}

/// The numeric engine the host dispatch layer drives. Wave 2 replaces the impl, not the trait.
pub trait OpBackend {
    /// Store `data` as a fresh tensor.
    fn create(&mut self, data: Vec<f32>) -> TensorId;
    /// A fresh `n`-element zero tensor.
    fn zeros(&mut self, n: usize) -> TensorId;
    /// A copy of `id` (a fresh tensor) — used for read-only "views" in the fake.
    fn clone_tensor(&mut self, id: TensorId) -> TensorId;
    /// Read a tensor's data.
    fn view(&self, id: TensorId) -> &[f32];
    /// Overwrite a tensor's data (assign / requantize).
    fn write(&mut self, id: TensorId, data: &[f32]);
    /// Release a tensor.
    fn free(&mut self, id: TensorId);

    /// `matmul@1`: `a[m,k] · b[k,n]` (fake: fixed-order fp32).
    fn matmul(&mut self, a: TensorId, m: usize, k: usize, b: TensorId, n: usize) -> TensorId;
    /// `add@1` (same-shape elementwise).
    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId;
    /// `add@1` with a trailing-dim bias broadcast (`a[.., cols] + b[cols]`).
    fn add_bias(&mut self, a: TensorId, b: TensorId, rows: usize, cols: usize) -> TensorId;
    /// `sub@1`.
    fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId;
    /// `mul@1`.
    fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId;
    /// `mul_s@1`.
    fn mul_s(&mut self, x: TensorId, s: f64) -> TensorId;
    /// `relu@1`.
    fn relu(&mut self, x: TensorId) -> TensorId;
    /// `cross_entropy@1` — mean over non-ignored rows; rank-0.
    fn cross_entropy(
        &mut self,
        logits: TensorId,
        rows: usize,
        cols: usize,
        targets: &[i64],
        ignore: i64,
    ) -> TensorId;

    // -- Wave-2 NN / shape forward (fake: forward only; HOST-9 autodiff parity is future) -------

    /// `embedding@1` — gather rows `d`-wide of `w` (`[vocab, d]`) by `ids`.
    fn embedding(&mut self, w: TensorId, ids: &[usize], d: usize) -> TensorId;
    /// `rmsnorm@1` — RMS norm of `x` (`rows × d`) scaled by `w` (`[d]`).
    fn rmsnorm(&mut self, x: TensorId, w: TensorId, rows: usize, d: usize, eps: f64) -> TensorId;
    /// `silu@1`.
    fn silu(&mut self, x: TensorId) -> TensorId;
    /// `softmax@1` over a `[outer, dimlen, inner]` view.
    fn softmax(&mut self, x: TensorId, outer: usize, dimlen: usize, inner: usize) -> TensorId;
    /// `rope@1` — rotary position embedding over `[rows, hd]` (rows count `seq`-periodic).
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
    ) -> TensorId;
    /// `flash_attn@1` — fused attention over `[bh, s, d]` groups.
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
    ) -> TensorId;
    /// `transpose@1` — swap axes `d0`/`d1` of a row-major tensor with `shape_in`.
    fn transpose(&mut self, x: TensorId, shape_in: &[usize], d0: usize, d1: usize) -> TensorId;
    /// `slice@1` — `start..end` along `dim` of a row-major tensor with `shape_in`.
    fn slice(
        &mut self,
        x: TensorId,
        shape_in: &[usize],
        dim: usize,
        start: usize,
        end: usize,
    ) -> TensorId;

    /// `backward@1` — Wave-1 fake keeps grads as-is (no tape); HOST-9 autodiff parity is Wave 2.
    fn backward(&mut self, _loss: TensorId) {}
    /// `adamw_step@1` — updates `master` in place (fused, ABI §5.7).
    fn adamw_step(
        &mut self,
        master: TensorId,
        grad: TensorId,
        m: TensorId,
        v: TensorId,
        hp: AdamwHp,
    );

    // -- det lane (real det-core kernels) -------------------------------------------------------

    /// `det_sum@1`.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on unequal operand lengths.
    fn det_sum(&mut self, xs: &[TensorId]) -> Result<TensorId, TrapCode>;
    /// `det_scale@1`.
    fn det_scale(&mut self, x: TensorId, alpha: f64) -> TensorId;
    /// `det_l2norm@1`.
    fn det_l2norm(&self, x: TensorId) -> f32;
    /// `det_sign@1`.
    fn det_sign(&mut self, x: TensorId) -> TensorId;
    /// `det_add@1`.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on unequal operand lengths.
    fn det_add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode>;
    /// `det_sub@1`.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on unequal operand lengths.
    fn det_sub(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode>;
    /// `det_mul@1`.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on unequal operand lengths.
    fn det_mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode>;
    /// `det_absmax_unpack@1`.
    ///
    /// # Errors
    /// [`TrapCode::BadEnum`] for a bad bit width; [`TrapCode::ShapeMismatch`] for a bad layout.
    fn det_absmax_unpack(
        &mut self,
        packed: TensorId,
        chunk: usize,
        bits: u32,
    ) -> Result<TensorId, TrapCode>;
    /// `det_chunk_scatter_add@1` (in place on `acc`).
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on a bad chunk layout or out-of-range index.
    fn det_chunk_scatter_add(
        &mut self,
        acc: TensorId,
        vals: TensorId,
        idx: TensorId,
        chunk: usize,
    ) -> Result<(), TrapCode>;
    /// `det_chunk_scatter@1`.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on a bad chunk layout or out-of-range index.
    fn det_chunk_scatter(
        &mut self,
        vals: TensorId,
        idx: TensorId,
        chunk: usize,
        out_len: usize,
    ) -> Result<TensorId, TrapCode>;
    /// `det_axpy_param@1` numeric core: `y += alpha · x` in place.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on unequal lengths.
    fn det_axpy(&mut self, y: TensorId, alpha: f64, x: TensorId) -> Result<(), TrapCode>;

    // -- Wave-2 compression natives (shared det-core reference; lane-agnostic tensors) ----------

    /// `topk_chunk@1` — per-chunk top-k by magnitude → `(values, indices)` tensors.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on a bad chunk/k layout.
    fn topk_chunk(
        &mut self,
        x: TensorId,
        chunk: usize,
        k: usize,
    ) -> Result<(TensorId, TensorId), TrapCode>;
    /// `absmax_pack@1` — per-chunk absmax codebook quant to a packed `U8` tensor (§6.6).
    ///
    /// # Errors
    /// [`TrapCode::BadEnum`] for a bad bit width; [`TrapCode::ShapeMismatch`] for a bad layout.
    fn absmax_pack(&mut self, x: TensorId, chunk: usize, bits: u32) -> Result<TensorId, TrapCode>;
    /// `dct2@1` — orthonormal 2-D DCT over `tile²` blocks.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on a bad tile layout.
    fn dct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode>;
    /// `idct2@1` / `det_idct2@1` — the inverse 2-D DCT.
    ///
    /// # Errors
    /// [`TrapCode::ShapeMismatch`] on a bad tile layout.
    fn idct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode>;
}

/// The Wave-1 CPU fake: a `Vec<f32>` tensor arena.
#[derive(Default)]
pub struct CpuBackend {
    tensors: Vec<Option<Vec<f32>>>,
    free: Vec<u32>,
}

impl CpuBackend {
    /// A fresh backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, data: Vec<f32>) -> TensorId {
        if let Some(id) = self.free.pop() {
            self.tensors[id as usize] = Some(data);
            id
        } else {
            self.tensors.push(Some(data));
            (self.tensors.len() - 1) as TensorId
        }
    }

    fn get(&self, id: TensorId) -> &[f32] {
        self.tensors[id as usize]
            .as_deref()
            .expect("live backend tensor")
    }

    fn u32s(&self, id: TensorId) -> Vec<u32> {
        self.get(id).iter().map(|&f| f as u32).collect()
    }
}

impl OpBackend for CpuBackend {
    fn create(&mut self, data: Vec<f32>) -> TensorId {
        self.insert(data)
    }
    fn zeros(&mut self, n: usize) -> TensorId {
        self.insert(vec![0.0; n])
    }
    fn clone_tensor(&mut self, id: TensorId) -> TensorId {
        let data = self.get(id).to_vec();
        self.insert(data)
    }
    fn view(&self, id: TensorId) -> &[f32] {
        self.get(id)
    }
    fn write(&mut self, id: TensorId, data: &[f32]) {
        self.tensors[id as usize] = Some(data.to_vec());
    }
    fn free(&mut self, id: TensorId) {
        if (id as usize) < self.tensors.len() && self.tensors[id as usize].is_some() {
            self.tensors[id as usize] = None;
            self.free.push(id);
        }
    }

    fn matmul(&mut self, a: TensorId, m: usize, k: usize, b: TensorId, n: usize) -> TensorId {
        let av = self.get(a);
        let bv = self.get(b);
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += av[i * k + p] * bv[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        self.insert(out)
    }
    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out: Vec<f32> = self
            .get(a)
            .iter()
            .zip(self.get(b).iter())
            .map(|(&x, &y)| x + y)
            .collect();
        self.insert(out)
    }
    fn add_bias(&mut self, a: TensorId, b: TensorId, rows: usize, cols: usize) -> TensorId {
        let mut out = self.get(a).to_vec();
        let bias = self.get(b);
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] += bias[j];
            }
        }
        self.insert(out)
    }
    fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out: Vec<f32> = self
            .get(a)
            .iter()
            .zip(self.get(b).iter())
            .map(|(&x, &y)| x - y)
            .collect();
        self.insert(out)
    }
    fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let out: Vec<f32> = self
            .get(a)
            .iter()
            .zip(self.get(b).iter())
            .map(|(&x, &y)| x * y)
            .collect();
        self.insert(out)
    }
    fn mul_s(&mut self, x: TensorId, s: f64) -> TensorId {
        let s = s as f32;
        let out: Vec<f32> = self.get(x).iter().map(|&e| e * s).collect();
        self.insert(out)
    }
    fn relu(&mut self, x: TensorId) -> TensorId {
        let out: Vec<f32> = self.get(x).iter().map(|&e| e.max(0.0)).collect();
        self.insert(out)
    }
    fn cross_entropy(
        &mut self,
        logits: TensorId,
        rows: usize,
        cols: usize,
        targets: &[i64],
        ignore: i64,
    ) -> TensorId {
        let lv = self.get(logits);
        let mut loss = 0.0_f32;
        let mut counted = 0.0_f32;
        for i in 0..rows {
            let t = targets.get(i).copied().unwrap_or(ignore);
            if t == ignore {
                continue;
            }
            let row = &lv[i * cols..(i + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = row.iter().map(|&e| (e - max).exp()).sum();
            let p = ((row[t as usize] - max).exp() / denom).max(1.0e-12);
            loss -= p.ln();
            counted += 1.0;
        }
        let mean = if counted > 0.0 { loss / counted } else { 0.0 };
        self.insert(vec![mean])
    }

    fn embedding(&mut self, w: TensorId, ids: &[usize], d: usize) -> TensorId {
        let wv = self.get(w);
        let mut out = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            out.extend_from_slice(&wv[id * d..id * d + d]);
        }
        self.insert(out)
    }
    fn rmsnorm(&mut self, x: TensorId, w: TensorId, rows: usize, d: usize, eps: f64) -> TensorId {
        let xv = self.get(x);
        let wv = self.get(w);
        let eps = eps as f32;
        let mut out = vec![0.0_f32; rows * d];
        for r in 0..rows {
            let row = &xv[r * d..(r + 1) * d];
            let ms = row.iter().map(|&e| e * e).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..d {
                out[r * d + i] = row[i] * inv * wv[i];
            }
        }
        self.insert(out)
    }
    fn silu(&mut self, x: TensorId) -> TensorId {
        let out: Vec<f32> = self
            .get(x)
            .iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        self.insert(out)
    }
    fn softmax(&mut self, x: TensorId, outer: usize, dimlen: usize, inner: usize) -> TensorId {
        let xv = self.get(x);
        let mut out = vec![0.0_f32; xv.len()];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * dimlen * inner + i;
                let mut max = f32::NEG_INFINITY;
                for t in 0..dimlen {
                    max = max.max(xv[base + t * inner]);
                }
                let mut denom = 0.0_f32;
                for t in 0..dimlen {
                    let e = (xv[base + t * inner] - max).exp();
                    out[base + t * inner] = e;
                    denom += e;
                }
                for t in 0..dimlen {
                    out[base + t * inner] /= denom;
                }
            }
        }
        self.insert(out)
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
        let xv = self.get(x).to_vec();
        let theta = theta as f32;
        let mut out = xv.clone();
        for r in 0..rows {
            let pos = (pos_start + (r % seq)) as f32;
            for j in 0..hd / 2 {
                let freq = 1.0 / theta.powf(2.0 * j as f32 / hd as f32);
                let angle = pos * freq;
                let (c, s) = (angle.cos(), angle.sin());
                let (ia, ib) = if interleaved {
                    (2 * j, 2 * j + 1)
                } else {
                    (j, j + hd / 2)
                };
                let (a, b) = (xv[r * hd + ia], xv[r * hd + ib]);
                out[r * hd + ia] = a * c - b * s;
                out[r * hd + ib] = a * s + b * c;
            }
        }
        self.insert(out)
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
        let qv = self.get(q).to_vec();
        let kv = self.get(k).to_vec();
        let vv = self.get(v).to_vec();
        let scale = scale as f32;
        let mut out = vec![0.0_f32; qv.len()];
        for g in 0..bh {
            let base = g * s * d;
            for i in 0..s {
                let jmax = if causal { i + 1 } else { s };
                let mut scores = vec![0.0_f32; jmax];
                let mut maxv = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0_f32;
                    for e in 0..d {
                        dot += qv[base + i * d + e] * kv[base + j * d + e];
                    }
                    *sc = dot * scale;
                    maxv = maxv.max(*sc);
                }
                let mut denom = 0.0_f32;
                for sc in &mut scores {
                    *sc = (*sc - maxv).exp();
                    denom += *sc;
                }
                for e in 0..d {
                    let mut acc = 0.0_f32;
                    for (j, &p) in scores.iter().enumerate() {
                        acc += (p / denom) * vv[base + j * d + e];
                    }
                    out[base + i * d + e] = acc;
                }
            }
        }
        self.insert(out)
    }
    fn transpose(&mut self, x: TensorId, shape_in: &[usize], d0: usize, d1: usize) -> TensorId {
        let out = permute_axes(self.get(x), shape_in, d0, d1);
        self.insert(out)
    }
    fn slice(
        &mut self,
        x: TensorId,
        shape_in: &[usize],
        dim: usize,
        start: usize,
        end: usize,
    ) -> TensorId {
        let out = slice_dim(self.get(x), shape_in, dim, start, end);
        self.insert(out)
    }

    fn adamw_step(
        &mut self,
        master: TensorId,
        grad: TensorId,
        m: TensorId,
        v: TensorId,
        hp: AdamwHp,
    ) {
        let g = self.get(grad).to_vec();
        let mut mv = self.get(m).to_vec();
        let mut vv = self.get(v).to_vec();
        let mut w = self.get(master).to_vec();
        let t = hp.step.max(1) as i32;
        let bc1 = 1.0 - hp.beta1.powi(t);
        let bc2 = 1.0 - hp.beta2.powi(t);
        for i in 0..w.len() {
            let gi = f64::from(g[i]);
            let mi = hp.beta1 * f64::from(mv[i]) + (1.0 - hp.beta1) * gi;
            let vi = hp.beta2 * f64::from(vv[i]) + (1.0 - hp.beta2) * gi * gi;
            mv[i] = mi as f32;
            vv[i] = vi as f32;
            let mut wi = f64::from(w[i]);
            wi -= hp.lr * hp.wd * wi;
            wi -= hp.lr * (mi / bc1) / ((vi / bc2).sqrt() + hp.eps);
            w[i] = wi as f32;
        }
        self.write(m, &mv);
        self.write(v, &vv);
        self.write(master, &w);
    }

    fn det_sum(&mut self, xs: &[TensorId]) -> Result<TensorId, TrapCode> {
        let vecs: Vec<Vec<f32>> = xs.iter().map(|&id| self.get(id).to_vec()).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(Vec::as_slice).collect();
        let out = det_core::det_sum(&refs).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_scale(&mut self, x: TensorId, alpha: f64) -> TensorId {
        let out = det_core::det_scale(self.get(x), alpha);
        self.insert(out)
    }
    fn det_l2norm(&self, x: TensorId) -> f32 {
        det_core::det_l2norm(self.get(x))
    }
    fn det_sign(&mut self, x: TensorId) -> TensorId {
        let out = det_core::det_sign(self.get(x));
        self.insert(out)
    }
    fn det_add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_add(self.get(a), self.get(b)).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_sub(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_sub(self.get(a), self.get(b)).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, TrapCode> {
        let out = det_core::det_mul(self.get(a), self.get(b)).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_absmax_unpack(
        &mut self,
        packed: TensorId,
        chunk: usize,
        bits: u32,
    ) -> Result<TensorId, TrapCode> {
        let bytes: Vec<u8> = self.get(packed).iter().map(|&f| f as u8).collect();
        let out = det_core::det_absmax_unpack(&bytes, chunk, bits).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_chunk_scatter_add(
        &mut self,
        acc: TensorId,
        vals: TensorId,
        idx: TensorId,
        chunk: usize,
    ) -> Result<(), TrapCode> {
        let valsv = self.get(vals).to_vec();
        let idxv = self.u32s(idx);
        let mut accv = self.get(acc).to_vec();
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
        let out = det_core::det_chunk_scatter(self.get(vals), &self.u32s(idx), chunk, out_len)
            .map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn det_axpy(&mut self, y: TensorId, alpha: f64, x: TensorId) -> Result<(), TrapCode> {
        let xv = self.get(x).to_vec();
        let mut yv = self.get(y).to_vec();
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
        let (vals, idx) = det_core::topk_chunk(self.get(x), chunk, k).map_err(det_trap)?;
        let ivals: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
        let vh = self.insert(vals);
        let ih = self.insert(ivals);
        Ok((vh, ih))
    }
    fn absmax_pack(&mut self, x: TensorId, chunk: usize, bits: u32) -> Result<TensorId, TrapCode> {
        let packed = det_core::absmax_pack(self.get(x), chunk, bits).map_err(det_trap)?;
        let vals: Vec<f32> = packed.iter().map(|&b| f32::from(b)).collect();
        Ok(self.insert(vals))
    }
    fn dct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode> {
        let out = det_core::dct2(self.get(x), tile).map_err(det_trap)?;
        Ok(self.insert(out))
    }
    fn idct2(&mut self, x: TensorId, tile: usize) -> Result<TensorId, TrapCode> {
        let out = det_core::idct2(self.get(x), tile).map_err(det_trap)?;
        Ok(self.insert(out))
    }
}

/// Swap axes `d0`/`d1` of a row-major tensor with `shape_in` (shared by transpose forward).
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

/// Copy the `start..end` sub-range along `dim` of a row-major tensor with `shape_in`.
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

fn det_trap(e: det_core::DetError) -> TrapCode {
    match e {
        det_core::DetError::UnsupportedBits { .. } => TrapCode::BadEnum,
        _ => TrapCode::ShapeMismatch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_and_relu_forward() {
        let mut b = CpuBackend::new();
        let a = b.create(vec![1.0, 2.0, 3.0, 4.0]); // [2,2]
        let w = b.create(vec![1.0, 0.0, 0.0, 1.0]); // identity
        let c = b.matmul(a, 2, 2, w, 2);
        assert_eq!(b.view(c), &[1.0, 2.0, 3.0, 4.0]);
        let neg = b.create(vec![-1.0, 2.0, -3.0]);
        let r = b.relu(neg);
        assert_eq!(b.view(r), &[0.0, 2.0, 0.0]);
    }

    #[test]
    fn det_ops_use_det_core() {
        let mut b = CpuBackend::new();
        let x = b.create(vec![1.0, 2.0, 3.0]);
        let y = b.create(vec![10.0, 20.0, 30.0]);
        let s = b.det_sum(&[x, y]).unwrap();
        assert_eq!(b.view(s), &[11.0, 22.0, 33.0]);
        let norm_in = b.create(vec![3.0, 4.0]);
        assert_eq!(b.det_l2norm(norm_in), 5.0);
    }

    #[test]
    fn det_shape_mismatch_traps() {
        let mut b = CpuBackend::new();
        let a = b.create(vec![1.0, 2.0]);
        let c = b.create(vec![1.0]);
        assert_eq!(b.det_add(a, c), Err(TrapCode::ShapeMismatch));
    }

    #[test]
    fn free_recycles_ids() {
        let mut b = CpuBackend::new();
        let x = b.create(vec![1.0]);
        b.free(x);
        let y = b.create(vec![2.0]);
        assert_eq!(x, y, "freed slot is reused");
    }

    #[test]
    fn embedding_gathers_rows() {
        let mut b = CpuBackend::new();
        let w = b.create(vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]); // [3,2]
        let out = b.embedding(w, &[2, 0], 2);
        assert_eq!(b.view(out), &[2.0, 2.1, 0.0, 0.1]);
    }

    #[test]
    fn rmsnorm_normalizes() {
        let mut b = CpuBackend::new();
        let x = b.create(vec![3.0, 4.0]); // 1 row, d=2
        let w = b.create(vec![1.0, 1.0]);
        let out = b.rmsnorm(x, w, 1, 2, 0.0);
        // ms = (9+16)/2 = 12.5; inv = 1/sqrt(12.5); y = x*inv.
        let inv = 1.0 / 12.5_f32.sqrt();
        assert!((b.view(out)[0] - 3.0 * inv).abs() < 1e-6);
        assert!((b.view(out)[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn flash_attn_causal_first_row_is_first_value() {
        let mut b = CpuBackend::new();
        // [b=1,h=1,s=2,d=1]; causal ⇒ row 0 only attends to key 0 ⇒ out[0] = v[0].
        let q = b.create(vec![1.0, 1.0]);
        let k = b.create(vec![1.0, 1.0]);
        let v = b.create(vec![5.0, 9.0]);
        let out = b.flash_attn(q, k, v, 1, 2, 1, true, 1.0);
        assert_eq!(b.view(out)[0], 5.0);
    }

    #[test]
    fn compression_ops_delegate_to_det_core() {
        let mut b = CpuBackend::new();
        let x = b.create(vec![0.1, -0.9, 0.2, 1.0]);
        let (vals, idx) = b.topk_chunk(x, 4, 2).unwrap();
        assert_eq!(b.view(vals), &[1.0, -0.9]);
        assert_eq!(b.view(idx), &[3.0, 1.0]);
        let cst = b.create(vec![2.0; 16]);
        let y = b.dct2(cst, 4).unwrap();
        assert!((b.view(y)[0] - 8.0).abs() < 1e-4);
        let back = b.idct2(y, 4).unwrap();
        assert!((b.view(back)[0] - 2.0).abs() < 1e-4);
    }
}
