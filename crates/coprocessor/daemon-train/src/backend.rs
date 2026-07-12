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
}
