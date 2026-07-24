// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The C3 re-authored tiny-LLaMA decoder — **ordinary Burn**, generic over the autodiff backend
//! (architecture §6 "Model code" row: `Autodiff<HostBackend>`, `burn::nn`-free by choice,
//! author-owned).
//!
//! **This file is compiled twice** (the det-crate dual-compile pattern): by the guest crate for
//! wasm32 over `Autodiff<HostBackend>`, and by the host parity test (`#[path]` include) natively
//! over `Autodiff<NdArray>` — the C3b lowering gate asserts the two runs are **bit-exact**, which
//! is only meaningful because both sides execute this exact source. Keep it self-contained over
//! `burn` + `serde` (no SDK imports).
//!
//! ## Determinism rules (why this file looks the way it does)
//!
//! - **No readbacks**: everything device-side; gradients stay `InnerBackend` tensors; parameter
//!   updates re-materialize leaves with `Tensor::from_inner` (over `HostBackend` a readback would
//!   panic by design — sdk-compute docs).
//! - **No host-libm transcendentals in guest code**: RoPE tables ride device ops
//!   (`exp(e·ln θ)`, `cos`, `sin`), never `f32::cos` (wasm32 libm ≠ native glibc in the last
//!   ulp); the AdamW bias correction uses [`pow_iter`] (a pure-IEEE f64 multiply loop), never
//!   `f64::powf`.
//! - The architecture mirrors the v1 `models::TinyLlama` (embedding → N×(rmsnorm → RoPE
//!   attention → rmsnorm → SwiGLU) → tied logits, shifted-max cross-entropy, AdamW inner) with
//!   the op-definition grounding of `tests/reference/mod.rs` — the tolerance-class comparison
//!   surface of the C3c parity lane.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{activation, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// The model dimensions + inner-AdamW hyperparameters (a flat, serde view of the v1
/// `TinyLlamaCfg` fields this model consumes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCfg {
    /// Residual width.
    pub d_model: u32,
    /// Transformer blocks.
    pub n_layers: u32,
    /// Attention heads (`n_kv_heads == n_heads`; GQA-repeat is future, as in v1).
    pub n_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// Vocabulary (tied input/output embedding).
    pub vocab: u32,
    /// Sequence length (predicts positions `1..seq` from `0..seq-1`).
    pub seq_len: u32,
    /// SwiGLU hidden = `ffn_mult · d_model`.
    pub ffn_mult: u32,
    /// RoPE base.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub rmsnorm_eps: f64,
    /// AdamW learning rate.
    pub lr: f64,
    /// AdamW β₁.
    pub beta1: f64,
    /// AdamW β₂.
    pub beta2: f64,
    /// AdamW ε.
    pub adam_eps: f64,
    /// AdamW decoupled weight decay.
    pub wd: f64,
}

impl ModelCfg {
    /// The canonical parameter element counts, in registration order (`tok`, per-block
    /// `attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down`, `norm`) — exactly the v1
    /// `canonical_param_layout` order the checkpoint/digest state rides.
    #[must_use]
    pub fn param_numels(&self) -> Vec<usize> {
        let d = self.d_model as usize;
        let qdim = (self.n_heads * self.head_dim) as usize;
        let hidden = (self.ffn_mult * self.d_model) as usize;
        let vocab = self.vocab as usize;
        let mut out = vec![vocab * d];
        for _ in 0..self.n_layers {
            out.extend([
                d,
                d * qdim,
                d * qdim,
                d * qdim,
                qdim * d,
                d,
                d * hidden,
                d * hidden,
                hidden * d,
            ]);
        }
        out.push(d);
        out
    }
}

/// `base^t` by binary-free sequential multiplication — pure IEEE f64 multiplies in a fixed
/// order, bit-identical on wasm32 and native (unlike `f64::powf`, which is libm).
#[must_use]
pub fn pow_iter(base: f64, t: u32) -> f64 {
    let mut acc = 1.0f64;
    for _ in 0..t {
        acc *= base;
    }
    acc
}

/// One block's parameters (flat rank-1 leaves; the forward reshapes to natural ranks).
struct Block<B: AutodiffBackend> {
    attn_norm: Tensor<B, 1>,
    wq: Tensor<B, 1>,
    wk: Tensor<B, 1>,
    wv: Tensor<B, 1>,
    wo: Tensor<B, 1>,
    ffn_norm: Tensor<B, 1>,
    w_gate: Tensor<B, 1>,
    w_up: Tensor<B, 1>,
    w_down: Tensor<B, 1>,
}

/// The decoder: flat AD leaves + `InnerBackend` AdamW moments, all device-resident.
pub struct TinyLlamaModel<B: AutodiffBackend> {
    cfg: ModelCfg,
    device: B::Device,
    tok: Tensor<B, 1>,
    blocks: Vec<Block<B>>,
    norm: Tensor<B, 1>,
    /// AdamW moments (plain device tensors, canonical order), zero-init like v1's local
    /// persistents; they survive ingest (v1 keeps them across the outer step).
    m: Vec<Tensor<B::InnerBackend, 1>>,
    v: Vec<Tensor<B::InnerBackend, 1>>,
}

fn leaf<B: AutodiffBackend>(device: &B::Device, data: &[f32]) -> Tensor<B, 1> {
    let n = data.len();
    Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [n]), device).require_grad()
}

fn zeros_inner<B: AutodiffBackend>(device: &B::Device, n: usize) -> Tensor<B::InnerBackend, 1> {
    Tensor::from_data(TensorData::new(vec![0.0f32; n], [n]), device)
}

impl<B: AutodiffBackend> TinyLlamaModel<B> {
    /// Build the model with every parameter and both AdamW moments allocated **on device** as
    /// zeros — `Tensor::zeros` is a device op, so not one element crosses the boundary and the
    /// guest holds no parameter-shaped buffer at all.
    ///
    /// This is the constructor every init form goes through ([`TinyLlamaModel::write_param_window`]
    /// then writes the real values in bounded windows): a guest-side flat state is O(family) —
    /// ~2.93 GiB per copy at the fleet-ceremony geometry — which no wasm32 linear memory holds.
    /// [`TinyLlamaModel::from_flat`] stays for the natively-driven lanes that already own a
    /// resident state (the dual-compiled oracle + the toy harnesses).
    #[must_use]
    pub fn zeros(cfg: ModelCfg, device: B::Device) -> Self {
        let numels = cfg.param_numels();
        let mut it = numels.iter();
        let leaf = |n: usize| Tensor::<B, 1>::zeros([n], &device).require_grad();
        let tok = leaf(*it.next().expect("tok"));
        let blocks = (0..cfg.n_layers)
            .map(|_| Block {
                attn_norm: leaf(*it.next().expect("attn_norm")),
                wq: leaf(*it.next().expect("wq")),
                wk: leaf(*it.next().expect("wk")),
                wv: leaf(*it.next().expect("wv")),
                wo: leaf(*it.next().expect("wo")),
                ffn_norm: leaf(*it.next().expect("ffn_norm")),
                w_gate: leaf(*it.next().expect("w_gate")),
                w_up: leaf(*it.next().expect("w_up")),
                w_down: leaf(*it.next().expect("w_down")),
            })
            .collect();
        let norm = leaf(*it.next().expect("norm"));
        let moments = || {
            numels
                .iter()
                .map(|&n| Tensor::<B::InnerBackend, 1>::zeros([n], &device))
                .collect()
        };
        let (m, v) = (moments(), moments());
        Self {
            cfg,
            device,
            tok,
            blocks,
            norm,
            m,
            v,
        }
    }

    /// Write one bounded `[off, off + vals.len())` element window into the parameter at `index`
    /// (canonical registration order) — the device-side counterpart of the state plane's window
    /// streaming: an init walks a family window-by-window and lands each window here, so the peak
    /// guest buffer is one window rather than one parameter (let alone one family).
    ///
    /// The parameter is re-materialized as a fresh AD leaf (`from_inner(...).require_grad()`), the
    /// same discipline as every other parameter write in this file. Init-time only: a window write
    /// during a live autodiff graph would detach the leaf the tape recorded.
    ///
    /// # Panics
    /// If `index` is not a canonical parameter position.
    pub fn write_param_window(&mut self, index: usize, off: usize, vals: &[f32]) {
        let len = vals.len();
        let patch = Tensor::<B::InnerBackend, 1>::from_data(
            TensorData::new(vals.to_vec(), [len]),
            &self.device,
        );
        let param = self.param_mut(index);
        let written = param.clone().inner().slice_assign([off..off + len], patch);
        *param = Tensor::from_inner(written).require_grad();
    }

    /// The parameter at canonical registration position `index` (`tok`, then per block
    /// `attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down`, then `norm`).
    fn param_mut(&mut self, index: usize) -> &mut Tensor<B, 1> {
        const PER_BLOCK: usize = 9;
        if index == 0 {
            return &mut self.tok;
        }
        let within = index - 1;
        let block = within / PER_BLOCK;
        if block >= self.blocks.len() {
            assert_eq!(
                within,
                self.blocks.len() * PER_BLOCK,
                "canonical param index"
            );
            return &mut self.norm;
        }
        let b = &mut self.blocks[block];
        match within % PER_BLOCK {
            0 => &mut b.attn_norm,
            1 => &mut b.wq,
            2 => &mut b.wk,
            3 => &mut b.wv,
            4 => &mut b.wo,
            5 => &mut b.ffn_norm,
            6 => &mut b.w_gate,
            7 => &mut b.w_up,
            _ => &mut b.w_down,
        }
    }

    /// Build from a RESIDENT canonical flat state (registration order — matched init). The guest
    /// never has one (it streams init windows into [`TinyLlamaModel::zeros`]); the natively
    /// compiled oracle lanes, which own their init image outright, do.
    ///
    /// # Panics
    /// If `flat` does not match [`ModelCfg::param_numels`].
    // Dead in the wasm32 guest compilation, live in the native dual-compile (this file is compiled
    // twice — see the module docs); the guest's own init path is `zeros` + `write_param_window`.
    #[allow(dead_code)]
    #[must_use]
    pub fn from_flat(cfg: ModelCfg, device: B::Device, flat: &[Vec<f32>]) -> Self {
        let numels = cfg.param_numels();
        assert_eq!(flat.len(), numels.len(), "init param count");
        for (f, n) in flat.iter().zip(numels.iter()) {
            assert_eq!(f.len(), *n, "init param numel");
        }
        let mut it = flat.iter();
        let tok = leaf::<B>(&device, it.next().expect("tok"));
        let blocks = (0..cfg.n_layers)
            .map(|_| Block {
                attn_norm: leaf::<B>(&device, it.next().expect("attn_norm")),
                wq: leaf::<B>(&device, it.next().expect("wq")),
                wk: leaf::<B>(&device, it.next().expect("wk")),
                wv: leaf::<B>(&device, it.next().expect("wv")),
                wo: leaf::<B>(&device, it.next().expect("wo")),
                ffn_norm: leaf::<B>(&device, it.next().expect("ffn_norm")),
                w_gate: leaf::<B>(&device, it.next().expect("w_gate")),
                w_up: leaf::<B>(&device, it.next().expect("w_up")),
                w_down: leaf::<B>(&device, it.next().expect("w_down")),
            })
            .collect();
        let norm = leaf::<B>(&device, it.next().expect("norm"));
        let m = numels
            .iter()
            .map(|&n| zeros_inner::<B>(&device, n))
            .collect();
        let v = numels
            .iter()
            .map(|&n| zeros_inner::<B>(&device, n))
            .collect();
        Self {
            cfg,
            device,
            tok,
            blocks,
            norm,
            m,
            v,
        }
    }

    /// The AD-leaf params in canonical registration order.
    fn flat_params(&self) -> Vec<Tensor<B, 1>> {
        let mut ps = vec![self.tok.clone()];
        for b in &self.blocks {
            ps.extend([
                b.attn_norm.clone(),
                b.wq.clone(),
                b.wk.clone(),
                b.wv.clone(),
                b.wo.clone(),
                b.ffn_norm.clone(),
                b.w_gate.clone(),
                b.w_up.clone(),
                b.w_down.clone(),
            ]);
        }
        ps.push(self.norm.clone());
        ps
    }

    /// Detached (inner) clones of the params, canonical order — the export surface (over
    /// `HostBackend` each is consumed by `compute@2::export`; natively by `into_data`).
    #[must_use]
    pub fn export_tensors(&self) -> Vec<Tensor<B::InnerBackend, 1>> {
        self.flat_params().into_iter().map(Tensor::inner).collect()
    }

    /// The AdamW moment tensors (all of `m`, then all of `v`, canonical order) — replica-local
    /// optimizer state the checkpoint walk exports (§10.2): the moments survive ingest, so a
    /// restore without them forks the next round's training trajectory.
    #[must_use]
    pub fn moment_tensors(&self) -> Vec<Tensor<B::InnerBackend, 1>> {
        self.m
            .iter()
            .cloned()
            .chain(self.v.iter().cloned())
            .collect()
    }

    /// Replace the AdamW moments from canonical per-param vectors (the restore counterpart of
    /// [`TinyLlamaModel::moment_tensors`]).
    ///
    /// # Panics
    /// If either moment set does not match [`ModelCfg::param_numels`].
    pub fn set_moments_from_flat(&mut self, m: &[Vec<f32>], v: &[Vec<f32>]) {
        let numels = self.cfg.param_numels();
        assert_eq!(m.len(), numels.len(), "m param count");
        assert_eq!(v.len(), numels.len(), "v param count");
        let upload = |d: &Vec<f32>| {
            Tensor::<B::InnerBackend, 1>::from_data(
                TensorData::new(d.clone(), [d.len()]),
                &self.device,
            )
        };
        self.m = m.iter().map(upload).collect();
        self.v = v.iter().map(upload).collect();
    }

    /// Replace the params with a new canonical flat state (post-ingest master upload). The AdamW
    /// moments are kept — the v1 outer step never resets them.
    pub fn set_params_from_flat(&mut self, flat: &[Vec<f32>]) {
        let mut it = flat.iter();
        self.tok = leaf::<B>(&self.device, it.next().expect("tok"));
        for b in &mut self.blocks {
            b.attn_norm = leaf::<B>(&self.device, it.next().expect("attn_norm"));
            b.wq = leaf::<B>(&self.device, it.next().expect("wq"));
            b.wk = leaf::<B>(&self.device, it.next().expect("wk"));
            b.wv = leaf::<B>(&self.device, it.next().expect("wv"));
            b.wo = leaf::<B>(&self.device, it.next().expect("wo"));
            b.ffn_norm = leaf::<B>(&self.device, it.next().expect("ffn_norm"));
            b.w_gate = leaf::<B>(&self.device, it.next().expect("w_gate"));
            b.w_up = leaf::<B>(&self.device, it.next().expect("w_up"));
            b.w_down = leaf::<B>(&self.device, it.next().expect("w_down"));
        }
        self.norm = leaf::<B>(&self.device, it.next().expect("norm"));
    }

    fn rmsnorm(&self, x: Tensor<B, 2>, w: &Tensor<B, 1>, d: usize) -> Tensor<B, 2> {
        let w2 = w.clone().reshape([1, d]);
        let ms = x.clone().powf_scalar(2.0).mean_dim(1); // [rows, 1]
        let inv = ms.add_scalar(self.cfg.rmsnorm_eps).sqrt().recip();
        x.mul(inv).mul(w2)
    }

    /// RoPE (half-split) on `[b, nh, s, hd]`. The cos/sin tables are computed **on-device**
    /// (`freq = exp(e · ln θ)`, `angle = pos · freq`, then device `cos`/`sin`) from
    /// exactly-representable f32 inputs, so the table bytes are a function of the backend's
    /// kernels — identical for the wasm32 guest and the native oracle (both ndarray).
    fn rope(&self, x: Tensor<B, 4>, s: usize) -> Tensor<B, 4> {
        let hd = self.cfg.head_dim as usize;
        let half = hd / 2;
        // e_j = −2j/hd: hd is a power of two in every config this model accepts, and j < hd, so
        // each value is exactly representable.
        #[allow(clippy::cast_precision_loss)]
        let exps: Vec<f32> = (0..half).map(|j| -2.0 * j as f32 / hd as f32).collect();
        #[allow(clippy::cast_precision_loss)]
        let pos: Vec<f32> = (0..s).map(|p| p as f32).collect();
        let e = Tensor::<B, 2>::from_data(TensorData::new(exps, [1, half]), &self.device);
        let p = Tensor::<B, 2>::from_data(TensorData::new(pos, [s, 1]), &self.device);
        #[allow(clippy::cast_possible_truncation)]
        let theta = Tensor::<B, 2>::from_data(
            TensorData::new(vec![self.cfg.rope_theta as f32], [1, 1]),
            &self.device,
        );
        // freq[1, half] = exp(e · ln θ); angles[s, half] = pos ⊗ freq (broadcast mul).
        let freq = e.mul(theta.log()).exp();
        let angles = p.mul(freq);
        let cos = angles.clone().cos().reshape([1, 1, s, half]);
        let sin = angles.sin().reshape([1, 1, s, half]);
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);
        let out1 = x1.clone().mul(cos.clone()).sub(x2.clone().mul(sin.clone()));
        let out2 = x1.mul(sin).add(x2.mul(cos));
        Tensor::cat(vec![out1, out2], 3)
    }

    /// One forward pass → the scaled mean cross-entropy loss tensor (for `backward`).
    fn forward(&self, tokens: &[u32], b: usize, seq: usize, loss_scale: f64) -> Tensor<B, 1> {
        let cfg = &self.cfg;
        let s = seq - 1;
        let d = cfg.d_model as usize;
        let nh = cfg.n_heads as usize;
        let hd = cfg.head_dim as usize;
        let qdim = nh * hd;
        let hidden = (cfg.ffn_mult * cfg.d_model) as usize;
        let vocab = cfg.vocab as usize;
        let rows = b * s;
        let scale = 1.0 / f64::from(cfg.head_dim).sqrt();

        // inp = tokens[:, 0..s]; tgt = tokens[:, 1..seq].
        let mut inp = Vec::with_capacity(rows);
        let mut tgt = Vec::with_capacity(rows);
        for bi in 0..b {
            for si in 0..s {
                inp.push(i64::from(tokens[bi * seq + si]));
                tgt.push(i64::from(tokens[bi * seq + si + 1]));
            }
        }

        // Embedding: rows of tok.[vocab, d] selected by inp → [rows, d].
        let tok2 = self.tok.clone().reshape([vocab, d]);
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(inp, [rows]), &self.device);
        let mut h = tok2.select(0, idx);

        for blk in &self.blocks {
            // Attention.
            let normed = self.rmsnorm(h.clone(), &blk.attn_norm, d);
            let mk = |w: &Tensor<B, 1>| -> Tensor<B, 4> {
                normed
                    .clone()
                    .matmul(w.clone().reshape([d, qdim]))
                    .reshape([b, s, nh, hd])
                    .swap_dims(1, 2) // [b, nh, s, hd]
            };
            let q = self.rope(mk(&blk.wq), s);
            let k = self.rope(mk(&blk.wk), s);
            let v = mk(&blk.wv);
            // Dense causal attention over [bh, s, hd].
            let bh = b * nh;
            let q3 = q.reshape([bh, s, hd]);
            let k3 = k.reshape([bh, s, hd]);
            let v3 = v.reshape([bh, s, hd]);
            #[allow(clippy::cast_possible_truncation)]
            let scores = q3.matmul(k3.swap_dims(1, 2)).mul_scalar(scale as f32);
            let mut mask = vec![0.0f32; s * s];
            for i in 0..s {
                for j in (i + 1)..s {
                    mask[i * s + j] = -1.0e30;
                }
            }
            let mask = Tensor::<B, 3>::from_data(TensorData::new(mask, [1, s, s]), &self.device);
            let probs = activation::softmax(scores.add(mask), 2);
            let attn = probs
                .matmul(v3) // [bh, s, hd]
                .reshape([b, nh, s, hd])
                .swap_dims(1, 2) // [b, s, nh, hd]
                .reshape([rows, qdim])
                .matmul(blk.wo.clone().reshape([qdim, d])); // [rows, d]
            h = h.add(attn);

            // SwiGLU FFN.
            let normed2 = self.rmsnorm(h.clone(), &blk.ffn_norm, d);
            let gate = activation::silu(
                normed2
                    .clone()
                    .matmul(blk.w_gate.clone().reshape([d, hidden])),
            );
            let up = normed2.matmul(blk.w_up.clone().reshape([d, hidden]));
            let ffn = gate.mul(up).matmul(blk.w_down.clone().reshape([hidden, d]));
            h = h.add(ffn);
        }

        let h = self.rmsnorm(h, &self.norm, d);
        // Tied embedding: logits = h · tokᵀ → [rows, vocab].
        let logits = h.matmul(self.tok.clone().reshape([vocab, d]).swap_dims(0, 1));

        // Shifted-max cross-entropy over all rows (no ignored targets in this harness).
        let max = logits.clone().max_dim(1).detach(); // [rows, 1]
        let shifted = logits.sub(max);
        let logsm = shifted.clone().sub(shifted.exp().sum_dim(1).log());
        let mut onehot = vec![0.0f32; rows * vocab];
        for (i, &t) in tgt.iter().enumerate() {
            #[allow(clippy::cast_sign_loss)]
            {
                onehot[i * vocab + t as usize] = 1.0;
            }
        }
        let oh = Tensor::<B, 2>::from_data(TensorData::new(onehot, [rows, vocab]), &self.device);
        #[allow(clippy::cast_precision_loss)]
        let denom = rows.max(1) as f32;
        #[allow(clippy::cast_possible_truncation)]
        let s_f32 = loss_scale as f32;
        logsm
            .mul(oh)
            .sum()
            .mul_scalar(-1.0 / denom)
            .mul_scalar(s_f32)
            .reshape([1])
    }

    /// Forward + backward: the (scale-weighted) per-param gradients, canonical order, as inner
    /// device tensors — pure enqueueing, zero readbacks (decisions D8).
    #[must_use]
    pub fn forward_backward(
        &self,
        tokens: &[u32],
        b: usize,
        seq: usize,
        loss_scale: f64,
    ) -> Vec<Tensor<B::InnerBackend, 1>> {
        let loss = self.forward(tokens, b, seq, loss_scale);
        let grads = loss.backward();
        self.flat_params()
            .iter()
            .map(|p| p.grad(&grads).expect("param participates in the loss"))
            .collect()
    }

    /// The fused-AdamW inner update from accumulated gradients at `inner_step` (0-based; the
    /// bias-correction step is `inner_step + 1`, round-local like v1). Everything stays
    /// device-side; the params re-materialize as fresh leaves.
    pub fn adamw_apply(&mut self, grads: &[Tensor<B::InnerBackend, 1>], inner_step: u32) {
        let t = inner_step + 1;
        #[allow(clippy::cast_possible_truncation)]
        let bc1 = (1.0 - pow_iter(self.cfg.beta1, t)) as f32;
        #[allow(clippy::cast_possible_truncation)]
        let bc2 = (1.0 - pow_iter(self.cfg.beta2, t)) as f32;
        #[allow(clippy::cast_possible_truncation)]
        let (b1, b2) = (self.cfg.beta1 as f32, self.cfg.beta2 as f32);
        #[allow(clippy::cast_possible_truncation)]
        let (lr, wd, eps) = (
            self.cfg.lr as f32,
            self.cfg.wd as f32,
            self.cfg.adam_eps as f32,
        );

        let params = self.flat_params();
        assert_eq!(grads.len(), params.len(), "grad count");
        let mut new_params = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let g = grads[i].clone();
            let m1 = self.m[i]
                .clone()
                .mul_scalar(b1)
                .add(g.clone().mul_scalar(1.0 - b1));
            let v1 = self.v[i]
                .clone()
                .mul_scalar(b2)
                .add(g.clone().mul(g).mul_scalar(1.0 - b2));
            let mhat = m1.clone().div_scalar(bc1);
            let vhat = v1.clone().div_scalar(bc2);
            let denom = vhat.sqrt().add_scalar(eps);
            let w1 = p
                .clone()
                .inner()
                .mul_scalar(1.0 - lr * wd)
                .sub(mhat.div(denom).mul_scalar(lr));
            self.m[i] = m1;
            self.v[i] = v1;
            new_params.push(Tensor::<B, 1>::from_inner(w1).require_grad());
        }
        let mut it = new_params.into_iter();
        self.tok = it.next().expect("tok");
        for blk in &mut self.blocks {
            blk.attn_norm = it.next().expect("attn_norm");
            blk.wq = it.next().expect("wq");
            blk.wk = it.next().expect("wk");
            blk.wv = it.next().expect("wv");
            blk.wo = it.next().expect("wo");
            blk.ffn_norm = it.next().expect("ffn_norm");
            blk.w_gate = it.next().expect("w_gate");
            blk.w_up = it.next().expect("w_up");
            blk.w_down = it.next().expect("w_down");
        }
        self.norm = it.next().expect("norm");
    }
}
