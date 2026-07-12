// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Optimization profiles as libraries (architecture §5.3, ABI §10.3).
//!
//! Each profile is first-party, golden-tested guest code an experiment composes and parameterizes
//! through its config — `SparseLoco` (the consumer-uplink flagship), `DiLoCo` (dense/int8 outer
//! Nesterov baseline), and `Demo` (per-step DeMo/DisTrO). They share one shape:
//!
//! - `manifest(&cfg) -> Manifest` — cadence + round modes the profile's math tolerates.
//! - `make_update(&mut self, params) -> UpdateBuilder` — **native lane, local math**: pseudo-gradient,
//!   error feedback / momentum, compression → the round payload.
//! - `ingest(&mut self, params, &UpdatesView)` — **det lane, canonical inputs**: decode → clip →
//!   aggregate (streaming `det_chunk_scatter_add`) → outer step, O(1)-tensor peak memory (§5.9).
//!
//! The profile struct holds its config on both sides, so `make_update` and `ingest` agree on
//! `(chunk, k, bits, tile)` without a wire header — payload sections carry only the packed values
//! and indices (the swarm never parses them, §4.3).

use crate::{
    det_zeros, DetPersistent, Dtype, Manifest, Param, Persistent, UpdateBuilder, UpdatesView,
};
use serde::{Deserialize, Serialize};

// ================================================================================================
// sparse_loco (§5.3.1) — the flagship
// ================================================================================================

/// `sparse_loco` config (ABI §10.3 schema). Chunk/top-k/bit defaults are the paper's real-model
/// values; the tiny reference model overrides `chunk` to a divisor of its parameter sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLocoCfg {
    /// Inner steps per round (H).
    pub h: u32,
    /// Error-feedback decay β (0.95).
    pub ef_decay: f64,
    /// 1-D chunk length (4096 for real models).
    pub chunk: u32,
    /// Top-k retained per chunk (64 → 1/64 density at chunk 4096).
    pub topk: u32,
    /// Value quantization bit width (2-bit, 4-level per-chunk absmax codebook).
    pub bits: u32,
    /// Outer step size α (1.0; optionally lowered late in training).
    pub outer_alpha: f64,
    /// Median-norm clip of contributions before aggregation (hardened default, §12).
    pub clip: bool,
}

impl Default for SparseLocoCfg {
    fn default() -> Self {
        Self {
            h: 30,
            ef_decay: 0.95,
            chunk: 4096,
            topk: 64,
            bits: 2,
            outer_alpha: 1.0,
            clip: true,
        }
    }
}

/// The `sparse_loco` profile: chunked top-k + 2-bit absmax values + error feedback (native lane),
/// det-lane ingest with absmax unpack + median-norm clip + scatter-add + rebase outer step.
pub struct SparseLoco {
    cfg: SparseLocoCfg,
    ef: Vec<Persistent>, // error-feedback residuals, one per param (native local)
}

impl SparseLoco {
    /// Register the error-feedback buffers (one per param). Call from `da_build`.
    #[must_use]
    pub fn new(cfg: SparseLocoCfg, params: &[Param]) -> Self {
        let ef = params
            .iter()
            .enumerate()
            .map(|(i, p)| Persistent::local(&format!("sl.ef{i}"), p.shape(), Dtype::F32))
            .collect();
        Self { cfg, ef }
    }

    /// Cadence: `steps_per_round = h`; barrier + pipelined; any interval.
    #[must_use]
    pub fn manifest(cfg: &SparseLocoCfg) -> Manifest {
        let mut m = Manifest::new("sparse_loco", env!("CARGO_PKG_VERSION"), cfg.h);
        m.round_modes = vec!["barrier".into(), "pipelined".into()];
        m
    }

    /// Δ = θ⁽ᵗ⁾ − θ → acc = β·ef + Δ → top-k chunk → 2-bit pack → push; ef ← acc − scatter(sent).
    pub fn make_update(&mut self, params: &[Param]) -> UpdateBuilder {
        let (chunk, k, bits) = (self.cfg.chunk, self.cfg.topk, self.cfg.bits);
        let mut ub = UpdateBuilder::new();
        for (i, p) in params.iter().enumerate() {
            let delta = p.round_base().sub(p.tensor());
            let acc = self.ef[i].tensor().mul_s(self.cfg.ef_decay).add(&delta);
            let (vals, idx) = acc.topk_chunk(chunk, k);
            let packed = vals.absmax_pack(k, bits); // per-top-k-row codebook
            ub.push_tensor(&packed);
            ub.push_tensor(&idx);
            // ef ← acc − chunk_scatter(dequant(sent))  (residual stays local; param-shaped)
            let sent_vals = packed.absmax_unpack(k, bits, Dtype::F32);
            let sent = sent_vals.chunk_scatter(&idx, chunk, p.shape());
            let ef_new = acc.sub(&sent);
            self.ef[i].assign(&ef_new);
        }
        ub
    }

    /// Streaming det-lane ingest: per param, (optionally median-norm-clip then) scatter-add every
    /// peer's decoded sparse Δ̂ into one accumulator, then rebase + apply the outer step.
    pub fn ingest(&mut self, params: &[Param], u: &UpdatesView) {
        let (chunk, k, bits) = (self.cfg.chunk, self.cfg.topk, self.cfg.bits);
        let count = u.len().max(1);
        for (i, p) in params.iter().enumerate() {
            let numel: u32 = p.shape().iter().product();
            let vsec = (2 * i) as u32;
            let isec = vsec + 1;

            // Pass 1 (clip only): per-peer contribution norm → median clip target.
            let clip_norms: Vec<f64> = if self.cfg.clip {
                (0..u.len())
                    .map(|j| {
                        let vals = u.get(j).tensor(vsec).absmax_unpack(k, bits);
                        vals.l2norm()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let median = median_of(&clip_norms);

            // Pass 2 (streaming): decode → clip-scale → scatter-add into one accumulator.
            let mut acc = det_zeros(&[numel]);
            for j in 0..u.len() {
                let ur = u.get(j);
                let vals = ur.tensor(vsec).absmax_unpack(k, bits);
                let idx = ur.tensor(isec);
                let scaled = if self.cfg.clip {
                    let norm = clip_norms[j as usize];
                    let s = if norm > median && norm > 0.0 {
                        median / norm
                    } else {
                        1.0
                    };
                    vals.scale(s)
                } else {
                    vals.scale(1.0)
                };
                acc.chunk_scatter_add(&scaled, &idx, chunk);
            }
            // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂  (rebase, then apply the canonical aggregate).
            p.det_reset_to_base();
            p.det_axpy(&acc, -self.cfg.outer_alpha / f64::from(count));
        }
    }
}

// ================================================================================================
// diloco (§5.3.2) — dense/int8 outer Nesterov baseline
// ================================================================================================

/// `diloco` config (ABI §10.3 schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiLoCoCfg {
    /// Inner steps per round (H).
    pub h: u32,
    /// Outer SGD learning rate (0.7).
    pub outer_lr: f64,
    /// Outer momentum (0.9).
    pub momentum: f64,
    /// Nesterov momentum (vs plain heavy-ball).
    pub nesterov: bool,
    /// Pseudo-gradient payload quantization: `0` = dense fp32, else bit width (e.g. 8).
    pub quant_bits: u32,
}

impl Default for DiLoCoCfg {
    fn default() -> Self {
        Self {
            h: 100,
            outer_lr: 0.7,
            momentum: 0.9,
            nesterov: true,
            quant_bits: 0,
        }
    }
}

/// The `diloco` profile: dense (or int8) pseudo-gradient + outer SGD with (Nesterov) momentum on a
/// **replicated** det persistent (the canonical example of consensus outer-optimizer state, §5.1).
pub struct DiLoCo {
    cfg: DiLoCoCfg,
    mom: Vec<DetPersistent>, // outer momentum, one per param (det, replicated)
}

impl DiLoCo {
    /// Register the replicated outer-momentum buffers. Call from `da_build`.
    #[must_use]
    pub fn new(cfg: DiLoCoCfg, params: &[Param]) -> Self {
        let mom = params
            .iter()
            .enumerate()
            .map(|(i, p)| DetPersistent::replicated(&format!("dl.mom{i}"), p.shape()))
            .collect();
        Self { cfg, mom }
    }

    /// Cadence: `steps_per_round = h`; barrier only (bandwidth-heavy).
    #[must_use]
    pub fn manifest(cfg: &DiLoCoCfg) -> Manifest {
        Manifest::new("diloco", env!("CARGO_PKG_VERSION"), cfg.h)
    }

    /// Δ = θ⁽ᵗ⁾ − θ, pushed dense (or int8-packed when `quant_bits != 0`).
    pub fn make_update(&mut self, params: &[Param]) -> UpdateBuilder {
        let mut ub = UpdateBuilder::new();
        for p in params {
            let numel: u32 = p.shape().iter().product();
            let delta = p.round_base().sub(p.tensor());
            if self.cfg.quant_bits == 0 {
                ub.push_tensor(&delta);
            } else {
                // int8 pseudo-grad codec: one absmax chunk over the whole tensor.
                let packed = delta.absmax_pack(numel, self.cfg.quant_bits);
                ub.push_tensor(&packed);
            }
        }
        ub
    }

    /// Aggregate the pseudo-gradient, advance the replicated momentum, and apply the outer
    /// (Nesterov) SGD step by rebasing to θ⁽ᵗ⁾ and subtracting `outer_lr · step`.
    pub fn ingest(&mut self, params: &[Param], u: &UpdatesView) {
        let count = u.len().max(1);
        for (i, p) in params.iter().enumerate() {
            let numel: u32 = p.shape().iter().product();
            // g = (1/R)·Σ Δ  (dense fp32 sum in record order).
            let mut acc = det_zeros(&[numel]);
            for j in 0..u.len() {
                let d = if self.cfg.quant_bits == 0 {
                    u.get(j).tensor(i as u32)
                } else {
                    u.get(j)
                        .tensor(i as u32)
                        .absmax_unpack(numel, self.cfg.quant_bits)
                };
                acc = acc.add(&d);
            }
            let g = acc.scale(1.0 / f64::from(count));
            // m ← momentum·m + g   (replicated momentum).
            let m_new = self.mom[i].tensor().scale(self.cfg.momentum).add(&g);
            self.mom[i].assign(&m_new);
            // step = nesterov ? g + momentum·m : m
            let step = if self.cfg.nesterov {
                g.add(&m_new.scale(self.cfg.momentum))
            } else {
                m_new
            };
            // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − outer_lr·step
            p.det_reset_to_base();
            p.det_axpy(&step, -self.cfg.outer_lr);
        }
    }
}

// ================================================================================================
// demo (§5.3.3) — per-step DeMo / DisTrO
// ================================================================================================

/// `demo` config (ABI §10.3 schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoCfg {
    /// Fast-momentum decay β (0.999).
    pub momentum_decay: f64,
    /// 2-D DCT tile size (`chunk = tile²`).
    pub tile: u32,
    /// Top-k DCT coefficients per tile (8..16).
    pub topk: u32,
    /// Sign-SGD learning rate at ingest.
    pub sign_lr: f64,
    /// Decoupled weight decay (0.1).
    pub wd: f64,
    /// Partial-subtraction factor α (0.2) removed from local momentum for what was sent.
    pub alpha: f64,
}

impl Default for DemoCfg {
    fn default() -> Self {
        Self {
            momentum_decay: 0.999,
            tile: 8,
            topk: 8,
            sign_lr: 0.01,
            wd: 0.1,
            alpha: 0.2,
        }
    }
}

/// The `demo` profile: per-step DCT energy extraction + top-k coefficients (native), det-lane ingest
/// sums coefficients, inverse-DCTs, and applies the **sign** of the aggregate (Signum-style).
pub struct Demo {
    cfg: DemoCfg,
    mom: Vec<Persistent>, // fast momentum, one per param (native local)
}

impl Demo {
    /// Register the momentum buffers. Call from `da_build`.
    #[must_use]
    pub fn new(cfg: DemoCfg, params: &[Param]) -> Self {
        let mom = params
            .iter()
            .enumerate()
            .map(|(i, p)| Persistent::local(&format!("demo.m{i}"), p.shape(), Dtype::F32))
            .collect();
        Self { cfg, mom }
    }

    /// Cadence: per step (H = 1), pipelined or a seconds-scale coordinator (§5.3.3).
    #[must_use]
    pub fn manifest(_cfg: &DemoCfg) -> Manifest {
        let mut m = Manifest::new("demo", env!("CARGO_PKG_VERSION"), 1);
        m.round_modes = vec!["barrier".into(), "pipelined".into()];
        m.min_round_interval_ms = 1000;
        m
    }

    /// M ← β·M + Δ; extract top-k DCT coefficients per tile; transmit them; M ← M − α·IDCT(sent).
    pub fn make_update(&mut self, params: &[Param]) -> UpdateBuilder {
        let (tile, k) = (self.cfg.tile, self.cfg.topk);
        let block = tile * tile;
        let mut ub = UpdateBuilder::new();
        for (i, p) in params.iter().enumerate() {
            let delta = p.round_base().sub(p.tensor());
            let m = self.mom[i]
                .tensor()
                .mul_s(self.cfg.momentum_decay)
                .add(&delta);
            let coeffs = m.dct2(tile);
            let (vals, idx) = coeffs.topk_chunk(block, k);
            ub.push_tensor(&vals);
            ub.push_tensor(&idx);
            // M ← M − α·IDCT(scatter(sent))  (param-shaped so the residual subtract type-checks)
            let sent_spatial = vals.chunk_scatter(&idx, block, p.shape()).idct2(tile);
            let m_new = m.sub(&sent_spatial.mul_s(self.cfg.alpha));
            self.mom[i].assign(&m_new);
        }
        ub
    }

    /// Sum sparse coefficients across peers, inverse-DCT, and apply `−sign_lr · sign(aggregate)`
    /// plus decoupled weight decay — all det lane (canonical inputs).
    pub fn ingest(&mut self, params: &[Param], u: &UpdatesView) {
        let (tile, block) = (self.cfg.tile, self.cfg.tile * self.cfg.tile);
        for (i, p) in params.iter().enumerate() {
            let numel: u32 = p.shape().iter().product();
            let vsec = (2 * i) as u32;
            let isec = vsec + 1;
            let mut coeff_acc = det_zeros(&[numel]);
            for j in 0..u.len() {
                let ur = u.get(j);
                let vals = ur.tensor(vsec);
                let idx = ur.tensor(isec);
                coeff_acc.chunk_scatter_add(&vals, &idx, block);
            }
            let spatial = coeff_acc.idct2(tile);
            let sign = spatial.sign();
            // θ ← θ⁽ᵗ⁾ − lr·wd·θ⁽ᵗ⁾ − lr·sign(aggregate)   (decoupled decay + sign-SGD)
            p.det_reset_to_base();
            let base = p.det_base();
            p.det_axpy(&base, -self.cfg.sign_lr * self.cfg.wd);
            p.det_axpy(&sign, -self.cfg.sign_lr);
        }
    }
}

/// The median of a list of contribution norms (guest `f64` math over det-lane `det_l2norm` results;
/// deterministic ⇒ safe on the agree-path, §7). Empty ⇒ `+∞` (no clip).
fn median_of(norms: &[f64]) -> f64 {
    if norms.is_empty() {
        return f64::INFINITY;
    }
    let mut v = norms.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
