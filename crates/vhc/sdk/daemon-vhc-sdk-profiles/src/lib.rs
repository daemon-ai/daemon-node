// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-sdk-profiles` — the comm/optimizer profiles over Burn tensors + det math
//! (Phase C, track C3; architecture §6 "Profiles" row, refactor §7 "models leave the SDK").
//!
//! The Phase-C re-expression of the v1 SDK profiles (`daemon-vhc-sdk::profiles`), shed of every
//! tabi handle: [`SparseLoco`] (the consumer-uplink flagship), [`DiLoCo`] (dense/int8 outer
//! Nesterov baseline), and [`Demo`] (per-step DeMo/DisTrO). **The math is the v1 math, verbatim**
//! — pinned bit-for-bit by this crate's golden suite against the current SDK implementation
//! (`tests/goldens.rs`; oracle provenance documented there).
//!
//! ## The two lanes (architecture §3.2/§3.6)
//!
//! - **Det lane (`ingest`)** — consensus math peers must agree on byte-for-byte. Every kernel
//!   delegates to the normative dual-compiled [`daemon_vhc_det`] crate, which compiles for wasm32
//!   with **zero host support**: a `compute@2` guest linking this crate reproduces the v1 host
//!   det ops bit-exactly (the C2 conformance gate pins host ≡ crate). Inputs are canonical f32
//!   slices — the round-base masters and the record-ordered committed payloads.
//! - **Native lane (`make_update`)** — local math over the *trained* parameters (Δ, error
//!   feedback, momentum) feeding the compression kernels. Tolerance-class per architecture §3.6;
//!   the profile state (`ef`/`mom`) is guest-side f32.
//!
//! ## The Burn doorway ([`burn_lane`])
//!
//! Model authors write ordinary Burn (`Autodiff<HostBackend>` in guests — the params live
//! device-side by handle). The profile consumes **materialized f32 slices**: over `HostBackend`,
//! extraction is the explicit budgeted `fence → export → Completion(BufferHandle) → read` walk
//! (architecture §3.2/§3.4 — there is no synchronous in-guest readback), driven by the module's
//! event loop; natively it is `into_data()`. [`burn_lane`] holds the generic model↔profile
//! composition helpers both bindings share.
//!
//! ## The payload wire
//!
//! [`Section`] is CBOR-identical to the v1 update-container wire (the host's `SectionWire`:
//! externally-tagged `Bytes`/`Tensor{data, shape}`), so a payload sealed by this crate decodes
//! under a v1 ingest and vice versa — the profile owns both ends, and the swarm never parses
//! payloads (opaque by invariant).

use serde::{Deserialize, Serialize};

// ================================================================================================
// payload wire
// ================================================================================================

/// One update-container section — CBOR-identical to the v1 host `SectionWire` (same variant and
/// field names, so the externally-tagged encodings are byte-equal).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Section {
    /// A profile-defined opaque byte section.
    Bytes(Vec<u8>),
    /// A tensor section: row-major fp32 data (packed `U8` payloads ride as one byte per f32
    /// element, exactly the v1 container convention).
    Tensor {
        /// Row-major fp32 data.
        data: Vec<f32>,
        /// The tensor shape.
        shape: Vec<u32>,
    },
}

/// Encode a payload (one round's sections) to the canonical container bytes.
///
/// # Panics
/// Never in practice: `Section` is always CBOR-serializable.
#[must_use]
pub fn encode_payload(sections: &[Section]) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(&sections.to_vec(), &mut buf).expect("sections encode");
    buf
}

/// Decode a committed payload's container bytes.
///
/// # Errors
/// A `String` describing the CBOR decode failure (a malformed payload — module policy decides;
/// the profiles refuse by propagating).
pub fn decode_payload(bytes: &[u8]) -> Result<Vec<Section>, String> {
    ciborium::from_reader(bytes).map_err(|e| format!("payload decode: {e}"))
}

/// A tensor section's f32 data, or an error naming the section.
fn tensor_data(sections: &[Section], idx: usize) -> Result<&[f32], String> {
    match sections.get(idx) {
        Some(Section::Tensor { data, .. }) => Ok(data),
        Some(Section::Bytes(_)) => Err(format!("section {idx}: expected Tensor, got Bytes")),
        None => Err(format!("section {idx}: missing")),
    }
}

/// Packed `U8` payload bytes from their f32-per-byte tensor ride (the v1 container convention).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bytes_of(data: &[f32]) -> Vec<u8> {
    data.iter().map(|&f| f as u8).collect()
}

/// Chunk-local u32 indices from their f32 tensor ride.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn indices_of(data: &[f32]) -> Vec<u32> {
    data.iter().map(|&f| f as u32).collect()
}

/// A packed byte vector as its f32-per-byte tensor ride.
fn f32_of_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| f32::from(b)).collect()
}

// ================================================================================================
// lane views
// ================================================================================================

/// The native-lane inputs of one parameter at `make_update`: the trained value and the round
/// base, both materialized f32 (see the crate doc for the `HostBackend` extraction walk).
#[derive(Debug, Clone, Copy)]
pub struct ParamView<'a> {
    /// The trained parameter θ (native lane).
    pub theta: &'a [f32],
    /// The det-lane canonical round base θ⁽ᵗ⁾.
    pub round_base: &'a [f32],
}

/// The det-lane state of one parameter at `ingest`: the canonical master (written) and the round
/// base θ⁽ᵗ⁾ it rebases to. The round-base *snapshot* (post-ingest master → next round base) is
/// choreography, owned by the caller — exactly as v1's engine owned it outside the profile.
#[derive(Debug)]
pub struct IngestParam<'a> {
    /// The canonical master, rewritten to θ⁽ᵗ⁺¹⁾.
    pub master: &'a mut [f32],
    /// The round base θ⁽ᵗ⁾ (unchanged).
    pub round_base: &'a [f32],
}

/// `Δ = θ⁽ᵗ⁾ − θ` (the v1 `p.round_base().sub(p.tensor())`, elementwise f32).
fn delta_of(p: &ParamView<'_>) -> Vec<f32> {
    p.round_base
        .iter()
        .zip(p.theta.iter())
        .map(|(&b, &t)| b - t)
        .collect()
}

/// Elementwise `a·s + b` in the v1 native op order (`mul_s` then `add`), f32.
#[allow(clippy::cast_possible_truncation)]
fn scale_add(a: &[f32], s: f64, b: &[f32]) -> Vec<f32> {
    let sf = s as f32;
    a.iter().zip(b.iter()).map(|(&x, &y)| x * sf + y).collect()
}

// ================================================================================================
// sparse_loco — the flagship (v1 §5.3.1, ported verbatim)
// ================================================================================================

/// `sparse_loco` config — field set and defaults identical to the v1 SDK schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLocoCfg {
    /// Inner steps per round (H).
    pub h: u32,
    /// Error-feedback decay β.
    pub ef_decay: f64,
    /// 1-D chunk length.
    pub chunk: u32,
    /// Top-k retained per chunk.
    pub topk: u32,
    /// Value quantization bit width.
    pub bits: u32,
    /// Outer step size α.
    pub outer_alpha: f64,
    /// Median-norm clip of contributions before aggregation.
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

/// The `sparse_loco` profile: chunked top-k + absmax-packed values + error feedback (native
/// lane); det-lane ingest with absmax unpack + median-norm clip + streaming scatter-add + rebase
/// outer step. The v1 math verbatim over slices.
pub struct SparseLoco {
    cfg: SparseLocoCfg,
    /// Error-feedback residuals, one per param (native local state, zero-init like the v1
    /// `Persistent::local`).
    ef: Vec<Vec<f32>>,
}

impl SparseLoco {
    /// Build with zeroed error-feedback residuals sized per param.
    #[must_use]
    pub fn new(cfg: SparseLocoCfg, numels: &[usize]) -> Self {
        let ef = numels.iter().map(|&n| vec![0.0f32; n]).collect();
        Self { cfg, ef }
    }

    /// `Δ = θ⁽ᵗ⁾ − θ → acc = β·ef + Δ → top-k chunk → absmax pack → push; ef ← acc − scatter(sent)`.
    ///
    /// Sections per param `i`: `2i` packed values (f32-per-byte tensor), `2i+1` chunk-local
    /// indices — the exact v1 layout.
    ///
    /// # Panics
    /// If a param's element count is not divisible by `chunk` (the v1 registration invariant the
    /// model's config guarantees).
    pub fn make_update(&mut self, params: &[ParamView<'_>]) -> Vec<Section> {
        let (chunk, k, bits) = (self.cfg.chunk, self.cfg.topk, self.cfg.bits);
        let mut sections = Vec::with_capacity(params.len() * 2);
        for (i, p) in params.iter().enumerate() {
            let numel = p.theta.len();
            let delta = delta_of(p);
            let acc = scale_add(&self.ef[i], self.cfg.ef_decay, &delta);
            let (vals, idx) =
                daemon_vhc_det::topk_chunk(&acc, chunk as usize, k as usize).expect("topk layout");
            // Per-top-k-row codebook: pack the [n_chunks, k] values with chunk = k (v1 verbatim).
            let packed = daemon_vhc_det::absmax_pack(&vals, k as usize, bits).expect("pack");
            let n_chunks = (numel / chunk as usize) as u32;
            sections.push(Section::Tensor {
                data: f32_of_bytes(&packed),
                shape: vec![packed.len() as u32],
            });
            #[allow(clippy::cast_precision_loss)]
            sections.push(Section::Tensor {
                data: idx.iter().map(|&v| v as f32).collect(),
                shape: vec![n_chunks, k],
            });
            // ef ← acc − chunk_scatter(dequant(sent)) (the residual stays local, param-shaped).
            let sent_vals =
                daemon_vhc_det::det_absmax_unpack(&packed, k as usize, bits).expect("unpack");
            let sent = daemon_vhc_det::det_chunk_scatter(&sent_vals, &idx, chunk as usize, numel)
                .expect("scatter");
            self.ef[i] = acc.iter().zip(sent.iter()).map(|(&a, &s)| a - s).collect();
        }
        sections
    }

    /// Streaming det-lane ingest (v1 verbatim): per param, (optionally median-norm-clip then)
    /// scatter-add every peer's decoded sparse Δ̂, then rebase + apply the outer step
    /// `θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂`.
    ///
    /// `payloads` are the record-ordered committed containers (decoded).
    ///
    /// # Errors
    /// A `String` on a malformed payload (missing/mistyped section, det-kernel layout refusal).
    pub fn ingest(
        &mut self,
        params: &mut [IngestParam<'_>],
        payloads: &[Vec<Section>],
    ) -> Result<(), String> {
        let (chunk, k, bits) = (self.cfg.chunk, self.cfg.topk, self.cfg.bits);
        let count = payloads.len().max(1);
        for (i, p) in params.iter_mut().enumerate() {
            let numel = p.master.len();
            let vsec = 2 * i;
            let isec = vsec + 1;

            // Pass 1 (clip only): per-peer contribution norm → median clip target.
            let clip_norms: Vec<f64> = if self.cfg.clip {
                payloads
                    .iter()
                    .map(|pl| {
                        let packed = bytes_of(tensor_data(pl, vsec)?);
                        let vals = daemon_vhc_det::det_absmax_unpack(&packed, k as usize, bits)
                            .map_err(|e| format!("unpack: {e:?}"))?;
                        Ok(f64::from(daemon_vhc_det::det_l2norm(&vals)))
                    })
                    .collect::<Result<_, String>>()?
            } else {
                Vec::new()
            };
            let median = median_of(&clip_norms);

            // Pass 2 (streaming): decode → clip-scale → scatter-add into one accumulator.
            let mut acc = vec![0.0f32; numel];
            for (j, pl) in payloads.iter().enumerate() {
                let packed = bytes_of(tensor_data(pl, vsec)?);
                let vals = daemon_vhc_det::det_absmax_unpack(&packed, k as usize, bits)
                    .map_err(|e| format!("unpack: {e:?}"))?;
                let idx = indices_of(tensor_data(pl, isec)?);
                let s = if self.cfg.clip {
                    let norm = clip_norms[j];
                    if norm > median && norm > 0.0 {
                        median / norm
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                let scaled = daemon_vhc_det::det_scale(&vals, s);
                daemon_vhc_det::det_chunk_scatter_add(&mut acc, &scaled, &idx, chunk as usize)
                    .map_err(|e| format!("scatter_add: {e:?}"))?;
            }
            // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂ (rebase, then apply the canonical aggregate).
            p.master.copy_from_slice(p.round_base);
            #[allow(clippy::cast_precision_loss)]
            daemon_vhc_det::det_axpy(p.master, -self.cfg.outer_alpha / count as f64, &acc)
                .map_err(|e| format!("axpy: {e:?}"))?;
        }
        Ok(())
    }

    /// The replicated (digest-covered) det state — `sparse_loco` has none (`ef` is local).
    #[must_use]
    pub fn replicated_state(&self) -> Vec<&[f32]> {
        Vec::new()
    }
}

// ================================================================================================
// diloco — dense/int8 outer Nesterov baseline (v1 §5.3.2, ported verbatim)
// ================================================================================================

/// `diloco` config — field set and defaults identical to the v1 SDK schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiLoCoCfg {
    /// Inner steps per round (H).
    pub h: u32,
    /// Outer SGD learning rate.
    pub outer_lr: f64,
    /// Outer momentum.
    pub momentum: f64,
    /// Nesterov momentum (vs plain heavy-ball).
    pub nesterov: bool,
    /// Pseudo-gradient quantization: `0` = dense fp32, else bit width.
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

/// The `diloco` profile: dense (or int8) pseudo-gradient + outer (Nesterov) SGD on a
/// **replicated** det momentum — the canonical consensus outer-optimizer state, covered by the
/// round digest ([`DiLoCo::replicated_state`]).
pub struct DiLoCo {
    cfg: DiLoCoCfg,
    /// Outer momentum, one per param (det-lane REPLICATED state, zero-init).
    mom: Vec<Vec<f32>>,
}

impl DiLoCo {
    /// Build with zeroed replicated momentum sized per param.
    #[must_use]
    pub fn new(cfg: DiLoCoCfg, numels: &[usize]) -> Self {
        let mom = numels.iter().map(|&n| vec![0.0f32; n]).collect();
        Self { cfg, mom }
    }

    /// `Δ = θ⁽ᵗ⁾ − θ`, pushed dense (or absmax-packed when `quant_bits != 0`); one section per
    /// param — the exact v1 layout.
    pub fn make_update(&mut self, params: &[ParamView<'_>]) -> Vec<Section> {
        let mut sections = Vec::with_capacity(params.len());
        for p in params {
            let numel = p.theta.len();
            let delta = delta_of(p);
            if self.cfg.quant_bits == 0 {
                sections.push(Section::Tensor {
                    data: delta,
                    shape: vec![numel as u32],
                });
            } else {
                let packed =
                    daemon_vhc_det::absmax_pack(&delta, numel, self.cfg.quant_bits).expect("pack");
                sections.push(Section::Tensor {
                    data: f32_of_bytes(&packed),
                    shape: vec![packed.len() as u32],
                });
            }
        }
        sections
    }

    /// Aggregate the pseudo-gradient, advance the replicated momentum, and apply the outer
    /// (Nesterov) SGD step by rebasing to θ⁽ᵗ⁾ and subtracting `outer_lr · step` (v1 verbatim).
    ///
    /// # Errors
    /// A `String` on a malformed payload.
    pub fn ingest(
        &mut self,
        params: &mut [IngestParam<'_>],
        payloads: &[Vec<Section>],
    ) -> Result<(), String> {
        let count = payloads.len().max(1);
        for (i, p) in params.iter_mut().enumerate() {
            let numel = p.master.len();
            // g = (1/R)·Σ Δ (dense fp32 fold in record order — the v1 `acc.add` chain).
            let mut acc = vec![0.0f32; numel];
            for pl in payloads {
                let data = tensor_data(pl, i)?;
                let d = if self.cfg.quant_bits == 0 {
                    data.to_vec()
                } else {
                    daemon_vhc_det::det_absmax_unpack(&bytes_of(data), numel, self.cfg.quant_bits)
                        .map_err(|e| format!("unpack: {e:?}"))?
                };
                acc = daemon_vhc_det::det_add(&acc, &d).map_err(|e| format!("add: {e:?}"))?;
            }
            #[allow(clippy::cast_precision_loss)]
            let g = daemon_vhc_det::det_scale(&acc, 1.0 / count as f64);
            // m ← momentum·m + g (replicated momentum).
            let m_new = daemon_vhc_det::det_add(
                &daemon_vhc_det::det_scale(&self.mom[i], self.cfg.momentum),
                &g,
            )
            .map_err(|e| format!("mom add: {e:?}"))?;
            self.mom[i] = m_new.clone();
            // step = nesterov ? g + momentum·m : m
            let step = if self.cfg.nesterov {
                daemon_vhc_det::det_add(&g, &daemon_vhc_det::det_scale(&m_new, self.cfg.momentum))
                    .map_err(|e| format!("nesterov add: {e:?}"))?
            } else {
                m_new
            };
            // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − outer_lr·step
            p.master.copy_from_slice(p.round_base);
            daemon_vhc_det::det_axpy(p.master, -self.cfg.outer_lr, &step)
                .map_err(|e| format!("axpy: {e:?}"))?;
        }
        Ok(())
    }

    /// The replicated (digest-covered) det state: the outer momentum, in registration order —
    /// exactly the v1 `DetPersistent::replicated("dl.mom{i}")` coverage.
    #[must_use]
    pub fn replicated_state(&self) -> Vec<&[f32]> {
        self.mom.iter().map(Vec::as_slice).collect()
    }
}

// ================================================================================================
// demo — per-step DeMo / DisTrO (v1 §5.3.3, ported verbatim)
// ================================================================================================

/// `demo` config — field set and defaults identical to the v1 SDK schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoCfg {
    /// Fast-momentum decay β.
    pub momentum_decay: f64,
    /// 2-D DCT tile size (`chunk = tile²`).
    pub tile: u32,
    /// Top-k DCT coefficients per tile.
    pub topk: u32,
    /// Sign-SGD learning rate at ingest.
    pub sign_lr: f64,
    /// Decoupled weight decay.
    pub wd: f64,
    /// Partial-subtraction factor α removed from local momentum for what was sent.
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

/// The `demo` profile: per-step DCT energy extraction + top-k coefficients (native lane);
/// det-lane ingest sums coefficients, inverse-DCTs, and applies the **sign** of the aggregate
/// plus decoupled decay.
pub struct Demo {
    cfg: DemoCfg,
    /// Fast momentum, one per param (native local state, zero-init).
    mom: Vec<Vec<f32>>,
}

impl Demo {
    /// Build with zeroed momentum sized per param.
    #[must_use]
    pub fn new(cfg: DemoCfg, numels: &[usize]) -> Self {
        let mom = numels.iter().map(|&n| vec![0.0f32; n]).collect();
        Self { cfg, mom }
    }

    /// `M ← β·M + Δ`; top-k DCT coefficients per tile; transmit; `M ← M − α·IDCT(sent)`.
    /// Sections per param `i`: `2i` raw coefficient values, `2i+1` indices (v1 layout).
    ///
    /// # Panics
    /// If a param's element count is not divisible by `tile²` (the registration invariant).
    pub fn make_update(&mut self, params: &[ParamView<'_>]) -> Vec<Section> {
        let (tile, k) = (self.cfg.tile, self.cfg.topk);
        let block = tile * tile;
        let mut sections = Vec::with_capacity(params.len() * 2);
        for (i, p) in params.iter().enumerate() {
            let numel = p.theta.len();
            let delta = delta_of(p);
            let m = scale_add(&self.mom[i], self.cfg.momentum_decay, &delta);
            let coeffs = daemon_vhc_det::dct2(&m, tile as usize).expect("dct2");
            let (vals, idx) =
                daemon_vhc_det::topk_chunk(&coeffs, block as usize, k as usize).expect("topk");
            let n_chunks = (numel / block as usize) as u32;
            sections.push(Section::Tensor {
                data: vals.clone(),
                shape: vec![n_chunks, k],
            });
            #[allow(clippy::cast_precision_loss)]
            sections.push(Section::Tensor {
                data: idx.iter().map(|&v| v as f32).collect(),
                shape: vec![n_chunks, k],
            });
            // M ← M − α·IDCT(scatter(sent)) (param-shaped residual subtract).
            let sent_spatial = daemon_vhc_det::idct2(
                &daemon_vhc_det::det_chunk_scatter(&vals, &idx, block as usize, numel)
                    .expect("scatter"),
                tile as usize,
            )
            .expect("idct2");
            #[allow(clippy::cast_possible_truncation)]
            let alpha = self.cfg.alpha as f32;
            self.mom[i] = m
                .iter()
                .zip(sent_spatial.iter())
                .map(|(&mv, &sv)| mv - sv * alpha)
                .collect();
        }
        sections
    }

    /// Sum sparse coefficients across peers, inverse-DCT, and apply
    /// `−sign_lr·wd·θ⁽ᵗ⁾ − sign_lr·sign(aggregate)` from the rebased θ⁽ᵗ⁾ — all det lane
    /// (v1 verbatim: rebase, decay-axpy over the base, sign-axpy).
    ///
    /// # Errors
    /// A `String` on a malformed payload.
    pub fn ingest(
        &mut self,
        params: &mut [IngestParam<'_>],
        payloads: &[Vec<Section>],
    ) -> Result<(), String> {
        let (tile, block) = (self.cfg.tile, self.cfg.tile * self.cfg.tile);
        for (i, p) in params.iter_mut().enumerate() {
            let numel = p.master.len();
            let vsec = 2 * i;
            let isec = vsec + 1;
            let mut coeff_acc = vec![0.0f32; numel];
            for pl in payloads {
                let vals = tensor_data(pl, vsec)?;
                let idx = indices_of(tensor_data(pl, isec)?);
                daemon_vhc_det::det_chunk_scatter_add(&mut coeff_acc, vals, &idx, block as usize)
                    .map_err(|e| format!("scatter_add: {e:?}"))?;
            }
            let spatial = daemon_vhc_det::idct2(&coeff_acc, tile as usize)
                .map_err(|e| format!("idct2: {e:?}"))?;
            let sign = daemon_vhc_det::det_sign(&spatial);
            // θ ← θ⁽ᵗ⁾ − lr·wd·θ⁽ᵗ⁾ − lr·sign(aggregate) (decoupled decay + sign-SGD).
            let base = p.round_base.to_vec();
            p.master.copy_from_slice(&base);
            daemon_vhc_det::det_axpy(p.master, -self.cfg.sign_lr * self.cfg.wd, &base)
                .map_err(|e| format!("decay axpy: {e:?}"))?;
            daemon_vhc_det::det_axpy(p.master, -self.cfg.sign_lr, &sign)
                .map_err(|e| format!("sign axpy: {e:?}"))?;
        }
        Ok(())
    }

    /// The replicated (digest-covered) det state — `demo` has none (`mom` is local).
    #[must_use]
    pub fn replicated_state(&self) -> Vec<&[f32]> {
        Vec::new()
    }
}

/// The median of a list of contribution norms (guest f64 math over det-lane `det_l2norm`
/// results; deterministic ⇒ safe on the agree-path). Empty ⇒ `+∞` (no clip). v1 verbatim.
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

// ================================================================================================
// the Burn doorway
// ================================================================================================

/// The model↔profile composition helpers, generic over any Burn backend — `HostBackend` in
/// guests, `NdArray` in native tests. The profile core consumes f32 slices (the det lane is
/// slice math by definition); these helpers are the Burn-tensor side of the seam.
pub mod burn_lane {
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, TensorData};

    /// Upload a canonical f32 master as a rank-1 device tensor (post-ingest: the new master
    /// becomes the working weights; guest-side this registers the data over `compute@2::import`).
    #[must_use]
    pub fn upload<B: Backend>(master: &[f32], device: &B::Device) -> Tensor<B, 1> {
        let n = master.len();
        Tensor::from_data(TensorData::new(master.to_vec(), [n]), device)
    }

    /// `Δ = θ⁽ᵗ⁾ − θ` on-device (both operands stay by handle; export the result — or export θ
    /// and difference guest-side, which is what the reference guest does for its ef math).
    #[must_use]
    pub fn delta_device<B: Backend>(round_base: Tensor<B, 1>, theta: Tensor<B, 1>) -> Tensor<B, 1> {
        round_base.sub(theta)
    }
}
