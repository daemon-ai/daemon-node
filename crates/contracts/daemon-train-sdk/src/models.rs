// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! First-party preset experiments built from the SDK (architecture §5.1/§10.5).
//!
//! [`TinyLlama`] is the reference LLaMA-family decoder — embedding → N×(rmsnorm → attention →
//! rmsnorm → SwiGLU) → tied logits, cross-entropy loss, AdamW inner — parameterized entirely by
//! `[experiment.config]` and wired to a comm profile ([`crate::profiles`]) by config. It is the
//! SDK's dogfood consumer: the `guests/tiny-llama` module is a one-line `experiment!(TinyLlama)`
//! over this type, and the sim tests drive the identical code path natively (ABI §10.4/§10.5).

use crate::profiles::{Demo, DemoCfg, DiLoCo, DiLoCoCfg, SparseLoco, SparseLocoCfg};
use crate::{
    embedding, zero_grads, Batch, Config, Dtype, Experiment, Init, Manifest, Param, Persistent,
    StepCtx, UpdateBuilder, UpdatesView,
};
use serde::{Deserialize, Serialize};

/// AdamW inner-optimizer hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWCfg {
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

impl Default for AdamWCfg {
    fn default() -> Self {
        Self {
            lr: 4.0e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1.0e-8,
            wd: 0.1,
        }
    }
}

/// tiny-llama experiment config (the frozen experiment schema for this preset).
///
/// All dimensions are chosen so every parameter's element count is a multiple of the comm
/// profiles' chunking (`sparse_loco.chunk` / `demo.tile²`), so no guest-side padding is needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyLlamaCfg {
    /// Residual width.
    pub d_model: u32,
    /// Transformer blocks.
    pub n_layers: u32,
    /// Attention heads.
    pub n_heads: u32,
    /// Key/value heads (GQA); v1 preset requires `== n_heads`.
    pub n_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// Vocabulary size (tied input/output embedding).
    pub vocab: u32,
    /// Sequence length (the LM predicts positions `1..seq_len` from `0..seq_len-1`).
    pub seq_len: u32,
    /// SwiGLU hidden = `ffn_mult · d_model`.
    pub ffn_mult: u32,
    /// RoPE base.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub rmsnorm_eps: f64,
    /// Inner AdamW.
    pub inner: AdamWCfg,
    /// Comm profile selector: `"sparse_loco" | "diloco" | "demo"`.
    pub profile: String,
    /// sparse_loco config (used when `profile == "sparse_loco"`).
    #[serde(default)]
    pub sparse_loco: SparseLocoCfg,
    /// diloco config (used when `profile == "diloco"`).
    #[serde(default)]
    pub diloco: DiLoCoCfg,
    /// demo config (used when `profile == "demo"`).
    #[serde(default)]
    pub demo: DemoCfg,
}

impl Default for TinyLlamaCfg {
    fn default() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 16,
            vocab: 64,
            seq_len: 9,
            ffn_mult: 2,
            rope_theta: 10_000.0,
            rmsnorm_eps: 1.0e-5,
            inner: AdamWCfg::default(),
            profile: "sparse_loco".to_string(),
            sparse_loco: SparseLocoCfg {
                h: 3,
                chunk: 64,
                topk: 8,
                clip: false,
                ..SparseLocoCfg::default()
            },
            diloco: DiLoCoCfg {
                h: 3,
                ..DiLoCoCfg::default()
            },
            demo: DemoCfg::default(),
        }
    }
}

impl TinyLlamaCfg {
    /// The **P1 160M preset** (architecture §5.1, spec §17): a LLaMA-family decoder scaled to the
    /// "160M on one GPU" gate — `d_model 768, n_layers 12, n_heads 12, head_dim 64, seq_len 1024`,
    /// SwiGLU hidden `ffn_mult·d_model = 3072`, tied embedding over the **GPT-2 BPE** vocabulary
    /// (50257; TinyStories' native GPT-Neo tokenizer, a GPT-2 BPE — `< 65536` ⇒ `u16` shards).
    ///
    /// GQA is deferred at this scale: `n_kv_heads == n_heads` (the `build()` assert is kept). The
    /// comm profile is `sparse_loco` at the golden cadence `H = 30`; `chunk` is `256` — the largest
    /// power-of-two dividing **every** parameter's element count without guest-side padding (the
    /// embedding `50257·768` has 2-adic valuation 8, so the real-model golden `chunk 4096` would not
    /// divide it), with `topk = 4` preserving the golden 1/64 density. See `swarm-ledger-m1.md`.
    #[must_use]
    pub fn llama_160m() -> Self {
        Self {
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            head_dim: 64,
            vocab: 50257,
            seq_len: 1024,
            ffn_mult: 4,
            rope_theta: 10_000.0,
            rmsnorm_eps: 1.0e-5,
            inner: AdamWCfg::default(),
            profile: "sparse_loco".to_string(),
            sparse_loco: SparseLocoCfg {
                h: 30,
                chunk: 256,
                topk: 4,
                bits: 2,
                ef_decay: 0.95,
                outer_alpha: 1.0,
                clip: true,
            },
            diloco: DiLoCoCfg::default(),
            demo: DemoCfg::default(),
        }
    }

    /// The canonical parameter state dict (name, shape) in **registration order** — the exact order
    /// [`TinyLlama::build`] registers params, which is the checkpoint tensor order, digest coverage,
    /// and safetensors layout (ABI §6.3). Kept in lockstep with `build()`; a single source of truth
    /// for the safetensors converter (M1) and the burn reference model (M2).
    #[must_use]
    pub fn canonical_param_layout(&self) -> Vec<(String, Vec<u32>)> {
        let d = self.d_model;
        let qdim = self.n_heads * self.head_dim;
        let hidden = self.ffn_mult * d;
        let mut out: Vec<(String, Vec<u32>)> = Vec::new();
        out.push(("tok.weight".to_string(), vec![self.vocab, d]));
        for l in 0..self.n_layers {
            out.push((format!("l{l}.attn_norm"), vec![d]));
            out.push((format!("l{l}.wq"), vec![d, qdim]));
            out.push((format!("l{l}.wk"), vec![d, qdim]));
            out.push((format!("l{l}.wv"), vec![d, qdim]));
            out.push((format!("l{l}.wo"), vec![qdim, d]));
            out.push((format!("l{l}.ffn_norm"), vec![d]));
            out.push((format!("l{l}.w_gate"), vec![d, hidden]));
            out.push((format!("l{l}.w_up"), vec![d, hidden]));
            out.push((format!("l{l}.w_down"), vec![hidden, d]));
        }
        out.push(("norm.weight".to_string(), vec![d]));
        out
    }

    /// The exact trainable parameter count (Σ element counts over [`Self::canonical_param_layout`]).
    /// Tied embedding ⇒ the `[vocab, d]` table is counted once.
    #[must_use]
    pub fn param_count(&self) -> u64 {
        self.canonical_param_layout()
            .iter()
            .map(|(_, s)| s.iter().map(|&d| u64::from(d)).product::<u64>())
            .sum()
    }
}

/// One transformer block's parameters.
struct Block {
    attn_norm: Param,
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    ffn_norm: Param,
    w_gate: Param,
    w_up: Param,
    w_down: Param,
}

/// The comm profile the experiment composes (selected by config).
enum Comm {
    SparseLoco(SparseLoco),
    DiLoCo(DiLoCo),
    Demo(Demo),
}

/// The tiny LLaMA-style decoder.
pub struct TinyLlama {
    cfg: TinyLlamaCfg,
    tok: Param,
    blocks: Vec<Block>,
    norm: Param,
    // Inner AdamW moments, one (m, v) pair per registered param, in registration order.
    m: Vec<Persistent>,
    v: Vec<Persistent>,
    comm: Comm,
}

impl TinyLlama {
    fn all_params(&self) -> Vec<&Param> {
        let mut ps = vec![&self.tok];
        for b in &self.blocks {
            ps.extend([
                &b.attn_norm,
                &b.wq,
                &b.wk,
                &b.wv,
                &b.wo,
                &b.ffn_norm,
                &b.w_gate,
                &b.w_up,
                &b.w_down,
            ]);
        }
        ps.push(&self.norm);
        ps
    }

    /// A flat clone of the params the comm profile operates over (registration order).
    fn params_vec(&self) -> Vec<Param> {
        // Params are cheap stable handles; the profile only reads round-base + writes via det ops.
        self.all_params()
            .into_iter()
            .map(Param::stable_view)
            .collect()
    }
}

impl Experiment for TinyLlama {
    fn manifest(cfg: &Config) -> Manifest {
        let cfg: TinyLlamaCfg = cfg.parse();
        // Cadence (H, round modes, interval) comes from the selected profile; the module name is
        // the experiment's own (a profile is a library the experiment composes, not its identity).
        let profile = match cfg.profile.as_str() {
            "diloco" => DiLoCo::manifest(&cfg.diloco),
            "demo" => Demo::manifest(&cfg.demo),
            _ => SparseLoco::manifest(&cfg.sparse_loco),
        };
        let mut m = Manifest::new(
            "tiny-llama",
            env!("CARGO_PKG_VERSION"),
            profile.steps_per_round,
        );
        m.round_modes = profile.round_modes;
        m.min_round_interval_ms = profile.min_round_interval_ms;
        m
    }

    fn build(cfg: &Config) -> Self {
        let cfg: TinyLlamaCfg = cfg.parse();
        assert_eq!(
            cfg.n_kv_heads, cfg.n_heads,
            "tiny-llama v1 preset requires n_kv_heads == n_heads (GQA-repeat is future)"
        );
        let d = cfg.d_model;
        let qdim = cfg.n_heads * cfg.head_dim;
        let hidden = cfg.ffn_mult * d;
        let normal =
            |name: &str, dims: &[u32]| Param::new(name, dims, Dtype::F32, Init::Normal, 0.0, 0.02);
        let ones = |name: &str| Param::new(name, &[d], Dtype::F32, Init::Ones, 0.0, 0.0);

        let tok = normal("tok.weight", &[cfg.vocab, d]);
        let blocks = (0..cfg.n_layers)
            .map(|l| Block {
                attn_norm: ones(&format!("l{l}.attn_norm")),
                wq: normal(&format!("l{l}.wq"), &[d, qdim]),
                wk: normal(&format!("l{l}.wk"), &[d, qdim]),
                wv: normal(&format!("l{l}.wv"), &[d, qdim]),
                wo: normal(&format!("l{l}.wo"), &[qdim, d]),
                ffn_norm: ones(&format!("l{l}.ffn_norm")),
                w_gate: normal(&format!("l{l}.w_gate"), &[d, hidden]),
                w_up: normal(&format!("l{l}.w_up"), &[d, hidden]),
                w_down: normal(&format!("l{l}.w_down"), &[hidden, d]),
            })
            .collect::<Vec<_>>();
        let norm = ones("norm.weight");

        // Register the AdamW moments (local persistents) in the same order as `all_params`.
        let mut me = Self {
            cfg: cfg.clone(),
            tok,
            blocks,
            norm,
            m: Vec::new(),
            v: Vec::new(),
            comm: Comm::Demo(Demo::new(DemoCfg::default(), &[])), // placeholder, replaced below
        };
        let shapes: Vec<Vec<u32>> = me.all_params().iter().map(|p| p.shape().to_vec()).collect();
        me.m = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| Persistent::local(&format!("adamw.m{i}"), s, Dtype::F32))
            .collect();
        me.v = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| Persistent::local(&format!("adamw.v{i}"), s, Dtype::F32))
            .collect();

        let params = me.params_vec();
        me.comm = match cfg.profile.as_str() {
            "diloco" => Comm::DiLoCo(DiLoCo::new(cfg.diloco.clone(), &params)),
            "demo" => Comm::Demo(Demo::new(cfg.demo.clone(), &params)),
            _ => Comm::SparseLoco(SparseLoco::new(cfg.sparse_loco.clone(), &params)),
        };
        me
    }

    fn step(&mut self, batch: &Batch, ctx: &StepCtx) {
        let cfg = &self.cfg;
        let b = batch.size();
        let seq = batch.seq_len();
        let s = seq - 1;
        let d = cfg.d_model;
        let nh = cfg.n_heads;
        let hd = cfg.head_dim;
        let rows = b * s;
        let scale = 1.0 / f64::from(hd).sqrt();

        let ids = batch.tokens(); // [b, seq]
        let inp = ids.slice(1, 0, seq - 1); // [b, s]
        let tgt = ids.slice(1, 1, seq); // [b, s]

        let emb = embedding(&self.tok, &inp); // [b, s, d]
        let mut h = emb.reshape(&[rows, d]); // [rows, d]

        for blk in &self.blocks {
            // Attention sub-block.
            let normed = h.rmsnorm(blk.attn_norm.tensor(), cfg.rmsnorm_eps);
            let q = normed
                .matmul(blk.wq.tensor())
                .reshape(&[b, s, nh, hd])
                .transpose(1, 2); // [b, nh, s, hd]
            let k = normed
                .matmul(blk.wk.tensor())
                .reshape(&[b, s, nh, hd])
                .transpose(1, 2);
            let v = normed
                .matmul(blk.wv.tensor())
                .reshape(&[b, s, nh, hd])
                .transpose(1, 2);
            let q = q.rope(0, cfg.rope_theta, false);
            let k = k.rope(0, cfg.rope_theta, false);
            let attn = q
                .flash_attn(&k, &v, true, scale) // [b, nh, s, hd]
                .transpose(1, 2) // [b, s, nh, hd]
                .reshape(&[rows, nh * hd])
                .matmul(blk.wo.tensor()); // [rows, d]
            h = h.add(&attn);

            // SwiGLU FFN sub-block.
            let normed2 = h.rmsnorm(blk.ffn_norm.tensor(), cfg.rmsnorm_eps);
            let gate = normed2.matmul(blk.w_gate.tensor()).silu();
            let up = normed2.matmul(blk.w_up.tensor());
            let ffn = gate.mul(&up).matmul(blk.w_down.tensor()); // [rows, d]
            h = h.add(&ffn);
        }

        let h = h.rmsnorm(self.norm.tensor(), cfg.rmsnorm_eps);
        // Tied embeddings: logits = h · tok.weightᵀ  →  [rows, vocab].
        let logits = h.matmul(&self.tok.transpose(0, 1));
        let targets = tgt.reshape(&[rows]);
        let loss = logits.cross_entropy(&targets, -1);
        loss.metric("loss");
        loss.mul_s(ctx.loss_scale(batch)).backward();
    }

    fn inner_update(&mut self, inner_step: u32) {
        let hp = self.cfg.inner.clone();
        let params = self.params_vec();
        for (i, p) in params.iter().enumerate() {
            p.adamw_step(
                &p.grad(),
                &self.m[i],
                &self.v[i],
                inner_step + 1,
                hp.lr,
                hp.beta1,
                hp.beta2,
                hp.eps,
                hp.wd,
            );
        }
        zero_grads();
    }

    fn make_update(&mut self, _round: u64) -> UpdateBuilder {
        let params = self.params_vec();
        match &mut self.comm {
            Comm::SparseLoco(p) => p.make_update(&params),
            Comm::DiLoCo(p) => p.make_update(&params),
            Comm::Demo(p) => p.make_update(&params),
        }
    }

    fn ingest(&mut self, _round: u64, updates: &UpdatesView) {
        let params = self.params_vec();
        match &mut self.comm {
            Comm::SparseLoco(p) => p.ingest(&params, updates),
            Comm::DiLoCo(p) => p.ingest(&params, updates),
            Comm::Demo(p) => p.ingest(&params, updates),
        }
    }

    /// The `[experiment.config]` defaults layer (ABI §6.2): the full [`TinyLlamaCfg`] default.
    fn defaults() -> Vec<u8> {
        let mut bytes = Vec::new();
        ciborium::into_writer(&TinyLlamaCfg::default(), &mut bytes)
            .expect("TinyLlamaCfg is always CBOR-serializable");
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact 160M-preset parameter count (reported in `swarm-ledger-m1.md`): tok 38,597,376 +
    /// 12·9,438,720 + 768 (final norm) = 151,862,784 (≈152M — within bounds of the "160M" spec row).
    const LLAMA_160M_PARAMS: u64 = 151_862_784;

    #[test]
    fn llama_160m_param_count_is_exact() {
        let cfg = TinyLlamaCfg::llama_160m();
        assert_eq!(cfg.param_count(), LLAMA_160M_PARAMS);
        // Spot-check the derived dims the preset promises.
        assert_eq!(cfg.n_kv_heads, cfg.n_heads, "GQA deferred: assert holds");
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.d_model, "qdim == d_model");
        assert_eq!(cfg.ffn_mult * cfg.d_model, 3072, "SwiGLU hidden");
        assert_eq!(cfg.vocab, 50257, "GPT-2 BPE vocab (u16 shards)");
    }

    #[test]
    fn canonical_layout_matches_registration_order() {
        // The layout lists 1 (tok) + n_layers·9 + 1 (final norm) params in build() order.
        let cfg = TinyLlamaCfg::llama_160m();
        let layout = cfg.canonical_param_layout();
        assert_eq!(layout.len(), 1 + (cfg.n_layers as usize) * 9 + 1);
        assert_eq!(layout.first().unwrap().0, "tok.weight");
        assert_eq!(layout.last().unwrap().0, "norm.weight");
        assert_eq!(layout[1].0, "l0.attn_norm");
        assert_eq!(layout[2].0, "l0.wq");
        assert_eq!(layout[9].0, "l0.w_down");
        assert_eq!(layout[10].0, "l1.attn_norm");
    }

    /// The 160M `sparse_loco` chunk (256) must divide **every** param element count so `make_update`
    /// needs no guest-side padding (the models.rs invariant). `topk` must not exceed `chunk`.
    #[test]
    fn llama_160m_chunk_divides_all_params() {
        let cfg = TinyLlamaCfg::llama_160m();
        let chunk = u64::from(cfg.sparse_loco.chunk);
        assert!(u64::from(cfg.sparse_loco.topk) <= chunk);
        for (name, shape) in cfg.canonical_param_layout() {
            let numel: u64 = shape.iter().map(|&d| u64::from(d)).product();
            assert_eq!(
                numel % chunk,
                0,
                "param {name} numel {numel} % {chunk} != 0"
            );
        }
    }

    /// `canonical_param_layout` is the single source of truth for `build()`: every listed param is
    /// actually registered (right name + element count) under the sim backend, in the same order.
    #[cfg(feature = "sim")]
    #[test]
    fn build_registers_the_canonical_layout() {
        use crate::sim;
        // A small (2-layer) config keeps the sim build cheap while exercising the full layout shape.
        let cfg = TinyLlamaCfg {
            n_layers: 2,
            ..TinyLlamaCfg::default()
        };
        sim::reset(0xDAE0_160D);
        let _exp = TinyLlama::build(&Config::from_value(&cfg));
        for (name, shape) in cfg.canonical_param_layout() {
            let master = sim::param_master(&name)
                .unwrap_or_else(|| panic!("param {name} not registered by build()"));
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            assert_eq!(master.len(), numel, "param {name} element count");
        }
        assert!(sim::param_master("l99.wq").is_none(), "no phantom params");
    }

    /// Meta-mode VRAM/RAM reconciliation vs spec §5.1 planning table (HOST-8/RUN-10 semantics). The
    /// spec row assumes **bf16 weights**; the P1 preset stores **fp32** masters+storage (det-lane
    /// exactness), so the footprint is larger per-tensor but must still land the spec's conclusion
    /// ("fits on an 8 GB card"). Recorded as a spec-amendment candidate in `swarm-ledger-m1.md`.
    #[test]
    fn llama_160m_footprint_reconciles_with_spec_table() {
        let cfg = TinyLlamaCfg::llama_160m();
        let n = cfg.param_count();
        let gib = 1u64 << 30;
        // fp32 everywhere on the P1 preset.
        let param_bytes = n * 4; // storage (fp32)
        let master_bytes = n * 4; // fp32 canonical master (ABI §5.9)
        let grad_bytes = n * 4; // fp32 grad accumulators (ABI §5.1)
        let adam_bytes = n * 4 * 2; // m + v (fp32)
                                    // The WasmBackend coarse footprint is params+master+grad = master_bytes·3 (wasm_backend.rs).
        let coarse_vram = master_bytes * 3;
        // Steady state incl. optimizer state (still excl. activations, which the meta pass adds).
        let steady_vram = param_bytes + master_bytes + grad_bytes + adam_bytes;

        // Spec §5.1 "160M" row conclusion: fits an 8 GB card. fp32 storage stays well within it.
        assert!(
            steady_vram < 8 * gib,
            "160M fp32 steady state must fit an 8 GB card"
        );
        // The coarse footprint is a lower bound (~1.7 GiB) below the spec's ~4.5 GB (which folds in
        // activations at seq 2048); the preset runs seq 1024, so the coarse number is expected low.
        assert!(
            coarse_vram >= gib && coarse_vram < 3 * gib,
            "coarse VRAM ~1.7 GiB"
        );
        // Host RAM: fp32 masters + round-base ≈ 2·params (spec §5.1 host-RAM row ≈ 2 GB at 160M).
        let host_ram = master_bytes * 2;
        assert!(
            host_ram >= gib && host_ram < 3 * gib,
            "host RAM ~1.1 GiB (spec ~2 GB)"
        );
    }
}
