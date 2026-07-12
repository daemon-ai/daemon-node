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
