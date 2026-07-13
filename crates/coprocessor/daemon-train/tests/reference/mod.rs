// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **reference-parity harness** (M2 — the P1 numeric exit-gate, spec §17 / TDD §8 "P1").
//!
//! An *independent-of-the-tabi-host* burn reference implementation of the 160M LLaMA
//! forward/backward/AdamW training step ([`RefLlama`]), plus drivers for the tabi (module) path and
//! the reference path that both consume the **same weights** (matched init via the tabi path's own
//! `Instance::param_master`) and the **same token batches**, so their loss curves are comparable.
//!
//! ## Independence
//!
//! The tabi path is `tiny_llama.wasm` → wasm ABI → [`daemon_train::WasmBackend`] → the `OpBackend`
//! engine (`BurnBackend<Autodiff<B>>`) → burn. [`RefLlama`] issues **burn tensor ops directly** — no
//! wasm sandbox, no ABI dispatch, no handle arena, no `OpBackend` indirection — differentiated by
//! burn's own `Autodiff` decorator (`Tensor::backward`). It is therefore independent of the *tabi
//! host* (the thing under test) while faithfully implementing the same architecture, so any per-step
//! divergence isolates to tolerance-class effects (burn kernel non-associativity across two op-issue
//! orders + f32 AdamW), never the det lane (spec §7.2, program "Determinism story").
//!
//! ## Op-definition grounding (mirrored from the tabi native lane, `burn_backend.rs`)
//!
//! RMSNorm `x·rsqrt(mean(x²)+eps)·w` (`burn_backend.rs:386`); RoPE half-split `freq_j=θ^(−2j/hd)`
//! (`:404`); dense causal attention `softmax(QKᵀ·scale + mask)·V`, mask `−1e30`, `scale=1/√hd`
//! (`:451`); SwiGLU `silu(x·Wgate)⊙(x·Wup)·Wdown` (`models.rs:380`); shifted-max cross-entropy over
//! counted rows (`:347`); fused f32 AdamW `w←w·(1−lr·wd)−lr·m̂/(√v̂+eps)`, step `inner_step+1`
//! (`:556`, `models.rs:400`); loss scaling `size/step_seqs` (`api.rs:685`) — the harness drives both
//! paths with `step_seqs = num_sequences` ⇒ scale `1.0`.
//!
//! This is a shared test submodule (a subdirectory `mod`, so cargo does not build it as its own test
//! binary); the ndarray + wgpu parity test files pass the backend as the generic parameter, exactly
//! like the G1/G2 `tolerance` harness.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{activation, Int, Tensor, TensorData};

use daemon_train::{BackendKind, EngineConfig, Worker};
use daemon_train_safetensors::StateDict;
use daemon_train_sdk::models::TinyLlamaCfg;

use crate::tolerance::{assert_close, tol_for, OpClass};

// -- guest build (shared with the other wasm-backed suites; G2 stale-guest guard) ---------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

/// Always rebuild the guests before loading (no-op when fresh) so a stale `tiny_llama.wasm` cannot
/// surface as NaN (Merge-1 adjudication note); `SWARM_TEST_GUEST_DIR` skips it (CI prebuilt).
fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
}

/// The tiny-llama guest module bytes (built on demand).
#[must_use]
pub fn tiny_llama_wasm() -> Vec<u8> {
    ensure_built();
    let path = guest_dir().join("tiny_llama.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The `[experiment.config]` CBOR bytes `da_build` receives.
#[must_use]
pub fn cfg_cbor(cfg: &TinyLlamaCfg) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(cfg, &mut b).expect("cbor");
    b
}

/// A host engine profile sized for `cfg` (self-protection budgets, not domain limits — ABI §8): a
/// real fp32 model's matmuls take longer wall-clock and touch far more handles than the tiny default.
#[must_use]
pub fn engine_for(cfg: &TinyLlamaCfg, backend: BackendKind) -> EngineConfig {
    let big = cfg.d_model >= 512;
    EngineConfig {
        backend,
        fuel_per_call: 1 << 36,
        epoch_deadline: std::time::Duration::from_secs(if big { 3600 } else { 120 }),
        op_budget: 1 << 32,
        max_step_handles: 1 << 26,
        ..EngineConfig::default()
    }
}

// -- token batches ------------------------------------------------------------------------------

/// A `[b, seq]` token batch, row-major, plus its dimensions. Both paths consume the identical tokens.
pub struct TokenBatch {
    /// `b · seq` token ids, row-major.
    pub tokens: Vec<u32>,
    /// Number of sequences.
    pub b: u32,
    /// Sequence length.
    pub seq: u32,
}

impl TokenBatch {
    /// A deterministic small-vocab batch (identical on both paths) for the reduced always-on config,
    /// whose vocab is smaller than GPT-2's so real TinyStories tokens would be out of range.
    #[must_use]
    pub fn deterministic(b: u32, seq: u32, vocab: u32) -> Self {
        let tokens = (0..u64::from(b) * u64::from(seq))
            .map(|i| (i.wrapping_mul(2_654_435_761) % u64::from(vocab)) as u32)
            .collect();
        Self { tokens, b, seq }
    }

    /// The first `b` sequences of the **real vendored TinyStories fixture** (M1), each `seq_len`
    /// GPT-2 tokens — the same corpus the swarm/tabi data path serves.
    #[must_use]
    pub fn tinystories(b: u32) -> Self {
        use daemon_swarm_run::data::Corpus;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../swarm/daemon-swarm-run/tests/fixtures/tinystories");
        let manifest_json =
            std::fs::read_to_string(dir.join("manifest.json")).expect("read fixture manifest");
        let manifest =
            daemon_swarm_run::data::Manifest::from_json(&manifest_json).expect("parse manifest");
        let shards: Vec<Vec<u8>> = manifest
            .shards
            .iter()
            .map(|s| std::fs::read(dir.join(&s.name)).expect("read shard"))
            .collect();
        let seq = manifest.seq_len;
        let corpus = Corpus::from_parts(manifest, shards).expect("blake3-verified corpus");
        let mut tokens = Vec::new();
        for i in 0..u64::from(b) {
            tokens.extend(corpus.sequence(i).expect("sequence"));
        }
        Self { tokens, b, seq }
    }

    /// Truncate each sequence to `seq` tokens (the fixture window is 1024; a smaller model uses a
    /// shorter `seq_len`). Both paths still consume the identical real tokens.
    #[must_use]
    pub fn truncate_seq(&self, seq: u32) -> Self {
        assert!(seq <= self.seq, "truncate_seq only shortens");
        let mut tokens = Vec::with_capacity((self.b * seq) as usize);
        for bi in 0..self.b as usize {
            let base = bi * self.seq as usize;
            tokens.extend_from_slice(&self.tokens[base..base + seq as usize]);
        }
        Self {
            tokens,
            b: self.b,
            seq,
        }
    }
}

// -- reference model (independent burn implementation) ------------------------------------------

/// The dimensions the reference forward needs (a flat view of the relevant [`TinyLlamaCfg`] fields).
#[derive(Clone)]
struct RefCfg {
    d: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    hidden: usize,
    vocab: usize,
    seq_len: usize,
    rope_theta: f64,
    eps: f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    adam_eps: f64,
    wd: f64,
}

impl RefCfg {
    fn from(cfg: &TinyLlamaCfg) -> Self {
        Self {
            d: cfg.d_model as usize,
            n_layers: cfg.n_layers as usize,
            n_heads: cfg.n_heads as usize,
            head_dim: cfg.head_dim as usize,
            hidden: (cfg.ffn_mult * cfg.d_model) as usize,
            vocab: cfg.vocab as usize,
            seq_len: cfg.seq_len as usize,
            rope_theta: cfg.rope_theta,
            eps: cfg.rmsnorm_eps,
            lr: cfg.inner.lr,
            beta1: cfg.inner.beta1,
            beta2: cfg.inner.beta2,
            adam_eps: cfg.inner.eps,
            wd: cfg.inner.wd,
        }
    }
}

/// One block's params (flat rank-1 tensors; reshaped in the forward).
struct RefBlock<B: AutodiffBackend> {
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

/// The independent burn LLaMA reference (params as flat leaves; forward reshapes to natural ranks).
pub struct RefLlama<B: AutodiffBackend> {
    cfg: RefCfg,
    device: B::Device,
    tok: Tensor<B, 1>,
    blocks: Vec<RefBlock<B>>,
    norm: Tensor<B, 1>,
    // AdamW moments in canonical registration order (tok, per-block×9, norm).
    m: Vec<Tensor<B, 1>>,
    v: Vec<Tensor<B, 1>>,
}

fn to_vec<B: burn::tensor::backend::Backend>(t: &Tensor<B, 1>) -> Vec<f32> {
    t.to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("f32 tensor data")
}

/// A fresh grad-carrying leaf on `device`.
fn leaf<B: AutodiffBackend>(device: &B::Device, data: Vec<f32>) -> Tensor<B, 1> {
    let n = data.len();
    Tensor::<B, 1>::from_data(TensorData::new(data, [n]), device).require_grad()
}

/// A fresh (non-grad) zero state tensor on `device`.
fn zeros<B: AutodiffBackend>(device: &B::Device, n: usize) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_data(TensorData::new(vec![0.0f32; n], [n]), device)
}

impl<B: AutodiffBackend> RefLlama<B> {
    /// Build the reference from a canonical state dict (matched init from the tabi path). The
    /// `sd` list is `(name, shape, data)` in registration order — exactly `Instance::params()` +
    /// `param_master`, i.e. `canonical_param_layout` order.
    #[must_use]
    pub fn from_state_dict(cfg: &TinyLlamaCfg, device: B::Device, sd: &StateDict) -> Self {
        let rc = RefCfg::from(cfg);
        let get = |name: &str| -> Vec<f32> {
            sd.tensors
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, _, d)| d.clone())
                .unwrap_or_else(|| panic!("state dict missing {name}"))
        };
        let tok = leaf::<B>(&device, get("tok.weight"));
        let blocks: Vec<RefBlock<B>> = (0..rc.n_layers)
            .map(|l| RefBlock {
                attn_norm: leaf::<B>(&device, get(&format!("l{l}.attn_norm"))),
                wq: leaf::<B>(&device, get(&format!("l{l}.wq"))),
                wk: leaf::<B>(&device, get(&format!("l{l}.wk"))),
                wv: leaf::<B>(&device, get(&format!("l{l}.wv"))),
                wo: leaf::<B>(&device, get(&format!("l{l}.wo"))),
                ffn_norm: leaf::<B>(&device, get(&format!("l{l}.ffn_norm"))),
                w_gate: leaf::<B>(&device, get(&format!("l{l}.w_gate"))),
                w_up: leaf::<B>(&device, get(&format!("l{l}.w_up"))),
                w_down: leaf::<B>(&device, get(&format!("l{l}.w_down"))),
            })
            .collect();
        let norm = leaf::<B>(&device, get("norm.weight"));
        // Moments start at zero (matching a fresh Adam state; the tabi path registers zero-init
        // `local` persistents for the same moments), one per param in canonical order.
        let sizes: Vec<usize> = cfg
            .canonical_param_layout()
            .iter()
            .map(|(_, shape)| shape.iter().map(|&d| d as usize).product())
            .collect();
        let m = sizes.iter().map(|&n| zeros::<B>(&device, n)).collect();
        let v = sizes.iter().map(|&n| zeros::<B>(&device, n)).collect();
        Self {
            cfg: rc,
            device,
            tok,
            blocks,
            norm,
            m,
            v,
        }
    }

    /// The params in canonical registration order (shared by AdamW + `state_dict`).
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

    fn rmsnorm(&self, x: Tensor<B, 2>, w: &Tensor<B, 1>, d: usize) -> Tensor<B, 2> {
        let w2 = w.clone().reshape([1, d]);
        let ms = x.clone().powf_scalar(2.0).mean_dim(1); // [rows,1]
        let inv = ms.add_scalar(self.cfg.eps).sqrt().recip();
        x.mul(inv).mul(w2)
    }

    /// RoPE (half-split), applied on `[b, nh, s, hd]`; position depends only on the `s` axis, so
    /// cos/sin broadcast over `(b, nh)` (`burn_backend.rs:404-450`).
    fn rope(&self, x: Tensor<B, 4>, s: usize) -> Tensor<B, 4> {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let theta = self.cfg.rope_theta as f32;
        let mut cosv = vec![0.0f32; s * half];
        let mut sinv = vec![0.0f32; s * half];
        for pos in 0..s {
            for j in 0..half {
                let freq = 1.0 / theta.powf(2.0 * j as f32 / hd as f32);
                let angle = pos as f32 * freq;
                cosv[pos * half + j] = angle.cos();
                sinv[pos * half + j] = angle.sin();
            }
        }
        let cos = Tensor::<B, 4>::from_data(TensorData::new(cosv, [1, 1, s, half]), &self.device);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sinv, [1, 1, s, half]), &self.device);
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);
        let out1 = x1.clone().mul(cos.clone()).sub(x2.clone().mul(sin.clone()));
        let out2 = x1.mul(sin).add(x2.mul(cos));
        Tensor::cat(vec![out1, out2], 3)
    }

    /// One forward pass returning the (unscaled) cross-entropy loss tensor for `backward`.
    fn forward(&self, tokens: &[u32], b: usize, seq: usize) -> Tensor<B, 1> {
        let cfg = &self.cfg;
        let s = seq - 1;
        let d = cfg.d;
        let nh = cfg.n_heads;
        let hd = cfg.head_dim;
        let qdim = nh * hd;
        let rows = b * s;
        let scale = 1.0 / (hd as f64).sqrt();

        // inp = tokens[:, 0..s]; tgt = tokens[:, 1..seq]
        let mut inp = Vec::with_capacity(rows);
        let mut tgt = Vec::with_capacity(rows);
        for bi in 0..b {
            for si in 0..s {
                inp.push(tokens[bi * seq + si] as i64);
                tgt.push(tokens[bi * seq + si + 1] as i64);
            }
        }

        // Embedding: select rows of tok.[vocab,d] by inp -> [rows, d].
        let tok2 = self.tok.clone().reshape([cfg.vocab, d]);
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(inp, [rows]), &self.device);
        let mut h = tok2.select(0, idx); // [rows, d]

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
                    .matmul(blk.w_gate.clone().reshape([d, cfg.hidden])),
            );
            let up = normed2.matmul(blk.w_up.clone().reshape([d, cfg.hidden]));
            let ffn = gate
                .mul(up)
                .matmul(blk.w_down.clone().reshape([cfg.hidden, d])); // [rows, d]
            h = h.add(ffn);
        }

        let h = self.rmsnorm(h, &self.norm, d);
        // Tied embedding: logits = h · tokᵀ -> [rows, vocab].
        let logits = h.matmul(self.tok.clone().reshape([cfg.vocab, d]).swap_dims(0, 1));

        // Shifted-max cross-entropy over counted rows (`burn_backend.rs:347`).
        let max = logits.clone().max_dim(1).detach(); // [rows,1]
        let shifted = logits.sub(max);
        let logsm = shifted.clone().sub(shifted.exp().sum_dim(1).log());
        let mut onehot = vec![0.0f32; rows * cfg.vocab];
        for (i, &t) in tgt.iter().enumerate() {
            onehot[i * cfg.vocab + t as usize] = 1.0;
        }
        let oh =
            Tensor::<B, 2>::from_data(TensorData::new(onehot, [rows, cfg.vocab]), &self.device);
        let denom = rows.max(1) as f32;
        logsm.mul(oh).sum().mul_scalar(-1.0 / denom).reshape([1])
    }

    /// One full training step: forward + backward + fused AdamW at `inner_step` (0-based; the AdamW
    /// bias-correction step is `inner_step+1`, `models.rs:400`). Returns the (unscaled) loss.
    pub fn step(&mut self, tokens: &[u32], b: usize, seq: usize, inner_step: u32) -> f32 {
        let loss = self.forward(tokens, b, seq);
        let loss_val = to_vec(&loss)[0];
        let grads = loss.backward();

        let params = self.flat_params();
        let hp_t = f64::from(inner_step + 1);
        let bc1 = (1.0 - self.cfg.beta1.powf(hp_t)) as f32;
        let bc2 = (1.0 - self.cfg.beta2.powf(hp_t)) as f32;
        let (b1, b2) = (self.cfg.beta1 as f32, self.cfg.beta2 as f32);
        let (lr, wd, eps) = (
            self.cfg.lr as f32,
            self.cfg.wd as f32,
            self.cfg.adam_eps as f32,
        );

        let mut new_params = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let g_inner = p.grad(&grads).expect("param grad");
            let g = Tensor::<B, 1>::from_data(g_inner.to_data(), &self.device);
            let m0 = self.m[i].clone();
            let v0 = self.v[i].clone();
            let w0 = p.clone().detach();
            let m1 = m0.mul_scalar(b1).add(g.clone().mul_scalar(1.0 - b1));
            let v1 = v0.mul_scalar(b2).add(g.clone().mul(g).mul_scalar(1.0 - b2));
            let mhat = m1.clone().div_scalar(bc1);
            let vhat = v1.clone().div_scalar(bc2);
            let denom = vhat.sqrt().add_scalar(eps);
            let w1 = w0
                .mul_scalar(1.0 - lr * wd)
                .sub(mhat.div(denom).mul_scalar(lr));
            self.m[i] = m1;
            self.v[i] = v1;
            // Re-materialize as a fresh leaf for the next step's graph.
            new_params.push(leaf::<B>(&self.device, to_vec(&w1)));
        }
        // Scatter the updated params back into the struct (canonical order).
        let mut it = new_params.into_iter();
        self.tok = it.next().unwrap();
        for blk in &mut self.blocks {
            blk.attn_norm = it.next().unwrap();
            blk.wq = it.next().unwrap();
            blk.wk = it.next().unwrap();
            blk.wv = it.next().unwrap();
            blk.wo = it.next().unwrap();
            blk.ffn_norm = it.next().unwrap();
            blk.w_gate = it.next().unwrap();
            blk.w_up = it.next().unwrap();
            blk.w_down = it.next().unwrap();
        }
        self.norm = it.next().unwrap();
        loss_val
    }

    /// The current params as a canonical state dict (for final-weights parity).
    #[must_use]
    pub fn state_dict(&self, cfg: &TinyLlamaCfg) -> StateDict {
        let layout = cfg.canonical_param_layout();
        let params = self.flat_params();
        let mut sd = StateDict::new();
        for ((name, shape), t) in layout.iter().zip(params.iter()) {
            sd.push(
                name.clone(),
                shape.iter().map(|&d| d as usize).collect(),
                to_vec(t),
            );
        }
        sd
    }
}

// -- drivers ------------------------------------------------------------------------------------

/// The result of driving a path for K steps.
pub struct PathRun {
    /// Per-step (unscaled) losses.
    pub losses: Vec<f32>,
    /// The final canonical state dict.
    pub final_state: StateDict,
    /// The initial canonical state dict (matched-init source for the reference).
    pub init_state: StateDict,
    /// Total wall-clock seconds for the measured steps (excludes build + init read).
    pub step_secs: f64,
    /// Per-step wall-clock seconds (index 0 carries lazy-bringup/kernel-compile warmup on wgpu).
    pub per_step_secs: Vec<f64>,
}

/// tokens/s + variance over the measured steps, dropping `warmup` leading steps.
#[must_use]
pub fn throughput_stats(run: &PathRun, b: u32, seq: u32, warmup: usize) -> (f64, f64, f64) {
    let measured: Vec<f64> = run.per_step_secs.iter().skip(warmup).copied().collect();
    let n = measured.len().max(1) as f64;
    let mean = measured.iter().sum::<f64>() / n;
    let var = measured.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    let toks_per_step = f64::from(b) * f64::from(seq - 1);
    let tps = if mean > 0.0 {
        toks_per_step / mean
    } else {
        0.0
    };
    (tps, mean, sd)
}

/// Drive the **tabi (module) path** for `steps` inner steps over `batch` on `backend`, returning the
/// per-step losses + the initial and final canonical state dicts (read via `Instance::param_master`).
pub fn drive_tabi(
    cfg: &TinyLlamaCfg,
    backend: BackendKind,
    batch: &TokenBatch,
    steps: u32,
) -> PathRun {
    let worker = Worker::new(engine_for(cfg, backend)).expect("worker");
    let module = worker.load_module(&tiny_llama_wasm()).expect("load module");
    let mut inst = worker.instantiate(&module).expect("instantiate");
    inst.build(&cfg_cbor(cfg)).expect("da_build");

    let read_state = |inst: &daemon_train::Instance| -> StateDict {
        let mut sd = StateDict::new();
        for p in inst.params() {
            let data = inst.param_master(&p.name).expect("param master");
            sd.push(
                p.name.clone(),
                p.shape.iter().map(|&d| d as usize).collect(),
                data,
            );
        }
        sd
    };
    let init_state = read_state(&inst);

    let mut losses = Vec::new();
    let mut per_step_secs = Vec::new();
    for step in 0..steps {
        let t = std::time::Instant::now();
        let h = inst.register_batch(batch.tokens.clone(), batch.b, batch.seq);
        // step_seqs = num sequences ⇒ loss_scale = size/step_seqs = 1.0 (plain mean-loss backward).
        inst.step(h, step, 0, 1, batch.b).expect("da_step");
        let loss = inst
            .metrics()
            .into_iter()
            .rev()
            .find(|(n, _)| n == "loss")
            .map_or(f32::NAN, |(_, v)| v);
        inst.inner_update(step).expect("da_inner_update");
        per_step_secs.push(t.elapsed().as_secs_f64());
        losses.push(loss);
    }
    let step_secs = per_step_secs.iter().sum();
    let final_state = read_state(&inst);
    PathRun {
        losses,
        final_state,
        init_state,
        step_secs,
        per_step_secs,
    }
}

/// Drive the **reference path** from `init` (matched init) for `steps` steps over `batch`.
pub fn drive_reference<B: AutodiffBackend>(
    cfg: &TinyLlamaCfg,
    device: B::Device,
    init: &StateDict,
    batch: &TokenBatch,
    steps: u32,
) -> PathRun {
    let mut model = RefLlama::<B>::from_state_dict(cfg, device, init);
    let mut losses = Vec::new();
    let mut per_step_secs = Vec::new();
    for step in 0..steps {
        let t = std::time::Instant::now();
        losses.push(model.step(&batch.tokens, batch.b as usize, batch.seq as usize, step));
        per_step_secs.push(t.elapsed().as_secs_f64());
    }
    let step_secs = per_step_secs.iter().sum();
    PathRun {
        losses,
        final_state: model.state_dict(cfg),
        init_state: init.clone(),
        step_secs,
        per_step_secs,
    }
}

// -- comparison ---------------------------------------------------------------------------------

/// The parity evidence (recorded in the ledger + throughput doc).
pub struct ParityReport {
    /// `|loss_tabi − loss_ref|` per step.
    pub per_step_delta: Vec<f32>,
    /// The maximum absolute per-element final-weight delta across all params.
    pub final_weight_max_delta: f32,
    /// The tolerance class used as the outer bound.
    pub class: OpClass,
}

/// Assert per-step loss parity + final-weights parity within `class` (Optimizer as the outer bound),
/// returning the achieved deltas for the record. `tabi` and `reference` must have run the same K
/// steps over identical batches from identical init.
pub fn assert_parity(
    tabi: &PathRun,
    reference: &PathRun,
    class: OpClass,
    ctx: &str,
) -> ParityReport {
    assert_eq!(
        tabi.losses.len(),
        reference.losses.len(),
        "{ctx}: step count mismatch"
    );
    // Init must be bit-identical (matched init): a sanity check that the reference started where the
    // tabi path started (else the whole comparison is meaningless).
    for ((n, _, a), (_, _, b)) in tabi
        .init_state
        .tensors
        .iter()
        .zip(reference.init_state.tensors.iter())
    {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                x.to_bits() == y.to_bits(),
                "{ctx}: matched-init mismatch on {n}"
            );
        }
    }

    let mut per_step_delta = Vec::with_capacity(tabi.losses.len());
    for (i, (&lt, &lr)) in tabi.losses.iter().zip(reference.losses.iter()).enumerate() {
        assert!(
            lt.is_finite() && lr.is_finite(),
            "{ctx}[{i}]: non-finite loss"
        );
        per_step_delta.push((lt - lr).abs());
        assert_close(&[lt], &[lr], class, &format!("{ctx} loss step {i}"));
    }

    // Final-weights parity: element-wise within the class, and record the max delta.
    let tol = tol_for(class);
    let mut max_delta = 0.0f32;
    for ((name, _, a), (_, _, b)) in tabi
        .final_state
        .tensors
        .iter()
        .zip(reference.final_state.tensors.iter())
    {
        assert_eq!(a.len(), b.len(), "{ctx}: final weight len mismatch {name}");
        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            max_delta = max_delta.max(diff);
            let bound = tol.atol + tol.rtol * y.abs();
            assert!(
                diff <= bound,
                "{ctx}: final weight {name} |{x} - {y}| = {diff} > {bound}"
            );
        }
    }

    ParityReport {
        per_step_delta,
        final_weight_max_delta: max_delta,
        class,
    }
}
