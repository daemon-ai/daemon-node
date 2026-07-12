// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `tiny-llama` — the reference guest experiment module.
//!
//! A `cdylib` compiled to `wasm32-unknown-unknown` and instantiated by the `daemon-train` host in
//! the wasm sandbox (tensor-ABI spec §5.1). Built through the SDK's [`experiment!`] macro, which
//! wires all `da_*` exports over the [`Experiment`] impl below. The real LLaMA-family decoder
//! (RMSNorm + SwiGLU + RoPE + GQA) lands in Wave 2 — this Wave-1 placeholder registers the canonical
//! state dict and round-trips the lifecycle so the host's `da_abi`/`da_build`/T3 path is exercised.

use daemon_train_sdk::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Cfg {
    d_model: u32,
    vocab: u32,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            d_model: 8,
            vocab: 16,
        }
    }
}

struct TinyLlama {
    params: Vec<Param>, // [tok.weight, norm.weight] — registration order = state dict
}

impl Experiment for TinyLlama {
    fn manifest(_cfg: &Config) -> Manifest {
        // A per-step cadence placeholder (H = 1); the real model reads H from its profile config.
        Manifest::new("tiny-llama", env!("CARGO_PKG_VERSION"), 1)
    }

    fn build(cfg: &Config) -> Self {
        let cfg: Cfg = cfg.parse();
        let tok = Param::new(
            "tok.weight",
            &[cfg.vocab, cfg.d_model],
            Dtype::F32,
            Init::Normal,
            0.0,
            0.02,
        );
        let norm = Param::new(
            "norm.weight",
            &[cfg.d_model],
            Dtype::F32,
            Init::Ones,
            0.0,
            0.0,
        );
        Self {
            params: vec![tok, norm],
        }
    }

    fn step(&mut self, _batch: &Batch, _ctx: &StepCtx) {
        // Placeholder: the Wave-2 model runs the forward/backward here.
    }

    fn inner_update(&mut self, _inner_step: u32) {}

    fn make_update(&mut self, _round: u64) -> UpdateBuilder {
        let mut ub = UpdateBuilder::new();
        for p in &self.params {
            let delta = p.round_base().sub(p.tensor());
            ub.push_tensor(&delta);
        }
        ub
    }

    fn ingest(&mut self, _round: u64, updates: &UpdatesView) {
        let count = updates.len().max(1) as f64;
        for (s, p) in self.params.iter().enumerate() {
            let mut acc = det_zeros(p.shape());
            for i in 0..updates.len() {
                acc = acc.add(&updates.get(i).tensor(s as u32));
            }
            let mean = acc.scale(1.0 / count);
            p.det_reset_to_base();
            p.det_axpy(&mean, -1.0);
        }
    }
}

daemon_train_sdk::experiment!(TinyLlama);
