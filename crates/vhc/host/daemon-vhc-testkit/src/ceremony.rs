// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The FROZEN fleet-ceremony trainer model configuration — the production validation tier.
//!
//! This is the single source of truth for the ceremony model geometry (the program spec's
//! ceremony section cites this module by path and restates the parameters for the reader).
//! It is a real multi-layer TinyLlama-class decoder sized for 24 GB-class GPU peers — a
//! ~0.79 B-parameter model whose training working set (~11.7 GiB of fp32 device state at
//! 16 B/param for master + gradient + both AdamW moments, before activations) deliberately
//! exceeds a 16 GiB unified-memory box — explicitly NOT the 64-dim structural acceptance tier
//! ([`crate::live_genesis`]) and NOT sized down to the smallest fleet peer.
//!
//! FROZEN: these values are ceremony inputs. Changing any of them re-derives the genesis, the
//! matched init, and every fleet-preflight sizing check — treat any edit as a new ceremony
//! candidate, never a tweak. The corpus pin (manifest hash + tokenizer) is frozen separately
//! when the ceremony corpus is published; [`ceremony_model_value`] deliberately builds only the
//! `model` half of the trainer config so the corpus half cannot be guessed at here.
//!
//! Nothing in the acceptance suite consumes this module (the acceptance tier stays 64-dim); it
//! is tracked so the ceremony genesis is authored from a reviewed, pinned artifact rather than
//! transcribed prose.

use ciborium::value::Value;

/// Residual width.
pub const CEREMONY_D_MODEL: u32 = 1536;
/// Transformer blocks (real multi-layer depth — the acceptance tier runs 2).
pub const CEREMONY_N_LAYERS: u32 = 24;
/// Attention heads (`n_kv_heads == n_heads` — the guest runs full MHA).
pub const CEREMONY_N_HEADS: u32 = 24;
/// Per-head width (`n_heads · head_dim == d_model`).
pub const CEREMONY_HEAD_DIM: u32 = 64;
/// Vocabulary (tied input/output embedding): a power-of-two ceiling over the ceremony
/// tokenizer's id space (the in-guest `token % vocab` clamp is the identity for well-formed
/// corpora, the established discipline).
pub const CEREMONY_VOCAB: u32 = 32_768;
/// Sequence length.
pub const CEREMONY_SEQ_LEN: u32 = 2_048;
/// SwiGLU hidden = `ffn_mult · d_model` (= 4608).
pub const CEREMONY_FFN_MULT: u32 = 3;

/// AdamW learning rate.
pub const CEREMONY_LR: f64 = 3.0e-4;
/// AdamW β₁.
pub const CEREMONY_BETA1: f64 = 0.9;
/// AdamW β₂.
pub const CEREMONY_BETA2: f64 = 0.95;
/// AdamW ε.
pub const CEREMONY_ADAM_EPS: f64 = 1.0e-8;
/// AdamW decoupled weight decay.
pub const CEREMONY_WD: f64 = 0.1;
/// RoPE base.
pub const CEREMONY_ROPE_THETA: f64 = 10_000.0;
/// RMSNorm epsilon.
pub const CEREMONY_RMSNORM_EPS: f64 = 1.0e-5;

/// The frozen per-parameter numels of the ceremony geometry, in the guest's registration order
/// (token embedding; per block: attn-norm, wq, wk, wv, wo, ffn-norm, w1, w3, w2; final norm) —
/// the same layout arithmetic the trainer guest derives from its `ModelCfg`.
#[must_use]
pub fn ceremony_param_numels() -> Vec<usize> {
    let d = CEREMONY_D_MODEL as usize;
    let qdim = (CEREMONY_N_HEADS * CEREMONY_HEAD_DIM) as usize;
    let hidden = (CEREMONY_FFN_MULT * CEREMONY_D_MODEL) as usize;
    let vocab = CEREMONY_VOCAB as usize;
    let mut out = vec![vocab * d];
    for _ in 0..CEREMONY_N_LAYERS {
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

/// The frozen ceremony trainer `model` config map (raw canonical-CBOR value against the trainer
/// guest's documented `ModelCfg` schema) — the reviewed artifact the ceremony genesis authoring
/// embeds verbatim. The corpus/`live` half is composed at genesis authoring, once the ceremony
/// corpus manifest is published and pinned.
#[must_use]
pub fn ceremony_model_value() -> Value {
    let text = |s: &str| Value::Text(s.into());
    let uint = |v: u32| Value::Integer(u64::from(v).into());
    Value::Map(vec![
        (text("d_model"), uint(CEREMONY_D_MODEL)),
        (text("n_layers"), uint(CEREMONY_N_LAYERS)),
        (text("n_heads"), uint(CEREMONY_N_HEADS)),
        (text("head_dim"), uint(CEREMONY_HEAD_DIM)),
        (text("vocab"), uint(CEREMONY_VOCAB)),
        (text("seq_len"), uint(CEREMONY_SEQ_LEN)),
        (text("ffn_mult"), uint(CEREMONY_FFN_MULT)),
        (text("rope_theta"), Value::Float(CEREMONY_ROPE_THETA)),
        (text("rmsnorm_eps"), Value::Float(CEREMONY_RMSNORM_EPS)),
        (text("lr"), Value::Float(CEREMONY_LR)),
        (text("beta1"), Value::Float(CEREMONY_BETA1)),
        (text("beta2"), Value::Float(CEREMONY_BETA2)),
        (text("adam_eps"), Value::Float(CEREMONY_ADAM_EPS)),
        (text("wd"), Value::Float(CEREMONY_WD)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen geometry's arithmetic, pinned: ~0.79 B parameters, ~11.7 GiB of fp32 training
    /// state at 16 B/param — inside a 24 GiB device budget with activation headroom, beyond a
    /// 16 GiB unified-memory box. A drift in any constant moves these sums and fails here.
    #[test]
    fn ceremony_geometry_is_frozen() {
        assert_eq!(
            CEREMONY_N_HEADS * CEREMONY_HEAD_DIM,
            CEREMONY_D_MODEL,
            "full-MHA width identity"
        );
        let numels = ceremony_param_numels();
        assert_eq!(
            numels.len(),
            (2 + 9 * CEREMONY_N_LAYERS) as usize,
            "embedding + 9 params/block + final norm"
        );
        let total: usize = numels.iter().sum();
        assert_eq!(total, 786_507_264, "the frozen parameter count");
        let state_bytes = total as u64 * 16; // fp32 master + grad + AdamW m + v
        assert!(
            state_bytes > 11 * (1 << 30),
            "the tier is NOT sized down to a 16 GiB-class peer"
        );
        assert!(
            state_bytes < 16 * (1 << 30),
            "the fp32 training state leaves activation headroom on a 24 GiB device"
        );
        // The largest single tensor (the tied embedding) stays far under per-buffer ceilings.
        let largest = numels.iter().max().copied().unwrap_or(0) as u64 * 4;
        assert!(largest < 1 << 30, "largest tensor under 1 GiB");
    }

    #[test]
    fn ceremony_model_value_round_trips_canonically() {
        let v = ceremony_model_value();
        let bytes = daemon_vhc_proto::to_canonical_vec(&v).expect("canonical encode");
        let back: Value = ciborium::de::from_reader(bytes.as_slice()).expect("decode");
        // Canonical encoding reorders map keys; the round trip preserves the ENTRIES.
        let pairs = |val: &Value| -> Vec<(String, Value)> {
            let Value::Map(entries) = val else {
                panic!("model config is a map");
            };
            let mut out: Vec<(String, Value)> = entries
                .iter()
                .map(|(k, v)| match k {
                    Value::Text(t) => (t.clone(), v.clone()),
                    other => panic!("non-text key {other:?}"),
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        assert_eq!(pairs(&v), pairs(&back));
        assert_eq!(pairs(&v).len(), 14, "the frozen field set");
    }
}
