// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The tiny-llama preset driven end to end through the `sim` backend (TDD §3.4/SDK-5-ish): build →
// (step × H → inner_update) → make_update → ingest across 2 simulated peers, for 3 rounds. Asserts:
//   * loss decreases over the 3 rounds (the model actually learns the synthetic successor task);
//   * the two peers' post-round param digests are bit-identical (the cross-peer agreement property,
//     ABI §7 — here on the det-lane outer step with self-inclusive identical updates).
// sim-only (the wasm guest exercises the identical `TinyLlama` code through the real ABI).
#![cfg(feature = "sim")]

use daemon_train_sdk::models::{TinyLlama, TinyLlamaCfg};
use daemon_train_sdk::sim;
use daemon_train_sdk::{Config, Experiment, StepCtx, UpdatesView};

/// A deterministic synthetic corpus: token `t+1 = (t·7 + 1) mod vocab` — a fixed successor map the
/// model can learn (next-token prediction), identical every call so runs are reproducible.
fn corpus(b: u32, seq: u32, vocab: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((b * seq) as usize);
    for row in 0..b {
        let mut t = (row * 13 + 1) % vocab;
        for _ in 0..seq {
            out.push(t);
            t = (t.wrapping_mul(7).wrapping_add(1)) % vocab;
        }
    }
    out
}

/// Digest of the full canonical state dict (all params, registration order): xxh-free — we compare
/// the raw fp32 master bytes, which is the strongest form of "digests equal".
fn state_digest(names: &[String]) -> Vec<u32> {
    names
        .iter()
        .flat_map(|n| sim::param_master(n).unwrap().into_iter().map(f32::to_bits))
        .collect()
}

fn cfg() -> Config {
    Config::from_value(&TinyLlamaCfg::default())
}

/// One peer: build, run `rounds` rounds of (H inner steps + self-inclusive 2-peer ingest), and
/// return (per-round mean loss, final state digest).
fn run_peer(seed: u64, rounds: u32) -> (Vec<f32>, Vec<u32>) {
    sim::reset(seed);
    let c = TinyLlamaCfg::default();
    let cfg = cfg();
    let h = TinyLlama::manifest(&cfg).steps_per_round;

    let mut exp = TinyLlama::build(&cfg);
    let param_names = state_dict_names(&c);

    let b = 4u32;
    let tokens = corpus(b, c.seq_len, c.vocab);
    let mut inner_step = 0u32;
    let mut round_losses = Vec::new();

    for _round in 0..rounds {
        let mut step_losses = Vec::new();
        for _ in 0..h {
            let batch = sim::make_batch(tokens.clone(), b, c.seq_len);
            let before = sim::metrics().len();
            exp.step(
                &batch,
                &StepCtx {
                    inner_step,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: b,
                },
            );
            // The loss reported by this step is the newest "loss" metric.
            if let Some((_, l)) = sim::metrics()
                .into_iter()
                .skip(before)
                .find(|(n, _)| n == "loss")
            {
                step_losses.push(l);
            }
            exp.inner_update(inner_step);
            inner_step += 1;
        }
        round_losses.push(step_losses.iter().sum::<f32>() / step_losses.len().max(1) as f32);

        // Round end: two self-inclusive peers commit the same update, ingest, snapshot base.
        let u1 = exp.make_update(0);
        sim::stage(&u1);
        let u2 = exp.make_update(0);
        sim::stage(&u2);
        exp.ingest(0, &UpdatesView::with_count(2));
        sim::snapshot_round_base();
    }
    (round_losses, state_digest(&param_names))
}

/// The registered param names in canonical (registration) order for the default config.
fn state_dict_names(c: &TinyLlamaCfg) -> Vec<String> {
    let mut ns = vec!["tok.weight".to_string()];
    for l in 0..c.n_layers {
        for suffix in [
            "attn_norm",
            "wq",
            "wk",
            "wv",
            "wo",
            "ffn_norm",
            "w_gate",
            "w_up",
            "w_down",
        ] {
            ns.push(format!("l{l}.{suffix}"));
        }
    }
    ns.push("norm.weight".to_string());
    ns
}

#[test]
fn tiny_llama_loss_decreases_over_three_rounds() {
    let (losses, _) = run_peer(0xDAE0_7E57, 3);
    eprintln!("tiny-llama per-round mean loss: {losses:?}");
    assert_eq!(losses.len(), 3);
    assert!(
        losses.iter().all(|l| l.is_finite() && *l > 0.0),
        "{losses:?}"
    );
    // The synthetic successor task is learnable; loss must fall from round 0 to round 2.
    assert!(
        losses[2] < losses[0],
        "loss should decrease over rounds: {losses:?}"
    );
}

#[test]
fn tiny_llama_two_peers_agree_bit_exactly() {
    // Two peers with the same seed + same (self-inclusive) updates compute the identical det-lane
    // outer step ⇒ bit-identical post-round state digests (ABI §7 cross-peer agreement).
    let (_, d1) = run_peer(0x5EED, 3);
    let (_, d2) = run_peer(0x5EED, 3);
    assert_eq!(d1, d2, "same inputs ⇒ identical canonical state");
    assert!(!d1.is_empty());
}
