// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// HOST-9 (ABI §4): `da_step` loss-scaling invariance — the same step data split into a different
// number of micro-batches produces the same accumulated gradient (the host owns micro-batch sizing
// via the OOM probe, so the accumulate boundary must be scaling-invariant). Each micro-batch scales
// its loss by size/step_seqs (`StepCtx::loss_scale`), so per-row contributions are identical and
// the accumulation matches within fp32 exactness on the CPU sim. sim-only.
#![cfg(feature = "sim")]

use daemon_train_sdk::sim;
use daemon_train_sdk::{Dtype, Init, Param, StepCtx, Tensor};

const TOTAL: u32 = 8;
const D: u32 = 4;
const VOCAB: u32 = 4;

/// A synthetic linear head: `logits = ones[rows, d] · w[d, vocab]`, cross-entropy against `tgt`.
/// Scaling the loss by `size/step_seqs` (loss_scale) then `backward` accumulates into `w.grad`.
fn micro_step(w: &Param, tgt_tokens: &[u32], step_seqs: u32) {
    let rows = tgt_tokens.len() as u32;
    let x = Tensor::ones(&[rows, D], Dtype::F32);
    let batch = sim::make_batch(tgt_tokens.to_vec(), rows, 1);
    let tgt = batch.tokens().reshape(&[rows]);
    let logits = x.matmul(w.tensor());
    let loss = logits.cross_entropy(&tgt, -1);
    let ctx = StepCtx {
        inner_step: 0,
        mb_index: 0,
        mb_count: 0,
        step_seqs,
    };
    loss.mul_s(ctx.loss_scale(&batch)).backward();
}

/// Accumulated `w.grad` after presenting the 8-row step as `mb_count` micro-batches.
fn grad_for_split(seed: u64, mb_count: u32) -> Vec<f32> {
    sim::reset(seed);
    let w = Param::new("w", &[D, VOCAB], Dtype::F32, Init::Normal, 0.0, 0.5);
    let targets: [u32; TOTAL as usize] = [0, 1, 2, 3, 0, 1, 2, 3];
    let per = (TOTAL / mb_count) as usize;
    for g in targets.chunks(per) {
        micro_step(&w, g, TOTAL); // step_seqs is the full step size for every micro-batch
    }
    sim::param_grad("w").unwrap()
}

#[test]
fn grads_invariant_to_micro_batch_split() {
    let whole = grad_for_split(0xDAE0_7E57, 1); // one micro-batch of 8
    let split4 = grad_for_split(0xDAE0_7E57, 4); // four micro-batches of 2

    assert_eq!(whole.len(), (D * VOCAB) as usize);
    // Same effective update within fp32 exactness (identical per-row contributions, regrouped).
    for (a, b) in whole.iter().zip(split4.iter()) {
        assert!(
            (a - b).abs() <= 1e-6 * (1.0 + a.abs()),
            "grad differs across micro-batch split: {a} vs {b}"
        );
    }
    // And the gradient is non-trivial (the test actually exercised backward).
    assert!(whole.iter().any(|&g| g.abs() > 1e-4));
}
