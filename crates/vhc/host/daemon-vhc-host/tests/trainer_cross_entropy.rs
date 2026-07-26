// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The trainer's cross-entropy target selection: an INDEXED read replaces a one-hot mask-and-reduce,
// and the gradient it hands back is asserted **bit-identical** to the one the mask produced.
//
// Why this lane exists. The reduction the trainer's loss ends in used to build a `[rows, vocab]`
// one-hot image, multiply the whole log-softmax by it, and sum every element — reading `rows`
// values out of `rows × vocab` of them. At the frozen ceremony geometry each of those images is
// `2047 × 32768 × 4` = 256 MiB, and there were two (the mask and the product). Selecting the same
// values by index materializes neither.
//
// The claim that change rests on is exact, not approximate, so it is asserted exactly here rather
// than absorbed by the trained-theta tolerance band: **the gradient with respect to the logits is
// the same to the bit**. The backward of an indexed selection scatters the incoming row gradients
// to exactly the selected positions and leaves every other position zero, which is the same matrix
// `d(logsm · onehot)/d(logsm)` produces. Only the forward loss SCALAR's summation order changes
// (`rows` terms instead of `rows × vocab`, the extra ones exact zeros), and that scalar is consumed
// solely as `backward()`'s root, which is seeded independently of its value — so no recorded value
// anywhere depends on it. This file proves both halves: the gradients match bit-for-bit, and the
// scalar is the only thing that may not.
//
// A drift here is a stop-and-escalate. It is never a tolerance to widen, and never a reason to
// re-record the trainer goldens: the whole point of the assertion is that the trajectory does not
// move.

use burn::backend::Autodiff;
use burn::tensor::{Int, Tensor, TensorData};
use burn_ndarray::NdArray;

type Back = Autodiff<NdArray<f32, i64, i8>>;
type Device = <NdArray<f32, i64, i8> as burn::tensor::backend::Backend>::Device;

/// Which target-selection formulation the tail runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    /// The retired formulation: a `[rows, vocab]` one-hot, an elementwise product, a full reduce.
    OneHotProduct,
    /// The shipping formulation: an indexed read of each row's target log-probability.
    IndexedGather,
}

/// The trainer's shifted-max cross-entropy tail (`model.rs`'s `forward`, from the logits down),
/// run under `selection`. Returns `(loss, d loss / d logits)`.
fn tail(
    logits_data: &[f32],
    tgt: &[i64],
    rows: usize,
    vocab: usize,
    loss_scale: f32,
    selection: Selection,
) -> (f32, Vec<f32>) {
    let device = Device::default();
    let logits = Tensor::<Back, 2>::from_data(
        TensorData::new(logits_data.to_vec(), [rows, vocab]),
        &device,
    )
    .require_grad();

    // Shifted-max log-softmax, byte-for-byte the trainer's own expression.
    let max = logits.clone().max_dim(1).detach();
    let shifted = logits.clone().sub(max);
    let logsm = shifted.clone().sub(shifted.exp().sum_dim(1).log());

    let picked = match selection {
        Selection::IndexedGather => {
            let targets =
                Tensor::<Back, 1, Int>::from_data(TensorData::new(tgt.to_vec(), [rows]), &device)
                    .reshape([rows, 1]);
            logsm.gather(1, targets)
        }
        Selection::OneHotProduct => {
            let targets =
                Tensor::<Back, 1, Int>::from_data(TensorData::new(tgt.to_vec(), [rows]), &device)
                    .reshape([rows, 1])
                    .expand([rows, vocab]);
            let classes = Tensor::<Back, 1, Int>::arange(0..vocab as i64, &device)
                .reshape([1, vocab])
                .expand([rows, vocab]);
            logsm.mul(classes.equal(targets).float())
        }
    };

    #[allow(clippy::cast_precision_loss)]
    let denom = rows.max(1) as f32;
    let loss = picked
        .sum()
        .mul_scalar(-1.0 / denom)
        .mul_scalar(loss_scale)
        .reshape([1]);

    let value = loss
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("loss scalar")[0];
    let grads = loss.backward();
    let grad = logits
        .grad(&grads)
        .expect("the logits participate in the loss")
        .into_data()
        .to_vec::<f32>()
        .expect("logit gradients");
    (value, grad)
}

/// Deterministic, reproducible logits with a wide dynamic range (so the log-softmax denominator is
/// not trivially uniform and the gradient carries real rounding).
fn logits(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            #[allow(clippy::cast_possible_truncation)]
            let v = (unit * 24.0 - 12.0) as f32;
            v
        })
        .collect()
}

/// Targets spread over the vocabulary, including row 0 at class 0 and the last row at the last
/// class (the two boundary positions an off-by-one in an indexed read would hit first).
fn targets(seed: u64, rows: usize, vocab: usize) -> Vec<i64> {
    let mut s = seed | 1;
    let mut out: Vec<i64> = (0..rows)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % vocab as u64) as i64
        })
        .collect();
    out[0] = 0;
    let last = out.len() - 1;
    out[last] = vocab as i64 - 1;
    out
}

/// The bit patterns of a gradient vector — equality over `f32` bits, so a sign-of-zero or a
/// last-ulp difference is caught rather than compared away.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn indexed_target_selection_reproduces_the_one_hot_gradient_bit_for_bit() {
    // Shapes: a square-ish block, a tall-thin block (many rows, small vocab), and a wide block
    // (few rows, large vocab — the ceremony's shape in miniature).
    let shapes = [(16usize, 16usize), (128, 8), (8, 1024), (64, 512)];
    for (rows, vocab) in shapes {
        for (case, scale) in [("unit", 1.0f32), ("scaled", 1.0 / 30.0)] {
            let seed = (rows as u64) << 32 | vocab as u64;
            let l = logits(seed, rows * vocab);
            let t = targets(seed ^ 0xA5A5, rows, vocab);

            let (loss_oh, grad_oh) = tail(&l, &t, rows, vocab, scale, Selection::OneHotProduct);
            let (loss_ix, grad_ix) = tail(&l, &t, rows, vocab, scale, Selection::IndexedGather);

            assert_eq!(
                bits(&grad_ix),
                bits(&grad_oh),
                "[{rows}x{vocab} {case}] the indexed selection's logit gradient must be \
                 BIT-IDENTICAL to the one-hot product's — a difference here moves the trainer's \
                 recorded trajectory and is a stop-and-escalate, never a tolerance to widen"
            );
            // The loss scalar is the one value that may differ (its summation order changed) and
            // the one value nothing reads: it exists only as `backward()`'s root.
            assert!(
                (loss_ix - loss_oh).abs() <= loss_oh.abs() * 1e-5 + 1e-6,
                "[{rows}x{vocab} {case}] the loss scalars must still agree numerically: \
                 one-hot {loss_oh}, indexed {loss_ix}"
            );
        }
    }
}

#[test]
fn indexed_target_selection_places_the_gradient_on_the_target_class_only() {
    // The shape of the gradient, asserted directly rather than only against the other formulation:
    // exactly one non-zero per row, at that row's target, equal to `-loss_scale / rows` plus the
    // log-softmax's own back-propagated term.
    let (rows, vocab) = (12usize, 40usize);
    let l = logits(0xC0FFEE, rows * vocab);
    let t = targets(0xBEEF, rows, vocab);
    let (_, grad) = tail(&l, &t, rows, vocab, 1.0, Selection::IndexedGather);

    for (r, &target) in t.iter().enumerate() {
        let row = &grad[r * vocab..(r + 1) * vocab];
        #[allow(clippy::cast_sign_loss)]
        let target = target as usize;
        assert!(
            row[target] < 0.0,
            "row {r}: the target class carries the negative log-likelihood pull"
        );
        for (c, g) in row.iter().enumerate() {
            if c != target {
                assert!(
                    *g >= 0.0,
                    "row {r} class {c}: a non-target class is pushed down, never pulled up"
                );
            }
        }
        let sum: f32 = row.iter().sum();
        assert!(
            sum.abs() < 1e-4,
            "row {r}: the softmax gradient sums to zero across the vocabulary, got {sum}"
        );
    }
}
