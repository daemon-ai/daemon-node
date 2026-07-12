// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `det-core` — fixed-order fp32 deterministic kernels.
//!
//! The bit-exact fp32 reference kernels shared verbatim by the host worker (`daemon-train`) and
//! the guest SDK simulator (`daemon-train-sdk`'s `sim` feature) so that "the sim matches the host"
//! is a property of one shared implementation rather than two that happen to agree
//! (`docs/specs/swarm-tensor-abi-spec.md` §5.9; `swarm-training-spec.md` §5.6/§10.1).
//!
//! These implement the **det lane** semantics (ABI §5.9): CPU fp32, fixed evaluation order,
//! bit-exact on every target and every vendor. That property is what makes the swarm's agree-path
//! (decode → clip → aggregate → outer step) cross-peer identical by construction — every reduction
//! here fixes its order explicitly, never leaving it to the compiler or a SIMD reassociation.
//!
//! **Zero third-party dependencies** (nothing beyond `std`): determinism must never hinge on a
//! transitive crate's floating-point behavior. The crate is also `wasm32`-clean (it rides the SDK's
//! `sim` path) and hand-rolls its error type (no `thiserror`).
//!
//! Scalar-cast rule (frozen at Merge 1, ABI §5.9): `f64` hyperparameters are cast to `f32` **inside**
//! the kernel, so the one cast site is shared by host and sim.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Errors returned by the deterministic kernels.
///
/// Hand-rolled to honor this crate's zero-dependency contract (no `thiserror`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetError {
    /// Two operands whose shapes must match did not.
    ShapeMismatch {
        /// The expected element count.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
    /// A length was not an exact multiple of a required divisor (chunking / block layout).
    NotDivisible {
        /// The length that failed to divide.
        len: usize,
        /// The divisor it had to be a multiple of.
        divisor: usize,
    },
    /// A scatter index fell outside its chunk bound.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The exclusive upper bound (the chunk size).
        bound: usize,
    },
    /// An `absmax`-packed decode requested a bit width other than 1/2/4/8.
    UnsupportedBits {
        /// The unsupported width.
        bits: u32,
    },
    /// An aggregation was handed an empty operand list (no shape to infer).
    Empty,
}

impl fmt::Display for DetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            Self::NotDivisible { len, divisor } => {
                write!(f, "length {len} is not a multiple of {divisor}")
            }
            Self::IndexOutOfRange { index, bound } => {
                write!(
                    f,
                    "scatter index {index} out of range for chunk size {bound}"
                )
            }
            Self::UnsupportedBits { bits } => {
                write!(f, "unsupported absmax bit width {bits} (expected 1/2/4/8)")
            }
            Self::Empty => write!(f, "empty operand list: no shape to infer"),
        }
    }
}

impl Error for DetError {}

// -- primitives ---------------------------------------------------------------------------------

/// A left-to-right (index-order) fp32 sum.
///
/// Reduction order is fixed by iteration order, never by the compiler or a SIMD reassociation, so
/// the result is bit-identical on every target (the invariant the tensor ABI's det lane leans on).
#[must_use]
pub fn fixed_order_sum(values: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &v in values {
        acc += v;
    }
    acc
}

/// A fixed-order fp32 dot product of two equal-length slices.
///
/// # Errors
///
/// Returns [`DetError::ShapeMismatch`] when the operands differ in length.
pub fn fixed_order_dot(a: &[f32], b: &[f32]) -> Result<f32, DetError> {
    if a.len() != b.len() {
        return Err(DetError::ShapeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mut acc = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    Ok(acc)
}

// -- det lane: aggregation ----------------------------------------------------------------------

/// Elementwise sum of `xs`, accumulated in **array order** (ABI `det_sum@1`).
///
/// Every tensor in `xs` must share the length of the first. Element `i` of the result is
/// `xs[0][i] + xs[1][i] + …` evaluated in that exact order, so the profiles' post-clip reduce is
/// bit-identical across peers. This is the batch form of [`det_chunk_scatter_add`]'s streaming path
/// (HOST-5: the two agree bit-for-bit).
///
/// # Errors
///
/// [`DetError::Empty`] if `xs` is empty; [`DetError::ShapeMismatch`] if the tensors differ in
/// length.
pub fn det_sum(xs: &[&[f32]]) -> Result<Vec<f32>, DetError> {
    let n = xs.first().ok_or(DetError::Empty)?.len();
    for x in xs {
        if x.len() != n {
            return Err(DetError::ShapeMismatch {
                expected: n,
                got: x.len(),
            });
        }
    }
    let mut acc = vec![0.0_f32; n];
    for x in xs {
        for (a, &v) in acc.iter_mut().zip(x.iter()) {
            *a += v;
        }
    }
    Ok(acc)
}

/// The L2 norm of `x`, fixed-order fp32 (ABI `det_l2norm@1`).
///
/// Squares are accumulated left-to-right, then a single `sqrt`. Because the inputs are det-lane
/// (canonical) the result is identical everywhere — it is one of the two readouts guest logic may
/// safely branch on inside `da_ingest_updates` (ABI §7).
#[must_use]
pub fn det_l2norm(x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for &v in x {
        acc += v * v;
    }
    acc.sqrt()
}

// -- det lane: elementwise ----------------------------------------------------------------------

/// In-place `y += alpha · x` (ABI `det_axpy@1`; the numeric core of `det_axpy_param@1`).
///
/// `alpha` is `f64` at the ABI boundary and cast to `f32` here (the single frozen cast site,
/// ABI §5.9). Accumulation is elementwise in index order.
///
/// # Errors
///
/// [`DetError::ShapeMismatch`] if `y` and `x` differ in length.
pub fn det_axpy(y: &mut [f32], alpha: f64, x: &[f32]) -> Result<(), DetError> {
    if y.len() != x.len() {
        return Err(DetError::ShapeMismatch {
            expected: y.len(),
            got: x.len(),
        });
    }
    let a = alpha as f32;
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
    Ok(())
}

/// Allocating `x · alpha` (ABI `det_scale@1`). `alpha` cast `f64 → f32` per §5.9.
#[must_use]
pub fn det_scale(x: &[f32], alpha: f64) -> Vec<f32> {
    let a = alpha as f32;
    x.iter().map(|&v| v * a).collect()
}

/// Allocating elementwise `a + b` (ABI `det_add@1`).
///
/// # Errors
///
/// [`DetError::ShapeMismatch`] if `a` and `b` differ in length.
pub fn det_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError> {
    binary(a, b, |x, y| x + y)
}

/// Allocating elementwise `a − b` (ABI `det_sub@1`).
///
/// # Errors
///
/// [`DetError::ShapeMismatch`] if `a` and `b` differ in length.
pub fn det_sub(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError> {
    binary(a, b, |x, y| x - y)
}

/// Allocating elementwise `a · b` (ABI `det_mul@1`).
///
/// # Errors
///
/// [`DetError::ShapeMismatch`] if `a` and `b` differ in length.
pub fn det_mul(a: &[f32], b: &[f32]) -> Result<Vec<f32>, DetError> {
    binary(a, b, |x, y| x * y)
}

/// Elementwise sign (ABI `det_sign@1`): `-1 / 0 / +1`; `sign(0) = 0` and `sign(NaN) = 0`.
#[must_use]
pub fn det_sign(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn binary(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> Result<Vec<f32>, DetError> {
    if a.len() != b.len() {
        return Err(DetError::ShapeMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect())
}

// -- det lane: chunked scatter ------------------------------------------------------------------

/// In-place `acc[c·chunk + idx[c,j]] += vals[c,j]` in fixed order (ABI `det_chunk_scatter_add@1`).
///
/// `acc` is any writable dense tensor whose length is a multiple of `chunk`; `vals` and `idx` are
/// the flattened `[n_chunks, k]` per-chunk sparse payload (`n_chunks = acc.len() / chunk`,
/// `k = vals.len() / n_chunks`). Chunks are visited in order and, within a chunk, entries in order —
/// the streaming-ingest hot path (decode one payload → scatter-add → drop), whose fixed order is
/// what makes it bit-equal to [`det_sum`] of the dense decodes (HOST-5).
///
/// # Errors
///
/// [`DetError::NotDivisible`] if `acc.len()` is not a multiple of `chunk` (or `chunk == 0`), or if
/// `vals.len()` is not a multiple of `n_chunks`; [`DetError::ShapeMismatch`] if `vals` and `idx`
/// differ in length; [`DetError::IndexOutOfRange`] if any index is `>= chunk`.
pub fn det_chunk_scatter_add(
    acc: &mut [f32],
    vals: &[f32],
    idx: &[u32],
    chunk: usize,
) -> Result<(), DetError> {
    if chunk == 0 || !acc.len().is_multiple_of(chunk) {
        return Err(DetError::NotDivisible {
            len: acc.len(),
            divisor: chunk,
        });
    }
    if vals.len() != idx.len() {
        return Err(DetError::ShapeMismatch {
            expected: vals.len(),
            got: idx.len(),
        });
    }
    let n_chunks = acc.len() / chunk;
    if n_chunks == 0 {
        // acc is empty: the only consistent payload is an empty one.
        return if vals.is_empty() {
            Ok(())
        } else {
            Err(DetError::ShapeMismatch {
                expected: 0,
                got: vals.len(),
            })
        };
    }
    if !vals.len().is_multiple_of(n_chunks) {
        return Err(DetError::NotDivisible {
            len: vals.len(),
            divisor: n_chunks,
        });
    }
    let k = vals.len() / n_chunks;
    for c in 0..n_chunks {
        let base = c * chunk;
        for j in 0..k {
            let p = c * k + j;
            let index = idx[p] as usize;
            if index >= chunk {
                return Err(DetError::IndexOutOfRange {
                    index,
                    bound: chunk,
                });
            }
            acc[base + index] += vals[p];
        }
    }
    Ok(())
}

/// Allocating dense-from-sparse (ABI `det_chunk_scatter@1`): zeros of length `out_len`, then a
/// single [`det_chunk_scatter_add`].
///
/// # Errors
///
/// As [`det_chunk_scatter_add`].
pub fn det_chunk_scatter(
    vals: &[f32],
    idx: &[u32],
    chunk: usize,
    out_len: usize,
) -> Result<Vec<f32>, DetError> {
    let mut acc = vec![0.0_f32; out_len];
    det_chunk_scatter_add(&mut acc, vals, idx, chunk)?;
    Ok(acc)
}

// -- det lane: absmax decode --------------------------------------------------------------------

/// Decode a blockwise absmax-packed `U8` payload to fp32 (ABI `det_absmax_unpack@1`).
///
/// Layout (frozen at Merge 1, ABI §6.6): per chunk a little-endian `f16` absmax scalar (2 bytes)
/// then `chunk` codes of `bits ∈ {1,2,4,8}` width, packed **LSB-first, chunk-major**, zero-padded
/// to a byte boundary. Dequant is the symmetric linear codebook
/// `value = absmax · (2·code / (2^bits − 1) − 1)`, so for `bits = 2` the four codes decode to
/// `−absmax, −absmax/3, +absmax/3, +absmax`. The `f16` codebook scalar is widened to `f32` exactly.
///
/// # Errors
///
/// [`DetError::UnsupportedBits`] for a width other than 1/2/4/8; [`DetError::NotDivisible`] if the
/// buffer is not an exact number of chunk records (or `chunk == 0`).
pub fn det_absmax_unpack(packed: &[u8], chunk: usize, bits: u32) -> Result<Vec<f32>, DetError> {
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Err(DetError::UnsupportedBits { bits });
    }
    if chunk == 0 {
        return Err(DetError::NotDivisible {
            len: packed.len(),
            divisor: 0,
        });
    }
    let code_bytes = (chunk * bits as usize).div_ceil(8);
    let stride = 2 + code_bytes;
    if !packed.len().is_multiple_of(stride) {
        return Err(DetError::NotDivisible {
            len: packed.len(),
            divisor: stride,
        });
    }
    let n_chunks = packed.len() / stride;
    let max_code = ((1u32 << bits) - 1) as f32;
    let mut out = Vec::with_capacity(n_chunks * chunk);
    for c in 0..n_chunks {
        let off = c * stride;
        let absmax = f16_bits_to_f32(u16::from_le_bytes([packed[off], packed[off + 1]]));
        let codes = &packed[off + 2..off + 2 + code_bytes];
        for e in 0..chunk {
            let code = read_bits_lsb(codes, e * bits as usize, bits);
            let level = (code as f32 / max_code) * 2.0 - 1.0;
            out.push(absmax * level);
        }
    }
    Ok(out)
}

/// Read a `bits`-wide unsigned code starting at `bit_pos`, LSB-first, from a byte slice.
fn read_bits_lsb(bytes: &[u8], bit_pos: usize, bits: u32) -> u32 {
    let mut v = 0_u32;
    for b in 0..bits as usize {
        let abs = bit_pos + b;
        let bit = (bytes[abs / 8] >> (abs % 8)) & 1;
        v |= (bit as u32) << b;
    }
    v
}

/// IEEE-754 half → single. Hand-rolled (zero-dep); exact widening.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0_f32 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    match exp {
        0 => sign * (mant as f32) * 2.0_f32.powi(-24), // zero / subnormal
        0x1f => {
            if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + (mant as f32) / 1024.0) * 2.0_f32.powi(exp as i32 - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture seed the TDD plan pins for any generated det-lane vector (§1.2 RNG caveat).
    const SEED: u64 = 0xDAE0_7E57;

    /// A tiny deterministic xorshift64* — used only to shuffle operand ORDER in tests, never to
    /// generate consensus-relevant data. Pinned to [`SEED`] so runs are reproducible.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn shuffle<T>(&mut self, s: &mut [T]) {
            for i in (1..s.len()).rev() {
                let j = (self.next_u64() % (i as u64 + 1)) as usize;
                s.swap(i, j);
            }
        }
    }

    #[test]
    fn sum_is_index_ordered() {
        assert_eq!(fixed_order_sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(fixed_order_sum(&[]), 0.0);
    }

    #[test]
    fn dot_checks_shapes() {
        assert_eq!(fixed_order_dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
        assert_eq!(
            fixed_order_dot(&[1.0], &[1.0, 2.0]),
            Err(DetError::ShapeMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn det_sum_golden() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [10.0_f32, 20.0, 30.0];
        let c = [0.5_f32, 0.5, 0.5];
        assert_eq!(det_sum(&[&a, &b, &c]).unwrap(), vec![11.5, 22.5, 33.5]);
        assert_eq!(det_sum(&[] as &[&[f32]]), Err(DetError::Empty));
        assert_eq!(
            det_sum(&[&a[..], &b[..2]]),
            Err(DetError::ShapeMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn det_sum_order_is_fixed_not_reassociated() {
        // A payload where left-to-right differs bitwise from a reassociated (pairwise) order — the
        // whole point of the det lane. `det_sum` must always take the array order.
        let big = 1.0e8_f32;
        let xs: [&[f32]; 3] = [&[big], &[1.0], &[-big]];
        // (big + 1) - big  ==  0  in fp32 (the 1 is lost).  1 + (big - big) == 1.
        assert_eq!(det_sum(&xs).unwrap(), vec![0.0]);
        // Repeated runs are bit-identical.
        assert_eq!(det_sum(&xs).unwrap()[0].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn l2norm_golden_and_branchable() {
        assert_eq!(det_l2norm(&[3.0, 4.0]), 5.0);
        assert_eq!(det_l2norm(&[]), 0.0);
        // Deterministic ⇒ safe to branch on.
        assert!(det_l2norm(&[0.6, 0.8]) < 1.0 + f32::EPSILON);
    }

    #[test]
    fn axpy_and_scale() {
        let mut y = vec![1.0_f32, 2.0, 3.0];
        det_axpy(&mut y, 2.0, &[10.0, 10.0, 10.0]).unwrap();
        assert_eq!(y, vec![21.0, 22.0, 23.0]);
        assert_eq!(det_scale(&[1.0, -2.0, 3.0], -1.0), vec![-1.0, 2.0, -3.0]);
        // f64→f32 cast happens inside the kernel.
        assert_eq!(det_scale(&[1.0], 0.5), vec![0.5]);
        assert_eq!(
            det_axpy(&mut [0.0], 1.0, &[0.0, 0.0]),
            Err(DetError::ShapeMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn elementwise_and_sign() {
        assert_eq!(det_add(&[1.0, 2.0], &[3.0, 4.0]).unwrap(), vec![4.0, 6.0]);
        assert_eq!(det_sub(&[3.0, 4.0], &[1.0, 1.0]).unwrap(), vec![2.0, 3.0]);
        assert_eq!(det_mul(&[2.0, 3.0], &[4.0, 5.0]).unwrap(), vec![8.0, 15.0]);
        assert_eq!(
            det_sign(&[-2.0, 0.0, 5.0, f32::NAN]),
            vec![-1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn chunk_scatter_add_golden() {
        // Two chunks of size 4, k=2 entries each.
        let mut acc = vec![0.0_f32; 8];
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let idx = [0_u32, 3, 1, 2];
        det_chunk_scatter_add(&mut acc, &vals, &idx, 4).unwrap();
        assert_eq!(acc, vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0]);

        // The allocating form matches.
        assert_eq!(det_chunk_scatter(&vals, &idx, 4, 8).unwrap(), acc);
    }

    #[test]
    fn chunk_scatter_add_errors() {
        assert_eq!(
            det_chunk_scatter_add(&mut [0.0; 6], &[1.0], &[0], 4),
            Err(DetError::NotDivisible { len: 6, divisor: 4 })
        );
        assert_eq!(
            det_chunk_scatter_add(&mut [0.0; 4], &[1.0, 2.0], &[0], 4),
            Err(DetError::ShapeMismatch {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            det_chunk_scatter_add(&mut [0.0; 4], &[1.0], &[4], 4),
            Err(DetError::IndexOutOfRange { index: 4, bound: 4 })
        );
    }

    /// HOST-5: streaming (scatter_add each decode, then drop) equals batch (`det_sum` of the dense
    /// decodes) **bit-for-bit**, for the record-order staged set.
    #[test]
    fn streaming_equals_batch_aggregation() {
        let chunk = 4;
        let out_len = 8; // 2 chunks
                         // Three "peer" sparse payloads (record order), each [n_chunks, k=2].
        let payloads: [(Vec<f32>, Vec<u32>); 3] = [
            (vec![0.25, 0.5, 1.5, 2.5], vec![0, 1, 2, 3]),
            (vec![-0.1, 0.2, 0.3, -0.4], vec![1, 3, 0, 2]),
            (vec![9.0, 8.0, 7.0, 6.0], vec![3, 0, 1, 1]),
        ];

        // Batch: densify each payload, then det_sum in record order.
        let dense: Vec<Vec<f32>> = payloads
            .iter()
            .map(|(v, i)| det_chunk_scatter(v, i, chunk, out_len).unwrap())
            .collect();
        let batch = det_sum(&dense.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();

        // Streaming: one accumulator, scatter-add each in record order, drop decodes.
        let mut acc = vec![0.0_f32; out_len];
        for (v, i) in &payloads {
            det_chunk_scatter_add(&mut acc, v, i, chunk).unwrap();
        }

        assert_eq!(acc, batch);
        for (s, b) in acc.iter().zip(batch.iter()) {
            assert_eq!(
                s.to_bits(),
                b.to_bits(),
                "streaming must equal batch bitwise"
            );
        }
    }

    /// Order is an **explicit argument** of the det-lane reduce, so there is no hidden
    /// nondeterminism: the same argument order always yields the same bits, and shuffling only the
    /// *physical storage* (while re-presenting the tensors in the record order the API demands)
    /// changes nothing. A control shows a genuinely different argument order is a different call
    /// (fp32 addition is non-associative), which is exactly why the host stages in record order
    /// (§5.11) and the guest never gets to reorder.
    #[test]
    fn sum_order_is_an_explicit_argument() {
        // Values crafted so that record order and reverse order round differently in fp32:
        // 1.0 survives when added before the large magnitude, and is lost when added after.
        let big = 3.0e7_f32; // ulp ≈ 4 at this magnitude
        let tensors = [[1.0_f32], [big], [-big]];

        // Store the tensors in a shuffled physical Vec, tagged with their record index, then
        // re-present them to det_sum in record order. The aggregate is bit-identical to the
        // straight call — the API fixed the order, not the memory layout.
        let mut tagged: Vec<(usize, &[f32])> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.as_slice()))
            .collect();
        Rng::new(SEED).shuffle(&mut tagged);
        tagged.sort_by_key(|(i, _)| *i); // record order, regardless of storage order
        let via_storage_shuffle = det_sum(&tagged.iter().map(|(_, t)| *t).collect::<Vec<_>>());

        let record_order: Vec<&[f32]> = tensors.iter().map(|t| t.as_slice()).collect();
        let canonical = det_sum(&record_order).unwrap();
        assert_eq!(via_storage_shuffle.unwrap(), canonical);

        // Repeated runs are bit-identical.
        for _ in 0..4 {
            assert_eq!(det_sum(&record_order).unwrap(), canonical);
        }

        // Control: a genuinely different argument order is allowed to diverge bitwise.
        let reversed: Vec<&[f32]> = tensors.iter().rev().map(|t| t.as_slice()).collect();
        let other = det_sum(&reversed).unwrap();
        assert_ne!(
            other[0].to_bits(),
            canonical[0].to_bits(),
            "fp32 sum is order-dependent; the det lane pins the (record) order by construction"
        );
    }

    #[test]
    fn absmax_unpack_2bit_golden() {
        // One chunk of 4 codes, bits=2 ⇒ code_bytes = 1. absmax = 1.0 (f16 bits 0x3C00).
        // codes 0,1,2,3 packed LSB-first into one byte: 0b11_10_01_00 = 0xE4.
        let packed = [0x00, 0x3C, 0xE4];
        let out = det_absmax_unpack(&packed, 4, 2).unwrap();
        // dequant = absmax * (2*code/3 - 1)  →  -1, -1/3, 1/3, 1
        assert_eq!(out[0], -1.0);
        assert!((out[1] - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((out[2] - (1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn absmax_unpack_1bit_and_8bit() {
        // 1-bit: absmax 2.0 (f16 0x4000), 8 codes packed one per bit: 0b0000_0011 → codes
        // 1,1,0,0,0,0,0,0 ⇒ +2, +2, -2, -2, -2, -2, -2, -2.
        let one = det_absmax_unpack(&[0x00, 0x40, 0x03], 8, 1).unwrap();
        assert_eq!(one[0], 2.0);
        assert_eq!(one[1], 2.0);
        assert_eq!(one[2], -2.0);

        // 8-bit: absmax 1.0, one chunk of 1 code = 255 ⇒ +absmax (full scale).
        let hi = det_absmax_unpack(&[0x00, 0x3C, 0xFF], 1, 8).unwrap();
        assert_eq!(hi[0], 1.0);
        // code 0 ⇒ -absmax.
        let lo = det_absmax_unpack(&[0x00, 0x3C, 0x00], 1, 8).unwrap();
        assert_eq!(lo[0], -1.0);
    }

    #[test]
    fn absmax_unpack_errors() {
        assert_eq!(
            det_absmax_unpack(&[0; 3], 4, 3),
            Err(DetError::UnsupportedBits { bits: 3 })
        );
        // 4 codes @ 2 bits = 1 code byte → stride 3; a 4-byte buffer is not a whole record.
        assert_eq!(
            det_absmax_unpack(&[0; 4], 4, 2),
            Err(DetError::NotDivisible { len: 4, divisor: 3 })
        );
    }

    #[test]
    fn repeated_runs_are_bit_identical() {
        let payload: Vec<f32> = (0..97).map(|n| (n as f32).sin() * 1234.5).collect();
        let first = det_l2norm(&payload);
        for _ in 0..8 {
            assert_eq!(det_l2norm(&payload).to_bits(), first.to_bits());
        }
        let s1 = det_scale(&payload, core::f64::consts::PI);
        let s2 = det_scale(&payload, core::f64::consts::PI);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
