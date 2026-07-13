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
    /// A per-chunk top-k requested more entries than the chunk holds.
    KTooLarge {
        /// The requested count.
        k: usize,
        /// The chunk size it exceeded.
        chunk: usize,
    },
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
            Self::KTooLarge { k, chunk } => {
                write!(f, "top-k k={k} exceeds chunk size {chunk}")
            }
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

/// Encode a blockwise fp32 payload to the absmax-packed `U8` layout (ABI `absmax_pack@1`, §6.6).
///
/// The exact inverse-direction partner of [`det_absmax_unpack`]: `x` is viewed flattened as
/// `[numel/chunk, chunk]`; for each chunk the per-chunk `absmax = max|value|` is stored as a
/// little-endian `f16` codebook scalar, then each value is quantized to the nearest symmetric code
/// `code = round((value/absmax + 1)/2 · (2^bits − 1))` (clamped to `[0, 2^bits − 1]`), packed
/// LSB-first, chunk-major, zero-padded to a byte boundary. A zero-absmax chunk encodes the midpoint
/// code (which [`det_absmax_unpack`] decodes back to `0`). `bits ∈ {1,2,4,8}`.
///
/// The pack→unpack round-trip is exact up to the codebook quantization; re-packing a decoded payload
/// is a fixed point (idempotent), which is what makes it a byte-stable wire form.
///
/// # Errors
///
/// [`DetError::UnsupportedBits`] for a width other than 1/2/4/8; [`DetError::NotDivisible`] if
/// `x.len()` is not a multiple of `chunk` (or `chunk == 0`).
pub fn absmax_pack(x: &[f32], chunk: usize, bits: u32) -> Result<Vec<u8>, DetError> {
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Err(DetError::UnsupportedBits { bits });
    }
    if chunk == 0 || !x.len().is_multiple_of(chunk) {
        return Err(DetError::NotDivisible {
            len: x.len(),
            divisor: chunk,
        });
    }
    let code_bytes = (chunk * bits as usize).div_ceil(8);
    let stride = 2 + code_bytes;
    let n_chunks = x.len() / chunk;
    let max_code = (1u32 << bits) - 1;
    let max_code_f = max_code as f32;
    let mut out = vec![0u8; n_chunks * stride];
    for c in 0..n_chunks {
        let block = &x[c * chunk..(c + 1) * chunk];
        let absmax = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        // Store the f16 absmax, then read it back so codes quantize against the *stored* scale (the
        // decoder only ever sees the f16 value) — keeps pack∘unpack∘pack a fixed point.
        let absmax_h = f32_to_f16_bits(absmax);
        let off = c * stride;
        out[off..off + 2].copy_from_slice(&absmax_h.to_le_bytes());
        let absmax_stored = f16_bits_to_f32(absmax_h);
        let codes = &mut out[off + 2..off + 2 + code_bytes];
        for (e, &v) in block.iter().enumerate() {
            let code = if absmax_stored == 0.0 {
                // Midpoint: decodes to 0 for even max_code; for odd bit widths the nearest even
                // split rounds to the lower-middle code, still the smallest-magnitude level.
                max_code / 2
            } else {
                let level = v / absmax_stored; // in [-1, 1]
                let q = ((level + 1.0) * 0.5 * max_code_f).round();
                (q.clamp(0.0, max_code_f)) as u32
            };
            write_bits_lsb(codes, e * bits as usize, bits, code);
        }
    }
    Ok(out)
}

/// Per-chunk top-k by magnitude (ABI `topk_chunk@1`, §5.8).
///
/// `x` is viewed flattened as `[numel/chunk, chunk]`; each chunk contributes its `k` largest-|value|
/// entries. Selection order within a chunk is descending magnitude, ties broken by ascending index
/// — a total order, so the result is deterministic. Returns `(values, indices)` each of length
/// `n_chunks · k`, with `values[c·k + j]` the signed value and `indices[c·k + j]` its position
/// **within** the chunk (`0..chunk`), matching [`det_chunk_scatter_add`]'s per-chunk index space.
///
/// # Errors
///
/// [`DetError::NotDivisible`] if `x.len()` is not a multiple of `chunk` (or `chunk == 0`);
/// [`DetError::KTooLarge`] if `k > chunk`.
pub fn topk_chunk(x: &[f32], chunk: usize, k: usize) -> Result<(Vec<f32>, Vec<u32>), DetError> {
    if chunk == 0 || !x.len().is_multiple_of(chunk) {
        return Err(DetError::NotDivisible {
            len: x.len(),
            divisor: chunk,
        });
    }
    if k > chunk {
        return Err(DetError::KTooLarge { k, chunk });
    }
    let n_chunks = x.len() / chunk;
    let mut vals = Vec::with_capacity(n_chunks * k);
    let mut idx = Vec::with_capacity(n_chunks * k);
    for c in 0..n_chunks {
        let block = &x[c * chunk..(c + 1) * chunk];
        // Fixed total order: larger magnitude first, then lower index. Sort a small index vector.
        let mut order: Vec<u32> = (0..chunk as u32).collect();
        order.sort_by(|&a, &b| {
            let (ma, mb) = (block[a as usize].abs(), block[b as usize].abs());
            // Descending magnitude; NaN-safe via total_cmp on the negated compare.
            mb.total_cmp(&ma).then(a.cmp(&b))
        });
        for &o in order.iter().take(k) {
            vals.push(block[o as usize]);
            idx.push(o);
        }
    }
    Ok((vals, idx))
}

/// Orthonormal 2-D DCT-II over `tile × tile` blocks (ABI `dct2@1`, §5.8).
///
/// `x` is viewed as a sequence of `tile × tile` row-major blocks (`x.len()` a multiple of `tile²`);
/// each block `X` is transformed to `Y = C · X · Cᵀ` where `C` is the orthonormal DCT-II matrix
/// (`C[k][j] = α(k)·cos(π(2j+1)k / 2·tile)`, `α(0)=√(1/tile)`, `α(k>0)=√(2/tile)`). Intermediate
/// accumulation is `f64` (one shared reference), cast to `f32` on store — deterministic on one
/// implementation, so the sim and the host CPU fake agree byte-for-byte (HOST-1).
///
/// # Errors
///
/// [`DetError::NotDivisible`] if `tile == 0` or `x.len()` is not a multiple of `tile²`.
pub fn dct2(x: &[f32], tile: usize) -> Result<Vec<f32>, DetError> {
    transform2(x, tile, false)
}

/// The inverse orthonormal 2-D DCT (DCT-III), `X = Cᵀ · Y · C` (ABI `idct2@1` / `det_idct2@1`).
///
/// Reconstructs [`dct2`]'s input to within fp32 rounding (≤ ~1e-5 relative for the specced tile
/// sizes). Same fixed-order `f64` accumulation as the forward transform.
///
/// # Errors
///
/// As [`dct2`].
pub fn idct2(x: &[f32], tile: usize) -> Result<Vec<f32>, DetError> {
    transform2(x, tile, true)
}

/// Shared 2-D DCT engine: `inverse=false` applies `C·X·Cᵀ`, `inverse=true` applies `Cᵀ·Y·C`.
fn transform2(x: &[f32], tile: usize, inverse: bool) -> Result<Vec<f32>, DetError> {
    let block = tile
        .checked_mul(tile)
        .filter(|&b| b != 0)
        .ok_or(DetError::NotDivisible {
            len: x.len(),
            divisor: 0,
        })?;
    if !x.len().is_multiple_of(block) {
        return Err(DetError::NotDivisible {
            len: x.len(),
            divisor: block,
        });
    }
    let c = dct_matrix(tile);
    let n_blocks = x.len() / block;
    let mut out = vec![0.0_f32; x.len()];
    let mut tmp = vec![0.0_f64; block];
    for bi in 0..n_blocks {
        let src = &x[bi * block..(bi + 1) * block];
        let dst = &mut out[bi * block..(bi + 1) * block];
        if inverse {
            // M[i][b] = Σ_a C[a][i]·Y[a][b]
            for i in 0..tile {
                for b in 0..tile {
                    let mut acc = 0.0_f64;
                    for a in 0..tile {
                        acc += c[a * tile + i] * f64::from(src[a * tile + b]);
                    }
                    tmp[i * tile + b] = acc;
                }
            }
            // X[i][j] = Σ_b M[i][b]·C[b][j]
            for i in 0..tile {
                for j in 0..tile {
                    let mut acc = 0.0_f64;
                    for b in 0..tile {
                        acc += tmp[i * tile + b] * c[b * tile + j];
                    }
                    dst[i * tile + j] = acc as f32;
                }
            }
        } else {
            // M[a][j] = Σ_i C[a][i]·X[i][j]
            for a in 0..tile {
                for j in 0..tile {
                    let mut acc = 0.0_f64;
                    for i in 0..tile {
                        acc += c[a * tile + i] * f64::from(src[i * tile + j]);
                    }
                    tmp[a * tile + j] = acc;
                }
            }
            // Y[a][b] = Σ_j M[a][j]·C[b][j]
            for a in 0..tile {
                for b in 0..tile {
                    let mut acc = 0.0_f64;
                    for j in 0..tile {
                        acc += tmp[a * tile + j] * c[b * tile + j];
                    }
                    dst[a * tile + b] = acc as f32;
                }
            }
        }
    }
    Ok(out)
}

/// The orthonormal DCT-II matrix (row-major `[tile][tile]`), in `f64`.
fn dct_matrix(tile: usize) -> Vec<f64> {
    let nf = tile as f64;
    let mut c = vec![0.0_f64; tile * tile];
    for k in 0..tile {
        let alpha = if k == 0 {
            (1.0 / nf).sqrt()
        } else {
            (2.0 / nf).sqrt()
        };
        for j in 0..tile {
            let angle = core::f64::consts::PI * (2.0 * j as f64 + 1.0) * k as f64 / (2.0 * nf);
            c[k * tile + j] = alpha * angle.cos();
        }
    }
    c
}

/// Write a `bits`-wide unsigned `code` starting at `bit_pos`, LSB-first, into a byte slice.
fn write_bits_lsb(bytes: &mut [u8], bit_pos: usize, bits: u32, code: u32) {
    for b in 0..bits as usize {
        let abs = bit_pos + b;
        let bit = ((code >> b) & 1) as u8;
        bytes[abs / 8] |= bit << (abs % 8);
    }
}

/// IEEE-754 single → half, round-to-nearest-even. Hand-rolled (zero-dep). Used by [`absmax_pack`].
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // inf / nan
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // underflow → signed zero
        }
        let m = mant | 0x80_0000; // restore implicit leading 1
        let shift = (14 - unbiased) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let round_bit = 1u32 << (shift - 1);
        let mut h = half;
        if rem > round_bit || (rem == round_bit && (half & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mant16 = mant >> 13;
    let rem = mant & 0x1fff;
    let mut h = ((unbiased as u32) << 10) | mant16;
    let round_bit = 0x1000;
    if rem > round_bit || (rem == round_bit && (mant16 & 1) == 1) {
        h += 1; // a carry into the exponent field is the intended behavior
    }
    sign | h as u16
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

    // -- HOST-3: absmax_pack ∘ det_absmax_unpack round-trip + layout --------------------------

    #[test]
    fn absmax_pack_is_inverse_of_unpack_2bit() {
        // A single chunk of 4, absmax exactly f16-representable so the codebook is clean.
        let x = [1.0_f32, -1.0, 1.0 / 3.0, -1.0 / 3.0];
        let packed = absmax_pack(&x, 4, 2).unwrap();
        // absmax = 1.0 (f16 0x3C00), codes: 1.0→3, -1.0→0, 1/3→2, -1/3→1  ⇒ 0b01_10_00_11 = 0x63.
        assert_eq!(packed, vec![0x00, 0x3C, 0x63]);
        let back = det_absmax_unpack(&packed, 4, 2).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn absmax_pack_is_idempotent_through_decode() {
        // Arbitrary payload, 2 chunks of 8, 2-bit: pack → unpack → re-pack must be byte-identical.
        let x: Vec<f32> = (0..16).map(|n| ((n as f32) - 7.5) * 0.3).collect();
        let p1 = absmax_pack(&x, 8, 2).unwrap();
        let decoded = det_absmax_unpack(&p1, 8, 2).unwrap();
        let p2 = absmax_pack(&decoded, 8, 2).unwrap();
        assert_eq!(p1, p2, "re-packing a decoded payload is a fixed point");
    }

    #[test]
    fn absmax_pack_8bit_full_range() {
        let x = [1.0_f32, -1.0, 0.0];
        let packed = absmax_pack(&x, 3, 8).unwrap();
        // absmax 1.0 → +1 encodes 255, -1 encodes 0, 0 encodes 128 (round((0+1)/2*255)=128).
        assert_eq!(packed[2], 255);
        assert_eq!(packed[3], 0);
        assert_eq!(packed[4], 128);
    }

    #[test]
    fn absmax_pack_zero_chunk_decodes_to_zero() {
        let x = [0.0_f32; 4];
        let packed = absmax_pack(&x, 4, 2).unwrap();
        let back = det_absmax_unpack(&packed, 4, 2).unwrap();
        assert!(back.iter().all(|&v| v == 0.0));
    }

    // -- HOST-2: topk_chunk ---------------------------------------------------------------------

    #[test]
    fn topk_chunk_selects_by_magnitude() {
        // Two chunks of 4, k=2. Chunk 0: |values| pick indices 3,1; chunk 1: pick 0,2.
        let x = [0.1_f32, -0.9, 0.2, 1.0, -5.0, 0.5, 3.0, -0.1];
        let (vals, idx) = topk_chunk(&x, 4, 2).unwrap();
        assert_eq!(idx, vec![3, 1, 0, 2]); // desc magnitude, ties by index
        assert_eq!(vals, vec![1.0, -0.9, -5.0, 3.0]);
    }

    #[test]
    fn topk_chunk_ties_break_by_index() {
        // Equal magnitudes ⇒ lower index wins, deterministically.
        let x = [1.0_f32, -1.0, 1.0, -1.0];
        let (vals, idx) = topk_chunk(&x, 4, 2).unwrap();
        assert_eq!(idx, vec![0, 1]);
        assert_eq!(vals, vec![1.0, -1.0]);
    }

    #[test]
    fn topk_chunk_errors() {
        assert_eq!(
            topk_chunk(&[1.0, 2.0, 3.0], 2, 1),
            Err(DetError::NotDivisible { len: 3, divisor: 2 })
        );
        assert_eq!(
            topk_chunk(&[1.0, 2.0], 2, 3),
            Err(DetError::KTooLarge { k: 3, chunk: 2 })
        );
    }

    // -- HOST-1: dct2 / idct2 -------------------------------------------------------------------

    #[test]
    fn dct2_dc_only_for_constant_block() {
        // A constant 4×4 block: all energy in the DC (0,0) coefficient, everything else ~0.
        let x = vec![2.0_f32; 16];
        let y = dct2(&x, 4).unwrap();
        assert!(
            (y[0] - 8.0).abs() < 1e-4,
            "DC = mean·N = 2·4 = 8, got {}",
            y[0]
        );
        for &v in &y[1..] {
            assert!(v.abs() < 1e-4, "AC coefficient should vanish, got {v}");
        }
    }

    #[test]
    fn dct2_idct2_reconstructs_per_tile() {
        for &tile in &[4_usize, 8, 16] {
            let block = tile * tile;
            let x: Vec<f32> = (0..block * 2)
                .map(|n| ((n as f32) * 0.017).sin() * 3.0 + 0.5)
                .collect();
            let y = dct2(&x, tile).unwrap();
            let back = idct2(&y, tile).unwrap();
            for (a, b) in x.iter().zip(back.iter()) {
                assert!((a - b).abs() < 1e-4, "tile {tile}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn dct2_is_orthonormal_energy_preserving() {
        // Parseval: ‖X‖² == ‖DCT(X)‖² for an orthonormal transform.
        let x: Vec<f32> = (0..64).map(|n| ((n as f32) * 0.3).cos()).collect();
        let y = dct2(&x, 8).unwrap();
        let ex: f32 = x.iter().map(|v| v * v).sum();
        let ey: f32 = y.iter().map(|v| v * v).sum();
        assert!((ex - ey).abs() / ex < 1e-4, "energy {ex} vs {ey}");
    }

    #[test]
    fn dct2_is_bit_reproducible() {
        let x: Vec<f32> = (0..64).map(|n| (n as f32).sin()).collect();
        let a = dct2(&x, 8).unwrap();
        let b = dct2(&x, 8).unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
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
