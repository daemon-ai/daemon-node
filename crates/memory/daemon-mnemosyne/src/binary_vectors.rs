// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! MIB (Maximally Informative Binarization) — port of `binary_vectors.py`.
//!
//! Sign-binarize each embedding dimension (`bit = 1 if x > 0`), pack MSB-first into bytes
//! (384 f32 -> 48 bytes), and compare with Hamming distance (XOR + popcount). The recall
//! `binary_bonus` (`beam.py` L5783) lives here as a pure helper.

/// Default embedding dimensionality (`embeddings.py` BGE-small).
pub const EMBEDDING_DIM: usize = 384;
/// Packed binary vector length in bytes (`binary_vectors.py` L39-L40): 384 / 8 = 48.
pub const BYTES_PER_VECTOR: usize = EMBEDDING_DIM / 8;

/// Sign-binarize + packbits (`binary_vectors.py` L104-L116). Truncates/pads to `EMBEDDING_DIM`.
/// `bit_i = 1` iff `emb[i] > 0` (positive only; zero -> 0). MSB-first packing.
pub fn maximally_informative_binarization(emb: &[f32]) -> Vec<u8> {
    let n = EMBEDDING_DIM;
    let mut out = vec![0u8; n / 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut byte = 0u8;
        for bit in 0..8 {
            let idx = i * 8 + bit;
            let v = emb.get(idx).copied().unwrap_or(0.0);
            if v > 0.0 {
                byte |= 1 << (7 - bit); // MSB-first, matching numpy.packbits
            }
        }
        *slot = byte;
    }
    out
}

/// Hamming distance via XOR + popcount (`binary_vectors.py` L133-L147). Range `[0, 8*len]`.
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Information-Theoretic Score `1 - distance/dim` (`binary_vectors.py` L163).
pub fn its(distance: u32, dim: usize) -> f64 {
    1.0 - (distance as f64) / (dim as f64)
}

/// The recall binary bonus `0.08 * (1 - tanh(normalized_dist * 3))` (`beam.py` L5783).
/// `normalized_dist` is `hamming / EMBEDDING_DIM` in `[0, 1]`; max bonus 0.08 at distance 0.
pub fn binary_bonus(normalized_dist: f64) -> f64 {
    0.08 * (1.0 - (normalized_dist * 3.0).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_to_48_bytes() {
        let emb = vec![0.5f32; EMBEDDING_DIM];
        let packed = maximally_informative_binarization(&emb);
        assert_eq!(packed.len(), BYTES_PER_VECTOR);
        assert!(packed.iter().all(|&b| b == 0xff)); // all positive -> all ones
    }

    #[test]
    fn sign_rule_positive_only() {
        let mut emb = vec![0.0f32; EMBEDDING_DIM];
        emb[0] = 1.0; // bit 0 set (MSB of byte 0)
        emb[7] = -1.0; // negative -> 0
        let packed = maximally_informative_binarization(&emb);
        assert_eq!(packed[0], 0b1000_0000);
    }

    #[test]
    fn hamming_and_its() {
        let a = maximally_informative_binarization(&vec![1.0f32; EMBEDDING_DIM]);
        let b = maximally_informative_binarization(&vec![-1.0f32; EMBEDDING_DIM]);
        let d = hamming_distance(&a, &b);
        assert_eq!(d, EMBEDDING_DIM as u32); // fully opposite
        assert!((its(d, EMBEDDING_DIM) - 0.0).abs() < 1e-9);
        assert!((its(0, EMBEDDING_DIM) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bonus_is_max_at_zero() {
        assert!((binary_bonus(0.0) - 0.08).abs() < 1e-9);
        assert!(binary_bonus(1.0) < binary_bonus(0.0));
    }
}
