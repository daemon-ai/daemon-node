// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `det-core` — fixed-order fp32 deterministic kernels.
//!
//! The bit-exact fp32 reference kernels shared verbatim by the host worker and the guest SDK
//! simulator so that "the sim matches the host" is a property of one shared implementation rather
//! than two that happen to agree (`docs/specs/swarm-tensor-abi-spec.md`; swarm-training-spec.md
//! §10.1).
//!
//! **Zero third-party dependencies** (nothing beyond `std`): determinism must never hinge on a
//! transitive crate's floating-point behavior. Every kernel here fixes reduction order explicitly.
//!
//! Wave-0 scaffold: only the fixed-order reduction primitive is present; the full kernel set lands
//! with lane **E**.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Errors returned by the deterministic kernels.
///
/// Hand-rolled to honor this crate's zero-dependency contract (no `thiserror`).
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetError {
    /// Two operands whose shapes must match did not.
    ShapeMismatch {
        /// The expected element count.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
}

impl fmt::Display for DetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
        }
    }
}

impl Error for DetError {}

/// A left-to-right (index-order) fp32 sum.
///
/// Reduction order is fixed by iteration order, never by the compiler or a SIMD reassociation, so
/// the result is bit-identical on every target (the invariant the tensor ABI leans on).
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
