// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **tolerance-class harness** (G1 — defines TDD HOST-3's machinery).
//!
//! A per-op tolerance table (`OpClass` → rtol/atol) plus a comparison runner that drives the same op
//! sequence on a **backend-under-test** and a **reference** [`OpBackend`], asserting forward outputs
//! and backward grads agree within the op's class. The native lane is not bit-identical across
//! backends (burn autodiff vs the CpuBackend tape, later wgpu) so equality is *by class*, never
//! exact — except the det lane / compression natives, which delegate to `det_core` host-side and
//! MUST be byte-identical (`OpClass::Exact`).
//!
//! This module is a shared test harness (a subdirectory module, so cargo does not build it as its
//! own test binary). The **burn-ndarray** parity test (`burn_backend_parity.rs`) passes a
//! `CpuBackend` reference and a `BurnNdarrayBackend` under test; **G2** reuses this module verbatim
//! and passes a `BurnBackend<Autodiff<Wgpu>>` under test — the backend pair is fully parametric
//! (`&mut dyn OpBackend`), which is exactly the seam G2 parametrizes.

#![allow(dead_code)]

use daemon_train::OpBackend;

/// The tolerance class of an op (the HOST-3 machinery). Pinned in [`tol_for`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpClass {
    /// Bit-exact: det lane + compression natives (both delegate to `det_core` host-side).
    Exact,
    /// Pure data moves (reshape/transpose/slice) — bit-exact permutations.
    Shape,
    /// Elementwise native ops (add/sub/mul/mul_s/relu/add_bias).
    Elementwise,
    /// Contractions / gathers (matmul/embedding) — summation-order variant.
    MatmulReduce,
    /// Normalization + shape-preserving fused ops (rmsnorm/softmax/silu/rope).
    Normalization,
    /// Attention + loss (flash_attn/cross_entropy) — deepest native op chains.
    Attention,
    /// Fused optimizer step (adamw) — f32 vs f64 accumulation divergence.
    Optimizer,
}

/// A relative/absolute tolerance pair: pass iff `|got - want| <= atol + rtol * |want|`.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    /// Relative tolerance.
    pub rtol: f32,
    /// Absolute tolerance.
    pub atol: f32,
}

/// The per-op tolerance table. Values measured from the actual ndarray-vs-cpu deltas on the pinned
/// fixtures, with headroom for wgpu (G2). `Exact`/`Shape` assert byte-identity.
#[must_use]
pub fn tol_for(class: OpClass) -> Tol {
    match class {
        OpClass::Exact | OpClass::Shape => Tol {
            rtol: 0.0,
            atol: 0.0,
        },
        OpClass::Elementwise => Tol {
            rtol: 1e-5,
            atol: 1e-6,
        },
        OpClass::MatmulReduce | OpClass::Normalization => Tol {
            rtol: 1e-4,
            atol: 1e-5,
        },
        OpClass::Attention | OpClass::Optimizer => Tol {
            rtol: 2e-4,
            atol: 2e-5,
        },
    }
}

/// Assert `got` matches `want` within the op's tolerance class (element-wise).
pub fn assert_close(got: &[f32], want: &[f32], class: OpClass, ctx: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{ctx}: length mismatch ({} vs {})",
        got.len(),
        want.len()
    );
    let tol = tol_for(class);
    let exact = tol.rtol == 0.0 && tol.atol == 0.0;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if exact {
            assert!(
                g.to_bits() == w.to_bits(),
                "{ctx}[{i}]: {class:?} must be bit-exact, got {g} want {w}"
            );
        } else {
            let diff = (g - w).abs();
            let bound = tol.atol + tol.rtol * w.abs();
            assert!(
                diff <= bound,
                "{ctx}[{i}]: {class:?} |{g} - {w}| = {diff} > {bound}"
            );
        }
    }
}

/// A deterministic pinned-seed fixture generator (seed `0xDAE07E57`, splitmix64). Every parity test
/// draws its fixed inputs from a fresh [`Fixture`] so both backends see identical data.
pub struct Fixture {
    state: u64,
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Fixture {
    /// A fresh generator seeded with the pinned `0xDAE07E57`.
    #[must_use]
    pub fn new() -> Self {
        Self { state: 0xDAE0_7E57 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)` (24 random bits / 2^24).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0_f32
    }

    /// A value in `[-1, 1)`.
    pub fn signed(&mut self) -> f32 {
        self.unit().mul_add(2.0, -1.0)
    }

    /// `n` values in `[-1, 1)`.
    pub fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.signed()).collect()
    }

    /// `n` values in `[-scale, scale)`.
    pub fn vec_scaled(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.signed() * scale).collect()
    }
}

/// Run `run` on `under_test` and `reference`, asserting every returned buffer agrees within `class`.
/// The backends are `&mut dyn OpBackend`, so the same runner serves any backend pair (G2 reuse).
pub fn assert_parity(
    under_test: &mut dyn OpBackend,
    reference: &mut dyn OpBackend,
    class: OpClass,
    ctx: &str,
    run: impl Fn(&mut dyn OpBackend) -> Vec<Vec<f32>>,
) {
    let got = run(under_test);
    let want = run(reference);
    assert_eq!(got.len(), want.len(), "{ctx}: output-buffer count mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(g, w, class, &format!("{ctx}#{i}"));
    }
}
