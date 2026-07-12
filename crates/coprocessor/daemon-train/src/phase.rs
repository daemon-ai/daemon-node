// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The phase-legality table (ABI §3.5) — normative.
//!
//! Imports are legal only inside specific guest entry points; calling one elsewhere traps
//! `PhaseViolation`. The table encodes the seam between local math (native lane, rounds) and
//! consensus math (det lane, ingest) at the type-system level of the ABI. Enforced in the host
//! dispatch layer ([`crate::runtime`]); the table here is the single source of truth, exercised
//! table-driven in tests.

use crate::trap::{Trap, TrapCode};

/// A guest lifecycle entry point (ABI §2.3/§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// `da_build`: register params/persistents.
    Build,
    /// `da_step`: forward + backward (accumulate).
    Step,
    /// `da_inner_update`: apply the inner optimizer.
    InnerUpdate,
    /// `da_make_update`: compress local progress into the update container.
    MakeUpdate,
    /// `da_ingest_updates`: decode + aggregate + outer step (det lane).
    Ingest,
}

impl Phase {
    /// The `da_*` export name for this phase.
    #[must_use]
    pub fn entry_name(self) -> &'static str {
        match self {
            Self::Build => "da_build",
            Self::Step => "da_step",
            Self::InnerUpdate => "da_inner_update",
            Self::MakeUpdate => "da_make_update",
            Self::Ingest => "da_ingest_updates",
        }
    }

    fn bit(self) -> u8 {
        match self {
            Self::Build => 1 << 0,
            Self::Step => 1 << 1,
            Self::InnerUpdate => 1 << 2,
            Self::MakeUpdate => 1 << 3,
            Self::Ingest => 1 << 4,
        }
    }
}

// Phase-mask shorthands for the table rows (ABI §3.5 columns).
const BUILD: u8 = 1 << 0;
const STEP: u8 = 1 << 1;
const INNER: u8 = 1 << 2;
const MAKE: u8 = 1 << 3;
const INGEST: u8 = 1 << 4;

/// The normative import→legal-phase table (ABI §3.5). Every `tabi@1` import wired by this lane
/// appears here; the mask is the bitwise-or of the phases in which the import is legal.
///
/// `drop` and the readouts (`scalar`/`metric`/`log`/introspection) are legal in every math/ingest
/// phase; registration is Build-only; the det lane and the container ingest side are Ingest-only;
/// container build + `param_round_base` are MakeUpdate-only; optimizer steps are InnerUpdate-only.
pub const PHASE_TABLE: &[(&str, u8)] = &[
    // registration
    ("param@1", BUILD),
    ("persistent@1", BUILD),
    ("det_persistent@1", BUILD),
    // creation / shape / math / NN (native lane)
    ("zeros@1", STEP | INNER | MAKE),
    ("ones@1", STEP | INNER | MAKE),
    ("full@1", STEP | INNER | MAKE),
    ("add@1", STEP | INNER | MAKE),
    ("sub@1", STEP | INNER | MAKE),
    ("mul@1", STEP | INNER | MAKE),
    ("mul_s@1", STEP | INNER | MAKE),
    ("matmul@1", STEP | INNER | MAKE),
    ("relu@1", STEP | INNER | MAKE),
    ("cross_entropy@1", STEP | INNER | MAKE),
    // assign / detach
    ("assign@1", STEP | INNER | MAKE),
    // autodiff
    ("backward@1", STEP | INNER),
    ("grad@1", STEP | INNER),
    ("zero_grads@1", STEP | INNER),
    // optimizer steps
    ("adamw_step@1", INNER),
    // param round-base (payload production baseline)
    ("param_round_base@1", MAKE),
    // update container build side
    ("upd_new@1", MAKE),
    ("upd_push_bytes@1", MAKE),
    ("upd_push_tensor@1", MAKE),
    // update container ingest side
    ("upd_sections@1", INGEST),
    ("upd_kind@1", INGEST),
    ("upd_bytes_len@1", INGEST),
    ("upd_read_bytes@1", INGEST),
    ("upd_tensor@1", INGEST),
    // det lane (ingest only)
    ("det_zeros@1", INGEST),
    ("det_sum@1", INGEST),
    ("det_scale@1", INGEST),
    ("det_l2norm@1", INGEST),
    ("det_sign@1", INGEST),
    ("det_add@1", INGEST),
    ("det_sub@1", INGEST),
    ("det_mul@1", INGEST),
    ("det_absmax_unpack@1", INGEST),
    ("det_chunk_scatter_add@1", INGEST),
    ("det_chunk_scatter@1", INGEST),
    ("det_assign@1", INGEST),
    ("det_param@1", INGEST),
    ("det_reset_param_to_base@1", INGEST),
    ("det_axpy_param@1", INGEST),
    // batch access (step only)
    ("batch_tokens@1", STEP),
    ("batch_size@1", STEP),
    ("batch_seq_len@1", STEP),
    // drop (step handles) + readouts / introspection — every math + ingest phase
    ("drop@1", STEP | INNER | MAKE | INGEST),
    ("scalar@1", STEP | INNER | MAKE | INGEST),
    ("metric@1", STEP | INNER | MAKE | INGEST),
    ("log@1", STEP | INNER | MAKE | INGEST),
    ("abi_minor@1", STEP | INNER | MAKE | INGEST),
];

/// The mask of phases in which `import` is legal, or `None` if the import is unknown to this host.
#[must_use]
pub fn legal_mask(import: &str) -> Option<u8> {
    PHASE_TABLE
        .iter()
        .find(|(name, _)| *name == import)
        .map(|(_, mask)| *mask)
}

/// Whether `import` is legal in `phase` (ABI §3.5).
#[must_use]
pub fn is_legal(import: &str, phase: Phase) -> bool {
    legal_mask(import).is_some_and(|mask| mask & phase.bit() != 0)
}

/// Enforce phase legality in the host dispatch layer; `Err(PhaseViolation)` otherwise.
///
/// # Errors
///
/// [`TrapCode::PhaseViolation`] if `import` is not legal in `phase`; [`TrapCode::BadModule`] if the
/// import is not one this host implements.
pub fn guard(import: &'static str, phase: Phase) -> Result<(), Trap> {
    match legal_mask(import) {
        None => Err(Trap::new(
            TrapCode::BadModule,
            import,
            Some(phase),
            "unknown import for this host",
        )),
        Some(mask) if mask & phase.bit() != 0 => Ok(()),
        Some(_) => Err(Trap::new(
            TrapCode::PhaseViolation,
            import,
            Some(phase),
            format!("{import} is not legal in {}", phase.entry_name()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Phase; 5] = [
        Phase::Build,
        Phase::Step,
        Phase::InnerUpdate,
        Phase::MakeUpdate,
        Phase::Ingest,
    ];

    #[test]
    fn registration_is_build_only() {
        for &p in &ALL {
            assert_eq!(is_legal("param@1", p), p == Phase::Build);
            assert_eq!(is_legal("det_persistent@1", p), p == Phase::Build);
        }
    }

    #[test]
    fn det_lane_is_ingest_only() {
        for import in [
            "det_sum@1",
            "det_axpy_param@1",
            "det_chunk_scatter_add@1",
            "det_param@1",
        ] {
            for &p in &ALL {
                assert_eq!(is_legal(import, p), p == Phase::Ingest, "{import} @ {p:?}");
            }
        }
    }

    #[test]
    fn optimizer_is_inner_update_only() {
        for &p in &ALL {
            assert_eq!(is_legal("adamw_step@1", p), p == Phase::InnerUpdate);
        }
    }

    #[test]
    fn container_build_is_make_update_only() {
        for import in ["upd_new@1", "upd_push_tensor@1", "param_round_base@1"] {
            for &p in &ALL {
                assert_eq!(is_legal(import, p), p == Phase::MakeUpdate, "{import}");
            }
        }
    }

    #[test]
    fn batch_access_is_step_only() {
        for &p in &ALL {
            assert_eq!(is_legal("batch_tokens@1", p), p == Phase::Step);
        }
    }

    #[test]
    fn drop_and_readouts_are_broad_but_not_build() {
        for import in ["drop@1", "scalar@1", "metric@1", "log@1"] {
            assert!(!is_legal(import, Phase::Build), "{import} not in build");
            assert!(is_legal(import, Phase::Step));
            assert!(is_legal(import, Phase::Ingest));
        }
    }

    #[test]
    fn guard_maps_to_typed_traps() {
        assert!(guard("param@1", Phase::Build).is_ok());
        let violation = guard("det_sum@1", Phase::Step).unwrap_err();
        assert_eq!(violation.code, TrapCode::PhaseViolation);
        let unknown = guard("nonexistent@1", Phase::Step).unwrap_err();
        assert_eq!(unknown.code, TrapCode::BadModule);
    }

    #[test]
    fn table_has_no_duplicate_imports() {
        let mut names: Vec<&str> = PHASE_TABLE.iter().map(|(n, _)| *n).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "phase table imports must be unique");
    }
}
