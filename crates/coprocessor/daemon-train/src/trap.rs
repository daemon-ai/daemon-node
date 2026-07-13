// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The complete trap taxonomy (ABI §3.6).
//!
//! Host functions never return status codes — misuse traps immediately with a typed code (T4). The
//! worker surfaces the code in the `Module` error class as `{code, import, entry_point, detail}`;
//! wasmtime's own traps (fuel/epoch/memory/`unreachable`) are mapped into the same taxonomy so a
//! trapping module is a typed local error, never a worker crash (ABI §3.6, architecture §13).

use std::fmt;

/// Every host-raised trap carries exactly one of these codes (ABI §3.6, normative).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrapCode {
    /// A handle was never valid (`0`, wrong class, or out of range).
    InvalidHandle,
    /// A step handle was used after it was freed / after its entry point returned.
    StaleHandle,
    /// A native op got a det handle or vice versa (ABI §3.4).
    LaneMismatch,
    /// An import was called in an entry point where it is not legal (ABI §3.5).
    PhaseViolation,
    /// Operand shapes are incompatible.
    ShapeMismatch,
    /// Operand dtypes are incompatible.
    DtypeMismatch,
    /// A tensor rank exceeded 8.
    RankOverflow,
    /// A guest memory span fell outside the exported linear memory.
    MemOob,
    /// The guest allocator returned `0`/misaligned for a host `da_alloc` request.
    AllocFail,
    /// A sealed payload exceeded `update_mb_max` (ABI §5.11).
    PayloadOverflow,
    /// The per-entry-point fuel budget was exhausted (ABI §8).
    BudgetFuel,
    /// The wall-clock epoch deadline fired (ABI §8).
    BudgetEpoch,
    /// The linear-memory cap was exceeded (ABI §8).
    BudgetMemory,
    /// The live step-handle cap was exceeded (ABI §8).
    BudgetHandles,
    /// The per-entry-point host-op-call cap was exceeded (ABI §8).
    BudgetOps,
    /// The guest executed `unreachable` (a guest-side panic, ABI §3.6).
    GuestPanic,
    /// A duplicate param/persistent name was registered (ABI §6.3).
    NameCollision,
    /// `scalar@1` was called on a tensor whose numel ≠ 1.
    NotScalar,
    /// An enum-valued argument (dtype/init/class/…) was out of range.
    BadEnum,
    /// The module's `da_abi` reported an incompatible major/minor (ABI §4).
    AbiMismatch,
    /// A required `da_*` export was missing or had the wrong signature (ABI §4/§2.1).
    BadModule,
}

impl TrapCode {
    /// A stable machine-readable slug for the code (worker error surface, architecture §13).
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::InvalidHandle => "InvalidHandle",
            Self::StaleHandle => "StaleHandle",
            Self::LaneMismatch => "LaneMismatch",
            Self::PhaseViolation => "PhaseViolation",
            Self::ShapeMismatch => "ShapeMismatch",
            Self::DtypeMismatch => "DtypeMismatch",
            Self::RankOverflow => "RankOverflow",
            Self::MemOob => "MemOob",
            Self::AllocFail => "AllocFail",
            Self::PayloadOverflow => "PayloadOverflow",
            Self::BudgetFuel => "BudgetFuel",
            Self::BudgetEpoch => "BudgetEpoch",
            Self::BudgetMemory => "BudgetMemory",
            Self::BudgetHandles => "BudgetHandles",
            Self::BudgetOps => "BudgetOps",
            Self::GuestPanic => "GuestPanic",
            Self::NameCollision => "NameCollision",
            Self::NotScalar => "NotScalar",
            Self::BadEnum => "BadEnum",
            Self::AbiMismatch => "AbiMismatch",
            Self::BadModule => "BadModule",
        }
    }
}

impl fmt::Display for TrapCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// A typed trap: the code plus the import, entry point, and a human detail (ABI §3.6).
#[derive(Debug, Clone)]
pub struct Trap {
    /// The trap code.
    pub code: TrapCode,
    /// The import that raised it (`""` for lifecycle/host-origin traps).
    pub import: &'static str,
    /// The entry point in flight when it raised (`None` outside an entry point).
    pub entry_point: Option<crate::phase::Phase>,
    /// A human-readable detail.
    pub detail: String,
}

impl Trap {
    /// Construct a trap.
    #[must_use]
    pub fn new(
        code: TrapCode,
        import: &'static str,
        entry_point: Option<crate::phase::Phase>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code,
            import,
            entry_point,
            detail: detail.into(),
        }
    }

    /// A bare-code trap with no import/entry context (host-origin).
    #[must_use]
    pub fn bare(code: TrapCode, detail: impl Into<String>) -> Self {
        Self::new(code, "", None, detail)
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trap {}", self.code)?;
        if !self.import.is_empty() {
            write!(f, " in {}", self.import)?;
        }
        if let Some(p) = self.entry_point {
            write!(f, " ({})", p.entry_name())?;
        }
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        Ok(())
    }
}

impl std::error::Error for Trap {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_are_unique_and_stable() {
        let codes = [
            TrapCode::InvalidHandle,
            TrapCode::StaleHandle,
            TrapCode::LaneMismatch,
            TrapCode::PhaseViolation,
            TrapCode::ShapeMismatch,
            TrapCode::DtypeMismatch,
            TrapCode::RankOverflow,
            TrapCode::MemOob,
            TrapCode::AllocFail,
            TrapCode::PayloadOverflow,
            TrapCode::BudgetFuel,
            TrapCode::BudgetEpoch,
            TrapCode::BudgetMemory,
            TrapCode::BudgetHandles,
            TrapCode::BudgetOps,
            TrapCode::GuestPanic,
            TrapCode::NameCollision,
            TrapCode::NotScalar,
            TrapCode::BadEnum,
            TrapCode::AbiMismatch,
            TrapCode::BadModule,
        ];
        let mut slugs: Vec<&str> = codes.iter().map(|c| c.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "trap slugs must be unique");
    }

    #[test]
    fn trap_renders_with_context() {
        let t = Trap::new(
            TrapCode::LaneMismatch,
            "matmul@1",
            Some(crate::phase::Phase::Step),
            "det handle in a native op",
        );
        let s = t.to_string();
        assert!(s.contains("LaneMismatch"));
        assert!(s.contains("matmul@1"));
        assert!(s.contains("da_step"));
    }
}
