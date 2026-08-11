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
    // -- major-2 codes (ABI Draft 3 §7.6, "New v2 trap codes") ------------------------------------
    /// A capability import was called in the assessment instance (deny-on-call stub, ABI §9.2).
    ClaimCapabilityDenied,
    /// A call exceeded a grant/channel bound: undeclared or rx-only channel, rate, per-slice
    /// readback bytes, a `read_back` kind not granted (ABI §7.6).
    GrantViolation,
    /// The guest violated the event protocol in a host-detectable way — e.g. breaking the
    /// mandatory `NeedCapacity` retry rule of `next_event`/`read_back` (ABI §4.1/§6.4).
    BadEvent,
    /// `read_back` named a `(src, kind)` that stages nothing (ABI §7.6).
    ReadBackUnavailable,
    /// `da_migrate` exceeded its bounded fuel/memory (ABI §10.3). Reserved until Phase E wires the
    /// upgrade transaction; part of the taxonomy now.
    MigrateBudget,
    /// The guest failed to return from a `Quiesce` drain before the effective deadline; forced
    /// epoch interruption (ABI §4.4/§11.3).
    QuiesceDeadlineExceeded,
    /// A `compute@2` op reported a deferred device execution error at a fence/readback (CUDA-style
    /// deferred error semantics, architecture §3.3; the Phase-C mapping of Burn's `ExecutionError`
    /// / runner faults, ABI §7.6/§15). A stale/unknown tensor handle is [`Self::StaleHandle`] /
    /// [`Self::InvalidHandle`] instead — this code is the *device* failure, never the handle one.
    ComputeFault,
    /// A misframed `state_emit` (ABI §12.14 [SF-4]): an empty chunk, a chunk larger than the
    /// run-pinned `state_chunk_size`, or an emit past the stream's declared `byte_len`. Framing
    /// is deliberately coarse — per-parameter tail alignment is a fold-identity concern, not a
    /// host trap. A grant breach is [`Self::GrantViolation`] instead.
    StateMisframedEmit,
    /// A `state_seal` on a stream whose emitted bytes ≠ its declared `byte_len` (ABI §12.14
    /// [SF-4]): the stream stays open (complete and retry); nothing was made durable.
    StateIncompleteSeal,
    /// The HOST's durable-storage substrate is out of capacity (ENOSPC / quota exceeded) on a
    /// load-bearing write (journal barrier, spill, payload staging). A HOST fault, never a
    /// module defect: the module did nothing wrong and a fresh instance hits the same wall.
    /// Recoverable — but only once capacity returns (the session classifies it storage-gated;
    /// the node redispatches only after a free-space check passes, ABI §3.6).
    HostStorageExhausted,
    /// The HOST's durable-storage substrate failed for a NON-capacity reason on a load-bearing
    /// write: permission denied, corruption, device error. A HOST fault, never a module defect —
    /// and terminal for this node until an operator repairs the substrate (freeing space cannot;
    /// distinct from [`Self::HostStorageExhausted`] for exactly that reason).
    HostStorageFailed,
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
            Self::ClaimCapabilityDenied => "ClaimCapabilityDenied",
            Self::GrantViolation => "GrantViolation",
            Self::BadEvent => "BadEvent",
            Self::ReadBackUnavailable => "ReadBackUnavailable",
            Self::MigrateBudget => "MigrateBudget",
            Self::QuiesceDeadlineExceeded => "QuiesceDeadlineExceeded",
            Self::ComputeFault => "ComputeFault",
            Self::StateMisframedEmit => "StateMisframedEmit",
            Self::StateIncompleteSeal => "StateIncompleteSeal",
            Self::HostStorageExhausted => "HostStorageExhausted",
            Self::HostStorageFailed => "HostStorageFailed",
        }
    }
}

impl fmt::Display for TrapCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// The FAILED completion the trapping event slice consumed, when it consumed one — the evidence
/// behind the REL-4 environmental-trap attribution heuristic (reliability spec §5).
///
/// Carried by construction, not reconstruction: the metadata lives on the active slice state
/// (set at delivery, cleared when the guest asks for the next event), so a trap between slices or
/// in a later slice cannot pick it up. Presence means exactly "the slice this trap occurred in was
/// the one that delivered this failed completion" — temporal adjacency, **never causal proof**.
#[derive(Debug, Clone)]
pub struct EnvCompletion {
    /// The op the completion answered.
    pub op: u64,
    /// The ABI §7.5 `comp-error` code (honest post-REL-3 classes).
    pub code: u64,
    /// The completion's human-readable detail (empty when it carried none).
    pub detail: String,
}

/// A typed trap: the code plus the import, entry point, and a human detail (ABI §3.6).
#[derive(Debug, Clone)]
pub struct Trap {
    /// The trap code.
    pub code: TrapCode,
    /// The import that raised it (`""` for lifecycle/host-origin traps).
    pub import: &'static str,
    /// The guest entry point in flight when it raised (`None` outside one). Since the Phase-E v1
    /// sunset retired the five-phase lifecycle (and its `Phase` enum with the phase-legality
    /// table, decisions D5), this is the entry's export name (e.g. `"da_init"`); v2 traps carry
    /// `None` — the slice context rides `detail`.
    pub entry_point: Option<&'static str>,
    /// A human-readable detail.
    pub detail: String,
    /// The failed completion the trapping slice consumed, when it consumed one (REL-4 evidence;
    /// see [`EnvCompletion`]). Attached at the single trap-consumption seam (`take_trap`), `None`
    /// everywhere a trap is constructed.
    pub env_completion: Option<EnvCompletion>,
}

impl Trap {
    /// Construct a trap.
    #[must_use]
    pub fn new(
        code: TrapCode,
        import: &'static str,
        entry_point: Option<&'static str>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            code,
            import,
            entry_point,
            detail: detail.into(),
            env_completion: None,
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
            write!(f, " ({p})")?;
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
            TrapCode::ClaimCapabilityDenied,
            TrapCode::GrantViolation,
            TrapCode::BadEvent,
            TrapCode::ReadBackUnavailable,
            TrapCode::MigrateBudget,
            TrapCode::QuiesceDeadlineExceeded,
            TrapCode::ComputeFault,
            TrapCode::StateMisframedEmit,
            TrapCode::StateIncompleteSeal,
            TrapCode::HostStorageExhausted,
            TrapCode::HostStorageFailed,
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
            Some("da_init"),
            "det handle in a native op",
        );
        let s = t.to_string();
        assert!(s.contains("LaneMismatch"));
        assert!(s.contains("matmul@1"));
        assert!(s.contains("da_init"));
    }
}
