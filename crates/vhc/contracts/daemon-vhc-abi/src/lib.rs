// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-abi` — the ABI surface both sides of the wasm boundary must agree on.
//!
//! The frozen `tabi@1` import vocabulary ([`TABI_IMPORTS`]) plus the ABI version constants
//! ([`DA_ABI_MAJOR`] / [`DA_ABI_MINOR`] / [`DA_ABI_VERSION`]). This is a `contracts/` crate: it is
//! dual-compiled for native *and* `wasm32-unknown-unknown`, links neither `sdk/*` nor `host/*`, and
//! carries no third-party dependencies.
//!
//! Extracted verbatim from the guest SDK (`daemon-vhc-sdk`) so the host runtime can assert its
//! `Linker` / phase-legality table against the frozen surface *without dev-depending on the SDK*
//! (the host->SDK dev-dep wart). The SDK re-exports these items, so guest-side and existing
//! consumers see an unchanged path. Growth is additive only (ABI §9) — append to [`TABI_IMPORTS`],
//! never reorder or remove.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

/// The tensor-ABI major version this contract is pinned to.
pub const DA_ABI_MAJOR: u32 = 1;
/// The tensor-ABI minor version this contract is pinned to.
pub const DA_ABI_MINOR: u32 = 0;

/// The tensor-ABI version, packed as `(major << 16) | minor`.
///
/// The guest advertises it via the `da_abi` export so the host can reject an incompatible module
/// before instantiation (swarm-tensor-abi-spec.md §4).
pub const DA_ABI_VERSION: u32 = (DA_ABI_MAJOR << 16) | DA_ABI_MINOR;

// ---------------------------------------------------------------------------------------------
// v2 (major 2) — the event-loop driver surface (ABI Draft 3 §1–§2)
//
// A0 lands the *contract*: the major-2 version constants, the import-namespace vocabulary the host
// consults to select a candidate driver (ABI §1.3 step 2), and the typed admission-refusal
// taxonomy (ABI §1.5). The major-2 *driver itself* (the `da_run`/`next_event` event loop) is Phase
// A2 — so a well-formed major-2 module is a clean [`AbiRefusalCode::AbiUnsupportedMajor`] here,
// naming the missing v2 driver, never a trap (ABI §1.2/§1.5, refactor §5 A0, decisions D2).
// ---------------------------------------------------------------------------------------------

/// The major-2 (event-loop driver) ABI major (ABI §0.4, §1.2).
pub const DA_ABI_MAJOR_V2: u32 = 2;
/// The major-2 ABI minor defined by ABI Draft 3 (`da_abi` major 2, minor 0).
pub const DA_ABI_MINOR_V2: u32 = 0;

/// The set of ABI **majors this host generation implements** (i.e. carries a driver for).
///
/// A0 ships the retained v1 five-phase driver only; the major-2 event-loop driver arrives in Phase
/// A2. A module whose declared major is not in this set is refused
/// [`AbiRefusalCode::AbiUnsupportedMajor`] *after* its declaration is cross-checked against its
/// import shape (ABI §1.3 step 5). Adding `2` here is the one-line switch A2 flips once the v2
/// driver + assessment linker land.
pub const HOST_IMPLEMENTED_MAJORS: &[u32] = &[DA_ABI_MAJOR];

/// The host's supported *minor* for an implemented `major` (`None` if the major is not implemented).
///
/// A module declaring `minor > host_minor_for(major)` is refused [`AbiRefusalCode::AbiMinorTooNew`]
/// (ABI §1.3 step 5); a module declaring a lower minor MUST be admitted (ABI §1.4).
#[must_use]
pub fn host_minor_for(major: u32) -> Option<u32> {
    match major {
        DA_ABI_MAJOR => Some(DA_ABI_MINOR),
        // DA_ABI_MAJOR_V2 => Some(DA_ABI_MINOR_V2),  // ← A2 enables this alongside the v2 driver.
        _ => None,
    }
}

// -- import namespaces (the wasm `import_module` names, ABI §2.2) -------------------------------

/// The frozen v1 tensor-ABI import namespace (`#[link(wasm_import_module = "tabi@1")]`).
///
/// Under major 2 this same namespace is the transitional **compute bridge** (ABI §2.5); its
/// *presence alone never makes a module major-1* — the candidate is major 2 whenever any `vhc@2`
/// symbol is imported (ABI §1.2/§1.3).
pub const NS_TABI_V1: &str = "tabi@1";
/// Loop mechanics: `next_event`, `read_back`, `stage_state`, `snapshot_state` (ABI §2.2, §2.6).
pub const NS_VHC_V2: &str = "vhc@2";
/// Network routing: `publish` at Phase A (ABI §2.2).
pub const NS_NET_V2: &str = "net@2";
/// Clock/timers + ambient telemetry: `set_timer`/`cancel_timer`/`now`/`emit_metric`/`log` (ABI §2.2).
pub const NS_SYS_V2: &str = "sys@2";
/// Artifact fetch by hash — no symbols at Phase A (Phase B, ABI §2.2).
pub const NS_DATA_V2: &str = "data@2";
/// Burn-shaped compute surface — no symbols at Phase A (Phase C, ABI §2.2).
pub const NS_COMPUTE_V2: &str = "compute@2";

/// The `vhc@2` Phase-A symbol vocabulary (ABI §2.2 table row `vhc@2`).
pub const VHC_V2_SYMBOLS: &[&str] = &["next_event", "read_back", "stage_state", "snapshot_state"];
/// The `net@2` Phase-A symbol vocabulary (ABI §2.2).
pub const NET_V2_SYMBOLS: &[&str] = &["publish"];
/// The `sys@2` Phase-A symbol vocabulary (ABI §2.2).
pub const SYS_V2_SYMBOLS: &[&str] = &["set_timer", "cancel_timer", "now", "emit_metric", "log"];

/// The v1 five-phase lifecycle exports whose presence (with a `tabi@1`-only import shape) marks a
/// **candidate major-1** module (ABI §1.3 step 2: "exports include the v1 lifecycle (`da_build` …)").
pub const V1_LIFECYCLE_EXPORTS: &[&str] = &[
    "da_build",
    "da_step",
    "da_inner_update",
    "da_make_update",
    "da_ingest_updates",
];

/// The exports a conforming major-2 module MUST provide (ABI §2.1). Checked (by name) once a
/// candidate major-2 driver is selected; the v2 *driver* that consumes them is Phase A2.
pub const V2_REQUIRED_EXPORTS: &[&str] = &[
    "da_abi",
    "da_alloc",
    "da_free",
    "da_manifest",
    "da_claim",
    "da_init",
    "da_run",
];

/// The Phase-A symbol vocabulary the host provides for a v2 import namespace, or `None` if `ns` is
/// not a known v2 namespace. `data@2`/`compute@2` are known but empty at Phase A.
#[must_use]
pub fn v2_namespace_symbols(ns: &str) -> Option<&'static [&'static str]> {
    match ns {
        NS_VHC_V2 => Some(VHC_V2_SYMBOLS),
        NS_NET_V2 => Some(NET_V2_SYMBOLS),
        NS_SYS_V2 => Some(SYS_V2_SYMBOLS),
        NS_DATA_V2 | NS_COMPUTE_V2 => Some(&[]),
        _ => None,
    }
}

/// The candidate driver selected from a module's *static import shape* (ABI §1.3 step 2). This is
/// the input the host acts on before it can call `da_abi` (which requires an already-linked,
/// instantiated module); `da_abi` is then the declaration cross-checked against this (ABI §1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateDriver {
    /// The retained v1 five-phase driver (`tabi@1`-only imports + v1 lifecycle exports).
    V1,
    /// The major-2 event-loop driver (a `vhc@2` symbol is imported).
    V2,
}

impl CandidateDriver {
    /// The `da_abi` major this candidate expects to be cross-checked against (ABI §1.3 step 5).
    #[must_use]
    pub fn major(self) -> u32 {
        match self {
            Self::V1 => DA_ABI_MAJOR,
            Self::V2 => DA_ABI_MAJOR_V2,
        }
    }
}

// -- typed admission-refusal taxonomy (ABI §1.5) ----------------------------------------------

/// The exposed, **split** admission-refusal codes (ABI §1.5). Version-negotiation and admission
/// failures are typed *admission outcomes*, surfaced to the node as `AssessRun`/instantiate
/// verdicts — never wasm traps, never worker crashes, never the reused v1 `TrapCode::AbiMismatch`
/// (decisions D2). Internally every one of these belongs to the broad `AbiMismatch` umbrella the v1
/// code retains; this is the exposed surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbiRefusalCode {
    /// blob blake3 ≠ envelope pin — raised **before compilation** (ABI §1.3 step 1).
    ModuleHashMismatch,
    /// Unrecognizable driver shape, unknown import namespace, or missing/mis-typed required export
    /// (ABI §1.3 steps 2, 6).
    BadModule,
    /// An imported symbol needs a namespace minor this host lacks (subsumes "missing import" for a
    /// known namespace; ABI §1.3 step 3).
    WorldMinorUnsupported,
    /// A major-2 module imports `tabi@1` on a bridge-retired host (Phase C; ABI §1.3 step 3, §2.5).
    BridgeRetired,
    /// The declared `major` is not implemented by this host (ABI §1.3 step 5). In A0 this is how a
    /// well-formed major-2 module is cleanly refused — the v2 event-loop driver arrives in A2.
    AbiUnsupportedMajor,
    /// The declared `minor` exceeds the host's for that major (ABI §1.3 step 5).
    AbiMinorTooNew,
    /// `da_abi` contradicts the import-derived candidate/tuple, or imports ⊄ manifest
    /// (ABI §1.3 step 5, §9.4 step 6).
    AbiDeclarationMismatch,
    /// The manifest requires worlds/ops/channels/depths beyond grants or lane bounds
    /// (ABI §9.4 step 6). Reserved for A2 admission; part of the taxonomy now.
    GrantsExceedLane,
    /// The claim exceeds lane claim-bounds or owner resource authorization (ABI §9.4 step 7/8).
    /// Reserved for A2; part of the taxonomy now.
    ClaimExceedsPolicy,
    /// Repeated `da_claim` invocations returned different bytes (ABI §9.4 step 7). Reserved for A2.
    ClaimInconsistent,
    /// `switch_module` targets a module without `da_migrate` — always an admission refusal, never a
    /// trap (ABI §1.5, §10.3). Reserved for Phase E; part of the taxonomy now.
    MigrateUnsupported,
}

impl AbiRefusalCode {
    /// The stable machine-readable slug (the node-facing admission-outcome surface, ABI §1.5).
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::ModuleHashMismatch => "ModuleHashMismatch",
            Self::BadModule => "BadModule",
            Self::WorldMinorUnsupported => "WorldMinorUnsupported",
            Self::BridgeRetired => "BridgeRetired",
            Self::AbiUnsupportedMajor => "AbiUnsupportedMajor",
            Self::AbiMinorTooNew => "AbiMinorTooNew",
            Self::AbiDeclarationMismatch => "AbiDeclarationMismatch",
            Self::GrantsExceedLane => "GrantsExceedLane",
            Self::ClaimExceedsPolicy => "ClaimExceedsPolicy",
            Self::ClaimInconsistent => "ClaimInconsistent",
            Self::MigrateUnsupported => "MigrateUnsupported",
        }
    }
}

impl fmt::Display for AbiRefusalCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// A typed admission refusal: the split [`AbiRefusalCode`] plus a human-readable `detail` naming the
/// offending value (observed vs supported), per ABI §1.5 (`{code, detail}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiRefusal {
    /// The split refusal code.
    pub code: AbiRefusalCode,
    /// A human-readable detail naming the offending value.
    pub detail: String,
}

impl AbiRefusal {
    /// Construct a typed refusal.
    #[must_use]
    pub fn new(code: AbiRefusalCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for AbiRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.slug(), self.detail)
    }
}

impl std::error::Error for AbiRefusal {}

// -- candidate selection + compatibility-tuple validation (ABI §1.3 steps 2–3) -----------------

/// Select the **candidate driver** from a module's static import namespaces + export names
/// (ABI §1.3 step 2). This is a pure function of the import/export *shape* — no `da_abi` call, no
/// instantiation. The declaration (`da_abi`) is cross-checked against `candidate.major()` afterwards
/// by the host (ABI §1.1, §1.3 step 5).
///
/// - any `vhc@2` import namespace ⇒ [`CandidateDriver::V2`];
/// - otherwise, namespaces ⊆ `{tabi@1}` **and** exports include the v1 lifecycle ⇒
///   [`CandidateDriver::V1`];
/// - otherwise ⇒ [`AbiRefusalCode::BadModule`] (no recognizable driver shape).
///
/// # Errors
///
/// [`AbiRefusalCode::BadModule`] when the shape matches no driver.
pub fn select_candidate(
    import_namespaces: &BTreeSet<&str>,
    exports: &BTreeSet<&str>,
) -> Result<CandidateDriver, AbiRefusal> {
    if import_namespaces.contains(NS_VHC_V2) {
        return Ok(CandidateDriver::V2);
    }
    let imports_within_tabi = import_namespaces.iter().all(|ns| *ns == NS_TABI_V1);
    let has_v1_lifecycle = V1_LIFECYCLE_EXPORTS.iter().all(|e| exports.contains(*e));
    if imports_within_tabi && has_v1_lifecycle {
        return Ok(CandidateDriver::V1);
    }
    Err(AbiRefusal::new(
        AbiRefusalCode::BadModule,
        format!(
            "no recognizable driver shape: import namespaces {:?}, v1 lifecycle exports {} present",
            import_namespaces,
            if has_v1_lifecycle { "all" } else { "not all" }
        ),
    ))
}

/// Validate the compatibility tuple derived from a module's static imports (ABI §1.3 step 3): every
/// imported `(namespace, symbol)` MUST be one the host provides at a version ≥ required.
///
/// - `tabi@1` symbols MUST be in the frozen [`TABI_IMPORTS`] vocabulary, else
///   [`AbiRefusalCode::WorldMinorUnsupported`] (a `tabi@1` symbol the host lacks);
/// - `vhc@2`/`net@2`/`sys@2`/`data@2`/`compute@2` symbols MUST be in that namespace's Phase-A
///   vocabulary ([`v2_namespace_symbols`]), else [`AbiRefusalCode::WorldMinorUnsupported`];
/// - any wholly unknown namespace ⇒ [`AbiRefusalCode::BadModule`].
///
/// This subsumes "missing import for a known namespace" (ABI §1.3 step 3). The `tabi@1`-under-major-2
/// bridge is accepted here (the bridge is advertised through Phase C, ABI §2.5); a bridge-retired
/// host is a Phase-C concern ([`AbiRefusalCode::BridgeRetired`]).
///
/// # Errors
///
/// [`AbiRefusalCode::WorldMinorUnsupported`] for an unknown symbol in a known namespace;
/// [`AbiRefusalCode::BadModule`] for an unknown namespace.
pub fn validate_imports(imports: &[(&str, &str)]) -> Result<(), AbiRefusal> {
    for (namespace, symbol) in imports {
        if *namespace == NS_TABI_V1 {
            if !TABI_IMPORTS.contains(symbol) {
                return Err(AbiRefusal::new(
                    AbiRefusalCode::WorldMinorUnsupported,
                    format!("tabi@1 symbol `{symbol}` is not in the frozen host vocabulary"),
                ));
            }
        } else if let Some(vocab) = v2_namespace_symbols(namespace) {
            if !vocab.contains(symbol) {
                return Err(AbiRefusal::new(
                    AbiRefusalCode::WorldMinorUnsupported,
                    format!(
                        "namespace `{namespace}` symbol `{symbol}` is beyond the host's Phase-A minor"
                    ),
                ));
            }
        } else {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!("unknown import namespace `{namespace}` (symbol `{symbol}`)"),
            ));
        }
    }
    Ok(())
}

/// The complete `tabi@1` import vocabulary the guest SDK binds (the extern block in the SDK's
/// `abi.rs`), in registration order: the Merge-1 frozen 50-import subset followed by the Wave-2
/// additions.
///
/// This is the **frozen surface**: the host `Linker` (`daemon-vhc-host`) and the phase-legality
/// table must agree with it name-for-name (asserted by `daemon-vhc-host/tests/abi_surface.rs`).
/// Growth is additive only (ABI §9) — append here, never reorder or remove.
pub const TABI_IMPORTS: &[&str] = &[
    // --- Merge-1 frozen subset (50) ---
    "param@1",
    "persistent@1",
    "det_persistent@1",
    "drop@1",
    "param_round_base@1",
    "backward@1",
    "grad@1",
    "zero_grads@1",
    "assign@1",
    "zeros@1",
    "ones@1",
    "full@1",
    "add@1",
    "sub@1",
    "mul@1",
    "mul_s@1",
    "matmul@1",
    "relu@1",
    "cross_entropy@1",
    "scalar@1",
    "metric@1",
    "log@1",
    "abi_minor@1",
    "adamw_step@1",
    "batch_tokens@1",
    "batch_size@1",
    "batch_seq_len@1",
    "upd_new@1",
    "upd_push_bytes@1",
    "upd_push_tensor@1",
    "upd_sections@1",
    "upd_kind@1",
    "upd_bytes_len@1",
    "upd_read_bytes@1",
    "upd_tensor@1",
    "det_zeros@1",
    "det_sum@1",
    "det_scale@1",
    "det_l2norm@1",
    "det_sign@1",
    "det_add@1",
    "det_sub@1",
    "det_mul@1",
    "det_absmax_unpack@1",
    "det_chunk_scatter_add@1",
    "det_chunk_scatter@1",
    "det_assign@1",
    "det_param@1",
    "det_reset_param_to_base@1",
    "det_axpy_param@1",
    // --- Wave-2 additions (16) ---
    "embedding@1",
    "rmsnorm@1",
    "softmax@1",
    "silu@1",
    "rope@1",
    "flash_attn@1",
    "reshape@1",
    "transpose@1",
    "slice@1",
    "topk_chunk@1",
    "chunk_scatter@1",
    "absmax_pack@1",
    "absmax_unpack@1",
    "dct2@1",
    "idct2@1",
    "det_idct2@1",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_packs_major_minor() {
        assert_eq!(DA_ABI_VERSION >> 16, 1);
        assert_eq!(DA_ABI_VERSION & 0xffff, 0);
    }

    fn ns_set(ns: &[&'static str]) -> BTreeSet<&'static str> {
        ns.iter().copied().collect()
    }

    #[test]
    fn v2_constants_pack_and_host_supports_only_v1_in_a0() {
        assert_eq!(DA_ABI_MAJOR_V2, 2);
        assert_eq!(HOST_IMPLEMENTED_MAJORS, &[1]);
        assert_eq!(host_minor_for(DA_ABI_MAJOR), Some(0));
        // The v2 driver is Phase A2 — A0 does NOT implement major 2 (a clean AbiUnsupportedMajor).
        assert_eq!(host_minor_for(DA_ABI_MAJOR_V2), None);
    }

    #[test]
    fn v1_shaped_module_selects_v1() {
        let ns = ns_set(&[NS_TABI_V1]);
        let exports = ns_set(&[
            "da_abi",
            "da_alloc",
            "da_build",
            "da_step",
            "da_inner_update",
            "da_make_update",
            "da_ingest_updates",
        ]);
        assert_eq!(
            select_candidate(&ns, &exports).unwrap(),
            CandidateDriver::V1
        );
    }

    #[test]
    fn any_vhc_v2_import_selects_v2_even_with_tabi_bridge() {
        // A major-2 module MAY also link the tabi@1 bridge (ABI §2.5); the candidate is still V2.
        let ns = ns_set(&[NS_VHC_V2, NS_SYS_V2, NS_TABI_V1]);
        let exports = ns_set(V2_REQUIRED_EXPORTS);
        assert_eq!(
            select_candidate(&ns, &exports).unwrap(),
            CandidateDriver::V2
        );
        assert_eq!(CandidateDriver::V2.major(), 2);
    }

    #[test]
    fn tabi_only_without_lifecycle_is_bad_module() {
        let ns = ns_set(&[NS_TABI_V1]);
        let exports = ns_set(&["da_abi", "da_alloc"]);
        assert_eq!(
            select_candidate(&ns, &exports).unwrap_err().code,
            AbiRefusalCode::BadModule
        );
    }

    #[test]
    fn unknown_namespace_is_bad_module() {
        let ns = ns_set(&["xyz@9"]);
        let exports = ns_set(&["da_abi"]);
        assert_eq!(
            select_candidate(&ns, &exports).unwrap_err().code,
            AbiRefusalCode::BadModule
        );
    }

    #[test]
    fn validate_imports_flags_unknown_tabi_symbol() {
        let err = validate_imports(&[("tabi@1", "add@1"), ("tabi@1", "bogus@1")]).unwrap_err();
        assert_eq!(err.code, AbiRefusalCode::WorldMinorUnsupported);
    }

    #[test]
    fn validate_imports_flags_unknown_v2_symbol_and_namespace() {
        assert_eq!(
            validate_imports(&[("vhc@2", "not_a_real_symbol")])
                .unwrap_err()
                .code,
            AbiRefusalCode::WorldMinorUnsupported
        );
        assert_eq!(
            validate_imports(&[("bogus@9", "foo")]).unwrap_err().code,
            AbiRefusalCode::BadModule
        );
        // Phase-A well-formed imports validate cleanly.
        validate_imports(&[
            ("tabi@1", "matmul@1"),
            ("vhc@2", "next_event"),
            ("sys@2", "now"),
        ])
        .unwrap();
    }

    #[test]
    fn refusal_slugs_are_unique_and_stable() {
        let codes = [
            AbiRefusalCode::ModuleHashMismatch,
            AbiRefusalCode::BadModule,
            AbiRefusalCode::WorldMinorUnsupported,
            AbiRefusalCode::BridgeRetired,
            AbiRefusalCode::AbiUnsupportedMajor,
            AbiRefusalCode::AbiMinorTooNew,
            AbiRefusalCode::AbiDeclarationMismatch,
            AbiRefusalCode::GrantsExceedLane,
            AbiRefusalCode::ClaimExceedsPolicy,
            AbiRefusalCode::ClaimInconsistent,
            AbiRefusalCode::MigrateUnsupported,
        ];
        let mut slugs: Vec<&str> = codes.iter().map(|c| c.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "refusal slugs must be unique");
    }

    #[test]
    fn tabi_imports_are_unique_and_complete() {
        // 50 Merge-1 frozen imports + 16 Wave-2 additions = the frozen v1 vocabulary.
        assert_eq!(TABI_IMPORTS.len(), 66);
        let mut names: Vec<&str> = TABI_IMPORTS.to_vec();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "tabi import names must be unique");
        // Every name carries an explicit @version (additive growth is by version, ABI §9).
        assert!(TABI_IMPORTS.iter().all(|n| n.contains('@')));
    }
}
