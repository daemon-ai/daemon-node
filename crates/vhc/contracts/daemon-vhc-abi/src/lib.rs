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

// ================================================================================================
// The journal (ABI companion §8): the normative record grammar + its framing constants.
//
// The crash-safe segmented journal makes policy-determinism operational (ABI §8, architecture
// §3.6). Its record format is a versioned ABI artifact that lives *here* (ABI §8.3: "it lives
// verbatim in `daemon-vhc-abi`"), because both the substrate that writes it (`daemon-vhc-observe`)
// and any conformance validator must agree on one authoritative grammar. This crate stays
// dependency-free / dual-compiled for wasm32, so it holds the grammar text + the framing constants
// only; the Rust record types + canonical CBOR codec live host-side in
// `daemon-vhc-observe::journal` and validate their output against [`JOURNAL_CDDL`].
// ================================================================================================

/// The normative, machine-valid, complete journal record grammar (ABI §8.2/§8.3/§8.5).
///
/// This is the authoritative CDDL artifact: it MUST validate as-is under `cddl-cat`, and tier-1 CI
/// validates every record of every conformance-run journal against it (ABI §13). Root rules:
/// `journal-record` (the §8.3 tagged union), `segment-header-body` (§8.2), and `sidecar-header`
/// (§8.5). Growth is additive by minor — tags 18–63 are reserved; tags are permanent.
pub const JOURNAL_CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/journal.cddl"));

/// The on-disk magic identifying a VHC journal segment file (ABI §8.2 header).
pub const JOURNAL_SEGMENT_MAGIC: &[u8; 8] = b"DVHCJRN2";

/// The on-disk magic identifying a VHC encrypted sidecar file (ABI §8.5 file layout).
pub const JOURNAL_SIDECAR_MAGIC: &[u8; 8] = b"DVHCSC01";

/// The journal format version carried in every segment header and the run-header record (ABI §8.2
/// `format_version`, §8.3 tag-0 `format`).
pub const JOURNAL_FORMAT_VERSION: u32 = 1;

/// The inline/sidecar threshold for `read_back` values (ABI §8.5): a value whose plaintext exceeds
/// this many bytes is stored as an encrypted content-addressed sidecar rather than inline in the
/// record. An ABI constant so writer and replay agree on the boundary.
pub const READBACK_INLINE_MAX: usize = 4096;

/// The numeric tag of every journal record variant (ABI §8.3). Tags are permanent and additive by
/// minor; the reserved range is 18–63. Kept as an explicit list so the substrate and any conformance
/// tool share one source of truth for "the complete record set".
pub const JOURNAL_RECORD_TAGS: &[u8] = &[
    0,  // run-header
    1,  // event
    2,  // read-back
    3,  // clock
    4,  // publish
    5,  // timer-arm
    6,  // timer-cancel
    7,  // drop
    8,  // throttle
    9,  // terminal
    10, // snapshot
    11, // init
    12, // signed-frame
    13, // instantiation
    14, // completion (reserved, Phase B)
    15, // device-profile (reserved, Phase B)
    16, // condition
    17, // seal
];

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

// ================================================================================================
// The major-2 event-loop wire vocabulary (ABI §4–§12): the numeric assignments the event-loop
// driver (`daemon-vhc-host`), the session event pump (`daemon-vhc-session`), the guest SDK
// (`daemon-vhc-sdk`), and the journal/replay verifier (`daemon-vhc-observe`) must ALL agree on.
//
// Like the A0 additions above (v2 namespaces, required exports, refusal taxonomy), these land in
// the shared contract crate *before* the driver that consumes them, so every side is compiled
// against one source of truth. They are pure normative constants — additive, no behavior change —
// and MUST NOT be reordered/renumbered (tags are permanent, ABI §5.2). `HOST_IMPLEMENTED_MAJORS`
// stays `[1]` here: these constants describe the wire the driver WILL speak; flipping the host to
// implement major 2 is coupled to the driver + assessment linker landing (see that constant's doc).
// ================================================================================================

// -- event tags (ABI §4.2): the leading integer of every canonical-CBOR event frame (§5.1) -------

/// `Frame` — a verified signed control frame (ABI §4.2/§4.3).
pub const EV_TAG_FRAME: u64 = 0;
/// `PayloadReady` — content-addressed staged bytes announced by `staging_id` (ABI §4.2/§4.3).
pub const EV_TAG_PAYLOAD_READY: u64 = 1;
/// `Timer` — a one-shot logical-clock timer elapsed (ABI §4.2/§4.3, §6.3).
pub const EV_TAG_TIMER: u64 = 2;
/// `Budget` — a host-initiated budget/pressure/throttle notification (ABI §4.2/§4.3).
pub const EV_TAG_BUDGET: u64 = 3;
/// `Stop` — terminal; after delivery every import traps `PhaseViolation` (ABI §4.2/§4.4).
pub const EV_TAG_STOP: u64 = 4;
/// `Fence` — RESERVED (Phase C `compute@2`); host MUST NOT deliver at major-2 minor 0 (ABI §4.6).
pub const EV_TAG_FENCE: u64 = 5;
/// `Completion` — RESERVED (Phase B async protocol); not delivered at minor 0 (ABI §4.6, §7.5).
pub const EV_TAG_COMPLETION: u64 = 6;
/// `Quiesce` — opens a bounded drain to `QuiesceReady` (ABI §4.2/§4.4).
pub const EV_TAG_QUIESCE: u64 = 7;

/// The Phase-A closed event subset a major-2-minor-0 host delivers and a module MUST handle
/// (ABI §4.2: `{Frame, PayloadReady, Timer, Budget, Stop, Quiesce}`).
pub const PHASE_A_EVENT_TAGS: &[u64] = &[
    EV_TAG_FRAME,
    EV_TAG_PAYLOAD_READY,
    EV_TAG_TIMER,
    EV_TAG_BUDGET,
    EV_TAG_STOP,
    EV_TAG_QUIESCE,
];

// -- `next_event` / `read_back` return status (ABI §4.1, §6.4): `(status << 32) | length` ---------

/// `next_event`/`read_back` status: the frame/value was written into the guest buffer.
pub const RET_STATUS_DELIVERED: u64 = 0;
/// `next_event`/`read_back` status: buffer too small; `length` = exact required capacity, value NOT
/// consumed. Mandatory-retry with an enlarged buffer before any other import (ABI §4.1, §6.4).
pub const RET_STATUS_NEED_CAPACITY: u64 = 1;

/// Pack a `(status, length)` pair into the `(status << 32) | length` return convention shared by
/// `next_event` (ABI §4.1) and `read_back` (ABI §6.4).
#[must_use]
pub const fn pack_status_len(status: u64, length: u32) -> u64 {
    (status << 32) | (length as u64)
}

// -- `read_back` kinds (ABI §6.4) -----------------------------------------------------------------

/// `read_back` kind 0: the staged payload bytes (raw); any `PayloadReady` with `meta.kind = 0`.
pub const READBACK_KIND_STAGED_BYTES: u32 = 0;
/// `read_back` kind 1: a `tabi@1` batch handle (bridge only, ABI §2.5; `meta.kind = 1`).
pub const READBACK_KIND_STAGED_BATCH: u32 = 1;
/// `read_back` kind 2: the staging index for `upd_*@1` (bridge only, ABI §2.5; `meta.kind = 2`).
pub const READBACK_KIND_STAGED_UPDATE: u32 = 2;
/// `read_back` kind 3: bytes of a named state-manifest section (migration restore, ABI §10.2;
/// requires the restore grant; the one kind legal during `da_migrate`, ABI §6.6).
pub const READBACK_KIND_STATE_SECTION: u32 = 3;
/// `read_back` kinds ≥ this are the reserved bridge-op journal kinds (ABI §2.5/§2.7); never valid as
/// call arguments (ABI §6.4).
pub const READBACK_KIND_BRIDGE_JOURNAL_MIN: u32 = 128;

// -- `da_run` Outcome codes (ABI §4.5) ------------------------------------------------------------

/// Outcome 0: clean finish after a `Stop` (ABI §4.5).
pub const OUTCOME_OK: u32 = 0;
/// Outcome 1: the module chose to leave the run (ABI §4.5).
pub const OUTCOME_LEFT: u32 = 1;
/// Outcome 2: returned during a `Quiesce` drain; snapshot manifest published (ABI §4.5, §10.2).
pub const OUTCOME_QUIESCE_READY: u32 = 2;
/// Outcomes ≥ this are module-defined; journaled verbatim, treated by the host as `Left` (ABI §4.5).
pub const OUTCOME_MODULE_DEFINED_MIN: u32 = 16;

// -- stop / quiesce reasons (ABI §4.2) ------------------------------------------------------------

/// `stop-reason` 0: the run completed (ABI §4.2).
pub const STOP_REASON_RUN_COMPLETE: u64 = 0;
/// `stop-reason` 1: leave requested (ABI §4.2).
pub const STOP_REASON_LEAVE_REQUESTED: u64 = 1;
/// `stop-reason` 2: fault (ABI §4.2; also the `SpoolExhausted`/`SequenceGapUnrecoverable` escalation
/// target, ABI §6.7).
pub const STOP_REASON_FAULT: u64 = 2;
/// `stop-reason` 3: owner policy (ABI §4.2).
pub const STOP_REASON_OWNER_POLICY: u64 = 3;

/// `quiesce-reason` 0: an epoch-fenced module upgrade drain (ABI §4.4, §10.3).
pub const QUIESCE_REASON_UPGRADE: u64 = 0;
/// `quiesce-reason` 1: a throttle drain (ABI §4.4, §11.3).
pub const QUIESCE_REASON_THROTTLE: u64 = 1;

// -- `payload-meta.kind` staged-kinds (ABI §4.2) --------------------------------------------------

/// Staged-kind 0: plain bytes (`read_back` kind 0).
pub const STAGED_KIND_BYTES: u64 = 0;
/// Staged-kind 1: a bridge batch (`read_back` kind 1, ABI §2.5).
pub const STAGED_KIND_BATCH: u64 = 1;
/// Staged-kind 2: a bridge update container (`read_back` kind 2, ABI §2.5).
pub const STAGED_KIND_UPDATE_CONTAINER: u64 = 2;

// -- the channel table (ABI §6.2) -----------------------------------------------------------------

/// Channel class 0: authoritative — reliable, ordered per sender, durable spool + gap detection
/// (ABI §4.7, §6.2).
pub const CHANNEL_CLASS_AUTHORITATIVE: u64 = 0;
/// Channel class 1: advisory/gossip — bounded queue, fixed coalescing, journaled drops (ABI §4.7).
pub const CHANNEL_CLASS_ADVISORY: u64 = 1;

/// Channel direction 0: rx-only (publishing traps `GrantViolation`, ABI §6.2).
pub const CHANNEL_DIR_RX_ONLY: u64 = 0;
/// Channel direction 1: tx-only (ABI §6.2).
pub const CHANNEL_DIR_TX_ONLY: u64 = 1;
/// Channel direction 2: bidirectional (ABI §6.2).
pub const CHANNEL_DIR_BIDIRECTIONAL: u64 = 2;

/// The `control` channel id in the Phase-A default channel table (ABI §6.2).
pub const DEFAULT_CHANNEL_CONTROL_ID: u32 = 0;
/// The `control` channel name (ABI §6.2).
pub const DEFAULT_CHANNEL_CONTROL_NAME: &str = "control";

/// The identity of a channel in the **Phase-A default channel table** (ABI §6.2). The table is a
/// driver-provided constant here, versioned with the ABI minor; at minor 0 it declares exactly one
/// channel (`control`). Size/rate/spool **bounds are NOT part of this identity** — they come from
/// the selected `ParticipationLane` profile at admission (ABI §6.2, §9.6), which is node-side
/// configuration, not an ABI constant. From D0 the full `channel-decl` (with bounds) moves into the
/// genesis envelope; the ABI surface (`publish(channel_id, …)`, `frame-ev.channel`, the §12 scope
/// tuple) is unchanged by that move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultChannel {
    /// The channel id the guest passes to `publish` / receives on `frame-ev.channel`.
    pub id: u32,
    /// The channel name.
    pub name: &'static str,
    /// The delivery class ([`CHANNEL_CLASS_AUTHORITATIVE`] / [`CHANNEL_CLASS_ADVISORY`]).
    pub class: u64,
    /// The direction ([`CHANNEL_DIR_RX_ONLY`] / `_TX_ONLY` / `_BIDIRECTIONAL`).
    pub direction: u64,
}

/// The Phase-A (major-2 minor-0) default channel table (ABI §6.2): exactly one authoritative,
/// bidirectional `control` channel that maps onto today's `SignedMessage` control plane. Its bounds
/// are supplied by the lane profile at admission.
pub const PHASE_A_DEFAULT_CHANNEL_TABLE: &[DefaultChannel] = &[DefaultChannel {
    id: DEFAULT_CHANNEL_CONTROL_ID,
    name: DEFAULT_CHANNEL_CONTROL_NAME,
    class: CHANNEL_CLASS_AUTHORITATIVE,
    direction: CHANNEL_DIR_BIDIRECTIONAL,
}];

// -- advisory-class coalescing rules (ABI §2.3 `coalesce`, §4.7) ----------------------------------

/// Coalesce rule 0: dedup-by-hash (`PayloadReady`, ABI §4.7).
pub const COALESCE_DEDUP_HASH: u64 = 0;
/// Coalesce rule 1: latest-wins (`Timer`/`Budget`, ABI §4.7).
pub const COALESCE_LATEST_WINS: u64 = 1;
/// Coalesce rule 2: drop-oldest (gossip, ABI §4.7).
pub const COALESCE_DROP_OLDEST: u64 = 2;

// -- the domain-separated signed-frame envelope (ABI §12.1 — lands at A2) --------------------------

/// The domain-separation tag every major-2 signed frame commits to (ABI §12.1). D1 adds certified
/// keys + `Authority` around this envelope but MUST NOT add/remove/change any envelope field — the
/// fields that give a Phase-A sequence its evidentiary meaning are frozen here at A2 (ABI §12.1/§12.2).
pub const FRAME_ENVELOPE_DOMAIN_V2: &str = "daemon-vhc/frame/2";

// -- `claim()` under-pressure degradation order (ABI §9.1) ----------------------------------------

/// `under_pressure` step 0: deny new buffers (ABI §9.1).
pub const CLAIM_PRESSURE_DENY_NEW_BUFFERS: u64 = 0;
/// `under_pressure` step 1: trap the current slice (ABI §9.1).
pub const CLAIM_PRESSURE_TRAP_CURRENT_SLICE: u64 = 1;

// -- migration scaffolding return codes (ABI §6.3, §10.2, §10.3) ----------------------------------

/// `cancel_timer` status 0: the timer was cancelled before firing; its `Timer` event MUST NOT be
/// delivered (ABI §6.3).
pub const CANCEL_TIMER_CANCELLED: u32 = 0;
/// `cancel_timer` status 1: already fired/delivered/cancelled or never issued (ABI §6.3).
pub const CANCEL_TIMER_ALREADY_FIRED_OR_UNKNOWN: u32 = 1;

/// `snapshot_state` return 0: the manifest was accepted (ABI §10.2).
pub const SNAPSHOT_STATE_ACCEPTED: u32 = 0;
/// `snapshot_state` return 1: a declared section was not staged (ABI §10.2).
pub const SNAPSHOT_STATE_SECTION_MISSING: u32 = 1;
/// `snapshot_state` return 2: a staged section's hash did not match its declaration (ABI §10.2).
pub const SNAPSHOT_STATE_HASH_MISMATCH: u32 = 2;
/// `snapshot_state` return 3: staging exceeded the migration grant (ABI §10.2).
pub const SNAPSHOT_STATE_GRANT_EXCEEDED: u32 = 3;

/// `da_migrate` return 0: state reconstructed and validated (ABI §10.2).
pub const DA_MIGRATE_READY: u32 = 0;
/// `da_migrate` return 1: this module cannot consume the descriptor's manifest (ABI §10.2).
pub const DA_MIGRATE_INCOMPATIBLE: u32 = 1;

/// The top bit set on **guest-created** staging IDs (`stage_state`, ABI §10.2), distinguishing them
/// from host-announced (`PayloadReady`) staging IDs (top bit clear) so the two namespaces never
/// collide and guest IDs need no journal record (counter-derived, replay-reproducible).
pub const GUEST_STAGING_ID_TOP_BIT: u64 = 1 << 63;

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
    fn event_tags_are_permanent_and_unique() {
        // Tags are permanent + never renumbered (ABI §5.2); the positional assignments of §4.2.
        let tags = [
            EV_TAG_FRAME,
            EV_TAG_PAYLOAD_READY,
            EV_TAG_TIMER,
            EV_TAG_BUDGET,
            EV_TAG_STOP,
            EV_TAG_FENCE,
            EV_TAG_COMPLETION,
            EV_TAG_QUIESCE,
        ];
        assert_eq!(tags, [0, 1, 2, 3, 4, 5, 6, 7]);
        let mut sorted = tags.to_vec();
        let n = sorted.len();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "event tags must be unique");
        // The Phase-A closed subset excludes the reserved Fence/Completion tags (ABI §4.2/§4.6).
        assert!(!PHASE_A_EVENT_TAGS.contains(&EV_TAG_FENCE));
        assert!(!PHASE_A_EVENT_TAGS.contains(&EV_TAG_COMPLETION));
        assert_eq!(PHASE_A_EVENT_TAGS.len(), 6);
    }

    #[test]
    fn status_len_packing_round_trips() {
        // The `(status << 32) | length` convention shared by next_event/read_back (ABI §4.1/§6.4).
        let packed = pack_status_len(RET_STATUS_NEED_CAPACITY, 4096);
        assert_eq!(packed >> 32, RET_STATUS_NEED_CAPACITY);
        assert_eq!(packed & 0xffff_ffff, 4096);
        assert_eq!(pack_status_len(RET_STATUS_DELIVERED, 0), 0);
    }

    #[test]
    fn phase_a_default_channel_table_is_single_control_channel() {
        // ABI §6.2 minor-0 default table: exactly one authoritative bidirectional `control` channel.
        assert_eq!(PHASE_A_DEFAULT_CHANNEL_TABLE.len(), 1);
        let control = PHASE_A_DEFAULT_CHANNEL_TABLE[0];
        assert_eq!(control.id, DEFAULT_CHANNEL_CONTROL_ID);
        assert_eq!(control.name, DEFAULT_CHANNEL_CONTROL_NAME);
        assert_eq!(control.class, CHANNEL_CLASS_AUTHORITATIVE);
        assert_eq!(control.direction, CHANNEL_DIR_BIDIRECTIONAL);
    }

    #[test]
    fn readback_kinds_and_bridge_reserve_do_not_overlap_call_kinds() {
        // Call kinds 0..=3 are distinct and below the reserved bridge-journal floor (ABI §6.4).
        let call_kinds = [
            READBACK_KIND_STAGED_BYTES,
            READBACK_KIND_STAGED_BATCH,
            READBACK_KIND_STAGED_UPDATE,
            READBACK_KIND_STATE_SECTION,
        ];
        assert_eq!(call_kinds, [0, 1, 2, 3]);
        assert!(call_kinds
            .iter()
            .all(|k| *k < READBACK_KIND_BRIDGE_JOURNAL_MIN));
        assert_eq!(READBACK_KIND_BRIDGE_JOURNAL_MIN, 128);
    }

    #[test]
    fn guest_staging_id_top_bit_is_high_bit() {
        // Guest-created staging IDs carry the top bit; host-announced ones clear it (ABI §10.2).
        assert_eq!(GUEST_STAGING_ID_TOP_BIT, 0x8000_0000_0000_0000);
        assert_eq!(GUEST_STAGING_ID_TOP_BIT & 1, 0);
    }

    #[test]
    fn frame_envelope_domain_is_major_scoped() {
        // The domain-separation tag is frozen at A2 and major-scoped (ABI §12.1).
        assert_eq!(FRAME_ENVELOPE_DOMAIN_V2, "daemon-vhc/frame/2");
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
