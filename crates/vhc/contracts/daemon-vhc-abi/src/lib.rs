// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-abi` — the ABI surface both sides of the wasm boundary must agree on.
//!
//! The major-2 (event-loop driver) version constants, the import-namespace vocabulary the host
//! consults to select a candidate driver (ABI §1.3 step 2), the typed admission-refusal taxonomy
//! (ABI §1.5), and the wire vocabularies the driver / SDK / journal share. This is a `contracts/`
//! crate: it is dual-compiled for native *and* `wasm32-unknown-unknown`, links neither `sdk/*`
//! nor `host/*`, and carries no third-party dependencies.
//!
//! **The retired v1 surface** survives only as its refusal-typing vocabulary: [`DA_ABI_MAJOR`]
//! (the major a synthetic v1-shaped module declares, refused `AbiUnsupportedMajor`),
//! [`V1_LIFECYCLE_EXPORTS`] (the export shape that makes a module a candidate major 1), and
//! [`NS_TABI_V1`] (the retired compute-bridge namespace — any import of it is a typed
//! [`AbiRefusalCode::BridgeRetired`]). The 66-import `tabi@1` vocabulary itself is deleted; its
//! registry lives in the frozen spec history.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

/// The RETIRED tensor-ABI major (the v1 five-phase surface). No driver for it exists — a module
/// declaring it meets a typed [`AbiRefusalCode::AbiUnsupportedMajor`]; the constant survives to
/// type that refusal and its synthetic test inputs.
pub const DA_ABI_MAJOR: u32 = 1;

// ---------------------------------------------------------------------------------------------
// v2 (major 2) — the event-loop driver surface (ABI Draft 3 §1–§2)
// ---------------------------------------------------------------------------------------------

/// The major-2 (event-loop driver) ABI major (ABI §0.4, §1.2).
pub const DA_ABI_MAJOR_V2: u32 = 2;
/// The major-2 ABI minor this host generation implements.
///
/// **Minor 1 is track B1's Phase-B surface** (buffers + the async completion protocol + net
/// payload put/get — the [`V2_SYMBOL_REGISTRY`] minor-1 entries), bumped in the same commit that
/// made `Event::Completion` delivery real in the driver — the ratified additive evolution path
/// (ABI §1.4/§4.6: reserved variants deliverable only above the minor that reserved them),
/// mirroring how the A2 majors flip was coupled to the working event-loop driver.
///
/// **Minor 2 is the Phase-C `compute@2` surface** ([`COMPUTE_MINOR_V2`]; decisions D8, ABI §15):
/// the four-symbol import shim over the `CBOR(burn_ir::OperationIr)` op-stream plus `Event::Fence`
/// (tag 5) delivery, bumped — per the same coupled-to-the-working-driver discipline — in the
/// commit that wired the compute command queue + fence delivery into the event-loop driver
/// (track C1 owns this bump; C2's named-op vocabulary registers at the same minor in its own
/// section). Minor-gating is structural, exactly the tag-6 completions precedent: a `Fence` event
/// can only ever arise from a `compute@2::fence` call, and importing any `compute@2` symbol
/// forces a declared minor ≥ 2 (§1.3 step 5 `AbiDeclarationMismatch` otherwise) — so a module
/// declaring minor ≤ 1 can never receive a tag-5 event. `AbiMinorTooNew` continues to refuse
/// declarations above this constant.
pub const DA_ABI_MINOR_V2: u32 = 2;

/// The set of ABI **majors this host generation implements** (i.e. carries a driver for).
///
/// Exactly `[2]`: the major-2 event-loop driver is the only driver. A module whose declared
/// major is `1` meets a clean [`AbiRefusalCode::AbiUnsupportedMajor`] *after* its declaration is
/// cross-checked against its import shape (ABI §1.3 step 5; pinned by a synthetic in-test
/// major-1 module — no recorded v1 artifact exists).
pub const HOST_IMPLEMENTED_MAJORS: &[u32] = &[DA_ABI_MAJOR_V2];

/// The host's supported *minor* for an implemented `major` (`None` if the major is not implemented).
///
/// A module declaring `minor > host_minor_for(major)` is refused [`AbiRefusalCode::AbiMinorTooNew`]
/// (ABI §1.3 step 5); a module declaring a lower minor MUST be admitted (ABI §1.4). Major 1 is
/// `None` since the Phase-E sunset (`AbiUnsupportedMajor`).
#[must_use]
pub fn host_minor_for(major: u32) -> Option<u32> {
    match major {
        DA_ABI_MAJOR_V2 => Some(DA_ABI_MINOR_V2),
        _ => None,
    }
}

// -- import namespaces (the wasm `import_module` names, ABI §2.2) -------------------------------

/// The RETIRED v1 tensor-ABI import namespace (`#[link(wasm_import_module = "tabi@1")]`) — the
/// transitional compute bridge's name. The bridge is gone: **any** import of this namespace is a
/// typed [`AbiRefusalCode::BridgeRetired`] at import validation. The constant survives only to
/// recognize the namespace and type that refusal; its presence never makes a module major-1
/// (the candidate is major 2 whenever any `vhc@2` symbol is imported, ABI §1.2/§1.3).
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
/// The `sys@2` **Phase-A** symbol vocabulary (ABI §2.2 table row `sys@2`, Phase A column).
pub const SYS_V2_PHASE_A_SYMBOLS: &[&str] =
    &["set_timer", "cancel_timer", "now", "emit_metric", "log"];

/// The `sys@2` **crypto acceleration** symbols (Phase B; ABI §2.2 "later phases", architecture
/// §3.2/§3.7): `hash` and `verify_sig`, following the det-lane pattern — semantics pinned by the
/// dual-compiled `daemon-vhc-proto::crypto` contract, an in-guest fallback always available, the
/// host import as the fast path. Their guest-visible results are **deterministic functions of
/// their inputs** (no host observation), so they carry no journal record — replay re-executes
/// them over the reproduced guest memory (the `dc`/`dd` replay classes, ABI §2.7).
///
/// Cross-track note (B1/B3): adding import symbols is an *additive-minor* growth (ABI §1.4). The
/// shared `DA_ABI_MINOR_V2` bump to the collective Phase-B minor — and the per-symbol
/// introduced-minor map §1.3 step 3 wants — is a single change all three Phase-B tracks share; it
/// is deliberately **not** made here so the tracks can agree on one Phase-B minor at merge and the
/// v2 guest blobs are re-pinned once, not three times. A module declaring minor 0 that does not
/// import these symbols is unaffected; the host links them for any module that does.
pub const SYS_V2_CRYPTO_SYMBOLS: &[&str] = &["hash", "verify_sig"];

/// The `sys@2` **ambient Phase-B** symbols (ABI §2.2 "later phases": "seeded RNG, device
/// profile"): `rng_seed` (the run-scoped deterministic seed — a pure function of the execution
/// identity, so deterministic per §2.7 dc class, no journal record; replay re-derives it) and
/// `device_profile` (the same profile the permanent probe measures, architecture §3.5 — a
/// **nondeterministic input**, journaled as the §8.3 tag-15 record on every delivery). The same
/// minor-bump note as [`SYS_V2_CRYPTO_SYMBOLS`] applies: the shared Phase-B minor lands once, at
/// the B1→B2→B3 merge.
pub const SYS_V2_AMBIENT_B_SYMBOLS: &[&str] = &["rng_seed", "device_profile"];

/// The full `sys@2` symbol vocabulary the host provides (Phase-A subset + the Phase-B crypto
/// accelerations + the Phase-B ambient symbols). The candidate-selection/import-validation path
/// ([`validate_imports`]) checks membership against this.
pub const SYS_V2_SYMBOLS: &[&str] = &[
    "set_timer",
    "cancel_timer",
    "now",
    "emit_metric",
    "log",
    "hash",
    "verify_sig",
    "rng_seed",
    "device_profile",
];

/// The `data@2` symbol vocabulary (Phase B, track B2; architecture §3.2 "the data world"):
/// `fetch(artifact_blake3, range_off, range_len) -> OpId`, completing
/// `Ok(BufferHandle)` through `Event::Completion` (tag 6) after **whole-artifact** blake3
/// verification host-side. The guest names artifacts **by content hash only** — never by URL or
/// locator: resolution happened once, at the edge (the envelope's committed artifact map,
/// architecture §5.1 snapshot pinning); hosts fetch-and-verify against the committed hash and
/// never resolve independently; fetch credentials are host-held and never enter the sandbox.
/// Which artifacts a module may touch is a **grant** — a fetch outside the admitted artifact
/// set traps `GrantViolation` (typed, attributable).
///
/// **Sub-resource (range) verification rule:** there are no per-range hashes in the artifact
/// map, so a range is verified as a slice OF hash-verified content — the host fetches and
/// verifies the whole artifact against its committed blake3 (LRU content cache makes repeated
/// ranged reads cheap), then slices `[range_off, range_off + range_len)` (`range_len == 0` =
/// to the end) into the completion buffer. An out-of-bounds range completes
/// `Err(StoreRefused)` — a completion error, not a trap (bounds are unknowable at call time).
pub const DATA_V2_SYMBOLS: &[&str] = &["fetch"];

/// The domain-separation context for the run-scoped RNG seed (`sys@2::rng_seed`): the seed is
/// `blake3::derive_key(RNG_SEED_DOMAIN_V2, material)` where `material` is the unambiguous
/// concatenation of the frozen execution identity (§8.1) —
/// `run_id ‖ epoch_le ‖ role_len_le ‖ role ‖ instance_le ‖ module_hash`. A pure function of the
/// identity: two role-instances never share a seed, a trap-restart of the SAME incarnation
/// reproduces it (policy determinism, architecture §3.6 claim 1), and replay re-derives it from
/// the journal's run header instead of recording it.
pub const RNG_SEED_DOMAIN_V2: &str = "daemon-vhc/rng/2";

/// The byte length of the `sys@2::rng_seed` output.
pub const RNG_SEED_LEN: usize = 32;

/// The `compute@2` symbol vocabulary (Phase C, track C1; architecture §3.2/§3.3/§3.4, ABI §15
/// ratified direction). The Burn-shaped compute world lowers ordinary `Autodiff<HostBackend>`
/// models to a stream of **CBOR-encoded [`burn_ir::OperationIr`] at the pinned Burn version**
/// (`burn = 0.21.0`; a Burn bump is an ABI event — decisions D8). The four symbols are the wasm
/// import shim over that stream:
///
/// - `submit_op(op_ptr, op_len)` — enqueue one CBOR `OperationIr` (command-queue semantics, ABI
///   §3.3: enqueue is **infallible** and returns immediately; errors surface only at fence/export).
///   Opaque `TensorId`s (guest-minted, guest reference-counted, released via `OperationIr::Drop`)
///   are carried *inside* the op blob — they are a guest-owned id space like staging/timer IDs,
///   **not** [§7.2](`pack_handle`) host-arena handles.
/// - `fence(fence_id)` — insert a fence marker into the compute queue; the host delivers
///   `Event::Fence(fence_id)` ([`EV_TAG_FENCE`]) when the device passes it (architecture §3.3).
///   Device errors surface here as a typed completion/trap.
/// - `export(ir_ptr, ir_len) -> OpId` — device tensor → staging: names a tensor by CBOR `TensorIr`
///   and completes `Ok(BufferHandle)` (kind 8) carrying the CBOR `TensorData`. Bulk bytes ride the
///   `BufferHandle`/`read_into` path (§3.4), never inline in the op-stream (§15 caveat).
/// - `import(buffer, meta_ptr, meta_len) -> OpId` — staging → device: registers a sealed buffer's
///   `TensorData` under the guest-minted `TensorId` named in `meta`, completing `Ok(())`.
///
/// **Reserved, refuse cleanly until specified (ABI §15):** quantization / `QFloat`
/// (`OperationIr::Float(Quantize/Dequantize)` do not lower today) and `OperationIr::Custom`
/// (custom/fused kernels stay in C2's host-side custom-op registry, architecture §3.2);
/// `OperationIr::Distributed` is out of scope. The rank-erased-primitive property of Burn is a
/// hard ABI floor.
///
/// **Minor-bump note (mirrors [`SYS_V2_CRYPTO_SYMBOLS`]):** these symbols register at
/// [`COMPUTE_MINOR_V2`] (the shared Phase-C minor). Track C1 owns the [`DA_ABI_MINOR_V2`] bump to
/// it (orchestrator ruling, sitting 2), landed — per B1's precedent — in the commit that wired the
/// compute command queue + `Event::Fence` delivery into the event-loop driver; C2's named-op
/// vocabulary registers at the same minor in its own delimited section and the merge unions.
pub const COMPUTE_V2_SYMBOLS: &[&str] = &["submit_op", "fence", "export", "import"];

/// The shared Phase-C major-2 minor at which the `compute@2` surface ([`COMPUTE_V2_SYMBOLS`]) and
/// the [`EV_TAG_FENCE`] event are introduced (ABI §15, decisions D8). [`DA_ABI_MINOR_V2`] reached
/// this minor with the C1 driver wiring (see its doc for the structural minor-gating argument).
pub const COMPUTE_MINOR_V2: u32 = 2;

/// The pinned Burn version the `compute@2` wire (`CBOR(burn_ir::OperationIr)`) is defined against
/// (ABI §15, decisions D8). The IR's variant set / discriminants / fields are Burn-version-
/// specific, so **a Burn version bump is an ABI event** (a variant insertion is a compute-major;
/// a provably additive change is at most a compute-minor). One pinned version per ABI major.
pub const COMPUTE_BURN_VERSION: &str = "0.21.0";

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

/// The symbol vocabulary the host provides for a v2 import namespace, or `None` if `ns` is
/// not a known v2 namespace. `data@2` populated at Phase B; `compute@2` at Phase C (registered at
/// [`COMPUTE_MINOR_V2`], see [`COMPUTE_V2_SYMBOLS`]).
#[must_use]
pub fn v2_namespace_symbols(ns: &str) -> Option<&'static [&'static str]> {
    match ns {
        NS_VHC_V2 => Some(VHC_V2_SYMBOLS),
        NS_NET_V2 => Some(NET_V2_SYMBOLS),
        NS_SYS_V2 => Some(SYS_V2_SYMBOLS),
        NS_DATA_V2 => Some(DATA_V2_SYMBOLS),
        NS_COMPUTE_V2 => Some(COMPUTE_V2_SYMBOLS),
        _ => None,
    }
}

// -- the v2 symbol registry: every symbol's introducing minor (ABI §1.3 step 3, §1.4) ------------
//
// "A0: `daemon-vhc-abi` gains the v2 constants AND the symbol registry §1.3 step 3 consults to map
// each static import to its introducing minor" (decisions D2). The registry is how additive minor
// growth stays checkable: a host admits a symbol iff its introducing minor ≤ the host's implemented
// minor for that namespace's major, and a module's DECLARED minor must be ≥ the highest introducing
// minor among its imports (a declared minor below what the imports require is
// `AbiDeclarationMismatch`, ABI §1.3 step 5). Minor-1 entries are track B1's Phase-B surface:
// buffers (§3.4/§7.4), the async completion protocol (§3.3/§7.5), and net payload put/get.

/// Every `(namespace, symbol, introducing minor)` of the major-2 import surface. Growth is
/// append-only; a symbol's introducing minor is permanent.
pub const V2_SYMBOL_REGISTRY: &[(&str, &str, u32)] = &[
    // -- minor 0 (Phase A) --
    (NS_VHC_V2, "next_event", 0),
    (NS_VHC_V2, "read_back", 0),
    (NS_VHC_V2, "stage_state", 0),
    (NS_VHC_V2, "snapshot_state", 0),
    (NS_NET_V2, "publish", 0),
    (NS_SYS_V2, "set_timer", 0),
    (NS_SYS_V2, "cancel_timer", 0),
    (NS_SYS_V2, "now", 0),
    (NS_SYS_V2, "emit_metric", 0),
    (NS_SYS_V2, "log", 0),
    // -- minor 1 (Phase B, track B1): buffers + the async completion protocol ------------------
    // The two budgeted linear-memory paths (§3.4): OUT of linear memory (sealed at creation) and
    // INTO it (charged against the per-slice readback allowance).
    (NS_VHC_V2, "create_from", 1),
    (NS_VHC_V2, "read_into", 1),
    // Deterministic buffer bookkeeping: length + explicit guest release (§3.4 ownership).
    (NS_VHC_V2, "buffer_len", 1),
    (NS_VHC_V2, "buffer_release", 1),
    // The completion protocol's cancellation import (§3.3/§7.5): the op completes `Cancelled`.
    (NS_VHC_V2, "cancel", 1),
    // Content-addressed payload plane by handle (§3.4): both complete via `Event::Completion`.
    (NS_NET_V2, "payload_put", 1),
    (NS_NET_V2, "payload_get", 1),
    // Direct peer streams under credit-based flow control (§3.3/§3.4): open/accept complete with
    // a kind-9 StreamHandle; writes consume writable credit (replenished by the receiver's reads,
    // surfacing as the writes' completions); reads complete with a kind-8 BufferHandle.
    (NS_NET_V2, "stream_open", 1),
    (NS_NET_V2, "stream_accept", 1),
    (NS_NET_V2, "stream_write", 1),
    (NS_NET_V2, "stream_read", 1),
    // -- minor 1 (Phase B, track B2): sys@2 crypto accels + ambient inputs ----------------------
    // The det-lane-pattern crypto accelerations (deterministic, unjournaled — §2.7 dc class).
    (NS_SYS_V2, "hash", 1),
    (NS_SYS_V2, "verify_sig", 1),
    // The identity-derived deterministic seed (unjournaled; replay re-derives it).
    (NS_SYS_V2, "rng_seed", 1),
    // The probe's device profile (nondeterministic input — journaled tag 15 per delivery).
    (NS_SYS_V2, "device_profile", 1),
    // -- minor 1 (Phase B, track B2): the data world (architecture §3.2) ------------------------
    // Artifact fetch by committed hash + range, completing Ok(BufferHandle) (see
    // `DATA_V2_SYMBOLS` for the pinning/grant/range rules).
    (NS_DATA_V2, "fetch", 1),
    // -- minor 2 (Phase C, track C1): the Burn-shaped compute world (ABI §15, decisions D8) ------
    // The wasm import shim over the CBOR(burn_ir::OperationIr) op-stream: enqueue (infallible),
    // fence (-> Event::Fence), and the two BufferHandle-mediated bulk crossings (export/import).
    // These register at COMPUTE_MINOR_V2 but the host advertises minor 2 (admitting them + the
    // Fence event) only after the shared Phase-C DA_ABI_MINOR_V2 bump at the track merge — the
    // same "registered before host-implemented" convention the Phase-B sys crypto/ambient symbols
    // used (see COMPUTE_V2_SYMBOLS). Until then validate_imports cleanly refuses a compute import
    // as WorldMinorUnsupported naming the introducing minor.
    (NS_COMPUTE_V2, "submit_op", COMPUTE_MINOR_V2),
    (NS_COMPUTE_V2, "fence", COMPUTE_MINOR_V2),
    (NS_COMPUTE_V2, "export", COMPUTE_MINOR_V2),
    (NS_COMPUTE_V2, "import", COMPUTE_MINOR_V2),
];

/// The minor at which `(namespace, symbol)` was introduced, or `None` if it is not a registered
/// v2 symbol (ABI §1.3 step 3).
#[must_use]
pub fn v2_symbol_minor(namespace: &str, symbol: &str) -> Option<u32> {
    V2_SYMBOL_REGISTRY
        .iter()
        .find(|(ns, sym, _)| *ns == namespace && *sym == symbol)
        .map(|(_, _, minor)| *minor)
}

/// The minimum major-2 minor a module's static v2 imports require: the highest introducing minor
/// among them (0 when none are v2 symbols). A module MUST declare at least this minor — declaring
/// below what the imports require is `AbiDeclarationMismatch` (ABI §1.3 step 5).
#[must_use]
pub fn required_v2_minor(imports: &[(&str, &str)]) -> u32 {
    imports
        .iter()
        .filter_map(|(ns, sym)| v2_symbol_minor(ns, sym))
        .max()
        .unwrap_or(0)
}

/// The candidate driver selected from a module's *static import shape* (ABI §1.3 step 2). This is
/// the input the host acts on before it can call `da_abi` (which requires an already-linked,
/// instantiated module); `da_abi` is then the declaration cross-checked against this (ABI §1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateDriver {
    /// The RETIRED v1 five-phase shape (`tabi@1`-only imports + v1 lifecycle exports) — selected
    /// only so its refusal is typed (`AbiUnsupportedMajor` at the declaration cross-check).
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
    /// The module imports the retired `tabi@1` compute bridge (ABI §1.3 step 3): the bridge no
    /// longer exists on any host — compute crosses the boundary through `compute@2` only.
    BridgeRetired,
    /// The declared `major` is not implemented by this host (ABI §1.3 step 5). Through A0 this is
    /// how a well-formed major-2 module was cleanly refused; since A2 landed the event-loop
    /// driver, it guards majors outside [`HOST_IMPLEMENTED_MAJORS`].
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
    /// The manifest requires a versioned custom op (`manifest.custom_ops`, ABI §2.3) this host does
    /// not advertise in its custom-op registry ([`HOST_CUSTOM_OPS`]) — a clean admission refusal,
    /// never a trap (architecture §3.2 "admission fails cleanly on hosts that lack them"). [C2:custom-op]
    CustomOpUnsupported,
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
            Self::CustomOpUnsupported => "CustomOpUnsupported",
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
/// - **any** `tabi@1` symbol ⇒ [`AbiRefusalCode::BridgeRetired`] — the transitional compute
///   bridge is retired on every host; compute crosses the boundary through `compute@2` only;
/// - `vhc@2`/`net@2`/`sys@2`/`data@2`/`compute@2` symbols MUST be in the
///   [`V2_SYMBOL_REGISTRY`] **at an introducing minor ≤ the host's implemented major-2 minor**,
///   else [`AbiRefusalCode::WorldMinorUnsupported`] (an unregistered symbol and a
///   registered-but-too-new symbol are the same refusal: this host cannot serve the import);
/// - any wholly unknown namespace ⇒ [`AbiRefusalCode::BadModule`].
///
/// This subsumes "missing import for a known namespace" (ABI §1.3 step 3). The complementary
/// declaration-side rule — a declared minor below [`required_v2_minor`] — is the selector's
/// step-5 cross-check (`AbiDeclarationMismatch`), not this function's.
///
/// # Errors
///
/// [`AbiRefusalCode::BridgeRetired`] for any `tabi@1` import;
/// [`AbiRefusalCode::WorldMinorUnsupported`] for an unknown/too-new symbol in a known namespace;
/// [`AbiRefusalCode::BadModule`] for an unknown namespace.
pub fn validate_imports(imports: &[(&str, &str)]) -> Result<(), AbiRefusal> {
    let host_v2_minor = host_minor_for(DA_ABI_MAJOR_V2).unwrap_or(0);
    for (namespace, symbol) in imports {
        if *namespace == NS_TABI_V1 {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BridgeRetired,
                format!(
                    "`tabi@1::{symbol}`: the transitional compute bridge is retired — compute \
                     crosses the boundary through `compute@2` only"
                ),
            ));
        } else if v2_namespace_symbols(namespace).is_some() {
            match v2_symbol_minor(namespace, symbol) {
                Some(introduced) if introduced <= host_v2_minor => {}
                Some(introduced) => {
                    return Err(AbiRefusal::new(
                        AbiRefusalCode::WorldMinorUnsupported,
                        format!(
                            "namespace `{namespace}` symbol `{symbol}` was introduced at minor \
                             {introduced}, beyond this host's implemented minor {host_v2_minor}"
                        ),
                    ));
                }
                None => {
                    return Err(AbiRefusal::new(
                        AbiRefusalCode::WorldMinorUnsupported,
                        format!(
                            "namespace `{namespace}` symbol `{symbol}` is not in the v2 symbol \
                             registry"
                        ),
                    ));
                }
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

// ================================================================================================
// Custom-op registry (Phase C; architecture §3.2, refactor §7).                    [C2:custom-op]
//
// Fused kernels (flash-attention variants, fused optimizer steps) cannot cross the wasm boundary
// as code; they register HOST-side under versioned names. A module's manifest lists the custom ops
// it needs (`manifest.custom_ops`, ABI §2.3) and admission fails CLEANLY — a typed refusal, never
// a trap — on a host that lacks a required one (architecture §3.2, §5.2). This is the named-op
// admission/dispatch seam that fills the RESERVED `compute@2` `OperationIr::Custom` variant
// (ABI §15): C1 owns the `OperationIr` wire and refuses the Custom IR variant; the host resolves
// the NAMED op through this registry. Coordinated by vocabulary, not shared code — kept in its own
// delimited section so the serial Phase-C merge with C1's compute@2 sections stays mechanical.
// ================================================================================================

/// The first registered custom op and the template for future fusions: a fused (causal/full)
/// attention kernel, versioned `@1`. Versioning is part of the name: a new kernel or an
/// incompatible revision is a distinct entry (`flash_attn@2`), never a silent redefinition — the
/// admission contract (`required ⊆ advertised`) is over these exact versioned strings.
pub const CUSTOM_OP_FLASH_ATTN_V1: &str = "flash_attn@1";

/// The versioned named custom ops this host advertises — the "advertised" set of admission's
/// `required ⊆ granted ⊆ advertised` (architecture §3.2). A module whose `manifest.custom_ops`
/// names an entry absent here is refused [`AbiRefusalCode::CustomOpUnsupported`] at admission (the
/// one place new research can still require a host release — by design rare). Growth is additive;
/// entries are permanent and version-suffixed, so a host only ever *adds* fusions it implements.
pub const HOST_CUSTOM_OPS: &[&str] = &[CUSTOM_OP_FLASH_ATTN_V1];

/// Whether this host advertises the versioned custom op `name` in its registry ([`HOST_CUSTOM_OPS`]).
#[must_use]
pub fn host_supports_custom_op(name: &str) -> bool {
    HOST_CUSTOM_OPS.contains(&name)
}

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
/// `Fence` — a compute-queue marker the guest inserted with `compute@2::fence(id)` has been passed
/// by the device (architecture §3.3, ABI §4.6/§15). Deliverable only at minor ≥ [`COMPUTE_MINOR_V2`]
/// (Phase C); a host below that minor MUST NOT deliver it (ABI §4.6).
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
/// Kind 1 — RETIRED with the compute bridge (was its staged-batch handle). The assignment is
/// permanent and never reused; the kind is no longer a valid `read_back` call argument.
pub const READBACK_KIND_STAGED_BATCH: u32 = 1;
/// Kind 2 — RETIRED with the compute bridge (was its staged-update-container index). Permanent,
/// never reused, no longer a valid call argument.
pub const READBACK_KIND_STAGED_UPDATE: u32 = 2;
/// `read_back` kind 3: bytes of a named state-manifest section (migration restore, ABI §10.2;
/// requires the restore grant; the one kind legal during `da_migrate`, ABI §6.6).
pub const READBACK_KIND_STATE_SECTION: u32 = 3;
/// Kind 4 — **journal-record kind only** (assigned by the B1 minor): the bytes behind a
/// `stream_read` completion. Stream payloads are opaque and NOT content-addressed, so — unlike
/// `payload_get` buffers, which replay re-fetches by hash (§8.7) — the received bytes are a
/// nondeterministic input journaled verbatim (a §8.3 tag-2 record, `src` = the read's `OpId`)
/// at completion arrival; replay materializes the completion's buffer from that record. Never a
/// valid `read_back` CALL argument at this minor (the same never-callable discipline as the
/// ≥ 128 bridge journal kinds).
pub const READBACK_KIND_STREAM_BYTES: u32 = 4;
/// Kind 5 — **journal-record kind only** (assigned by the C1 compute minor): the `CBOR(TensorData)`
/// bytes behind a `compute@2::export` completion. An exported tensor's bytes are a
/// **nondeterministic input** (device-produced — native-lane arithmetic, architecture §3.6 claim
/// 3), so — exactly like kind-4 stream bytes — they are journaled verbatim (a §8.3 tag-2 record,
/// `src` = the export's `OpId`) at completion arrival; replay materializes the completion's
/// buffer from that record and re-executes no kernel (§8.7). Never a valid `read_back` CALL
/// argument (the same never-callable discipline as kind 4 and the ≥ 128 bridge journal kinds).
pub const READBACK_KIND_TENSOR_EXPORT: u32 = 5;
/// `read_back` kinds ≥ this were the retired bridge's reserved journal kinds (permanent, never
/// reused); never valid as
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
/// Staged-kind 1 — RETIRED with the compute bridge (permanent assignment, never reused).
pub const STAGED_KIND_BATCH: u64 = 1;
/// Staged-kind 2 — RETIRED with the compute bridge (permanent assignment, never reused).
pub const STAGED_KIND_UPDATE_CONTAINER: u64 = 2;
/// Staged-kind 3: a migration state section (`read_back` kind 3, legal only during `da_migrate` —
/// ABI §6.4/§6.6/§10.2). Host-staged on the NEW run instance from the accepted snapshot; the
/// restore staging IDs travel in the migration descriptor, never as `PayloadReady` events.
pub const STAGED_KIND_STATE_SECTION: u64 = 3;

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

// -- the `da_claim` memory-claim contract (ABI §9.1) -----------------------------------------------
//
// The tiered memory envelope every major-2 module reports as a deterministic, cheap, compute-free
// function of (config, grants). The canonical-CBOR map keys are fixed here so the guest-side
// authors/SDK and the host-side evaluator (`daemon-vhc-host::v2::admission`) agree on one
// vocabulary; the schema itself is ratified in the ABI companion (§9.1 `memory-claim`).

/// `memory-claim` key: resources the host meters EXACTLY — the enforceable cap (breach is a typed
/// attributable trap, ABI §9.1).
pub const CLAIM_KEY_HARD_ACCOUNTABLE: &str = "hard_accountable";
/// `memory-claim` key: the expected high-water mark, judged at admission against owner policy.
pub const CLAIM_KEY_DECLARED_PEAK: &str = "declared_peak";
/// `memory-claim` key: host-side costs the module cannot see and is never blamed for.
pub const CLAIM_KEY_WORKSPACE: &str = "workspace";
/// `memory-claim` key: the ordered degradation steps (`CLAIM_PRESSURE_*`).
pub const CLAIM_KEY_UNDER_PRESSURE: &str = "under_pressure";
/// `memory-claim` optional key: free-form notes.
pub const CLAIM_KEY_NOTES: &str = "notes";
/// `tier-bytes` key: device-tier bytes.
pub const CLAIM_TIER_KEY_DEVICE: &str = "device";
/// `tier-bytes` key: host-tier bytes.
pub const CLAIM_TIER_KEY_HOST: &str = "host";

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

// ================================================================================================
// Resource handles: the bit layout, the kinds, and the three resource classes (ABI §7.1/§7.2).
//
// A handle is an opaque nonzero `u64` naming a host-side resource; `0` is never a live handle. The
// v1 bit layout (`daemon-vhc-host::handle`) is retained verbatim and lifted here so every side of
// the boundary — the host arena, the guest SDK, the journal/replay verifier — packs and inspects
// handles against one source of truth. Kinds 1–7 were the retired bridge's resources (their
// numbering is permanent and never reused); kinds
// 8/9/10 (BufferHandle / StreamHandle / OpId) are Phase B (this track) — reserved-numbered here so
// the buffer + completion layers land without renumbering. Growth is additive (kinds 11–255
// reserved); a kind's class is permanent.
// ================================================================================================

/// The bit position of the 8-bit `kind` field in a handle (ABI §7.2).
pub const HANDLE_KIND_SHIFT: u32 = 56;
/// The bit position of the 24-bit `generation` field in a handle (ABI §7.2).
pub const HANDLE_GENERATION_SHIFT: u32 = 32;
/// The 24-bit mask applied to a handle's `generation` field (ABI §7.2).
pub const HANDLE_GENERATION_MASK: u64 = 0x00FF_FFFF;
/// The largest representable handle generation (24 bits). A slot whose generation would wrap past
/// this MUST be **permanently retired** (never returned to the free list), making ABA reuse of a
/// `(kind, generation, index)` triple impossible within an instance (ABI §7.1).
pub const HANDLE_MAX_GENERATION: u32 = 0x00FF_FFFF;
/// The 32-bit mask applied to a handle's `index` field (1-based, ABI §7.2).
pub const HANDLE_INDEX_MASK: u64 = 0xFFFF_FFFF;

/// Handle kind 1: a native step tensor — retired bridge, slice-class (permanent, ABI §7.2).
pub const HANDLE_KIND_STEP_TENSOR_NATIVE: u8 = 1;
/// Handle kind 2: a det step tensor — retired bridge, slice-class (permanent, ABI §7.2).
pub const HANDLE_KIND_STEP_TENSOR_DET: u8 = 2;
/// Handle kind 3: a param — retired bridge, registered-class (permanent, ABI §7.2).
pub const HANDLE_KIND_PARAM: u8 = 3;
/// Handle kind 4: a persistent — retired bridge, registered-class (permanent, ABI §7.2).
pub const HANDLE_KIND_PERSISTENT: u8 = 4;
/// Handle kind 5: a det persistent — retired bridge, registered-class (permanent, ABI §7.2).
pub const HANDLE_KIND_DET_PERSISTENT: u8 = 5;
/// Handle kind 6: an update container — retired bridge, instance-class (permanent, ABI §7.2).
pub const HANDLE_KIND_UPDATE_CONTAINER: u8 = 6;
/// Handle kind 7: a batch — retired bridge, instance-class (permanent, ABI §7.2).
pub const HANDLE_KIND_BATCH: u8 = 7;
/// Handle kind 8: a [`BufferHandle`](https://example.invalid) — the sealed, host-owned byte region
/// every world speaks (ABI §7.4); instance-class. **Phase B (this track).**
pub const HANDLE_KIND_BUFFER: u8 = 8;
/// Handle kind 9: a stream handle for a direct peer stream (ABI §7.2); instance-class. **Phase B.**
pub const HANDLE_KIND_STREAM: u8 = 9;
/// Handle kind 10: an `OpId` — an outstanding async operation completing via `Event::Completion`
/// (ABI §7.2/§7.5); instance-class. **Phase B (this track).**
pub const HANDLE_KIND_OP_ID: u8 = 10;

/// The three resource classes (ABI §7.1): the class fixes a handle's lifetime, its generation
/// behavior, and its restart semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceClass {
    /// Registered in `da_init`; lives the run instance; handles re-derived deterministically from
    /// 1-based registration order (generation 0) after any restart (ABI §7.1).
    Registered,
    /// Lives until explicit release or instance end; a generational handle from a dead instance
    /// traps `StaleHandle`; re-acquired through capability calls (ABI §7.1).
    Instance,
    /// Lives until the current event slice ends; invalidated wholesale at each slice boundary
    /// (ABI §7.1).
    Slice,
}

/// The [`ResourceClass`] of a handle `kind`, or `None` if `kind` is unassigned (ABI §7.1/§7.2).
///
/// The kind→class mapping is permanent: kinds 1–2 are slice-class, 3–5 registered-class, 6–10
/// instance-class (buffers/streams/`OpId`s share the instance class with the retired bridge kinds).
#[must_use]
pub fn handle_class(kind: u8) -> Option<ResourceClass> {
    match kind {
        HANDLE_KIND_STEP_TENSOR_NATIVE | HANDLE_KIND_STEP_TENSOR_DET => Some(ResourceClass::Slice),
        HANDLE_KIND_PARAM | HANDLE_KIND_PERSISTENT | HANDLE_KIND_DET_PERSISTENT => {
            Some(ResourceClass::Registered)
        }
        HANDLE_KIND_UPDATE_CONTAINER
        | HANDLE_KIND_BATCH
        | HANDLE_KIND_BUFFER
        | HANDLE_KIND_STREAM
        | HANDLE_KIND_OP_ID => Some(ResourceClass::Instance),
        _ => None,
    }
}

/// Pack a `(kind, generation, index)` triple into the opaque `u64` handle layout (ABI §7.2):
/// `(kind << 56) | ((generation & 0xFF_FFFF) << 32) | index`. `index` is 1-based; a `0` index (with
/// `kind`/`generation` also 0) yields the reserved non-handle `0`.
#[must_use]
pub const fn pack_handle(kind: u8, generation: u32, index: u32) -> u64 {
    ((kind as u64) << HANDLE_KIND_SHIFT)
        | (((generation as u64) & HANDLE_GENERATION_MASK) << HANDLE_GENERATION_SHIFT)
        | (index as u64)
}

/// The 8-bit `kind` field of a handle (ABI §7.2).
#[must_use]
pub const fn handle_kind(handle: u64) -> u8 {
    (handle >> HANDLE_KIND_SHIFT) as u8
}

/// The 24-bit `generation` field of a handle (ABI §7.2).
#[must_use]
pub const fn handle_generation(handle: u64) -> u32 {
    ((handle >> HANDLE_GENERATION_SHIFT) & HANDLE_GENERATION_MASK) as u32
}

/// The 32-bit 1-based `index` field of a handle (ABI §7.2).
#[must_use]
pub const fn handle_index(handle: u64) -> u32 {
    (handle & HANDLE_INDEX_MASK) as u32
}

// ================================================================================================
// The async completion-result wire vocabulary (ABI §7.5): the numeric assignments the completion
// codec (`daemon-vhc-host::v2`), the guest SDK, and the journal/replay verifier must agree on.
//
// Any capability call that cannot complete immediately returns an `OpId` (kind 10) and completes
// via `Event::Completion(op, result)` (event tag 6). None of it is linked at Phase A, but §7.5
// FIXES the wire encoding now so journals (tag 14, opaque `bstr`) and SDKs are stable. The grammar
// of the decoded result bytes is [`COMPLETION_RESULT_CDDL`]; these constants are its numeric
// assignments. The variant set is additive within major 2; unknown `comp-error` codes fail closed
// (ABI §5.2).
// ================================================================================================

/// The normative `completion-result` wire grammar (ABI §7.5): the decoded shape of a
/// `Event::Completion` result and of a journal tag-14 `completion-rec.result` byte string.
///
/// Root rule `completion-result`. It MUST validate as-is under `cddl-cat`; the `completion_grammar`
/// test asserts it is machine-valid and that a representative of each variant validates. The journal
/// stores these bytes opaquely (`completion-rec.result: bstr`, [`JOURNAL_CDDL`]).
pub const COMPLETION_RESULT_CDDL: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/completion.cddl"));

/// `completion-result` variant discriminant 0: success (ABI §7.5 `[0, success-payload]`).
pub const COMPLETION_RESULT_OK: u64 = 0;
/// `completion-result` variant discriminant 1: failure (ABI §7.5 `[1, comp-error]`).
pub const COMPLETION_RESULT_ERR: u64 = 1;

/// `comp-error` code 0: the operation was cancelled (`vhc@2::cancel`, ABI §7.5).
pub const COMP_ERR_CANCELLED: u64 = 0;
/// `comp-error` code 1: the network destination was unreachable (ABI §7.5).
pub const COMP_ERR_NET_UNREACHABLE: u64 = 1;
/// `comp-error` code 2: the operation timed out (guest timer policy surfaces here, ABI §7.5).
pub const COMP_ERR_TIMEOUT: u64 = 2;
/// `comp-error` code 3: the payload store refused the operation (ABI §7.5).
pub const COMP_ERR_STORE_REFUSED: u64 = 3;
/// `comp-error` code 4: a fetched/verified byte range did not match its content hash (ABI §7.5).
pub const COMP_ERR_HASH_MISMATCH: u64 = 4;
/// `comp-error` code 5: a stream write exceeded its writable credit (ABI §7.5, credit flow control).
pub const COMP_ERR_CREDIT_EXHAUSTED: u64 = 5;
/// `comp-error` code 6: the peer closed the stream (ABI §7.5).
pub const COMP_ERR_PEER_CLOSED: u64 = 6;
/// `comp-error` code 7: a grant/quota bound was exhausted (ABI §7.5).
pub const COMP_ERR_GRANT_EXHAUSTED: u64 = 7;
/// `comp-error` code 8: a **deferred device execution error** surfaced at a `compute@2`
/// export/readback completion (CUDA-style deferred semantics, architecture §3.3; the C1 compute
/// minor's addition — additive per §7.5's "8–63 reserved (additive by minor)"). The trap-shaped
/// twin (a device error at a *fence*) is `TrapCode::ComputeFault`; a stale/unknown tensor handle
/// is never this code — it is the typed `StaleHandle`/`InvalidHandle` trap at the call (ABI §15).
pub const COMP_ERR_DEVICE: u64 = 8;
/// `comp-error` codes at or above this are reserved (additive by minor); a guest MUST fail closed on
/// an unknown code (ABI §5.2/§7.5).
pub const COMP_ERR_RESERVED_MIN: u64 = 9;

/// The stable machine-readable slug for a `comp-error` code, or `None` if `code` is reserved/unknown
/// (ABI §7.5). Used to render the `detail`-free failure category in host logs / node surfaces.
#[must_use]
pub fn comp_err_slug(code: u64) -> Option<&'static str> {
    match code {
        COMP_ERR_CANCELLED => Some("Cancelled"),
        COMP_ERR_NET_UNREACHABLE => Some("NetUnreachable"),
        COMP_ERR_TIMEOUT => Some("Timeout"),
        COMP_ERR_STORE_REFUSED => Some("StoreRefused"),
        COMP_ERR_HASH_MISMATCH => Some("HashMismatch"),
        COMP_ERR_CREDIT_EXHAUSTED => Some("CreditExhausted"),
        COMP_ERR_PEER_CLOSED => Some("PeerClosed"),
        COMP_ERR_GRANT_EXHAUSTED => Some("GrantExhausted"),
        COMP_ERR_DEVICE => Some("DeviceError"),
        _ => None,
    }
}

// ================================================================================================
// The grants + manifest vocabulary (ABI §2.3 / §2.6): the single canonical shape naming everything
// a role-instance may reach.
//
// This is the vocabulary B1 extends for the Phase-B worlds — per-grant bounds (net topics/rates/
// payload bytes via `grant-bound`), buffer quota (`buffer-req`), and advisory queue depths
// (`event-caps`) — additively on the v1 schema (refactor §6); the full envelope reshape is D0. It is
// the vocabulary B2/B3 consume: the guest SDK authors a `manifest`, the host derives the admitted
// `grants-doc` as `lane ∩ envelope ∩ owner ∩ manifest` (§2.6), and both key against these exact
// field names. The abi crate is dependency-free / dual-compiled, so it holds the grammar +
// key-name constants; the Rust request/grant types live host-side.
// ================================================================================================

/// The normative grants + manifest grammar (ABI §2.3 `manifest`/`world-req`/`event-caps`/
/// `buffer-req`/`grant-bound`; §2.6 `grants-doc`/`world-grant`/`migration-grant`/`channel-decl`).
///
/// Root rules `grants-doc` (the admitted grants) and `manifest` (the module request). It MUST
/// validate as-is under `cddl-cat`; the `grants_grammar` test asserts it is machine-valid and that
/// representatives of both roots validate. The admitted `grants-doc` bytes are journaled verbatim in
/// the run header ([`JOURNAL_CDDL`] tag 0 `grants`).
pub const GRANTS_CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/grants.cddl"));

/// The grants-document schema version this contract defines (`grants-doc.version`, ABI §2.6).
pub const GRANTS_DOC_VERSION: u64 = 1;

// -- `grant-bound` keys (ABI §2.3): the shared per-grant bound vocabulary ------------------------

/// `grant-bound` key: per-item byte ceiling (publish payload, readback value, payload bytes) (§2.3).
pub const GRANT_BOUND_KEY_MAX_BYTES: &str = "max_bytes";
/// `grant-bound` key: per-event-slice call ceiling for this grant (§2.3).
pub const GRANT_BOUND_KEY_MAX_PER_SLICE: &str = "max_per_slice";
/// `grant-bound` key: sustained rate ceiling (token bucket, per minute) (§2.3).
pub const GRANT_BOUND_KEY_RATE_PER_MIN: &str = "rate_per_min";
/// `grant-bound` key: concurrent-operation ceiling — the Phase-B completion outstanding cap (§2.3).
pub const GRANT_BOUND_KEY_MAX_OUTSTANDING: &str = "max_outstanding";
/// `grant-bound` key: the enumerated allowed values (topics, dataset hashes, sources) (§2.3).
pub const GRANT_BOUND_KEY_VALUES: &str = "values";

// -- `world-req` / `world-grant` keys (ABI §2.3 / §2.6) ------------------------------------------

/// `world-req` key: the world/namespace name (`"vhc"`/`"net"`/`"sys"`/`"data"`/`"compute"`).
pub const WORLD_KEY_WORLD: &str = "world";
/// `world-req`/`world-grant` key: the requested/admitted namespace minor (§2.3/§2.6).
pub const WORLD_KEY_MINOR: &str = "minor";
/// `world-req` key: the requested per-grant bounds map (§2.3).
pub const WORLD_KEY_GRANTS: &str = "grants";
/// `world-grant` key: the admitted per-grant bounds map (§2.6).
pub const WORLD_GRANT_KEY_BOUNDS: &str = "bounds";

// -- `buffer-req` keys (ABI §2.3): the live-resource + linear-memory-crossing quotas -------------

/// `buffer-req` key: standing live-resource ceiling across all instance-class handles (§2.3/§7.3).
pub const BUFFER_REQ_KEY_MAX_LIVE_HANDLES: &str = "max_live_handles";
/// `buffer-req` key: standing live-buffer byte ceiling — the Phase-B buffer quota (§2.3/§7.4).
pub const BUFFER_REQ_KEY_MAX_LIVE_BYTES: &str = "max_live_bytes";
/// `buffer-req` key: per-slice ceiling on bytes crossing into linear memory via `read_into` (§2.3).
pub const BUFFER_REQ_KEY_MAX_READBACK_BYTES: &str = "max_readback_bytes";

// -- `event-caps` keys + advisory class names (ABI §2.3 / §4.7): advisory queue depths ------------

/// `event-caps` per-class key: the declared advisory queue depth (§2.3).
pub const EVENT_CAP_KEY_DEPTH: &str = "depth";
/// `event-caps` per-class key: the fixed coalescing rule (`COALESCE_*`) (§2.3).
pub const EVENT_CAP_KEY_COALESCE: &str = "coalesce";
/// `event-caps` class key: the `PayloadReady` advisory class (dedup-by-hash, §4.7).
pub const EVENT_CLASS_PAYLOAD_READY: &str = "payload-ready";
/// `event-caps` class key: the `Timer` advisory class (latest-wins, §4.7).
pub const EVENT_CLASS_TIMER: &str = "timer";
/// `event-caps` class key: the gossip advisory class (drop-oldest, §4.7).
pub const EVENT_CLASS_GOSSIP: &str = "gossip";

#[cfg(test)]
mod tests {
    use super::*;

    fn ns_set(ns: &[&'static str]) -> BTreeSet<&'static str> {
        ns.iter().copied().collect()
    }

    #[test]
    fn only_major_2_is_implemented() {
        assert_eq!(DA_ABI_MAJOR_V2, 2);
        // Major 1 carries no driver (AbiUnsupportedMajor); an unimplemented major (e.g. a
        // future 3) stays a clean AbiUnsupportedMajor too.
        assert_eq!(HOST_IMPLEMENTED_MAJORS, &[2]);
        assert_eq!(host_minor_for(DA_ABI_MAJOR), None);
        // B1 bumped the major-2 minor to 1 with Completion delivery; C1 bumped it to 2 with
        // Fence delivery + the compute@2 shim (the ratified coupled-to-the-working-driver path,
        // ABI §1.4/§4.6); minor 3 stays AbiMinorTooNew.
        assert_eq!(host_minor_for(DA_ABI_MAJOR_V2), Some(COMPUTE_MINOR_V2));
        assert_eq!(host_minor_for(3), None);
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
    fn a_tabi_importing_module_still_selects_v2_then_refuses_bridge_retired() {
        // Candidate selection is a pure shape read: a `vhc@2` import makes the candidate V2 even
        // when the retired bridge namespace is also imported — the REFUSAL is import
        // validation's, typed BridgeRetired (never a mis-selection, never BadModule).
        let ns = ns_set(&[NS_VHC_V2, NS_SYS_V2, NS_TABI_V1]);
        let exports = ns_set(V2_REQUIRED_EXPORTS);
        assert_eq!(
            select_candidate(&ns, &exports).unwrap(),
            CandidateDriver::V2
        );
        assert_eq!(CandidateDriver::V2.major(), 2);
        let err = validate_imports(&[("vhc@2", "next_event"), ("tabi@1", "add@1")]).unwrap_err();
        assert_eq!(err.code, AbiRefusalCode::BridgeRetired);
        assert_eq!(err.code.slug(), "BridgeRetired");
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
    fn validate_imports_refuses_any_tabi_import_bridge_retired() {
        // The retired-bridge pin at the vocabulary layer: ANY tabi@1 symbol — a formerly-real op
        // or a made-up one alike — is the typed BridgeRetired refusal.
        for sym in ["add@1", "matmul@1", "bogus@1"] {
            let err = validate_imports(&[("tabi@1", sym)]).unwrap_err();
            assert_eq!(err.code, AbiRefusalCode::BridgeRetired, "tabi@1::{sym}");
        }
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
        // Well-formed v2 imports validate cleanly.
        validate_imports(&[("vhc@2", "next_event"), ("sys@2", "now")]).unwrap();
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
        // Kinds 4/5 are JOURNAL-record kinds (stream-read completion bytes, B1; tensor-export
        // completion bytes, C1) — distinct from every call kind and below the bridge journal
        // floor, never callable.
        assert_eq!(READBACK_KIND_STREAM_BYTES, 4);
        assert_eq!(READBACK_KIND_TENSOR_EXPORT, 5);
        assert!(!call_kinds.contains(&READBACK_KIND_STREAM_BYTES));
        assert!(!call_kinds.contains(&READBACK_KIND_TENSOR_EXPORT));
        assert_ne!(READBACK_KIND_TENSOR_EXPORT, READBACK_KIND_STREAM_BYTES);
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
    fn sys_v2_vocab_is_phase_a_plus_crypto_plus_ambient() {
        // The full sys@2 vocabulary is the Phase-A subset + the Phase-B crypto accels + the
        // Phase-B ambient symbols; every sub-list is a subset, and `validate_imports` accepts
        // every member (the Phase-B symbols are minor-1 registry entries, admitted since the B1
        // minor bump).
        for s in SYS_V2_PHASE_A_SYMBOLS {
            assert!(SYS_V2_SYMBOLS.contains(s), "phase-a symbol {s} missing");
        }
        for s in SYS_V2_CRYPTO_SYMBOLS {
            assert!(SYS_V2_SYMBOLS.contains(s), "crypto symbol {s} missing");
        }
        for s in SYS_V2_AMBIENT_B_SYMBOLS {
            assert!(SYS_V2_SYMBOLS.contains(s), "ambient symbol {s} missing");
        }
        assert_eq!(
            SYS_V2_SYMBOLS.len(),
            SYS_V2_PHASE_A_SYMBOLS.len()
                + SYS_V2_CRYPTO_SYMBOLS.len()
                + SYS_V2_AMBIENT_B_SYMBOLS.len()
        );
        validate_imports(&[
            ("sys@2", "hash"),
            ("sys@2", "verify_sig"),
            ("sys@2", "rng_seed"),
            ("sys@2", "device_profile"),
        ])
        .unwrap();
        // The B2 sys@2 Phase-B surface registers at minor 1 (the shared Phase-B minor).
        for sym in ["hash", "verify_sig", "rng_seed", "device_profile"] {
            assert_eq!(v2_symbol_minor(NS_SYS_V2, sym), Some(1), "sys@2::{sym}");
        }
    }

    #[test]
    fn compute_v2_vocab_registers_at_the_phase_c_minor() {
        // ABI §15 / decisions D8: the compute@2 shim symbols register at COMPUTE_MINOR_V2 (the
        // shared Phase-C minor), following the same "registered before host-implemented"
        // convention as the Phase-B sys crypto/ambient symbols.
        assert_eq!(COMPUTE_MINOR_V2, 2);
        assert_eq!(COMPUTE_BURN_VERSION, "0.21.0");
        assert_eq!(
            v2_namespace_symbols(NS_COMPUTE_V2),
            Some(COMPUTE_V2_SYMBOLS)
        );
        for sym in COMPUTE_V2_SYMBOLS {
            assert_eq!(
                v2_symbol_minor(NS_COMPUTE_V2, sym),
                Some(COMPUTE_MINOR_V2),
                "compute@2::{sym} registers at the Phase-C minor"
            );
        }
        // The Fence event tag is the Phase-C-minor compute-queue marker (ABI §4.6/§15): reserved
        // out of the Phase-A closed subset until the compute minor lands.
        assert_eq!(EV_TAG_FENCE, 5);
        assert!(!PHASE_A_EVENT_TAGS.contains(&EV_TAG_FENCE));
        // required_v2_minor lifts a compute import to the Phase-C minor (ABI §1.3 step 5).
        assert_eq!(
            required_v2_minor(&[("vhc@2", "next_event"), ("compute@2", "submit_op")]),
            COMPUTE_MINOR_V2
        );
    }

    #[test]
    fn compute_imports_refused_until_the_phase_c_minor_bump() {
        // Registered-but-not-yet-host-implemented: a compute@2 import is a clean typed refusal
        // (WorldMinorUnsupported naming the introducing minor) until DA_ABI_MINOR_V2 is bumped to
        // COMPUTE_MINOR_V2 at the Phase-C track merge — never a BadModule, never a trap.
        let r = validate_imports(&[("compute@2", "submit_op")]);
        if host_minor_for(DA_ABI_MAJOR_V2).unwrap_or(0) >= COMPUTE_MINOR_V2 {
            r.unwrap();
        } else {
            let err = r.unwrap_err();
            assert_eq!(err.code, AbiRefusalCode::WorldMinorUnsupported);
            assert!(
                err.detail.contains("introduced at minor 2"),
                "{}",
                err.detail
            );
        }
        // An unregistered compute symbol is the same refusal (never BadModule — the namespace is
        // known), and a bogus compute symbol too.
        assert_eq!(
            validate_imports(&[("compute@2", "not_a_compute_symbol")])
                .unwrap_err()
                .code,
            AbiRefusalCode::WorldMinorUnsupported
        );
    }

    #[test]
    fn handle_layout_round_trips_and_masks_fields() {
        // ABI §7.2: (kind << 56) | ((gen & 0xFF_FFFF) << 32) | index; index 1-based, 32 bits.
        let h = pack_handle(HANDLE_KIND_BUFFER, 0x00AB_CDEF, 0x1234_5678);
        assert_eq!(handle_kind(h), HANDLE_KIND_BUFFER);
        assert_eq!(handle_generation(h), 0x00AB_CDEF);
        assert_eq!(handle_index(h), 0x1234_5678);
        // Generation is 24 bits: a value with high bits set is masked down, not aliased into kind.
        let wrapped = pack_handle(HANDLE_KIND_OP_ID, 0xFFFF_FFFF, 1);
        assert_eq!(
            handle_kind(wrapped),
            HANDLE_KIND_OP_ID,
            "gen overflow never bleeds into kind"
        );
        assert_eq!(handle_generation(wrapped), HANDLE_MAX_GENERATION);
        assert_eq!(handle_index(wrapped), 1);
        // `0` is never a live handle (ABI §7.2): the all-zero triple packs to 0.
        assert_eq!(pack_handle(0, 0, 0), 0);
    }

    #[test]
    fn handle_kinds_are_assigned_and_classed_permanently() {
        // Kinds 1..=10 are the assigned set; 8/9/10 are the Phase-B buffer/stream/op-id (ABI §7.2).
        assert_eq!(
            [
                HANDLE_KIND_STEP_TENSOR_NATIVE,
                HANDLE_KIND_STEP_TENSOR_DET,
                HANDLE_KIND_PARAM,
                HANDLE_KIND_PERSISTENT,
                HANDLE_KIND_DET_PERSISTENT,
                HANDLE_KIND_UPDATE_CONTAINER,
                HANDLE_KIND_BATCH,
                HANDLE_KIND_BUFFER,
                HANDLE_KIND_STREAM,
                HANDLE_KIND_OP_ID,
            ],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        // The kind→class mapping (ABI §7.1): slice 1–2, registered 3–5, instance 6–10.
        assert_eq!(
            handle_class(HANDLE_KIND_STEP_TENSOR_NATIVE),
            Some(ResourceClass::Slice)
        );
        assert_eq!(
            handle_class(HANDLE_KIND_STEP_TENSOR_DET),
            Some(ResourceClass::Slice)
        );
        for k in [
            HANDLE_KIND_PARAM,
            HANDLE_KIND_PERSISTENT,
            HANDLE_KIND_DET_PERSISTENT,
        ] {
            assert_eq!(handle_class(k), Some(ResourceClass::Registered));
        }
        for k in [
            HANDLE_KIND_UPDATE_CONTAINER,
            HANDLE_KIND_BATCH,
            HANDLE_KIND_BUFFER,
            HANDLE_KIND_STREAM,
            HANDLE_KIND_OP_ID,
        ] {
            assert_eq!(
                handle_class(k),
                Some(ResourceClass::Instance),
                "buffers/streams/op-ids are instance-class (ABI §7.1)"
            );
        }
        // An unassigned kind has no class (reserved 11–255, ABI §7.2).
        assert_eq!(handle_class(11), None);
        assert_eq!(handle_class(255), None);
    }

    #[test]
    fn completion_result_and_comp_error_codes_are_assigned_and_additive() {
        // ABI §7.5: variant discriminants + the 0..=7 comp-error codes, 8..=63 reserved.
        assert_eq!((COMPLETION_RESULT_OK, COMPLETION_RESULT_ERR), (0, 1));
        let codes = [
            COMP_ERR_CANCELLED,
            COMP_ERR_NET_UNREACHABLE,
            COMP_ERR_TIMEOUT,
            COMP_ERR_STORE_REFUSED,
            COMP_ERR_HASH_MISMATCH,
            COMP_ERR_CREDIT_EXHAUSTED,
            COMP_ERR_PEER_CLOSED,
            COMP_ERR_GRANT_EXHAUSTED,
            COMP_ERR_DEVICE,
        ];
        assert_eq!(codes, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(COMP_ERR_RESERVED_MIN, 9);
        // Every assigned code has a unique, stable slug; reserved/unknown codes have none (fail
        // closed, ABI §5.2/§7.5).
        let mut slugs: Vec<&str> = codes.iter().map(|c| comp_err_slug(*c).unwrap()).collect();
        let n = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "comp-error slugs must be unique");
        assert_eq!(comp_err_slug(COMP_ERR_RESERVED_MIN), None);
        assert_eq!(comp_err_slug(u64::MAX), None);
    }

    #[test]
    fn v2_symbol_registry_is_consistent_and_derives_required_minor() {
        // Registry entries are unique per (ns, symbol) and the introducing minor is permanent.
        let mut keys: Vec<(&str, &str)> = V2_SYMBOL_REGISTRY
            .iter()
            .map(|(ns, sym, _)| (*ns, *sym))
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n, "registry entries must be unique");
        // The minor-0 slice of the registry is exactly the Phase-A vocabulary consts (for sys@2
        // that is the Phase-A subset — the B2 crypto/ambient symbols are minor-1 entries).
        for (ns, vocab) in [
            (NS_VHC_V2, VHC_V2_SYMBOLS),
            (NS_NET_V2, NET_V2_SYMBOLS),
            (NS_SYS_V2, SYS_V2_PHASE_A_SYMBOLS),
        ] {
            let minor0: Vec<&str> = V2_SYMBOL_REGISTRY
                .iter()
                .filter(|(rns, _, m)| *rns == ns && *m == 0)
                .map(|(_, sym, _)| *sym)
                .collect();
            assert_eq!(minor0, vocab.to_vec(), "minor-0 registry slice for {ns}");
        }
        // Every provided sys@2 symbol is registered (full-vocab ↔ registry agreement).
        for sym in SYS_V2_SYMBOLS {
            assert!(
                v2_symbol_minor(NS_SYS_V2, sym).is_some(),
                "sys@2::{sym} missing from the registry"
            );
        }
        // Required-minor derivation (ABI §1.3 step 5): the max introducing minor of the imports.
        assert_eq!(
            required_v2_minor(&[("vhc@2", "next_event"), ("net@2", "publish")]),
            0
        );
        assert_eq!(
            required_v2_minor(&[("vhc@2", "next_event"), ("net@2", "payload_put")]),
            1,
            "a minor-1 import raises the required minor"
        );
        assert_eq!(
            required_v2_minor(&[("bogus@9", "foo")]),
            0,
            "unregistered namespaces contribute no required minor"
        );
        // The B1 minor-1 surface is registered.
        for sym in [
            "create_from",
            "read_into",
            "buffer_len",
            "buffer_release",
            "cancel",
        ] {
            assert_eq!(v2_symbol_minor(NS_VHC_V2, sym), Some(1), "vhc@2::{sym}");
        }
        for sym in [
            "payload_put",
            "payload_get",
            "stream_open",
            "stream_accept",
            "stream_write",
            "stream_read",
        ] {
            assert_eq!(v2_symbol_minor(NS_NET_V2, sym), Some(1), "net@2::{sym}");
        }
        assert_eq!(v2_symbol_minor(NS_VHC_V2, "bogus"), None);
    }

    #[test]
    fn validate_imports_gates_minor1_symbols_on_the_host_minor() {
        // A minor-1 symbol is admitted iff the host's implemented major-2 minor is >= 1; below
        // that it is a typed WorldMinorUnsupported naming the introducing minor (ABI §1.3 step 3).
        let r = validate_imports(&[("net@2", "payload_put")]);
        if host_minor_for(DA_ABI_MAJOR_V2).unwrap_or(0) >= 1 {
            r.unwrap();
        } else {
            let err = r.unwrap_err();
            assert_eq!(err.code, AbiRefusalCode::WorldMinorUnsupported);
            assert!(
                err.detail.contains("introduced at minor 1"),
                "{}",
                err.detail
            );
        }
    }

    #[test]
    fn grant_vocabulary_keys_are_unique_and_stable() {
        // ABI §2.3/§2.6: the structural key names both `manifest` and `grants-doc` consumers key
        // against. They must be unique so B2/B3 (adding their worlds' bounds) reference one set.
        let keys = [
            GRANT_BOUND_KEY_MAX_BYTES,
            GRANT_BOUND_KEY_MAX_PER_SLICE,
            GRANT_BOUND_KEY_RATE_PER_MIN,
            GRANT_BOUND_KEY_MAX_OUTSTANDING,
            GRANT_BOUND_KEY_VALUES,
            WORLD_KEY_WORLD,
            WORLD_KEY_MINOR,
            WORLD_KEY_GRANTS,
            WORLD_GRANT_KEY_BOUNDS,
            BUFFER_REQ_KEY_MAX_LIVE_HANDLES,
            BUFFER_REQ_KEY_MAX_LIVE_BYTES,
            BUFFER_REQ_KEY_MAX_READBACK_BYTES,
            EVENT_CAP_KEY_DEPTH,
            EVENT_CAP_KEY_COALESCE,
            EVENT_CLASS_PAYLOAD_READY,
            EVENT_CLASS_TIMER,
            EVENT_CLASS_GOSSIP,
        ];
        let mut sorted: Vec<&str> = keys.to_vec();
        let n = sorted.len();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "grant vocabulary keys must be unique");
        assert_eq!(GRANTS_DOC_VERSION, 1);
        // The advisory event classes align with their fixed coalescing rules (§4.7): payload-ready
        // dedup-by-hash, timer latest-wins, gossip drop-oldest.
        assert_eq!(COALESCE_DEDUP_HASH, 0);
        assert_eq!(COALESCE_LATEST_WINS, 1);
        assert_eq!(COALESCE_DROP_OLDEST, 2);
    }
}
