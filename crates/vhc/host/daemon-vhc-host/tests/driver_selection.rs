// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A0 driver-selection conformance (ABI Draft 3 §1.3/§1.5; decisions D2; refactor §5 A0).
//!
//! Every module here is **hand-built** with `wasm-encoder` — precise control over the static
//! import namespaces, the export list, and the `da_abi` constant, with no wasm32 toolchain in the
//! tier-1 lane. The suite pins the A0 acceptance: a v2-major module is cleanly refused by the
//! typed `AbiUnsupportedMajor` (naming the missing A2 event-loop driver); a declaration that
//! contradicts the import shape is `AbiDeclarationMismatch`; a hash mismatch refuses **before
//! compile**; unknown symbols/namespaces split into `WorldMinorUnsupported`/`BadModule` — all as
//! admission refusals ([`daemon_vhc_abi::AbiRefusal`]), never traps.

use daemon_vhc_abi::{AbiRefusalCode, CandidateDriver, DA_ABI_MAJOR_V2};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    Module, TypeSection, ValType,
};

/// Assemble a minimal valid module: `imports` as `(namespace, symbol)` function imports (all typed
/// `() -> i32`), `exports` as named functions (all `() -> i32` returning 0), plus a `da_abi`
/// export returning `da_abi_value`.
fn build_module(imports: &[(&str, &str)], exports: &[&str], da_abi_value: u32) -> Vec<u8> {
    let mut module = Module::new();

    // Type 0: () -> i32 — shared by every import and export in these fixtures.
    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I32]);
    module.section(&types);

    let mut import_sec = ImportSection::new();
    for (ns, sym) in imports {
        import_sec.import(ns, sym, EntityType::Function(0));
    }
    module.section(&import_sec);

    let n_imports = imports.len() as u32;
    let n_funcs = exports.len() as u32 + 1; // + da_abi
    let mut funcs = FunctionSection::new();
    for _ in 0..n_funcs {
        funcs.function(0);
    }
    module.section(&funcs);

    let mut export_sec = ExportSection::new();
    for (i, name) in exports.iter().enumerate() {
        export_sec.export(name, ExportKind::Func, n_imports + i as u32);
    }
    export_sec.export("da_abi", ExportKind::Func, n_imports + exports.len() as u32);
    module.section(&export_sec);

    let mut code = CodeSection::new();
    for _ in 0..exports.len() {
        let mut f = Function::new([]);
        f.instructions().i32_const(0).end();
        code.function(&f);
    }
    let mut da_abi = Function::new([]);
    da_abi.instructions().i32_const(da_abi_value as i32).end();
    code.function(&da_abi);
    module.section(&code);

    module.finish()
}

/// The v1 lifecycle export set (candidate-major-1 shape; no imports needed — ∅ ⊆ {tabi@1}).
const V1_EXPORTS: &[&str] = &[
    "da_alloc",
    "da_free",
    "da_manifest",
    "da_build",
    "da_step",
    "da_inner_update",
    "da_make_update",
    "da_ingest_updates",
];

/// The major-2 required export set minus `da_abi` (which `build_module` always appends).
const V2_EXPORTS: &[&str] = &[
    "da_alloc",
    "da_free",
    "da_manifest",
    "da_claim",
    "da_init",
    "da_run",
];

fn worker() -> Worker {
    Worker::new(EngineConfig::default()).expect("engine")
}

fn pack(major: u32, minor: u32) -> u32 {
    (major << 16) | minor
}

// -- Phase-E sunset FLIP (decisions D5): the v1 shape is refused typed, never admitted ------------
// (Through the dual-driver transition this was the positive "v1 module admitted to the v1 driver"
// pin; the sunset removed the driver, so the SAME well-formed module now meets the clean
// AbiUnsupportedMajor — the candidate/declaration cross-check passes, then the host refuses the
// unimplemented major. The A0 frozen fixture pins this over the real pre-refactor bytes.)

#[test]
fn v1_shaped_module_is_refused_abi_unsupported_major_post_sunset() {
    let wasm = build_module(&[], V1_EXPORTS, pack(1, 0));
    let err = select_driver(&worker(), &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect_err("a v1 module is refused on a post-sunset host");
    assert_eq!(err.code, AbiRefusalCode::AbiUnsupportedMajor);
    assert!(err.detail.contains("major 1"));
}

// -- A2: a well-formed major-2 module is ADMITTED to the event-loop driver (was: A0's clean
// AbiUnsupportedMajor refusal; the expectation flipped in the same commit the driver first ran a
// module end-to-end — refactor §5 A2) --------------------------------------------------------------

#[test]
fn module_selects_the_event_loop_driver() {
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(2, 0));
    let sel = select_driver(&worker(), &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("major 2 selects the A2 event-loop driver");
    assert_eq!(sel.driver, CandidateDriver::V2);
    assert_eq!((sel.major, sel.minor), (2, 0));
}

/// THE RETIRED-BRIDGE PIN: a synthetic major-2 module importing `tabi@1` meets the typed
/// `BridgeRetired` refusal at the §1.3 front door — never a mis-selection to major 1 (ABI §1.2:
/// bridge imports never make a module major-1), never a `BadModule`, never a trap. The offending
/// input is hand-assembled in-test; no recorded bridge artifact exists.
#[test]
fn module_importing_the_retired_bridge_is_refused_bridge_retired() {
    let wasm = build_module(
        &[("vhc@2", "next_event"), ("tabi@1", "batch_size@1")],
        V2_EXPORTS,
        pack(2, 0),
    );
    let err = select_driver(&worker(), &wasm, None)
        .expect_err("a bridge-importing module is refused typed");
    assert_eq!(err.code, AbiRefusalCode::BridgeRetired);
    assert_eq!(err.code.slug(), "BridgeRetired");
    assert!(err.detail.contains("compute@2"), "{}", err.detail);
}

/// The AbiUnsupportedMajor guard survives the flip, one major further out: a future major 3 is
/// not implemented. (Since both import-shape candidates now map to implemented majors, the
/// selection path reaches this refusal only via a future import shape — the constant probe is
/// the honest pin; a declared major 3 over a v2 shape is a declaration mismatch first, asserted
/// by `v2_shape_declaring_major1_is_declaration_mismatch`'s mirror below.)
#[test]
fn future_major_is_not_implemented() {
    assert_eq!(daemon_vhc_abi::host_minor_for(3), None);
    assert!(!daemon_vhc_abi::HOST_IMPLEMENTED_MAJORS.contains(&3));
    // A v2-shaped module declaring major 3: the import-derived candidate (2) wins — declaration
    // mismatch, not a silent admit.
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(3, 0));
    let err = select_driver(&worker(), &wasm, None).expect_err("major 3 contradiction");
    assert_eq!(err.code, AbiRefusalCode::AbiDeclarationMismatch);
}

// -- da_abi cross-check: declaration contradicting the import shape --------------------------------

#[test]
fn v1_shape_declaring_major2_is_declaration_mismatch() {
    // Import shape says v1 (tabi@1-only + lifecycle exports); da_abi lies and says major 2.
    let wasm = build_module(&[], V1_EXPORTS, pack(DA_ABI_MAJOR_V2, 0));
    let err = select_driver(&worker(), &wasm, None).expect_err("contradiction must refuse");
    assert_eq!(err.code, AbiRefusalCode::AbiDeclarationMismatch);
}

#[test]
fn v2_shape_declaring_major1_is_declaration_mismatch() {
    // Import shape says v2 (a vhc@2 import); da_abi declares major 1. The cross-check fires
    // BEFORE the host-support check, so this is a declaration mismatch, not unsupported-major.
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(1, 0));
    let err = select_driver(&worker(), &wasm, None).expect_err("contradiction must refuse");
    assert_eq!(err.code, AbiRefusalCode::AbiDeclarationMismatch);
}

// -- minor gate -----------------------------------------------------------------------------------

#[test]
fn v1_any_minor_is_unsupported_major_post_sunset() {
    // Pre-sunset this pinned AbiMinorTooNew (minor 7 > host v1 minor 0); with major 1 no longer
    // implemented the major check fires first — the refusal is AbiUnsupportedMajor regardless of
    // the declared minor (the minor gate remains pinned by the v2 arms below).
    let wasm = build_module(&[], V1_EXPORTS, pack(1, 7));
    let err = select_driver(&worker(), &wasm, None).expect_err("major 1 is not implemented");
    assert_eq!(err.code, AbiRefusalCode::AbiUnsupportedMajor);
}

// -- the B1 minor-0→1 bump: both-side pins (ABI §1.3 step 5, §1.4) ----------------------------------

#[test]
fn minor1_import_with_minor1_declaration_is_admitted() {
    // The bump's positive selection pin: a module importing the B1 surface and declaring 2.1.
    let wasm = build_module(
        &[("vhc@2", "next_event"), ("net@2", "payload_put")],
        V2_EXPORTS,
        pack(2, 1),
    );
    let sel = select_driver(&worker(), &wasm, None).expect("minor-1 module admitted");
    assert_eq!((sel.major, sel.minor), (2, 1));
}

#[test]
fn minor1_import_with_minor0_declaration_is_declaration_mismatch() {
    // "A declared minor below what the imports require is AbiDeclarationMismatch" (§1.3 step 5):
    // the module imports payload_put (introduced at minor 1) but declares 2.0.
    let wasm = build_module(
        &[("vhc@2", "next_event"), ("net@2", "payload_put")],
        V2_EXPORTS,
        pack(2, 0),
    );
    let err = select_driver(&worker(), &wasm, None).expect_err("lying declaration must refuse");
    assert_eq!(err.code, AbiRefusalCode::AbiDeclarationMismatch);
    assert!(err.detail.contains("require minor 1"), "{}", err.detail);
}

#[test]
fn minor0_declaration_without_minor1_imports_stays_admitted() {
    // Additive discipline: the Phase-A shape is untouched by the bump.
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(2, 0));
    let sel = select_driver(&worker(), &wasm, None).expect("minor-0 module still admitted");
    assert_eq!((sel.major, sel.minor), (2, 0));
}

#[test]
fn minor6_declaration_is_admitted() {
    // The REL-6 bump's positive selection pin (ABI §4.5 minor 6 — Outcome 4 EnvStarved): minor 6
    // assigns an outcome code, no new symbol, so a module importing any existing surface and
    // declaring 2.6 negotiates cleanly. Together with `v2_minor_above_host_is_minor_too_new`
    // (2.7 refused) this is the §1.4 compatibility pair: an older host REFUSES the newer module
    // at the front door rather than ever reading outcome 4 as a generic terminal.
    const {
        assert!(
            daemon_vhc_abi::DA_ABI_MINOR_V2 >= 6,
            "the EnvStarved assignment is implemented"
        );
    }
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(2, 6));
    let sel = select_driver(&worker(), &wasm, None).expect("a minor-6 module is admitted");
    assert_eq!((sel.major, sel.minor), (2, 6));
}

#[test]
fn v2_minor_above_host_is_minor_too_new() {
    // One above the host's implemented minor (derived, not hard-coded — the constant moves with
    // each phase's bump; C1's Phase-C bump took it to 2): AbiMinorTooNew continues to protect
    // the other direction.
    let above = daemon_vhc_abi::DA_ABI_MINOR_V2 + 1;
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(2, above));
    let err = select_driver(&worker(), &wasm, None)
        .expect_err("a declared minor above the host's must refuse");
    assert_eq!(err.code, AbiRefusalCode::AbiMinorTooNew);
}

// -- step 1: hash mismatch refuses BEFORE compile ---------------------------------------------------

#[test]
fn hash_mismatch_refused_before_compile() {
    // Deliberately NOT valid wasm: if compilation ran before the hash check this would surface as
    // BadModule. The typed ModuleHashMismatch proves no byte reached wasmtime (ABI §1.3 step 1).
    let not_wasm = b"definitely not a wasm module";
    let wrong_pin = *blake3::hash(b"some other bytes").as_bytes();
    let err =
        select_driver(&worker(), not_wasm, Some(&wrong_pin)).expect_err("pin mismatch must refuse");
    assert_eq!(err.code, AbiRefusalCode::ModuleHashMismatch);
}

#[test]
fn correct_pin_with_invalid_wasm_is_bad_module() {
    // Control for the test above: the hash gate passes, then compilation fails typed.
    let not_wasm = b"definitely not a wasm module";
    let err = select_driver(&worker(), not_wasm, Some(blake3::hash(not_wasm).as_bytes()))
        .expect_err("invalid wasm must refuse");
    assert_eq!(err.code, AbiRefusalCode::BadModule);
}

// -- step 3: compatibility tuple ---------------------------------------------------------------------

#[test]
fn unknown_symbol_in_known_namespace_is_world_minor_unsupported() {
    let wasm = build_module(
        &[("vhc@2", "next_event"), ("vhc@2", "symbol_from_the_future")],
        V2_EXPORTS,
        pack(2, 0),
    );
    let err = select_driver(&worker(), &wasm, None).expect_err("unknown symbol must refuse");
    assert_eq!(err.code, AbiRefusalCode::WorldMinorUnsupported);
    assert!(err.detail.contains("symbol_from_the_future"));
}

#[test]
fn any_tabi_import_is_bridge_retired() {
    // ANY tabi@1 symbol — formerly-real or made-up — is the typed BridgeRetired refusal at
    // import validation, on a v1-shaped candidate too.
    let wasm = build_module(&[("tabi@1", "bogus_op@1")], V1_EXPORTS, pack(1, 0));
    let err = select_driver(&worker(), &wasm, None).expect_err("tabi import must refuse");
    assert_eq!(err.code, AbiRefusalCode::BridgeRetired);
}

#[test]
fn unknown_namespace_is_bad_module() {
    let wasm = build_module(
        &[("wasi_snapshot_preview1", "fd_write")],
        V1_EXPORTS,
        pack(1, 0),
    );
    let err = select_driver(&worker(), &wasm, None).expect_err("unknown namespace must refuse");
    assert_eq!(err.code, AbiRefusalCode::BadModule);
}

// -- step 2: no recognizable driver shape ------------------------------------------------------------

#[test]
fn tabi_only_without_v1_lifecycle_is_bad_module() {
    let wasm = build_module(&[], &["da_alloc", "da_free"], pack(1, 0));
    let err = select_driver(&worker(), &wasm, None).expect_err("shapeless module must refuse");
    assert_eq!(err.code, AbiRefusalCode::BadModule);
}

// -- step 6: required exports for the selected major -------------------------------------------------

#[test]
fn v1_candidate_missing_da_manifest_is_unsupported_major_post_sunset() {
    // Pre-sunset this pinned the step-6 BadModule ("required export `da_manifest` missing");
    // post-sunset the step-5 major refusal fires first — the v1 candidate never reaches the
    // export check (the v2 step-6 arm keeps that stage pinned via V2_REQUIRED_EXPORTS tests).
    let wasm = build_module(
        &[],
        &[
            "da_alloc",
            "da_free",
            // no da_manifest
            "da_build",
            "da_step",
            "da_inner_update",
            "da_make_update",
            "da_ingest_updates",
        ],
        pack(1, 0),
    );
    let err = select_driver(&worker(), &wasm, None).expect_err("v1 candidate must refuse");
    assert_eq!(err.code, AbiRefusalCode::AbiUnsupportedMajor);
}

// -- refusals are admission outcomes, not traps ------------------------------------------------------

#[test]
fn refusals_never_reuse_the_v1_trap_slug_vocabulary() {
    // The split codes are their own taxonomy: none of them stringifies to the retained v1 trap
    // umbrella `AbiMismatch` (decisions D2: never a reused TrapCode::AbiMismatch). Since A2
    // admits well-formed major-2 modules, the refused probe is a minor-too-new declaration.
    let wasm = build_module(&[("vhc@2", "next_event")], V2_EXPORTS, pack(2, 9));
    let err = select_driver(&worker(), &wasm, None).unwrap_err();
    assert_eq!(err.code, AbiRefusalCode::AbiMinorTooNew);
    assert_ne!(err.code.slug(), "AbiMismatch");
    assert_ne!(
        err.code.slug(),
        daemon_vhc_host::TrapCode::AbiMismatch.slug()
    );
}
