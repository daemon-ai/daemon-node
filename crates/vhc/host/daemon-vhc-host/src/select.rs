// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Admission-time ABI selection (ABI Draft 3 §1.3) — the A0 dual-dispatch front door.
//!
//! The normative order, implemented exactly (decisions D2; refactor §5 A0):
//!
//! 1. **Verify before compiling** — the module blob's blake3 vs the envelope pin; a mismatch is
//!    [`AbiRefusalCode::ModuleHashMismatch`] and **no byte of the blob reaches wasmtime**.
//! 2. **Compile and inspect** — read the static import section + export list; select the
//!    **candidate driver** ([`daemon_vhc_abi::select_candidate`]): any `vhc@2` import ⇒ candidate
//!    major 2; `tabi@1`-only imports + the v1 lifecycle exports ⇒ candidate major 1; otherwise
//!    [`AbiRefusalCode::BadModule`].
//! 3. **Derive + validate the compatibility tuple** — every imported symbol must be one the host
//!    provides ([`daemon_vhc_abi::validate_imports`]): unknown symbol in a known namespace ⇒
//!    [`AbiRefusalCode::WorldMinorUnsupported`]; any `tabi@1` import (the retired compute
//!    bridge) ⇒ [`AbiRefusalCode::BridgeRetired`]; unknown namespace ⇒ `BadModule`.
//! 4. **Instantiate with the assessment linker** — a **deny-on-call** linker (every static import
//!    resolves; calling any of them traps — ABI §9.2), under minimal fuel + a tight epoch
//!    deadline. Both candidates read their declaration through the assessment stubs — `da_abi`
//!    is import-free (§1.1).
//! 5. **Cross-check the declaration** — call `da_abi()`; `major` ≠ candidate ⇒
//!    [`AbiRefusalCode::AbiDeclarationMismatch`]; `major` not implemented by this host ⇒
//!    [`AbiRefusalCode::AbiUnsupportedMajor`] — **this is where every well-formed major-1 module
//!    is refused**: candidate major 1 cross-checks clean and then meets the typed refusal, never
//!    a crash, never a silent hang (the synthetic major-1 module pins exactly this). `minor`
//!    above the host's ⇒ [`AbiRefusalCode::AbiMinorTooNew`].
//! 6. **Check required exports** for the selected major ⇒ `BadModule` when missing/mis-typed.
//!
//! Every failure is a **typed admission refusal** ([`AbiRefusal`]) — an `AssessRun`/instantiate
//! *outcome* the node consumes, never a wasm trap, never a runtime `Event::Error`, and never the
//! reused v1 `TrapCode::AbiMismatch` (ABI §1.5; decisions D2). No `da_init`/`da_run`/`da_step`
//! guest code executes on a refusal.

use wasmtime::{ExternType, Linker, Module, Store};

use daemon_vhc_abi::{
    host_minor_for, select_candidate, validate_imports, AbiRefusal, AbiRefusalCode,
    CandidateDriver, V2_REQUIRED_EXPORTS,
};

use crate::runtime::Worker;

/// A successful ABI §1.3 selection: the driver the module gets, plus its cross-checked declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    /// The selected driver (== the import-derived candidate, after the `da_abi` cross-check).
    pub driver: CandidateDriver,
    /// The declared `da_abi` major (cross-checked against the candidate).
    pub major: u32,
    /// The declared `da_abi` minor (≤ the host's minor for the major).
    pub minor: u32,
}

/// The fuel budget for the selection/assessment instantiation + `da_abi` call (ABI §9.2: "minimal
/// fuel and a tight epoch deadline"). `da_abi` is a pure compile-time constant (§1.1); anything
/// that exhausts this is nonconforming and refused.
const ASSESS_FUEL: u64 = 1 << 20;

/// Run the full ABI §1.3 selection order over a module blob.
///
/// `expected_blake3` is the envelope's per-role pin; `None` means the caller has already verified
/// content addressing out-of-band (e.g. the `ArtifactResolver` fetch path, which blake3-verifies
/// on fetch) — step 1 is then a no-op rather than unverified.
///
/// On success the module is a **candidate for the returned driver**; the caller instantiates the
/// *run* instance itself (for v1: the unchanged `Worker::instantiate` path). The instance created
/// here is used only for the declaration cross-check and is discarded (ABI §9.2: assessment
/// instances are never promoted).
///
/// # Errors
///
/// A typed [`AbiRefusal`] per the §1.3/§1.5 tables (see the module docs for the step → code map).
pub fn select_driver(
    worker: &Worker,
    wasm: &[u8],
    expected_blake3: Option<&[u8; 32]>,
) -> Result<Selection, AbiRefusal> {
    // Step 1 — hash-verify BEFORE compile. On mismatch, no byte reaches wasmtime.
    if let Some(pin) = expected_blake3 {
        let got = blake3::hash(wasm);
        if got.as_bytes() != pin {
            return Err(AbiRefusal::new(
                AbiRefusalCode::ModuleHashMismatch,
                format!(
                    "module blob blake3 {} does not match the envelope pin {}",
                    got.to_hex(),
                    hex32(pin)
                ),
            ));
        }
    }

    // Step 2 — compile (validate) + inspect the static import section and export list.
    let module = Module::new(worker.engine(), wasm).map_err(|e| {
        AbiRefusal::new(
            AbiRefusalCode::BadModule,
            format!("module failed wasm validation/compilation: {e}"),
        )
    })?;
    let import_pairs: Vec<(String, String)> = module
        .imports()
        .map(|i| (i.module().to_string(), i.name().to_string()))
        .collect();
    let namespaces = import_pairs
        .iter()
        .map(|(ns, _)| ns.as_str())
        .collect::<std::collections::BTreeSet<&str>>();
    let exports = module
        .exports()
        .map(|e| e.name().to_string())
        .collect::<Vec<String>>();
    let export_set = exports
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<&str>>();
    let candidate = select_candidate(&namespaces, &export_set)?;

    // Step 3 — derive + validate the compatibility tuple against the host's symbol registry.
    let import_refs: Vec<(&str, &str)> = import_pairs
        .iter()
        .map(|(ns, sym)| (ns.as_str(), sym.as_str()))
        .collect();
    validate_imports(&import_refs)?;

    // Steps 4–5 — instantiate with the candidate linker and cross-check `da_abi`.
    let declared = read_declaration(worker, &module, candidate)?;
    let (major, minor) = (declared >> 16, declared & 0xffff);

    if major != candidate.major() {
        return Err(AbiRefusal::new(
            AbiRefusalCode::AbiDeclarationMismatch,
            format!(
                "da_abi declares major {major} but the static import shape selects major {} \
                 (the module's declaration contradicts its own imports)",
                candidate.major()
            ),
        ));
    }
    let Some(host_minor) = host_minor_for(major) else {
        return Err(AbiRefusal::new(
            AbiRefusalCode::AbiUnsupportedMajor,
            format!(
                "module declares abi major {major}, but this host implements only majors {:?}",
                daemon_vhc_abi::HOST_IMPLEMENTED_MAJORS
            ),
        ));
    };
    if minor > host_minor {
        return Err(AbiRefusal::new(
            AbiRefusalCode::AbiMinorTooNew,
            format!("module declares abi {major}.{minor}, host supports {major}.{host_minor}"),
        ));
    }
    // A declared minor below what the imports require is a lying declaration (ABI §1.3 step 5:
    // "a declared `minor` below what the imports require is `AbiDeclarationMismatch`").
    if candidate == CandidateDriver::V2 {
        let required = daemon_vhc_abi::required_v2_minor(&import_refs);
        if minor < required {
            return Err(AbiRefusal::new(
                AbiRefusalCode::AbiDeclarationMismatch,
                format!(
                    "da_abi declares minor {minor} but the static imports require minor \
                     {required} (the declaration is below what the imports need)"
                ),
            ));
        }
    }

    // Step 6 — required exports for the selected major, by name (signatures are enforced by the
    // driver's typed `get_typed_func` calls before any guest code runs).
    let required: &[&str] = match candidate {
        CandidateDriver::V1 => &["da_abi", "da_alloc", "da_free", "da_manifest"],
        CandidateDriver::V2 => V2_REQUIRED_EXPORTS,
    };
    for name in required {
        if !export_set.contains(name) {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!("required export `{name}` is missing for major {}", major),
            ));
        }
    }

    Ok(Selection {
        driver: candidate,
        major,
        minor,
    })
}

/// Steps 4–5's mechanics: instantiate `module` under the deny-on-call assessment linker
/// (ABI §9.2 — every static import resolves at link time to a deterministic stub that traps if
/// invoked) with minimal budgets, call `da_abi()`, discard the instance, and return the packed
/// declaration. `da_abi` is import-free (§1.1), so reading the declaration never touches a stub.
/// Both candidates use this path, so every step-5 refusal is typed — never an instantiation
/// crash.
fn read_declaration(
    worker: &Worker,
    module: &Module,
    _candidate: CandidateDriver,
) -> Result<u32, AbiRefusal> {
    let mut store: Store<()> = Store::new(worker.engine(), ());
    store.set_fuel(ASSESS_FUEL).map_err(|e| {
        AbiRefusal::new(
            AbiRefusalCode::BadModule,
            format!("fuel seeding failed: {e}"),
        )
    })?;
    store.set_epoch_deadline(worker.epoch_ticks_pub());

    let instance = {
        let mut linker: Linker<()> = Linker::new(worker.engine());
        for import in module.imports() {
            let ExternType::Func(func_ty) = import.ty() else {
                return Err(AbiRefusal::new(
                    AbiRefusalCode::BadModule,
                    format!(
                        "non-function import `{}::{}` (memories/globals/tables are not part \
                         of the ABI)",
                        import.module(),
                        import.name()
                    ),
                ));
            };
            let denied = format!(
                "ClaimCapabilityDenied: capability import `{}::{}` called during assessment \
                 (ABI §9.2 deny-on-call stub)",
                import.module(),
                import.name()
            );
            linker
                .func_new(import.module(), import.name(), func_ty, move |_, _, _| {
                    Err(wasmtime::Error::msg(denied.clone()))
                })
                .map_err(|e| {
                    AbiRefusal::new(
                        AbiRefusalCode::BadModule,
                        format!("assessment stub link failed: {e}"),
                    )
                })?;
        }
        linker.instantiate(&mut store, module).map_err(|e| {
            AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!("candidate (assessment) instantiation failed: {e}"),
            )
        })?
    };

    let da_abi = instance
        .get_typed_func::<(), u32>(&mut store, "da_abi")
        .map_err(|_| {
            AbiRefusal::new(
                AbiRefusalCode::BadModule,
                "missing or mis-typed `da_abi` export",
            )
        })?;
    da_abi.call(&mut store, ()).map_err(|e| {
        AbiRefusal::new(
            AbiRefusalCode::BadModule,
            format!("da_abi() must be callable immediately after instantiation and pure: {e}"),
        )
    })
    // `store`/`instance` drop here — the cross-check instance is discarded, never promoted.
}

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
