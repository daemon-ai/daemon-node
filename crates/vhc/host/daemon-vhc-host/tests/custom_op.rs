// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The custom-op registry conformance gate (tier-1; architecture §3.2, refactor §7).
//
// Fused kernels register HOST-side under versioned names. A module's manifest lists the custom ops
// it needs (`manifest.custom_ops`, ABI §2.3); admission fails CLEANLY — a typed
// `CustomOpUnsupported` refusal, never a trap — on a host that lacks a required one. `flash_attn@1`
// is the first registered fusion and the template for future entries. This gate pins the shared
// ABI vocabulary (the `required ⊆ advertised` contract, coordinated with C1's compute@2 track by
// vocabulary rather than shared code) and the registry's admission behaviour. CPU-deterministic
// (no wasm host, no GPU, no network) — a `vhc-ci-det` citizen.

use daemon_vhc_abi::{
    host_supports_custom_op, AbiRefusalCode, CUSTOM_OP_FLASH_ATTN_V1, HOST_CUSTOM_OPS,
};
use daemon_vhc_host::run::CustomOpRegistry;

#[test]
fn abi_vocabulary_pins_flash_attn_v1_as_the_first_entry() {
    assert_eq!(CUSTOM_OP_FLASH_ATTN_V1, "flash_attn@1");
    assert!(
        HOST_CUSTOM_OPS.contains(&CUSTOM_OP_FLASH_ATTN_V1),
        "flash_attn@1 must be advertised by the host registry"
    );
    // Versioning is part of the name — a bare/other-version name is a distinct, unadvertised op.
    assert!(host_supports_custom_op("flash_attn@1"));
    assert!(!host_supports_custom_op("flash_attn"));
    assert!(!host_supports_custom_op("flash_attn@2"));
    // The refusal code has a stable slug (the node-facing admission-outcome surface, §1.5).
    assert_eq!(
        AbiRefusalCode::CustomOpUnsupported.slug(),
        "CustomOpUnsupported"
    );
}

#[test]
fn registry_default_mirrors_the_abi_vocabulary() {
    let reg = CustomOpRegistry::new();
    // The default host registry advertises exactly the shared ABI vocabulary.
    let mut advertised = reg.advertised();
    advertised.sort_unstable();
    let mut expected: Vec<&str> = HOST_CUSTOM_OPS.to_vec();
    expected.sort_unstable();
    assert_eq!(advertised, expected);
    assert!(reg.supports(CUSTOM_OP_FLASH_ATTN_V1));
}

#[test]
fn admission_admits_advertised_and_refuses_absent_custom_ops() {
    let reg = CustomOpRegistry::new();
    // No custom ops required → admitted; a required advertised op → admitted.
    assert!(reg.admit(&[]).is_ok());
    assert!(reg.admit(&["flash_attn@1".to_string()]).is_ok());

    // A required op the host does not advertise → CLEAN typed refusal naming the offender.
    let err = reg
        .admit(&["flash_attn@1".to_string(), "fused_moe@1".to_string()])
        .unwrap_err();
    assert_eq!(err.code, AbiRefusalCode::CustomOpUnsupported);
    assert!(
        err.detail.contains("fused_moe@1"),
        "refusal names the offending op: {}",
        err.detail
    );

    // A future host build registers a fusion it implements; admission then passes.
    let mut reg = CustomOpRegistry::new();
    reg.register("fused_moe@1");
    assert!(reg.admit(&["fused_moe@1".to_string()]).is_ok());
}
