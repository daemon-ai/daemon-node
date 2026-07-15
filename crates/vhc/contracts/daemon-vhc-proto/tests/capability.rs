// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capability subset admission (TDD PROTO-12, spec §6.5/§16).

use daemon_swarm_proto::capability::CapabilitySet;
use daemon_swarm_proto::to_canonical_vec;

fn advertised() -> CapabilitySet {
    CapabilitySet::from_tokens([
        "tensor-abi@1",
        "rmsnorm@1",
        "flash_attn@1",
        "adamw_step@1",
        "topk_chunk@1",
        "absmax_pack@1",
        "det_chunk_scatter_add@1",
        "det_sum@1",
        "det_axpy_param@1",
    ])
    .unwrap()
}

#[test]
fn assess_subset_ok() {
    let required =
        CapabilitySet::from_tokens(["tensor-abi@1", "adamw_step@1", "det_sum@1"]).unwrap();
    assert!(advertised().admits(&required).is_ok());
    assert!(advertised().missing(&required).is_empty());
}

#[test]
fn assess_missing_op_rejected() {
    // A required op the peer does not advertise, and a version-major mismatch, both fail.
    let required = CapabilitySet::from_tokens([
        "tensor-abi@1",
        "grassmann_refresh@1", // not advertised at all
        "det_sum@2",           // advertised only at major 1
    ])
    .unwrap();
    let err = advertised().admits(&required).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("grassmann_refresh@1"), "{msg}");
    assert!(msg.contains("det_sum@2"), "{msg}");

    let missing = advertised().missing(&required);
    assert_eq!(missing.len(), 2);
}

#[test]
fn capability_set_is_canonical_token_array() {
    // Wire form is a sorted array of name@version tokens, independent of insertion order.
    let a = CapabilitySet::from_tokens(["b@1", "a@1", "a@2"]).unwrap();
    let b = CapabilitySet::from_tokens(["a@2", "a@1", "b@1"]).unwrap();
    assert_eq!(to_canonical_vec(&a).unwrap(), to_canonical_vec(&b).unwrap());
}
