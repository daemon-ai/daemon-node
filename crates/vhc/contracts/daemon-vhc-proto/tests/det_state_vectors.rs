// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared det-state chunk-geometry vectors (`tests/fixtures/det-state-geometry-vectors.json`)
//! executed against the normative genesis-authoring rules
//! ([`daemon_vhc_proto::validate_profile_chunk`], [`daemon_vhc_proto::validate_state_chunk_size`],
//! [`daemon_vhc_proto::derive_state_chunk_size`],
//! [`daemon_vhc_proto::validate_checkpoint_cadence`]).
//!
//! The fixture is the **cross-implementation contract** (the seat-lease / roster vector
//! pattern): every conforming genesis authoring/validation seat must reproduce these decisions
//! exactly. The ceremony-geometry vectors are the frozen-model refusal cases: the compression
//! profile's chunk must divide every parameter numel, and the 1536-wide norm parameters make
//! `chunk | 1536` the binding constraint — the profile default 4096 is a refusal, not a default.

use daemon_vhc_proto::{
    derive_state_chunk_size, validate_checkpoint_cadence, validate_profile_chunk,
    validate_state_chunk_size,
};
use serde_json::Value;

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/det-state-geometry-vectors.json"
));

fn u64_of(v: &Value, key: &str) -> u64 {
    v.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("vector field `{key}`"))
}

fn expect_accept(v: &Value) -> bool {
    match v.get("expect").and_then(Value::as_str) {
        Some("accept") => true,
        Some("refuse") => false,
        other => panic!("vector expect: {other:?}"),
    }
}

/// The frozen ceremony parameter layout, re-derived from the fixture's pinned geometry (token
/// embedding; per block attn-norm, wq, wk, wv, wo, ffn-norm, w1, w3, w2; final norm).
fn ceremony_numels(geometry: &Value) -> Vec<u64> {
    let d = u64_of(geometry, "d_model");
    let layers = u64_of(geometry, "n_layers");
    let vocab = u64_of(geometry, "vocab");
    let qdim = u64_of(geometry, "qdim");
    let hidden = u64_of(geometry, "hidden");
    let mut out = vec![vocab * d];
    for _ in 0..layers {
        out.extend([
            d,
            d * qdim,
            d * qdim,
            d * qdim,
            qdim * d,
            d,
            d * hidden,
            d * hidden,
            hidden * d,
        ]);
    }
    out.push(d);
    out
}

#[test]
fn the_rules_reproduce_every_shared_geometry_vector() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("fixture parses");
    let numels = ceremony_numels(fixture.get("ceremony_geometry").expect("geometry"));
    assert_eq!(numels.len(), 2 + 9 * 24, "registration-order entry count");
    assert_eq!(
        numels.iter().sum::<u64>(),
        786_507_264,
        "the frozen ceremony parameter count"
    );

    for v in fixture["profile_chunk_vectors"].as_array().expect("array") {
        let name = v["name"].as_str().expect("name");
        let got = validate_profile_chunk(u64_of(v, "chunk"), &numels);
        assert_eq!(
            got.is_ok(),
            expect_accept(v),
            "profile-chunk vector `{name}`: got {got:?}"
        );
    }

    for v in fixture["state_chunk_size_vectors"]
        .as_array()
        .expect("array")
    {
        let name = v["name"].as_str().expect("name");
        let got = validate_state_chunk_size(u64_of(v, "state_chunk_size"), u64_of(v, "chunk"));
        assert_eq!(
            got.is_ok(),
            expect_accept(v),
            "state-chunk-size vector `{name}`: got {got:?}"
        );
    }

    for v in fixture["derive_state_chunk_size_vectors"]
        .as_array()
        .expect("array")
    {
        let chunk = u64_of(v, "chunk");
        assert_eq!(
            derive_state_chunk_size(chunk),
            u64_of(v, "derived"),
            "derive vector chunk {chunk}"
        );
    }

    for v in fixture["cadence_retention_vectors"]
        .as_array()
        .expect("array")
    {
        let name = v["name"].as_str().expect("name");
        let got = validate_checkpoint_cadence(u64_of(v, "cadence"), u64_of(v, "retention"));
        assert_eq!(
            got.is_ok(),
            expect_accept(v),
            "cadence-retention vector `{name}`: got {got:?}"
        );
    }
}
