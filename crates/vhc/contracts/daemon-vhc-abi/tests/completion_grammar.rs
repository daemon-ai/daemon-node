// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Completion-result grammar conformance (ABI companion §7.5).
//!
//! The normative artifact [`daemon_vhc_abi::COMPLETION_RESULT_CDDL`] fixes the decoded shape of an
//! `Event::Completion(op, result)` and of a journal tag-14 `completion-rec.result` byte string
//! (which the journal stores opaquely). Per ABI §7.5 the wire is fixed *before* completions are
//! linked (Phase B), so journals and SDKs stay stable across the phase that makes it real. This
//! dependency-free contract crate owns the grammar and proves it machine-valid: each variant of
//! `completion-result` has a representative CBOR sample that validates, and malformed shapes are
//! rejected. The host-side encode/decode round-trip over the real Rust types lives in
//! `daemon-vhc-host::v2` (the codec that produces these bytes), mirroring how `journal_grammar`
//! proves the §8.3 grammar and `daemon-vhc-observe::journal` owns the record round-trip.

use ciborium::value::Value;
use daemon_vhc_abi::{
    COMPLETION_RESULT_CDDL, COMPLETION_RESULT_ERR, COMPLETION_RESULT_OK, COMP_ERR_CANCELLED,
    COMP_ERR_HASH_MISMATCH,
};

fn enc(v: &Value) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::ser::into_writer(v, &mut b).expect("encode");
    b
}

fn u(n: u64) -> Value {
    Value::Integer(n.into())
}

/// Whether `bytes` validate against the `completion-result` root of the normative grammar.
fn ok(bytes: &[u8]) -> bool {
    cddl_cat::validate_cbor_bytes("completion-result", COMPLETION_RESULT_CDDL, bytes).is_ok()
}

#[test]
fn grammar_is_machine_valid_and_every_variant_validates() {
    // success — a handle payload (uint): `[0, <handle>]`.
    let ok_handle = Value::Array(vec![u(COMPLETION_RESULT_OK), u(0x0800_0000_0000_0001)]);
    assert!(ok(&enc(&ok_handle)), "ok/handle validates");

    // success — a 32-byte content hash payload (payload_put): `[0, h32]`.
    let ok_hash = Value::Array(vec![u(COMPLETION_RESULT_OK), Value::Bytes(vec![7u8; 32])]);
    assert!(ok(&enc(&ok_hash)), "ok/hash validates");

    // success — unit (stream_write / publish-ack / cancel-target): `[0, null]`.
    let ok_unit = Value::Array(vec![u(COMPLETION_RESULT_OK), Value::Null]);
    assert!(ok(&enc(&ok_unit)), "ok/unit validates");

    // failure — a comp-error map, detail-free: `[1, {code}]`.
    let err_bare = Value::Array(vec![
        u(COMPLETION_RESULT_ERR),
        Value::Map(vec![(Value::Text("code".into()), u(COMP_ERR_CANCELLED))]),
    ]);
    assert!(ok(&enc(&err_bare)), "err/bare validates");

    // failure — a comp-error map with detail: `[1, {code, detail}]`.
    let err_detail = Value::Array(vec![
        u(COMPLETION_RESULT_ERR),
        Value::Map(vec![
            (Value::Text("code".into()), u(COMP_ERR_HASH_MISMATCH)),
            (
                Value::Text("detail".into()),
                Value::Text("blake3 mismatch".into()),
            ),
        ]),
    ]);
    assert!(ok(&enc(&err_detail)), "err/detail validates");
}

#[test]
fn grammar_rejects_malformed_completion_results() {
    // Not an array at all.
    assert!(!ok(&enc(&Value::Null)));
    // A failure variant whose second element is not a comp-error map.
    let bad_err = Value::Array(vec![u(COMPLETION_RESULT_ERR), u(0)]);
    assert!(!ok(&enc(&bad_err)), "failure must carry a comp-error map");
    // A comp-error map missing its mandatory `code`.
    let no_code = Value::Array(vec![
        u(COMPLETION_RESULT_ERR),
        Value::Map(vec![(
            Value::Text("detail".into()),
            Value::Text("x".into()),
        )]),
    ]);
    assert!(!ok(&enc(&no_code)), "comp-error requires `code`");
}
