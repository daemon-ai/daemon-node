// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Grants + manifest grammar conformance (ABI companion §2.3 / §2.6).
//!
//! The normative artifact [`daemon_vhc_abi::GRANTS_CDDL`] fixes the single canonical vocabulary a
//! module requests (`manifest`) and the host admits (`grants-doc`). This dependency-free contract
//! crate owns the grammar and proves it machine-valid: representatives of both roots validate, and
//! a request beyond the grammar's shape is rejected. It is the vocabulary B1 extends for the
//! Phase-B worlds and B2/B3 consume; landing it here (before the derivation/consumers) is the
//! "coordinate via the abi contracts" discipline. Mirrors `journal_grammar` / `completion_grammar`.

use ciborium::value::Value;
use daemon_vhc_abi::GRANTS_CDDL;

fn enc(v: &Value) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::ser::into_writer(v, &mut b).expect("encode");
    b
}

fn u(n: u64) -> Value {
    Value::Integer(n.into())
}
fn t(s: &str) -> Value {
    Value::Text(s.into())
}
fn b(bytes: &[u8]) -> Value {
    Value::Bytes(bytes.to_vec())
}
fn map(pairs: Vec<(&str, Value)>) -> Value {
    Value::Map(pairs.into_iter().map(|(k, v)| (t(k), v)).collect())
}

fn ok(root: &str, bytes: &[u8]) -> bool {
    cddl_cat::validate_cbor_bytes(root, GRANTS_CDDL, bytes).is_ok()
}

const H32: [u8; 32] = [7u8; 32];

fn event_caps() -> Value {
    map(vec![
        (
            "payload-ready",
            map(vec![("depth", u(8)), ("coalesce", u(0))]),
        ),
        ("timer", map(vec![("depth", u(1)), ("coalesce", u(1))])),
    ])
}

fn buffer_req() -> Value {
    map(vec![
        ("max_live_handles", u(64)),
        ("max_live_bytes", u(8_000_000_000)),
        ("max_readback_bytes", u(4096)),
    ])
}

fn grant_bound_full() -> Value {
    // Exercise every optional `grant-bound` key (topics via `values`, rates, payload bytes, the
    // completion outstanding cap) — the Phase-B per-world bound vocabulary.
    map(vec![
        ("max_bytes", u(1_048_576)),
        ("max_per_slice", u(4)),
        ("rate_per_min", u(600)),
        ("max_outstanding", u(16)),
        ("values", Value::Array(vec![t("topic-a"), t("topic-b")])),
    ])
}

#[test]
fn manifest_root_validates() {
    let world = map(vec![
        ("world", t("net")),
        ("minor", u(0)),
        ("grants", map(vec![("publish", grant_bound_full())])),
    ]);
    let manifest = map(vec![
        ("name", t("tiny-llama")),
        ("version", t("1.0.0")),
        ("sdk", t("daemon-vhc-sdk")),
        ("abi", u(2 << 16)),
        ("worlds", Value::Array(vec![world])),
        ("custom_ops", Value::Array(vec![])),
        ("channels", Value::Array(vec![u(0)])),
        ("events", event_caps()),
        ("buffers", buffer_req()),
        ("migratable", Value::Bool(false)),
    ]);
    assert!(ok("manifest", &enc(&manifest)), "manifest validates");
}

#[test]
fn grants_doc_root_validates() {
    let channel = map(vec![
        ("id", u(0)),
        ("name", t("control")),
        ("class", u(0)),
        ("direction", u(2)),
        ("max_frame_bytes", u(1_048_576)),
        ("rate_per_min", u(600)),
        ("spool_frames", u(1024)),
        ("replay_window", u(4096)),
        ("per_sender_quota", u(256)),
    ]);
    let world_grant = map(vec![
        ("minor", u(0)),
        ("bounds", map(vec![("payload_put", grant_bound_full())])),
    ]);
    let grants_doc = map(vec![
        ("version", u(1)),
        ("run_id", b(&H32)),
        ("epoch", u(0)),
        ("role", t("trainer")),
        ("instance", u(42)),
        ("lane", t("trainer")),
        ("lane_version", u(1)),
        ("worlds", map(vec![("net", world_grant)])),
        ("custom_ops", Value::Array(vec![])),
        ("channels", Value::Array(vec![channel])),
        ("events", event_caps()),
        ("buffers", buffer_req()),
        ("artifacts", Value::Array(vec![b(&H32)])),
        (
            "migration",
            map(vec![
                ("restore", Value::Bool(true)),
                ("max_sections", u(8)),
                ("max_section_bytes", u(1_000_000)),
            ]),
        ),
    ]);
    assert!(ok("grants-doc", &enc(&grants_doc)), "grants-doc validates");
}

#[test]
fn grammar_rejects_malformed() {
    // A grants-doc missing its mandatory `run_id`.
    let no_run_id = map(vec![
        ("version", u(1)),
        ("epoch", u(0)),
        ("role", t("trainer")),
        ("instance", u(1)),
        ("lane", t("trainer")),
        ("lane_version", u(1)),
        ("worlds", map(vec![])),
        ("custom_ops", Value::Array(vec![])),
        ("channels", Value::Array(vec![])),
        ("events", map(vec![])),
        ("buffers", buffer_req()),
    ]);
    assert!(!ok("grants-doc", &enc(&no_run_id)), "run_id is mandatory");
    // A buffer-req missing a mandatory quota.
    let bad_buffers = map(vec![("max_live_handles", u(1))]);
    assert!(!ok("buffer-req", &enc(&bad_buffers)));
}
