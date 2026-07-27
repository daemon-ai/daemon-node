// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Journal record grammar conformance (ABI companion §8.2/§8.3/§8.5, §13).
//!
//! The normative artifact [`daemon_vhc_abi::JOURNAL_CDDL`] MUST validate as-is under `cddl-cat`
//! (ABI §8.3) — tier-1 CI validates every conformance-run journal record against it. This crate is
//! dependency-free on its runtime surface, so it owns the *grammar* and proves it is machine-valid:
//! every §8.3 tag has a representative CBOR sample that validates against the `journal-record` root,
//! and the §8.2 / §8.5 header grammars validate too. The exhaustive serde↔grammar round-trip over
//! the real Rust record types lives in `daemon-vhc-observe::journal` (the substrate that encodes
//! them), mirroring how `daemon-vhc-proto` owns `daemon-vhc.cddl` and validates its own types.

use ciborium::value::Value;
use daemon_vhc_abi::{JOURNAL_CDDL, JOURNAL_RECORD_TAGS};

const H32: [u8; 32] = [7u8; 32];

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

/// `[tag, ord, body]`.
fn rec(tag: u64, ord: u64, body: Value) -> Value {
    Value::Array(vec![u(tag), u(ord), body])
}

fn validate(root: &str, bytes: &[u8]) {
    try_validate(root, bytes).unwrap_or_else(|e| panic!("`{root}` failed to validate: {e}"));
}

/// The fallible form, for the negative assertions: a shape the grammar must *refuse*.
fn try_validate(root: &str, bytes: &[u8]) -> Result<(), String> {
    cddl_cat::validate_cbor_bytes(root, JOURNAL_CDDL, bytes).map_err(|e| format!("{e:?}"))
}

/// One representative record per §8.3 tag, in tag order.
fn sample_records() -> Vec<(u8, Value)> {
    let sidecar_ref = map(vec![("hash", b(&H32)), ("size", u(999_999)), ("seg", u(3))]);
    vec![
        (
            0,
            rec(
                0,
                0,
                map(vec![
                    ("run_id", b(&H32)),
                    ("epoch", u(1)),
                    ("role", t("trainer")),
                    ("instance", u(42)),
                    ("module", b(&H32)),
                    ("abi", u(2 << 16)),
                    (
                        "worlds",
                        Value::Map(vec![(t("vhc"), u(0)), (t("net"), u(0))]),
                    ),
                    ("bridge", Value::Bool(true)),
                    ("manifest", b(b"m")),
                    ("config", b(b"c")),
                    ("grants", b(b"g")),
                    ("claim", b(b"cl")),
                    ("channels", b(b"ch")),
                    ("device", b(b"d")),
                    ("format", u(1)),
                ]),
            ),
        ),
        // tag 0, certification variant: `claim` is ABSENT and the four replacement values plus
        // their digests are mandatory. The legacy sample above proves the other alternative, so
        // both minor-selected shapes are exercised.
        (
            0,
            rec(
                0,
                19,
                map(vec![
                    ("run_id", b(&H32)),
                    ("epoch", u(1)),
                    ("role", t("trainer")),
                    ("instance", u(42)),
                    ("module", b(&H32)),
                    ("abi", u((2 << 16) | 5)),
                    (
                        "worlds",
                        Value::Map(vec![(t("vhc"), u(0)), (t("net"), u(0))]),
                    ),
                    ("bridge", Value::Bool(false)),
                    ("manifest", b(b"m")),
                    ("config", b(b"c")),
                    ("grants", b(b"g")),
                    ("resource_plan", b(b"plan")),
                    ("resource_plan_hash", b(&H32)),
                    ("physical_estimate", b(b"claim")),
                    ("physical_estimate_hash", b(&H32)),
                    ("aggregate_estimate", b(b"aggregate")),
                    ("aggregate_estimate_hash", b(&H32)),
                    ("execution_grant", b(b"grant")),
                    ("execution_grant_hash", b(&H32)),
                    ("channels", b(b"ch")),
                    ("device", b(b"d")),
                    ("format", u(1)),
                ]),
            ),
        ),
        (
            1,
            rec(1, 1, map(vec![("at", u(12)), ("frame", b(b"frame-bytes"))])),
        ),
        // read-back: inline value branch of the group choice.
        (
            2,
            rec(
                2,
                2,
                map(vec![
                    ("src", u(0)),
                    ("kind", u(1)),
                    ("status", u(0)),
                    ("value", b(b"small")),
                ]),
            ),
        ),
        (3, rec(3, 3, map(vec![("now", u(123_456))]))),
        (
            4,
            rec(
                4,
                4,
                map(vec![
                    ("channel", u(0)),
                    ("seq", u(1)),
                    ("hash", b(&H32)),
                    ("frame", b(b"signed-wire-frame")),
                ]),
            ),
        ),
        (
            5,
            rec(
                5,
                5,
                map(vec![("id", u(1)), ("delay", u(1000)), ("armed_at", u(50))]),
            ),
        ),
        (6, rec(6, 6, map(vec![("id", u(1)), ("status", u(0))]))),
        (
            7,
            rec(
                7,
                7,
                map(vec![
                    ("class", u(0)),
                    ("rule", u(0)),
                    ("dropped", map(vec![("hash", b(&H32)), ("seq", u(9))])),
                ]),
            ),
        ),
        (
            8,
            rec(
                8,
                8,
                map(vec![
                    ("paused", Value::Bool(false)),
                    ("duty_pct", u(80)),
                    ("vram_cap_bytes", u(8_000_000_000)),
                ]),
            ),
        ),
        // terminal: outcome branch (kind 0).
        (9, rec(9, 9, map(vec![("kind", u(0)), ("outcome", u(0))]))),
        (
            10,
            rec(10, 10, map(vec![("manifest", b(b"state-manifest"))])),
        ),
        (
            11,
            rec(
                11,
                11,
                map(vec![
                    ("config_hash", b(&H32)),
                    ("grants_hash", b(&H32)),
                    ("status", u(0)),
                ]),
            ),
        ),
        // signed-frame: inline frame branch.
        (
            12,
            rec(
                12,
                12,
                map(vec![
                    ("channel", u(0)),
                    ("seq", u(1)),
                    ("sender", b(&H32)),
                    ("frame", b(b"original-signed-frame")),
                ]),
            ),
        ),
        (
            13,
            rec(
                13,
                13,
                map(vec![("counter", u(0)), ("reason", u(0)), ("at", u(7))]),
            ),
        ),
        (
            14,
            rec(14, 14, map(vec![("op", u(0)), ("result", b(b"r"))])),
        ),
        (15, rec(15, 15, map(vec![("profile", b(b"prof"))]))),
        (
            16,
            rec(
                16,
                16,
                map(vec![("code", t("SpoolExhausted")), ("detail", t("x"))]),
            ),
        ),
        (
            17,
            rec(
                17,
                17,
                map(vec![("segment_blake3", b(&H32)), ("records", u(18))]),
            ),
        ),
        // tag 18: the certification minor's grant-application result.
        (
            18,
            rec(
                18,
                20,
                map(vec![("execution_grant_hash", b(&H32)), ("status", u(0))]),
            ),
        ),
        // read-back: sidecar branch of the group choice (exercise the other alternative).
        (
            2,
            rec(
                2,
                18,
                map(vec![
                    ("src", u(0)),
                    ("kind", u(1)),
                    ("status", u(0)),
                    ("sidecar", sidecar_ref),
                ]),
            ),
        ),
    ]
}

#[test]
fn grammar_is_machine_valid_and_every_tag_validates() {
    let samples = sample_records();
    // Every §8.3 tag has at least one representative that validates as `journal-record`.
    let mut seen: Vec<u8> = samples.iter().map(|(tag, _)| *tag).collect();
    for (tag, value) in &samples {
        validate("journal-record", &enc(value));
        let _ = tag;
    }
    seen.sort_unstable();
    seen.dedup();
    for tag in JOURNAL_RECORD_TAGS {
        assert!(
            seen.contains(tag),
            "tag {tag} has no representative sample in the grammar test"
        );
    }
    assert_eq!(
        JOURNAL_RECORD_TAGS.len(),
        19,
        "the complete §8.3 record set is 19 tags (0..=18); 19..=63 reserved"
    );
}

/// A certification-variant run header MUST NOT carry the legacy `claim` member, and MUST carry
/// every replacement value with its digest. An optional field would not express this: a reader has
/// to be able to tell which statement the header is making, and "absent" is not a statement.
#[test]
fn the_certification_run_header_forbids_the_legacy_claim_and_requires_its_replacements() {
    let base: Vec<(&str, Value)> = vec![
        ("run_id", b(&H32)),
        ("epoch", u(1)),
        ("role", t("trainer")),
        ("instance", u(42)),
        ("module", b(&H32)),
        ("abi", u((2 << 16) | 5)),
        ("worlds", Value::Map(vec![(t("vhc"), u(0))])),
        ("bridge", Value::Bool(false)),
        ("manifest", b(b"m")),
        ("config", b(b"c")),
        ("grants", b(b"g")),
        ("resource_plan", b(b"plan")),
        ("resource_plan_hash", b(&H32)),
        ("physical_estimate", b(b"claim")),
        ("physical_estimate_hash", b(&H32)),
        ("aggregate_estimate", b(b"aggregate")),
        ("aggregate_estimate_hash", b(&H32)),
        ("execution_grant", b(b"grant")),
        ("execution_grant_hash", b(&H32)),
        ("channels", b(b"ch")),
        ("device", b(b"d")),
        ("format", u(1)),
    ];
    validate(
        "certification-run-header-rec",
        &enc(&rec(0, 0, map(base.clone()))),
    );

    // Adding the legacy member back is not a certification header.
    let mut with_legacy = base.clone();
    with_legacy.push(("claim", b(b"cl")));
    assert!(
        try_validate(
            "certification-run-header-rec",
            &enc(&rec(0, 0, map(with_legacy)))
        )
        .is_err(),
        "the certification variant forbids the legacy `claim` member"
    );

    // Dropping any replacement value or digest is not a certification header either.
    for omitted in [
        "resource_plan",
        "resource_plan_hash",
        "physical_estimate",
        "physical_estimate_hash",
        "aggregate_estimate",
        "aggregate_estimate_hash",
        "execution_grant",
        "execution_grant_hash",
    ] {
        let reduced: Vec<(&str, Value)> = base
            .iter()
            .filter(|(k, _)| *k != omitted)
            .cloned()
            .collect();
        assert!(
            try_validate(
                "certification-run-header-rec",
                &enc(&rec(0, 0, map(reduced)))
            )
            .is_err(),
            "`{omitted}` is mandatory in the certification run header"
        );
    }
}

#[test]
fn segment_and_sidecar_header_grammars_validate() {
    let seg = map(vec![
        ("run_id", b(&H32)),
        ("epoch", u(1)),
        ("role", t("trainer")),
        ("instance", u(42)),
        ("module", b(&H32)),
        ("segment", u(0)),
    ]);
    validate("segment-header-body", &enc(&seg));

    let sidecar = map(vec![
        ("run_id", b(&H32)),
        ("epoch", u(1)),
        ("role", t("trainer")),
        ("instance", u(42)),
        ("module", b(&H32)),
        ("ord", u(7)),
        ("hash", b(&H32)),
        ("size", u(1_000_000)),
    ]);
    validate("sidecar-header", &enc(&sidecar));
}

#[test]
fn grammar_rejects_malformed_records() {
    // Not an array at all.
    assert!(
        cddl_cat::validate_cbor_bytes("journal-record", JOURNAL_CDDL, &enc(&Value::Null)).is_err()
    );
    // An unknown tag (64 is outside 0..=17 and the reserved 18..=63 range is not yet assigned).
    let unknown = rec(64, 0, map(vec![("x", u(0))]));
    assert!(
        cddl_cat::validate_cbor_bytes("journal-record", JOURNAL_CDDL, &enc(&unknown)).is_err(),
        "an unassigned tag must not validate"
    );
    // A clock record missing its `now` key.
    let bad_clock = rec(3, 0, map(vec![]));
    assert!(
        cddl_cat::validate_cbor_bytes("journal-record", JOURNAL_CDDL, &enc(&bad_clock)).is_err()
    );
}
