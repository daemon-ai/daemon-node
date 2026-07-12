// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Canonical CBOR conformance (TDD HOST-13, proto half): the consensus-critical encoder is checked
//! against the RFC 8949 Appendix A vectors and adversarial key orders / floats. If any of these
//! drift, every downstream hash and signature drifts with them (spec §5.6).

use ciborium::value::Value;
use daemon_swarm_proto::to_canonical_vec;

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn enc(v: &Value) -> String {
    hex(&to_canonical_vec(v).expect("canonical encode"))
}

fn int(n: i128) -> Value {
    Value::Integer(n.try_into().expect("fits cbor integer"))
}

#[test]
fn rfc8949_integer_vectors() {
    // RFC 8949 Appendix A — shortest-form unsigned/negative integers.
    let cases: &[(i128, &str)] = &[
        (0, "00"),
        (1, "01"),
        (10, "0a"),
        (23, "17"),
        (24, "1818"),
        (25, "1819"),
        (100, "1864"),
        (1000, "1903e8"),
        (1_000_000, "1a000f4240"),
        (1_000_000_000_000, "1b000000e8d4a51000"),
        (-1, "20"),
        (-10, "29"),
        (-100, "3863"),
        (-1000, "3903e7"),
    ];
    for (n, expect) in cases {
        assert_eq!(enc(&int(*n)), *expect, "integer {n}");
    }
    // u64::MAX and the most-negative 64-bit CBOR integer.
    assert_eq!(
        enc(&Value::Integer(u64::MAX.into())),
        "1bffffffffffffffff",
        "u64::MAX"
    );
}

#[test]
fn rfc8949_float_vectors() {
    // RFC 8949 Appendix A — shortest floating-point form that round-trips the value.
    let cases: &[(f64, &str)] = &[
        (0.0, "f90000"),
        (-0.0, "f98000"),
        (1.0, "f93c00"),
        (1.1, "fb3ff199999999999a"),
        (1.5, "f93e00"),
        (65504.0, "f97bff"),
        (100000.0, "fa47c35000"),
        (f64::from(f32::MAX), "fa7f7fffff"),
        (1.0e300, "fb7e37e43c8800759c"),
        (2.0f64.powi(-24), "f90001"),
        (2.0f64.powi(-14), "f90400"),
        (-4.0, "f9c400"),
        (-4.1, "fbc010666666666666"),
        (f64::INFINITY, "f97c00"),
        (f64::NEG_INFINITY, "f9fc00"),
    ];
    for (f, expect) in cases {
        assert_eq!(enc(&Value::Float(*f)), *expect, "float {f}");
    }
    assert_eq!(
        enc(&Value::Float(f64::NAN)),
        "f97e00",
        "NaN → canonical half"
    );
}

#[test]
fn nan_payloads_all_normalize() {
    // Every NaN, regardless of payload/sign, must encode to the one canonical half.
    let quiet = f64::from_bits(0x7ff8_0000_0000_0000);
    let signalling = f64::from_bits(0x7ff0_0000_0000_0001);
    let negative = f64::from_bits(0xfff8_dead_beef_cafe);
    for nan in [quiet, signalling, negative] {
        assert_eq!(enc(&Value::Float(nan)), "f97e00");
    }
}

#[test]
fn adversarial_map_key_order_is_sorted() {
    // Keys presented out of order (and of mixed types/lengths) must come out in RFC 8949 §4.2.1
    // bytewise-lexicographic order of their encodings: 1 (0x01) < 10 (0x0a) < "z" (0x61 7a) <
    // "aa" (0x62 61 61) — shorter text sorts first because its length byte is smaller.
    let map = Value::Map(vec![
        (Value::Text("aa".into()), int(4)),
        (Value::Text("z".into()), int(3)),
        (int(10), int(2)),
        (int(1), int(1)),
    ]);
    assert_eq!(
        enc(&map),
        // map(4) | 01 01 | 0a 02 | 61 7a 03 | 62 6161 04
        "a401010a02617a03626161 04".replace(' ', "")
    );
}

#[test]
fn nested_map_keys_sorted_recursively() {
    let inner_unsorted = Value::Map(vec![
        (Value::Text("b".into()), int(2)),
        (Value::Text("a".into()), int(1)),
    ]);
    let outer = Value::Map(vec![(Value::Text("k".into()), inner_unsorted)]);
    // a1 | 616b | a2 6161 01 6162 02
    assert_eq!(enc(&outer), "a1616ba26161016162 02".replace(' ', ""));
}

#[test]
fn equal_values_encode_identically_regardless_of_source_order() {
    let a = Value::Map(vec![
        (Value::Text("one".into()), int(1)),
        (Value::Text("two".into()), int(2)),
        (Value::Text("three".into()), int(3)),
    ]);
    let b = Value::Map(vec![
        (Value::Text("three".into()), int(3)),
        (Value::Text("two".into()), int(2)),
        (Value::Text("one".into()), int(1)),
    ]);
    assert_eq!(to_canonical_vec(&a).unwrap(), to_canonical_vec(&b).unwrap());
}

#[test]
fn definite_lengths_and_struct_roundtrip() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Demo {
        // Declared out of sorted order on purpose; canonical output sorts by key.
        zebra: u32,
        alpha: String,
        nums: Vec<i32>,
    }
    let d = Demo {
        zebra: 7,
        alpha: "hi".into(),
        nums: vec![1, 2, 3],
    };
    let bytes = to_canonical_vec(&d).unwrap();
    // Map head is major-5 definite length 3 (0xa3); keys sort "nums" (length 4) < "alpha" < "zebra"
    // (both length 5, then bytewise: 0x61 < 0x7a).
    assert_eq!(bytes[0], 0xa3, "definite-length map head");
    let back: Demo = daemon_swarm_proto::from_canonical_slice(&bytes).unwrap();
    assert_eq!(back, d);
}
