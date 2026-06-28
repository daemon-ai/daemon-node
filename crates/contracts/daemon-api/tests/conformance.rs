//! CDDL conformance gate.
//!
//! The Rust serde types are the source of truth for the wire format; `daemon-api.cddl` is the
//! single authoritative schema. This test turns any drift between them into a failing build:
//!
//! * positive: every committed fixture (real `ciborium` output emitted by `xtask api-fixtures`)
//!   must validate against the CDDL under its `api-request` / `api-response` root;
//! * negative: clearly-invalid payloads must be rejected, proving the schema is actually
//!   discriminating and not vacuously accepting everything.
//!
//! `cddl-cat` parses the full file in-process, so no external CDDL toolchain is required.

use std::fs;
use std::path::Path;

const CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/daemon-api.cddl"));

fn root_for(fixture_stem: &str) -> &'static str {
    if fixture_stem.starts_with("wire-c2s-") {
        "wire-c2s"
    } else if fixture_stem.starts_with("wire-s2c-") {
        "wire-s2c"
    } else if fixture_stem.starts_with("request-") {
        "api-request"
    } else {
        "api-response"
    }
}

#[test]
fn fixtures_validate_against_cddl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cbor");
    let mut checked = 0usize;
    for entry in fs::read_dir(&dir).expect("read fixtures/cbor") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("cbor") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let root = root_for(stem);
        let bytes = fs::read(&path).unwrap();
        cddl_cat::validate_cbor_bytes(root, CDDL, &bytes).unwrap_or_else(|e| {
            panic!("fixture `{stem}` failed to validate against `{root}`: {e:?}")
        });
        checked += 1;
    }
    assert!(
        checked >= 20,
        "expected the committed fixtures to be present, only validated {checked}"
    );
}

#[test]
fn invalid_payloads_are_rejected() {
    use ciborium::value::Integer;
    use ciborium::Value;

    fn enc(v: &Value) -> Vec<u8> {
        let mut b = Vec::new();
        ciborium::ser::into_writer(v, &mut b).unwrap();
        b
    }
    let int = |n: i64| Value::Integer(Integer::from(n));

    let cases: Vec<(&str, Vec<u8>)> = vec![
        // null is not a member of the api-request union.
        ("null", enc(&Value::Null)),
        // a bare string that is not one of the unit variants ("Health", "Models", ...).
        (
            "unknown unit variant",
            enc(&Value::Text("NotAVariant".into())),
        ),
        // a single-key map whose key is not a known externally-tagged variant.
        (
            "unknown map variant",
            enc(&Value::Map(vec![(
                Value::Text("Bogus".into()),
                Value::Text("x".into()),
            )])),
        ),
        // a known variant (Subscribe) but with a wrong-typed field (`session` must be tstr).
        (
            "wrong field type",
            enc(&Value::Map(vec![(
                Value::Text("Subscribe".into()),
                Value::Map(vec![
                    (Value::Text("session".into()), int(1)),
                    (Value::Text("after_seq".into()), int(0)),
                    (Value::Text("max".into()), int(0)),
                ]),
            )])),
        ),
    ];

    for (label, bytes) in cases {
        let res = cddl_cat::validate_cbor_bytes("api-request", CDDL, &bytes);
        assert!(
            res.is_err(),
            "expected `{label}` to be rejected by the CDDL, but it validated"
        );
    }
}
