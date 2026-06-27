//! Property-based CDDL conformance (the comprehensive drift gate).
//!
//! `tests/conformance.rs` validates representative, real fixtures. This test goes further: it uses
//! the feature-gated `Arbitrary` derives to synthesize values across the *entire* `ApiRequest` /
//! `ApiResponse` variant space (and field edge cases), serializes each with `ciborium`, and asserts
//! the bytes validate against the authoritative `daemon-api.cddl`. Any Rust shape the schema does
//! not describe becomes a failing case.
//!
//! Gated on the `arbitrary` feature so the `arbitrary` dependency never ships in production builds.
//! Run with: `cargo test -p daemon-api --features arbitrary`.
#![cfg(feature = "arbitrary")]

use arbitrary::{Arbitrary, Unstructured};
use cddl_cat::context::BasicContext;
use cddl_cat::flatten::flatten_from_str;
use daemon_api::{ApiRequest, ApiResponse};
use proptest::prelude::*;

const CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/daemon-api.cddl"));

thread_local! {
    // Flatten the CDDL once per test thread instead of reparsing on every value.
    static CTX: BasicContext = BasicContext::new(
        flatten_from_str(CDDL).expect("daemon-api.cddl should flatten"),
    );
}

/// Validate one serialized value against `root`. Returns the validation error message, if any.
fn validate(root: &str, buf: &[u8]) -> Result<(), String> {
    CTX.with(|ctx| {
        let rule = ctx
            .rules
            .get(root)
            .unwrap_or_else(|| panic!("missing root rule `{root}`"));
        let value: ciborium::value::Value =
            ciborium::from_reader(buf).map_err(|e| format!("cbor decode: {e}"))?;
        cddl_cat::cbor::validate_cbor(rule, &value, ctx).map_err(|e| format!("{e:?}"))
    })
}

/// Build an arbitrary value, serialize it, and validate. Cases without enough entropy or that fail
/// to serialize are skipped (they are not conformance failures).
fn check<T>(root: &str, bytes: &[u8]) -> Result<(), TestCaseError>
where
    T: for<'a> Arbitrary<'a> + serde::Serialize + std::fmt::Debug,
{
    let mut u = Unstructured::new(bytes);
    let Ok(value) = T::arbitrary(&mut u) else {
        return Ok(());
    };
    let mut buf = Vec::new();
    if ciborium::ser::into_writer(&value, &mut buf).is_err() {
        return Ok(());
    }
    validate(root, &buf)
        .map_err(|e| TestCaseError::fail(format!("{root} rejected an arbitrary {value:?}: {e}")))
}

/// Enumerate ALL failing variants in one run (a development aid; `#[ignore]`d so CI relies on the
/// proptest cases below). Run with: `cargo test -p daemon-api --features arbitrary -- --ignored
/// audit_all_failing_variants --nocapture`.
#[test]
#[ignore]
fn audit_all_failing_variants() {
    use std::collections::BTreeMap;

    fn first_token(dbg: &str) -> String {
        dbg.split([' ', '(', '{']).next().unwrap_or(dbg).to_string()
    }
    fn audit<T>(root: &str, fails: &mut BTreeMap<String, String>)
    where
        T: for<'a> Arbitrary<'a> + serde::Serialize + std::fmt::Debug,
    {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..150_000u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i);
            let len = 24 + (state as usize % 600);
            let mut bytes = vec![0u8; len];
            let mut s = state;
            for b in bytes.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *b = s as u8;
            }
            let mut u = Unstructured::new(&bytes);
            let Ok(value) = T::arbitrary(&mut u) else {
                continue;
            };
            let mut buf = Vec::new();
            if ciborium::ser::into_writer(&value, &mut buf).is_err() {
                continue;
            }
            if let Err(e) = validate(root, &buf) {
                let key = format!("{root}::{}", first_token(&format!("{value:?}")));
                fails
                    .entry(key)
                    .or_insert_with(|| format!("{e} | {value:?}"));
            }
        }
    }
    let mut fails = BTreeMap::new();
    audit::<ApiRequest>("api-request", &mut fails);
    audit::<ApiResponse>("api-response", &mut fails);
    eprintln!("\n==== {} failing variants ====", fails.len());
    for (k, v) in &fails {
        let v = if v.len() > 900 { &v[..900] } else { v.as_str() };
        eprintln!("{k}\n    {v}\n");
    }
    assert!(
        fails.is_empty(),
        "{} variants fail CDDL validation",
        fails.len()
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn arbitrary_api_request_matches_cddl(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        check::<ApiRequest>("api-request", &bytes)?;
    }

    #[test]
    fn arbitrary_api_response_matches_cddl(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        check::<ApiResponse>("api-response", &bytes)?;
    }
}
