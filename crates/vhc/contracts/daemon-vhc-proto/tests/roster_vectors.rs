// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared roster-slot fold vectors (`tests/fixtures/roster-vectors.json`) executed against
//! the normative fold ([`RosterSlot::fold`]).
//!
//! The fixture is the **cross-implementation contract**: every conforming roster registry — this
//! fold, the local fake in `daemon-vhc-net`, and the cloud registry's port — must reproduce these
//! decisions and slot transitions exactly. The vectors are structural by construction (the
//! registry never verifies signatures or judges authority), so records are built here with dummy
//! certificates and unset signatures: if the fold ever starts caring about either, these vectors
//! break loudly — which would itself be a posture violation to escalate, never to "fix" by
//! signing the fixtures.

use daemon_vhc_proto::bytes::IrohId;
use daemon_vhc_proto::cert::{RunKeyCertBody, RunKeyCertificate};
use daemon_vhc_proto::{
    Hash, PeerId, RosterDecision, RosterRecord, RosterRecordBody, RosterSlot, Signature,
    CERT_DOMAIN_V2,
};
use serde_json::Value;

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/roster-vectors.json"
));

fn u64_of(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

/// Build a record body from a vector spec layered over the fixture's `record_defaults`.
fn record_body(defaults: &Value, spec: &Value) -> RosterRecordBody {
    let get = |key: &str| spec.get(key).filter(|v| !v.is_null()).or(defaults.get(key));
    let get_u64 = |key: &str| {
        get(key)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("vector record field `{key}`"))
    };
    let get_fill = |key: &str| u8::try_from(get_u64(key)).expect("fill byte");
    let direct_addrs: Vec<String> = match spec.get("direct_addrs").or(defaults.get("direct_addrs"))
    {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().expect("addr string").to_string())
            .collect(),
        other => panic!("vector direct_addrs: {other:?}"),
    };
    let relay_url = match spec.get("relay_url") {
        // An explicit `"relay_url": null` in the spec means "no relay" (the structural case).
        Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        None => defaults
            .get("relay_url")
            .and_then(Value::as_str)
            .map(str::to_string),
        other => panic!("vector relay_url: {other:?}"),
    };
    RosterRecordBody {
        domain: get("domain")
            .and_then(Value::as_str)
            .expect("domain")
            .to_string(),
        run_id: Hash([get_fill("run"); 32]),
        role: get("role").and_then(Value::as_str).expect("role").into(),
        epoch: get_u64("epoch"),
        incarnation: get_u64("incarnation"),
        sender: PeerId([get_fill("sender"); 32]),
        module_hash: Hash([get_fill("module"); 32]),
        endpoint_id: IrohId([get_fill("endpoint"); 32]),
        direct_addrs,
        relay_url,
        issued_at_ms: get_u64("issued_at_ms"),
    }
}

/// Wrap a body as a record with a dummy certificate + unset signature (the fold is structural —
/// it must never read either; see the module doc).
fn record(body: RosterRecordBody) -> RosterRecord {
    let certificate = RunKeyCertificate {
        body: RunKeyCertBody {
            domain: CERT_DOMAIN_V2.to_string(),
            scope: body.cert_scope(),
            run_key: body.sender,
        },
        base_identity: PeerId([0; 32]),
        sig: Signature([0; 64]),
    };
    RosterRecord {
        body,
        certificate,
        sig: Signature([0; 64]),
    }
}

fn decision_kind(d: &RosterDecision) -> &'static str {
    match d {
        RosterDecision::Accepted => "accepted",
        RosterDecision::RejectedStructural { .. } => "rejected_structural",
        RosterDecision::RejectedStale { .. } => "rejected_stale",
    }
}

fn assert_slot_matches(name: &str, step: usize, slot: &RosterSlot, expect: &Value) {
    match (expect.is_null(), &slot.record) {
        (true, None) => {}
        (true, Some(r)) => panic!(
            "{name}[{step}]: expected an empty slot, stored (incarnation {}, issued {})",
            r.body.incarnation, r.body.issued_at_ms
        ),
        (false, None) => panic!("{name}[{step}]: expected a stored record, slot is empty"),
        (false, Some(r)) => {
            assert_eq!(
                u64_of(expect, "incarnation").expect("expect_slot.incarnation"),
                r.body.incarnation,
                "{name}[{step}]: stored incarnation"
            );
            assert_eq!(
                u64_of(expect, "issued_at_ms").expect("expect_slot.issued_at_ms"),
                r.body.issued_at_ms,
                "{name}[{step}]: stored issued_at_ms"
            );
        }
    }
}

#[test]
fn the_shared_roster_vectors_hold_against_the_normative_fold() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("parse roster-vectors.json");
    let defaults = fixture.get("record_defaults").expect("record_defaults");
    let vectors = fixture
        .get("vectors")
        .and_then(Value::as_array)
        .expect("vectors");
    assert!(!vectors.is_empty());

    for vector in vectors {
        let name = vector
            .get("name")
            .and_then(Value::as_str)
            .expect("vector name");
        let mut slot = match vector.get("initial") {
            Some(Value::Null) | None => RosterSlot::new(),
            Some(spec) => RosterSlot {
                record: Some(record(record_body(defaults, spec))),
            },
        };
        let steps = vector
            .get("steps")
            .and_then(Value::as_array)
            .expect("steps");
        for (i, step) in steps.iter().enumerate() {
            let published = record(record_body(
                defaults,
                step.get("record").expect("step.record"),
            ));
            let (next, decision) = slot.fold(&published);
            let expected = step
                .get("expect")
                .and_then(|e| e.get("kind"))
                .and_then(Value::as_str)
                .expect("expect.kind");
            assert_eq!(
                decision_kind(&decision),
                expected,
                "{name}[{i}]: decision (got {decision:?})"
            );
            assert_slot_matches(
                name,
                i,
                &next,
                step.get("expect_slot").expect("expect_slot"),
            );
            slot = next;
        }
    }
}
