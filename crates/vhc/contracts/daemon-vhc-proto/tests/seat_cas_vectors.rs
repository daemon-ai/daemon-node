// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The shared seat-slot CAS vectors (`tests/fixtures/seat-cas-vectors.json`) executed against the
//! normative fold ([`SeatSlot::fold`]).
//!
//! The fixture is the **cross-implementation contract**: every conforming seat registry — this
//! fold, the local fake in `daemon-vhc-net`, and the cloud registry's port — must reproduce these
//! decisions and slot transitions exactly. The vectors are structural by construction (the
//! registry never verifies signatures or judges authority), so leases are built here with dummy
//! certificates and unset signatures: if the fold ever starts caring about either, these vectors
//! break loudly — which would itself be a posture violation to escalate, never to "fix" by
//! signing the fixtures.

use daemon_vhc_proto::cert::{RunKeyCertBody, RunKeyCertificate};
use daemon_vhc_proto::domains::{SEAT_LEASE_DOMAIN, SEAT_RELEASE_DOMAIN};
use daemon_vhc_proto::{
    ControlEndpoint, Hash, PeerId, SeatDecision, SeatLease, SeatLeaseBody, SeatRelease,
    SeatReleaseBody, SeatRequest, SeatSlot, Signature, CERT_DOMAIN_V2,
};
use serde_json::Value;

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/seat-cas-vectors.json"
));

fn u64_of(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

fn fill(v: &Value, key: &str) -> Option<u8> {
    u64_of(v, key).map(|n| u8::try_from(n).expect("fill byte"))
}

/// Build a lease body from a vector spec layered over the fixture's `lease_defaults`.
fn lease_body(defaults: &Value, spec: &Value) -> SeatLeaseBody {
    let get = |key: &str| spec.get(key).filter(|v| !v.is_null()).or(defaults.get(key));
    let get_u64 = |key: &str| {
        get(key)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("vector lease field `{key}`"))
    };
    let get_fill = |key: &str| u8::try_from(get_u64(key)).expect("fill byte");
    let endpoint_ws = match spec.get("endpoint_ws") {
        // An explicit `"endpoint_ws": null` in the spec means "no endpoint" (the structural case).
        Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        None => defaults
            .get("endpoint_ws")
            .and_then(Value::as_str)
            .map(str::to_string),
        other => panic!("vector endpoint_ws: {other:?}"),
    };
    SeatLeaseBody {
        domain: get("domain")
            .and_then(Value::as_str)
            .expect("domain")
            .to_string(),
        run_id: Hash([get_fill("run"); 32]),
        role: get("role")
            .and_then(Value::as_str)
            .expect("role")
            .to_string(),
        epoch: get_u64("epoch"),
        incarnation: get_u64("incarnation"),
        leadership_term: get_u64("leadership_term"),
        claimant: PeerId([get_fill("claimant"); 32]),
        module_hash: Hash([get_fill("module"); 32]),
        endpoint: ControlEndpoint {
            ws: endpoint_ws,
            iroh_ticket: None,
        },
        issued_at_ms: get_u64("issued_at_ms"),
        expires_at_ms: get_u64("expires_at_ms"),
        heartbeat_interval_ms: get_u64("heartbeat_interval_ms"),
    }
}

/// Wrap a body as a stored/presented lease. Certificate + signature are DUMMIES: the fold is
/// structural and must never read them.
fn lease_of(body: SeatLeaseBody) -> SeatLease {
    let certificate = RunKeyCertificate {
        body: RunKeyCertBody {
            domain: CERT_DOMAIN_V2.to_string(),
            scope: body.cert_scope(),
            run_key: body.claimant,
        },
        base_identity: PeerId([0u8; 32]),
        sig: Signature([0u8; 64]),
    };
    SeatLease {
        body,
        certificate,
        sig: Signature([0u8; 64]),
    }
}

fn release_of(defaults: &Value, spec: &Value) -> SeatRelease {
    let run = fill(spec, "run")
        .or_else(|| fill(defaults, "run"))
        .expect("run fill");
    let role = spec
        .get("role")
        .or_else(|| defaults.get("role"))
        .and_then(Value::as_str)
        .expect("role")
        .to_string();
    SeatRelease {
        body: SeatReleaseBody {
            domain: spec
                .get("domain")
                .and_then(Value::as_str)
                .unwrap_or(SEAT_RELEASE_DOMAIN)
                .to_string(),
            run_id: Hash([run; 32]),
            role,
            incarnation: u64_of(spec, "incarnation").expect("incarnation"),
            leadership_term: u64_of(spec, "leadership_term").expect("leadership_term"),
            claimant: PeerId([fill(spec, "claimant").expect("claimant"); 32]),
        },
        sig: Signature([0u8; 64]),
    }
}

fn slot_of(defaults: &Value, spec: &Value) -> SeatSlot {
    let lease = spec
        .get("lease")
        .filter(|v| !v.is_null())
        .map(|l| lease_of(lease_body(defaults, l)));
    SeatSlot {
        lease,
        last_leadership_term: u64_of(spec, "last_leadership_term"),
    }
}

fn assert_decision(name: &str, step: usize, expect: &Value, got: &SeatDecision) {
    let kind = expect.get("kind").and_then(Value::as_str).expect("kind");
    let ok = match (kind, got) {
        ("accepted", SeatDecision::Accepted)
        | ("rejected_structural", SeatDecision::RejectedStructural { .. })
        | ("rejected_held", SeatDecision::RejectedHeld)
        | ("rejected_not_held", SeatDecision::RejectedNotHeld) => true,
        ("rejected_fencing_conflict", SeatDecision::RejectedFencingConflict { expected, got }) => {
            u64_of(expect, "expected") == Some(*expected) && u64_of(expect, "got") == Some(*got)
        }
        _ => false,
    };
    assert!(
        ok,
        "vector `{name}` step {step}: expected {expect}, got {got:?}"
    );
}

fn assert_slot(name: &str, step: usize, expect: &Value, slot: &SeatSlot) {
    let leased_term = slot.lease.as_ref().map(|l| l.body.leadership_term);
    let leased_incarnation = slot.lease.as_ref().map(|l| l.body.incarnation);
    let leased_claimant = slot.lease.as_ref().map(|l| u64::from(l.body.claimant.0[0]));
    assert_eq!(
        leased_term,
        u64_of(expect, "leased_term"),
        "vector `{name}` step {step}: leased_term"
    );
    assert_eq!(
        leased_incarnation,
        u64_of(expect, "leased_incarnation"),
        "vector `{name}` step {step}: leased_incarnation"
    );
    assert_eq!(
        leased_claimant,
        u64_of(expect, "leased_claimant"),
        "vector `{name}` step {step}: leased_claimant"
    );
    assert_eq!(
        slot.last_leadership_term,
        u64_of(expect, "last_leadership_term"),
        "vector `{name}` step {step}: last_leadership_term"
    );
}

#[test]
fn the_fold_reproduces_every_shared_cas_vector() {
    let fixture: Value = serde_json::from_str(VECTORS).expect("fixture parses");
    let defaults = fixture.get("lease_defaults").expect("lease_defaults");
    assert_eq!(
        defaults.get("domain").and_then(Value::as_str),
        Some(SEAT_LEASE_DOMAIN),
        "the fixture pins the live seat-lease domain string"
    );
    let vectors = fixture
        .get("vectors")
        .and_then(Value::as_array)
        .expect("vectors");
    assert!(!vectors.is_empty());

    for vector in vectors {
        let name = vector.get("name").and_then(Value::as_str).expect("name");
        let skew_ms = u64_of(vector, "skew_ms").expect("skew_ms");
        let mut slot = slot_of(defaults, vector.get("initial").expect("initial"));
        let steps = vector
            .get("steps")
            .and_then(Value::as_array)
            .expect("steps");
        for (i, step) in steps.iter().enumerate() {
            let now_ms = u64_of(step, "now_ms").expect("now_ms");
            let request = step.get("request").expect("request");
            let request = if let Some(spec) = request.get("claim") {
                SeatRequest::Claim(lease_of(lease_body(defaults, spec)))
            } else if let Some(spec) = request.get("renew") {
                SeatRequest::Renew(lease_of(lease_body(defaults, spec)))
            } else if let Some(spec) = request.get("release") {
                SeatRequest::Release(release_of(defaults, spec))
            } else {
                panic!("vector `{name}` step {i}: unknown request kind");
            };
            let (next, decision) = slot.fold(&request, now_ms, skew_ms);
            assert_decision(name, i, step.get("expect").expect("expect"), &decision);
            assert_slot(
                name,
                i,
                step.get("expect_slot").expect("expect_slot"),
                &next,
            );
            slot = next;
        }
    }
}
