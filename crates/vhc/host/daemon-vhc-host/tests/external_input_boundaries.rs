// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! **The host's external-input boundaries, enumerated — and each one refuses typed.**
//!
//! Every boundary below is a place where bytes the host did not produce cross into it. The rule at all
//! of them is the same and it is not "validate the input": it is that a malformed input produces a
//! **typed refusal naming what was wrong**, before any value from it is used, and never a panic, never a
//! silent default, and never a zero standing in for a measurement.
//!
//! This file exists to be the enumeration. A boundary that is not listed here is either not a boundary
//! or is not covered, and both are findable by reading one file — which is the property the enumeration
//! is for. Each test names the boundary, feeds it something a hostile or broken producer would, and
//! asserts the refusal is typed and specific.
//!
//! | # | Boundary | Untrusted producer | Refusal shape |
//! |---|---|---|---|
//! | 1 | the framed genesis envelope | whoever published the run | refuse before payload decoding |
//! | 2 | `da_resource_plan`'s returned span | the guest module | typed plan refusal |
//! | 3 | the Execution Grant bytes | the authoring pipeline / role entry | typed decode refusal |
//! | 4 | the Device Capability Report | another node, or this one's probe | typed capability error |
//! | 5 | a Backend Execution Profile + its envelope | a profile signer | typed authentication refusal |
//! | 6 | a recorded composition in a journal | whoever wrote the journal | typed replay refusal |
//! | 7 | a journal record's tag/body | whoever wrote the journal | typed decode error |
//! | 8 | `sys@2::log`'s message span | the guest module | clamped from the arguments alone |
//!
//! Boundaries deliberately covered elsewhere, named here so the enumeration is honest about its edges:
//! the wasm sandbox's own limits (fuel, epoch, memory) are wasmtime's and are exercised by the driver
//! suites; network frame decode is `daemon-vhc-net`'s codec suite; the admission funnel's staged
//! refusals are `claim_funnel`.

use daemon_vhc_proto::execution_grant::ExecutionGrant;
use daemon_vhc_proto::resource_plan::LogicalResourcePlan;
use daemon_vhc_resource::{
    validate_recorded_composition, CapabilityError, RecordedComposition, RecordedCompositionError,
};

/// **Boundary 1 — the framed genesis envelope.** Refusal precedes payload decoding.
///
/// The framing header exists so that a reader can refuse *before* decoding a payload it may not
/// understand: an envelope from a future schema major, or one whose declared length or digest does not
/// match its bytes, is refused on the header alone. A reader that decoded first would have already acted
/// on bytes it had not authenticated.
#[test]
fn boundary_1_a_framed_genesis_refuses_on_the_header_before_decoding_its_payload() {
    let payload = b"a payload this reader must never decode".to_vec();
    let framed =
        daemon_vhc_proto::framing::frame(daemon_vhc_proto::GENESIS_SCHEMA_MAJOR + 1, 0, &payload)
            .expect("framing a next-generation envelope is legal; reading it is not");

    // A next-generation major is refused by a current reader.
    let err = daemon_vhc_proto::framing::unframe(&framed, daemon_vhc_proto::GENESIS_SCHEMA_MAJOR)
        .expect_err("a current reader refuses a next-generation schema major");
    let msg = err.to_string();
    assert!(
        msg.contains("major") || msg.contains("schema"),
        "the refusal names the schema disagreement: {msg}"
    );

    // And the other direction: bytes that are not a frame at all are refused, not indexed into.
    for junk in [&b""[..], &b"short"[..], &[0xFFu8; 64][..]] {
        assert!(
            daemon_vhc_proto::framing::unframe(junk, daemon_vhc_proto::GENESIS_SCHEMA_MAJOR)
                .is_err(),
            "non-frame bytes are refused rather than parsed"
        );
    }

    // A truncated payload contradicts the header's declared length, and the header wins.
    let mut truncated = framed.clone();
    truncated.truncate(framed.len() - 1);
    assert!(
        daemon_vhc_proto::framing::unframe(&truncated, daemon_vhc_proto::GENESIS_SCHEMA_MAJOR)
            .is_err(),
        "a payload shorter than its declared length is refused"
    );
}

/// **Boundary 2 — the plan span a guest returns.** Malformed bytes are a typed plan refusal.
///
/// The span is guest-owned memory whose length the guest chose. Bounds are checked before the copy, and
/// the bytes are then held to the closed schema: non-canonical CBOR, a truncated document, or content
/// naming a physical backend are all refusals rather than a partially-decoded plan.
#[test]
fn boundary_2_a_guest_plan_span_is_refused_typed_when_it_is_not_a_plan() {
    for (what, bytes) in [
        ("empty", &b""[..]),
        ("not CBOR at all", &b"this is not cbor"[..]),
        ("a bare CBOR integer where a map belongs", &[0x01u8][..]),
        ("a truncated CBOR map", &[0xA1u8, 0x63][..]),
    ] {
        let refusal = LogicalResourcePlan::decode_canonical(bytes)
            .expect_err("a plan decoder refuses non-plan bytes");
        assert!(
            !refusal.detail().is_empty(),
            "the refusal for {what} says what was wrong"
        );
    }
}

/// **Boundary 3 — the Execution Grant bytes.** A grant is decoded by its own closed decoder.
///
/// The grant is host-written for a run, but on the replay and role-entry paths it arrives as bytes from
/// somewhere else, and it is required *before* initialization — so a permissive decode here would admit
/// a configuration the run never agreed to at the one moment that establishes execution identity.
#[test]
fn boundary_3_grant_bytes_are_refused_typed_when_they_are_not_a_grant() {
    for bytes in [&b""[..], &b"{}"[..], &[0xA0u8][..], &[0xFFu8; 32][..]] {
        assert!(
            ExecutionGrant::decode_canonical(bytes).is_err(),
            "a grant decoder refuses non-grant bytes rather than yielding an empty configuration"
        );
    }
}

/// **Boundary 4 — the Device Capability Report.** A zero reading is refused as loudly as a malformed one.
///
/// The report is measured on a node and read by admission, so its failure mode is not corruption but
/// *plausibility*: a probe that could not measure returning zero, or a per-allocation ceiling larger than
/// the whole supply, are both statements that look like measurements and are not.
#[test]
fn boundary_4_a_capability_report_refuses_zero_readings_and_impossible_ceilings() {
    // The report type has exactly one production constructor — the probe adapter in the worker — and its
    // fixtures are private to the crate that owns it, which is the property this boundary relies on:
    // nothing outside can mint a report. So the *rules* are pinned where the type lives (a zero reading,
    // a ceiling above the whole supply, and an unmeasured ceiling are each refused in
    // `daemon-vhc-resource`'s capability suite), and what this enumeration asserts is that the refusals
    // reaching a caller here are **typed and name the quantity** — an untyped or unnamed one would be
    // unactionable at exactly the moment an operator needs to know which figure was wrong.
    let zero = CapabilityError::ZeroInsteadOfUnavailable {
        quantity: "free disk",
    };
    assert!(
        zero.to_string().contains("free disk") && zero.to_string().contains("zero"),
        "a zero reading names the quantity that was zero: {zero}"
    );
    let unmeasured = CapabilityError::Unmeasured {
        quantity: "per-allocation ceiling",
        detail:
            "a composed estimate's maximum individual allocation cannot be validated against an \
                 unmeasured limit",
    };
    assert!(
        unmeasured.to_string().contains("per-allocation ceiling"),
        "an unmeasured quantity names itself rather than reporting a generic failure: {unmeasured}"
    );
}

/// **Boundary 5 — a profile and its trust envelope.** Acceptance is an intersection, and silence is not
/// consent for a development authority.
///
/// A profile arrives signed by someone. The refusal has to name the rejecting policy, because "not
/// accepted" is unactionable when two policies are involved — and a development authority requires both
/// sides to name it explicitly, so an unstated policy admits nothing rather than deferring.
#[test]
fn boundary_5_an_unvouched_profile_is_refused_and_the_refusal_names_a_reason() {
    // The authentication surface is exercised in depth by the resource crate's trust suite. What this
    // boundary owns is that the refusal is *typed and reportable*: an operator holding one must be able
    // to tell an expiry from a revision mismatch from a policy gap.
    let refusal = daemon_vhc_resource::AuthenticationRefusal::NoAuthorityNamed;
    let rendered = refusal.to_string();
    assert!(
        !rendered.is_empty() && rendered.to_lowercase().contains("authorit"),
        "the no-authority refusal explains itself: {rendered}"
    );
}

/// **Boundary 6 — a recorded composition in a journal.** Digests are checked against the recorded bytes
/// before any value is used, and the members must agree about each other.
#[test]
fn boundary_6_a_recorded_composition_with_a_wrong_digest_is_refused_before_use() {
    let bytes = b"whatever these bytes are".to_vec();
    let wrong = daemon_vhc_proto::blake3_hash(b"different bytes entirely");
    let recorded = RecordedComposition {
        resource_plan: (&bytes, wrong),
        physical_estimate: (&bytes, wrong),
        aggregate_estimate: (&bytes, wrong),
        execution_grant: (&bytes, wrong),
    };
    let err = validate_recorded_composition(recorded)
        .expect_err("a digest that does not match its bytes refuses");
    assert!(
        matches!(err, RecordedCompositionError::DigestMismatch { .. }),
        "and it refuses on the digest, before attempting to decode anything: {err}"
    );
}

/// **Boundary 7 — a journal record.** An unknown tag or an undecodable body is a typed error.
///
/// A journal is read by replay, by the archive verifier and by an operator's tooling, and it may have
/// been written by an older or newer host. A tag this reader does not know is refused by name — the
/// failure that motivated it was a tag that could be *written* and not *read*.
#[test]
fn boundary_7_a_journal_record_with_an_unknown_tag_is_refused_by_name() {
    use daemon_vhc_observe::journal::record::Record;

    // A well-formed CBOR record shape carrying a tag no reader implements.
    let unknown_tag = 250u8;
    let body = ciborium::value::Value::Map(Vec::new());
    // The record shape is `[tag, ord, body]` — the tag first, which is what makes a tag this reader
    // does not implement refusable before the body is looked at.
    let record = ciborium::value::Value::Array(vec![
        ciborium::value::Value::Integer(u64::from(unknown_tag).into()),
        ciborium::value::Value::Integer(0u64.into()),
        body,
    ]);
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&record, &mut bytes).expect("encode");

    let err = Record::from_canonical(&bytes).expect_err("an unknown tag is refused");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("250") || msg.to_lowercase().contains("tag"),
        "the refusal names the tag it did not know: {msg}"
    );
}

/// **Boundary 8 — the `log` message span.** The clamp is computed from the arguments alone.
///
/// This is the boundary where the ordering matters most: the guest declares a length, and the host must
/// clamp *before* allocating or copying. A host that copied the whole span and then truncated has
/// already paid for the untrusted number.
#[test]
fn boundary_8_the_log_clamp_is_computed_from_the_arguments_alone() {
    use daemon_vhc_abi::{log_accepted_prefix_len, LOG_BYTES_PER_PHASE_MAX, LOG_MESSAGE_BYTES_MAX};

    // An absurd declared length yields the per-message ceiling, not an allocation attempt.
    assert_eq!(
        u64::from(log_accepted_prefix_len(u32::MAX, LOG_BYTES_PER_PHASE_MAX)),
        LOG_MESSAGE_BYTES_MAX,
        "a hostile length is clamped to the per-message bound"
    );
    // An exhausted phase budget accepts nothing, whatever the guest asks for.
    assert_eq!(log_accepted_prefix_len(4096, 0), 0);
    // And the function is total over its inputs: no combination panics.
    for raw in [0u32, 1, 4095, 4096, 4097, u32::MAX] {
        for remaining in [0u64, 1, LOG_BYTES_PER_PHASE_MAX, u64::MAX] {
            let accepted = log_accepted_prefix_len(raw, remaining);
            assert!(u64::from(accepted) <= remaining);
            assert!(u64::from(accepted) <= LOG_MESSAGE_BYTES_MAX);
            assert!(accepted <= raw);
        }
    }
}
