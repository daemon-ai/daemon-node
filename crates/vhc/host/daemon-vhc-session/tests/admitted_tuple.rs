// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `[DI-9]` admitted tuple's **certification branch**, built from a real composition.
//!
//! The tuple is the assessment's statement of what will run, re-derived at join and compared member by
//! member. Below the certification minor its resource statement is the digest of the claim the module
//! declared. At the certification minor the module declares no claim, so the statement is the plan that
//! was priced plus the composition that priced it — and the reason this file exists is that the failure
//! mode of getting it wrong is silent: a certification-minor tuple that kept digesting the (now empty)
//! declared claim bytes carries `blake3("")`, a constant, which compares equal between two runs that
//! have nothing whatsoever in common. Join would re-verify it, report a match, and have checked nothing.
//!
//! So these tests assert against real composed values, not against a shape. The composition comes from
//! `daemon-vhc-resource`'s fixture assemblers (its `test-support` feature), because an authenticated
//! profile is the one input a test cannot fabricate.

use daemon_vhc_abi::CandidateDriver;
use daemon_vhc_host::run::{Admission, ComposedAgainst};
use daemon_vhc_host::Selection;
use daemon_vhc_resource::revision::BackendClass;
use daemon_vhc_resource::{
    test_support, AdmissionInputs, AdmittedComposition, ReservationIdentity,
};
use daemon_vhc_session::protocol::{AdmittedResource, ComposedResource};

const DEVICE: &str = "vulkan:0000:c4:00.0";

/// Compose for real, then hand back both halves the admission carries.
fn composed_admission() -> (AdmittedComposition, ComposedAgainst, Vec<u8>) {
    let (store, running) = test_support::stocked_profile_store(BackendClass::Vulkan);
    let policy = test_support::accepting_policy();
    let profile = store
        .select(&test_support::authentication_context(&running, &policy))
        .expect("the fixture profile authenticates");
    let report = test_support::capability_report(BackendClass::Vulkan);
    let lane_bounds = test_support::generous_lane_bounds();
    let plan = test_support::trivial_plan();

    let composition = daemon_vhc_resource::admit_composition(&AdmissionInputs {
        plan: &plan,
        profile: &profile,
        report: &report,
        owner_cap: None,
        lane: "gpu-small",
        lane_bounds: &lane_bounds,
        co_resident_roles: 1,
        reservation_identity: ReservationIdentity {
            role: "trainer".into(),
            incarnation: 3,
            device_identity: DEVICE.into(),
            sequence: 7,
        },
        frozen_binding: None,
    })
    .expect("the plan composes against the authenticated profile");

    let against = ComposedAgainst {
        profile_digest: profile.digest().0,
        profile_authority: profile.authenticating_authority().0,
        backend_class: profile.profile().backend_class.slug().to_string(),
        backend_implementation_revision: profile.profile().implementation_revision.clone(),
        capability_report_digest: report.report_digest().expect("the report digests").0,
        planner_version: daemon_vhc_resource::PLANNER_VERSION,
        device_identity: DEVICE.into(),
        reservation_sequence: 7,
    };
    let plan_bytes = plan.to_canonical_bytes().expect("the plan encodes");
    (composition, against, plan_bytes)
}

fn admission(
    minor: u32,
    composition: Option<AdmittedComposition>,
    against: Option<ComposedAgainst>,
    plan_bytes: Vec<u8>,
) -> Admission {
    Admission {
        selection: Selection {
            driver: CandidateDriver::V2,
            major: 2,
            minor,
        },
        claim: None,
        composition,
        composed_against: against,
        claim_bytes: Vec::new(),
        resource_plan_bytes: plan_bytes,
        manifest_bytes: Vec::new(),
        quotas: None,
    }
}

/// The certification branch carries the plan digest and the composed values — each digest agreeing
/// with the bytes it stands for.
#[test]
fn a_composed_admission_states_the_plan_it_priced_and_what_priced_it() {
    let (composition, against, plan_bytes) = composed_admission();
    let expected_plan_hash = *blake3::hash(&plan_bytes).as_bytes();
    let admission = admission(5, Some(composition), Some(against.clone()), plan_bytes);

    let AdmittedResource::Composed(composed) =
        AdmittedResource::from_admission(&admission).expect("the admission states its identity")
    else {
        panic!("a composed admission must produce the composed branch, not a declared claim hash");
    };

    // The renamed member is the PLAN's digest — not `blake3("")`, which is what digesting the absent
    // declared claim would have produced, and which would have compared equal across unrelated runs.
    assert_eq!(composed.logical_resource_plan_hash, expected_plan_hash);
    assert_ne!(
        composed.logical_resource_plan_hash,
        *blake3::hash(b"").as_bytes(),
        "the plan digest is the digest of empty bytes — the composed branch is digesting the \
         absent declared claim instead of the plan"
    );

    // Every digest agrees with the bytes recorded beside it. A record whose digest does not cover its
    // own bytes is worse than no record: it invites a reader to verify and be reassured.
    assert_eq!(
        composed.physical_estimate_hash,
        *blake3::hash(&composed.physical_estimate).as_bytes()
    );
    assert_eq!(
        composed.aggregate_estimate_hash,
        *blake3::hash(&composed.aggregate_estimate).as_bytes()
    );
    assert_eq!(
        composed.execution_grant_hash,
        *blake3::hash(&composed.selected_configuration).as_bytes()
    );

    // And the authorities it was composed against are the ones that were used, not defaults.
    assert_eq!(composed.profile_digest, against.profile_digest);
    assert_eq!(composed.profile_authority, against.profile_authority);
    assert_eq!(composed.backend_class, "vulkan");
    assert_eq!(
        composed.backend_implementation_revision,
        against.backend_implementation_revision
    );
    assert_eq!(
        composed.capability_report_digest,
        against.capability_report_digest
    );
    assert_eq!(composed.device_identity, DEVICE);
    assert_eq!(composed.reservation_sequence, 7);
    assert_ne!(
        composed.reservation_digest, [0u8; 32],
        "the reservation digest is absent, so the record locates a reservation without proving \
         what was charged"
    );
    assert!(
        !composed.physical_estimate.is_empty() && !composed.aggregate_estimate.is_empty(),
        "a composed statement with no claim bytes records that a composition happened and nothing \
         about what it produced"
    );
}

/// Re-deriving the same admission agrees member for member; a re-derivation against a **different**
/// capability report does not, and names the member that moved.
///
/// This is the join-time check in miniature. The report is the interesting mover because nothing else
/// changes with it: the module, config, grants and profile are all identical, so a tuple that did not
/// carry the report digest would call this a match.
#[test]
fn a_re_derivation_agrees_and_a_moved_capability_report_is_named() {
    let (composition, against, plan_bytes) = composed_admission();
    let assessed = admission(
        5,
        Some(composition),
        Some(against.clone()),
        plan_bytes.clone(),
    );
    let admitted = AdmittedResource::from_admission(&assessed).expect("states its identity");
    let rederived = AdmittedResource::from_admission(&assessed).expect("states its identity");
    assert_eq!(
        admitted.first_mismatch(&rederived),
        None,
        "the same admission must re-derive to the same statement"
    );

    let mut moved = against;
    moved.capability_report_digest = [0xFF; 32];
    let (composition, _, _) = composed_admission();
    let reprobed =
        AdmittedResource::from_admission(&admission(5, Some(composition), Some(moved), plan_bytes))
            .expect("states its identity");
    assert_eq!(
        admitted.first_mismatch(&reprobed),
        Some("capability_report_digest")
    );
}

/// An admission with no composition takes the legacy branch, under the member name and the meaning it
/// has always had — the rename does not reach back and reinterpret it.
#[test]
fn an_admission_without_a_composition_keeps_the_declared_member() {
    let legacy = admission(4, None, None, Vec::new());
    let resource = AdmittedResource::from_admission(&legacy).expect("states its identity");
    assert_eq!(
        resource,
        AdmittedResource::Declared {
            claim_hash: *blake3::hash(b"").as_bytes()
        },
        "a legacy admission must digest its declared claim bytes, whatever they are"
    );
}

/// The composed statement survives the wire it travels on.
///
/// It rides inside `Command`/`Event` frames, and its members are the evidence join compares. A member
/// that did not round-trip would be compared against a decoded default and mismatch forever — or worse,
/// default on both sides and match forever.
#[test]
fn the_composed_statement_round_trips_through_the_frame_codec() {
    let (composition, against, plan_bytes) = composed_admission();
    let admission = admission(5, Some(composition), Some(against), plan_bytes);
    let resource = AdmittedResource::from_admission(&admission).expect("states its identity");

    let mut bytes = Vec::new();
    ciborium::into_writer(&resource, &mut bytes).expect("encodes");
    let back: AdmittedResource = ciborium::from_reader(bytes.as_slice()).expect("decodes");
    assert_eq!(resource, back);
    assert_eq!(resource.first_mismatch(&back), None);

    // And the branch survives too — a composed statement must not decode as a declaration.
    assert!(matches!(back, AdmittedResource::Composed(_)));
    let _: &ComposedResource = match &back {
        AdmittedResource::Composed(c) => c,
        AdmittedResource::Declared { .. } => panic!("the composed branch decoded as declared"),
    };
}
