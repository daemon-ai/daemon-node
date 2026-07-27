// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Assembling a [`ResourceAuthority`] from **outside** the resource crate.
//!
//! At the certification minor a module declares no physical figure, so admission composes one from the
//! plan, an authenticated Backend Execution Profile and this node's capability report; with no
//! authority to compose against, the run is refused `ClaimNotComposable`. That refusal is the correct
//! floor, but it left a hole in what was testable: authentication is reachable only through the
//! resource crate's private machinery, so until now nothing outside that crate could build the
//! authority the positive path needs — this file is the first place anywhere that does.
//!
//! It exists to keep that ability honest. `daemon-vhc-resource/test-support` is what makes it possible,
//! and a break here means either the feature stopped exposing enough to assemble an authority or the
//! assembly itself drifted — both of which would otherwise be discovered by whoever next tried to test
//! the composed path, and read to them as their own mistake.
//!
//! The floor itself (the refusal when the authority is absent, and the driver's fail-closed refusal
//! rather than a run header with empty members) is covered where those decisions are made, in
//! `run::admission` and `run::driver`.

use daemon_vhc_host::run::admission::ResourceAuthority;
use daemon_vhc_resource::revision::BackendClass;
use daemon_vhc_resource::test_support;
use daemon_vhc_resource::{AdmissionInputs, ReservationIdentity};

/// The full assembly, in the order a node performs it: stock a store, select against both policy
/// sides, and hold the borrow while composing.
#[test]
fn the_test_support_feature_assembles_an_authority_that_composes_a_claim() {
    let (store, running) = test_support::stocked_profile_store(BackendClass::Vulkan);
    let policy = test_support::accepting_policy();
    let profile = store
        .select(&test_support::authentication_context(&running, &policy))
        .expect("the fixture profile authenticates under the fixture policy");

    let report = test_support::capability_report(BackendClass::Vulkan);
    let lane_bounds = test_support::generous_lane_bounds();
    let plan = test_support::trivial_plan();

    let authority = ResourceAuthority {
        profile: &profile,
        report: &report,
        lane_bounds: &lane_bounds,
        co_resident_roles: 1,
        reservation_identity: ReservationIdentity {
            role: "trainer".into(),
            incarnation: 1,
            device_identity: "test-device".into(),
            sequence: 1,
        },
        frozen_binding: None,
    };

    let composed = daemon_vhc_resource::admit_composition(&AdmissionInputs {
        plan: &plan,
        profile: authority.profile,
        report: authority.report,
        owner_cap: None,
        lane: "gpu-small",
        lane_bounds: authority.lane_bounds,
        co_resident_roles: authority.co_resident_roles,
        reservation_identity: authority.reservation_identity.clone(),
        frozen_binding: authority.frozen_binding,
    })
    .expect("a plan, an authenticated profile and a measured report compose a claim");

    // A composed claim, not a placeholder: the plan declares a persistent parameter tensor, so a
    // figure of zero would mean the composition silently produced nothing while reporting success.
    assert!(
        composed.claim().device_total_bytes() > 0,
        "the composed claim reserves no device memory for a plan that declares a persistent tensor"
    );
    // The grant is what the guest actually receives, and a claim without one is a figure the module
    // can never be told about.
    assert!(
        !composed.grant().values.is_empty(),
        "the composition produced no Execution Grant values for the guest to read"
    );
}

/// A **frozen** binding is verified rather than reselected — the uniform-run path, and the one a fleet
/// participant takes. Assembling it from outside must work too, or the only externally testable
/// composition would be the one no real run performs.
#[test]
fn a_frozen_binding_composes_from_outside_the_resource_crate_too() {
    let (store, running) = test_support::stocked_profile_store(BackendClass::Vulkan);
    let policy = test_support::accepting_policy();
    let profile = store
        .select(&test_support::authentication_context(&running, &policy))
        .expect("the fixture profile authenticates");

    let report = test_support::capability_report(BackendClass::Vulkan);
    let lane_bounds = test_support::generous_lane_bounds();
    let plan = test_support::trivial_plan();
    let frozen = test_support::trivial_binding(2);

    let composed = daemon_vhc_resource::admit_composition(&AdmissionInputs {
        plan: &plan,
        profile: &profile,
        report: &report,
        owner_cap: None,
        lane: "gpu-small",
        lane_bounds: &lane_bounds,
        co_resident_roles: 1,
        reservation_identity: ReservationIdentity {
            role: "trainer".into(),
            incarnation: 1,
            device_identity: "test-device".into(),
            sequence: 1,
        },
        frozen_binding: Some(&frozen),
    })
    .expect("the frozen configuration is within what this machine admits, so it verifies");

    // The frozen configuration was honoured, not re-derived: the minimum binding this plan would
    // select on its own is 1, so a claim equal to the minimum's would mean the freeze was ignored —
    // which is the failure mode that surfaces far away, as a digest mismatch at another participant.
    let minimum = daemon_vhc_resource::admit_composition(&AdmissionInputs {
        plan: &plan,
        profile: &profile,
        report: &report,
        owner_cap: None,
        lane: "gpu-small",
        lane_bounds: &lane_bounds,
        co_resident_roles: 1,
        reservation_identity: ReservationIdentity {
            role: "trainer".into(),
            incarnation: 1,
            device_identity: "test-device".into(),
            sequence: 2,
        },
        frozen_binding: None,
    })
    .expect("the unfrozen selection composes");

    assert!(
        composed.claim().device_total_bytes() > minimum.claim().device_total_bytes(),
        "the frozen micro_batch=2 claim ({}) is not larger than the minimum selection's ({}) — the \
         frozen binding was reselected rather than verified",
        composed.claim().device_total_bytes(),
        minimum.claim().device_total_bytes()
    );
}
