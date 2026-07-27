// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fixture assemblers for **tests in other crates** — behind the `test-support` feature, never default.
//!
//! ## Why this exists, and why it is fenced
//!
//! At the certification minor a module declares no physical figure, so admission composes one from a
//! plan, an **authenticated** certified profile and the node's capability report. A test that drives a
//! certification-minor guest through admission therefore needs all three, and the only honest way to
//! obtain an authenticated profile is the way production obtains one: stock a store and select against
//! an owner policy and a run policy. That machinery is deliberately crate-private — nothing outside can
//! mint a profile — which is exactly the property that would otherwise make such a test impossible to
//! write.
//!
//! So the fixtures are exposed under a feature that:
//!
//! - is **not** in `default`;
//! - MUST appear only in `dev-dependencies` edges, never in a `[dependencies]` edge of any crate — a
//!   production build that could switch it on could mint a profile, and `xtask vhc-dep-check` fails the
//!   gate if any production edge enables it;
//! - carries no production code paths: everything here is a constructor for a value a test asserts
//!   against, and none of it is reachable from a shipping binary.
//!
//! ## What a caller assembles
//!
//! ```ignore
//! let (store, running) = test_support::stocked_profile_store(BackendClass::Vulkan);
//! let policy = test_support::accepting_policy();
//! let profile = store.select(&test_support::authentication_context(&running, &policy))?;
//! let report = test_support::capability_report(BackendClass::Vulkan);
//! let bounds = test_support::generous_lane_bounds();
//! // -> everything `daemon_vhc_host::run::ResourceAuthority` needs.
//! ```
//!
//! The store must outlive the selected profile: an `AuthenticatedProfile` borrows the store it was
//! selected from, because a verdict that outlived its store would be a verdict about a profile nobody
//! holds any more.

use crate::capability::DeviceCapabilityReport;
use crate::planner::LaneEstimateBounds;
use crate::revision::{BackendClass, BackendImplementationRevision};
use crate::store::{AuthenticationContext, ProfileStore};
use crate::trust::ProfileAcceptancePolicy;
use daemon_vhc_proto::resource_plan::{Binding, LogicalResourcePlan};

/// A store holding one profile that authenticates, and the running revision it is priced for.
///
/// Returned as a pair because authentication compares the record's backend class and implementation
/// revision against the profile's: a caller that sourced the two independently would be testing its own
/// fixture wiring rather than the rule.
#[must_use]
pub fn stocked_profile_store(class: BackendClass) -> (ProfileStore, BackendImplementationRevision) {
    let running = crate::revision::fixtures::revision(class);
    let profile = crate::trust::fixtures::profile_for(&running);
    let envelope = crate::trust::fixtures::envelope_for(&profile, &running);
    let mut store = ProfileStore::new();
    store
        .insert(profile, envelope)
        .expect("the fixture profile and envelope agree, so the store accepts them");
    (store, running)
}

/// A policy that accepts the fixture authority — usable as both the owner's and the run's side.
///
/// Release-only: it names no development authority, because accepting one is a separate, explicit act
/// that **both** policies must perform (`[PC-12]`, ratified 2026-07-26). A test that wants the
/// development path names it on both sides deliberately.
#[must_use]
pub fn accepting_policy() -> ProfileAcceptancePolicy {
    crate::trust::fixtures::policy_for(&ProfileStore::new())
}

/// The authentication context for a selection: both policy sides, the running revision, the planner
/// version the profile is priced for, and the fixture clock.
#[must_use]
pub fn authentication_context<'a>(
    running: &'a BackendImplementationRevision,
    policy: &'a ProfileAcceptancePolicy,
) -> AuthenticationContext<'a> {
    AuthenticationContext {
        owner: policy,
        run: policy,
        running,
        planner_version: 1,
        now_ms: crate::trust::fixtures::NOW,
    }
}

/// A complete, valid capability report **carrying a measured per-allocation ceiling**.
///
/// The ceiling matters: admission's pool step refuses a report that has none, which is correct — an
/// unmeasured device is not a device that passed — and is also the single most likely reason a test
/// assembling an authority by hand sees an unexpected refusal.
#[must_use]
pub fn capability_report(class: BackendClass) -> DeviceCapabilityReport {
    crate::capability::fixtures::report(class)
}

/// The fixture report with its derived usable device supply raised to `usable_bytes`.
///
/// For gates whose subject is something other than device supply, run at geometries whose
/// conservative composed estimate exceeds the stock fixture figure: raising the fixture's supply is
/// fixture policy — the same act as picking the stock figure — not a statement about hardware, and
/// it keeps the refusal path (`admit_node_memory_bytes`) exercised where supply IS the subject.
#[must_use]
pub fn capability_report_with_supply(
    class: BackendClass,
    usable_bytes: u64,
) -> DeviceCapabilityReport {
    let mut report = crate::capability::fixtures::report(class);
    report.device_supply = crate::revision::Maybe::Available(
        crate::capability::fixtures::derived_supply(usable_bytes),
    );
    report
}

/// Lane bounds wide enough for any composed estimate, for a test that is not about lane bounds.
///
/// Every backend class is populated on purpose. A class with no entry refuses
/// `LaneProfileUnsupported` — silence is not permission — so an incomplete map here would surface as a
/// refusal that looks like a lane-bounds failure in a test about something else.
#[must_use]
pub fn generous_lane_bounds() -> LaneEstimateBounds {
    let mut bounds = LaneEstimateBounds::default();
    for class in ["vulkan", "metal", "dx12", "cuda", "cpu"] {
        bounds
            .by_backend_class
            .insert(class.to_string(), [0, u64::MAX]);
    }
    bounds
}

/// A small, well-formed Logical Resource Plan: one bounded dimension, a persistent and a transient
/// tensor, one priced operation, one transfer, one linear-memory term.
///
/// The same plan the crate's own composition tests use, so a consumer's expectations and the planner's
/// canonical vectors are talking about one object.
#[must_use]
pub fn trivial_plan() -> LogicalResourcePlan {
    crate::planner::fixtures::plan()
}

/// A binding of [`trivial_plan`]'s single dimension, for a caller that wants a frozen configuration.
#[must_use]
pub fn trivial_binding(micro_batch: u64) -> Binding {
    crate::planner::fixtures::binding(micro_batch)
}
