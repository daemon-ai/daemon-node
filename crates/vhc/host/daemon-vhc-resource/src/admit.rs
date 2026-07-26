// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The admission sequence over the three objects: **compose, bound, authorize, reserve**
//! (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-4]`, `[RC-10]`, `[RC-13]`).
//!
//! One implementation, four callers — authoring validation, node admission, sealed-binary conformance
//! and ceremony preflight (`[DI-10]`). The sequence lives here rather than inside the node's funnel
//! because a second copy of it in the funnel would be a second answer to "may this role run", and the
//! whole point of composing a claim from a plan and a certified profile is that authoring, admission
//! and conformance reach the same one.
//!
//! ## The order is the design, not an implementation detail
//!
//! Every step can refuse, and which step refuses first decides what an operator is told:
//!
//! 1. **Compose** the selected configuration into a Physical Claim. A plan that cannot be priced by
//!    this profile fails here, and it is a composition fault — not a small machine.
//! 2. **Bound it by the participation lane**, keyed by the backend class the profile prices. A claim
//!    absurd for the lane is a lane violation. Reported as a capability refusal instead, it would send
//!    someone to inspect hardware for a fault that lives in a plan or a profile.
//! 3. **Authorize it**, against measured supply and the owner's optional cap, as **two independent
//!    comparisons with two attributions**. An operator whose own cap is the binding constraint must
//!    not be told their hardware is too small.
//! 4. **Reserve it**, against the profile's *configured* pool worst case. Not against a measurement:
//!    certified evidence has to bound every admitted behaviour, and one run that stayed inside a bound
//!    is a single sample from a distribution whose maximum is already known.
//!
//! Nothing here reads instantaneous device state. Supply is the capability report's stable statement,
//! and volatile occupancy pressure belongs to the governor at run time.

use daemon_vhc_proto::execution_grant::ExecutionGrant;
use daemon_vhc_proto::resource_plan::{Binding, LogicalResourcePlan};

use crate::capability::{
    admit_node_memory_bytes, CapabilityError, DeviceAdmissionRefusal, DeviceCapabilityReport,
    OwnerDeviceCap,
};
use crate::governor::{
    check_pool_admissible, derive_reservation, GovernorError, PoolAdmission, Reservation,
    ReservationIdentity,
};
use crate::planner::{
    aggregate, check_claim_against_lane, compose_selection, minimum_binding, AggregateClaim,
    LaneClaimBounds, PhysicalClaim, PlannerError, Selection,
};
use crate::store::AuthenticatedProfile;

/// What admission needs on the node to compose and authorize one role instance.
///
/// A struct rather than a parameter list because the members are a set: an admission that had some of
/// them would not be a cheaper admission, it would be an unauthorized one.
pub struct AdmissionInputs<'a> {
    /// The module's Logical Resource Plan, as its assessment emitted it.
    pub plan: &'a LogicalResourcePlan,
    /// The **authenticated** certified profile for this node's backend implementation.
    ///
    /// Authenticated rather than raw so the trust gate cannot be skipped by a caller who did not know
    /// it was there: the only way to hold one of these is to have selected it against an owner and a
    /// run policy.
    pub profile: &'a AuthenticatedProfile<'a>,
    /// This node's Device Capability Report — its stable statement of supply.
    pub report: &'a DeviceCapabilityReport,
    /// The owner's optional cap on this device. Node policy, applied after supply and only tightening.
    pub owner_cap: Option<OwnerDeviceCap>,
    /// The participation lane this role would run in.
    pub lane: &'a str,
    /// The lane's profile-keyed claim sanity bounds.
    pub lane_bounds: &'a LaneClaimBounds,
    /// How many role instances share this device, this one included.
    pub co_resident_roles: u64,
    /// The reservation identity this admission would hold.
    pub reservation_identity: ReservationIdentity,
    /// The frozen configuration a `UniformRun` participant must run, when the signed role entry
    /// carries one.
    ///
    /// `Some` means **verify, do not reselect**: every participant in a uniform run consumes the same
    /// grant bytes, so a participant that re-selected locally would be running a different
    /// configuration than the run agreed on, and the divergence would surface as a digest mismatch far
    /// from its cause. `None` selects the largest configuration this machine admits.
    pub frozen_binding: Option<&'a Binding>,
}

/// What admission produced: the claim, what the node reserved, and the grant the guest will consume.
#[derive(Clone, Debug)]
pub struct AdmittedComposition {
    /// The selected configuration, its Execution Grant, and the composed claim.
    pub selection: Selection,
    /// The node/device aggregate the instance was admitted within.
    pub aggregate: AggregateClaim,
    /// The single memory reservation the governor holds and the owner's ledger projects.
    pub reservation: Reservation,
    /// The pool bound that was checked, carried so evidence can state what admission compared.
    pub pool: PoolAdmission,
}

impl AdmittedComposition {
    /// The composed role Physical Claim.
    #[must_use]
    pub fn claim(&self) -> &PhysicalClaim {
        &self.selection.claim
    }

    /// The Execution Grant the run instance receives.
    #[must_use]
    pub fn grant(&self) -> &ExecutionGrant {
        &self.selection.grant
    }
}

/// Why admission refused, attributed to the step that refused.
///
/// Typed per step because the four are acted on differently, and a single message would collapse the
/// distinction the ratified divergence rule exists to preserve: a composition fault belongs to the plan
/// or the profile, a lane violation to the node's own policy, an authorization refusal to supply or to
/// the owner, and a pool refusal to the profile's configured allocator behaviour.
#[derive(Clone, Debug, thiserror::Error)]
pub enum AdmissionRefusal {
    /// The capability report is not usable, so nothing can be compared against it.
    #[error("this node's capability report cannot be used: {0}")]
    CapabilityUnusable(#[from] CapabilityError),
    /// The plan could not be composed against this profile.
    #[error("the logical resource plan could not be composed into a physical claim: {0}")]
    NotComposable(PlannerError),
    /// The composed claim falls outside the lane's bounds for the priced backend class.
    #[error("{0}")]
    ExceedsLane(PlannerError),
    /// The claim exceeds measured supply, the owner's cap, or the pool they share.
    #[error("{0}")]
    Unauthorized(#[from] DeviceAdmissionRefusal),
    /// The reservation could not be derived, or the configured pool worst case does not fit.
    #[error("{0}")]
    PoolBound(#[from] GovernorError),
    /// The node/device aggregate could not be formed — co-resident profiles disagree about a term.
    #[error("{0}")]
    Aggregation(PlannerError),
}

impl AdmissionRefusal {
    /// The step that refused, as a stable slug for evidence and logs.
    #[must_use]
    pub fn stage(&self) -> &'static str {
        match self {
            Self::CapabilityUnusable(_) => "capability-report",
            Self::NotComposable(_) => "composition",
            Self::ExceedsLane(_) => "lane-bounds",
            Self::Unauthorized(_) => "device-authorization",
            Self::PoolBound(_) => "pool-bound",
            Self::Aggregation(_) => "aggregation",
        }
    }
}

/// Compose, bound, authorize and reserve one role instance — the whole admission sequence, in order.
///
/// # Errors
/// [`AdmissionRefusal`], naming the step that refused.
pub fn admit_composition(
    inputs: &AdmissionInputs<'_>,
) -> Result<AdmittedComposition, AdmissionRefusal> {
    let profile = inputs.profile.profile();
    inputs.report.validate()?;

    // 1 — compose the configuration this instance will run.
    let binding = match inputs.frozen_binding {
        Some(frozen) => frozen.clone(),
        None => minimum_binding(inputs.plan).map_err(AdmissionRefusal::NotComposable)?,
    };
    let selection = compose_selection(
        inputs.plan,
        &binding,
        profile,
        inputs.co_resident_roles,
        inputs.plan.selection_scope,
    )
    .map_err(AdmissionRefusal::NotComposable)?;

    // 2 — the lane bounds the composed claim, before any statement about the machine.
    check_claim_against_lane(&selection.claim, profile, inputs.lane, inputs.lane_bounds)
        .map_err(AdmissionRefusal::ExceedsLane)?;

    // 3 — supply and the owner's cap, independently, with the joint pool comparison where the device
    // and the host draw on one DRAM.
    admit_node_memory_bytes(
        selection.claim.total_peak_bytes,
        selection.claim.linear_memory_bytes,
        inputs.report,
        inputs.owner_cap,
    )?;

    // 4 — what the node reserves, and the configured pool bound it must fit.
    //
    // The aggregate is this instance's projection: other co-resident instances contribute through the
    // node's own occupancy ledger, which is where their claims live. Composing it here from one role
    // keeps the reservation derivable at admission time, before any other role's admission is known.
    let aggregate = aggregate(&[(
        inputs.reservation_identity.role.clone(),
        selection.claim.clone(),
        selection.occupancy.clone(),
    )])
    .map_err(AdmissionRefusal::Aggregation)?;
    let reservation = derive_reservation(
        inputs.reservation_identity.clone(),
        &selection.claim,
        &selection.occupancy,
        &aggregate,
        profile,
    )?;
    let pool = check_pool_admissible(&reservation, &selection.claim, inputs.report, profile)?;

    Ok(AdmittedComposition {
        selection,
        aggregate,
        reservation,
        pool,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{DeviceMemorySource, DeviceMemorySupply, MemoryPoolTopology};
    use crate::planner::fixtures::{binding, plan};
    use crate::revision::{BackendClass, BackendImplementationRevision, Maybe};
    use crate::store::{AuthenticationContext, ProfileStore};
    use crate::trust::fixtures as trust_fixtures;

    /// A store holding one profile that authenticates, and the revision it is priced for.
    fn stocked() -> (ProfileStore, BackendImplementationRevision) {
        let running = crate::revision::fixtures::revision(BackendClass::Vulkan);
        let profile = trust_fixtures::profile_for(&running);
        let envelope = trust_fixtures::envelope_for(&profile, &running);
        let mut store = ProfileStore::new();
        store.insert(profile, envelope).expect("stocks");
        (store, running)
    }

    /// The authenticated profile, obtained the only way one can be: selected under both policies.
    fn authenticate<'a>(
        store: &'a ProfileStore,
        running: &'a BackendImplementationRevision,
        policy: &'a crate::trust::ProfileAcceptancePolicy,
    ) -> AuthenticatedProfile<'a> {
        store
            .select(&AuthenticationContext {
                owner: policy,
                run: policy,
                running,
                planner_version: 1,
                now_ms: trust_fixtures::NOW,
            })
            .expect("the fixture profile authenticates")
    }

    /// A report whose measured ceiling is present: the pool step refuses without one, deliberately, so
    /// a fixture exercising the earlier steps has to carry it.
    fn report_with_measured_ceiling() -> DeviceCapabilityReport {
        let mut report = crate::capability::fixtures::report(BackendClass::Vulkan);
        report.measured_max_allocation_bytes = Maybe::Available(1 << 30);
        report
    }

    /// The order of refusals is what these assert, so each fixture makes exactly one step fail.
    fn inputs<'a>(
        plan: &'a LogicalResourcePlan,
        profile: &'a AuthenticatedProfile<'a>,
        report: &'a DeviceCapabilityReport,
        bounds: &'a LaneClaimBounds,
    ) -> AdmissionInputs<'a> {
        AdmissionInputs {
            plan,
            profile,
            report,
            owner_cap: None,
            lane: "trainer",
            lane_bounds: bounds,
            co_resident_roles: 1,
            reservation_identity: ReservationIdentity {
                role: "trainer".into(),
                incarnation: 1,
                device_identity: "device-0".into(),
                sequence: 1,
            },
            frozen_binding: None,
        }
    }

    fn generous_bounds() -> LaneClaimBounds {
        let mut bounds = LaneClaimBounds::default();
        for class in ["vulkan", "metal", "dx12", "cuda", "cpu"] {
            bounds
                .by_backend_class
                .insert(class.to_string(), [0, u64::MAX]);
        }
        bounds
    }

    /// A complete admission produces a claim, a grant, an aggregate and a reservation that agree.
    #[test]
    fn a_composed_admission_yields_a_claim_a_grant_and_the_reservation_it_authorized() {
        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let report = report_with_measured_ceiling();
        let bounds = generous_bounds();

        let admitted = admit_composition(&inputs(&plan, &profile, &report, &bounds))
            .expect("a plan, a certified profile and a machine that fits it admit");

        assert_eq!(
            admitted.grant().logical_resource_plan_hash,
            plan.plan_hash().unwrap(),
            "the grant names the plan it configures"
        );
        assert_eq!(
            admitted.claim().execution_grant_hash,
            admitted.grant().grant_hash().unwrap(),
            "the claim names the grant it prices"
        );
        assert_eq!(admitted.reservation.identity.role, "trainer");
        assert!(admitted.pool.configured_worst_case_bytes() > 0);
    }

    /// The lane refuses before the machine is consulted, so the operator is told which authority said no.
    ///
    /// Both steps would refuse this fixture: the claim is above the lane's ceiling *and* above the
    /// device's supply. The lane is the one that must answer, because a claim absurd for the lane is a
    /// fault in the plan or the profile and not a small machine.
    #[test]
    fn the_lane_refuses_before_supply_does() {
        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let mut report = report_with_measured_ceiling();
        report.device_supply = Maybe::Available(DeviceMemorySupply {
            usable_bytes: 1 << 20,
            source: DeviceMemorySource::LinuxUnifiedMemoryBudget,
        });
        // The ceiling stays under the supply, or the report itself is invalid and refuses a step
        // earlier — which would make this test pass for the wrong reason.
        report.measured_max_allocation_bytes = Maybe::Available(4096);

        let mut bounds = LaneClaimBounds::default();
        bounds.by_backend_class.insert("vulkan".into(), [0, 4096]);

        let refusal =
            admit_composition(&inputs(&plan, &profile, &report, &bounds)).expect_err("refuses");
        assert_eq!(refusal.stage(), "lane-bounds");
        assert!(matches!(refusal, AdmissionRefusal::ExceedsLane(_)));
    }

    /// With the lane satisfied, supply and the owner's cap answer separately.
    #[test]
    fn supply_and_the_owner_cap_are_attributed_separately() {
        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let mut report = report_with_measured_ceiling();
        report.memory_pool = MemoryPoolTopology::Separate;
        let bounds = generous_bounds();

        // Supply too small: a hardware refusal.
        let mut tight = report.clone();
        tight.device_supply = Maybe::Available(DeviceMemorySupply {
            usable_bytes: 4096,
            source: DeviceMemorySource::LinuxUnifiedMemoryBudget,
        });
        // Kept under the supply so the report is valid and the supply comparison is what refuses.
        tight.measured_max_allocation_bytes = Maybe::Available(4096);
        let refusal =
            admit_composition(&inputs(&plan, &profile, &tight, &bounds)).expect_err("refuses");
        assert_eq!(refusal.stage(), "device-authorization");
        assert!(matches!(
            refusal,
            AdmissionRefusal::Unauthorized(DeviceAdmissionRefusal::ExceedsSupply { .. })
        ));

        // Supply ample, owner cap tight: a policy refusal, and it says so.
        let mut capped = inputs(&plan, &profile, &report, &bounds);
        capped.owner_cap = Some(OwnerDeviceCap { max_bytes: 4096 });
        let refusal = admit_composition(&capped).expect_err("refuses");
        assert_eq!(refusal.stage(), "device-authorization");
        assert!(matches!(
            refusal,
            AdmissionRefusal::Unauthorized(DeviceAdmissionRefusal::ExceedsOwnerCap { .. })
        ));
    }

    /// An unmeasured per-allocation ceiling refuses at the pool step rather than admitting.
    ///
    /// This is the state of every real box until an allocation probe exists, and it is deliberately
    /// fail-closed: admitting against an absent measurement admits everything.
    #[test]
    fn an_unmeasured_allocation_ceiling_refuses_at_the_pool_bound() {
        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let mut unmeasured = report_with_measured_ceiling();
        unmeasured.measured_max_allocation_bytes = Maybe::default();
        let bounds = generous_bounds();

        let refusal =
            admit_composition(&inputs(&plan, &profile, &unmeasured, &bounds)).expect_err("refuses");
        assert_eq!(refusal.stage(), "pool-bound");
    }

    /// A frozen configuration is verified, not re-selected.
    #[test]
    fn a_frozen_binding_is_the_configuration_that_gets_composed() {
        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let report = report_with_measured_ceiling();
        let bounds = generous_bounds();

        let frozen = binding(4);
        let mut with_frozen = inputs(&plan, &profile, &report, &bounds);
        with_frozen.frozen_binding = Some(&frozen);
        let admitted = admit_composition(&with_frozen).expect("the frozen configuration admits");
        assert_eq!(admitted.selection.binding, frozen);
    }
}
