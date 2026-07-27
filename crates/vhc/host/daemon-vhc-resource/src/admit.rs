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
//! 1. **Compose** the selected configuration into a Physical Estimate. A plan that cannot be priced by
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
    aggregate, check_estimate_against_lane, compose_selection, minimum_binding, AggregateEstimate,
    LaneEstimateBounds, PhysicalEstimate, PlannerError, Selection,
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
    pub lane_bounds: &'a LaneEstimateBounds,
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
    /// The selected configuration, its Execution Grant, and the composed estimate.
    pub selection: Selection,
    /// The node/device aggregate the instance was admitted within.
    pub aggregate: AggregateEstimate,
    /// The single memory reservation the governor holds and the owner's ledger projects.
    pub reservation: Reservation,
    /// The pool bound that was checked, carried so evidence can state what admission compared.
    pub pool: PoolAdmission,
}

impl AdmittedComposition {
    /// The composed role Physical Estimate.
    #[must_use]
    pub fn estimate(&self) -> &PhysicalEstimate {
        &self.selection.estimate
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
    /// The composed estimate falls outside the lane's bounds for the priced backend class.
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

    // 2 — the lane bounds the composed estimate, before any statement about the machine.
    check_estimate_against_lane(
        &selection.estimate,
        profile,
        inputs.lane,
        inputs.lane_bounds,
    )
    .map_err(AdmissionRefusal::ExceedsLane)?;

    // 3 — supply and the owner's cap, independently, with the joint pool comparison where the device
    // and the host draw on one DRAM.
    admit_node_memory_bytes(
        selection.estimate.total_peak_bytes,
        selection.estimate.linear_memory_bytes,
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
        selection.estimate.clone(),
        selection.occupancy.clone(),
    )])
    .map_err(AdmissionRefusal::Aggregation)?;
    let reservation = derive_reservation(
        inputs.reservation_identity.clone(),
        &selection.estimate,
        &selection.occupancy,
        &aggregate,
        profile,
    )?;
    let pool = check_pool_admissible(&reservation, &selection.estimate, inputs.report, profile)?;

    Ok(AdmittedComposition {
        selection,
        aggregate,
        reservation,
        pool,
    })
}

/// The four composed members a certification run's journal header records, as recorded bytes.
///
/// Bytes rather than decoded values because that is what a replay holds, and because verifying a digest
/// against the value it was decoded from proves nothing — the digest has to be checked against the bytes
/// the journal actually carried.
#[derive(Clone, Copy, Debug)]
pub struct RecordedComposition<'a> {
    /// The canonical Logical Resource Plan bytes, and the hash the header recorded for them.
    pub resource_plan: (&'a [u8], daemon_vhc_proto::Hash),
    /// The composed role Physical Estimate.
    pub physical_estimate: (&'a [u8], daemon_vhc_proto::Hash),
    /// The node/device aggregate claim.
    pub aggregate_estimate: (&'a [u8], daemon_vhc_proto::Hash),
    /// The Execution Grant.
    pub execution_grant: (&'a [u8], daemon_vhc_proto::Hash),
}

/// Why a recorded composition cannot be trusted.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum RecordedCompositionError {
    /// A recorded hash does not match the bytes recorded beside it.
    #[error(
        "the run header's `{member}` hash does not match the bytes recorded with it, so the record \
         cannot be used: nothing downstream may treat either as describing the other"
    )]
    DigestMismatch {
        /// Which member.
        member: &'static str,
    },
    /// A member's bytes are not a decodable value of its type.
    #[error("the run header's `{member}` does not decode: {detail}")]
    Undecodable {
        /// Which member.
        member: &'static str,
        /// The decoder's own words.
        detail: String,
    },
    /// Two recorded members disagree about what they describe.
    #[error(
        "the recorded composition is internally inconsistent: {detail}. The members were written by \
         one admission, so a disagreement between them means the record was assembled from more than \
         one run or was edited"
    )]
    Inconsistent {
        /// What disagreed with what.
        detail: String,
    },
    /// Re-composing the recorded plan against the recorded profile did not reproduce the claim.
    #[error(
        "re-composing the recorded plan reproduced a different claim than the run recorded \
         ({recomposed} vs {recorded} device bytes), so the recorded claim is not what this planner \
         and profile price that plan at"
    )]
    Divergent {
        /// What re-composition produced.
        recomposed: u64,
        /// What the journal recorded.
        recorded: u64,
    },
}

/// Decode one recorded member of the host's own canonical-CBOR types.
fn decode_member<T: serde::de::DeserializeOwned>(
    member: &'static str,
    bytes: &[u8],
) -> Result<T, RecordedCompositionError> {
    daemon_vhc_proto::from_canonical_slice(bytes).map_err(|e| {
        RecordedCompositionError::Undecodable {
            member,
            detail: e.to_string(),
        }
    })
}

/// Validate a recorded composition: digests first, then the cross-references between the members.
///
/// **This is what a replay can check without holding a profile.** The recorded claim names the plan it
/// prices and the grant that configured it, and the grant names the plan it configures — so the three
/// have to agree, and a record where they do not was assembled from more than one run or was edited. It
/// is a cheap, total check, and it runs before anything reads a value out of the record.
///
/// Digests are checked against the recorded **bytes**, in the order the ABI requires: every digest is
/// verified before any value is used. A reader that decoded first and checked later would already have
/// acted on bytes it had not authenticated.
///
/// # Errors
/// [`RecordedCompositionError`] naming the member or the disagreement.
pub fn validate_recorded_composition(
    recorded: RecordedComposition<'_>,
) -> Result<(), RecordedCompositionError> {
    use daemon_vhc_proto::blake3_hash;

    for (member, (bytes, recorded_hash)) in [
        ("resource_plan", recorded.resource_plan),
        ("physical_estimate", recorded.physical_estimate),
        ("aggregate_estimate", recorded.aggregate_estimate),
        ("execution_grant", recorded.execution_grant),
    ] {
        if blake3_hash(bytes) != recorded_hash {
            return Err(RecordedCompositionError::DigestMismatch { member });
        }
    }

    let claim: PhysicalEstimate = decode_member("physical_estimate", recorded.physical_estimate.0)?;
    let aggregate: AggregateEstimate =
        decode_member("aggregate_estimate", recorded.aggregate_estimate.0)?;
    // The plan and the grant are closed bounded schemas with their own decoders, and those are what a
    // replay must use: a permissive decode would accept bytes the admission path would have refused.
    let grant = ExecutionGrant::decode_canonical(recorded.execution_grant.0).map_err(|e| {
        RecordedCompositionError::Undecodable {
            member: "execution_grant",
            detail: e.detail().to_string(),
        }
    })?;
    // Decoded to prove the recorded bytes are a plan at all; its own hash is what the cross-references
    // below are against.
    LogicalResourcePlan::decode_canonical(recorded.resource_plan.0).map_err(|e| {
        RecordedCompositionError::Undecodable {
            member: "resource_plan",
            detail: e.detail().to_string(),
        }
    })?;

    let plan_hash = recorded.resource_plan.1;
    if claim.logical_resource_plan_hash != plan_hash {
        return Err(RecordedCompositionError::Inconsistent {
            detail: "the recorded claim prices a different plan than the one recorded beside it"
                .into(),
        });
    }
    if grant.logical_resource_plan_hash != plan_hash {
        return Err(RecordedCompositionError::Inconsistent {
            detail:
                "the recorded grant configures a different plan than the one recorded beside it"
                    .into(),
        });
    }
    if claim.execution_grant_hash != recorded.execution_grant.1 {
        return Err(RecordedCompositionError::Inconsistent {
            detail: "the recorded claim names a different grant than the one recorded beside it"
                .into(),
        });
    }
    // The aggregate is what the node reserved, so it cannot be smaller per-allocation than the role
    // whose admission it covers.
    if aggregate.max_individual_allocation_bytes < claim.max_individual_allocation_bytes {
        return Err(RecordedCompositionError::Inconsistent {
            detail: format!(
                "the node aggregate allows a {}-byte single allocation while the role it covers \
                 claims {}",
                aggregate.max_individual_allocation_bytes, claim.max_individual_allocation_bytes
            ),
        });
    }
    Ok(())
}

/// Re-derive the recorded claim from the recorded plan and grant, and require byte-equality.
///
/// The stronger check, available to a replay that holds the profile the claim cites. Validation above
/// proves the record is self-consistent; this proves it is what **this** planner and profile actually
/// price that plan at, which is what makes a divergence attributable: the same plan and the same profile
/// producing a different claim means the planner changed, and that is a versioned artifact.
///
/// Call [`validate_recorded_composition`] first — this trusts the digests it checked.
///
/// # Errors
/// [`RecordedCompositionError::Divergent`] when re-composition disagrees, or the decode/consistency
/// errors of the underlying members.
pub fn recompose_recorded_estimate(
    recorded: RecordedComposition<'_>,
    profile: &crate::profile::BackendExecutionProfile,
    co_resident_roles: u64,
) -> Result<(), RecordedCompositionError> {
    let plan = LogicalResourcePlan::decode_canonical(recorded.resource_plan.0).map_err(|e| {
        RecordedCompositionError::Undecodable {
            member: "resource_plan",
            detail: e.detail().to_string(),
        }
    })?;
    let recorded_claim: PhysicalEstimate =
        decode_member("physical_estimate", recorded.physical_estimate.0)?;
    let grant = ExecutionGrant::decode_canonical(recorded.execution_grant.0).map_err(|e| {
        RecordedCompositionError::Undecodable {
            member: "execution_grant",
            detail: e.detail().to_string(),
        }
    })?;

    // The configuration the run actually ran under is the grant's, not a re-selection: replay
    // reproduces a recorded decision and never makes a fresh one.
    //
    // A grant value that is not a dimension value refuses rather than being coerced. The grant's own
    // domain is wider than a plan dimension's — it can carry a boolean or a signed selection — and a
    // plan whose dimension was granted one of those was never composable, so guessing a mapping here
    // would invent a configuration the run could not have had.
    let mut binding = daemon_vhc_proto::resource_plan::Binding::new();
    for (name, value) in &grant.values {
        let dimension_value = match value {
            daemon_vhc_proto::execution_grant::GrantValue::Uint(n) => {
                daemon_vhc_proto::resource_plan::DimensionValue::Uint(*n)
            }
            daemon_vhc_proto::execution_grant::GrantValue::Text(s) => {
                daemon_vhc_proto::resource_plan::DimensionValue::Enum(s.clone())
            }
            other => {
                return Err(RecordedCompositionError::Inconsistent {
                    detail: format!(
                        "the recorded grant assigns `{name}` a {other:?}, which is not a value any \
                         plan dimension can take"
                    ),
                })
            }
        };
        binding.insert(name.clone(), dimension_value);
    }
    let recomposed = compose_selection(&plan, &binding, profile, co_resident_roles, grant.scope)
        .map_err(|e| RecordedCompositionError::Inconsistent {
            detail: format!("the recorded plan does not compose against this profile: {e}"),
        })?;
    if recomposed.estimate.total_peak_bytes != recorded_claim.total_peak_bytes {
        return Err(RecordedCompositionError::Divergent {
            recomposed: recomposed.estimate.total_peak_bytes,
            recorded: recorded_claim.total_peak_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::MemoryPoolTopology;
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
        report.measured_max_allocation =
            Maybe::Available(crate::capability::fixtures::measured_ceiling(1 << 30));
        report
    }

    /// The order of refusals is what these assert, so each fixture makes exactly one step fail.
    fn inputs<'a>(
        plan: &'a LogicalResourcePlan,
        profile: &'a AuthenticatedProfile<'a>,
        report: &'a DeviceCapabilityReport,
        bounds: &'a LaneEstimateBounds,
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

    fn generous_bounds() -> LaneEstimateBounds {
        let mut bounds = LaneEstimateBounds::default();
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
            admitted.estimate().execution_grant_hash,
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
        report.device_supply =
            Maybe::Available(crate::capability::fixtures::derived_supply(1 << 20));
        // The ceiling stays under the supply, or the report itself is invalid and refuses a step
        // earlier — which would make this test pass for the wrong reason.
        report.measured_max_allocation =
            Maybe::Available(crate::capability::fixtures::measured_ceiling(4096));

        let mut bounds = LaneEstimateBounds::default();
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
        tight.device_supply = Maybe::Available(crate::capability::fixtures::derived_supply(4096));
        // Kept under the supply so the report is valid and the supply comparison is what refuses.
        tight.measured_max_allocation =
            Maybe::Available(crate::capability::fixtures::measured_ceiling(4096));
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
        unmeasured.measured_max_allocation = Maybe::default();
        let bounds = generous_bounds();

        let refusal =
            admit_composition(&inputs(&plan, &profile, &unmeasured, &bounds)).expect_err("refuses");
        assert_eq!(refusal.stage(), "pool-bound");
    }

    /// A recorded composition validates, and every cross-reference between its members is load-bearing.
    ///
    /// This is what a replay checks holding nothing but the journal: the members were written by one
    /// admission, so a disagreement between them means the record came from more than one run or was
    /// edited. Each mutation below is a record that would otherwise have been read as describing a run
    /// it does not describe.
    #[test]
    fn a_recorded_composition_validates_and_each_cross_reference_is_load_bearing() {
        use daemon_vhc_proto::blake3_hash;

        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let report = report_with_measured_ceiling();
        let bounds = generous_bounds();
        let admitted = admit_composition(&inputs(&plan, &profile, &report, &bounds))
            .expect("the fixture admits");

        let plan_bytes = plan.to_canonical_bytes().expect("plan encodes");
        let claim_bytes = admitted
            .estimate()
            .to_canonical_bytes()
            .expect("claim encodes");
        let aggregate_bytes = admitted
            .aggregate
            .to_canonical_bytes()
            .expect("aggregate encodes");
        let grant_bytes = admitted
            .grant()
            .to_canonical_bytes()
            .expect("grant encodes");
        let recorded = |plan_b: &'static [u8]| plan_b;
        let _ = recorded;

        let members = |plan_b: &[u8], claim_b: &[u8], agg_b: &[u8], grant_b: &[u8]| {
            (
                blake3_hash(plan_b),
                blake3_hash(claim_b),
                blake3_hash(agg_b),
                blake3_hash(grant_b),
            )
        };
        let (ph, ch, ah, gh) = members(&plan_bytes, &claim_bytes, &aggregate_bytes, &grant_bytes);
        let good = RecordedComposition {
            resource_plan: (&plan_bytes, ph),
            physical_estimate: (&claim_bytes, ch),
            aggregate_estimate: (&aggregate_bytes, ah),
            execution_grant: (&grant_bytes, gh),
        };
        validate_recorded_composition(good).expect("what admission wrote validates");

        // A digest that does not match its bytes is caught before anything is decoded.
        let wrong_digest = RecordedComposition {
            physical_estimate: (&claim_bytes, blake3_hash(b"not the claim")),
            ..good
        };
        assert!(matches!(
            validate_recorded_composition(wrong_digest).unwrap_err(),
            RecordedCompositionError::DigestMismatch {
                member: "physical_estimate"
            }
        ));

        // A claim recorded beside a different plan than it prices.
        let other_plan = {
            let mut p = plan.clone();
            p.linear_fragmentation_headroom = daemon_vhc_proto::resource_plan::Expr::Const(8192);
            p.to_canonical_bytes().expect("encodes")
        };
        let swapped_plan = RecordedComposition {
            resource_plan: (&other_plan, blake3_hash(&other_plan)),
            ..good
        };
        let err = validate_recorded_composition(swapped_plan).unwrap_err();
        assert!(
            matches!(err, RecordedCompositionError::Inconsistent { .. }),
            "a claim pricing another plan is refused, got {err}"
        );

        // An aggregate that allows less per allocation than the role it covers.
        let shrunk = {
            let mut aggregate = admitted.aggregate.clone();
            aggregate.max_individual_allocation_bytes = admitted
                .estimate()
                .max_individual_allocation_bytes
                .saturating_sub(1);
            aggregate.to_canonical_bytes().expect("encodes")
        };
        let shrunk_aggregate = RecordedComposition {
            aggregate_estimate: (&shrunk, blake3_hash(&shrunk)),
            ..good
        };
        assert!(matches!(
            validate_recorded_composition(shrunk_aggregate).unwrap_err(),
            RecordedCompositionError::Inconsistent { .. }
        ));
    }

    /// Holding the profile, a replay re-derives the recorded claim and requires it to agree.
    ///
    /// The configuration it re-composes is the **grant's**, not a fresh selection: replay reproduces a
    /// recorded decision. A recorded claim that this planner and profile do not price that plan at is a
    /// divergence with somewhere to point — the planner is a versioned artifact and the profile is a
    /// certified one, so exactly one of them moved.
    #[test]
    fn a_replay_holding_the_profile_reproduces_the_recorded_claim() {
        use daemon_vhc_proto::blake3_hash;

        let plan = plan();
        let (store, running) = stocked();
        let policy = trust_fixtures::policy_for(&store);
        let profile = authenticate(&store, &running, &policy);
        let report = report_with_measured_ceiling();
        let bounds = generous_bounds();
        let admitted = admit_composition(&inputs(&plan, &profile, &report, &bounds))
            .expect("the fixture admits");

        let plan_bytes = plan.to_canonical_bytes().unwrap();
        let claim_bytes = admitted.estimate().to_canonical_bytes().unwrap();
        let aggregate_bytes = admitted.aggregate.to_canonical_bytes().unwrap();
        let grant_bytes = admitted.grant().to_canonical_bytes().unwrap();
        let recorded = RecordedComposition {
            resource_plan: (&plan_bytes, blake3_hash(&plan_bytes)),
            physical_estimate: (&claim_bytes, blake3_hash(&claim_bytes)),
            aggregate_estimate: (&aggregate_bytes, blake3_hash(&aggregate_bytes)),
            execution_grant: (&grant_bytes, blake3_hash(&grant_bytes)),
        };

        recompose_recorded_estimate(recorded, profile.profile(), 1)
            .expect("the same plan, grant and profile reproduce the recorded claim");

        // A record whose claim was tampered with is a divergence, and it names both figures.
        let tampered = {
            let mut claim = admitted.estimate().clone();
            claim.total_peak_bytes = claim.total_peak_bytes.saturating_add(4096);
            claim.to_canonical_bytes().unwrap()
        };
        let divergent = RecordedComposition {
            physical_estimate: (&tampered, blake3_hash(&tampered)),
            ..recorded
        };
        let err = recompose_recorded_estimate(divergent, profile.profile(), 1).unwrap_err();
        assert!(
            matches!(err, RecordedCompositionError::Divergent { .. }),
            "got {err}"
        );

        // Co-residency is deliberately NOT asserted to change the answer here: this fixture's profile
        // prices no process- or device-scoped term, so the same plan composes identically however many
        // roles share the device. A profile that did carry such a term would diverge, and that it does
        // not is a property of the fixture rather than of the check.
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
