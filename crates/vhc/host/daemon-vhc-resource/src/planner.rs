// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **composition planner** — `PhysicalClaim = compose(LogicalResourcePlan, BackendExecutionProfile)`
//! (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-4]`, `[RC-10]`; §5.4 `[DI-10]`).
//!
//! One implementation, four callers: authoring validation, node admission, sealed-binary
//! conformance, and ceremony preflight. Four independently written planners would disagree about
//! the same machine, and the disagreement would surface as an admission that authoring said was
//! fine. Where a consumer cannot share this implementation — across a language boundary — it is
//! bound instead by the canonical vectors in [`vectors`], gated on both sides.
//!
//! ## Generic over algorithms
//!
//! The planner consumes a plan and a profile and knows nothing about training. It never re-derives
//! logical demand: the plan's own arithmetic produces the logical footprint, and the profile prices
//! it. Those equations are plan semantics, not profile discretion.
//!
//! ## One plan, one claim per backend — never a maximum across backends
//!
//! The same plan composed with a Vulkan profile yields a Vulkan claim; composed with a Metal profile
//! it yields a Metal claim. Each participant is admitted against the claim for the backend it will
//! actually use. Maximizing across backends would let the least efficient backend's overhead refuse
//! a machine that is amply capable on the backend it would have run.

use std::collections::{BTreeMap, BTreeSet};

use daemon_vhc_proto::resource_plan::{Binding, DimensionValue, Domain, LogicalResourcePlan};
use daemon_vhc_proto::{ExecutionGrant, GrantValue, Hash, SelectionScope};
use serde::{Deserialize, Serialize};

use crate::capability::{CapabilityError, DeviceCapabilityReport};
use crate::profile::{
    AllocationScope, BackendExecutionProfile, CompositionRule, CostInput, CostInputs, CostTerm,
    EnforcementClass, ProfileError,
};

/// The planner implementation identity.
///
/// A planner fix invalidates every prior composition result, whether or not its bytes ultimately
/// change, so this is a **candidate** identity: it rides the frozen candidate tuple and a profile
/// names the planner versions it was priced for. Bump it in the change that alters composition.
pub const PLANNER_VERSION: u32 = 1;

/// The composed physical claim for one role instance on one backend.
///
/// The two enforcement classes are carried **distinctly**, per `[RC-10]`: a certification statement
/// that reported one enforcement property over both would be making a claim about the driver's
/// internals that nobody verified.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalClaim {
    /// Persistent device residency.
    pub persistent_device_bytes: u64,
    /// The maximal concurrently-live transient set, priced.
    pub transient_peak_bytes: u64,
    /// Backend workspace across the plan's operations.
    pub workspace_bytes: u64,
    /// Retained allocator pool reservations.
    pub retained_pool_bytes: u64,
    /// First-use compilation allocations.
    pub compilation_bytes: u64,
    /// Transfer staging.
    pub staging_bytes: u64,
    /// The maximum **individual physical** allocation this role will require. Composed from the
    /// plan's largest logical object and the profile's alignment, workspace, pooling and staging
    /// behavior — never authored, and never taken from model geometry alone.
    pub max_individual_allocation_bytes: u64,
    /// The sum of the occupancy terms, before headroom.
    pub subtotal_bytes: u64,
    /// The profile's stated headroom, as bytes over the subtotal.
    pub headroom_bytes: u64,
    /// The profiled hidden-overhead reserve.
    pub hidden_overhead_reserve_bytes: u64,
    /// Total peak: subtotal + headroom + hidden reserve.
    pub total_peak_bytes: u64,
    /// Of the total, the part the governor can intercept and attribute per role.
    pub directly_enforceable_bytes: u64,
    /// Of the total, the part that is budgeted from the profile and observed in aggregate only.
    pub profiled_and_measured_bytes: u64,
    /// The wasm linear-memory cap, taken from the guest's own backend-neutral host-memory terms —
    /// never from a host constant, and never from the device side.
    pub linear_memory_bytes: u64,
    /// The planner that produced this claim.
    pub planner_version: u32,
    /// The profile it was composed with.
    pub profile_digest: Hash,
    /// The plan it prices.
    pub logical_resource_plan_hash: Hash,
    /// The grant that selected the logical configuration.
    pub execution_grant_hash: Hash,
}

impl PhysicalClaim {
    /// The claim's canonical CBOR bytes — what the admitted tuple and the journal record verbatim.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, PlannerError> {
        daemon_vhc_proto::to_canonical_vec(self)
            .map_err(|e| PlannerError::Invalid(format!("claim encoding: {e}")))
    }

    /// blake3 of the canonical bytes — the `physical_claim_hash`.
    pub fn claim_digest(&self) -> Result<Hash, PlannerError> {
        Ok(daemon_vhc_proto::blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Total device residency this claim asks for, across every device-side term.
    ///
    /// Saturating rather than wrapping: an overflow here would silently produce a small total and
    /// admit a claim far above any entitlement, which is the one arithmetic outcome that must not be
    /// possible.
    #[must_use]
    pub fn device_total_bytes(&self) -> u64 {
        self.persistent_device_bytes
            .saturating_add(self.transient_peak_bytes)
            .saturating_add(self.workspace_bytes)
            .saturating_add(self.retained_pool_bytes)
            .saturating_add(self.compilation_bytes)
            .saturating_add(self.staging_bytes)
    }
}

/// One term's contribution, kept with everything the aggregate needs to compose it at its real
/// scope rather than guessing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedTerm {
    /// The term's stable aggregation key.
    pub aggregation_key: String,
    /// The scope it is allocated at.
    pub scope: AllocationScope,
    /// How simultaneous role claims combine.
    pub composition_rule: CompositionRule,
    /// Whether the governor can enforce it.
    pub enforcement: EnforcementClass,
    /// The evaluated bytes.
    pub bytes: u64,
}

/// A role's per-term breakdown, which the node/device aggregate composes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedOccupancy {
    /// The terms, keyed by aggregation key.
    pub terms: Vec<ScopedTerm>,
}

/// The node/device aggregate claim over every admitted co-resident role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateClaim {
    /// The occupancy bytes reserved for this node/device.
    pub occupancy_bytes: u64,
    /// The largest single allocation any co-resident role will require. A **maximum constraint**,
    /// checked against the capability report's limit — never summed, and not itself a reservation.
    pub max_individual_allocation_bytes: u64,
    /// Of the occupancy, the directly enforceable part.
    pub directly_enforceable_bytes: u64,
    /// Of the occupancy, the profiled-and-measured part.
    pub profiled_and_measured_bytes: u64,
    /// The roles this aggregate covers, in deterministic order.
    pub roles: Vec<String>,
    /// The planner that produced it.
    pub planner_version: u32,
}

impl AggregateClaim {
    /// The aggregate's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, PlannerError> {
        daemon_vhc_proto::to_canonical_vec(self)
            .map_err(|e| PlannerError::Invalid(format!("aggregate encoding: {e}")))
    }

    /// blake3 of the canonical bytes — the `aggregate_claim_hash`.
    pub fn claim_digest(&self) -> Result<Hash, PlannerError> {
        Ok(daemon_vhc_proto::blake3_hash(&self.to_canonical_bytes()?))
    }
}

/// Which authority a divergence is attributed to (`[RC-6]`).
///
/// With three artifacts and a planner between the algorithm and the machine, "measured exceeded
/// declared" is too coarse to act on: it does not say which authority is wrong, and each has a
/// different owner, a different fix and a different re-freeze consequence. A gate that cannot
/// distinguish them sends the wrong wave to do the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceAuthority {
    /// A logical term is missing or under-bounded.
    LogicalResourcePlan,
    /// A workspace formula, pooling model or staging behavior under-predicts.
    BackendExecutionProfile,
    /// Composition or logical-choice selection is wrong.
    PlannerOrSelector,
    /// The machine does not have what it reported.
    CapabilityProbe,
    /// Interception, reservation, scoped aggregation, runtime accounting or measurement is wrong.
    ResourceGovernor,
}

/// Why a composition or a selection failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlannerError {
    /// The inputs are structurally unusable.
    #[error("composition input is invalid: {0}")]
    Invalid(String),
    /// The plan and the profile do not fit each other.
    #[error(transparent)]
    Profile(#[from] ProfileError),
    /// The capability report cannot be used.
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    /// The plan itself was refused.
    #[error("logical resource plan refused: {0}")]
    Plan(String),
    /// No admissible logical configuration exists on this machine.
    #[error(
        "no admissible logical configuration exists: the smallest configuration the plan permits \
         composes to {required} bytes against a {available}-byte budget"
    )]
    NoAdmissibleConfiguration {
        /// What the minimum configuration needs.
        required: u64,
        /// What the machine offers.
        available: u64,
    },
    /// The composed claim exceeds the machine's per-allocation limit.
    #[error(
        "the composed claim requires a {required}-byte single allocation, above this device's \
         measured {limit}-byte limit"
    )]
    AllocationCeilingExceeded {
        /// The claim's largest single allocation.
        required: u64,
        /// The device's measured limit.
        limit: u64,
    },
    /// The composed Physical Claim falls outside the participation lane's sanity bounds.
    #[error(
        "the composed physical claim's {total} device bytes fall outside lane `{lane}`'s bounds \
         [{min}, {max}] for backend class `{backend_class}`"
    )]
    PhysicalClaimExceedsLane {
        /// The lane.
        lane: String,
        /// The claim's device total.
        total: u64,
        /// The lane's lower bound for this backend class.
        min: u64,
        /// The lane's upper bound for this backend class.
        max: u64,
        /// The class whose bounds were applied.
        backend_class: &'static str,
    },
    /// The lane states no sanity bounds for the backend class the claim was priced for.
    #[error(
        "lane `{lane}` states no claim bounds for backend class `{backend_class}`, so a claim priced \
         for it cannot be sanity-checked; a lane that has not been given bounds for a backend does \
         not admit against that backend"
    )]
    LaneStatesNoBoundsForClass {
        /// The lane.
        lane: String,
        /// The class that has no bounds.
        backend_class: &'static str,
    },
    /// Two profiles disagree about how one shared term composes.
    #[error(
        "aggregation key `{key}` is given incompatible scope/rule pairs by the co-resident roles' \
         profiles; admission refuses rather than choosing one"
    )]
    IncompatibleSharedTerm {
        /// The key in question.
        key: String,
    },
}

impl From<daemon_vhc_proto::PlanRefusal> for PlannerError {
    fn from(value: daemon_vhc_proto::PlanRefusal) -> Self {
        Self::Plan(value.to_string())
    }
}

/// The logical inputs the planner supplies to a profile formula, derived from the plan.
fn base_inputs(
    plan: &LogicalResourcePlan,
    binding: &Binding,
    co_resident_roles: u64,
) -> Result<CostInputs, PlannerError> {
    let footprint = plan.footprint(binding)?;
    Ok(CostInputs::from([
        (
            CostInput::PersistentLogicalBytes,
            footprint.device_persistent_bytes,
        ),
        (
            CostInput::TransientPeakLogicalBytes,
            footprint.device_transient_peak_bytes,
        ),
        (
            CostInput::LargestLogicalObjectBytes,
            footprint.largest_logical_tensor_bytes,
        ),
        (
            CostInput::TransferWindowBytes,
            footprint.largest_transfer_window_bytes,
        ),
        (
            CostInput::LogicalBytes,
            footprint.largest_logical_tensor_bytes,
        ),
        (CostInput::ElementCount, 0),
        (CostInput::InFlight, 1),
        (CostInput::CoResidentRoleCount, co_resident_roles),
    ]))
}

fn add(a: u64, b: u64) -> Result<u64, PlannerError> {
    a.checked_add(b)
        .ok_or_else(|| PlannerError::Invalid("checked u64 overflow composing a claim".into()))
}

/// Compose one role's Physical Claim, and the per-term breakdown the aggregate needs.
///
/// The claim is a **conservative upper bound for the admitted configuration**, not an average-case
/// prediction.
pub fn compose(
    plan: &LogicalResourcePlan,
    binding: &Binding,
    profile: &BackendExecutionProfile,
    co_resident_roles: u64,
) -> Result<(PhysicalClaim, ScopedOccupancy), PlannerError> {
    profile.validate()?;
    plan.validate()?;

    // A plan needing an operation class or dtype the backend does not have is a typed refusal, never
    // a fallback to an estimate.
    let families: BTreeSet<String> = plan.operations.iter().map(|o| o.family.clone()).collect();
    let dtypes: BTreeSet<String> = plan
        .tensors
        .iter()
        .map(|t| t.dtype.spelling().to_string())
        .collect();
    profile.supports(&families, &dtypes)?;

    let footprint = plan.footprint(binding)?;
    let inputs = base_inputs(plan, binding, co_resident_roles)?;
    let mut occupancy = ScopedOccupancy::default();

    let mut record = |term: &CostTerm, bytes: u64| {
        occupancy.terms.push(ScopedTerm {
            aggregation_key: term.aggregation_key.clone(),
            scope: term.scope,
            composition_rule: term.composition_rule,
            enforcement: term.enforcement,
            bytes,
        });
    };

    // Workspace, per operation, priced by that operation's own family formula.
    let mut workspace_bytes = 0u64;
    for op in &plan.operations {
        let Some(formula) = profile.workspace_for(&op.family) else {
            return Err(PlannerError::Invalid(format!(
                "operation `{}` has family `{}`, which the profile does not price",
                op.name, op.family
            )));
        };
        let mut op_inputs = inputs.clone();
        // The priced object is the largest tensor this operation touches.
        let mut largest = 0u64;
        for name in op.inputs.iter().chain(op.outputs.iter()) {
            if let Some(tensor) = plan.tensors.iter().find(|t| &t.name == name) {
                largest = largest.max(LogicalResourcePlan::tensor_bytes(tensor, binding)?);
            }
        }
        op_inputs.insert(CostInput::LogicalBytes, largest);
        op_inputs.insert(CostInput::InFlight, op.max_in_flight.evaluate(binding)?);
        let bytes = formula.term.formula.evaluate(&op_inputs)?;
        record(&formula.term, bytes);
        workspace_bytes = add(workspace_bytes, bytes)?;
    }

    // Standing terms.
    let mut persistent_device_bytes = 0u64;
    let mut max_individual_allocation_bytes = 0u64;
    let mut standing_occupancy = 0u64;
    for term in &profile.standing_terms {
        let bytes = term.formula.evaluate(&inputs)?;
        record(term, bytes);
        match term.scope {
            // A per-allocation term is a maximum constraint, not bytes to reserve. Summing it would
            // turn a ceiling into occupancy and refuse machines for memory nobody is holding.
            AllocationScope::PerAllocation => {
                max_individual_allocation_bytes = max_individual_allocation_bytes.max(bytes);
            }
            AllocationScope::PerRoleInstance => {
                persistent_device_bytes = add(persistent_device_bytes, bytes)?;
                standing_occupancy = add(standing_occupancy, bytes)?;
            }
            AllocationScope::PerProcess | AllocationScope::PerDevice => {
                standing_occupancy = add(standing_occupancy, bytes)?;
            }
        }
    }

    let transient_peak_bytes = footprint.device_transient_peak_bytes;
    let retained_pool_bytes = profile.pooling.retained_reservation.evaluate(&inputs)?;
    let compilation_bytes = profile.compilation.allocations.evaluate(&inputs)?;
    let staging_bytes = profile.staging.per_window.evaluate(&inputs)?;

    let subtotal_bytes = [
        standing_occupancy,
        transient_peak_bytes,
        workspace_bytes,
        retained_pool_bytes,
        compilation_bytes,
        staging_bytes,
    ]
    .into_iter()
    .try_fold(0u64, add)?;

    let with_headroom = profile.headroom.apply(subtotal_bytes)?;
    let headroom_bytes = with_headroom.saturating_sub(subtotal_bytes);
    let hidden_overhead_reserve_bytes = profile.headroom.hidden_overhead_reserve_bytes;
    let total_peak_bytes = add(with_headroom, hidden_overhead_reserve_bytes)?;

    // A pool block larger than any single tensor is still a single allocation the driver must
    // satisfy, so it participates in the per-allocation maximum.
    max_individual_allocation_bytes = max_individual_allocation_bytes
        .max(profile.pooling.reservation_block_bytes)
        .max(staging_bytes);

    // The two enforcement classes, split. Everything the governor cannot intercept — the profiled
    // terms plus the hidden reserve — is budgeted and observed in aggregate, never individually
    // enforced.
    let mut directly_enforceable_bytes = 0u64;
    let mut profiled_and_measured_bytes = hidden_overhead_reserve_bytes;
    for term in occupancy.terms.iter().filter(|t| t.scope.is_occupancy()) {
        match term.enforcement {
            EnforcementClass::DirectlyEnforceable => {
                directly_enforceable_bytes = add(directly_enforceable_bytes, term.bytes)?;
            }
            EnforcementClass::ProfiledAndMeasured => {
                profiled_and_measured_bytes = add(profiled_and_measured_bytes, term.bytes)?;
            }
        }
    }
    // Transients, pool retention, compilation and staging are the backend's own doing; the governor
    // pre-authorizes them from the profile rather than intercepting each one.
    profiled_and_measured_bytes = [
        transient_peak_bytes,
        retained_pool_bytes,
        compilation_bytes,
        staging_bytes,
        headroom_bytes,
    ]
    .into_iter()
    .try_fold(profiled_and_measured_bytes, add)?;

    Ok((
        PhysicalClaim {
            persistent_device_bytes,
            transient_peak_bytes,
            workspace_bytes,
            retained_pool_bytes,
            compilation_bytes,
            staging_bytes,
            max_individual_allocation_bytes,
            subtotal_bytes,
            headroom_bytes,
            hidden_overhead_reserve_bytes,
            total_peak_bytes,
            directly_enforceable_bytes,
            profiled_and_measured_bytes,
            linear_memory_bytes: footprint.linear_peak_bytes,
            planner_version: PLANNER_VERSION,
            profile_digest: profile.profile_digest()?,
            logical_resource_plan_hash: plan.plan_hash()?,
            execution_grant_hash: Hash([0u8; 32]),
        },
        occupancy,
    ))
}

/// Compose the node/device aggregate over every admitted co-resident role, at each term's real
/// scope (`[RC-10]`, `[NC-1]`).
///
/// Per-role terms compose per role; process-scoped terms compose once per process; device-scoped
/// once over every process using the device; per-allocation aggregates by maximum and is never
/// summed. Shared terms match by aggregation key, and profiles that give the same key incompatible
/// scopes or rules are mutually incompatible — admission refuses rather than choosing one.
pub fn aggregate(
    roles: &[(String, PhysicalClaim, ScopedOccupancy)],
) -> Result<AggregateClaim, PlannerError> {
    let mut occupancy_bytes = 0u64;
    let mut max_individual_allocation_bytes = 0u64;
    let mut enforceable = 0u64;
    let mut profiled = 0u64;
    // Shared terms are charged once per key; the first sighting fixes the scope and rule.
    let mut shared: BTreeMap<String, (AllocationScope, CompositionRule, u64, EnforcementClass)> =
        BTreeMap::new();

    for (_, claim, occ) in roles {
        max_individual_allocation_bytes =
            max_individual_allocation_bytes.max(claim.max_individual_allocation_bytes);
        for term in &occ.terms {
            if let Some((scope, rule, _, _)) = shared.get(&term.aggregation_key) {
                if *scope != term.scope || *rule != term.composition_rule {
                    return Err(PlannerError::IncompatibleSharedTerm {
                        key: term.aggregation_key.clone(),
                    });
                }
            }
            match term.scope {
                AllocationScope::PerAllocation => {
                    max_individual_allocation_bytes =
                        max_individual_allocation_bytes.max(term.bytes);
                }
                AllocationScope::PerRoleInstance => match term.enforcement {
                    EnforcementClass::DirectlyEnforceable => {
                        occupancy_bytes = add(occupancy_bytes, term.bytes)?;
                        enforceable = add(enforceable, term.bytes)?;
                    }
                    EnforcementClass::ProfiledAndMeasured => {
                        occupancy_bytes = add(occupancy_bytes, term.bytes)?;
                        profiled = add(profiled, term.bytes)?;
                    }
                },
                AllocationScope::PerProcess | AllocationScope::PerDevice => {
                    let entry = shared.entry(term.aggregation_key.clone()).or_insert((
                        term.scope,
                        term.composition_rule,
                        0,
                        term.enforcement,
                    ));
                    entry.2 = match term.composition_rule {
                        // Charged once per key, however many roles share it: multiplying a shared
                        // process pool by the role count is the mistake this scope exists to avoid.
                        CompositionRule::OncePerKey => entry.2.max(term.bytes),
                        CompositionRule::Max => entry.2.max(term.bytes),
                        CompositionRule::Sum => add(entry.2, term.bytes)?,
                    };
                }
            }
        }
        // The per-role parts of the claim that are not individual terms.
        occupancy_bytes = add(
            occupancy_bytes,
            claim
                .transient_peak_bytes
                .saturating_add(claim.workspace_bytes)
                .saturating_add(claim.retained_pool_bytes)
                .saturating_add(claim.compilation_bytes)
                .saturating_add(claim.staging_bytes),
        )?;
        profiled = add(
            profiled,
            claim
                .transient_peak_bytes
                .saturating_add(claim.retained_pool_bytes)
                .saturating_add(claim.compilation_bytes)
                .saturating_add(claim.staging_bytes),
        )?;
        enforceable = add(enforceable, claim.workspace_bytes)?;
    }

    for (_, (_, _, bytes, enforcement)) in shared {
        occupancy_bytes = add(occupancy_bytes, bytes)?;
        match enforcement {
            EnforcementClass::DirectlyEnforceable => enforceable = add(enforceable, bytes)?,
            EnforcementClass::ProfiledAndMeasured => profiled = add(profiled, bytes)?,
        }
    }

    Ok(AggregateClaim {
        occupancy_bytes,
        max_individual_allocation_bytes,
        directly_enforceable_bytes: enforceable,
        profiled_and_measured_bytes: profiled,
        roles: roles.iter().map(|(r, _, _)| r.clone()).collect(),
        planner_version: PLANNER_VERSION,
    })
}

/// A participation lane's claim sanity bounds, keyed by the backend class a profile prices.
///
/// The legacy bounds were one `[min, max]` pair applied to a figure the guest declared, and at the
/// certification minor the guest declares no physical figure at all: the claim is composed here, from a
/// plan and a profile, and the same plan prices differently on every backend. One pair of scalars
/// therefore cannot sanity-bound them — a bound loose enough for the most expensive backend is no bound
/// on the cheapest, which is the failure mode a sanity check exists to catch.
///
/// **Keyed by class, not by profile digest.** A digest key would re-key on every profile revision, so a
/// driver update would silently leave a lane with no bounds; the class is the coarsest key that still
/// separates the backends whose prices differ. It is the same key the lane already uses to decide which
/// backends it will host.
///
/// A class with no entry is **not** unbounded. It is a lane that has not been configured for that
/// backend, and it refuses — see [`PlannerError::LaneStatesNoBoundsForClass`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaneClaimBounds {
    /// `[min, max]` device bytes per backend class slug.
    pub by_backend_class: std::collections::BTreeMap<String, [u64; 2]>,
}

impl LaneClaimBounds {
    /// The bounds this lane states for one backend class, if any.
    #[must_use]
    pub fn for_class(&self, class: crate::revision::BackendClass) -> Option<[u64; 2]> {
        self.by_backend_class.get(class.slug()).copied()
    }
}

/// Apply a lane's **profile-keyed** sanity bounds to a composed role Physical Claim.
///
/// Stage order is load-bearing and is the reason this is its own function: the lane check happens
/// **after composition and before capability or owner authorization**. A claim that is absurd for the
/// lane — a plan and profile combination pricing a role at ten times what this lane ever hosts — should
/// be refused as a lane violation, not reported as a machine that is too small. The two refusals send an
/// operator to different places, and the order decides which one they get.
///
/// The bound applied is the claim's device total, which is the figure the lane's own device minima and
/// the owner's cap are expressed in.
///
/// # Errors
/// [`PlannerError::PhysicalClaimExceedsLane`] when the total falls outside the class's bounds, and
/// [`PlannerError::LaneStatesNoBoundsForClass`] when the lane states none for it.
pub fn check_claim_against_lane(
    claim: &PhysicalClaim,
    profile: &BackendExecutionProfile,
    lane: &str,
    bounds: &LaneClaimBounds,
) -> Result<(), PlannerError> {
    let class = profile.backend_class;
    let Some([min, max]) = bounds.for_class(class) else {
        return Err(PlannerError::LaneStatesNoBoundsForClass {
            lane: lane.to_string(),
            backend_class: class.slug(),
        });
    };
    let total = claim.total_peak_bytes;
    if total < min || total > max {
        return Err(PlannerError::PhysicalClaimExceedsLane {
            lane: lane.to_string(),
            total,
            min,
            max,
            backend_class: class.slug(),
        });
    }
    Ok(())
}

/// Validate a composed claim against a capability report and, where set, the owner's cap.
///
/// Admission evaluates the **composed claim**, not the plan.
///
/// The device-memory question is **two independent comparisons**, against host-measured supply and
/// against the owner's optional cap. They used to be one `min` of a single field that could hold either
/// a platform measurement or an operator's number depending on a sibling enum — so the two authorities
/// overwrote each other, and a refusal could not say which had refused. An operator whose own cap is the
/// binding constraint should not be told their hardware is too small.
///
/// The owner's cap only ever *tightens* supply; it can never supply the mandatory hardware figure, which
/// is measured on this node because this node is the thing that can measure it.
///
/// # Errors
/// [`PlannerError`] when the claim exceeds measured supply, exceeds the owner's cap, exceeds the
/// per-allocation ceiling, or when this platform has no trustworthy supply derivation at all.
pub fn validate_against(
    claim: &PhysicalClaim,
    report: &DeviceCapabilityReport,
    owner_cap: Option<crate::capability::OwnerDeviceCap>,
) -> Result<(), PlannerError> {
    report.validate()?;
    // Both comparisons, and the typed refusal is preserved in the message so a reader can tell a
    // policy refusal from a hardware one — while the shape stays the one callers already match on.
    //
    // The claim's host side rides along because on a unified device it is not a separate budget: the
    // linear-memory cap and the device residency come out of one DRAM pool, so a role that fits each
    // figure separately can still over-commit the machine. Passing both is what lets the report's
    // topology decide whether a joint comparison applies, instead of this call site assuming.
    if let Err(refusal) = crate::capability::admit_node_memory_bytes(
        claim.total_peak_bytes,
        claim.linear_memory_bytes,
        report,
        owner_cap,
    ) {
        return Err(PlannerError::NoAdmissibleConfiguration {
            required: refusal.claimed_bytes(),
            available: refusal.binding_limit_bytes(),
        });
    }
    let limit = report.max_allocation_bytes()?;
    if claim.max_individual_allocation_bytes > limit {
        return Err(PlannerError::AllocationCeilingExceeded {
            required: claim.max_individual_allocation_bytes,
            limit,
        });
    }
    Ok(())
}

/// How the host chooses among the plan's admissible logical configurations.
///
/// Selection policy is **host and owner policy, not guest policy**.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionPolicy {
    /// Raise each dimension as far as the machine admits, in declaration order.
    LargestAdmissible,
    /// Use exactly this configuration, and refuse if it is not admissible. This is what a
    /// participant does with a frozen uniform grant: verify, do not reselect.
    Fixed(Binding),
}

/// The selected configuration and the claim it produces.
#[derive(Clone, Debug)]
pub struct Selection {
    /// The chosen binding.
    pub binding: Binding,
    /// The Execution Grant carrying it — logical values only.
    pub grant: ExecutionGrant,
    /// The claim for the chosen configuration, with the grant digest stamped in.
    pub claim: PhysicalClaim,
    /// The per-term breakdown for the aggregate.
    pub occupancy: ScopedOccupancy,
}

fn grant_for(
    plan: &LogicalResourcePlan,
    binding: &Binding,
    scope: SelectionScope,
) -> Result<ExecutionGrant, PlannerError> {
    let values = binding
        .iter()
        .map(|(name, value)| {
            let v = match value {
                DimensionValue::Uint(n) => GrantValue::Uint(*n),
                DimensionValue::Enum(s) => GrantValue::Text(s.clone()),
            };
            (name.clone(), v)
        })
        .collect();
    Ok(ExecutionGrant {
        logical_resource_plan_hash: plan.plan_hash()?,
        scope,
        values,
    })
}

/// The all-minimum binding: the smallest configuration the plan permits.
pub(crate) fn minimum_binding(plan: &LogicalResourcePlan) -> Result<Binding, PlannerError> {
    let mut binding = Binding::new();
    for dim in &plan.dimensions {
        let value = match &dim.domain {
            Domain::UintRange { lo, .. } => DimensionValue::Uint(*lo),
            Domain::Enum(values) => DimensionValue::Enum(
                values
                    .first()
                    .ok_or_else(|| {
                        PlannerError::Invalid(format!("dimension `{}` has an empty enum", dim.name))
                    })?
                    .clone(),
            ),
        };
        binding.insert(dim.name.clone(), value);
    }
    Ok(binding)
}

/// Compose one configuration and the Execution Grant that carries it — **without authorizing it**.
///
/// Separate from [`select`] because the authorization steps have a required order that a function
/// doing both cannot express: a composed claim is bounded by the participation lane *before* it is
/// compared against the machine's supply or the owner's cap, so that an absurd claim is refused as a
/// lane violation rather than reported as a machine that is too small. A caller needing that order —
/// admission is the one that does — composes here and then authorizes in sequence.
///
/// The grant's digest is stamped into the claim, so the claim names the configuration it prices.
///
/// # Errors
/// [`PlannerError`] when the plan and profile do not compose, or the grant cannot be encoded.
pub fn compose_selection(
    plan: &LogicalResourcePlan,
    binding: &Binding,
    profile: &BackendExecutionProfile,
    co_resident_roles: u64,
    scope: SelectionScope,
) -> Result<Selection, PlannerError> {
    let (mut claim, occupancy) = compose(plan, binding, profile, co_resident_roles)?;
    let grant = grant_for(plan, binding, scope)?;
    claim.execution_grant_hash = grant.grant_hash()?;
    Ok(Selection {
        binding: binding.clone(),
        grant,
        claim,
        occupancy,
    })
}

/// Compose over the plan's bounded choice set, select an admissible configuration under `policy`,
/// and deliver it as an Execution Grant (`[RC-11]`).
///
/// The capability report is **validation supply**; it is not an input to `compose`. That is what
/// keeps the claim a function of the plan and the profile alone, so the same configuration prices
/// identically on two machines with the same backend.
///
/// [`SelectionPolicy::LargestAdmissible`] performs a deterministic **coordinate-wise ascent**: from
/// the smallest admissible configuration, each dimension in declaration order is raised as far as
/// the machine still admits, holding the others fixed. This is not a search for a global optimum,
/// and it does not claim to be one — selection is host policy, and what the specification requires
/// of it is that two hosts given the same inputs reach the same answer. An exhaustive search over a
/// multi-dimensional choice set would be unbounded, and an unbounded search on the admission path
/// is not a policy anyone can operate.
pub fn select(
    plan: &LogicalResourcePlan,
    profile: &BackendExecutionProfile,
    report: &DeviceCapabilityReport,
    owner_cap: Option<crate::capability::OwnerDeviceCap>,
    co_resident_roles: u64,
    scope: SelectionScope,
    policy: &SelectionPolicy,
) -> Result<Selection, PlannerError> {
    let finish = |binding: Binding| -> Result<Selection, PlannerError> {
        compose_selection(plan, &binding, profile, co_resident_roles, scope)
    };

    match policy {
        SelectionPolicy::Fixed(binding) => {
            let selection = finish(binding.clone())?;
            validate_against(&selection.claim, report, owner_cap)?;
            Ok(selection)
        }
        SelectionPolicy::LargestAdmissible => {
            let mut binding = minimum_binding(plan)?;
            // The floor has to be admissible, or nothing is. That refusal is the composed claim,
            // the profile and the report working as designed — the machine genuinely cannot host
            // this run — and it names the numbers rather than the machine.
            let floor = finish(binding.clone())?;
            validate_against(&floor.claim, report, owner_cap)?;

            for dim in &plan.dimensions {
                let candidates: Vec<DimensionValue> = match &dim.domain {
                    Domain::UintRange { lo, hi } => {
                        // Ascend from the top so the first admissible value found is the largest.
                        (*lo..=*hi).rev().map(DimensionValue::Uint).collect()
                    }
                    Domain::Enum(values) => values
                        .iter()
                        .rev()
                        .cloned()
                        .map(DimensionValue::Enum)
                        .collect(),
                };
                let held = binding.get(&dim.name).cloned();
                for candidate in candidates {
                    binding.insert(dim.name.clone(), candidate);
                    let trial = finish(binding.clone())?;
                    if validate_against(&trial.claim, report, owner_cap).is_ok() {
                        break;
                    }
                    if let Some(previous) = &held {
                        binding.insert(dim.name.clone(), previous.clone());
                    }
                }
            }
            finish(binding)
        }
    }
}

/// Canonical cross-language conformance vectors for the planner (`[DI-10]` form 2).
///
/// A consumer that cannot link this implementation — the registry service validates on its own
/// side, in another language — is bound by these instead: `(plan, profile, expected claim)`,
/// versioned with the planner and gated on both sides. Two hand-maintained planners with nothing
/// binding them would disagree about the same machine, and the disagreement would surface as an
/// admission that authoring said was fine.
pub mod vectors {
    use super::*;

    /// One vector: the inputs and the claim they must produce.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct PlannerVector {
        /// A stable name for the case.
        pub name: String,
        /// The planner version this vector was produced by.
        pub planner_version: u32,
        /// The plan's canonical bytes.
        pub plan_bytes: Vec<u8>,
        /// The profile's canonical bytes.
        pub profile_bytes: Vec<u8>,
        /// The selected binding, as the grant's canonical bytes.
        pub grant_bytes: Vec<u8>,
        /// The number of co-resident roles the vector composes for.
        pub co_resident_roles: u64,
        /// The claim's canonical bytes.
        pub expected_claim_bytes: Vec<u8>,
    }

    /// Build a vector from real inputs by running the planner over them.
    pub fn derive(
        name: &str,
        plan: &LogicalResourcePlan,
        binding: &Binding,
        profile: &BackendExecutionProfile,
        co_resident_roles: u64,
    ) -> Result<PlannerVector, PlannerError> {
        let (mut claim, _) = compose(plan, binding, profile, co_resident_roles)?;
        let grant = grant_for(plan, binding, plan.selection_scope)?;
        claim.execution_grant_hash = grant.grant_hash()?;
        Ok(PlannerVector {
            name: name.to_string(),
            planner_version: PLANNER_VERSION,
            plan_bytes: plan.to_canonical_bytes()?,
            profile_bytes: profile.to_canonical_bytes()?,
            grant_bytes: grant
                .to_canonical_bytes()
                .map_err(|e| PlannerError::Plan(e.to_string()))?,
            co_resident_roles,
            expected_claim_bytes: claim.to_canonical_bytes()?,
        })
    }

    /// Re-run a vector through this planner and confirm it reproduces the recorded claim exactly.
    pub fn check(vector: &PlannerVector) -> Result<(), PlannerError> {
        if vector.planner_version != PLANNER_VERSION {
            return Err(PlannerError::Invalid(format!(
                "vector `{}` was produced by planner version {}, but this planner is version \
                 {PLANNER_VERSION}; a planner change invalidates prior composition evidence",
                vector.name, vector.planner_version
            )));
        }
        let plan = LogicalResourcePlan::decode_canonical(&vector.plan_bytes)?;
        let profile: BackendExecutionProfile =
            daemon_vhc_proto::from_canonical_slice(&vector.profile_bytes)
                .map_err(|e| PlannerError::Invalid(format!("vector profile: {e}")))?;
        let grant = ExecutionGrant::decode_canonical(&vector.grant_bytes)
            .map_err(|e| PlannerError::Plan(e.to_string()))?;
        let binding = grant
            .bind_to(&plan)
            .map_err(|e| PlannerError::Plan(e.to_string()))?;
        let (mut claim, _) = compose(&plan, &binding, &profile, vector.co_resident_roles)?;
        claim.execution_grant_hash = grant.grant_hash()?;
        if claim.to_canonical_bytes()? != vector.expected_claim_bytes {
            return Err(PlannerError::Invalid(format!(
                "vector `{}` did not reproduce its recorded claim",
                vector.name
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use daemon_vhc_proto::resource_plan::{
        Dimension, Dtype, Expr, Lifetime, LinearLifetime, LinearMemoryTerm, OperationDecl,
        Retention, TensorDecl, TransferDecl, TransferKind,
    };

    pub(crate) fn plan() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![Dimension {
                name: "micro_batch".into(),
                domain: Domain::UintRange { lo: 1, hi: 8 },
            }],
            tensors: vec![
                TensorDecl {
                    name: "activations".into(),
                    shape: vec![Expr::Dimension("micro_batch".into()), Expr::Const(4096)],
                    dtype: Dtype::F32,
                    layout: vec![],
                    lifetime: Lifetime::Transient("forward".into()),
                },
                TensorDecl {
                    name: "params".into(),
                    shape: vec![Expr::Const(1_048_576)],
                    dtype: Dtype::F32,
                    layout: vec![],
                    lifetime: Lifetime::Persistent(Retention::Run),
                },
            ],
            operations: vec![OperationDecl {
                name: "matmul".into(),
                family: "gemm".into(),
                inputs: vec!["params".into()],
                outputs: vec!["activations".into()],
                workspace_class: Some("reduction".into()),
                max_in_flight: Expr::Const(2),
            }],
            transfers: vec![TransferDecl {
                name: "window".into(),
                kind: TransferKind::Ingest,
                window_bytes: Expr::Const(65_536),
                max_in_flight: Expr::Const(1),
            }],
            linear_memory: vec![LinearMemoryTerm {
                name: "index".into(),
                lifetime: LinearLifetime::Persistent,
                bytes: Expr::Const(32_768),
            }],
            transient_live_sets: vec![vec!["forward".into()]],
            linear_fragmentation_headroom: Expr::Const(4096),
        }
    }

    pub(crate) fn binding(micro_batch: u64) -> Binding {
        Binding::from([("micro_batch".to_string(), DimensionValue::Uint(micro_batch))])
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{binding, plan};
    use super::*;
    use crate::capability::fixtures::report;
    use crate::profile::fixtures::profile;
    use crate::revision::BackendClass;

    #[test]
    fn composition_is_deterministic_and_stamps_its_authorities() {
        let (a, _) = compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        let (b, _) = compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.planner_version, PLANNER_VERSION);
        assert_eq!(
            a.profile_digest,
            profile(BackendClass::Vulkan).profile_digest().unwrap()
        );
        assert_eq!(a.logical_resource_plan_hash, plan().plan_hash().unwrap());
        assert!(a.total_peak_bytes > a.subtotal_bytes, "headroom is added");
    }

    /// One plan, one claim per backend — and the claims differ, which is the point.
    #[test]
    fn the_same_plan_prices_differently_per_backend_and_is_never_maximized_across_them() {
        let mut metal = profile(BackendClass::Metal);
        metal.pooling.reservation_block_bytes = 4 << 20;
        metal.compilation.allocations = crate::profile::CostExpr::Const(4 << 20);
        let (vulkan_claim, _) =
            compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        let (metal_claim, _) = compose(&plan(), &binding(4), &metal, 1).unwrap();
        assert_ne!(vulkan_claim.total_peak_bytes, metal_claim.total_peak_bytes);
        assert_ne!(vulkan_claim.profile_digest, metal_claim.profile_digest);
    }

    /// The claim scales with the selected configuration, which is what makes selection meaningful.
    #[test]
    fn a_larger_configuration_composes_to_a_larger_claim() {
        let small = compose(&plan(), &binding(1), &profile(BackendClass::Vulkan), 1)
            .unwrap()
            .0;
        let large = compose(&plan(), &binding(8), &profile(BackendClass::Vulkan), 1)
            .unwrap()
            .0;
        assert!(large.total_peak_bytes > small.total_peak_bytes);
    }

    /// The two enforcement classes are recorded distinctly and account for the whole total.
    #[test]
    fn the_enforceable_and_profiled_classes_are_recorded_distinctly() {
        let (claim, _) = compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        assert!(claim.directly_enforceable_bytes > 0);
        assert!(claim.profiled_and_measured_bytes > 0);
        assert_eq!(
            claim.directly_enforceable_bytes + claim.profiled_and_measured_bytes,
            claim.total_peak_bytes,
            "every byte of the claim belongs to exactly one enforcement class"
        );
    }

    /// The linear-memory cap comes from the guest's own host-memory terms, never from the device
    /// side and never from a host constant.
    #[test]
    fn the_linear_memory_cap_comes_from_the_plans_own_terms() {
        let (claim, _) = compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        let footprint = plan().footprint(&binding(4)).unwrap();
        assert_eq!(claim.linear_memory_bytes, footprint.linear_peak_bytes);
        assert_eq!(claim.linear_memory_bytes, 32_768 + 4096);
    }

    /// A plan needing an operation family the backend does not have is a typed refusal, never a
    /// fallback to an estimate.
    #[test]
    fn an_unsupported_operation_family_is_a_typed_refusal() {
        let mut p = plan();
        p.operations[0].family = "flash_attention".into();
        let err = compose(&p, &binding(1), &profile(BackendClass::Vulkan), 1).unwrap_err();
        assert!(err.to_string().contains("does not support"));
    }

    #[test]
    fn admission_evaluates_the_composed_claim_against_supply() {
        let (claim, _) = compose(&plan(), &binding(4), &profile(BackendClass::Vulkan), 1).unwrap();
        let r = report(BackendClass::Vulkan);
        validate_against(&claim, &r, None).expect("fits within measured supply");

        // An owner cap below the claim refuses, and the refusal says it is a POLICY refusal rather
        // than a hardware one — the device has the memory; its owner has chosen not to lend it.
        let err = validate_against(
            &claim,
            &r,
            Some(crate::capability::OwnerDeviceCap { max_bytes: 1024 }),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                PlannerError::NoAdmissibleConfiguration {
                    available: 1024,
                    ..
                }
            ),
            "the cap is the binding limit: {err}"
        );
        // And the typed comparison underneath says WHICH refused, so an operator whose own policy is
        // the constraint is not told their hardware is too small.
        let typed = crate::capability::admit_device_bytes(
            claim.total_peak_bytes,
            &r,
            Some(crate::capability::OwnerDeviceCap { max_bytes: 1024 }),
        )
        .unwrap_err();
        assert!(
            matches!(
                typed,
                crate::capability::DeviceAdmissionRefusal::ExceedsOwnerCap { .. }
            ),
            "a cap refusal is typed as a cap refusal: {typed}"
        );

        // And supply itself refuses on its own terms when the device genuinely lacks the memory.
        let mut small = r.clone();
        small.device_supply =
            crate::revision::Maybe::Available(crate::capability::fixtures::derived_supply(1024));
        small.measured_max_allocation =
            crate::revision::Maybe::Available(crate::capability::fixtures::measured_ceiling(1024));
        let supply_typed =
            crate::capability::admit_device_bytes(claim.total_peak_bytes, &small, None)
                .unwrap_err();
        assert!(
            matches!(
                supply_typed,
                crate::capability::DeviceAdmissionRefusal::ExceedsSupply { .. }
            ),
            "a supply refusal is typed as a supply refusal, distinct from a cap refusal: \
             {supply_typed}"
        );
    }

    #[test]
    fn a_claim_above_the_devices_measured_allocation_limit_is_refused() {
        let (mut claim, _) =
            compose(&plan(), &binding(1), &profile(BackendClass::Vulkan), 1).unwrap();
        claim.max_individual_allocation_bytes = 8 << 30;
        let err = validate_against(&claim, &report(BackendClass::Vulkan), None).unwrap_err();
        assert!(matches!(
            err,
            PlannerError::AllocationCeilingExceeded { .. }
        ));
    }

    /// Selection raises the configuration as far as the machine admits, deterministically, and
    /// delivers logical values only.
    #[test]
    fn selection_is_deterministic_and_delivers_logical_values_only() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let r = report(BackendClass::Vulkan);
        let first = select(
            &p,
            &prof,
            &r,
            None,
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::LargestAdmissible,
        )
        .unwrap();
        let second = select(
            &p,
            &prof,
            &r,
            None,
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::LargestAdmissible,
        )
        .unwrap();
        assert_eq!(first.binding, second.binding);
        assert_eq!(first.grant, second.grant);
        assert_eq!(
            first.binding.get("micro_batch"),
            Some(&DimensionValue::Uint(8)),
            "an ample machine takes the largest admissible configuration"
        );

        // The grant carries the selected dimension and nothing else — no backend identity, no
        // memory figure, no profile content.
        assert_eq!(first.grant.values.len(), 1);
        assert!(first.grant.values.contains_key("micro_batch"));
        assert_eq!(
            first.claim.execution_grant_hash,
            first.grant.grant_hash().unwrap()
        );
    }

    /// The lane's sanity bounds are keyed by the class the profile prices, and applied to the composed
    /// claim rather than to anything the guest declared.
    #[test]
    fn lane_claim_bounds_apply_per_backend_class_to_the_composed_claim() {
        let p = plan();
        let vulkan = profile(BackendClass::Vulkan);
        let claim = compose(&p, &binding(1), &vulkan, 1).unwrap().0;
        let total = claim.total_peak_bytes;

        let mut bounds = LaneClaimBounds::default();
        bounds
            .by_backend_class
            .insert("vulkan".to_string(), [0, total]);
        check_claim_against_lane(&claim, &vulkan, "trainer", &bounds)
            .expect("a claim at the bound is inside it");

        // A hair over the lane's ceiling refuses as a LANE violation, naming the class whose bounds
        // were applied — not as a machine that is too small.
        bounds
            .by_backend_class
            .insert("vulkan".to_string(), [0, total - 1]);
        let refusal = check_claim_against_lane(&claim, &vulkan, "trainer", &bounds).unwrap_err();
        assert!(matches!(
            refusal,
            PlannerError::PhysicalClaimExceedsLane {
                backend_class: "vulkan",
                ..
            }
        ));

        // Below the floor is equally a lane violation: a claim far under what the lane hosts is a
        // composition that went wrong, not a cheap role.
        bounds
            .by_backend_class
            .insert("vulkan".to_string(), [total + 1, total * 2]);
        assert!(matches!(
            check_claim_against_lane(&claim, &vulkan, "trainer", &bounds).unwrap_err(),
            PlannerError::PhysicalClaimExceedsLane { .. }
        ));
    }

    /// A lane with no bounds for the priced class refuses rather than admitting unbounded.
    ///
    /// The bounds are per class precisely because prices differ per backend, so a lane configured for one
    /// backend has said nothing about another — and silence is not permission.
    #[test]
    fn a_lane_without_bounds_for_the_priced_class_refuses() {
        let p = plan();
        let metal = profile(BackendClass::Metal);
        let claim = compose(&p, &binding(1), &metal, 1).unwrap().0;
        let mut bounds = LaneClaimBounds::default();
        bounds
            .by_backend_class
            .insert("vulkan".to_string(), [0, u64::MAX]);

        let refusal = check_claim_against_lane(&claim, &metal, "trainer", &bounds).unwrap_err();
        assert!(matches!(
            refusal,
            PlannerError::LaneStatesNoBoundsForClass {
                backend_class: "metal",
                ..
            }
        ));
        assert!(refusal.to_string().contains("does not admit"));
    }

    /// A tighter machine selects a smaller configuration rather than refusing outright.
    #[test]
    fn a_tighter_machine_selects_a_smaller_configuration() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let r = report(BackendClass::Vulkan);
        let ample = select(
            &p,
            &prof,
            &r,
            None,
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::LargestAdmissible,
        )
        .unwrap();
        let floor = compose(&p, &binding(1), &prof, 1).unwrap().0;
        let tight = select(
            &p,
            &prof,
            &r,
            Some(crate::capability::OwnerDeviceCap {
                max_bytes: floor.total_peak_bytes,
            }),
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::LargestAdmissible,
        )
        .unwrap();
        let chosen = |selection: &Selection| match selection.binding.get("micro_batch") {
            Some(DimensionValue::Uint(n)) => *n,
            other => panic!("expected a numeric selection, got {other:?}"),
        };
        assert!(
            chosen(&tight) < chosen(&ample),
            "a tighter budget selects a smaller configuration"
        );
    }

    /// When even the smallest configuration does not fit, the refusal is the model working as
    /// designed and it names the numbers.
    #[test]
    fn a_machine_that_cannot_host_the_floor_is_refused_with_both_numbers() {
        let err = select(
            &plan(),
            &profile(BackendClass::Vulkan),
            &report(BackendClass::Vulkan),
            Some(crate::capability::OwnerDeviceCap { max_bytes: 1024 }),
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::LargestAdmissible,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PlannerError::NoAdmissibleConfiguration { .. }
        ));
        assert!(err.to_string().contains("1024"));
    }

    /// A participant with a frozen uniform grant verifies it; it does not reselect.
    #[test]
    fn a_fixed_selection_verifies_rather_than_reselects() {
        let selection = select(
            &plan(),
            &profile(BackendClass::Vulkan),
            &report(BackendClass::Vulkan),
            None,
            1,
            SelectionScope::UniformRun,
            &SelectionPolicy::Fixed(binding(3)),
        )
        .unwrap();
        assert_eq!(
            selection.binding.get("micro_batch"),
            Some(&DimensionValue::Uint(3))
        );
    }

    /// A per-allocation term aggregates by maximum; a shared process/device term is charged once,
    /// however many roles share it.
    #[test]
    fn the_aggregate_composes_at_each_terms_real_scope() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim_a, occ_a) = compose(&p, &binding(2), &prof, 2).unwrap();
        let (claim_b, occ_b) = compose(&p, &binding(2), &prof, 2).unwrap();

        let one = aggregate(&[("trainer".into(), claim_a.clone(), occ_a.clone())]).unwrap();
        let two = aggregate(&[
            ("trainer".into(), claim_a, occ_a),
            ("seat".into(), claim_b, occ_b),
        ])
        .unwrap();

        assert_eq!(two.roles, vec!["trainer".to_string(), "seat".to_string()]);
        // The shared device-scoped context is charged ONCE across both roles, so two roles cost
        // strictly less than twice one role.
        assert!(two.occupancy_bytes < one.occupancy_bytes * 2);
        assert!(two.occupancy_bytes > one.occupancy_bytes);
        // The per-allocation maximum is a maximum, not a sum.
        assert_eq!(
            two.max_individual_allocation_bytes,
            one.max_individual_allocation_bytes
        );
        assert_eq!(
            two.directly_enforceable_bytes + two.profiled_and_measured_bytes,
            two.occupancy_bytes
        );
    }

    /// Two profiles giving one aggregation key incompatible scopes is a refusal, not a choice
    /// between them.
    #[test]
    fn an_incompatible_shared_term_is_refused_rather_than_resolved() {
        let p = plan();
        let (claim, occ) = compose(&p, &binding(1), &profile(BackendClass::Vulkan), 2).unwrap();
        let mut clashing = occ.clone();
        for term in &mut clashing.terms {
            if term.scope == AllocationScope::PerDevice {
                term.scope = AllocationScope::PerProcess;
            }
        }
        let err = aggregate(&[
            ("a".into(), claim.clone(), occ),
            ("b".into(), claim, clashing),
        ])
        .unwrap_err();
        assert!(matches!(err, PlannerError::IncompatibleSharedTerm { .. }));
    }

    /// The canonical vectors are what binds a cross-language consumer, so they must reproduce
    /// exactly through this planner.
    #[test]
    fn canonical_vectors_reproduce_their_recorded_claims() {
        let vector = vectors::derive(
            "gemm-microbatch-4",
            &plan(),
            &binding(4),
            &profile(BackendClass::Vulkan),
            1,
        )
        .unwrap();
        vectors::check(&vector).expect("the vector reproduces");

        // A vector from a different planner version is refused rather than compared: a planner
        // change invalidates prior composition evidence.
        let mut stale = vector.clone();
        stale.planner_version = PLANNER_VERSION + 1;
        assert!(vectors::check(&stale)
            .unwrap_err()
            .to_string()
            .contains("invalidates prior composition evidence"));

        // A tampered expectation fails.
        let mut tampered = vector;
        tampered.expected_claim_bytes.push(0);
        assert!(vectors::check(&tampered)
            .unwrap_err()
            .to_string()
            .contains("did not reproduce"));
    }

    /// Every divergence authority is nameable, so a breach records who to send.
    #[test]
    fn the_five_divergence_authorities_are_all_expressible() {
        let all = [
            DivergenceAuthority::LogicalResourcePlan,
            DivergenceAuthority::BackendExecutionProfile,
            DivergenceAuthority::PlannerOrSelector,
            DivergenceAuthority::CapabilityProbe,
            DivergenceAuthority::ResourceGovernor,
        ];
        let unique: BTreeSet<_> = all.iter().collect();
        assert_eq!(unique.len(), 5);
    }
}
