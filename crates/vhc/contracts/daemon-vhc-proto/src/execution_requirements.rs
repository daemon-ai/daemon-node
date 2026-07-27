// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **role execution-requirement structure** — what a frozen envelope says a role requires
//! (`docs/specs/vhc-architecture-spec.md` §5.4 `[DI-6]`).
//!
//! The envelope carries **no physical figure at all**. No device-memory floor, no per-allocation
//! ceiling, no measured profile. A physical figure in an envelope is a claim about a machine the
//! envelope has never seen, and it would stand as a second authority beside the composed claim.
//!
//! What the envelope carries instead is the role's *requirement*, stated logically: the canonical
//! Logical Resource Plan (or its bytes and digest), the backend classes the run allows, the
//! profile-certification requirements it will accept, the hardware-independent minima, and the
//! selection scope with either its frozen uniform Execution Grant or its normalization contract.
//! The **target profiles** state what the intended machines provide, physically; they are an
//! authoring input and certification evidence, never envelope content. The planner is what relates
//! the two.
//!
//! Embedding the plan is what permits **planning before download**: a node composes and pre-screens
//! from the envelope alone. It does not make the envelope the source of truth — the guest remains
//! the source, and at participation the module MUST reproduce the byte-identical plan from the
//! admitted configuration and Capability Grants. A mismatch is `ResourcePlanInconsistent`, which is
//! its own refusal and deliberately not the legacy `ClaimInconsistent`: those name different
//! objects and equating them would silently reinterpret old evidence.

use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;

use crate::bytes::{Hash, PeerId};
use crate::error::VhcProtoError;
use crate::execution_grant::ExecutionGrant;
use crate::hash::blake3_hash;
use crate::resource_plan::{LogicalResourcePlan, SelectionScope};

/// Whether a role needs an accelerator at all. This is a statement about the *role*, not about any
/// machine: the seat role performs no accelerator computation and says so, which is what lets it
/// claim zero duty beside a training role on the same node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorRequirement {
    /// The role must not be placed on an accelerator.
    Forbidden,
    /// The role runs with or without one.
    #[default]
    Optional,
    /// The role requires one.
    Required,
}

/// The run's hardware-independent minima.
///
/// Deliberately small. Every memory and allocation figure that used to live in the per-role device
/// section is **absent by design**: those are members of the composed Physical Claim, produced by
/// the planner from this role's plan and the participant's certified profile. What remains are
/// facts about the role and the network rather than about a device's capacity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareIndependentMinima {
    /// Whether the role needs an accelerator.
    pub accelerator: AcceleratorRequirement,
    /// Minimum sustained uplink, bits/s. A link rate is a property of the participant's
    /// connectivity, not of its device memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_uplink_bps: Option<u64>,
    /// Minimum sustained downlink, bits/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_downlink_bps: Option<u64>,
}

/// What the run will accept as authority for a Backend Execution Profile (`[PC-12]`).
///
/// Content addressing proves **identity**, not **authority**: a digest says which bytes these are,
/// not that they may be trusted to price a machine. Admission authenticates a profile against
/// these requirements *intersected with* the owner's own policy, before composing with it. Neither
/// policy may broaden the other, and a refusal names which one rejected.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileCertificationRequirements {
    /// The signing/release authorities this run accepts. Empty means the run defers entirely to
    /// owner policy — it never means "any authority".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted_authorities: Vec<PeerId>,
    /// The **development** authorities this run accepts, a separate and deliberately awkward
    /// decision from accepting a release authority (`[PC-12]`).
    ///
    /// Deferral-on-silence applies to release authorities only: for a development authority,
    /// silence is never consent, so acceptance requires the owner's policy **and** this run-side
    /// set to each name the authority explicitly. A development-authenticated profile satisfies
    /// integration evidence and never ceremony certification — the class fence lives in the
    /// authentication result, not in this list's good intentions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted_development_authorities: Vec<PeerId>,
    /// The lowest profile schema version this run accepts.
    #[serde(default)]
    pub min_profile_schema: u32,
    /// The planner versions this run accepts a profile to have been priced for. A profile priced
    /// for one composition algorithm is not valid under another.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted_planner_versions: Vec<u32>,
    /// Whether the profile must carry a conformance-evidence digest (`[PC-10]`).
    #[serde(default)]
    pub require_conformance_evidence: bool,
}

/// How the role's bounded logical choices are resolved, and the artifact that freezes them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionRequirement {
    /// One configuration, chosen at authoring, that every intended participant satisfies. The
    /// canonical Execution Grant bytes and digest are frozen **inside the signed role entry**, and
    /// every participant verifies and consumes those exact bytes — a locally reselected value
    /// would make participants disagree about a choice the run committed to.
    UniformRun {
        /// The canonical Execution Grant bytes.
        execution_grant: Vec<u8>,
        /// blake3 of `execution_grant`.
        execution_grant_hash: Hash,
    },
    /// Each admitting host selects locally, inside the frozen choice set, under the module's
    /// declared normalization/equivalence contract. Legal only when the module's own contract
    /// makes heterogeneous choices semantically valid, and the selected values must be made
    /// peer-visible under the module protocol's authentication rules — a local admitted tuple no
    /// peer can inspect is insufficient.
    PerParticipant {
        /// blake3 of the module's canonical normalization/equivalence contract.
        equivalence_contract_hash: Hash,
    },
}

/// What a role requires in order to execute, as carried by the signed envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleExecutionRequirements {
    /// The canonical Logical Resource Plan bytes, as produced by the module's own assessment path.
    pub logical_resource_plan: Vec<u8>,
    /// blake3 of `logical_resource_plan` — the admitted tuple's `logical_resource_plan_hash`.
    pub logical_resource_plan_hash: Hash,
    /// The backend classes this run admits for the role. Derived from the intended target set;
    /// this is the one thing that legitimately derives from what the target machines are.
    pub allowed_backend_classes: Vec<String>,
    /// What the run accepts as profile authority.
    pub profile_certification: ProfileCertificationRequirements,
    /// The hardware-independent minima.
    pub minima: HardwareIndependentMinima,
    /// How the plan's bounded choices are resolved.
    pub selection: SelectionRequirement,
}

impl RoleExecutionRequirements {
    /// Build the structure for a role from its plan and a resolved selection, deriving both the
    /// plan digest and (for a uniform run) the grant bytes and digest rather than accepting them.
    /// A value that can be derived and is instead supplied is a contradiction waiting for an
    /// author to make it (`[DI-4]`).
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] if the plan or the grant does not check out.
    pub fn derive(
        plan: &LogicalResourcePlan,
        allowed_backend_classes: Vec<String>,
        profile_certification: ProfileCertificationRequirements,
        minima: HardwareIndependentMinima,
        uniform_grant: Option<&ExecutionGrant>,
    ) -> Result<Self, VhcProtoError> {
        plan.validate()?;
        let logical_resource_plan = plan.to_canonical_bytes()?;
        let logical_resource_plan_hash = blake3_hash(&logical_resource_plan);

        let selection =
            match (plan.selection_scope, uniform_grant) {
                (SelectionScope::UniformRun, Some(grant)) => {
                    grant.bind_to(plan)?;
                    let bytes = grant.to_canonical_bytes()?;
                    SelectionRequirement::UniformRun {
                        execution_grant_hash: blake3_hash(&bytes),
                        execution_grant: bytes,
                    }
                }
                (SelectionScope::UniformRun, None) => return Err(VhcProtoError::Validation(
                    "a uniform-run plan must freeze one Execution Grant in its signed role entry \
                     ([RC-11])"
                        .into(),
                )),
                (SelectionScope::PerParticipant, _) => {
                    let Some(contract) = plan.equivalence_contract_hash else {
                        return Err(VhcProtoError::Validation(
                            "a per-participant plan must declare its normalization/equivalence \
                         contract ([RC-11])"
                                .into(),
                        ));
                    };
                    SelectionRequirement::PerParticipant {
                        equivalence_contract_hash: contract,
                    }
                }
            };

        Ok(Self {
            logical_resource_plan,
            logical_resource_plan_hash,
            allowed_backend_classes,
            profile_certification,
            minima,
            selection,
        })
    }

    /// A role structure over the **canonical trivial plan**, for negative and conformance fixtures.
    ///
    /// **This is not a production authoring path and must never become one.** A production role's
    /// structure is derived from the plan that role's own module emitted through its assessment
    /// export; the owner withdrew the compute-free carve-out precisely so no second source for a
    /// derived value exists. What this serves is `[DI-2]`'s fixture exemption: the upgrade
    /// transaction, the switch-module path, the join refusals and the worker protocol all need a
    /// *runnable-shaped* envelope to exercise behaviour that has nothing to do with resources, and
    /// before this member existed they got one for free.
    ///
    /// It uses [`LogicalResourcePlan::trivial`] — the same shared construction every compute-free
    /// module emits — rather than spelling a plan out again, so even the fixture path cannot drift
    /// from the format.
    ///
    /// The `fixture_` prefix is load-bearing: it is what a scan can key on to prove no production
    /// path calls this.
    #[must_use]
    pub fn fixture_over_trivial_plan(allowed_backend_classes: Vec<String>) -> Self {
        let plan = LogicalResourcePlan::trivial(crate::WASM_GUEST_LINEAR_FLOOR_BYTES);
        let grant = ExecutionGrant {
            logical_resource_plan_hash: plan.plan_hash().expect("trivial plan hashes"),
            scope: SelectionScope::UniformRun,
            values: std::collections::BTreeMap::new(),
        };
        Self::derive(
            &plan,
            allowed_backend_classes,
            ProfileCertificationRequirements::default(),
            HardwareIndependentMinima::default(),
            Some(&grant),
        )
        .expect("the canonical trivial plan and its empty grant are valid by construction")
    }

    /// Decode and validate the embedded plan, checking the digest and the selection coupling.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] if the plan is malformed, its digest disagrees, or the frozen
    /// selection does not match the plan's declared scope.
    pub fn decode_plan(&self) -> Result<LogicalResourcePlan, VhcProtoError> {
        if blake3_hash(&self.logical_resource_plan) != self.logical_resource_plan_hash {
            return Err(VhcProtoError::Validation(
                "role execution requirements: the embedded plan does not match its digest".into(),
            ));
        }
        let plan = LogicalResourcePlan::decode_canonical(&self.logical_resource_plan)?;
        match (&self.selection, plan.selection_scope) {
            (SelectionRequirement::UniformRun { .. }, SelectionScope::UniformRun)
            | (SelectionRequirement::PerParticipant { .. }, SelectionScope::PerParticipant) => {}
            // A per-participant plan may legitimately be narrowed to a uniform run by the
            // envelope; the reverse would broaden what the module's semantics permit.
            (SelectionRequirement::UniformRun { .. }, SelectionScope::PerParticipant) => {}
            (SelectionRequirement::PerParticipant { .. }, SelectionScope::UniformRun) => {
                return Err(VhcProtoError::Validation(
                    "the envelope selects per-participant for a plan that permits only a uniform \
                     run; an envelope may narrow a scope, never broaden it ([RC-11])"
                        .into(),
                ))
            }
        }
        Ok(plan)
    }

    /// The frozen uniform Execution Grant, decoded and checked against its digest and its plan.
    /// Every participant of a `UniformRun` role calls this and consumes exactly these bytes.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] if the selection is per-participant, or the grant is
    /// substituted, non-canonical, or does not bind to the embedded plan.
    pub fn frozen_grant(&self) -> Result<ExecutionGrant, VhcProtoError> {
        let SelectionRequirement::UniformRun {
            execution_grant,
            execution_grant_hash,
        } = &self.selection
        else {
            return Err(VhcProtoError::Validation(
                "this role selects per participant; there is no frozen grant to consume".into(),
            ));
        };
        if blake3_hash(execution_grant) != *execution_grant_hash {
            return Err(VhcProtoError::Validation(
                "the frozen Execution Grant does not match the digest in the signed role entry"
                    .into(),
            ));
        }
        let grant = ExecutionGrant::decode_canonical(execution_grant)?;
        let plan = self.decode_plan()?;
        grant.bind_to(&plan)?;
        Ok(grant)
    }

    /// Envelope-internal validation: the plan resolves, the digests hold, the selection matches,
    /// and at least one backend class is allowed.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] naming the first failure.
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.allowed_backend_classes.is_empty() {
            return Err(VhcProtoError::Validation(
                "a role must allow at least one backend class; an empty list admits nothing and \
                 says so nowhere"
                    .into(),
            ));
        }
        self.decode_plan()?;
        if matches!(self.selection, SelectionRequirement::UniformRun { .. }) {
            self.frozen_grant()?;
        }
        Ok(())
    }
}

/// A Logical Resource Plan **together with where it came from**.
///
/// The derivation invariant is that the module's own assessment output is the only source of a
/// plan — an authoring seat that composed one itself would be stating a second, unverifiable
/// opinion about the algorithm's needs, and the two would diverge silently. The plan is a plain
/// struct with public fields, so nothing stops code from building one; what this wrapper does is
/// make the *provenance* travel with it, so a seat can require a plan that came from a module and
/// a fixture's plan cannot be mistaken for one.
///
/// The two constructors are the whole surface, and which code may call them is a scanned rule
/// rather than a naming convention: `from_module_assessment` belongs to the assessment path alone,
/// and `fixture` to test targets alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleDerivedPlan {
    plan: LogicalResourcePlan,
    module_derived: bool,
}

impl ModuleDerivedPlan {
    /// Wrap the plan a module's own assessment export produced.
    ///
    /// **The assessment path is the only caller.** Everything else takes one of these as an input;
    /// see the type's own note on why that is a scanned rule.
    #[must_use]
    pub fn from_module_assessment(plan: LogicalResourcePlan) -> Self {
        Self {
            plan,
            module_derived: true,
        }
    }

    /// Wrap a plan that no module produced, for a fixture that has no module to ask.
    ///
    /// Named so it cannot be mistaken for the real thing at a call site, and reported honestly by
    /// [`Self::is_module_derived`] rather than passing as an assessment result.
    #[must_use]
    pub fn fixture(plan: LogicalResourcePlan) -> Self {
        Self {
            plan,
            module_derived: false,
        }
    }

    /// The plan itself.
    #[must_use]
    pub fn plan(&self) -> &LogicalResourcePlan {
        &self.plan
    }

    /// Whether a module's assessment produced this plan.
    ///
    /// A gate that requires real provenance checks this rather than trusting the call site.
    #[must_use]
    pub fn is_module_derived(&self) -> bool {
        self.module_derived
    }
}

/// The per-role execution requirements an authoring seat **places into** a genesis envelope.
///
/// Authoring seats do not construct execution requirements. They are handed this and place what is
/// in it, which is what makes the module the single source: a seat has no way to state a resource
/// requirement of its own, because the only thing it can do with this type is look a role up in it.
///
/// A role that is missing from here gets no requirement structure, and `validate` refuses a runnable
/// envelope carrying none — so an incomplete derivation fails closed and loudly at the existing gate
/// rather than needing a new one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthoredExecution {
    per_role: BTreeMap<String, RoleExecutionRequirements>,
}

impl AuthoredExecution {
    /// Nothing derived yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Derive one role's requirements from the plan its module produced, and add them.
    ///
    /// This is the only way requirements enter, and it is [`RoleExecutionRequirements::derive`]
    /// underneath — so the plan is validated, its digest computed here rather than accepted, and a
    /// uniform-run plan's Execution Grant frozen into the signed role entry as a by-product of
    /// authoring. Nothing about grant freezing is optional or separately invocable.
    ///
    /// # Errors
    /// [`VhcProtoError::Validation`] when the plan is invalid, when a uniform-run plan arrives
    /// without the grant it must freeze, when a per-participant plan declares no equivalence
    /// contract, or when the grant does not bind to the plan.
    pub fn derive(
        mut self,
        role: &str,
        plan: &ModuleDerivedPlan,
        allowed_backend_classes: Vec<String>,
        profile_certification: ProfileCertificationRequirements,
        minima: HardwareIndependentMinima,
        uniform_grant: Option<&ExecutionGrant>,
    ) -> Result<Self, VhcProtoError> {
        let requirements = RoleExecutionRequirements::derive(
            plan.plan(),
            allowed_backend_classes,
            profile_certification,
            minima,
            uniform_grant,
        )?;
        self.per_role.insert(role.to_string(), requirements);
        Ok(self)
    }

    /// One role's requirements, or `None` when nothing was derived for it.
    ///
    /// `None` is placed as-is: `validate` refuses a runnable envelope whose role carries no
    /// requirement structure, so a role the derivation missed is refused at the gate that already
    /// exists instead of being papered over with a default here. A default would be this seat
    /// stating a resource requirement, which is the whole thing the type exists to prevent.
    #[must_use]
    pub fn for_role(&self, role: &str) -> Option<RoleExecutionRequirements> {
        self.per_role.get(role).cloned()
    }

    /// The roles requirements were derived for, in canonical order.
    pub fn roles(&self) -> impl Iterator<Item = &str> {
        self.per_role.keys().map(String::as_str)
    }

    /// Whether every one of `roles` had requirements derived for it.
    ///
    /// For a caller that would rather refuse before authoring than author an envelope its own
    /// `validate` will reject.
    #[must_use]
    pub fn covers(&self, roles: &[&str]) -> bool {
        roles.iter().all(|role| self.per_role.contains_key(*role))
    }
}

/// A minimal but real execution-requirement structure for in-crate tests that need a *runnable*
/// role entry: a one-dimension plan and the uniform grant that resolves it.
#[cfg(test)]
pub(crate) fn sample_for_tests(backend_class: &str) -> RoleExecutionRequirements {
    use crate::execution_grant::GrantValue;
    use crate::resource_plan::{Dimension, Domain, Dtype, Expr, Lifetime, Retention, TensorDecl};

    let plan = LogicalResourcePlan {
        selection_scope: SelectionScope::UniformRun,
        equivalence_contract_hash: None,
        dimensions: vec![Dimension {
            name: "micro_batch".into(),
            domain: Domain::UintRange { lo: 1, hi: 4 },
        }],
        tensors: vec![TensorDecl {
            name: "params".into(),
            shape: vec![Expr::Const(1024)],
            dtype: Dtype::F32,
            layout: vec![],
            lifetime: Lifetime::Persistent(Retention::Run),
        }],
        operations: vec![],
        transfers: vec![],
        linear_memory: vec![],
        transient_live_sets: vec![],
        linear_fragmentation_headroom: Expr::Const(0),
    };
    let grant = ExecutionGrant {
        logical_resource_plan_hash: plan.plan_hash().expect("plan hash"),
        scope: SelectionScope::UniformRun,
        values: std::collections::BTreeMap::from([(
            "micro_batch".to_string(),
            GrantValue::Uint(2),
        )]),
    };
    RoleExecutionRequirements::derive(
        &plan,
        vec![backend_class.to_string()],
        ProfileCertificationRequirements {
            min_profile_schema: 1,
            accepted_planner_versions: vec![1],
            require_conformance_evidence: true,
            ..Default::default()
        },
        HardwareIndependentMinima {
            accelerator: AcceleratorRequirement::Required,
            ..Default::default()
        },
        Some(&grant),
    )
    .expect("execution requirements")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_grant::GrantValue;
    use crate::resource_plan::{
        Dimension, Domain, Dtype, Expr, Lifetime, LogicalResourcePlan, Retention, TensorDecl,
    };
    use std::collections::BTreeMap;

    fn plan(scope: SelectionScope) -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: scope,
            equivalence_contract_hash: match scope {
                SelectionScope::UniformRun => None,
                SelectionScope::PerParticipant => Some(Hash([3u8; 32])),
            },
            dimensions: vec![Dimension {
                name: "micro_batch".into(),
                domain: Domain::UintRange { lo: 1, hi: 8 },
            }],
            tensors: vec![TensorDecl {
                name: "params".into(),
                shape: vec![Expr::Const(1024)],
                dtype: Dtype::F32,
                layout: vec![],
                lifetime: Lifetime::Persistent(Retention::Run),
            }],
            operations: vec![],
            transfers: vec![],
            linear_memory: vec![],
            transient_live_sets: vec![],
            linear_fragmentation_headroom: Expr::Const(0),
        }
    }

    fn grant(plan: &LogicalResourcePlan, micro_batch: u64) -> ExecutionGrant {
        ExecutionGrant {
            logical_resource_plan_hash: plan.plan_hash().unwrap(),
            scope: SelectionScope::UniformRun,
            values: BTreeMap::from([("micro_batch".to_string(), GrantValue::Uint(micro_batch))]),
        }
    }

    fn requirements(
        plan: &LogicalResourcePlan,
        grant: Option<&ExecutionGrant>,
    ) -> RoleExecutionRequirements {
        RoleExecutionRequirements::derive(
            plan,
            vec!["vulkan".into()],
            ProfileCertificationRequirements {
                min_profile_schema: 1,
                accepted_planner_versions: vec![1],
                require_conformance_evidence: true,
                ..Default::default()
            },
            HardwareIndependentMinima {
                accelerator: AcceleratorRequirement::Required,
                ..Default::default()
            },
            grant,
        )
        .expect("derive")
    }

    #[test]
    fn a_uniform_run_role_freezes_the_grant_every_participant_consumes() {
        let plan = plan(SelectionScope::UniformRun);
        let grant = grant(&plan, 4);
        let reqs = requirements(&plan, Some(&grant));
        reqs.validate().unwrap();

        let recovered = reqs.frozen_grant().unwrap();
        assert_eq!(recovered, grant);
        assert_eq!(reqs.decode_plan().unwrap(), plan);
        // The digests are derived, not supplied.
        assert_eq!(reqs.logical_resource_plan_hash, plan.plan_hash().unwrap());
    }

    #[test]
    fn a_substituted_grant_is_refused_against_the_signed_digest() {
        let plan = plan(SelectionScope::UniformRun);
        let mut reqs = requirements(&plan, Some(&grant(&plan, 4)));
        let substitute = grant(&plan, 8).to_canonical_bytes().unwrap();
        let SelectionRequirement::UniformRun {
            execution_grant, ..
        } = &mut reqs.selection
        else {
            unreachable!()
        };
        *execution_grant = substitute;
        assert!(reqs
            .frozen_grant()
            .unwrap_err()
            .to_string()
            .contains("does not match the digest"));
    }

    #[test]
    fn a_uniform_run_plan_must_carry_a_frozen_grant() {
        let plan = plan(SelectionScope::UniformRun);
        let err = RoleExecutionRequirements::derive(
            &plan,
            vec!["vulkan".into()],
            ProfileCertificationRequirements::default(),
            HardwareIndependentMinima::default(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must freeze one Execution Grant"));
    }

    #[test]
    fn a_per_participant_role_carries_its_equivalence_contract() {
        let plan = plan(SelectionScope::PerParticipant);
        let reqs = requirements(&plan, None);
        reqs.validate().unwrap();
        assert_eq!(
            reqs.selection,
            SelectionRequirement::PerParticipant {
                equivalence_contract_hash: Hash([3u8; 32])
            }
        );
        assert!(reqs
            .frozen_grant()
            .unwrap_err()
            .to_string()
            .contains("no frozen grant"));
    }

    /// The envelope may narrow the plan's scope and must never broaden it.
    #[test]
    fn the_envelope_may_narrow_a_scope_but_never_broaden_one() {
        let permissive = plan(SelectionScope::PerParticipant);
        let mut narrowed = requirements(&permissive, None);
        narrowed.selection = SelectionRequirement::UniformRun {
            execution_grant: vec![],
            execution_grant_hash: blake3_hash(&[]),
        };
        // Narrowing a per-participant plan to a uniform run is legal at the scope level.
        assert!(narrowed.decode_plan().is_ok());

        let strict = plan(SelectionScope::UniformRun);
        let mut broadened = requirements(&strict, Some(&grant(&strict, 1)));
        broadened.selection = SelectionRequirement::PerParticipant {
            equivalence_contract_hash: Hash([7u8; 32]),
        };
        assert!(broadened
            .decode_plan()
            .unwrap_err()
            .to_string()
            .contains("never broaden"));
    }

    /// The structure carries no physical figure: there is nowhere in it to author a device-memory
    /// floor or a per-allocation ceiling, because both are members of the composed claim.
    #[test]
    fn the_structure_carries_no_physical_figure() {
        let plan = plan(SelectionScope::UniformRun);
        let reqs = requirements(&plan, Some(&grant(&plan, 2)));
        let json = format!("{reqs:?}").to_lowercase();
        for physical in [
            "vram",
            "device_memory",
            "max_alloc",
            "per_allocation",
            "ram_bytes",
            "disk_bytes",
        ] {
            assert!(
                !json.contains(physical),
                "the role structure must not carry `{physical}`"
            );
        }
    }

    #[test]
    fn an_empty_backend_class_list_is_refused() {
        let plan = plan(SelectionScope::UniformRun);
        let mut reqs = requirements(&plan, Some(&grant(&plan, 2)));
        reqs.allowed_backend_classes.clear();
        assert!(reqs
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least one backend class"));
    }
}

#[cfg(test)]
mod authoring_interface_tests {
    use super::*;
    use crate::resource_plan::{Dimension, Domain, Expr, LinearLifetime, LinearMemoryTerm};

    fn plan_with_dimensions() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![
                Dimension {
                    name: "micro_batch".into(),
                    domain: Domain::UintRange { lo: 2, hi: 16 },
                },
                Dimension {
                    name: "precision".into(),
                    domain: Domain::Enum(vec!["bf16".into(), "f32".into()]),
                },
            ],
            tensors: vec![],
            operations: vec![],
            transfers: vec![],
            linear_memory: vec![LinearMemoryTerm {
                name: "floor".into(),
                lifetime: LinearLifetime::Persistent,
                bytes: Expr::Const(1 << 20),
            }],
            transient_live_sets: vec![],
            linear_fragmentation_headroom: Expr::Const(0),
        }
    }

    /// A plan carries where it came from, so a fixture's plan cannot pass as a module's.
    ///
    /// This is the whole mechanism behind the derivation invariant at the authoring seats: the plan
    /// type has public fields and always will, so provenance has to travel beside it rather than be
    /// inferred from the shape of what arrived.
    #[test]
    fn a_plan_reports_whether_a_module_produced_it() {
        let assessed = ModuleDerivedPlan::from_module_assessment(plan_with_dimensions());
        assert!(assessed.is_module_derived());

        let fixture = ModuleDerivedPlan::fixture(plan_with_dimensions());
        assert!(
            !fixture.is_module_derived(),
            "a fixture's plan must not report itself as an assessment result"
        );
        // Same plan either way — the difference is only ever the provenance.
        assert_eq!(assessed.plan(), fixture.plan());
    }

    /// An authoring seat can only look a role up; it has no way to state a requirement.
    ///
    /// A role nobody derived comes back `None`, which the envelope's own `validate` refuses for a
    /// runnable role. Defaulting instead would be the seat inventing a resource requirement, which is
    /// the failure this type exists to prevent — so the absence is preserved all the way to the gate.
    #[test]
    fn an_unauthored_role_yields_nothing_rather_than_a_default() {
        let plan = ModuleDerivedPlan::from_module_assessment(plan_with_dimensions());
        let grant = ExecutionGrant::selecting_domain_minimum(plan.plan()).expect("floor grant");
        let authored = AuthoredExecution::new()
            .derive(
                "trainer",
                &plan,
                vec!["cuda".into()],
                ProfileCertificationRequirements::default(),
                HardwareIndependentMinima::default(),
                Some(&grant),
            )
            .expect("the trainer derives");

        assert!(authored.for_role("trainer").is_some());
        assert!(
            authored.for_role("coordinator").is_none(),
            "a role nobody derived must not acquire a default requirement here"
        );
        assert!(!authored.covers(&["trainer", "coordinator"]));
        assert!(authored.covers(&["trainer"]));
        assert_eq!(authored.roles().collect::<Vec<_>>(), vec!["trainer"]);
    }

    /// Deriving freezes the uniform-run grant as a by-product — there is no separate step to forget.
    ///
    /// The frozen bytes are canonical and the digest is computed here rather than accepted, so a
    /// caller cannot hand in a grant digest that names something other than the grant it shipped.
    #[test]
    fn deriving_a_uniform_run_role_freezes_its_grant_and_computes_the_digest_itself() {
        let plan = ModuleDerivedPlan::from_module_assessment(plan_with_dimensions());
        let grant = ExecutionGrant::selecting_domain_minimum(plan.plan()).expect("floor grant");
        let authored = AuthoredExecution::new()
            .derive(
                "trainer",
                &plan,
                vec!["cuda".into()],
                ProfileCertificationRequirements::default(),
                HardwareIndependentMinima::default(),
                Some(&grant),
            )
            .expect("derives");
        let requirements = authored.for_role("trainer").expect("present");

        let SelectionRequirement::UniformRun {
            execution_grant_hash,
            execution_grant,
        } = &requirements.selection
        else {
            panic!("a uniform-run plan must freeze a uniform-run selection");
        };
        assert_eq!(
            *execution_grant,
            grant.to_canonical_bytes().expect("canonical grant"),
            "the frozen bytes are the grant's own canonical encoding"
        );
        assert_eq!(
            *execution_grant_hash,
            crate::hash::blake3_hash(execution_grant)
        );
        assert_eq!(
            requirements.logical_resource_plan_hash,
            crate::hash::blake3_hash(&requirements.logical_resource_plan)
        );
    }

    /// A uniform-run plan cannot be authored without the grant it must freeze.
    #[test]
    fn a_uniform_run_role_refuses_to_derive_without_a_grant() {
        let plan = ModuleDerivedPlan::from_module_assessment(plan_with_dimensions());
        let refusal = AuthoredExecution::new()
            .derive(
                "trainer",
                &plan,
                vec!["cuda".into()],
                ProfileCertificationRequirements::default(),
                HardwareIndependentMinima::default(),
                None,
            )
            .expect_err("a uniform-run plan without its grant must refuse");
        assert!(
            format!("{refusal}").contains("must freeze one Execution Grant"),
            "the refusal says what is missing: {refusal}"
        );
    }

    /// The domain-minimum policy selects each dimension's smallest admissible value.
    ///
    /// Sound as a floor because every operator in the plan's grammar is monotone non-decreasing in
    /// its dimension arguments, so no other admissible selection costs less. It is a named choice,
    /// not a default: a run that means to train at a real batch size selects for itself.
    #[test]
    fn the_domain_minimum_policy_selects_the_floor_of_every_dimension() {
        let plan = plan_with_dimensions();
        let grant = ExecutionGrant::selecting_domain_minimum(&plan).expect("floor grant");

        assert_eq!(grant.scope, SelectionScope::UniformRun);
        assert_eq!(
            grant.values.get("micro_batch"),
            Some(&crate::execution_grant::GrantValue::Uint(2)),
            "the range's inclusive lower bound, not zero — zero is not admissible here"
        );
        assert_eq!(
            grant.values.get("precision"),
            Some(&crate::execution_grant::GrantValue::Text("bf16".into())),
            "the first spelling of a domain validated as sorted and unique"
        );
        // It names the plan it resolves, so it cannot be applied to a different one.
        assert_eq!(
            grant.logical_resource_plan_hash,
            plan.plan_hash().expect("plan hash")
        );
        grant
            .bind_to(&plan)
            .expect("the floor grant binds to its plan");
    }

    /// An empty enum domain has no smallest admissible value, and saying so beats picking one.
    #[test]
    fn an_empty_enum_domain_has_no_minimum_and_refuses() {
        let mut plan = plan_with_dimensions();
        plan.dimensions[1].domain = Domain::Enum(vec![]);
        assert!(ExecutionGrant::selecting_domain_minimum(&plan).is_err());
    }
}
