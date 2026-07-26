// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **Backend Execution Profile** — the host backend implementation's statement of what it
//! physically costs to execute a logical plan (`docs/specs/vhc-architecture-spec.md` §9.6
//! `[RC-4]`(2), `[RC-10]`; §9.7 `[PC-10]`).
//!
//! A Logical Resource Plan says what an algorithm needs, logically. A profile says what *this*
//! backend implementation, on *this* allocator, through *this* driver, actually allocates to
//! deliver it. The two are composed by the planner into a Physical Claim. Splitting them puts each
//! fact with the party that can establish it: the module knows its tensors and lifetimes; the
//! backend knows what it allocates to execute them; the machine knows what it has.
//!
//! ## Why this type is not in a guest-linked crate
//!
//! Most of a device peak is **not the module's**: pooling and retention, workspace, compilation
//! allocations and staging are properties of a backend implementation the module cannot see and
//! must not know. If the profile type lived in a crate the guests link, every profile revision
//! would change every guest hash — a driver update would re-pin and re-certify a training algorithm
//! that had not changed. That is the coupling the whole redesign exists to remove, and it would be
//! reintroduced by a crate-layout choice. Hence: host-side only, enforced by the dependency gate.
//!
//! ## Every cost term declares its scope
//!
//! A shared process pool is not multiplied by the role count; a per-role workspace is not silently
//! treated as shared. So each term carries an [`AllocationScope`], a stable [`CostTerm::aggregation_key`],
//! and the associative, deterministic [`CompositionRule`] by which simultaneous role claims compose
//! at that scope. A term whose sharing behavior is unknown takes the conservative non-sharing rule
//! until conformance evidence justifies another — the conservative direction over-reserves, and
//! over-reserving refuses a machine that would have worked, which is recoverable; under-reserving
//! admits a machine that then dies, which is not.
//!
//! ## The two enforcement classes are recorded distinctly
//!
//! A host cannot intercept every allocation: a compute framework, an allocator, a shader compiler
//! and a graphics driver all allocate below any boundary a host can stand on. Every term therefore
//! declares its [`EnforcementClass`], and a certification statement reports the two classes
//! separately. A certification that reports one enforcement property over both is making a claim
//! about the driver's internals that nobody verified.

use std::collections::{BTreeMap, BTreeSet};

use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

use crate::revision::{BackendClass, Maybe};

/// The profile schema this build authors and accepts.
pub const BACKEND_EXECUTION_PROFILE_SCHEMA: u32 = 1;

/// The scope at which a physical cost term is actually allocated (`[RC-4]`(2), `[RC-10]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationScope {
    /// A constraint on any **single** allocation, not an occupancy reservation. It aggregates by
    /// maximum across every candidate allocation and role, is checked against the capability
    /// report's largest-single-allocation limit, and is never summed. Bytes such an allocation
    /// contributes to occupancy are represented separately at role, process or device scope.
    PerAllocation,
    /// Allocated once per admitted role instance.
    PerRoleInstance,
    /// Allocated once per host process, however many role instances it holds.
    PerProcess,
    /// Allocated once per device, across every process using it.
    PerDevice,
}

impl AllocationScope {
    /// Whether this scope contributes to an occupancy reservation. [`AllocationScope::PerAllocation`]
    /// does not: it is a maximum constraint that admission validates, not bytes it reserves.
    pub fn is_occupancy(self) -> bool {
        !matches!(self, Self::PerAllocation)
    }
}

/// How simultaneous role claims combine at a term's scope. Associative and deterministic, so the
/// aggregate does not depend on the order roles happened to attach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionRule {
    /// Each consumer pays. This is the **conservative non-sharing rule**, and it is the required
    /// default for any term whose sharing behavior conformance evidence has not established.
    Sum,
    /// The largest single value governs. Correct for [`AllocationScope::PerAllocation`], and never
    /// a substitute for summing occupancy.
    Max,
    /// Charged once per aggregation key, however many consumers share it.
    OncePerKey,
}

/// Whether the governor can actually enforce a term, or only budget and observe it (`[RC-10]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementClass {
    /// The host or backend surface exposes the allocation at creation, so the governor intercepts
    /// it, attributes it to a role instance, and enforces the admitted claim against it.
    DirectlyEnforceable,
    /// The allocation happens below any boundary the host can stand on. It is **budgeted from the
    /// profile and observed in aggregate**, never individually enforced — and this says so rather
    /// than implying otherwise.
    ProfiledAndMeasured,
}

/// A logical quantity the planner supplies when it evaluates a profile formula. The formula
/// language is deliberately over *logical* inputs: a profile prices logical demand, it does not
/// re-derive it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostInput {
    /// The logical byte size of the object being priced.
    LogicalBytes,
    /// Its element count.
    ElementCount,
    /// The plan's declared maximum simultaneous in-flight count for the priced operation.
    InFlight,
    /// The plan's device-logical persistent floor.
    PersistentLogicalBytes,
    /// The plan's device-logical transient peak.
    TransientPeakLogicalBytes,
    /// The plan's largest single logical object.
    LargestLogicalObjectBytes,
    /// The declared transfer window being priced.
    TransferWindowBytes,
    /// The number of admitted co-resident role instances sharing this term's scope.
    CoResidentRoleCount,
}

/// A profile cost formula: a closed, bounded, deterministic expression over [`CostInput`]s.
///
/// Physical concerns the plan is forbidden to express live here and only here — alignment and
/// padding are the clear case: a plan's layout constraints add no physical padding, because
/// alignment belongs to the backend that imposes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostExpr {
    /// A literal byte count.
    Const(u64),
    /// A logical quantity supplied by the planner.
    Input(CostInput),
    /// Two or more terms, summed.
    Add(Vec<CostExpr>),
    /// A product of two terms.
    Mul(Box<CostExpr>, Box<CostExpr>),
    /// One or more terms, maximized.
    Max(Vec<CostExpr>),
    /// Ceiling division by a positive divisor.
    CeilDiv(Box<CostExpr>, u64),
    /// Round up to a physical alignment boundary.
    AlignUp(Box<CostExpr>, u64),
    /// Scale by an exact rational. Stated as a ratio rather than a float so two hosts composing the
    /// same plan and profile reach byte-identical claims; a float would make the claim
    /// platform-dependent in its last bits.
    Ratio {
        /// The term being scaled.
        term: Box<CostExpr>,
        /// Numerator.
        numerator: u64,
        /// Denominator, greater than zero.
        denominator: u64,
    },
}

/// A complete assignment of the logical inputs a formula may read.
pub type CostInputs = BTreeMap<CostInput, u64>;

/// Why a profile or a formula was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProfileError {
    /// The profile is not a well-formed schema-1 document, or its semantics do not check out.
    #[error("backend execution profile is invalid: {0}")]
    Invalid(String),
    /// A formula could not be evaluated.
    #[error("backend execution profile formula could not be evaluated: {0}")]
    Formula(String),
    /// A required measurement is missing, so the property that depends on it is unverifiable.
    #[error("backend execution profile is not certified for {property}: {detail}")]
    NotCertified {
        /// The property that cannot be certified.
        property: &'static str,
        /// Why.
        detail: String,
    },
}

impl CostExpr {
    /// Evaluate under a complete set of logical inputs, in checked `u64` arithmetic. Overflow is a
    /// refusal, not a wrap: a claim that silently wrapped would admit a machine on a figure smaller
    /// than the workload.
    pub fn evaluate(&self, inputs: &CostInputs) -> Result<u64, ProfileError> {
        let overflow =
            |what: &str| ProfileError::Formula(format!("checked u64 overflow in {what}"));
        match self {
            Self::Const(v) => Ok(*v),
            Self::Input(input) => inputs.get(input).copied().ok_or_else(|| {
                ProfileError::Formula(format!("no value supplied for input {input:?}"))
            }),
            Self::Add(terms) => {
                if terms.len() < 2 {
                    return Err(ProfileError::Formula("add takes two or more terms".into()));
                }
                terms.iter().try_fold(0u64, |acc, t| {
                    acc.checked_add(t.evaluate(inputs)?)
                        .ok_or_else(|| overflow("add"))
                })
            }
            Self::Mul(a, b) => a
                .evaluate(inputs)?
                .checked_mul(b.evaluate(inputs)?)
                .ok_or_else(|| overflow("mul")),
            Self::Max(terms) => {
                if terms.is_empty() {
                    return Err(ProfileError::Formula("max takes one or more terms".into()));
                }
                let mut best = 0u64;
                for term in terms {
                    best = best.max(term.evaluate(inputs)?);
                }
                Ok(best)
            }
            Self::CeilDiv(inner, divisor) => {
                if *divisor == 0 {
                    return Err(ProfileError::Formula("ceil-div by zero".into()));
                }
                Ok(inner.evaluate(inputs)?.div_ceil(*divisor))
            }
            Self::AlignUp(inner, alignment) => {
                if *alignment == 0 {
                    return Err(ProfileError::Formula("align-up to zero".into()));
                }
                let value = inner.evaluate(inputs)?;
                value
                    .div_ceil(*alignment)
                    .checked_mul(*alignment)
                    .ok_or_else(|| overflow("align-up"))
            }
            Self::Ratio {
                term,
                numerator,
                denominator,
            } => {
                if *denominator == 0 {
                    return Err(ProfileError::Formula(
                        "ratio with a zero denominator".into(),
                    ));
                }
                let value = term.evaluate(inputs)?;
                // Multiply before dividing, and round UP: a headroom factor that rounded down
                // would quietly shave the allowance it exists to provide.
                value
                    .checked_mul(*numerator)
                    .map(|scaled| scaled.div_ceil(*denominator))
                    .ok_or_else(|| overflow("ratio"))
            }
        }
    }

    /// Every input this formula reads.
    pub fn inputs(&self) -> BTreeSet<CostInput> {
        let mut out = BTreeSet::new();
        self.collect_inputs(&mut out);
        out
    }

    fn collect_inputs(&self, out: &mut BTreeSet<CostInput>) {
        match self {
            Self::Const(_) => {}
            Self::Input(i) => {
                out.insert(*i);
            }
            Self::Add(terms) | Self::Max(terms) => {
                for t in terms {
                    t.collect_inputs(out);
                }
            }
            Self::Mul(a, b) => {
                a.collect_inputs(out);
                b.collect_inputs(out);
            }
            Self::CeilDiv(inner, _) | Self::AlignUp(inner, _) => inner.collect_inputs(out),
            Self::Ratio { term, .. } => term.collect_inputs(out),
        }
    }
}

/// One physical cost term, with everything the aggregate needs to compose it correctly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostTerm {
    /// A stable, human-readable term name.
    pub name: String,
    /// The scope at which the allocation actually happens.
    pub scope: AllocationScope,
    /// The stable key shared terms match by. Profiles that give the same key incompatible scopes or
    /// composition rules are mutually incompatible, and admission refuses rather than choosing one.
    pub aggregation_key: String,
    /// How simultaneous role claims combine at this term's scope.
    pub composition_rule: CompositionRule,
    /// Whether the governor can enforce this term or only budget and observe it.
    pub enforcement: EnforcementClass,
    /// The formula.
    pub formula: CostExpr,
    /// Whether conformance evidence established this term's sharing behavior. When false the term
    /// MUST use [`CompositionRule::Sum`].
    pub sharing_evidenced: bool,
}

impl CostTerm {
    fn validate(&self) -> Result<(), ProfileError> {
        if self.name.is_empty() || self.aggregation_key.is_empty() {
            return Err(ProfileError::Invalid(
                "every cost term needs a name and a stable aggregation key".into(),
            ));
        }
        if self.scope == AllocationScope::PerAllocation
            && self.composition_rule != CompositionRule::Max
        {
            return Err(ProfileError::Invalid(format!(
                "term `{}` is per-allocation, which aggregates by maximum and is never summed",
                self.name
            )));
        }
        if !self.sharing_evidenced && self.composition_rule != CompositionRule::Sum {
            return Err(ProfileError::Invalid(format!(
                "term `{}` claims a sharing rule that no conformance evidence establishes; a term \
                 with unknown sharing takes the conservative non-sharing rule",
                self.name
            )));
        }
        // A formula reading the co-resident role count at a per-role scope would double-count: the
        // aggregate already composes per-role terms across roles.
        if self.scope == AllocationScope::PerRoleInstance
            && self
                .formula
                .inputs()
                .contains(&CostInput::CoResidentRoleCount)
        {
            return Err(ProfileError::Invalid(format!(
                "term `{}` is per-role but reads the co-resident role count; the aggregate already \
                 composes per-role terms across roles, so this would count them twice",
                self.name
            )));
        }
        Ok(())
    }
}

/// The per-allocation ceiling, with **reported and measured carried as distinct members**.
///
/// They disagree in both directions on real hardware, so neither can stand for the other. One
/// platform reports a ceiling well below what its driver actually enforces; another reports a
/// compile-time constant unrelated to the card in the machine. A "verify the reported ceiling"
/// conformance clause is therefore unsatisfiable by reporting alone, and the profile has to say
/// which figure it was calibrated against.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCeilings {
    /// What the platform API reports. Recorded always; trusted alone never.
    pub reported_bytes: u64,
    /// What the driver was **measured** to actually enforce, by probing. Typed unavailability when
    /// the probe has not been run — not a zero, and not a copy of the reported figure.
    pub measured_bytes: Maybe<u64>,
}

impl AllocationCeilings {
    /// The ceiling the planner may compose against.
    ///
    /// The measured figure when one exists; otherwise a typed refusal. Falling back to the reported
    /// figure would silently produce claims priced against a number known to be wrong on two of
    /// three platforms — and the claim would carry a certified profile's provenance while doing it.
    pub fn effective_bytes(&self) -> Result<u64, ProfileError> {
        match self.measured_bytes.value() {
            Some(measured) => Ok(*measured),
            None => Err(ProfileError::NotCertified {
                property: "the per-allocation ceiling",
                detail: format!(
                    "the driver-enforced ceiling has not been measured; the platform reports \
                     {} bytes, which is not evidence of what the driver enforces",
                    self.reported_bytes
                ),
            }),
        }
    }

    /// Whether reported and measured disagree — worth surfacing in a conformance record even when
    /// the measured figure is the larger one, because the disagreement is a fact about the platform.
    pub fn disagree(&self) -> bool {
        self.measured_bytes
            .value()
            .is_some_and(|m| *m != self.reported_bytes)
    }
}

/// Allocator pooling and retention behavior.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolingBehavior {
    /// The block size the allocator reserves in, where it reserves in blocks rather than per
    /// request. A pool block larger than any single tensor is still a single allocation the driver
    /// must satisfy.
    pub reservation_block_bytes: u64,
    /// Whether freed storage is returned to the driver, or retained in the pool. Pools that never
    /// shrink make the peak the standing reservation.
    pub returns_to_driver: bool,
    /// The retained pool reservation, as a formula.
    pub retained_reservation: CostExpr,
}

/// First-use compilation behavior.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationBehavior {
    /// Device-side allocations a first-use compilation materializes — kernel binaries and caches.
    /// These belong **in** the composed claim, not beside it.
    pub allocations: CostExpr,
    /// Whether compiled artifacts persist for the instance's life.
    pub cached_for_instance: bool,
}

/// Import, export and readback staging behavior.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingBehavior {
    /// Staging bytes per in-flight transfer window.
    pub per_window: CostExpr,
    /// Whether staging buffers are pooled across transfers.
    pub pooled: bool,
}

/// The profile's stated uncertainty and the headroom it therefore requires.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Headroom {
    /// The headroom factor applied to the composed total, as an exact ratio.
    pub numerator: u64,
    /// Its denominator, greater than zero.
    pub denominator: u64,
    /// A flat reserve held against allocations the host cannot intercept at all — the
    /// profiled hidden-overhead reserve of `[RC-10]`(3).
    pub hidden_overhead_reserve_bytes: u64,
    /// Plain-language statement of what the uncertainty is and where it comes from. A headroom
    /// figure with no stated basis is a guess wearing a measurement's clothes.
    pub basis: String,
}

impl Headroom {
    fn validate(&self) -> Result<(), ProfileError> {
        if self.denominator == 0 {
            return Err(ProfileError::Invalid(
                "headroom denominator must be greater than zero".into(),
            ));
        }
        if self.numerator < self.denominator {
            return Err(ProfileError::Invalid(
                "headroom must not shrink the composed total; a factor below one is a discount, \
                 not an allowance"
                    .into(),
            ));
        }
        if self.basis.trim().is_empty() {
            return Err(ProfileError::Invalid(
                "headroom must state the basis of its uncertainty".into(),
            ));
        }
        Ok(())
    }

    /// Apply the headroom factor to a composed total.
    pub fn apply(&self, total: u64) -> Result<u64, ProfileError> {
        CostExpr::Ratio {
            term: Box::new(CostExpr::Const(total)),
            numerator: self.numerator,
            denominator: self.denominator,
        }
        .evaluate(&CostInputs::new())
    }
}

/// The workspace formula for one logical operation class, plus the ceiling it is certified against.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceFormula {
    /// The logical operation family this prices — the plan's `family` spelling.
    pub operation_family: String,
    /// The workspace cost.
    pub term: CostTerm,
}

/// The host backend implementation's statement of what it costs to execute a logical plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendExecutionProfile {
    /// Schema version.
    pub schema: u32,
    /// The backend class this profile prices.
    pub backend_class: BackendClass,
    /// The exact backend implementation revision this profile is valid for. A profile is valid only
    /// for the revisions it names; an unmatched revision prevents role execution until revalidation
    /// succeeds.
    pub implementation_revision: String,
    /// The allocator implementation revision its pooling terms were calibrated against.
    pub allocator_revision: String,
    /// The operation families this backend supports.
    pub supported_operation_families: BTreeSet<String>,
    /// The dtype spellings it supports.
    pub supported_dtypes: BTreeSet<String>,
    /// The alignment the backend imposes on a single allocation.
    pub allocation_alignment_bytes: u64,
    /// The per-allocation ceiling, reported and measured as distinct members.
    pub allocation_ceilings: AllocationCeilings,
    /// Workspace formulas, one per priced operation family.
    pub workspace_formulas: Vec<WorkspaceFormula>,
    /// Terms that are not per-operation: persistent residency, pool reservations, process-scoped
    /// context, device-scoped overhead.
    pub standing_terms: Vec<CostTerm>,
    /// Allocator pooling and retention behavior.
    pub pooling: PoolingBehavior,
    /// First-use compilation behavior.
    pub compilation: CompilationBehavior,
    /// Transfer staging behavior.
    pub staging: StagingBehavior,
    /// Stated uncertainty and required headroom.
    pub headroom: Headroom,
    /// The conformance-suite version that certified this profile.
    pub conformance_suite_version: u32,
    /// The digest of the conformance evidence that certified it.
    pub conformance_evidence_digest: Hash,
}

impl BackendExecutionProfile {
    /// The profile's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ProfileError> {
        to_canonical_vec(self).map_err(|e| ProfileError::Invalid(format!("profile encoding: {e}")))
    }

    /// blake3 of the profile's canonical bytes — the profile digest the admitted tuple and the
    /// composition evidence record carry.
    pub fn profile_digest(&self) -> Result<Hash, ProfileError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Every cost term, workspace and standing alike.
    pub fn terms(&self) -> impl Iterator<Item = &CostTerm> {
        self.workspace_formulas
            .iter()
            .map(|w| &w.term)
            .chain(self.standing_terms.iter())
    }

    /// Full schema-1 validation.
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.schema != BACKEND_EXECUTION_PROFILE_SCHEMA {
            return Err(ProfileError::Invalid(format!(
                "unknown profile schema {} (this build understands \
                 {BACKEND_EXECUTION_PROFILE_SCHEMA})",
                self.schema
            )));
        }
        if self.implementation_revision.is_empty() || self.allocator_revision.is_empty() {
            return Err(ProfileError::Invalid(
                "a profile must name the exact backend implementation and allocator revisions it \
                 is valid for; a profile valid for everything is valid for nothing"
                    .into(),
            ));
        }
        if self.allocation_alignment_bytes == 0 {
            return Err(ProfileError::Invalid(
                "allocation alignment must be greater than zero".into(),
            ));
        }
        self.headroom.validate()?;

        // One aggregation key must mean one scope and one rule, or the aggregate is ambiguous.
        let mut keys: BTreeMap<&str, (AllocationScope, CompositionRule)> = BTreeMap::new();
        for term in self.terms() {
            term.validate()?;
            match keys.get(term.aggregation_key.as_str()) {
                Some((scope, rule)) if *scope != term.scope || *rule != term.composition_rule => {
                    return Err(ProfileError::Invalid(format!(
                        "aggregation key `{}` is given two different scope/rule pairs; a shared \
                         term cannot compose two ways at once",
                        term.aggregation_key
                    )));
                }
                Some(_) => {}
                None => {
                    keys.insert(
                        term.aggregation_key.as_str(),
                        (term.scope, term.composition_rule),
                    );
                }
            }
        }

        // A workspace formula for an operation family the backend does not support prices something
        // it cannot run; an unpriced supported family is a term nobody accounted.
        for workspace in &self.workspace_formulas {
            if !self
                .supported_operation_families
                .contains(&workspace.operation_family)
            {
                return Err(ProfileError::Invalid(format!(
                    "a workspace formula prices operation family `{}`, which this backend does not \
                     declare support for",
                    workspace.operation_family
                )));
            }
        }
        let priced: BTreeSet<&String> = self
            .workspace_formulas
            .iter()
            .map(|w| &w.operation_family)
            .collect();
        for family in &self.supported_operation_families {
            if !priced.contains(family) {
                return Err(ProfileError::Invalid(format!(
                    "operation family `{family}` is supported but unpriced; an unpriced family is a \
                     cost the composition cannot see"
                )));
            }
        }
        Ok(())
    }

    /// Whether this profile may price a plan that needs `families` and `dtypes`. A plan requiring
    /// an operation class the backend does not have is a typed refusal, never a fallback to an
    /// estimate.
    pub fn supports(
        &self,
        families: &BTreeSet<String>,
        dtypes: &BTreeSet<String>,
    ) -> Result<(), ProfileError> {
        for family in families {
            if !self.supported_operation_families.contains(family) {
                return Err(ProfileError::Invalid(format!(
                    "the plan requires operation family `{family}`, which backend class `{}` does \
                     not support",
                    self.backend_class.slug()
                )));
            }
        }
        for dtype in dtypes {
            if !self.supported_dtypes.contains(dtype) {
                return Err(ProfileError::Invalid(format!(
                    "the plan requires dtype `{dtype}`, which backend class `{}` does not support",
                    self.backend_class.slug()
                )));
            }
        }
        Ok(())
    }

    /// The workspace formula for one operation family.
    pub fn workspace_for(&self, family: &str) -> Option<&WorkspaceFormula> {
        self.workspace_formulas
            .iter()
            .find(|w| w.operation_family == family)
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    pub(crate) fn term(
        name: &str,
        scope: AllocationScope,
        rule: CompositionRule,
        enforcement: EnforcementClass,
        formula: CostExpr,
    ) -> CostTerm {
        CostTerm {
            name: name.into(),
            scope,
            aggregation_key: name.into(),
            composition_rule: rule,
            enforcement,
            formula,
            sharing_evidenced: rule != CompositionRule::Sum,
        }
    }

    /// A small but complete and valid profile.
    pub(crate) fn profile(class: BackendClass) -> BackendExecutionProfile {
        BackendExecutionProfile {
            schema: BACKEND_EXECUTION_PROFILE_SCHEMA,
            backend_class: class,
            implementation_revision: "0.10.0".into(),
            allocator_revision: "0.10.0".into(),
            supported_operation_families: ["gemm".to_string()].into_iter().collect(),
            supported_dtypes: ["f32".to_string(), "bool1".to_string()]
                .into_iter()
                .collect(),
            allocation_alignment_bytes: 256,
            allocation_ceilings: AllocationCeilings {
                reported_bytes: 2 << 30,
                measured_bytes: Maybe::Available(4 << 30),
            },
            workspace_formulas: vec![WorkspaceFormula {
                operation_family: "gemm".into(),
                term: term(
                    "gemm_workspace",
                    AllocationScope::PerRoleInstance,
                    CompositionRule::Sum,
                    EnforcementClass::DirectlyEnforceable,
                    CostExpr::Ratio {
                        term: Box::new(CostExpr::Input(CostInput::LogicalBytes)),
                        numerator: 1,
                        denominator: 8,
                    },
                ),
            }],
            standing_terms: vec![
                term(
                    "resident_tensors",
                    AllocationScope::PerRoleInstance,
                    CompositionRule::Sum,
                    EnforcementClass::DirectlyEnforceable,
                    CostExpr::AlignUp(
                        Box::new(CostExpr::Input(CostInput::PersistentLogicalBytes)),
                        256,
                    ),
                ),
                term(
                    "device_context",
                    AllocationScope::PerDevice,
                    CompositionRule::OncePerKey,
                    EnforcementClass::ProfiledAndMeasured,
                    CostExpr::Const(64 << 20),
                ),
                term(
                    "largest_single_allocation",
                    AllocationScope::PerAllocation,
                    CompositionRule::Max,
                    EnforcementClass::DirectlyEnforceable,
                    CostExpr::AlignUp(
                        Box::new(CostExpr::Input(CostInput::LargestLogicalObjectBytes)),
                        256,
                    ),
                ),
            ],
            pooling: PoolingBehavior {
                reservation_block_bytes: 32 << 20,
                returns_to_driver: false,
                retained_reservation: CostExpr::Input(CostInput::TransientPeakLogicalBytes),
            },
            compilation: CompilationBehavior {
                allocations: CostExpr::Const(16 << 20),
                cached_for_instance: true,
            },
            staging: StagingBehavior {
                per_window: CostExpr::Mul(
                    Box::new(CostExpr::Input(CostInput::TransferWindowBytes)),
                    Box::new(CostExpr::Const(2)),
                ),
                pooled: true,
            },
            headroom: Headroom {
                numerator: 21,
                denominator: 20,
                hidden_overhead_reserve_bytes: 128 << 20,
                basis: "measured allocator high-water spread across the conformance workloads"
                    .into(),
            },
            conformance_suite_version: 1,
            conformance_evidence_digest: Hash([9u8; 32]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{profile, term};
    use super::*;

    fn inputs() -> CostInputs {
        CostInputs::from([
            (CostInput::LogicalBytes, 8192),
            (CostInput::PersistentLogicalBytes, 4000),
            (CostInput::TransientPeakLogicalBytes, 2048),
            (CostInput::LargestLogicalObjectBytes, 1000),
            (CostInput::TransferWindowBytes, 65_536),
            (CostInput::InFlight, 2),
            (CostInput::ElementCount, 1024),
            (CostInput::CoResidentRoleCount, 2),
        ])
    }

    #[test]
    fn a_complete_profile_validates_and_digests_deterministically() {
        let p = profile(BackendClass::Vulkan);
        p.validate().expect("valid");
        assert_eq!(
            p.profile_digest().unwrap(),
            profile(BackendClass::Vulkan).profile_digest().unwrap(),
            "the digest is a deterministic function of the profile's content"
        );
        assert_ne!(
            p.profile_digest().unwrap(),
            profile(BackendClass::Metal).profile_digest().unwrap()
        );
    }

    /// Reported and measured ceilings are distinct members, because they disagree in both
    /// directions on real hardware and neither can stand for the other.
    #[test]
    fn the_per_allocation_ceiling_carries_reported_and_measured_separately() {
        let p = profile(BackendClass::Vulkan);
        assert_eq!(p.allocation_ceilings.reported_bytes, 2 << 30);
        assert_eq!(p.allocation_ceilings.effective_bytes().unwrap(), 4 << 30);
        assert!(
            p.allocation_ceilings.disagree(),
            "a platform reporting below what its driver enforces is recorded as a disagreement"
        );
    }

    /// Without a measured ceiling the profile is not certified for it, and composing anyway would
    /// price the claim against a figure known to be wrong on real platforms.
    #[test]
    fn an_unmeasured_ceiling_is_a_typed_refusal_not_a_fallback_to_the_reported_figure() {
        let mut p = profile(BackendClass::Dx12);
        p.allocation_ceilings = AllocationCeilings {
            reported_bytes: u64::from(u32::MAX / 2),
            measured_bytes: Maybe::default(),
        };
        let err = p.allocation_ceilings.effective_bytes().unwrap_err();
        assert!(matches!(err, ProfileError::NotCertified { .. }));
        assert!(err.to_string().contains("has not been measured"));
        assert!(!p.allocation_ceilings.disagree());
    }

    /// A per-allocation term is a maximum constraint. Summing it would turn a ceiling into an
    /// occupancy reservation and refuse machines for memory nobody is holding.
    #[test]
    fn a_per_allocation_term_must_aggregate_by_maximum() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms.push(term(
            "bad_ceiling",
            AllocationScope::PerAllocation,
            CompositionRule::Sum,
            EnforcementClass::DirectlyEnforceable,
            CostExpr::Const(1),
        ));
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("never summed"));
        assert!(!AllocationScope::PerAllocation.is_occupancy());
        assert!(AllocationScope::PerProcess.is_occupancy());
    }

    /// A term whose sharing conformance has not established takes the conservative non-sharing
    /// rule. Over-reserving refuses a machine that would have worked; under-reserving admits one
    /// that then dies.
    #[test]
    fn unknown_sharing_takes_the_conservative_non_sharing_rule() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms.push(CostTerm {
            name: "guessed_pool".into(),
            scope: AllocationScope::PerProcess,
            aggregation_key: "guessed_pool".into(),
            composition_rule: CompositionRule::OncePerKey,
            enforcement: EnforcementClass::ProfiledAndMeasured,
            formula: CostExpr::Const(1),
            sharing_evidenced: false,
        });
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("conservative non-sharing rule"));
    }

    #[test]
    fn one_aggregation_key_means_one_scope_and_one_rule() {
        let mut p = profile(BackendClass::Vulkan);
        let mut clash = term(
            "device_context",
            AllocationScope::PerProcess,
            CompositionRule::OncePerKey,
            EnforcementClass::ProfiledAndMeasured,
            CostExpr::Const(1),
        );
        clash.aggregation_key = "device_context".into();
        p.standing_terms.push(clash);
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("two different scope/rule pairs"));
    }

    /// The two enforcement classes are carried on every term, so a certification statement can
    /// report them separately instead of claiming one property over both.
    #[test]
    fn every_term_declares_which_enforcement_class_it_belongs_to() {
        let p = profile(BackendClass::Vulkan);
        let enforceable = p
            .terms()
            .filter(|t| t.enforcement == EnforcementClass::DirectlyEnforceable)
            .count();
        let profiled = p
            .terms()
            .filter(|t| t.enforcement == EnforcementClass::ProfiledAndMeasured)
            .count();
        assert!(enforceable > 0 && profiled > 0);
        assert_eq!(enforceable + profiled, p.terms().count());
    }

    #[test]
    fn a_supported_family_must_be_priced_and_a_priced_family_supported() {
        let mut p = profile(BackendClass::Vulkan);
        p.supported_operation_families.insert("reduce".into());
        assert!(p.validate().unwrap_err().to_string().contains("unpriced"));

        let mut p = profile(BackendClass::Vulkan);
        p.workspace_formulas.push(WorkspaceFormula {
            operation_family: "unsupported".into(),
            term: term(
                "x",
                AllocationScope::PerRoleInstance,
                CompositionRule::Sum,
                EnforcementClass::DirectlyEnforceable,
                CostExpr::Const(1),
            ),
        });
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("does not declare support for"));
    }

    #[test]
    fn a_plan_needing_an_unsupported_operation_or_dtype_is_a_typed_refusal() {
        let p = profile(BackendClass::Vulkan);
        let families = ["flash_attention".to_string()].into_iter().collect();
        let dtypes = ["f32".to_string()].into_iter().collect();
        assert!(p
            .supports(&families, &dtypes)
            .unwrap_err()
            .to_string()
            .contains("does not support"));

        let ok_families = ["gemm".to_string()].into_iter().collect();
        let bad_dtypes = ["f8".to_string()].into_iter().collect();
        assert!(p.supports(&ok_families, &bad_dtypes).is_err());
        assert!(p.supports(&ok_families, &dtypes).is_ok());
    }

    /// Formulas are exact rationals, never floats, so two hosts composing the same plan and
    /// profile reach byte-identical claims. A headroom factor rounds UP.
    #[test]
    fn formulas_are_exact_and_headroom_rounds_up() {
        let expr = CostExpr::Ratio {
            term: Box::new(CostExpr::Const(100)),
            numerator: 21,
            denominator: 20,
        };
        assert_eq!(expr.evaluate(&CostInputs::new()).unwrap(), 105);
        let odd = CostExpr::Ratio {
            term: Box::new(CostExpr::Const(1)),
            numerator: 21,
            denominator: 20,
        };
        assert_eq!(odd.evaluate(&CostInputs::new()).unwrap(), 2, "rounds up");

        let p = profile(BackendClass::Vulkan);
        assert_eq!(p.headroom.apply(1000).unwrap(), 1050);
    }

    #[test]
    fn alignment_belongs_to_the_profile_and_is_applied_by_it() {
        let expr = CostExpr::AlignUp(Box::new(CostExpr::Const(4000)), 256);
        assert_eq!(expr.evaluate(&inputs()).unwrap(), 4096);
    }

    #[test]
    fn overflow_is_a_refusal_and_a_missing_input_is_named() {
        let overflowing = CostExpr::Mul(
            Box::new(CostExpr::Const(u64::MAX)),
            Box::new(CostExpr::Const(2)),
        );
        assert!(overflowing
            .evaluate(&inputs())
            .unwrap_err()
            .to_string()
            .contains("overflow"));

        let missing = CostExpr::Input(CostInput::LogicalBytes);
        assert!(missing
            .evaluate(&CostInputs::new())
            .unwrap_err()
            .to_string()
            .contains("no value supplied"));
    }

    /// A per-role term that reads the co-resident role count would be counted twice: once by the
    /// formula and once by the aggregate composing per-role terms across roles.
    #[test]
    fn a_per_role_term_must_not_read_the_co_resident_role_count() {
        let mut p = profile(BackendClass::Vulkan);
        p.standing_terms.push(term(
            "double_counted",
            AllocationScope::PerRoleInstance,
            CompositionRule::Sum,
            EnforcementClass::DirectlyEnforceable,
            CostExpr::Input(CostInput::CoResidentRoleCount),
        ));
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("count them twice"));
    }

    #[test]
    fn headroom_must_not_shrink_the_total_and_must_state_its_basis() {
        let mut p = profile(BackendClass::Vulkan);
        p.headroom.numerator = 19;
        assert!(p.validate().unwrap_err().to_string().contains("discount"));

        let mut p = profile(BackendClass::Vulkan);
        p.headroom.basis = "  ".into();
        assert!(p.validate().unwrap_err().to_string().contains("basis"));
    }

    #[test]
    fn a_profile_must_name_the_revisions_it_is_valid_for() {
        let mut p = profile(BackendClass::Vulkan);
        p.implementation_revision.clear();
        assert!(p
            .validate()
            .unwrap_err()
            .to_string()
            .contains("valid for everything is valid for nothing"));
    }
}
