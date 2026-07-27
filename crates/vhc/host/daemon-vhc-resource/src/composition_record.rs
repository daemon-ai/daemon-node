// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **two-layer freeze**: the frozen candidate tuple, the append-only composition evidence
//! ledger, and the co-citation validator that joins them
//! (`docs/specs/vhc-fleet-ceremony-runbook.md` §7 T7; `[PC-11]`, `[PC-13]`, `[DI-9]`, `[DI-10]`).
//!
//! ## Why two layers
//!
//! The freeze discipline pins a **candidate tuple** and requires every subsequent artifact to cite
//! it; a membership change re-freezes the candidate and re-runs the full battery. That worked while
//! resources were one number. Under the three-object model it breaks in both directions:
//!
//! - putting profile digests, capability reports and grants **into** the tuple would force a full
//!   re-freeze and a full battery for a driver update — exactly the coupling `[PC-11]` removes;
//! - leaving them out **entirely** means `[PC-13]`'s recertification triggers have nothing to
//!   compare against: a profile could change and no artifact would record that it had.
//!
//! So: layer 1 keeps its semantics unchanged and gains the four identities whose change is already
//! a candidate recertification. Layer 2 is an append-only ledger of composition evidence whose
//! lifecycle is `[PC-13]`'s consequence table, and which **never forces a battery by itself**. The
//! two rows that *do* reach layer 1 — a planner fix and a governor fix — are exactly the two
//! identities layer 1 now carries. That join is the design, not a coincidence.
//!
//! ## The tuple never points forward
//!
//! [`CandidateTuple`] contains no reference to any composition record, and it must not gain one. A
//! frozen artifact cannot reference evidence created after it was frozen; the join is made by the
//! **citing artifact**, at the time it is written, and by nothing else. If a composition-record
//! field ever appears on the tuple, the design has been inverted.

use std::collections::{BTreeMap, BTreeSet};

use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId};
use serde::{Deserialize, Serialize};

use crate::governor::{ReservationComponents, ReservationIdentity};
use crate::planner::{AggregateEstimate, PhysicalEstimate};

/// The number of normative bindings a layer-2 composition evidence record carries.
///
/// Asserted rather than merely documented, because the count is the thing a reader checks a record
/// against and it has already grown once.
pub const COMPOSITION_RECORD_MEMBERS: usize = 12;

/// The resource governor's implementation identity.
///
/// A governor or scoped-aggregation fix invalidates reservation and enforcement evidence and is a
/// **candidate recertification** under `[PC-13]`, which is why this rides the frozen tuple beside
/// the planner version. Bump it in the change that alters reservation or aggregation behavior.
pub const GOVERNOR_VERSION: u32 = 1;

/// The layer-1 **frozen candidate tuple**.
///
/// The first four members are the pre-existing freeze and their semantics are **unchanged**: a
/// membership change re-freezes the candidate and re-runs the full battery. The last four are
/// added because `[PC-13]` already classes a change in any of them as candidate recertification, so
/// a tuple that did not record them could not be compared against.
///
/// All four additions are **contract identifiers, not release versions**. They move under their own
/// contract rules; no `VERSION` file is involved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateTuple {
    /// The node commit the candidate was built from.
    pub node_commit: String,
    /// The cloud commit plus the deployed version id it serves.
    pub cloud_deployment: String,
    /// The digest over the pinned guest module hashes.
    pub guests_digest: Hash,
    /// The digests of the tooling binaries the candidate ships.
    pub tooling_digests: BTreeMap<String, Hash>,
    /// The composition planner identity (`[DI-10]`, `[PC-13]` planner fix).
    pub planner_version: u32,
    /// The resource governor identity (`[PC-13]` governor or scoped-aggregation fix).
    pub governor_version: u32,
    /// The **full** module ABI, major and minor. Not the major alone: the certification minor
    /// carries pre-loop diagnostic logging and both resource exports together.
    pub abi_major: u32,
    /// See [`CandidateTuple::abi_major`].
    pub abi_minor: u32,
    /// The genesis envelope schema major. The bounded framing header and the role
    /// execution-requirement structure ride it.
    pub genesis_schema_major: u32,
}

impl CandidateTuple {
    /// The tuple's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, EvidenceError> {
        to_canonical_vec(self).map_err(|e| EvidenceError::Invalid(format!("tuple encoding: {e}")))
    }

    /// blake3 of the canonical bytes — the `candidate_tuple_digest` every artifact cites.
    pub fn tuple_digest(&self) -> Result<Hash, EvidenceError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }
}

/// Which `[PC-13]` row caused this record to exist. A closed enumeration: a trigger that is not one
/// of these is not a recertification trigger, and calling it one would let a documentation update
/// masquerade as evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsequenceClass {
    /// A profile was corrected or re-issued.
    ProfileCorrection,
    /// The planner was fixed. Reaches layer 1.
    PlannerFix,
    /// The capability probe was fixed.
    CapabilityProbeFix,
    /// The governor or its scoped aggregation was fixed. Reaches layer 1.
    GovernorFix,
    /// The backend implementation revision changed.
    BackendImplementationRevision,
    /// The first record for this subject — no prior evidence to supersede.
    InitialComposition,
}

impl ConsequenceClass {
    /// Whether this class also invalidates a **layer-1** identity, and therefore requires a
    /// candidate recertification rather than only a re-composition.
    pub fn reaches_candidate_layer(self) -> bool {
        matches!(self, Self::PlannerFix | Self::GovernorFix)
    }
}

/// What a composition record is *about*: one participant, one backend, one admitted role instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EvidenceSubject {
    /// The participant's node identity.
    pub participant: PeerId,
    /// The backend class slug the role executes on.
    pub backend_class: String,
    /// The device the role is bound to.
    pub device_identity: String,
    /// The run role.
    pub role: String,
    /// The role instance's incarnation.
    pub incarnation: u64,
}

/// One layer-2 **composition evidence record**. Immutable once appended.
///
/// It carries [`COMPOSITION_RECORD_MEMBERS`] normative bindings: the role/incarnation and
/// participant/device identity, the profile digest and its authority, the capability-report digest,
/// the plan hash, the grant digest, the composed role claim and the node/device aggregate, the
/// planner identity, the governor identity, the reservation identity, the reservation digest, and
/// the scope-separated reservation components. The lifecycle members (`supersedes`, the trigger
/// reason and the consequence class) are the record's own bookkeeping and are counted separately.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionEvidenceRecord {
    /// What this record is about.
    pub subject: EvidenceSubject,
    /// The Backend Execution Profile that priced the claim. The profile **transitively** pins the
    /// backend implementation revision and the permitted driver/API ranges it names, so those are
    /// deliberately not separate members: duplicating them would create a second spelling of the
    /// same fact with nothing binding the two.
    pub profile_digest: Hash,
    /// Who vouched for that profile. Required, because content addressing proves identity and not
    /// authority (`[PC-12]`).
    pub profile_authority: PeerId,
    /// The planner versions the profile names itself compatible with. Carried so a citation can be
    /// checked without re-fetching the profile bytes.
    pub profile_compatible_planner_versions: BTreeSet<u32>,
    /// The Device Capability Report the claim was validated against.
    pub capability_report_digest: Hash,
    /// The Logical Resource Plan that was priced.
    pub logical_resource_plan_hash: Hash,
    /// The digest of the Execution Grant's **canonical bytes**.
    pub execution_grant_digest: Hash,
    /// The composed role Physical Estimate.
    pub physical_estimate: PhysicalEstimate,
    /// The node/device aggregate claim reserved for this incarnation.
    pub aggregate_estimate: AggregateEstimate,
    /// The planner that composed it.
    pub planner_version: u32,
    /// The governor that reserved it.
    pub governor_version: u32,
    /// The identity of the **single reservation** this admission took, as recorded in the admitted
    /// tuple. The owner-ledger charge and the governor's occupancy reservation are the same
    /// reservation seen twice, and this is the identity both views refer to.
    pub reservation_identity: ReservationIdentity,
    /// The digest of that reservation's canonical encoding.
    ///
    /// Identity alone only **locates** a reservation; it does not prove what was charged. A record
    /// carrying identity and no digest lets an auditor find the reservation and then trust its
    /// current contents.
    pub reservation_digest: Hash,
    /// The reserved amounts separated by allocation scope, with the hidden-overhead reserve as its
    /// own visible component and the two enforcement classes separated.
    ///
    /// Totals without scope separation cannot distinguish a correctly-shared process-scoped term
    /// from a double-counted per-role one — which is precisely the colocation defect the legacy
    /// per-role tier sum produced.
    pub reservation_components: ReservationComponents,
    /// The record this one replaces, or `None` for the first record of its subject.
    pub supersedes: Option<Hash>,
    /// Plain language: why this record exists. A human reads this first.
    pub trigger_reason: String,
    /// Which `[PC-13]` row caused it.
    pub consequence_class: ConsequenceClass,
}

impl CompositionEvidenceRecord {
    /// The record's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, EvidenceError> {
        to_canonical_vec(self).map_err(|e| EvidenceError::Invalid(format!("record encoding: {e}")))
    }

    /// blake3 of the canonical bytes — the `composition_evidence_digest` an artifact cites.
    pub fn record_digest(&self) -> Result<Hash, EvidenceError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    fn validate(&self) -> Result<(), EvidenceError> {
        if self.trigger_reason.trim().is_empty() {
            return Err(EvidenceError::Invalid(
                "a composition record must state, in plain language, why it exists".into(),
            ));
        }
        match (self.consequence_class, self.supersedes) {
            (ConsequenceClass::InitialComposition, Some(_)) => Err(EvidenceError::Invalid(
                "an initial composition supersedes nothing".into(),
            )),
            (ConsequenceClass::InitialComposition, None) => Ok(()),
            (_, None) => Err(EvidenceError::Invalid(
                "a record triggered by a recertification row must name the record it replaces"
                    .into(),
            )),
            (_, Some(_)) => Ok(()),
        }
    }
}

/// An appended revocation of a record found unsound. Itself immutable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revocation {
    /// The record being revoked.
    pub record_digest: Hash,
    /// Plain language: why.
    pub reason: String,
}

/// A record's standing, evaluated at the moment a statement is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordStanding {
    /// Current: not superseded, not revoked.
    Current,
    /// Replaced by a successor.
    Superseded,
    /// Explicitly revoked.
    Revoked,
}

/// Why evidence or a citation was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EvidenceError {
    /// The record or tuple is structurally unusable.
    #[error("composition evidence is invalid: {0}")]
    Invalid(String),
    /// An append would mutate history.
    #[error("a composition record is immutable and append-only; {0}")]
    AppendOnly(String),
    /// A cited digest resolves to nothing.
    #[error("cited composition evidence {} resolves to no record", .0.to_hex())]
    Dangling(Hash),
    /// A `supersedes` chain does not resolve.
    #[error("the supersedes chain from {} does not resolve to a current record", .0.to_hex())]
    BrokenChain(Hash),
    /// A cited record has been replaced.
    #[error(
        "cited composition evidence {} has been superseded; a statement is valid only against \
         current evidence",
        .0.to_hex()
    )]
    Superseded(Hash),
    /// A cited record has been revoked.
    #[error("cited composition evidence {} has been revoked", .0.to_hex())]
    Revoked(Hash),
    /// An artifact cited only one of the two digests.
    #[error(
        "an artifact must cite BOTH the candidate tuple and its composition evidence; citing one \
         is incomplete and is not admissible evidence ({0})"
    )]
    IncompleteCitation(&'static str),
    /// The tuple cited is not the tuple supplied.
    #[error("the artifact cites a different candidate tuple than the one being validated against")]
    TupleMismatch,
    /// A record's layer-1 identity disagrees with the tuple's.
    #[error(
        "cross-layer disagreement: the composition record was produced by {which} {record}, but \
         the frozen candidate carries {tuple}"
    )]
    CrossLayerMismatch {
        /// Which identity disagreed.
        which: &'static str,
        /// What the record says.
        record: u32,
        /// What the tuple says.
        tuple: u32,
    },
    /// The profile is not priced for the planner that composed the claim.
    #[error(
        "the profile names compatible planner versions {named:?}, which exclude the planner that \
         composed this claim ({planner}); the claim is not composable evidence"
    )]
    ProfilePlannerIncompatible {
        /// What the profile names.
        named: BTreeSet<u32>,
        /// What composed the claim.
        planner: u32,
    },
}

/// The append-only composition evidence ledger.
///
/// Records are never edited in place and never deleted. A record may be **revoked** — by a successor
/// that supersedes it, or by an explicit revocation for a record found unsound — and a revocation is
/// itself an appended, immutable entry.
#[derive(Clone, Debug, Default)]
pub struct CompositionEvidenceLedger {
    records: BTreeMap<Hash, CompositionEvidenceRecord>,
    superseded_by: BTreeMap<Hash, Hash>,
    revocations: BTreeMap<Hash, Revocation>,
    append_order: Vec<Hash>,
}

impl CompositionEvidenceLedger {
    /// A new, empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a record, returning its digest.
    ///
    /// Refuses to re-append an existing digest, to supersede a record the ledger does not hold, and
    /// to supersede a record that has already been superseded — the last of these is what keeps a
    /// chain linear and therefore resolvable to exactly one current record.
    pub fn append(&mut self, record: CompositionEvidenceRecord) -> Result<Hash, EvidenceError> {
        record.validate()?;
        let digest = record.record_digest()?;
        if self.records.contains_key(&digest) {
            return Err(EvidenceError::AppendOnly(format!(
                "record {} is already in the ledger",
                digest.to_hex()
            )));
        }
        if let Some(previous) = record.supersedes {
            if !self.records.contains_key(&previous) {
                return Err(EvidenceError::Dangling(previous));
            }
            if let Some(existing) = self.superseded_by.get(&previous) {
                return Err(EvidenceError::AppendOnly(format!(
                    "record {} is already superseded by {}; a chain must stay linear so it \
                     resolves to exactly one current record",
                    previous.to_hex(),
                    existing.to_hex()
                )));
            }
            self.superseded_by.insert(previous, digest);
        }
        self.records.insert(digest, record);
        self.append_order.push(digest);
        Ok(digest)
    }

    /// Append a revocation for a record found unsound.
    pub fn revoke(&mut self, record_digest: Hash, reason: &str) -> Result<(), EvidenceError> {
        if !self.records.contains_key(&record_digest) {
            return Err(EvidenceError::Dangling(record_digest));
        }
        if reason.trim().is_empty() {
            return Err(EvidenceError::Invalid(
                "a revocation must state why the record is unsound".into(),
            ));
        }
        self.revocations.insert(
            record_digest,
            Revocation {
                record_digest,
                reason: reason.to_string(),
            },
        );
        Ok(())
    }

    /// A record by digest.
    pub fn get(&self, digest: &Hash) -> Option<&CompositionEvidenceRecord> {
        self.records.get(digest)
    }

    /// How many records have been appended.
    pub fn len(&self) -> usize {
        self.append_order.len()
    }

    /// Whether the ledger holds nothing.
    pub fn is_empty(&self) -> bool {
        self.append_order.is_empty()
    }

    /// A record's standing right now.
    pub fn standing(&self, digest: &Hash) -> Result<RecordStanding, EvidenceError> {
        if !self.records.contains_key(digest) {
            return Err(EvidenceError::Dangling(*digest));
        }
        if self.revocations.contains_key(digest) {
            return Ok(RecordStanding::Revoked);
        }
        if self.superseded_by.contains_key(digest) {
            return Ok(RecordStanding::Superseded);
        }
        Ok(RecordStanding::Current)
    }

    /// Follow a `supersedes` chain forward to the one current record that replaced `digest`.
    ///
    /// A chain that does not resolve — a missing link, or a cycle — fails closed. There is no
    /// "probably still fine" path: the whole point of resolving is to know which record a statement
    /// is actually standing on.
    pub fn resolve_current(&self, digest: &Hash) -> Result<Hash, EvidenceError> {
        if !self.records.contains_key(digest) {
            return Err(EvidenceError::Dangling(*digest));
        }
        let mut cursor = *digest;
        let mut seen = BTreeSet::new();
        while let Some(next) = self.superseded_by.get(&cursor) {
            if !seen.insert(cursor) {
                return Err(EvidenceError::BrokenChain(*digest));
            }
            if !self.records.contains_key(next) {
                return Err(EvidenceError::BrokenChain(*digest));
            }
            cursor = *next;
        }
        if self.revocations.contains_key(&cursor) {
            return Err(EvidenceError::Revoked(cursor));
        }
        Ok(cursor)
    }

    /// The current record for a subject, if the ledger holds one.
    pub fn current_for(&self, subject: &EvidenceSubject) -> Option<&CompositionEvidenceRecord> {
        self.append_order
            .iter()
            .filter_map(|d| self.records.get(d).map(|r| (d, r)))
            .find(|(d, r)| {
                &r.subject == subject
                    && !self.superseded_by.contains_key(*d)
                    && !self.revocations.contains_key(*d)
            })
            .map(|(_, r)| r)
    }
}

/// What an artifact produced after the freeze cites.
///
/// Every certification statement, journal evidence bundle, replay verdict, preflight record and
/// conformance record cites the **pair**. An artifact legitimately spanning several participants
/// cites one candidate-tuple digest and the *set* of composition-evidence digests it covers, naming
/// each.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCitation {
    /// The frozen candidate this artifact was produced against.
    pub candidate_tuple_digest: Hash,
    /// Every composition record it covers.
    pub composition_evidence_digests: Vec<Hash>,
}

/// Validate a citation against the frozen candidate and the ledger.
///
/// Four things, all fail-closed:
///
/// 1. **Both digests present.** One is incomplete and not admissible evidence.
/// 2. **The tuple never points forward** — structural, and asserted by the tuple type carrying no
///    composition-record member at all.
/// 3. **Cross-layer agreement.** A record's planner and governor identities must equal the tuple's,
///    and a profile whose named compatible planner versions exclude that planner is not composable.
/// 4. **Validity, evaluated now.** Every cited digest must resolve to a record that is current and
///    not revoked. A superseded record, a revoked record, a dangling digest, or an unresolvable
///    chain fails the statement closed.
pub fn validate_citation(
    citation: &EvidenceCitation,
    tuple: &CandidateTuple,
    ledger: &CompositionEvidenceLedger,
) -> Result<(), EvidenceError> {
    // 1. Both digests present.
    if citation.candidate_tuple_digest == Hash([0u8; 32]) {
        return Err(EvidenceError::IncompleteCitation(
            "no candidate tuple digest",
        ));
    }
    if citation.composition_evidence_digests.is_empty() {
        return Err(EvidenceError::IncompleteCitation(
            "no composition evidence digest",
        ));
    }
    if citation.candidate_tuple_digest != tuple.tuple_digest()? {
        return Err(EvidenceError::TupleMismatch);
    }

    for digest in &citation.composition_evidence_digests {
        // 4. Validity first — a dangling digest cannot be cross-checked against anything.
        let record = ledger.get(digest).ok_or(EvidenceError::Dangling(*digest))?;
        match ledger.standing(digest)? {
            RecordStanding::Current => {}
            RecordStanding::Superseded => return Err(EvidenceError::Superseded(*digest)),
            RecordStanding::Revoked => return Err(EvidenceError::Revoked(*digest)),
        }
        // The chain from this record must also resolve — a citation of a current record whose own
        // ancestry is broken is standing on evidence nobody can reconstruct.
        ledger.resolve_current(digest)?;

        // 3. Cross-layer agreement.
        if record.planner_version != tuple.planner_version {
            return Err(EvidenceError::CrossLayerMismatch {
                which: "planner version",
                record: record.planner_version,
                tuple: tuple.planner_version,
            });
        }
        if record.governor_version != tuple.governor_version {
            return Err(EvidenceError::CrossLayerMismatch {
                which: "governor version",
                record: record.governor_version,
                tuple: tuple.governor_version,
            });
        }
        if !record
            .profile_compatible_planner_versions
            .contains(&record.planner_version)
        {
            return Err(EvidenceError::ProfilePlannerIncompatible {
                named: record.profile_compatible_planner_versions.clone(),
                planner: record.planner_version,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::fixtures::report;
    use crate::planner::{aggregate, compose, PLANNER_VERSION};
    use crate::profile::fixtures::profile;
    use crate::revision::BackendClass;
    use daemon_vhc_proto::resource_plan::{
        Binding, Dimension, DimensionValue, Domain, Dtype, Expr, Lifetime, LogicalResourcePlan,
        Retention, SelectionScope, TensorDecl,
    };

    fn tuple() -> CandidateTuple {
        CandidateTuple {
            node_commit: "51cb03af".into(),
            cloud_deployment: "cloud@1234".into(),
            guests_digest: Hash([3u8; 32]),
            tooling_digests: BTreeMap::from([("worker".to_string(), Hash([4u8; 32]))]),
            planner_version: PLANNER_VERSION,
            governor_version: GOVERNOR_VERSION,
            abi_major: daemon_vhc_abi_major(),
            abi_minor: daemon_vhc_abi_minor(),
            genesis_schema_major: daemon_vhc_proto::GENESIS_SCHEMA_MAJOR,
        }
    }

    // The tuple carries the FULL ABI, so the test names both halves explicitly rather than packing
    // them — a packed value is exactly what let the minor go unrecorded.
    fn daemon_vhc_abi_major() -> u32 {
        2
    }
    fn daemon_vhc_abi_minor() -> u32 {
        5
    }

    fn plan() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![Dimension {
                name: "micro_batch".into(),
                domain: Domain::UintRange { lo: 1, hi: 4 },
            }],
            tensors: vec![TensorDecl {
                name: "params".into(),
                shape: vec![Expr::Const(65_536)],
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

    fn subject(role: &str, incarnation: u64) -> EvidenceSubject {
        EvidenceSubject {
            participant: PeerId([9u8; 32]),
            backend_class: "vulkan".into(),
            device_identity: "0000:c4:00.0".into(),
            role: role.into(),
            incarnation,
        }
    }

    fn record(
        role: &str,
        incarnation: u64,
        supersedes: Option<Hash>,
        class: ConsequenceClass,
    ) -> CompositionEvidenceRecord {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let binding = Binding::from([("micro_batch".to_string(), DimensionValue::Uint(2))]);
        let (claim, occ) = compose(&p, &binding, &prof, 1).unwrap();
        let agg = aggregate(&[(role.to_string(), claim.clone(), occ.clone())]).unwrap();
        let reservation = crate::governor::derive_reservation(
            ReservationIdentity {
                role: role.into(),
                incarnation,
                device_identity: "0000:c4:00.0".into(),
                sequence: 1,
            },
            &claim,
            &occ,
            &agg,
            &prof,
        )
        .unwrap();
        CompositionEvidenceRecord {
            reservation_identity: reservation.identity.clone(),
            reservation_digest: reservation.reservation_digest().unwrap(),
            reservation_components: reservation.bounds().components,
            subject: subject(role, incarnation),
            profile_digest: prof.profile_digest().unwrap(),
            profile_authority: PeerId([7u8; 32]),
            profile_compatible_planner_versions: [PLANNER_VERSION].into_iter().collect(),
            capability_report_digest: report(BackendClass::Vulkan).report_digest().unwrap(),
            logical_resource_plan_hash: p.plan_hash().unwrap(),
            execution_grant_digest: Hash([8u8; 32]),
            physical_estimate: claim,
            aggregate_estimate: agg,
            planner_version: PLANNER_VERSION,
            governor_version: GOVERNOR_VERSION,
            supersedes,
            trigger_reason: "first composition for this role instance".into(),
            consequence_class: class,
        }
    }

    fn cite(digests: Vec<Hash>) -> EvidenceCitation {
        EvidenceCitation {
            candidate_tuple_digest: tuple().tuple_digest().unwrap(),
            composition_evidence_digests: digests,
        }
    }

    /// Round-trip: encode → decode → re-encode is byte-identical, with and without `supersedes`.
    #[test]
    fn a_record_round_trips_byte_identically_with_and_without_a_predecessor() {
        for supersedes in [None, Some(Hash([1u8; 32]))] {
            let class = if supersedes.is_some() {
                ConsequenceClass::ProfileCorrection
            } else {
                ConsequenceClass::InitialComposition
            };
            let r = record("trainer", 1, supersedes, class);
            let bytes = r.to_canonical_bytes().unwrap();
            let decoded: CompositionEvidenceRecord =
                daemon_vhc_proto::from_canonical_slice(&bytes).unwrap();
            assert_eq!(decoded, r);
            assert_eq!(decoded.to_canonical_bytes().unwrap(), bytes);
        }
    }

    /// A chain of three resolves to exactly one current record.
    #[test]
    fn a_supersedes_chain_resolves_to_exactly_one_current_record() {
        let mut ledger = CompositionEvidenceLedger::new();
        let first = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();

        let mut second_rec = record(
            "trainer",
            1,
            Some(first),
            ConsequenceClass::ProfileCorrection,
        );
        second_rec.trigger_reason = "profile re-issued after a workspace formula correction".into();
        let second = ledger.append(second_rec).unwrap();

        let mut third_rec = record(
            "trainer",
            1,
            Some(second),
            ConsequenceClass::CapabilityProbeFix,
        );
        third_rec.trigger_reason = "capability probe corrected; report regenerated".into();
        let third = ledger.append(third_rec).unwrap();

        assert_eq!(ledger.len(), 3);
        assert_eq!(ledger.resolve_current(&first).unwrap(), third);
        assert_eq!(ledger.resolve_current(&second).unwrap(), third);
        assert_eq!(ledger.resolve_current(&third).unwrap(), third);
        assert_eq!(ledger.standing(&first).unwrap(), RecordStanding::Superseded);
        assert_eq!(ledger.standing(&third).unwrap(), RecordStanding::Current);
        assert_eq!(
            ledger
                .current_for(&subject("trainer", 1))
                .unwrap()
                .supersedes,
            Some(second)
        );
    }

    #[test]
    fn a_valid_citation_of_a_current_record_passes() {
        let mut ledger = CompositionEvidenceLedger::new();
        let d = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        validate_citation(&cite(vec![d]), &tuple(), &ledger).expect("valid");
    }

    /// Citing a superseded record fails closed.
    #[test]
    fn citing_a_superseded_record_fails_closed() {
        let mut ledger = CompositionEvidenceLedger::new();
        let first = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        ledger
            .append(record(
                "trainer",
                1,
                Some(first),
                ConsequenceClass::PlannerFix,
            ))
            .unwrap();
        assert_eq!(
            validate_citation(&cite(vec![first]), &tuple(), &ledger),
            Err(EvidenceError::Superseded(first))
        );
    }

    /// Citing a revoked record fails closed.
    #[test]
    fn citing_a_revoked_record_fails_closed() {
        let mut ledger = CompositionEvidenceLedger::new();
        let d = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        ledger
            .revoke(d, "the profile it cites was found to understate staging")
            .unwrap();
        assert_eq!(
            validate_citation(&cite(vec![d]), &tuple(), &ledger),
            Err(EvidenceError::Revoked(d))
        );
        assert_eq!(ledger.standing(&d).unwrap(), RecordStanding::Revoked);
        // A revocation is an appended entry, not a deletion: the record is still retrievable.
        assert!(ledger.get(&d).is_some());
    }

    /// Citing a dangling digest fails closed.
    #[test]
    fn citing_a_dangling_digest_fails_closed() {
        let ledger = CompositionEvidenceLedger::new();
        let ghost = Hash([0xAB; 32]);
        assert_eq!(
            validate_citation(&cite(vec![ghost]), &tuple(), &ledger),
            Err(EvidenceError::Dangling(ghost))
        );
    }

    /// A broken chain fails closed: a record cannot supersede one the ledger does not hold, and a
    /// record already superseded cannot be superseded twice.
    #[test]
    fn a_broken_supersedes_chain_fails_closed() {
        let mut ledger = CompositionEvidenceLedger::new();
        let orphan = record(
            "trainer",
            1,
            Some(Hash([0xCD; 32])),
            ConsequenceClass::ProfileCorrection,
        );
        assert_eq!(
            ledger.append(orphan),
            Err(EvidenceError::Dangling(Hash([0xCD; 32])))
        );

        let first = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        ledger
            .append(record(
                "trainer",
                1,
                Some(first),
                ConsequenceClass::PlannerFix,
            ))
            .unwrap();
        let mut fork = record("trainer", 1, Some(first), ConsequenceClass::GovernorFix);
        fork.trigger_reason = "a second successor, which would fork the chain".into();
        assert!(matches!(
            ledger.append(fork),
            Err(EvidenceError::AppendOnly(_))
        ));
    }

    /// A planner identity mismatch between record and tuple fails closed.
    #[test]
    fn a_planner_identity_mismatch_fails_closed() {
        let mut ledger = CompositionEvidenceLedger::new();
        let mut r = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        r.planner_version = PLANNER_VERSION + 1;
        r.profile_compatible_planner_versions = [PLANNER_VERSION + 1].into_iter().collect();
        let d = ledger.append(r).unwrap();
        assert!(matches!(
            validate_citation(&cite(vec![d]), &tuple(), &ledger),
            Err(EvidenceError::CrossLayerMismatch {
                which: "planner version",
                ..
            })
        ));
    }

    /// A governor identity mismatch between record and tuple fails closed.
    #[test]
    fn a_governor_identity_mismatch_fails_closed() {
        let mut ledger = CompositionEvidenceLedger::new();
        let mut r = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        r.governor_version = GOVERNOR_VERSION + 1;
        let d = ledger.append(r).unwrap();
        assert!(matches!(
            validate_citation(&cite(vec![d]), &tuple(), &ledger),
            Err(EvidenceError::CrossLayerMismatch {
                which: "governor version",
                ..
            })
        ));
    }

    /// A profile whose named compatible planner versions exclude the composing planner is not
    /// composable evidence.
    #[test]
    fn a_profile_not_priced_for_the_composing_planner_is_refused() {
        let mut ledger = CompositionEvidenceLedger::new();
        let mut r = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        r.profile_compatible_planner_versions = [PLANNER_VERSION + 7].into_iter().collect();
        let d = ledger.append(r).unwrap();
        assert!(matches!(
            validate_citation(&cite(vec![d]), &tuple(), &ledger),
            Err(EvidenceError::ProfilePlannerIncompatible { .. })
        ));
    }

    /// An artifact citing only one of the two digests is rejected.
    #[test]
    fn an_artifact_citing_only_one_digest_is_rejected() {
        let mut ledger = CompositionEvidenceLedger::new();
        let d = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();

        let no_records = EvidenceCitation {
            candidate_tuple_digest: tuple().tuple_digest().unwrap(),
            composition_evidence_digests: vec![],
        };
        assert!(matches!(
            validate_citation(&no_records, &tuple(), &ledger),
            Err(EvidenceError::IncompleteCitation(_))
        ));

        let no_tuple = EvidenceCitation {
            candidate_tuple_digest: Hash([0u8; 32]),
            composition_evidence_digests: vec![d],
        };
        assert!(matches!(
            validate_citation(&no_tuple, &tuple(), &ledger),
            Err(EvidenceError::IncompleteCitation(_))
        ));
    }

    /// A multi-participant artifact cites one tuple digest and N record digests, and fails if any
    /// one of the N is superseded.
    #[test]
    fn a_multi_participant_citation_validates_and_fails_if_any_record_is_superseded() {
        let mut ledger = CompositionEvidenceLedger::new();
        let trainer = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        let seat = ledger
            .append(record(
                "seat",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();

        validate_citation(&cite(vec![trainer, seat]), &tuple(), &ledger)
            .expect("both current, both cited");

        ledger
            .append(record(
                "seat",
                1,
                Some(seat),
                ConsequenceClass::ProfileCorrection,
            ))
            .unwrap();
        assert_eq!(
            validate_citation(&cite(vec![trainer, seat]), &tuple(), &ledger),
            Err(EvidenceError::Superseded(seat)),
            "one superseded record fails the whole statement"
        );
    }

    /// The tuple never points forward. It carries no composition-record member at all, and adding
    /// one would invert the design: the join is made by the citing artifact, at the time it is
    /// written, and by nothing else.
    #[test]
    fn the_candidate_tuple_carries_no_reference_to_any_composition_record() {
        let encoded = tuple().to_canonical_bytes().unwrap();
        let text = String::from_utf8_lossy(&encoded).to_lowercase();
        for forward_reference in [
            "composition",
            "evidence",
            "supersedes",
            "record_digest",
            "profile_digest",
        ] {
            assert!(
                !text.contains(forward_reference),
                "the frozen tuple must not reference `{forward_reference}`"
            );
        }
    }

    /// The record binds the single reservation: its identity locates it, its digest proves what was
    /// charged, and the scope-separated components make a shared term distinguishable from a
    /// double-counted per-role one.
    #[test]
    fn the_record_binds_the_reservation_by_identity_digest_and_scope_separated_components() {
        assert_eq!(COMPOSITION_RECORD_MEMBERS, 12);

        let r = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        assert_eq!(r.reservation_identity.role, "trainer");
        assert_eq!(r.reservation_identity.incarnation, 1);
        assert_ne!(r.reservation_digest, Hash([0u8; 32]));

        let c = r.reservation_components;
        // Hidden overhead is its own visible component, not folded into the per-role total.
        assert!(c.hidden_overhead_bytes > 0);
        assert_ne!(c.per_role.total(), c.occupancy_bytes());
        // The two enforcement classes are separated, and together they account for the occupancy.
        assert_eq!(
            c.directly_enforceable_bytes() + c.profiled_and_measured_bytes(),
            c.occupancy_bytes()
        );
        // The per-allocation constraint is carried but is not occupancy.
        assert!(c.max_individual_allocation_bytes > 0);

        // Identity alone does not prove what was charged: a record whose components differ has a
        // different digest even at the same identity.
        let mut altered = r.clone();
        altered
            .reservation_components
            .per_role
            .directly_enforceable_bytes += 1;
        assert_eq!(altered.reservation_identity, r.reservation_identity);
        assert_ne!(altered.record_digest().unwrap(), r.record_digest().unwrap());
    }

    /// Layer 1 carries exactly the two identities whose `[PC-13]` row reaches it, and the full ABI
    /// rather than the major alone.
    #[test]
    fn layer_one_carries_the_identities_whose_consequence_class_reaches_it() {
        let t = tuple();
        assert_eq!(t.planner_version, PLANNER_VERSION);
        assert_eq!(t.governor_version, GOVERNOR_VERSION);
        assert_eq!(t.abi_minor, 5, "the minor is recorded, not just the major");
        assert_eq!(
            t.genesis_schema_major,
            daemon_vhc_proto::GENESIS_SCHEMA_MAJOR
        );

        assert!(ConsequenceClass::PlannerFix.reaches_candidate_layer());
        assert!(ConsequenceClass::GovernorFix.reaches_candidate_layer());
        for local in [
            ConsequenceClass::ProfileCorrection,
            ConsequenceClass::CapabilityProbeFix,
            ConsequenceClass::BackendImplementationRevision,
            ConsequenceClass::InitialComposition,
        ] {
            assert!(
                !local.reaches_candidate_layer(),
                "{local:?} is a layer-2 lifecycle event and must not force a battery by itself"
            );
        }
    }

    /// A record must say why it exists, and a recertification-triggered record must name what it
    /// replaces — otherwise the chain has no ancestry to resolve.
    #[test]
    fn a_record_must_state_its_trigger_and_its_predecessor_consistently() {
        let mut ledger = CompositionEvidenceLedger::new();
        let mut blank = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        blank.trigger_reason = "   ".into();
        assert!(matches!(
            ledger.append(blank),
            Err(EvidenceError::Invalid(_))
        ));

        let orphan_correction = record("trainer", 1, None, ConsequenceClass::ProfileCorrection);
        assert!(matches!(
            ledger.append(orphan_correction),
            Err(EvidenceError::Invalid(_))
        ));

        let initial_with_parent = record(
            "trainer",
            1,
            Some(Hash([1u8; 32])),
            ConsequenceClass::InitialComposition,
        );
        assert!(matches!(
            ledger.append(initial_with_parent),
            Err(EvidenceError::Invalid(_))
        ));
    }

    #[test]
    fn a_citation_against_a_different_candidate_is_refused() {
        let mut ledger = CompositionEvidenceLedger::new();
        let d = ledger
            .append(record(
                "trainer",
                1,
                None,
                ConsequenceClass::InitialComposition,
            ))
            .unwrap();
        let mut other = tuple();
        other.node_commit = "deadbeef".into();
        assert_eq!(
            validate_citation(&cite(vec![d]), &other, &ledger),
            Err(EvidenceError::TupleMismatch)
        );
    }

    #[test]
    fn a_record_cannot_be_appended_twice_and_a_ghost_cannot_be_revoked() {
        let mut ledger = CompositionEvidenceLedger::new();
        let r = record("trainer", 1, None, ConsequenceClass::InitialComposition);
        ledger.append(r.clone()).unwrap();
        assert!(matches!(
            ledger.append(r),
            Err(EvidenceError::AppendOnly(_))
        ));
        assert!(matches!(
            ledger.revoke(Hash([0xEE; 32]), "nope"),
            Err(EvidenceError::Dangling(_))
        ));
    }
}
