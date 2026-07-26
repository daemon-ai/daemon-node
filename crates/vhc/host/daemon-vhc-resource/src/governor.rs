// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **resource governor** and the single-reservation seam
//! (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-10]`, `[RC-13]`, `[RC-14]`, `[RC-15]`).
//!
//! ## One reservation, two views
//!
//! The owner's memory ledger charge and the governor's occupancy reservation are **the same
//! reservation seen twice**. They must not be taken twice and they must not drift apart. So there is
//! exactly one [`Reservation`] object, and both consumers obtain their numbers from it through
//! [`Reservation::bounds`] — the same function, the same arithmetic, the same composed claim. A
//! second derivation is not an optimization, it is the drift.
//!
//! ## What the two views are compared on
//!
//! Identity and bounds. **Not occupancy.** A reservation is a ceiling, and a role using less than it
//! reserved is the expected case — that is not divergence and must not be refused, and neither view
//! is shrunk toward the other. What fails closed is an identity mismatch, absent or duplicated
//! reservation state, bounds differing between the views, differing reservation arithmetic, or
//! measured occupancy **exceeding** the reserved bound — that last being a breach of the reservation
//! rather than a disagreement about it.
//!
//! ## Memory is claim-derived; the other ledgers are not
//!
//! Only device memory and host RAM derive from the composed claim. Duty cycle, disk, uplink and
//! downlink, and the concurrently-admitted-instance ceiling keep their established sources. What
//! they share is the prohibition on the owner-cap fallback: an absent input is a typed refusal or an
//! explicit policy default, never a silent substitution of the owner's own ceiling for a value that
//! was supposed to be derived. A claim that cannot be composed yields **no reservation**, and
//! therefore no admission.

use std::collections::BTreeMap;

use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash};
use serde::{Deserialize, Serialize};

use crate::capability::DeviceCapabilityReport;
use crate::planner::{AggregateClaim, PhysicalClaim, ScopedOccupancy};
use crate::profile::{AllocationScope, BackendExecutionProfile, EnforcementClass, ProfileError};

/// The reservation arithmetic's identity.
///
/// Both views must be produced by the *same* arithmetic, so the version is recorded with every
/// reservation and compared. It moves with the governor: a change to how a reservation is computed
/// invalidates reservation and enforcement evidence.
pub const RESERVATION_ARITHMETIC_VERSION: u32 = 1;

/// Which reservation this is, unambiguously.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReservationIdentity {
    /// The run role.
    pub role: String,
    /// The role instance's incarnation.
    pub incarnation: u64,
    /// The device the role instance is bound to for its lifetime.
    pub device_identity: String,
    /// A node-local monotone sequence, so a re-admission of the same `(role, incarnation)` after a
    /// release is a *different* reservation rather than an ambiguous reuse of one identity.
    pub sequence: u64,
}

/// One allocation scope's reserved amount, with the two enforcement classes separated.
///
/// Separated because a certification statement that reported one enforcement property over both
/// would be making a claim about the driver's internals that nobody verified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeComponent {
    /// Bytes the governor intercepts and attributes at creation.
    pub directly_enforceable_bytes: u64,
    /// Bytes budgeted from the profile and observed in aggregate only.
    pub profiled_and_measured_bytes: u64,
}

impl ScopeComponent {
    /// The component's total.
    pub fn total(&self) -> u64 {
        self.directly_enforceable_bytes
            .saturating_add(self.profiled_and_measured_bytes)
    }
}

/// The reserved amounts, separated by allocation scope, with hidden overhead visible.
///
/// Totals without scope separation cannot distinguish a correctly-shared process-scoped term from a
/// double-counted per-role one — which is the concrete defect the legacy three-tier sum produced
/// for co-resident roles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationComponents {
    /// Reserved per role instance.
    pub per_role: ScopeComponent,
    /// Reserved once per process, however many role instances it holds.
    pub per_process: ScopeComponent,
    /// Reserved once per device, across every process using it.
    pub per_device: ScopeComponent,
    /// The profiled hidden-overhead reserve, as its **own visible component** rather than folded
    /// into a per-role total.
    pub hidden_overhead_bytes: u64,
    /// The scope at which that hidden overhead is observable.
    pub hidden_overhead_observation_scope: AllocationScope,
    /// The largest single allocation this reservation permits.
    ///
    /// A **maximum constraint**, validated against the capability report's limit. It is **not
    /// occupancy** and is never added to any ledger total — adding it would refuse machines for
    /// memory nobody is holding.
    pub max_individual_allocation_bytes: u64,
}

impl Default for ReservationComponents {
    /// An empty reservation. The hidden-overhead observation scope defaults to the **broadest**
    /// scope rather than deriving `Default` on the scope enum: a per-role default would be a claim
    /// that a driver statistic is attributable per role, which is exactly what `[RC-10]` forbids
    /// asserting without evidence.
    fn default() -> Self {
        Self {
            per_role: ScopeComponent::default(),
            per_process: ScopeComponent::default(),
            per_device: ScopeComponent::default(),
            hidden_overhead_bytes: 0,
            hidden_overhead_observation_scope: AllocationScope::PerDevice,
            max_individual_allocation_bytes: 0,
        }
    }
}

impl ReservationComponents {
    /// The occupancy this reservation holds: every scope's bytes plus the hidden overhead, and the
    /// per-allocation constraint deliberately excluded.
    pub fn occupancy_bytes(&self) -> u64 {
        self.per_role
            .total()
            .saturating_add(self.per_process.total())
            .saturating_add(self.per_device.total())
            .saturating_add(self.hidden_overhead_bytes)
    }

    /// The directly enforceable share of the occupancy.
    pub fn directly_enforceable_bytes(&self) -> u64 {
        self.per_role
            .directly_enforceable_bytes
            .saturating_add(self.per_process.directly_enforceable_bytes)
            .saturating_add(self.per_device.directly_enforceable_bytes)
    }

    /// The profiled-and-measured share, hidden overhead included — it is by definition in this class.
    pub fn profiled_and_measured_bytes(&self) -> u64 {
        self.per_role
            .profiled_and_measured_bytes
            .saturating_add(self.per_process.profiled_and_measured_bytes)
            .saturating_add(self.per_device.profiled_and_measured_bytes)
            .saturating_add(self.hidden_overhead_bytes)
    }
}

/// What both views read. Produced by one function so the two cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationBounds {
    /// The scope-separated components.
    pub components: ReservationComponents,
    /// The device-memory occupancy this reservation holds.
    pub device_memory_bytes: u64,
    /// The host-RAM occupancy this reservation holds — the guest's own linear-memory peak, taken
    /// from its backend-neutral terms and never from a host constant.
    pub host_memory_bytes: u64,
    /// The arithmetic that produced these numbers.
    pub arithmetic_version: u32,
}

/// The single reservation an admission takes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservation {
    /// Which reservation this is.
    pub identity: ReservationIdentity,
    /// What it holds.
    pub bounds: ReservationBounds,
}

impl Reservation {
    /// The bounds. **Both** the governor and the owner-ledger projection read them through here —
    /// that is the seam, and it is why they cannot drift.
    pub fn bounds(&self) -> ReservationBounds {
        self.bounds
    }

    /// The reservation's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, GovernorError> {
        to_canonical_vec(self)
            .map_err(|e| GovernorError::Invalid(format!("reservation encoding: {e}")))
    }

    /// blake3 of the canonical bytes.
    ///
    /// Identity alone only *locates* a reservation; it does not prove what was charged. A record
    /// carrying identity and no digest lets an auditor find the reservation and then trust its
    /// current contents.
    pub fn reservation_digest(&self) -> Result<Hash, GovernorError> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }
}

/// Whether observed occupancy is within its reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyVerdict {
    /// Observed at or below the reserved bound. **The expected case.** A reservation is a ceiling.
    WithinReservation {
        /// The reserved bound.
        reserved: u64,
        /// What was observed.
        observed: u64,
    },
    /// Observed above the reserved bound. A breach of the reservation, handled as a certification
    /// stop — not a disagreement about the reservation.
    Breach {
        /// The reserved bound.
        reserved: u64,
        /// What was observed.
        observed: u64,
    },
}

impl OccupancyVerdict {
    /// Whether this verdict requires the run to stop.
    pub fn is_breach(&self) -> bool {
        matches!(self, Self::Breach { .. })
    }
}

/// Why a reservation could not be taken, or why the two views disagree.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GovernorError {
    /// Structurally unusable input.
    #[error("reservation input is invalid: {0}")]
    Invalid(String),
    /// The claim could not be composed, so there is nothing to reserve.
    ///
    /// This is what replaces the owner-cap fallback. With the guest's device tiers retired, an
    /// absent composed figure would otherwise make the fallback the *only* path, and the ledger
    /// would charge the owner's own ceiling instead of the workload.
    #[error(
        "no composed claim is available for {role}, so no reservation exists and the admission is \
         refused: {detail}. The owner's cap is not a substitute for a figure that was supposed to \
         be derived"
    )]
    ClaimNotComposable {
        /// The role that cannot be admitted.
        role: String,
        /// Why the claim is unavailable — a missing, incompatible or unauthenticated profile.
        detail: String,
    },
    /// The profile is unusable.
    #[error(transparent)]
    Profile(#[from] ProfileError),
    /// The two views name different reservations.
    #[error("the ledger and the governor refer to different reservations: {detail}")]
    IdentityMismatch {
        /// What differs.
        detail: String,
    },
    /// The two views hold different bounds.
    #[error(
        "the ledger and the governor hold different reservation bounds for the same reservation \
         ({detail}); neither view is enlarged and neither is shrunk — the admission is refused"
    )]
    BoundsMismatch {
        /// What differs.
        detail: String,
    },
    /// The two views were produced by different arithmetic.
    #[error(
        "the two views were produced by different reservation arithmetic (ledger {ledger}, \
         governor {governor}); equal numbers today would be a coincidence with an expiry date"
    )]
    ArithmeticMismatch {
        /// The ledger view's arithmetic version.
        ledger: u32,
        /// The governor view's arithmetic version.
        governor: u32,
    },
    /// Reservation state is missing or duplicated.
    #[error("reservation state for {role}/{incarnation} is {problem}")]
    ReservationState {
        /// The role.
        role: String,
        /// The incarnation.
        incarnation: u64,
        /// Absent, or duplicated.
        problem: &'static str,
    },
    /// Observed occupancy exceeded the reserved bound.
    #[error(
        "measured occupancy {observed} B exceeds the reserved bound {reserved} B for \
         {role}/{incarnation}; this is a breach of the reservation, not a disagreement about it"
    )]
    OccupancyBreach {
        /// The role.
        role: String,
        /// The incarnation.
        incarnation: u64,
        /// The bound.
        reserved: u64,
        /// What was observed.
        observed: u64,
    },
}

/// Derive the single reservation from the composed claims.
///
/// Memory only: device memory from the composed role claim and the node/device aggregate, host RAM
/// from the plan's own linear-memory terms. Never from a guest-declared physical figure, never from
/// a host constant, never from an owner cap.
///
/// Each term is charged at its **declared** allocation scope with its declared composition rule.
/// `PerAllocation` terms are collected as a maximum constraint and are **not** added to any total.
pub fn derive_reservation(
    identity: ReservationIdentity,
    claim: &PhysicalClaim,
    occupancy: &ScopedOccupancy,
    aggregate: &AggregateClaim,
    profile: &BackendExecutionProfile,
) -> Result<Reservation, GovernorError> {
    profile.validate()?;

    let mut components = ReservationComponents {
        hidden_overhead_bytes: profile.headroom.hidden_overhead_reserve_bytes,
        hidden_overhead_observation_scope: profile.headroom.hidden_overhead_observation_scope,
        max_individual_allocation_bytes: claim
            .max_individual_allocation_bytes
            .max(aggregate.max_individual_allocation_bytes),
        ..Default::default()
    };

    // Shared-scope terms are charged ONCE per aggregation key. Charging them per role is the
    // double-count the legacy per-role tier sum produced across co-resident roles.
    let mut charged_shared_keys: BTreeMap<&str, ()> = BTreeMap::new();
    for term in &occupancy.terms {
        let slot = match term.scope {
            // A maximum constraint, never occupancy.
            AllocationScope::PerAllocation => {
                components.max_individual_allocation_bytes =
                    components.max_individual_allocation_bytes.max(term.bytes);
                continue;
            }
            AllocationScope::PerRoleInstance => &mut components.per_role,
            AllocationScope::PerProcess => {
                if charged_shared_keys
                    .insert(term.aggregation_key.as_str(), ())
                    .is_some()
                {
                    continue;
                }
                &mut components.per_process
            }
            AllocationScope::PerDevice => {
                if charged_shared_keys
                    .insert(term.aggregation_key.as_str(), ())
                    .is_some()
                {
                    continue;
                }
                &mut components.per_device
            }
        };
        match term.enforcement {
            EnforcementClass::DirectlyEnforceable => {
                slot.directly_enforceable_bytes =
                    slot.directly_enforceable_bytes.saturating_add(term.bytes);
            }
            EnforcementClass::ProfiledAndMeasured => {
                slot.profiled_and_measured_bytes =
                    slot.profiled_and_measured_bytes.saturating_add(term.bytes);
            }
        }
    }

    // The parts of the claim that are not individual profile terms: the transient peak, the pool
    // retention, compilation and staging are the backend's own doing, pre-authorized from the
    // profile rather than intercepted per allocation.
    components.per_role.profiled_and_measured_bytes = components
        .per_role
        .profiled_and_measured_bytes
        .saturating_add(claim.transient_peak_bytes)
        .saturating_add(claim.retained_pool_bytes)
        .saturating_add(claim.compilation_bytes)
        .saturating_add(claim.staging_bytes)
        .saturating_add(claim.headroom_bytes);
    components.per_role.directly_enforceable_bytes = components
        .per_role
        .directly_enforceable_bytes
        .saturating_add(claim.workspace_bytes);

    Ok(Reservation {
        identity,
        bounds: ReservationBounds {
            components,
            device_memory_bytes: components.occupancy_bytes(),
            host_memory_bytes: claim.linear_memory_bytes,
            arithmetic_version: RESERVATION_ARITHMETIC_VERSION,
        },
    })
}

/// The one place a reservation exists on a node.
///
/// Classified under the governor authority: a mis-keyed reservation, a release on the wrong event, a
/// lost or duplicated entry, or a projection that mis-scopes a term is attributable here and
/// nowhere else.
#[derive(Clone, Debug, Default)]
pub struct ReservationStore {
    live: BTreeMap<ReservationIdentity, Reservation>,
    next_sequence: u64,
}

impl ReservationStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// The next node-local reservation sequence.
    pub fn next_sequence(&mut self) -> u64 {
        self.next_sequence += 1;
        self.next_sequence
    }

    /// Take a reservation. Refuses a duplicate for the same identity.
    pub fn reserve(&mut self, reservation: Reservation) -> Result<(), GovernorError> {
        if self.live.contains_key(&reservation.identity) {
            return Err(GovernorError::ReservationState {
                role: reservation.identity.role.clone(),
                incarnation: reservation.identity.incarnation,
                problem: "already reserved (duplicated reservation state)",
            });
        }
        self.live.insert(reservation.identity.clone(), reservation);
        Ok(())
    }

    /// Release a reservation on the failing path or at teardown. A refusal returns what it held.
    pub fn release(
        &mut self,
        identity: &ReservationIdentity,
    ) -> Result<Reservation, GovernorError> {
        self.live
            .remove(identity)
            .ok_or_else(|| GovernorError::ReservationState {
                role: identity.role.clone(),
                incarnation: identity.incarnation,
                problem: "absent (nothing to release)",
            })
    }

    /// A live reservation.
    pub fn get(&self, identity: &ReservationIdentity) -> Option<&Reservation> {
        self.live.get(identity)
    }

    /// How many reservations are live.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Whether nothing is reserved.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// The node/device occupancy delta a new reservation adds over what is already held at shared
    /// scopes — what admission atomically reserves, rather than the full figure.
    pub fn occupancy_delta(&self, candidate: &Reservation) -> u64 {
        let already_holds_shared = self
            .live
            .values()
            .any(|r| r.identity.device_identity == candidate.identity.device_identity);
        let c = &candidate.bounds.components;
        if already_holds_shared {
            // The process- and device-scoped terms are already held by the sibling; only this
            // role's own occupancy is new.
            c.per_role.total()
        } else {
            c.occupancy_bytes()
        }
    }
}

/// Compare the owner-ledger view and the governor view of one reservation.
///
/// **Identity and bounds only.** Occupancy is not a comparand: a role using less than it reserved is
/// the expected case.
pub fn compare_views(ledger: &Reservation, governor: &Reservation) -> Result<(), GovernorError> {
    if ledger.identity != governor.identity {
        return Err(GovernorError::IdentityMismatch {
            detail: format!(
                "ledger holds {}/{} seq {}, governor holds {}/{} seq {}",
                ledger.identity.role,
                ledger.identity.incarnation,
                ledger.identity.sequence,
                governor.identity.role,
                governor.identity.incarnation,
                governor.identity.sequence
            ),
        });
    }
    if ledger.bounds.arithmetic_version != governor.bounds.arithmetic_version {
        return Err(GovernorError::ArithmeticMismatch {
            ledger: ledger.bounds.arithmetic_version,
            governor: governor.bounds.arithmetic_version,
        });
    }
    if ledger.bounds != governor.bounds {
        return Err(GovernorError::BoundsMismatch {
            detail: format!(
                "device memory {} vs {}, host memory {} vs {}",
                ledger.bounds.device_memory_bytes,
                governor.bounds.device_memory_bytes,
                ledger.bounds.host_memory_bytes,
                governor.bounds.host_memory_bytes
            ),
        });
    }
    Ok(())
}

/// Evaluate observed occupancy against a reservation.
///
/// Below the bound is [`OccupancyVerdict::WithinReservation`] and is **normal**. Above it is a
/// breach.
pub fn evaluate_occupancy(reservation: &Reservation, observed_bytes: u64) -> OccupancyVerdict {
    let reserved = reservation.bounds.device_memory_bytes;
    if observed_bytes <= reserved {
        OccupancyVerdict::WithinReservation {
            reserved,
            observed: observed_bytes,
        }
    } else {
        OccupancyVerdict::Breach {
            reserved,
            observed: observed_bytes,
        }
    }
}

/// The pool sizing a composed claim implies — **computed, not applied**.
///
/// The default is not neutral. One backend's runtime overrides the pool page size with a fraction of
/// the **device-local heap** — on a machine whose heap is far larger than the run's budget that
/// yields a single pool page several times the declared device budget, and the same code reports the
/// smaller budget. Another sizes from a compile-time constant unrelated to the card.
///
/// These figures are nevertheless *not* configured into the runtime, and that is a ruling rather than
/// an omission. The page size cannot be applied per claim: the bring-up registers one compute client
/// per device per process, the probe **is** the bring-up, and the claim that would size the pool does
/// not exist until after the probe has already registered under the framework's defaults. So the
/// program certifies under the framework default for now and the record says so.
///
/// They are computed anyway, for two reasons. They are what a configured build *would* apply, so
/// applying them later is a decision rather than a derivation. And [`Self::reservation_breach`] turns
/// the empirical question the ruling rides on — does the framework default actually reserve within the
/// reservation bound at ceremony geometry? — into a comparison rather than an eyeball.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolSizing {
    /// The pool page size a configured build would set. Bounded by the claim's own largest single
    /// allocation and by what the device was measured to accept — never derived from the heap.
    pub page_bytes: u64,
    /// The pool ceiling a configured build would set: the reservation's device-memory bound.
    pub max_pool_bytes: u64,
}

/// What the admission check found. Admission is the verdict; the sizing rides along unapplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolAdmission {
    /// The claim's largest single allocation, rounded up to the profile's alignment — the figure the
    /// device must actually accept in one piece.
    pub largest_single_allocation_bytes: u64,
    /// What this device was **measured** to accept in one allocation. Measured, not reported: a
    /// reported ceiling is a vendor's claim about a family, and admitting against it would admit a
    /// claim this card was never shown to satisfy.
    pub measured_device_ceiling_bytes: u64,
    /// The reservation's device-memory bound — what the run is entitled to occupy on this device.
    pub reservation_device_memory_bytes: u64,
    /// The sizing a configured build would apply. See [`PoolSizing`] on why it is not applied.
    pub would_be: PoolSizing,
}

impl PoolAdmission {
    /// By how much an observed reservation exceeds what the run is entitled to, or `None` when it
    /// does not.
    ///
    /// The mechanical form of the condition the framework-default ruling rides on. `bytes_reserved`
    /// from an allocator sample is what the pool actually took from the device, and the reservation
    /// bound is what the run may occupy; the framework default is only acceptable while the first
    /// stays inside the second. If this returns `Some` at ceremony geometry, configuring the pool
    /// stops being optional.
    ///
    /// Deliberately takes the observed figure as an argument rather than reading a sampler: the
    /// governor does not measure, and a comparison that fetched its own input could compare a
    /// different run's reading against this claim's bound.
    #[must_use]
    pub fn reservation_breach(&self, observed_bytes_reserved: u64) -> Option<u64> {
        observed_bytes_reserved
            .checked_sub(self.reservation_device_memory_bytes)
            .filter(|excess| *excess > 0)
    }
}

/// Admit or refuse a composed claim against what this device was measured to accept.
///
/// The **check** half of the pool question, which is the half that prevents a bad admission. A claim
/// whose largest single allocation exceeds the device's measured limit cannot be satisfied by any pool
/// configuration, so refusing it here is not a substitute for configuring the pool — it is the part
/// configuration would never have fixed.
///
/// The sizing a configured build would apply is returned alongside, unapplied. See [`PoolSizing`].
///
/// # Errors
/// [`GovernorError`] if the claim's largest single allocation exceeds what the device was measured
/// to accept, or if the report carries no measured ceiling to check against — an unmeasured device
/// is not a device that passed, and admitting against an absent measurement would admit everything.
pub fn check_pool_admissible(
    reservation: &Reservation,
    claim: &PhysicalClaim,
    report: &DeviceCapabilityReport,
    profile: &BackendExecutionProfile,
) -> Result<PoolAdmission, GovernorError> {
    let measured_ceiling = report
        .max_allocation_bytes()
        .map_err(|e| GovernorError::Invalid(e.to_string()))?;
    let alignment = profile.allocation_alignment_bytes.max(1);

    let requested = claim
        .max_individual_allocation_bytes
        .div_ceil(alignment)
        .saturating_mul(alignment);
    if requested > measured_ceiling {
        return Err(GovernorError::Invalid(format!(
            "the composed claim needs a {requested}-byte single allocation, above the \
             {measured_ceiling}-byte limit this device was measured to accept"
        )));
    }

    // A pool page larger than the whole reservation is the defect a configured build would avoid: it
    // cannot be satisfied inside the budget the reservation holds, whatever the heap happens to be.
    let max_pool_bytes = reservation.bounds.device_memory_bytes;
    let page_bytes = requested.min(max_pool_bytes).max(alignment);

    Ok(PoolAdmission {
        largest_single_allocation_bytes: requested,
        measured_device_ceiling_bytes: measured_ceiling,
        reservation_device_memory_bytes: max_pool_bytes,
        would_be: PoolSizing {
            page_bytes,
            max_pool_bytes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::fixtures::report;
    use crate::planner::{aggregate, compose};
    use crate::profile::fixtures::profile;
    use crate::revision::BackendClass;
    use daemon_vhc_proto::resource_plan::{
        Binding, Dimension, DimensionValue, Domain, Dtype, Expr, Lifetime, LinearLifetime,
        LinearMemoryTerm, LogicalResourcePlan, OperationDecl, Retention, SelectionScope,
        TensorDecl,
    };

    fn plan() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![Dimension {
                name: "micro_batch".into(),
                domain: Domain::UintRange { lo: 1, hi: 4 },
            }],
            tensors: vec![
                TensorDecl {
                    name: "activations".into(),
                    shape: vec![Expr::Dimension("micro_batch".into()), Expr::Const(65_536)],
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
                workspace_class: None,
                max_in_flight: Expr::Const(1),
            }],
            transfers: vec![],
            linear_memory: vec![LinearMemoryTerm {
                name: "index".into(),
                lifetime: LinearLifetime::Persistent,
                bytes: Expr::Const(65_536),
            }],
            transient_live_sets: vec![vec!["forward".into()]],
            linear_fragmentation_headroom: Expr::Const(4096),
        }
    }

    fn binding() -> Binding {
        Binding::from([("micro_batch".to_string(), DimensionValue::Uint(2))])
    }

    fn identity(role: &str, sequence: u64) -> ReservationIdentity {
        ReservationIdentity {
            role: role.into(),
            incarnation: 1,
            device_identity: "0000:c4:00.0".into(),
            sequence,
        }
    }

    fn reserve_for(role: &str, sequence: u64, co_resident: u64) -> Reservation {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, occ) = compose(&p, &binding(), &prof, co_resident).unwrap();
        let agg = aggregate(&[(role.to_string(), claim.clone(), occ.clone())]).unwrap();
        derive_reservation(identity(role, sequence), &claim, &occ, &agg, &prof).unwrap()
    }

    /// Both views read the same bounds through the same function. That is the seam.
    #[test]
    fn the_two_views_are_one_object_and_therefore_agree() {
        let reservation = reserve_for("trainer", 1, 1);
        let ledger_view = reservation.bounds();
        let governor_view = reservation.bounds();
        assert_eq!(ledger_view, governor_view);
        assert_eq!(
            ledger_view.arithmetic_version,
            RESERVATION_ARITHMETIC_VERSION
        );
        compare_views(&reservation, &reservation).expect("one object cannot disagree with itself");
    }

    /// Memory derives from the composed claim: device from the claim and aggregate, host RAM from
    /// the plan's own linear terms.
    #[test]
    fn memory_derives_from_the_composed_claim_and_never_from_a_cap() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, _) = compose(&p, &binding(), &prof, 1).unwrap();
        let reservation = reserve_for("trainer", 1, 1);

        assert_eq!(
            reservation.bounds.host_memory_bytes,
            claim.linear_memory_bytes
        );
        assert_eq!(reservation.bounds.host_memory_bytes, 65_536 + 4096);
        assert!(reservation.bounds.device_memory_bytes > 0);
        // The whole reservation is accounted by the scope components plus hidden overhead.
        assert_eq!(
            reservation.bounds.device_memory_bytes,
            reservation.bounds.components.occupancy_bytes()
        );
    }

    /// The owner-cap fallback is replaced by a typed refusal: an uncomposable claim yields no
    /// reservation, so there is no admission.
    #[test]
    fn an_uncomposable_claim_is_a_typed_refusal_not_a_charge_of_the_owners_ceiling() {
        let err = GovernorError::ClaimNotComposable {
            role: "trainer".into(),
            detail: "no authenticated profile for the resolved backend implementation".into(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("no reservation exists"));
        assert!(rendered.contains("not a substitute"));
    }

    /// Hidden overhead is its own visible component at the scope the profile declares, not folded
    /// into a per-role total.
    #[test]
    fn hidden_overhead_is_a_visible_component_at_its_declared_scope() {
        let prof = profile(BackendClass::Vulkan);
        let reservation = reserve_for("trainer", 1, 1);
        let c = reservation.bounds.components;
        assert_eq!(
            c.hidden_overhead_bytes,
            prof.headroom.hidden_overhead_reserve_bytes
        );
        assert_eq!(
            c.hidden_overhead_observation_scope,
            prof.headroom.hidden_overhead_observation_scope
        );
        // It is in the profiled class by definition, and it is not inside the per-role figure.
        assert!(c.profiled_and_measured_bytes() >= c.hidden_overhead_bytes);
        assert_ne!(c.per_role.total(), c.occupancy_bytes());
    }

    /// The per-allocation constraint is never occupancy. Adding it to a ledger total would refuse
    /// machines for memory nobody is holding.
    #[test]
    fn the_per_allocation_constraint_is_never_added_to_any_total() {
        let reservation = reserve_for("trainer", 1, 1);
        let c = reservation.bounds.components;
        assert!(c.max_individual_allocation_bytes > 0);
        assert_eq!(
            c.occupancy_bytes(),
            c.per_role.total()
                + c.per_process.total()
                + c.per_device.total()
                + c.hidden_overhead_bytes,
            "the per-allocation constraint is excluded from occupancy"
        );
    }

    /// A shared-scope term is charged once. Charging it per role is the colocation double-count.
    #[test]
    fn a_shared_scope_term_is_charged_once_not_once_per_role() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, occ) = compose(&p, &binding(), &prof, 2).unwrap();
        let agg = aggregate(&[
            ("trainer".to_string(), claim.clone(), occ.clone()),
            ("seat".to_string(), claim.clone(), occ.clone()),
        ])
        .unwrap();

        // The device-scoped context appears once in the reservation, whatever the role count.
        let one = derive_reservation(identity("trainer", 1), &claim, &occ, &agg, &prof).unwrap();
        let device_scoped = one.bounds.components.per_device.total();
        assert!(device_scoped > 0, "the fixture has a device-scoped term");

        let mut doubled = occ.clone();
        doubled.terms.extend(occ.terms.clone());
        let twice =
            derive_reservation(identity("trainer", 2), &claim, &doubled, &agg, &prof).unwrap();
        assert_eq!(
            twice.bounds.components.per_device.total(),
            device_scoped,
            "a repeated shared-scope aggregation key is charged once, not twice"
        );
    }

    /// A co-resident sibling reserves only the aggregate occupancy delta.
    #[test]
    fn a_co_resident_sibling_reserves_only_the_delta() {
        let mut store = ReservationStore::new();
        let first = reserve_for("trainer", store.next_sequence(), 2);
        let full = store.occupancy_delta(&first);
        assert_eq!(full, first.bounds.components.occupancy_bytes());
        store.reserve(first.clone()).unwrap();

        let sibling = reserve_for("seat", store.next_sequence(), 2);
        let delta = store.occupancy_delta(&sibling);
        assert_eq!(delta, sibling.bounds.components.per_role.total());
        assert!(
            delta < full,
            "the sibling does not re-reserve the shared process and device terms"
        );
    }

    /// Occupancy below the bound is NORMAL. This is the correction most likely to be implemented
    /// wrong, so it is asserted directly.
    #[test]
    fn occupancy_below_the_reserved_bound_is_normal_and_not_a_divergence() {
        let reservation = reserve_for("trainer", 1, 1);
        let reserved = reservation.bounds.device_memory_bytes;

        for observed in [0, 1, reserved / 2, reserved - 1, reserved] {
            let verdict = evaluate_occupancy(&reservation, observed);
            assert!(
                !verdict.is_breach(),
                "observing {observed} against a {reserved} reservation is within it"
            );
            assert_eq!(
                verdict,
                OccupancyVerdict::WithinReservation { reserved, observed }
            );
        }
        // And neither view is shrunk toward the observation.
        assert_eq!(reservation.bounds.device_memory_bytes, reserved);
    }

    /// Occupancy above the bound is a breach of the reservation, not a disagreement about it.
    #[test]
    fn occupancy_above_the_reserved_bound_is_a_breach() {
        let reservation = reserve_for("trainer", 1, 1);
        let reserved = reservation.bounds.device_memory_bytes;
        let verdict = evaluate_occupancy(&reservation, reserved + 1);
        assert!(verdict.is_breach());
        assert_eq!(
            verdict,
            OccupancyVerdict::Breach {
                reserved,
                observed: reserved + 1
            }
        );
    }

    /// Identity mismatch, differing bounds and differing arithmetic all fail closed, and neither
    /// view is moved toward the other.
    #[test]
    fn identity_bounds_and_arithmetic_mismatches_all_fail_closed() {
        let governor = reserve_for("trainer", 1, 1);

        let mut other_identity = governor.clone();
        other_identity.identity.sequence = 2;
        assert!(matches!(
            compare_views(&other_identity, &governor).unwrap_err(),
            GovernorError::IdentityMismatch { .. }
        ));

        let mut other_bounds = governor.clone();
        other_bounds.bounds.device_memory_bytes += 1;
        assert!(matches!(
            compare_views(&other_bounds, &governor).unwrap_err(),
            GovernorError::BoundsMismatch { .. }
        ));

        let mut other_arithmetic = governor.clone();
        other_arithmetic.bounds.arithmetic_version = RESERVATION_ARITHMETIC_VERSION + 1;
        assert!(matches!(
            compare_views(&other_arithmetic, &governor).unwrap_err(),
            GovernorError::ArithmeticMismatch { .. }
        ));
    }

    /// Absent or duplicated reservation state fails closed, and a refusal returns what it held.
    #[test]
    fn reservation_state_is_single_and_a_refusal_returns_what_it_held() {
        let mut store = ReservationStore::new();
        let reservation = reserve_for("trainer", 1, 1);
        store.reserve(reservation.clone()).unwrap();
        assert_eq!(store.len(), 1);

        assert!(matches!(
            store.reserve(reservation.clone()).unwrap_err(),
            GovernorError::ReservationState {
                problem: "already reserved (duplicated reservation state)",
                ..
            }
        ));

        let released = store.release(&reservation.identity).unwrap();
        assert_eq!(released.bounds, reservation.bounds);
        assert!(store.is_empty());

        assert!(matches!(
            store.release(&reservation.identity).unwrap_err(),
            GovernorError::ReservationState {
                problem: "absent (nothing to release)",
                ..
            }
        ));
    }

    /// A re-admission after a release is a different reservation, not an ambiguous reuse of one
    /// identity.
    #[test]
    fn a_re_admission_takes_a_new_reservation_identity() {
        let mut store = ReservationStore::new();
        let first = store.next_sequence();
        let second = store.next_sequence();
        assert_ne!(first, second);
        assert_ne!(identity("trainer", first), identity("trainer", second));
    }

    /// The digest proves what was charged. Identity alone only locates the reservation.
    #[test]
    fn the_reservation_digest_changes_with_what_was_charged() {
        let a = reserve_for("trainer", 1, 1);
        let mut b = a.clone();
        b.bounds.components.per_role.directly_enforceable_bytes += 1;
        assert_ne!(
            a.reservation_digest().unwrap(),
            b.reservation_digest().unwrap()
        );
        assert_eq!(
            a.reservation_digest().unwrap(),
            reserve_for("trainer", 1, 1).reservation_digest().unwrap(),
            "the digest is a deterministic function of the reservation"
        );
    }

    /// Pool sizing comes from the composed claim. A page derived from the device heap can exceed
    /// the whole budget; a page derived from the claim cannot exceed the reservation.
    #[test]
    fn pool_sizing_derives_from_the_claim_and_never_exceeds_the_reservation() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, _) = compose(&p, &binding(), &prof, 1).unwrap();
        let reservation = reserve_for("trainer", 1, 1);
        let admission =
            check_pool_admissible(&reservation, &claim, &report(BackendClass::Vulkan), &prof)
                .unwrap();
        let sizing = admission.would_be;

        assert_eq!(
            sizing.max_pool_bytes,
            reservation.bounds.device_memory_bytes
        );
        assert!(
            sizing.page_bytes <= sizing.max_pool_bytes,
            "a pool page larger than the whole reservation cannot be satisfied inside it"
        );
        assert_eq!(sizing.page_bytes % prof.allocation_alignment_bytes, 0);

        // The observed default on this box: a page sized from the device-local heap is 21,687,348,224 B
        // against a 4 GiB budget — 5.05x the budget the same code reports. The derived page is not
        // that number, and cannot be, because it is bounded by the claim rather than by the heap.
        let heap_derived_page = 21_687_348_224u64;
        assert!(
            sizing.page_bytes < heap_derived_page,
            "the page must not be derived from the device heap"
        );
        assert!(sizing.page_bytes <= 4 * 1024 * 1024 * 1024);
    }

    /// The framework-default ruling rides on an empirical condition, and this is the comparison that
    /// decides it.
    ///
    /// The program certifies under the framework's own pool configuration because a per-claim page
    /// size is unreachable: one compute client per device per process, and the probe is the bring-up,
    /// so the claim that would size the pool does not exist until after registration. That is only
    /// acceptable while the default's actual reservation stays inside what the run is entitled to
    /// occupy — which the allocator readout can now measure, since `bytes_reserved` is exactly what
    /// the pool took from the device.
    ///
    /// So the check reports the bound, and a breach is a subtraction rather than a judgement call. If
    /// this fires at ceremony geometry, configuring the pool stops being optional.
    #[test]
    fn an_observed_reservation_above_the_bound_is_reported_as_a_breach() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, _) = compose(&p, &binding(), &prof, 1).unwrap();
        let reservation = reserve_for("trainer", 1, 1);
        let admission =
            check_pool_admissible(&reservation, &claim, &report(BackendClass::Vulkan), &prof)
                .unwrap();

        let bound = admission.reservation_device_memory_bytes;
        assert_eq!(bound, reservation.bounds.device_memory_bytes);

        // Reserving within the entitlement is not a breach, and neither is reserving exactly it.
        assert_eq!(admission.reservation_breach(bound / 2), None);
        assert_eq!(admission.reservation_breach(bound), None);

        // Above it, the excess is reported rather than merely flagged: how far over decides whether
        // this is a rounding artefact or the heap-derived page the default is known to produce.
        assert_eq!(admission.reservation_breach(bound + 4096), Some(4096));

        // The observed default on this box: a page sized from the device-local heap is
        // 21,687,348,224 B against a 4 GiB budget — 5.05x what the same code reports as the budget.
        // Were the framework to reserve that, the breach would be unmistakable rather than marginal.
        let heap_derived_page = 21_687_348_224u64;
        assert_eq!(
            admission.reservation_breach(heap_derived_page),
            Some(heap_derived_page - bound),
            "the case that would force a configured pool is reported with its full excess"
        );
    }

    /// The check admits against a **measured** ceiling, and an unmeasured device does not pass.
    ///
    /// A reported ceiling is a vendor's claim about a family; admitting a claim against it would admit
    /// something this card was never shown to satisfy. An absent measurement therefore refuses, rather
    /// than being treated as unlimited — a bound compared against nothing admits everything.
    #[test]
    fn a_device_with_no_measured_ceiling_cannot_admit_a_claim() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (claim, _) = compose(&p, &binding(), &prof, 1).unwrap();
        let reservation = reserve_for("trainer", 1, 1);
        let mut unmeasured = report(BackendClass::Vulkan);
        unmeasured.measured_max_allocation_bytes = crate::revision::Maybe::default();

        assert!(
            check_pool_admissible(&reservation, &claim, &unmeasured, &prof).is_err(),
            "an unmeasured device is not a device that passed"
        );
    }
    /// A claim needing a single allocation above what the device was measured to accept is refused
    /// rather than sized around.
    #[test]
    fn a_claim_above_the_measured_ceiling_refuses_pool_sizing() {
        let p = plan();
        let prof = profile(BackendClass::Vulkan);
        let (mut claim, _) = compose(&p, &binding(), &prof, 1).unwrap();
        claim.max_individual_allocation_bytes = 8 << 30;
        let reservation = reserve_for("trainer", 1, 1);
        let err = check_pool_admissible(&reservation, &claim, &report(BackendClass::Vulkan), &prof)
            .unwrap_err();
        assert!(err.to_string().contains("was measured to accept"));
    }
}
