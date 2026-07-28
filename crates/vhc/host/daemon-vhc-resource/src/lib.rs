// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-resource` — the **host** side of the three-object resource model
//! (`docs/specs/vhc-architecture-spec.md` §9.6, §9.7).
//!
//! A module's memory footprint used to be one number with one owner. It is now three artifacts with
//! three owners and three independent lifecycles:
//!
//! | Artifact | Owner | Says |
//! |---|---|---|
//! | Logical Resource Plan | the guest | what the algorithm needs, logically |
//! | **Backend Execution Profile** | the host backend implementation | what it physically costs to deliver that |
//! | **Device Capability Report** | the participating node | what the machine actually has |
//!
//! The plan and the Execution Grant are guest-facing and live in `daemon-vhc-proto`. Everything in
//! *this* crate is host-side, and deliberately so.
//!
//! ## The crate boundary is load-bearing
//!
//! **Nothing here may be linked by `contracts/*` or `sdk/*`, and therefore never by a guest.** Most
//! of a device peak is not the module's: pooling and retention, workspace, compilation allocations
//! and staging are properties of a backend implementation the module cannot see and must not know.
//! If the profile type lived in a crate the guests link, every profile revision would change every
//! guest hash — a driver update would re-pin and re-certify a training algorithm that had not
//! changed. That is the exact coupling the redesign exists to remove, and it would be reintroduced
//! by nothing more than a crate-layout choice. `xtask vhc-dep-check` enforces the direction
//! mechanically, the way it already enforces host↔SDK.

#![forbid(unsafe_code)]

pub mod admit;
pub mod capability;
pub mod composition_record;
pub mod governor;
pub mod planner;
pub mod profile;
pub mod provision;
pub mod revision;
pub mod store;
// Fixture assemblers for tests in OTHER crates. Behind a non-default feature that no production
// dependency edge may enable (`xtask vhc-dep-check` fails the gate if one does): the constructors it
// exposes can mint a profile, which is precisely the act the store's crate-private surface exists to
// prevent a shipping binary from performing.
pub mod probe;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod trust;
pub mod verdict;

pub use admit::{
    admit_composition, recompose_recorded_estimate, validate_recorded_composition, AdmissionInputs,
    AdmissionRefusal, AdmittedComposition, RecordedComposition, RecordedCompositionError,
};
pub use capability::{
    admit_device_bytes, admit_node_memory_bytes, derive_device_supply, revision_record_digest,
    AllocationProbeMethod, CapabilityError, DeviceAdmissionRefusal, DeviceCapabilityReport,
    DeviceMemorySource, DeviceMemorySupply, HostDeviceFacts, LinkCapacity,
    MeasuredAllocationCeiling, MemoryPoolTopology, OwnerDeviceCap, SupplyPlatform,
    DEVICE_CAPABILITY_REPORT_SCHEMA,
};
pub use composition_record::{
    validate_citation, CandidateTuple, CompositionEvidenceLedger, CompositionEvidenceRecord,
    ConsequenceClass, EvidenceCitation, EvidenceError, EvidenceSubject, RecordStanding, Revocation,
    COMPOSITION_RECORD_MEMBERS, GOVERNOR_VERSION,
};
pub use governor::{
    check_pool_admissible, compare_views, derive_reservation, evaluate_occupancy, GovernorError,
    OccupancyVerdict, PoolAdmission, PoolSizing, Reservation, ReservationBounds,
    ReservationComponents, ReservationIdentity, ReservationStore, ScopeComponent,
    RESERVATION_ARITHMETIC_VERSION,
};
pub use planner::{
    aggregate, check_estimate_against_lane, compose, compose_selection, AggregateEstimate,
    DivergenceAuthority, LaneEstimateBounds, PhysicalEstimate, PlannerError, ScopedOccupancy,
    ScopedTerm, Selection, PLANNER_VERSION,
};
pub use profile::{
    AllocationCeilings, AllocationScope, BackendExecutionProfile, CompilationBehavior,
    CompositionRule, CostExpr, CostInput, CostInputs, CostTerm, EnforcementClass, Headroom,
    PoolingBehavior, ProfileError, StagingBehavior, WorkspaceFormula,
    BACKEND_EXECUTION_PROFILE_SCHEMA,
};
pub use provision::{
    load as load_provisioned_profiles, load_from_env as load_provisioned_profiles_from_env,
    write as write_provisioned_profiles, ProvisionError, ProvisionedEntry, ProvisionedProfiles,
    PROFILES_FILE_NAME, PROFILE_DIR_ENV, PROVISIONED_PROFILES_SCHEMA,
};
pub use revision::{
    Adapter, AdapterDeviceType, AdapterIdentity, AllocatorImplementation, ApiSelectionSource,
    BackendClass, BackendImplementation, BackendImplementationRevision, ComputeFramework,
    ComputeStackIdentity, DriverApi, DriverRevision, Maybe, OperatingSystem, OsFamily, PlatformApi,
    ProbeObservation, ProducedBy, RevisionRefusal, RevisionSignal, SealedBinaryIdentity,
    Unavailable,
};
pub use store::{
    AuthenticatedProfile, AuthenticationContext, ProfileStore, SelectionRefusal, StoreRefusal,
    StoredProfile,
};
pub use trust::{
    authenticate, AuthenticationRefusal, PolicySide, ProfileAcceptancePolicy, ProfileAuthority,
    ProfileTrustEnvelope, RevisionNumbering, RevisionRange, ValidityPolicy,
};
pub use verdict::{
    FitOutcome, FitProbeKey, FitVerdict, FitVerdictStore, VerdictError, FIT_VERDICT_SCHEMA,
};
