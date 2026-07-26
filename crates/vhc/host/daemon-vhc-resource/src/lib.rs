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

pub mod profile;
pub mod revision;
pub mod trust;

pub use profile::{
    AllocationCeilings, AllocationScope, BackendExecutionProfile, CompilationBehavior,
    CompositionRule, CostExpr, CostInput, CostInputs, CostTerm, EnforcementClass, Headroom,
    PoolingBehavior, ProfileError, StagingBehavior, WorkspaceFormula,
    BACKEND_EXECUTION_PROFILE_SCHEMA,
};
pub use revision::{
    Adapter, AdapterDeviceType, AdapterIdentity, AllocatorImplementation, ApiSelectionSource,
    BackendClass, BackendImplementation, BackendImplementationRevision, ComputeFramework,
    DriverApi, DriverRevision, Maybe, OperatingSystem, OsFamily, PlatformApi, ProducedBy,
    RevisionRefusal, RevisionSignal, SealedBinaryIdentity, Unavailable,
};
pub use trust::{
    authenticate, AuthenticationRefusal, PolicySide, ProfileAcceptancePolicy, ProfileAuthority,
    ProfileTrustEnvelope, RevisionNumbering, RevisionRange, ValidityPolicy,
};
