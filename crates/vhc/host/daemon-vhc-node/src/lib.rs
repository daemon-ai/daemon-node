// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-node` — the node-side vhc-training service (swarm-training-spec.md §10.3/§10.4).
//!
//! The node is the single authority for vhc participation state; the app is a thin mirror
//! (ADR-003). This crate is that authority's runtime:
//!
//! - [`VhcStore`] — the durable `vhc.db` (spec §10.3): joined-run intents + status
//!   (`vhc_runs`), per-run contribution counters (`vhc_contrib`), and a windowed event log
//!   (`vhc_events`). Durable join-intent drives restart re-convergence.
//! - [`VhcService`] — supervises **N worker instances** (one sandbox = one role-instance,
//!   decisions D1/D6; each a [`WorkerControl`], in production a `daemon-vhc-supervisor`
//!   `TrainSupervisor` child) under the owner's aggregate grants, translates worker events into
//!   [`VhcEvent`](daemon_api::VhcEvent)s (persisted + fanned out + `NodeEvent::VhcChanged`
//!   on the node feed), re-issues `JoinRun` for persisted intents on start, and implements
//!   [`daemon_api::VhcApi`]. **OFF by default** — a disabled service never spawns a worker.
//! - [`OwnerArbiter`] — the D6 owner-scoped resource arbiter (Phase E): per-device + host-wide
//!   typed ledgers, atomic check-and-reserve admission, release-before-replacement preemption
//!   ordering, and crash reconciliation, across every role-instance on the host.
//!
//! The node binds an `Arc<VhcService>` as its `Arc<dyn VhcApi>` (via `NodeApiImpl::with_vhc`)
//! only when `[vhc] enabled = true`.

#![forbid(unsafe_code)]

pub mod arbiter;
pub mod credentials;
pub mod discovery;
pub mod seat;
pub mod service;
pub mod store;

pub use arbiter::{
    AdmitRefusal, BudgetSnapshot, ClaimTiers, InstanceCharge, OwnerArbiter, OwnerBudget,
    RoleInstanceId, TierBytes,
};
pub use discovery::{CheckpointPointer, DiscoveredRun, EgressRunDiscovery, RunDiscovery};
// Re-exported so the boot site constructs the registry-backed discovery seam without a direct
// `daemon-vhc-net` dep edge (A3 boot wiring; additive).
pub use daemon_vhc_net::RegistryClient;
pub use service::{NodeFeed, VhcError, VhcService, VhcServiceParts, WorkerControl};
pub use store::{DesiredState, PersistedRun, StoreError, VhcStore, EVENT_WINDOW};
