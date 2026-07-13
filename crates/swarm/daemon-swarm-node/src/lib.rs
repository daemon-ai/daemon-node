// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-node` — the node-side swarm-training service (swarm-training-spec.md §10.3/§10.4).
//!
//! The node is the single authority for swarm participation state; the app is a thin mirror
//! (ADR-003). This crate is that authority's runtime:
//!
//! - [`SwarmStore`] — the durable `swarm.db` (spec §10.3): joined-run intents + status
//!   (`swarm_runs`), per-run contribution counters (`swarm_contrib`), and a windowed event log
//!   (`swarm_events`). Durable join-intent drives restart re-convergence.
//! - [`SwarmService`] — owns a [`WorkerControl`] (the `daemon-train-client` `TrainSupervisor` seam),
//!   translates worker events into [`SwarmEvent`](daemon_api::SwarmEvent)s (persisted + fanned out +
//!   `NodeEvent::SwarmChanged` on the node feed), re-issues `JoinRun` for persisted intents on start,
//!   and implements [`daemon_api::SwarmApi`]. **OFF by default** — a disabled service never spawns a
//!   worker.
//!
//! The node binds an `Arc<SwarmService>` as its `Arc<dyn SwarmApi>` (via `NodeApiImpl::with_swarm`)
//! only when `[swarm] enabled = true`.

#![forbid(unsafe_code)]

pub mod service;
pub mod store;

pub use service::{NodeFeed, SwarmError, SwarmService, SwarmServiceParts, WorkerControl};
pub use store::{DesiredState, PersistedRun, StoreError, SwarmStore, EVENT_WINDOW};
