// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The orchestration-fleet composition glue (moved here from `bins/daemon` so the binary and the
//! conformance harness share one implementation):
//!
//! - [`spawner`]: the profile-driven placement seam ([`ProfileChildSpawner`]).
//! - [`acp_live`]: foreign-engine (ACP) resolution for the live interactive session seam.
//! - [`job_worker`]: the durable delegation worker ([`FleetJobWorker`]).
//! - [`view`]: the durable-graph projection of the management tree ([`FleetViewImpl`]).

pub(crate) mod acp_live;
pub mod job_worker;
pub mod spawner;
pub mod view;

pub use job_worker::FleetJobWorker;
pub use spawner::{AgentBackend, ForeignProtocol, LaunchProfile, ProfileChildSpawner};
pub use view::FleetViewImpl;
