// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The orchestration-fleet composition glue (moved here from `bins/daemon` so the binary and the
//! conformance harness share one implementation):
//!
//! - [`spawner`]: the profile-driven placement seam ([`ProfileChildSpawner`]).
//! - [`foreign_live`]: foreign-engine (ACP / stream-json) resolution for the live interactive
//!   session seam.
//! - [`job_worker`]: the durable delegation worker ([`FleetJobWorker`]).
//! - [`notice_worker`]: the detached-delegation completion-notice worker ([`NoticeWorker`]).
//! - [`view`]: the durable-graph projection of the management tree ([`FleetViewImpl`]).
//! - [`reaper`]: the ephemeral-subagent reaper ([`EphemeralReaper`]).

pub(crate) mod foreign_incarnation;
pub(crate) mod foreign_live;
pub mod job_worker;
pub mod notice_worker;
pub mod reaper;
pub mod spawner;
pub mod view;

pub use job_worker::FleetJobWorker;
pub use notice_worker::NoticeWorker;
pub use reaper::{EphemeralReaper, ReaperConfig};
pub use spawner::{AgentBackend, ForeignProtocol, LaunchProfile, ProfileChildSpawner};
pub use view::FleetViewImpl;
