// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-node` — the single host-composition root.
//!
//! Phases 1-11 grew the node's wiring (durable store + resident services, the orchestration fleet as
//! the real job worker, the credential broker, and the live session surface) inline in `bins/daemon`,
//! with a near-identical copy in the conformance harness. [`assemble`] collapses that into one place:
//! both the binary and the gate build their node through it, so there is exactly one composition to
//! keep correct. It lives above `daemon-host` because the fleet + orchestrate-tool glue is
//! composition-layer policy — `daemon-host` deliberately does not depend on `daemon-orchestration`.
//!
//! Callers supply only *policy*: the store, the [`ProviderRegistry`](daemon_core::ProviderRegistry)
//! (provider selection seam), optional brokered credentials, the session/credential
//! [`ProfileRef`](daemon_common::ProfileRef), and the engine [`Config`](daemon_core::Config).
//! [`assemble`] does the standard plumbing (three role `EngineProfile`s, the fleet, the durable
//! factory, the host, and the [`NodeApiImpl`](daemon_host::NodeApiImpl)).
//!
//! This crate root is a thin facade: the composition is split across [`types`] (the policy
//! inputs/outputs), [`profiles`] (role-profile dressing + per-session resolution), [`fleet`] (the
//! placement seam, durable worker, and tree projection), [`cron`] (the resident scheduler), and
//! [`assembly`] (the phase-wired [`assemble`]). The public paths are re-exported here unchanged.

#![forbid(unsafe_code)]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

mod assembly;
pub mod cron;
pub mod fleet;
pub mod profiles;
pub mod types;

pub use assembly::assemble;
pub use cron::{CronSkillLoader, CronWorker};
pub use fleet::{
    AgentBackend, EphemeralReaper, FleetJobWorker, FleetViewImpl, ForeignProtocol, LaunchProfile,
    ProfileChildSpawner, ReaperConfig,
};
pub use types::{
    AssembledNode, GatewayBinding, GatewayCoords, GatewayLease, GatewayTokenMinter, NodeAssembly,
    OrchestrateCaps, PromptAssembly, PromptPolicy, ProviderResolver, ResolvedSkills,
    SkillsResolver,
};
