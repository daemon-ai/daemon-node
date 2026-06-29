// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The child-placement seam (layout §4: the per-child mechanism).
//!
//! The runtime never builds engines itself — that would couple it to `daemon-core` and weld it to
//! the agent-driven mode. Instead, *how* a child is materialized is an injected [`ChildSpawner`]
//! that returns an upward-facing [`ManagedUnit`]. The engine-backed spawner (wrapping a
//! `daemon-core` engine in `daemon_host::EngineUnit`) lives where `daemon-core` + `daemon-host` are
//! already in scope — the conformance harness today, `bins/daemon` in production. A future
//! process/container placement (phase 5) is just a different `ChildSpawner`.

use async_trait::async_trait;
use daemon_common::UnitId;
use daemon_supervision::{DelegationSpec, ManagedUnit};
use std::sync::Arc;

/// Materializes a child unit on demand — the runtime's only handle onto placement.
///
/// Given the child's `id` and its [`DelegationSpec`] (work + attenuated toolset + budget), produce
/// the upward [`ManagedUnit`] the runtime will register, drive, and observe. The runtime installs
/// its own request handler and subscribes to events *after* this returns, so the spawner must not
/// start work itself (work begins on the first [`daemon_supervision::ManageCommand::Assign`]).
#[async_trait]
pub trait ChildSpawner: Send + Sync {
    /// Build the child unit identified by `id` from `spec`.
    async fn spawn(&self, id: UnitId, spec: &DelegationSpec) -> Arc<dyn ManagedUnit>;
}
