// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node-managed backend-resource contract.
//!
//! The host's fixed resident services run as interval loops under the one-for-one
//! [`Supervisor`](crate::supervisor::Supervisor). A *managed resource* is the sibling abstraction
//! for a standing backend that owns its own listener/worker lifecycle rather than a tick loop — the
//! OpenAI-compatible gateway (a bound TCP listener) and local inference (a supervised
//! `daemon-infer` worker). Both are brought up (`activate`), observed (`health`), and taken down
//! (`stop`) through one uniform contract, and both surface as [`ServiceHealth`] entries in
//! [`ControlApi::health`](daemon_api::ControlApi::health) alongside the resident services.
//!
//! The trait lives here (in the substrate that projects health) but the implementations are
//! injected by the assembling binary — `daemon-host` links neither the gateway HTTP surface
//! (`daemon-gateway`) nor the provider stack (`daemon-providers`), mirroring the
//! [`AgentDiscovery`](crate::node_api::AgentDiscovery) / [`ForeignSessionFactory`](crate::node_api::ForeignSessionFactory)
//! injection seams.

use crate::supervisor::ServiceError;
use async_trait::async_trait;
use daemon_api::{ApiError, GatewayStatus, ServiceHealth};

/// A node-managed backend resource with a uniform lifecycle: bring it up (`activate`), observe it
/// (`health`), and take it down (`stop`). Reported in `HealthReport.services` alongside the
/// resident-service supervisor's children.
#[async_trait]
pub trait ManagedResource: Send + Sync {
    /// The stable service name — the [`ServiceHealth::name`] and health-map key (e.g. `"gateway"`,
    /// `"local-inference"`).
    fn name(&self) -> &str;

    /// Bring the resource up (idempotent): a no-op if it is already active, or if it is configured
    /// off. An `Err` means the bring-up failed (e.g. the gateway could not bind); the resource stays
    /// down and its `health` reports the failure.
    async fn activate(&self) -> Result<(), ServiceError>;

    /// The resource's current health as a wire [`ServiceHealth`] line.
    async fn health(&self) -> ServiceHealth;

    /// Take the resource down (idempotent, best-effort). Called on node shutdown; a stopped resource
    /// can be re-`activate`d.
    async fn stop(&self);
}

/// The typed control seam for the node-owned OpenAI-compatible gateway resource. Extends
/// [`ManagedResource`] (so it is reported in health like any other managed backend) with the
/// wire-configurable enable/rebind ops backing [`ControlApi::gateway_get`](daemon_api::ControlApi::gateway_get)
/// / [`gateway_set`](daemon_api::ControlApi::gateway_set). The implementation owns durable
/// persistence of the override (store-backed, boot config as the default) and hot-(re)binds the
/// listener on change.
#[async_trait]
pub trait GatewayControl: ManagedResource {
    /// The gateway's current runtime status (enabled/addr/listening/last_error).
    async fn get(&self) -> GatewayStatus;

    /// Enable/disable the gateway and optionally rebind its listener, persisting the new state and
    /// hot-(re)binding. `addr = None` keeps the current/boot address. Returns the resulting status.
    async fn set(&self, enabled: bool, addr: Option<String>) -> Result<GatewayStatus, ApiError>;
}
