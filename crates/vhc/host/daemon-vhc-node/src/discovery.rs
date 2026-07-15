// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`RunDiscovery`] — the run-discovery + envelope-fetch seam the join flow drives (spec §6.1/§6.5;
//! A1).
//!
//! [`SwarmService::swarm_join`](crate::SwarmService) used to derive eligibility from a hardware
//! probe against a hardcoded allowlist coordinator (W1 placeholder). A1 replaces that with real
//! discovery: resolve the run from the coordinator registry, fetch + blake3-verify the frozen
//! envelope, and hand it to the worker's existing `AssessRun` for a real §6.5 verdict **before**
//! `JoinRun`. This trait is the seam (a [`EgressRunDiscovery`] over
//! [`daemon_swarm_net::RegistryClient`] in production, a fake in tests) so the service is testable
//! without a live coordinator.

use async_trait::async_trait;
use daemon_swarm_net::{RegistryClient, RunId};

use crate::service::SwarmError;

/// A discovered run: the coordination facts the node needs to assess + join (never experiment
/// config or module bytes — the seam rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredRun {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// The coordinator endpoint the run is served from (the WS `base_url` + presign base).
    pub coordinator: String,
    /// blake3 of the frozen envelope (hex) — the assert the peer joins under (§6.5).
    pub envelope_hash: String,
    /// The swarm proto version the run is pinned to (§16).
    pub proto_version: u32,
}

/// The discovery seam: list/resolve runs from the coordinator registry and fetch a run's frozen
/// envelope bytes (blake3-verified). Implemented by [`EgressRunDiscovery`] (real) + test fakes.
#[async_trait]
pub trait RunDiscovery: Send + Sync {
    /// Discover all runs the coordinator advertises (registry `GET /runs`).
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, SwarmError>;
    /// Resolve one run (`GET /runs/:id`); `None` if the coordinator does not know it.
    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, SwarmError>;
    /// Fetch the run's frozen envelope bytes (presigned GET + blake3-verify). Errors if the run is
    /// unknown or the bytes do not match the descriptor's hash.
    async fn fetch_envelope(&self, run_id: &str) -> Result<Vec<u8>, SwarmError>;
}

/// The production [`RunDiscovery`]: a [`RegistryClient`] against a swarm coordinator base.
pub struct EgressRunDiscovery {
    registry: RegistryClient,
    coordinator: String,
}

impl EgressRunDiscovery {
    /// Wrap a configured [`RegistryClient`]; its base URL is the coordinator endpoint the discovered
    /// runs are served from (the WS + presign base).
    pub fn new(registry: RegistryClient) -> Self {
        let coordinator = registry.base_url().to_string();
        Self {
            registry,
            coordinator,
        }
    }
}

#[async_trait]
impl RunDiscovery for EgressRunDiscovery {
    async fn list_runs(&self) -> Result<Vec<DiscoveredRun>, SwarmError> {
        let runs = self
            .registry
            .list_runs()
            .await
            .map_err(|e| SwarmError::Discovery(e.to_string()))?;
        Ok(runs
            .into_iter()
            .map(|d| DiscoveredRun {
                run_id: d.run_id,
                coordinator: self.coordinator.clone(),
                envelope_hash: d.envelope_hash,
                proto_version: d.proto_version,
            })
            .collect())
    }

    async fn get_run(&self, run_id: &str) -> Result<Option<DiscoveredRun>, SwarmError> {
        let run = self
            .registry
            .get_run(run_id)
            .await
            .map_err(|e| SwarmError::Discovery(e.to_string()))?;
        Ok(run.map(|d| DiscoveredRun {
            run_id: d.run_id,
            coordinator: self.coordinator.clone(),
            envelope_hash: d.envelope_hash,
            proto_version: d.proto_version,
        }))
    }

    async fn fetch_envelope(&self, run_id: &str) -> Result<Vec<u8>, SwarmError> {
        let descriptor = self
            .registry
            .get_run(run_id)
            .await
            .map_err(|e| SwarmError::Discovery(e.to_string()))?
            .ok_or_else(|| SwarmError::Discovery(format!("run {run_id} not found in registry")))?;
        self.registry
            .fetch_envelope(&RunId::new(run_id), &descriptor)
            .await
            .map_err(|e| SwarmError::Discovery(e.to_string()))
    }
}
