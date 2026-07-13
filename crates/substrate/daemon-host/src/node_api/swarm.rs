// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`SwarmApi`] on [`NodeApiImpl`] — a thin forwarding seam onto the optional node swarm service
//! (spec §10.4). The real request→supervisor-command + store-read mapping lives in the
//! `daemon-swarm-node` service (bound via [`NodeApiImpl::with_swarm`] only when `[swarm] enabled`);
//! absent it, every op resolves to [`ApiError::Unsupported`] / an empty stream, so a node built
//! without swarm training (the default) never spawns a training worker.

use super::*;
use daemon_api::{
    SwarmApi, SwarmEventStream, SwarmHardwareReport, SwarmLeaveMode, SwarmPolicy, SwarmRunDetail,
    SwarmRunSummary,
};

#[async_trait]
impl SwarmApi for NodeApiImpl {
    async fn swarm_run_list(&self) -> Result<Vec<SwarmRunSummary>, ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_run_list().await,
            None => Err(ApiError::Unsupported("swarm_run_list".into())),
        }
    }

    async fn swarm_run_detail(&self, run_id: String) -> Result<Option<SwarmRunDetail>, ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_run_detail(run_id).await,
            None => Err(ApiError::Unsupported("swarm_run_detail".into())),
        }
    }

    async fn swarm_join(
        &self,
        run_id: String,
        policy: SwarmPolicy,
        op_id: String,
    ) -> Result<(), ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_join(run_id, policy, op_id).await,
            None => Err(ApiError::Unsupported("swarm_join".into())),
        }
    }

    async fn swarm_leave(
        &self,
        run_id: String,
        mode: SwarmLeaveMode,
        op_id: String,
    ) -> Result<(), ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_leave(run_id, mode, op_id).await,
            None => Err(ApiError::Unsupported("swarm_leave".into())),
        }
    }

    async fn swarm_set_policy(&self, policy: SwarmPolicy) -> Result<(), ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_set_policy(policy).await,
            None => Err(ApiError::Unsupported("swarm_set_policy".into())),
        }
    }

    async fn swarm_hardware_report(&self) -> Result<SwarmHardwareReport, ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_hardware_report().await,
            None => Err(ApiError::Unsupported("swarm_hardware_report".into())),
        }
    }

    async fn swarm_subscribe(&self, run_id: Option<String>) -> Result<SwarmEventStream, ApiError> {
        match self.swarm.get() {
            Some(s) => s.swarm_subscribe(run_id).await,
            None => Ok(stream::empty().boxed()),
        }
    }
}
