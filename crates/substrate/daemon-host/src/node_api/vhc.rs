// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`VhcApi`] on [`NodeApiImpl`] — a thin forwarding seam onto the optional node vhc service
//! (spec §10.4). The real request→supervisor-command + store-read mapping lives in the
//! `daemon-vhc-node` service (bound via [`NodeApiImpl::with_vhc`] only when `[vhc] enabled`);
//! absent it, every op resolves to [`ApiError::Unsupported`] / an empty stream, so a node built
//! without vhc training (the default) never spawns a training worker.

use super::*;
use daemon_api::{
    VhcApi, VhcEventStream, VhcHardwareReport, VhcLeaveMode, VhcPolicy, VhcRunDetail, VhcRunSummary,
};

#[async_trait]
impl VhcApi for NodeApiImpl {
    async fn vhc_run_list(&self) -> Result<Vec<VhcRunSummary>, ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_run_list().await,
            None => Err(ApiError::Unsupported("vhc_run_list".into())),
        }
    }

    async fn vhc_run_detail(&self, run_id: String) -> Result<Option<VhcRunDetail>, ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_run_detail(run_id).await,
            None => Err(ApiError::Unsupported("vhc_run_detail".into())),
        }
    }

    async fn vhc_join(
        &self,
        run_id: String,
        policy: VhcPolicy,
        op_id: String,
    ) -> Result<(), ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_join(run_id, policy, op_id).await,
            None => Err(ApiError::Unsupported("vhc_join".into())),
        }
    }

    async fn vhc_leave(
        &self,
        run_id: String,
        mode: VhcLeaveMode,
        op_id: String,
    ) -> Result<(), ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_leave(run_id, mode, op_id).await,
            None => Err(ApiError::Unsupported("vhc_leave".into())),
        }
    }

    async fn vhc_pause(&self, run_id: String, op_id: String) -> Result<(), ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_pause(run_id, op_id).await,
            None => Err(ApiError::Unsupported("vhc_pause".into())),
        }
    }

    async fn vhc_resume(&self, run_id: String, op_id: String) -> Result<(), ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_resume(run_id, op_id).await,
            None => Err(ApiError::Unsupported("vhc_resume".into())),
        }
    }

    async fn vhc_set_policy(&self, policy: VhcPolicy) -> Result<(), ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_set_policy(policy).await,
            None => Err(ApiError::Unsupported("vhc_set_policy".into())),
        }
    }

    async fn vhc_hardware_report(&self) -> Result<VhcHardwareReport, ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_hardware_report().await,
            None => Err(ApiError::Unsupported("vhc_hardware_report".into())),
        }
    }

    async fn vhc_subscribe(&self, run_id: Option<String>) -> Result<VhcEventStream, ApiError> {
        match self.vhc.get() {
            Some(s) => s.vhc_subscribe(run_id).await,
            None => Ok(stream::empty().boxed()),
        }
    }
}
