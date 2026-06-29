// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Host configuration: partition ownership and the resident-service cadence/policy.

use crate::supervisor::{Backoff, MeltdownPolicy};
use daemon_common::PartitionId;
use std::time::Duration;

/// Configuration for a [`Host`](crate::Host) and its resident-service tree.
#[derive(Clone, Copy, Debug)]
pub struct HostConfig {
    /// The partition this host owns.
    pub partition: PartitionId,
    /// How often the wake/job dispatchers poll the durable outboxes.
    pub dispatch_interval: Duration,
    /// How often the recovery scanner re-checks for resumable sessions whose wake was lost.
    pub scan_interval: Duration,
    /// How often the cron scheduler (I15) checks for due jobs. Coarser than the dispatch cadence —
    /// cron resolution is seconds, not milliseconds — to keep the idle tick cheap.
    pub schedule_interval: Duration,
    /// Restart backoff applied to every resident service.
    pub backoff: Backoff,
    /// Meltdown threshold for the resident tree.
    pub meltdown: MeltdownPolicy,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            partition: PartitionId::DEFAULT,
            dispatch_interval: Duration::from_millis(2),
            scan_interval: Duration::from_millis(10),
            schedule_interval: Duration::from_secs(1),
            backoff: Backoff::default(),
            meltdown: MeltdownPolicy::default(),
        }
    }
}
