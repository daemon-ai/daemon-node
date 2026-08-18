// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Host configuration: partition ownership and the resident-service cadence/policy.

use crate::supervisor::{Backoff, MeltdownPolicy};
use daemon_common::PartitionId;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for a [`Host`](crate::Host) and its resident-service tree.
#[derive(Clone, Debug)]
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
    /// Root directory for node-owned per-agent state homes (wire v47 A4). When set, stream-json
    /// foreign agents spawn with a `Clean` scrubbed environment and `HOME`/`XDG_*` repointed to
    /// `<root>/<agent-name>`, so agent dotfile state is node-owned and no daemon-ambient secret
    /// leaks into the child. `None` (the default) preserves the historical `InheritFull` spawn.
    /// ACP agents are NOT isolated by this knob: the ACP transport owns process creation without
    /// an env-scrubbing hook (a documented deferral).
    pub agent_state_root: Option<PathBuf>,
    /// Commit-then-linger residency (session-unification §8): how long a durable incarnation
    /// stays hydrated after a non-terminal turn commit awaiting the next wake (no rehydrate cost
    /// per message). The timeout only passivates the ALREADY-COMMITTED incarnation — no commit is
    /// owed at passivation. `None` disables lingering (every commit passivates immediately).
    pub linger: Option<Duration>,
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
            agent_state_root: None,
            linger: Some(Duration::from_secs(30)),
        }
    }
}
