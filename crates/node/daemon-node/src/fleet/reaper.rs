// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The ephemeral-subagent reaper: a host-owned background sweep that **archives**
//! [`SessionRole::EphemeralSubagent`] sessions a grace period after they reach a terminal state
//! (the [`SessionMeta::terminal_ms`](daemon_store::SessionMeta::terminal_ms) clock stamped by
//! `mark_completed`). Archiving drops the child out of the default roster/tree scopes (it stays
//! reachable under the explicit `Archived` scope — nothing is deleted), so transient-subagent
//! churn ages out of the GUI without operator gardening. Managed children, primaries, live
//! sessions, already-archived rows, and legacy rows without a terminal stamp are never touched.

use std::sync::Arc;
use std::time::Duration;

use daemon_host::NodeEventFeed;
use daemon_store::{SessionRole, SessionStatus, SessionStore};

/// Unix-millis now — the live sweep's clock (tests inject their own via [`EphemeralReaper::sweep_at`]).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The reaper policy carried on `NodeAssembly`: whether the sweep runs, how long after a terminal
/// state an ephemeral child is archived, and the sweep cadence.
#[derive(Clone, Copy, Debug)]
pub struct ReaperConfig {
    /// Whether the reaper task is spawned at assembly.
    pub enabled: bool,
    /// How long after `terminal_ms` an ephemeral subagent is archived.
    pub grace: Duration,
    /// The sweep interval (the first sweep runs one interval after start, so short-lived test
    /// nodes never observe one).
    pub interval: Duration,
}

impl Default for ReaperConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            grace: Duration::from_secs(300),
            interval: Duration::from_secs(60),
        }
    }
}

/// The sweep itself: scans the durable session graph and archives eligible ephemeral subagents.
pub struct EphemeralReaper {
    store: Arc<dyn SessionStore>,
    grace: Duration,
    /// The node-wide event feed: a sweep that archived something emits a `FleetChanged` pointer so
    /// live tree/roster consumers refetch instead of polling. `None` sweeps silently.
    events: Option<Arc<NodeEventFeed>>,
}

impl EphemeralReaper {
    /// A reaper over `store` archiving ephemeral subagents `grace` after their terminal state.
    pub fn new(store: Arc<dyn SessionStore>, grace: Duration) -> Self {
        Self {
            store,
            grace,
            events: None,
        }
    }

    /// Emit a `FleetChanged` node event after a sweep that archived at least one session.
    pub fn with_events(mut self, events: Arc<NodeEventFeed>) -> Self {
        self.events = Some(events);
        self
    }

    /// Spawn the periodic sweep loop (interval-first: the initial immediate tick is consumed, so
    /// the first sweep lands one `interval` after start).
    pub fn spawn(self, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                self.sweep_at(now_ms()).await;
            }
        })
    }

    /// One sweep pass evaluated at `now_ms` (injected for deterministic tests): archive every
    /// `Completed` `EphemeralSubagent` whose terminal stamp is at least `grace` old. Returns how
    /// many sessions were archived.
    pub async fn sweep_at(&self, now_ms: u64) -> usize {
        let grace_ms = self.grace.as_millis() as u64;
        let mut archived = 0usize;
        for (session, status) in self.store.list_sessions().await {
            if !matches!(status, SessionStatus::Completed) {
                continue;
            }
            let Some(mut meta) = self.store.session_meta(&session).await else {
                continue;
            };
            if meta.role != Some(SessionRole::EphemeralSubagent) || meta.archived {
                continue;
            }
            // Legacy rows (no terminal stamp) are never reaped — forward-looking only.
            let due = meta
                .terminal_ms
                .is_some_and(|t| now_ms.saturating_sub(t) >= grace_ms);
            if !due {
                continue;
            }
            meta.archived = true;
            if self.store.set_session_meta(&session, meta).await.is_ok() {
                archived += 1;
            }
        }
        if archived > 0 {
            if let Some(events) = &self.events {
                let rev = events.note_fleet_change();
                events.emit(daemon_api::NodeEvent::FleetChanged { rev });
            }
        }
        archived
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Epoch, PartitionId, SessionId, SnapshotBlob};
    use daemon_store::{Checkpoint, InMemoryStore, SessionMeta};

    /// Seed one durable session in the given terminal/role/clock shape.
    async fn seed(
        store: &Arc<dyn SessionStore>,
        id: &str,
        role: SessionRole,
        completed: bool,
        terminal_ms: Option<u64>,
        archived: bool,
    ) -> SessionId {
        let session = SessionId::new(id);
        store
            .create_session(
                session.clone(),
                PartitionId::DEFAULT,
                SnapshotBlob::default(),
            )
            .await
            .unwrap();
        if completed {
            let fence = store.acquire_activation_lease(&session).await.unwrap();
            store
                .mark_completed(
                    Checkpoint::new(session.clone(), Epoch(1), SnapshotBlob::default()),
                    fence,
                )
                .await
                .unwrap();
        }
        // Overwrite the meta with the controlled test shape (mark_completed stamped "now").
        let meta = SessionMeta {
            role: Some(role),
            terminal_ms,
            archived,
            ..SessionMeta::default()
        };
        store.set_session_meta(&session, meta).await.unwrap();
        session
    }

    /// Only terminal ephemeral subagents past the grace window are archived; managed children,
    /// live sessions, fresh completions, already-archived rows, and legacy rows (no terminal
    /// stamp) all survive.
    #[tokio::test]
    async fn sweep_archives_only_due_terminal_ephemerals() {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let grace = Duration::from_secs(300);
        let grace_ms = grace.as_millis() as u64;
        let base = 1_000_000u64;

        let due = seed(
            &store,
            "p/c1",
            SessionRole::EphemeralSubagent,
            true,
            Some(base),
            false,
        )
        .await;
        let fresh = seed(
            &store,
            "p/c2",
            SessionRole::EphemeralSubagent,
            true,
            Some(base + grace_ms / 2),
            false,
        )
        .await;
        let managed = seed(
            &store,
            "p/c3",
            SessionRole::ManagedChild,
            true,
            Some(base),
            false,
        )
        .await;
        let live = seed(
            &store,
            "p/c4",
            SessionRole::EphemeralSubagent,
            false,
            Some(base),
            false,
        )
        .await;
        let legacy = seed(
            &store,
            "p/c5",
            SessionRole::EphemeralSubagent,
            true,
            None,
            false,
        )
        .await;
        let already = seed(
            &store,
            "p/c6",
            SessionRole::EphemeralSubagent,
            true,
            Some(base),
            true,
        )
        .await;

        let reaper = EphemeralReaper::new(store.clone(), grace);
        let archived = reaper.sweep_at(base + grace_ms).await;
        assert_eq!(archived, 1, "exactly the due ephemeral is archived");

        let archived_of = |id: &SessionId| {
            let store = store.clone();
            let id = id.clone();
            async move { store.session_meta(&id).await.unwrap().archived }
        };
        assert!(archived_of(&due).await);
        assert!(!archived_of(&fresh).await);
        assert!(!archived_of(&managed).await);
        assert!(!archived_of(&live).await);
        assert!(!archived_of(&legacy).await);
        assert!(
            archived_of(&already).await,
            "stays archived, not re-counted"
        );

        // A second sweep is a no-op (idempotent).
        assert_eq!(reaper.sweep_at(base + grace_ms).await, 0);
        // The fresh one becomes due once its grace elapses.
        assert_eq!(reaper.sweep_at(base + 2 * grace_ms).await, 1);
        assert!(archived_of(&fresh).await);
    }
}
