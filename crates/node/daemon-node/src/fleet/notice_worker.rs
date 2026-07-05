// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The detached-delegation notice worker: a host-owned background loop that drains the durable
//! completion-notice outbox and delivers each notice to its parent as a fresh reactive turn.
//!
//! When a detached (`spawn wait:false`) child reaches a terminal state, `mark_completed` pushes a
//! [`CompletionNotice`](daemon_store::CompletionNotice) — the child never fulfilled a parent job (the
//! parent did not suspend), so it rides the notice outbox rather than the `waiting_for`/
//! `completion_inbox` rail. This worker decodes the opaque payload (a CBOR `DelegationResult`) and
//! injects `[subagent {child} completed] {summary}` into the parent through the one lifecycle-aware
//! seam ([`NodeApiImpl::inject_session_input`]): a live parent takes a reactive `StartTurn`, a parked
//! durable parent gets a pending input + wake (drained at hydrate), and a settled parent cleanly
//! drops it. Failed and cancelled children deliver the same way — the parent always learns the
//! outcome.

use std::sync::Arc;
use std::time::Duration;

use daemon_host::NodeApiImpl;
use daemon_protocol::DelegationResult;
use daemon_store::SessionStore;

/// The notice worker: drains completion notices and injects them into their parent sessions.
pub struct NoticeWorker {
    store: Arc<dyn SessionStore>,
    /// The assembled node surface, for the one lifecycle-aware `inject_session_input` seam. Bound
    /// after `NodeApiImpl` exists (mirroring the `NodeProcessNotifier` late-bind).
    node: Arc<NodeApiImpl>,
}

impl NoticeWorker {
    /// A notice worker over `store`, delivering through `node`.
    pub fn new(store: Arc<dyn SessionStore>, node: Arc<NodeApiImpl>) -> Self {
        Self { store, node }
    }

    /// Drain every pending completion notice, injecting each into its parent. Returns how many were
    /// delivered (a delivery that the parent dropped — e.g. a settled session — still counts as
    /// drained). Idempotent at the outbox level: each notice is dequeued exactly once.
    pub async fn drain_once(&self) -> usize {
        let mut delivered = 0usize;
        while let Some(notice) = self.store.dequeue_completion_notice().await {
            let result = DelegationResult::decode(&notice.payload);
            let text = format!("[subagent {} completed] {}", notice.child, result.summary);
            if let Err(e) = self.node.inject_session_input(&notice.parent, text).await {
                tracing::warn!(
                    parent = %notice.parent,
                    child = %notice.child,
                    error = %e,
                    "completion notice could not be injected"
                );
            }
            delivered += 1;
        }
        delivered
    }

    /// Spawn the periodic drain loop at `interval` (the wake/job dispatch cadence). The first tick
    /// fires immediately (a harmless empty drain), then every `interval`.
    pub fn spawn(self, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                self.drain_once().await;
            }
        })
    }
}
