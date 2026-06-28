//! `daemon-delivery` — the reusable **pull** half of the §5.9.3 outbound routing capability.
//!
//! The host owns inbound routing (origin -> session -> profile) and outbound *storage* (each live
//! session's `Primary`/`Spectator` [`DeliveryTarget`]s) plus two primitives a transport needs to
//! deliver replies symmetrically:
//!
//! - [`SessionApi::delivery_sessions`](daemon_api::SessionApi::delivery_sessions) — *which* sessions
//!   this transport instance currently owns (its `Primary`), the discovery a reconnecting transport
//!   runs to find what it must resume posting for; and
//! - [`SessionApi::subscribe`](daemon_api::SessionApi::subscribe) — the live merged-log stream to
//!   project into outbound messages.
//!
//! This crate stitches those into a turnkey loop so an adapter does not re-implement discovery,
//! fan-in, and handover-stop each time: [`serve_delivery`] discovers the owned sessions, subscribes
//! each, hands every entry to an adapter-owned [`Projector`], and **stops** a session's stream once
//! the transport is handed over (demoted from `Primary` to `Spectator`). It is the cross-process /
//! reconnect-safe path; the in-process push path is `daemon_api::DeliverySink` (registered with the
//! host directly, no socket). Pure-contracts deps only (no host, no substrate), so any adapter — the
//! HTTP surface, a future matrix transport — can build on it.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_api::{NodeApi, SessionLogEntry};
use daemon_common::SessionId;
use daemon_protocol::{SinkKind, TransportId};
use futures::StreamExt;
use std::sync::Arc;
use tokio::task::JoinHandle;

/// The adapter-owned policy that turns a session's merged [`SessionLogEntry`] stream into whatever
/// the transport posts (a Matrix message, an SSE frame, a webhook). Projection lives here, not in
/// the host or this crate, so each transport keeps its own rendering/coalescing rules.
#[async_trait]
pub trait Projector: Send + Sync {
    /// Handle one outbound `entry` for `session` (already filtered to a session this transport owns).
    async fn project(&self, session: SessionId, entry: SessionLogEntry);
}

/// A live outbound delivery subscription: one background task per owned session, each forwarding the
/// session's merged log into the [`Projector`] until handover demotes the transport or the task is
/// [`abort`](DeliverySubscription::abort)ed. Dropping it aborts every task.
pub struct DeliverySubscription {
    tasks: Vec<JoinHandle<()>>,
}

impl DeliverySubscription {
    /// The number of owned sessions currently being delivered (one task each).
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the transport currently owns no sessions for delivery.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Abort every per-session delivery task immediately.
    pub fn abort(&self) {
        for task in &self.tasks {
            task.abort();
        }
    }

    /// Await every per-session task to completion (each ends on handover demotion or stream close).
    pub async fn join(mut self) {
        for task in std::mem::take(&mut self.tasks) {
            let _ = task.await;
        }
    }
}

impl Drop for DeliverySubscription {
    fn drop(&mut self) {
        self.abort();
    }
}

/// Whether `transport` is still the `Primary` reply sink of `session` (i.e. still owns delivery).
async fn still_owns(api: &Arc<dyn NodeApi>, session: &SessionId, transport: &TransportId) -> bool {
    api.delivery_targets(session.clone())
        .await
        .iter()
        .any(|t| t.kind == SinkKind::Primary && &t.transport == transport)
}

/// Discover every session `transport` currently owns ([`delivery_sessions`]), subscribe each merged
/// log, and forward entries to `projector` — the reconnect-safe outbound loop a transport runs on
/// (re)connect. Each session's task **stops** once the transport is no longer the session's
/// `Primary` (handover demotion), so a handed-over session falls off delivery without explicit
/// teardown. Returns a [`DeliverySubscription`] handle owning the per-session tasks.
///
/// `after_seq = 0` backfills each session from the start of its live log before going live (a
/// reconnecting transport re-posts from where it can dedupe); pass a per-session high-water mark via
/// the projector if exactly-once posting matters.
pub async fn serve_delivery(
    api: Arc<dyn NodeApi>,
    transport: TransportId,
    projector: Arc<dyn Projector>,
) -> DeliverySubscription {
    let sessions = api.delivery_sessions(transport.clone()).await;
    let mut tasks = Vec::with_capacity(sessions.len());
    for session in sessions {
        let api = api.clone();
        let transport = transport.clone();
        let projector = projector.clone();
        let task = tokio::spawn(async move {
            let mut stream = match api.subscribe(session.clone(), 0).await {
                Ok(s) => s,
                Err(_) => return,
            };
            while let Some(item) = stream.next().await {
                // A lossy lag is best-effort-skipped here (the prior silent-drop behavior); durable
                // delivery re-baseline is future work (it would re-subscribe from 0 and dedup).
                let entry = match item {
                    daemon_api::LogStreamItem::Entry(e) => e,
                    daemon_api::LogStreamItem::Lagged => continue,
                };
                // Re-check ownership before projecting: subscribe streams the full merged log to every
                // reader (Primary or Spectator), so a demoted transport keeps receiving entries — it
                // must drop out itself once it is no longer the Primary (handover stop).
                if !still_owns(&api, &session, &transport).await {
                    break;
                }
                projector.project(session.clone(), entry).await;
            }
        });
        tasks.push(task);
    }
    DeliverySubscription { tasks }
}
