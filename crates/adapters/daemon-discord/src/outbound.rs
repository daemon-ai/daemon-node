// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Outbound: the session's merged log -> Discord channel messages, via a `daemon-delivery` `Projector`.
//!
//! [`DiscordProjector`] implements the reusable `daemon_delivery::Projector`: it projects the outbound
//! `AgentEvent` stream down to channel messages and, in the same callback, drives the inbound gate's
//! busy state from `TurnStarted`/`TurnFinished` (so `daemon-ingest` needs no second subscription).
//!
//! [`DeliveryManager`] wraps the delivery primitives (`subscribe` / `delivery_targets`) in an
//! incremental, dedup'd per-session subscriber so a freshly-opened session gets its reply stream
//! immediately; each task backfills from seq 0 (at-least-once re-post on reconnect is the documented
//! tradeoff). It is the same shape as the Matrix adapter's manager — only the wire send differs.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;

use daemon_api::{NodeApi, SessionLogEntry};
use daemon_common::SessionId;
use daemon_delivery::Projector;
use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::Ingestor;
use daemon_protocol::{
    AgentEvent, HostRequest, HostRequestKind, SessionPayload, SinkKind, TransportId,
};
use serenity_self::http::Http;
use serenity_self::model::id::ChannelId;

/// Projects a session's merged log into Discord posts and drives the gate's busy state.
pub struct DiscordProjector {
    api: Arc<dyn NodeApi>,
    ingestor: Arc<Ingestor>,
    /// The per-account REST handles, keyed by their instance-qualified transport id (the `Primary`'s
    /// transport selects which account posts the reply).
    clients: HashMap<TransportId, Arc<Http>>,
}

impl DiscordProjector {
    /// Construct a projector over `api`, the shared `ingestor`, and the brought-up account REST handles.
    pub fn new(
        api: Arc<dyn NodeApi>,
        ingestor: Arc<Ingestor>,
        clients: HashMap<TransportId, Arc<Http>>,
    ) -> Self {
        Self {
            api,
            ingestor,
            clients,
        }
    }

    /// Post `text` to the session's `Primary` channel (the account + channel the opening `Origin`
    /// seeded). The route *is* the channel id (`Group { chat = channel_id, thread = None }`).
    async fn post(&self, session: &SessionId, text: &str) {
        let targets = self.api.delivery_targets(session.clone()).await;
        let Some(primary) = targets.iter().find(|t| t.kind == SinkKind::Primary) else {
            return;
        };
        let Some(http) = self.clients.get(&primary.transport) else {
            return;
        };
        let channel_id = match primary.route.as_str().parse::<ChannelId>() {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(route = primary.route.as_str(), error = %e, "discord: unparseable reply channel id");
                return;
            }
        };
        if let Err(e) = channel_id.say(http.as_ref(), text).await {
            tracing::warn!(channel = %channel_id.get(), error = %e, "discord: sending reply failed");
        }
    }
}

/// The chat-rendering of a blocking host request (spec §4). Approval/Choice/Input become a prompt
/// message; the user's reaction/reply -> `respond` round-trip is deferred, so this only posts.
fn prompt_text(req: &HostRequest) -> Option<String> {
    match &req.kind {
        HostRequestKind::Approval { prompt, .. } => Some(format!(
            "[approval needed] {prompt}\n(reply to approve — reaction capture coming soon)"
        )),
        HostRequestKind::Input { prompt } => Some(format!("[input needed] {prompt}")),
        HostRequestKind::Choice { prompt, options } => {
            let mut s = format!("[choose one] {prompt}");
            for (i, opt) in options.iter().enumerate() {
                s.push_str(&format!("\n{}. {opt}", i + 1));
            }
            Some(s)
        }
        // Delegation / spawn are host-internal; nothing to surface to the chat.
        _ => None,
    }
}

#[async_trait]
impl Projector for DiscordProjector {
    async fn project(&self, session: SessionId, entry: SessionLogEntry) {
        match &entry.payload {
            SessionPayload::Event(AgentEvent::TurnStarted { .. }) => {
                self.ingestor.note_turn_started(&session);
            }
            SessionPayload::Event(AgentEvent::TurnFinished { summary, .. }) => {
                if let Some(text) = &summary.final_text {
                    if !text.is_empty() {
                        self.post(&session, text).await;
                    }
                }
                // Idle the gate and flush any addressed messages queued mid-turn.
                if let Err(e) = self.ingestor.note_turn_finished(&session).await {
                    tracing::warn!(error = %e, "discord: gate flush failed");
                }
            }
            SessionPayload::Request(req) => {
                if let Some(prompt) = prompt_text(req) {
                    self.post(&session, &prompt).await;
                }
            }
            _ => {}
        }
    }
}

/// Incremental, dedup'd outbound delivery: one backfilling subscribe task per owned session, reusing
/// the `daemon-delivery` `Projector`. Seed it at bring-up from `delivery_sessions` and on every
/// inbound `Ingestor::receive` (which returns the opened session id).
pub struct DeliveryManager {
    api: Arc<dyn NodeApi>,
    projector: Arc<DiscordProjector>,
    active: Mutex<HashSet<SessionId>>,
}

/// Whether `transport` is still the session's `Primary` (delivery ownership; stops on handover).
async fn still_owns(api: &Arc<dyn NodeApi>, session: &SessionId, transport: &TransportId) -> bool {
    api.delivery_targets(session.clone())
        .await
        .iter()
        .any(|t| t.kind == SinkKind::Primary && &t.transport == transport)
}

impl DeliveryManager {
    /// Construct a delivery manager over `api` and the shared `projector`.
    pub fn new(api: Arc<dyn NodeApi>, projector: Arc<DiscordProjector>) -> Self {
        Self {
            api,
            projector,
            active: Mutex::new(HashSet::new()),
        }
    }

    /// Ensure a delivery subscription exists for `session` (owned by `transport`). Idempotent: a
    /// session already being delivered is a no-op. The task backfills from seq 0, forwards each entry
    /// to the projector, and stops when the transport loses `Primary` (handover) or the stream closes.
    pub fn ensure(self: &Arc<Self>, session: SessionId, transport: TransportId) {
        {
            let mut active = self.active.lock().unwrap();
            if !active.insert(session.clone()) {
                return;
            }
        }
        let me = self.clone();
        // Bind the in-process `internal` principal for the whole detached delivery task: a spawned
        // task inherits no request context, so the ownership-gated `subscribe` / `delivery_targets`
        // (and the projector's `submit` on turn-finish) would otherwise run with `None` (deny).
        tokio::spawn(with_request_context(
            RequestContext::internal(),
            async move {
                let mut stream = match me.api.subscribe(session.clone(), 0).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "discord: delivery subscribe failed");
                        me.active.lock().unwrap().remove(&session);
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
                    // Best-effort-skip a lossy lag; durable delivery re-baseline is future work.
                    let entry = match item {
                        daemon_api::LogStreamItem::Entry(e) => e,
                        daemon_api::LogStreamItem::Lagged => continue,
                    };
                    if !still_owns(&me.api, &session, &transport).await {
                        break;
                    }
                    me.projector.project(session.clone(), entry).await;
                }
                me.active.lock().unwrap().remove(&session);
            },
        ));
    }

    /// The number of sessions currently being delivered (test/observability helper).
    pub fn active_count(&self) -> usize {
        self.active.lock().unwrap().len()
    }
}
