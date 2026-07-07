// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Outbound: the session's merged log -> Telegram messages, via a `daemon-delivery` [`Projector`].
//!
//! [`TelegramProjector`] implements the reusable `daemon_delivery::Projector` (the inbound mirror of
//! the `daemon-ingest` gate): it projects the outbound `AgentEvent` stream down to chat messages
//! and, in the same callback, drives the inbound gate's busy state from `TurnStarted`/`TurnFinished`
//! (so `daemon-ingest` needs no second subscription).
//!
//! Like the Matrix adapter, a chat transport creates sessions at runtime (on first inbound), so
//! [`DeliveryManager`] wraps the delivery-crate `Projector` + wire primitives (`delivery_sessions` /
//! `subscribe` / `delivery_targets`) in an incremental, dedup'd per-session subscriber. Sends go
//! through the SDK-agnostic [`TelegramClient`] trait, so this module never touches grammers.

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

use crate::mapping::parse_chat_id;
use crate::TelegramClient;

/// Projects a session's merged log into Telegram messages and drives the gate's busy state.
pub struct TelegramProjector {
    api: Arc<dyn NodeApi>,
    ingestor: Arc<Ingestor>,
    /// The per-account clients, keyed by their instance-qualified transport id (the `Primary`'s
    /// transport selects which account posts the reply).
    clients: HashMap<TransportId, Arc<dyn TelegramClient>>,
}

impl TelegramProjector {
    /// Construct a projector over `api`, the shared `ingestor`, and the brought-up account clients.
    pub fn new(
        api: Arc<dyn NodeApi>,
        ingestor: Arc<Ingestor>,
        clients: HashMap<TransportId, Arc<dyn TelegramClient>>,
    ) -> Self {
        Self {
            api,
            ingestor,
            clients,
        }
    }

    /// Post `text` to the session's `Primary` chat (the account + chat the opening `Origin` seeded).
    async fn post(&self, session: &SessionId, text: &str) {
        let targets = self.api.delivery_targets(session.clone()).await;
        let Some(primary) = targets.iter().find(|t| t.kind == SinkKind::Primary) else {
            return;
        };
        let Some(client) = self.clients.get(&primary.transport) else {
            return;
        };
        // The route *is* the chat id (the inbound `Group` origin seeds `chat = chat id`).
        let Some(chat_id) = parse_chat_id(primary.route.as_str()) else {
            tracing::warn!(
                route = primary.route.as_str(),
                "telegram: unparseable reply chat id"
            );
            return;
        };
        if let Err(e) = client.send_text(chat_id, text).await {
            tracing::warn!(%chat_id, error = %e, "telegram: sending reply failed");
        }
    }
}

/// The chat-rendering of a blocking host request. Approval/Choice/Input become a prompt message; the
/// user's reaction/reply -> `respond` round-trip is deferred, so this only posts.
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
impl Projector for TelegramProjector {
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
                    tracing::warn!(error = %e, "telegram: gate flush failed");
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
    projector: Arc<TelegramProjector>,
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
    pub fn new(api: Arc<dyn NodeApi>, projector: Arc<TelegramProjector>) -> Self {
        Self {
            api,
            projector,
            active: Mutex::new(HashSet::new()),
        }
    }

    /// Ensure a delivery subscription exists for `session` (owned by `transport`). Idempotent: a
    /// session already being delivered is a no-op. The task backfills from seq 0, forwards each
    /// entry to the projector, and stops when the transport loses `Primary` (handover) or the stream
    /// closes.
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
                        tracing::warn!(error = %e, "telegram: delivery subscribe failed");
                        me.active.lock().unwrap().remove(&session);
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
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
}
