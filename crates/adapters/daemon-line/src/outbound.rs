// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Outbound: the session's merged log → LINE push messages, via a `daemon-delivery` [`Projector`].
//!
//! [`LineProjector`] implements the reusable `daemon_delivery::Projector` (the outbound mirror of the
//! `daemon-ingest` gate): it projects the outbound `AgentEvent` stream down to LINE push messages
//! and, in the same callback, drives the inbound gate's busy state from `TurnStarted`/`TurnFinished`
//! (so `daemon-ingest` needs no second subscription).
//!
//! LINE has no reply token for asynchronous turns (reply tokens are single-use and expire), so every
//! outbound message uses the **push** API (`push_message`, `to = <conversation id>`). The
//! conversation id is the `Primary` sink's route — which is exactly the LINE user/group/room id the
//! inbound `Origin` seeded (see [`crate::mapping::scope_for`]). [`DeliveryManager`] mirrors the
//! Matrix adapter's incremental, dedup'd per-session subscriber so a freshly-opened session gets its
//! reply stream immediately (backfilled from seq 0; the at-least-once re-post on reconnect is the
//! documented tradeoff).

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

use line_bot_sdk_rust::line_messaging_api::apis::MessagingApiApi;
use line_bot_sdk_rust::line_messaging_api::models::{Message, PushMessageRequest, TextMessage};

use crate::account::LineAccount;

/// Projects a session's merged log into LINE push messages and drives the gate's busy state.
pub struct LineProjector {
    api: Arc<dyn NodeApi>,
    ingestor: Arc<Ingestor>,
    /// The per-account bot clients, keyed by their instance-qualified transport id (the `Primary`'s
    /// transport selects which channel pushes the reply).
    clients: HashMap<TransportId, LineAccount>,
}

impl LineProjector {
    /// Construct a projector over `api`, the shared `ingestor`, and the brought-up account clients.
    pub fn new(
        api: Arc<dyn NodeApi>,
        ingestor: Arc<Ingestor>,
        clients: HashMap<TransportId, LineAccount>,
    ) -> Self {
        Self {
            api,
            ingestor,
            clients,
        }
    }

    /// Push `text` to the session's `Primary` conversation (the account + LINE id the opening
    /// `Origin` seeded). The `Primary` route is the LINE `to` id (user/group/room).
    async fn post(&self, session: &SessionId, text: &str) {
        let targets = self.api.delivery_targets(session.clone()).await;
        let Some(primary) = targets.iter().find(|t| t.kind == SinkKind::Primary) else {
            return;
        };
        let Some(account) = self.clients.get(&primary.transport) else {
            return;
        };
        let request = PushMessageRequest {
            to: primary.route.as_str().to_string(),
            messages: vec![Message::TextMessage(TextMessage::new(text.to_string()))],
            notification_disabled: Some(false),
            custom_aggregation_units: None,
        };
        if let Err(e) = account
            .line
            .messaging_api_client
            .push_message(request, None)
            .await
        {
            tracing::warn!(route = primary.route.as_str(), error = ?e, "line: push reply failed");
        }
    }
}

/// The chat-rendering of a blocking host request (approval/choice/input become a prompt message; the
/// user's reply → `respond` round-trip is deferred, so this only posts).
fn prompt_text(req: &HostRequest) -> Option<String> {
    match &req.kind {
        HostRequestKind::Approval { prompt, .. } => Some(format!(
            "[approval needed] {prompt}\n(reply to approve — structured capture coming soon)"
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
impl Projector for LineProjector {
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
                    tracing::warn!(error = %e, "line: gate flush failed");
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
/// inbound `Ingestor::receive` (which returns the opened session id). Mirrors the Matrix adapter's
/// `DeliveryManager`.
pub struct DeliveryManager {
    api: Arc<dyn NodeApi>,
    projector: Arc<LineProjector>,
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
    pub fn new(api: Arc<dyn NodeApi>, projector: Arc<LineProjector>) -> Self {
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
        // would otherwise run with `None` (deny). The `internal` marker is the trusted embedded-caller
        // identity.
        tokio::spawn(with_request_context(
            RequestContext::internal(),
            async move {
                let mut stream = match me.api.subscribe(session.clone(), 0).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "line: delivery subscribe failed");
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

    /// The number of sessions currently being delivered (test/observability helper).
    pub fn active_count(&self) -> usize {
        self.active.lock().unwrap().len()
    }
}
