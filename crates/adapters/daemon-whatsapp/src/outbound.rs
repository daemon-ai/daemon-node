// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Outbound: the session's merged log -> WhatsApp messages, via a `daemon-delivery` [`Projector`].
//!
//! [`WhatsappProjector`] projects the outbound `AgentEvent` stream down to chat messages and, in the
//! same callback, drives the inbound gate's busy state from `TurnStarted`/`TurnFinished`. The
//! [`DeliveryManager`] is the incremental, dedup'd per-session subscriber the Matrix adapter uses,
//! adapted to resolve the sending account from the live [`crate::LiveBackends`] registry so it works
//! for both the user (WhatsApp Web) and bot (Cloud API) backends.

use std::collections::HashSet;
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

use crate::LiveBackends;

/// Projects a session's merged log into WhatsApp posts and drives the gate's busy state.
pub struct WhatsappProjector {
    api: Arc<dyn NodeApi>,
    ingestor: Arc<Ingestor>,
    backends: LiveBackends,
}

impl WhatsappProjector {
    /// Construct a projector over `api`, the shared `ingestor`, and the live backend registry.
    pub fn new(api: Arc<dyn NodeApi>, ingestor: Arc<Ingestor>, backends: LiveBackends) -> Self {
        Self {
            api,
            ingestor,
            backends,
        }
    }

    /// Post `text` to the session's `Primary` chat (the account + chat the opening `Origin` seeded).
    async fn post(&self, session: &SessionId, text: &str) {
        let targets = self.api.delivery_targets(session.clone()).await;
        let Some(primary) = targets.iter().find(|t| t.kind == SinkKind::Primary) else {
            return;
        };
        let backend = {
            let guard = self.backends.read().await;
            guard.get(&primary.transport).cloned()
        };
        let Some(backend) = backend else {
            tracing::warn!(transport = %primary.transport.as_str(), "whatsapp: no live backend for reply");
            return;
        };
        // The reply route is the chat id (the inbound `Group` origin seeds `chat = <jid/recipient>`).
        if let Err(e) = backend.send_text(primary.route.as_str(), text).await {
            tracing::warn!(route = primary.route.as_str(), error = %e, "whatsapp: sending reply failed");
        }
    }
}

/// The chat-rendering of a blocking host request (approval / input / choice); the response
/// round-trip is deferred, so this only posts the prompt (mirrors the Matrix adapter).
fn prompt_text(req: &HostRequest) -> Option<String> {
    match &req.kind {
        HostRequestKind::Approval { prompt, .. } => {
            Some(format!("[approval needed] {prompt}\n(reply to approve)"))
        }
        HostRequestKind::Input { prompt } => Some(format!("[input needed] {prompt}")),
        HostRequestKind::Choice { prompt, options } => {
            let mut s = format!("[choose one] {prompt}");
            for (i, opt) in options.iter().enumerate() {
                s.push_str(&format!("\n{}. {opt}", i + 1));
            }
            Some(s)
        }
        _ => None,
    }
}

#[async_trait]
impl Projector for WhatsappProjector {
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
                if let Err(e) = self.ingestor.note_turn_finished(&session).await {
                    tracing::warn!(error = %e, "whatsapp: gate flush failed");
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
    projector: Arc<WhatsappProjector>,
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
    pub fn new(api: Arc<dyn NodeApi>, projector: Arc<WhatsappProjector>) -> Self {
        Self {
            api,
            projector,
            active: Mutex::new(HashSet::new()),
        }
    }

    /// Ensure a delivery subscription exists for `session` (owned by `transport`). Idempotent.
    pub fn ensure(self: &Arc<Self>, session: SessionId, transport: TransportId) {
        {
            let mut active = self.active.lock().unwrap();
            if !active.insert(session.clone()) {
                return;
            }
        }
        let me = self.clone();
        tokio::spawn(with_request_context(
            RequestContext::internal(),
            async move {
                let mut stream = match me.api.subscribe(session.clone(), 0).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "whatsapp: delivery subscribe failed");
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
