// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: Telegram message -> `Origin` + `Reception` -> the `daemon-ingest` gate.
//!
//! The confined grammers layer ([`crate::client`]) extracts the transport-neutral [`InboundEvent`]
//! from its SDK `Message` and hands it here. This module keeps only the *transport-specific* piece —
//! classifying whether a message is *addressed* (mention / DM / `!command`) — and hands a normalised
//! [`Reception`] to the reusable [`Ingestor`], which owns the transport-agnostic command selection
//! (`StartTurn` / `Observe` / busy policy). A Telegram chat id is the natural session/route key, so
//! every chat (including private DMs) maps to `OriginScope::Group { chat }`; DM-ness only influences
//! the *addressed* flag, not the scope (keeping outbound routing trivial: the reply route is the
//! chat id). SDK-free, so it is unit testable without a live client.

use std::sync::Arc;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::config::{self, TelegramRoute};
use crate::mapping::chat_conv_id;
use crate::outbound::DeliveryManager;

/// The transport-neutral projection of one inbound Telegram message — extracted by the confined
/// grammers layer so the routing/addressing logic is SDK-free and testable.
#[derive(Clone, Debug)]
pub struct InboundEvent {
    /// The chat (peer) the message was sent to — the session/route key.
    pub chat_id: i64,
    /// The sender's Telegram user id (the immutable platform identity the ingest gate keys on).
    pub sender_id: i64,
    /// The sender's display name, if known (attribution only; never the routing identity).
    pub sender_display: Option<String>,
    /// The message text.
    pub text: String,
    /// Whether this is a private (1:1) chat.
    pub is_dm: bool,
    /// Whether the account was mentioned/replied-to (grammers `Message::mentioned`).
    pub mentioned: bool,
}

/// The shared, cloneable context threaded into the per-account update loop.
#[derive(Clone)]
pub struct InboundCtx {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session this account opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-chat / addressing route table.
    pub routes: Arc<Vec<TelegramRoute>>,
    /// This account's bare id — the route matcher key.
    pub bare: String,
    /// This account's instance-qualified transport id.
    pub transport: TransportId,
}

/// Whether `ev` addresses the agent, per `route`: in a mention-gated route only a DM, a `!command`,
/// or an explicit mention/reply turns the agent; otherwise every message is treated as addressed.
pub fn is_addressed(route: &TelegramRoute, ev: &InboundEvent) -> bool {
    if route.mention_gating {
        ev.is_dm || ev.text.trim_start().starts_with('!') || ev.mentioned
    } else {
        true
    }
}

/// Build the normalised [`Reception`] for `ev` on `transport`. The scope is always
/// `Group { chat = chat id }`; attribution (who spoke) rides inside the text, adapter-formatted (the
/// authoritative sender is the structured [`SenderId`], never parsed back out of the body).
pub fn build_reception(transport: &TransportId, ev: &InboundEvent, addressed: bool) -> Reception {
    let origin = Origin::new(
        transport.clone(),
        OriginScope::Group {
            chat: chat_conv_id(ev.chat_id),
            thread: None,
        },
    );
    let who = ev
        .sender_display
        .clone()
        .unwrap_or_else(|| ev.sender_id.to_string());
    let attributed = format!("{who}: {}", ev.text);
    Reception {
        origin,
        // The immutable platform identity: the numeric Telegram user id, supplied structurally.
        sender: SenderId::new(ev.sender_id.to_string()),
        input: UserMsg::new(attributed),
        addressed,
    }
}

/// Gate one inbound event: pick its route (ignore chats no route matches), classify addressing,
/// build a [`Reception`], run it through the [`Ingestor`] under the trusted in-process `internal`
/// principal, and ensure the opened session has an outbound delivery subscription. Mirrors the
/// Matrix adapter's `on_room_message`, minus the SDK event plumbing.
pub async fn handle(ctx: &InboundCtx, ev: InboundEvent) {
    let chat = chat_conv_id(ev.chat_id);
    let route = match config::route_for(&ctx.routes, &ctx.bare, &chat, ev.is_dm) {
        Some(r) => r,
        None => return,
    };
    let addressed = is_addressed(&route, &ev);
    let reception = build_reception(&ctx.transport, &ev, addressed);

    // Bind the in-process `internal` principal: `receive` drives `submit_routed`, whose ownership
    // check denies a `None` principal. The update loop runs in a spawned task with no request
    // context, so it supplies the trusted embedded-caller identity explicitly.
    let received =
        with_request_context(RequestContext::internal(), ctx.ingestor.receive(reception)).await;
    match received {
        Ok(session) => ctx.delivery.ensure(session, ctx.transport.clone()),
        Err(e) => tracing::warn!(error = %e, "telegram: ingest receive failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(text: &str, is_dm: bool, mentioned: bool) -> InboundEvent {
        InboundEvent {
            chat_id: -100,
            sender_id: 42,
            sender_display: Some("Alice".into()),
            text: text.to_string(),
            is_dm,
            mentioned,
        }
    }

    #[test]
    fn mention_gating_requires_dm_command_or_mention() {
        let gated = TelegramRoute::default();
        assert!(gated.mention_gating);
        assert!(
            is_addressed(&gated, &ev("hello", true, false)),
            "dm addresses"
        );
        assert!(
            is_addressed(&gated, &ev("!help", false, false)),
            "bang command addresses"
        );
        assert!(
            is_addressed(&gated, &ev("hey @bot", false, true)),
            "mention addresses"
        );
        assert!(
            !is_addressed(&gated, &ev("ambient chatter", false, false)),
            "ambient group chatter is not addressed"
        );
    }

    #[test]
    fn ungated_route_addresses_everything() {
        let route = TelegramRoute {
            mention_gating: false,
            ..Default::default()
        };
        assert!(is_addressed(&route, &ev("anything", false, false)));
    }

    #[test]
    fn reception_carries_scope_sender_and_attribution() {
        let transport = TransportId::new("telegram/777");
        let r = build_reception(&transport, &ev("ping", true, false), true);
        assert!(r.addressed);
        // Immutable sender identity is the numeric user id, not the display name.
        assert_eq!(r.sender.as_str(), "42");
        // Attribution rides inside the text; the routing scope keys on the chat id.
        assert_eq!(r.input.text, "Alice: ping");
        match &r.origin.scope {
            OriginScope::Group { chat, thread } => {
                assert_eq!(chat, "-100");
                assert!(thread.is_none());
            }
            other => panic!("expected Group scope, got {other:?}"),
        }
        assert_eq!(r.origin.transport.as_str(), "telegram/777");
    }

    #[test]
    fn reception_falls_back_to_numeric_sender_when_no_display() {
        let transport = TransportId::new("telegram/1");
        let mut e = ev("hi", true, false);
        e.sender_display = None;
        let r = build_reception(&transport, &e, true);
        assert_eq!(r.input.text, "42: hi");
    }
}
