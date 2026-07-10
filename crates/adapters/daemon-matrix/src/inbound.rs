// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: Matrix room messages -> `Origin` + `Reception` -> the `daemon-ingest` gate, PLUS the
//! wire-v38 journal obligation (every inbound message is reported through the node
//! [`LifecycleSink`](daemon_api::LifecycleSink) so the conversation's durable chat history grows
//! and `MessagesChanged` fires — in addition to the ingest routing, never instead of it).
//!
//! The adapter keeps only the *transport-specific* piece — classifying whether a message is
//! *addressed* (mention / DM / `!command`) — and hands a normalised [`Reception`] to the reusable
//! [`Ingestor`], which owns the transport-agnostic command selection (`StartTurn` / `Observe` /
//! busy policy) over `submit_routed` (spec §3.1, event-io §5.9.1). A Matrix room id is the natural
//! session/route key, so every room (including DM rooms) maps to `OriginScope::Group { chat: room_id }`;
//! DM-ness only influences the *addressed* flag, not the scope (keeping outbound routing trivial:
//! the reply route is the room id).

use std::sync::Arc;

use matrix_sdk::event_handler::Ctx;
use matrix_sdk::ruma::events::room::message::{MessageType, OriginalSyncRoomMessageEvent};
use matrix_sdk::ruma::OwnedUserId;
use matrix_sdk::Room;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::config::{self, MatrixRoute};
use crate::outbound::DeliveryManager;

/// The shared, cloneable context threaded into the per-account `m.room.message` handler.
#[derive(Clone)]
pub struct InboundCtx {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session this account opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-room / addressing route table.
    pub routes: Arc<Vec<MatrixRoute>>,
    /// This account's bare user id (`@bot:hs.org`) — the route matcher key.
    pub bare: String,
    /// This account's instance-qualified transport id.
    pub transport: TransportId,
    /// This account's own user id — messages from it are ignored (no self-loop).
    pub me: OwnedUserId,
    /// The node-owned lifecycle sink (wire v38): every inbound message is reported through it so
    /// the node journals a `Chat` record on `conv:<transport>:<room>` and emits `MessagesChanged` —
    /// in ADDITION to the agent-session `Ingestor` routing below, never instead of it. `None` in
    /// unit tests that never wire the node.
    pub sink: Option<Arc<dyn daemon_api::LifecycleSink>>,
}

/// Whether `body`/`mentions` address `me`: an explicit `m.mentions` entry, or the user id / localpart
/// appearing in the text (a lightweight fallback for clients that don't populate `m.mentions`).
fn mentions_me(
    mentions: Option<&matrix_sdk::ruma::events::Mentions>,
    body: &str,
    me: &OwnedUserId,
) -> bool {
    if let Some(m) = mentions {
        if m.user_ids.contains(me) {
            return true;
        }
    }
    body.contains(me.as_str()) || body.contains(me.localpart())
}

/// The registered `m.room.message` handler. Classifies addressing, builds a [`Reception`], gates it
/// through the [`Ingestor`], and ensures the opened session has an outbound delivery subscription.
pub async fn on_room_message(ev: OriginalSyncRoomMessageEvent, room: Room, ctx: Ctx<InboundCtx>) {
    let ctx = ctx.0;
    // Never react to our own posts (the outbound reply path would otherwise loop).
    if ev.sender == ctx.me {
        return;
    }
    let body = match &ev.content.msgtype {
        MessageType::Text(t) => t.body.clone(),
        // v1 handles text only; other msgtypes (media, notice) are ignored for now.
        _ => return,
    };

    let room_id = room.room_id().as_str().to_string();

    // Journal obligation (wire v38): report EVERY inbound message through the node sink, which
    // appends a `Chat` record onto `conv:<transport>:<room>` and emits `MessagesChanged`. This is
    // the transport-level conversation history, so it happens BEFORE the route table below — that
    // table only gates *agent engagement*. The record carries the RAW body + the sender's MXID as
    // the structured author (attribution never rides the text).
    if let Some(sink) = &ctx.sink {
        let mut msg = daemon_api::ChatMessage::new(
            Some(daemon_api::Participant::Contact(daemon_api::ContactInfo {
                id: ev.sender.to_string(),
                ..Default::default()
            })),
            body.clone(),
        );
        msg.id = Some(ev.event_id.to_string());
        msg.timestamp = Some(u64::from(ev.origin_server_ts.get()) / 1000);
        // Inbound delivery: no local op token (null provenance, rung 3 api/39).
        sink.chat_message(ctx.transport.clone(), room_id.clone(), msg, None)
            .await;
    }

    let is_dm = room.is_direct().await.unwrap_or(false);

    let route = match config::route_for(&ctx.routes, &ctx.bare, &room_id, is_dm) {
        Some(r) => r,
        // A configured route table that doesn't match this room: the adapter ignores it (the
        // history record above still landed — journaling is transport-level, not route-gated).
        None => return,
    };

    let addressed = if route.mention_gating {
        is_dm
            || body.trim_start().starts_with('!')
            || mentions_me(ev.content.mentions.as_ref(), &body, &ctx.me)
    } else {
        true
    };

    let origin = Origin::new(
        ctx.transport.clone(),
        OriginScope::Group {
            chat: room_id,
            thread: None,
        },
    );
    // Attribution (who spoke) rides inside the text, adapter-formatted (ingest treats input opaquely).
    let attributed = format!("{}: {}", ev.sender, body);
    let reception = Reception {
        origin,
        // The immutable platform identity: the Matrix MXID (`@user:hs`), never the room display name.
        // This is what the ingest `SenderPolicy` gate keys on — supplied structurally, never parsed
        // back out of `attributed`.
        sender: SenderId::new(ev.sender.as_str()),
        input: UserMsg::new(attributed),
        addressed,
    };

    // Bind the in-process `internal` principal: `receive` drives `submit_routed`, whose Auth 4
    // ownership check now denies a `None` principal. This inbound handler runs in a matrix-sdk event
    // task with no request context, so it supplies the trusted embedded-caller identity explicitly
    // (a fresh chat session is then stamped `owner = "internal"` — see the plan's stamping note).
    let received =
        with_request_context(RequestContext::internal(), ctx.ingestor.receive(reception)).await;
    match received {
        Ok(session) => ctx.delivery.ensure(session, ctx.transport.clone()),
        Err(e) => tracing::warn!(error = %e, "matrix: ingest receive failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_sdk::ruma::{owned_user_id, user_id};

    #[test]
    fn mention_via_mentions_field() {
        let me = owned_user_id!("@bot:hs.org");
        let mut m = matrix_sdk::ruma::events::Mentions::new();
        m.user_ids.insert(user_id!("@bot:hs.org").to_owned());
        assert!(mentions_me(Some(&m), "hello there", &me));
    }

    #[test]
    fn mention_via_text_fallback() {
        let me = owned_user_id!("@bot:hs.org");
        assert!(mentions_me(None, "hey @bot:hs.org help", &me));
        assert!(mentions_me(None, "hey bot help", &me)); // localpart
        assert!(!mentions_me(None, "unrelated chatter", &me));
    }
}
