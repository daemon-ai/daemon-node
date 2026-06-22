//! Inbound: Matrix room messages -> `Origin` + `Reception` -> the `daemon-ingest` gate.
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

use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, TransportId, UserMsg};

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
    let is_dm = room.is_direct().await.unwrap_or(false);

    let route = match config::route_for(&ctx.routes, &ctx.bare, &room_id, is_dm) {
        Some(r) => r,
        // A configured route table that doesn't match this room: the adapter ignores it.
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
        input: UserMsg::new(attributed),
        addressed,
    };

    match ctx.ingestor.receive(reception).await {
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
