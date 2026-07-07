// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: Discord gateway messages -> `Origin` + `Reception` -> the `daemon-ingest` gate.
//!
//! The adapter keeps only the *transport-specific* piece — classifying whether a message is
//! *addressed* (mention / DM / `!command`) — and hands a normalised [`Reception`] to the reusable
//! [`Ingestor`], which owns the transport-agnostic command selection over `submit_routed`. A Discord
//! **channel id** is the natural session/route key, so every message (guild channel *and* DM) maps to
//! `OriginScope::Group { chat: channel_id }`; DM-ness only influences the *addressed* flag, not the
//! scope — keeping outbound routing trivial (the reply route is the channel id).

use std::sync::Arc;

use serenity_self::async_trait;
use serenity_self::client::{Context, EventHandler};
use serenity_self::model::channel::Message;
use serenity_self::model::gateway::Ready;
use serenity_self::model::id::UserId;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::config::{self, DiscordRoute};
use crate::outbound::DeliveryManager;

/// The per-account serenity [`EventHandler`]: normalises inbound messages and gates them through the
/// [`Ingestor`]. One handler per account (each account owns its own gateway client), so the account
/// identity (`me`, `bare`, `transport`) is fixed on the handler.
pub struct DiscordHandler {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session this account opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-channel / addressing route table.
    pub routes: Arc<Vec<DiscordRoute>>,
    /// This account's bare Discord user id (`1234`) — the route matcher key.
    pub bare: String,
    /// This account's instance-qualified transport id (`discord/1234`).
    pub transport: TransportId,
    /// This account's own user id — messages from it are ignored (no self-loop).
    pub me: UserId,
}

/// Whether a message addresses the agent, given its route's gating policy. With gating off, every
/// message in an engaged channel is addressed; with gating on, only a DM, a `!command`, or an
/// explicit mention of the account turns the agent. Pure (no SDK types) so it is unit-testable.
pub(crate) fn classify_addressed(
    mention_gating: bool,
    is_dm: bool,
    content: &str,
    mentions_me: bool,
) -> bool {
    if !mention_gating {
        return true;
    }
    is_dm || content.trim_start().starts_with('!') || mentions_me
}

#[async_trait]
impl EventHandler for DiscordHandler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        tracing::info!(
            instance = %self.transport.as_str(),
            user = %ready.user.name,
            "discord: gateway ready"
        );
    }

    async fn message(&self, _ctx: Context, msg: Message) {
        // Never react to our own posts (the outbound reply path would otherwise loop), nor to webhook
        // deliveries (which we cannot attribute to a Discord user).
        if msg.author.id == self.me || msg.webhook_id.is_some() {
            return;
        }
        let content = msg.content.clone();
        if content.is_empty() {
            return;
        }

        let channel = msg.channel_id.get().to_string();
        let is_dm = msg.guild_id.is_none();

        let route = match config::route_for(&self.routes, &self.bare, &channel, is_dm) {
            Some(r) => r,
            // A configured route table that doesn't match this channel: the adapter ignores it.
            None => return,
        };

        let mentions_me = msg.mentions.iter().any(|u| u.id == self.me);
        let addressed = classify_addressed(route.mention_gating, is_dm, &content, mentions_me);

        let origin = Origin::new(
            self.transport.clone(),
            OriginScope::Group {
                chat: channel,
                thread: None,
            },
        );
        // Attribution (who spoke) rides inside the text, adapter-formatted (ingest treats input
        // opaquely); the authoritative sender is the immutable numeric Discord user id.
        let attributed = format!("{}: {}", msg.author.name, content);
        let reception = Reception {
            origin,
            sender: SenderId::new(msg.author.id.get().to_string()),
            input: UserMsg::new(attributed),
            addressed,
        };

        // Bind the in-process `internal` principal: `receive` drives `submit_routed`, whose ownership
        // check denies a `None` principal. This handler runs in a serenity event task with no request
        // context, so it supplies the trusted embedded-caller identity explicitly.
        let received =
            with_request_context(RequestContext::internal(), self.ingestor.receive(reception))
                .await;
        match received {
            Ok(session) => self.delivery.ensure(session, self.transport.clone()),
            Err(e) => tracing::warn!(error = %e, "discord: ingest receive failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gating_off_addresses_everything() {
        assert!(classify_addressed(false, false, "idle chatter", false));
    }

    #[test]
    fn gating_on_requires_dm_command_or_mention() {
        // DM always addressed.
        assert!(classify_addressed(true, true, "hello", false));
        // `!command` prefix.
        assert!(classify_addressed(true, false, "  !help", false));
        // explicit mention.
        assert!(classify_addressed(true, false, "hey bot", true));
        // plain channel chatter, no mention -> not addressed.
        assert!(!classify_addressed(true, false, "unrelated chatter", false));
    }
}
