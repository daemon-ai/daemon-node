// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: Slack **Socket Mode** push events -> `Origin` + `Reception` -> the `daemon-ingest` gate.
//!
//! Socket Mode (a WSS connection opened with the app-level token) delivers events without a public
//! webhook. The adapter keeps only the *transport-specific* piece — extracting a message + deciding
//! whether it is *addressed* (mention / DM / `!command`) — and hands a normalised [`Reception`] to
//! the reusable [`Ingestor`], which owns the transport-agnostic command selection. A Slack channel id
//! is the natural session/route key, so every message maps to `OriginScope::Group { chat: channel }`
//! (IM channels included); DM-ness only influences the *addressed* flag, keeping outbound routing
//! trivial (the reply route is the channel id).
//!
//! The Socket Mode push callback is a bare `fn` (it cannot capture), so the per-account context
//! ([`InboundState`]) is seeded into the listener's user-state storage and read back inside the
//! callback — the slack-morphism idiom.

use std::collections::HashMap;
use std::sync::Arc;

use slack_morphism::prelude::{
    SlackClientEventsUserState, SlackEventCallbackBody, SlackHyperClient, SlackMessageEvent,
    SlackPushEventCallback, UserCallbackResult,
};

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::config::{self, SlackRoute};
use crate::outbound::DeliveryManager;

/// One bot account's inbound identity, resolved from a push event's `team_id`. A single Socket Mode
/// connection (opened with the node's app-level token) can carry events for every workspace the app
/// is installed in, so events are routed to the right account by team.
#[derive(Clone)]
pub struct InboundAccount {
    /// This account's bare label (`<label>` in `slack/<label>`) — the route matcher key.
    pub bare: String,
    /// This account's instance-qualified transport id.
    pub transport: TransportId,
    /// This account's own bot user id (`U…`) — messages from it are ignored (no self-loop).
    pub bot_user: String,
}

/// The router threaded into the Socket Mode push callback via the listener's user-state storage.
/// Cloneable (all shared state is `Arc`) so the callback can clone it out from under the user-state
/// read lock before awaiting. Shared ingest/delivery/routes across accounts; per-workspace identity
/// resolved from the event's team id.
#[derive(Clone)]
pub struct InboundState {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session an account opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-channel / addressing route table.
    pub routes: Arc<Vec<SlackRoute>>,
    /// The bot accounts served by this Socket Mode connection, keyed by their team id (`T…`).
    pub accounts: Arc<HashMap<String, InboundAccount>>,
}

/// The transport-relevant fields extracted from a Slack `message` event.
struct MsgParts {
    channel: String,
    user: String,
    text: String,
    thread: Option<String>,
    is_dm: bool,
}

/// Pull the `(channel, user, text, thread, is_dm)` out of a plain `message` event, or `None` for a
/// non-plain message (any `subtype` — bot posts, joins, edits, …), a message with no channel/user, or
/// an empty body. `channel_type == "im"` marks a DM.
fn message_parts(msg: &SlackMessageEvent) -> Option<MsgParts> {
    if msg.subtype.is_some() {
        return None;
    }
    let channel = msg.origin.channel.as_ref()?.0.clone();
    let user = msg.sender.user.as_ref()?.0.clone();
    let text = msg.content.as_ref().and_then(|c| c.text.clone())?;
    if text.is_empty() {
        return None;
    }
    let thread = msg.origin.thread_ts.as_ref().map(|t| t.0.clone());
    let is_dm = msg
        .origin
        .channel_type
        .as_ref()
        .map(|t| t.0 == "im")
        .unwrap_or(false);
    Some(MsgParts {
        channel,
        user,
        text,
        thread,
        is_dm,
    })
}

/// Whether a message is *addressed* under `route`: with mention-gating on, a DM, a `!command`, or an
/// explicit `<@bot>` mention counts; with gating off, every message is addressed.
pub(crate) fn is_addressed(route: &SlackRoute, is_dm: bool, text: &str, bot_user: &str) -> bool {
    if !route.mention_gating {
        return true;
    }
    is_dm || text.trim_start().starts_with('!') || text.contains(&format!("<@{bot_user}>"))
}

/// Build the normalised [`Reception`] for an inbound Slack message. The channel id is the session/
/// route key (`OriginScope::Group`); the immutable Slack user id is the structural [`SenderId`], and
/// a human-readable `user: text` attribution rides inside the body (ingest treats it opaquely).
pub(crate) fn build_reception(
    transport: &TransportId,
    channel: &str,
    thread: Option<String>,
    sender: &str,
    text: &str,
    addressed: bool,
) -> Reception {
    let origin = Origin::new(
        transport.clone(),
        OriginScope::Group {
            chat: channel.to_string(),
            thread,
        },
    );
    Reception {
        origin,
        sender: SenderId::new(sender),
        input: UserMsg::new(format!("{sender}: {text}")),
        addressed,
    }
}

/// The registered Socket Mode push-events callback. Extracts a plain message, classifies addressing
/// against the account's route table, gates it through the [`Ingestor`], and ensures the opened
/// session has an outbound delivery subscription. A bare `fn` (Socket Mode callbacks cannot capture),
/// so it reads its [`InboundState`] from the listener's user-state storage.
pub async fn on_push_event(
    event: SlackPushEventCallback,
    _client: Arc<SlackHyperClient>,
    states: SlackClientEventsUserState,
) -> UserCallbackResult<()> {
    let team_id = event.team_id.0.clone();
    let SlackEventCallbackBody::Message(msg) = event.event else {
        return Ok(());
    };
    let Some(parts) = message_parts(&msg) else {
        return Ok(());
    };
    let state = {
        let guard = states.read().await;
        guard.get_user_state::<InboundState>().cloned()
    };
    let Some(state) = state else {
        tracing::warn!("slack: inbound state missing from listener user-state; dropping event");
        return Ok(());
    };
    // Resolve which bound bot account this workspace's event belongs to.
    let Some(account) = state.accounts.get(&team_id).cloned() else {
        return Ok(());
    };
    // Never react to our own posts (the outbound reply path would otherwise loop).
    if parts.user == account.bot_user {
        return Ok(());
    }

    let route = match config::route_for(&state.routes, &account.bare, &parts.channel, parts.is_dm) {
        Some(r) => r,
        // A configured route table that doesn't match this channel: the adapter ignores it.
        None => return Ok(()),
    };
    let addressed = is_addressed(&route, parts.is_dm, &parts.text, &account.bot_user);
    let reception = build_reception(
        &account.transport,
        &parts.channel,
        parts.thread,
        &parts.user,
        &parts.text,
        addressed,
    );

    // Bind the in-process `internal` principal: `receive` drives `submit_routed`, whose ownership
    // check denies a `None` principal, and this Socket Mode callback runs with no request context.
    let received = with_request_context(
        RequestContext::internal(),
        state.ingestor.receive(reception),
    )
    .await;
    match received {
        Ok(session) => state.delivery.ensure(session, account.transport.clone()),
        Err(e) => tracing::warn!(error = %e, "slack: ingest receive failed"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addressing_honours_mention_gating() {
        let gated = SlackRoute::default();
        assert!(gated.mention_gating);
        // DM is always addressed.
        assert!(is_addressed(&gated, true, "hello", "U0BOT"));
        // Bang-command is addressed.
        assert!(is_addressed(&gated, false, "!status", "U0BOT"));
        // Explicit mention is addressed.
        assert!(is_addressed(&gated, false, "hey <@U0BOT> help", "U0BOT"));
        // Ambient chatter is NOT addressed.
        assert!(!is_addressed(&gated, false, "unrelated chatter", "U0BOT"));

        // Gating off: everything is addressed.
        let open = SlackRoute {
            mention_gating: false,
            ..Default::default()
        };
        assert!(is_addressed(&open, false, "unrelated chatter", "U0BOT"));
    }

    #[test]
    fn reception_keys_on_channel_and_immutable_sender() {
        let transport = TransportId::new("slack/T1");
        let r = build_reception(
            &transport,
            "C123",
            Some("1700000000.000100".into()),
            "U777",
            "deploy the thing",
            true,
        );
        assert_eq!(r.sender, SenderId::new("U777"));
        assert!(r.addressed);
        assert!(r.input.text.starts_with("U777: "));
        match r.origin.scope {
            OriginScope::Group { chat, thread } => {
                assert_eq!(chat, "C123");
                assert_eq!(thread.as_deref(), Some("1700000000.000100"));
            }
            other => panic!("expected Group scope, got {other:?}"),
        }
    }
}
