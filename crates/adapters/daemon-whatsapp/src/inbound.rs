// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: normalised [`WaInbound`] messages -> `Origin` + `Reception` -> the `daemon-ingest` gate.
//!
//! The adapter keeps only the transport-specific piece — classifying whether a message is
//! *addressed* — and hands a normalised [`Reception`] to the reusable [`Ingestor`]. Every WhatsApp
//! chat (DM or group) maps to `OriginScope::Group { chat = jid }`, so the reply route is trivially the
//! chat id; DM-ness only influences the addressed flag (mirrors the Matrix adapter).

use std::sync::Arc;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::backend::WaInbound;
use crate::config::{self, WhatsappRoute};
use crate::mapping::classify_addressed;
use crate::outbound::DeliveryManager;

/// The context threaded into a per-account inbound drain loop.
#[derive(Clone)]
pub struct InboundCtx {
    /// The reusable inbound gate (shared with the outbound projector that drives its busy state).
    pub ingestor: Arc<Ingestor>,
    /// Ensures an outbound delivery subscription exists for any session this account opens.
    pub delivery: Arc<DeliveryManager>,
    /// The engaged-chat / addressing route table.
    pub routes: Arc<Vec<WhatsappRoute>>,
    /// This account's bare handle — the route matcher key.
    pub account: String,
    /// This account's instance-qualified transport id.
    pub transport: TransportId,
}

/// Build the normalised [`Reception`] for an inbound message under the route table, or `None` when a
/// configured route table does not engage the chat.
pub fn build_reception(
    routes: &[WhatsappRoute],
    account: &str,
    transport: &TransportId,
    inbound: &WaInbound,
) -> Option<Reception> {
    let is_dm = !inbound.is_group;
    let route = config::route_for(routes, account, &inbound.chat, is_dm)?;
    let addressed = classify_addressed(&inbound.text, inbound.is_group, route.mention_gating);

    let origin = Origin::new(
        transport.clone(),
        OriginScope::Group {
            chat: inbound.chat.clone(),
            thread: None,
        },
    );
    // Attribution (who spoke) rides inside the text, adapter-formatted (ingest treats input opaquely).
    let attributed = format!("{}: {}", inbound.sender, inbound.text);
    Some(Reception {
        origin,
        // The immutable platform identity (the sender JID), supplied structurally — never parsed back
        // out of `attributed`. The ingest `SenderPolicy` gate keys on it.
        sender: SenderId::new(&inbound.sender),
        input: UserMsg::new(attributed),
        addressed,
    })
}

/// Gate one inbound message through the [`Ingestor`] and ensure the opened session has an outbound
/// delivery subscription. Runs under the in-process `internal` principal (the trusted embedded-caller
/// identity), exactly like the Matrix adapter's event handler.
pub async fn handle_inbound(ctx: &InboundCtx, inbound: WaInbound) {
    let Some(reception) = build_reception(&ctx.routes, &ctx.account, &ctx.transport, &inbound)
    else {
        return;
    };
    let received =
        with_request_context(RequestContext::internal(), ctx.ingestor.receive(reception)).await;
    match received {
        Ok(session) => ctx.delivery.ensure(session, ctx.transport.clone()),
        Err(e) => tracing::warn!(error = %e, "whatsapp: ingest receive failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inbound(chat: &str, text: &str, is_group: bool) -> WaInbound {
        WaInbound {
            chat: chat.to_string(),
            sender: "15550000000@s.whatsapp.net".to_string(),
            text: text.to_string(),
            is_group,
        }
    }

    #[test]
    fn dm_is_addressed_and_scoped_to_chat() {
        let t = TransportId::new("whatsapp/15551234567");
        let r = build_reception(
            &[],
            "15551234567",
            &t,
            &inbound("1@s.whatsapp.net", "hi", false),
        )
        .expect("default route engages the chat");
        assert!(r.addressed, "a DM is always addressed");
        match &r.origin.scope {
            OriginScope::Group { chat, thread } => {
                assert_eq!(chat, "1@s.whatsapp.net");
                assert!(thread.is_none());
            }
            other => panic!("expected group scope, got {other:?}"),
        }
        // Sender is carried structurally AND prefixed into the attributed body.
        assert_eq!(r.sender.as_str(), "15550000000@s.whatsapp.net");
        assert!(r.input.text.contains("15550000000@s.whatsapp.net: hi"));
    }

    #[test]
    fn group_message_gated_on_command() {
        let t = TransportId::new("whatsapp/me");
        let plain = build_reception(&[], "me", &t, &inbound("120@g.us", "chatter", true)).unwrap();
        assert!(!plain.addressed, "ambient group chatter is not addressed");
        let cmd = build_reception(&[], "me", &t, &inbound("120@g.us", "!help", true)).unwrap();
        assert!(cmd.addressed, "a !command addresses the agent");
    }

    #[test]
    fn configured_route_table_ignores_unmatched_chat() {
        let t = TransportId::new("whatsapp/me");
        let routes = vec![WhatsappRoute {
            chat_glob: Some("*@g.us".into()),
            ..Default::default()
        }];
        assert!(
            build_reception(&routes, "me", &t, &inbound("1@s.whatsapp.net", "hi", false)).is_none(),
            "a DM does not match a group-only route table"
        );
    }
}
