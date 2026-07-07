// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: iLink long-poll updates -> `Origin` + `Reception` -> the `daemon-ingest` gate.
//!
//! WeChat's iLink bot delivers messages by long-poll (`getupdates`), not a push socket, so the
//! adapter owns the poll loop ([`run_inbound`]): it walks each batch, remembers the peer's reply
//! context-token, normalises the message into a [`Reception`], and hands it to the reusable
//! [`Ingestor`]. WeChat is a **DM-only** transport, so every user message addresses the agent
//! (`addressed = true`) and the session/reply key is the peer user id
//! (`OriginScope::Dm { user: <peer> }`). The bot's own echoes are filtered by
//! [`IncomingMessage::from_wire`] (which returns `None` for non-user messages), so there is no
//! self-loop guard to write here.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use wechatbot::types::IncomingMessage;

use daemon_host::{with_request_context, RequestContext};
use daemon_ingest::{Ingestor, Reception};
use daemon_protocol::{Origin, OriginScope, SenderId, TransportId, UserMsg};

use crate::account::LiveAccount;
use crate::outbound::DeliveryManager;

/// The initial reconnect backoff after a failed poll (doubles up to [`MAX_BACKOFF`]).
const BASE_BACKOFF: Duration = Duration::from_secs(1);
/// The reconnect backoff ceiling.
const MAX_BACKOFF: Duration = Duration::from_secs(10);

/// Normalise a parsed WeChat message into a [`Reception`] for the ingest gate. WeChat DMs are always
/// addressed; the routing/reply key is the peer user id. Attribution (`<peer>: <text>`) rides inside
/// the body (ingest treats it opaquely), while the authoritative sender is the immutable peer id.
pub fn build_reception(transport: &TransportId, msg: &IncomingMessage) -> Reception {
    let origin = Origin::new(
        transport.clone(),
        OriginScope::Dm {
            user: msg.user_id.clone(),
        },
    );
    let attributed = format!("{}: {}", msg.user_id, msg.text);
    Reception {
        origin,
        // The immutable platform identity is the peer's iLink user id — supplied structurally, never
        // parsed back out of the attributed body. The ingest `SenderPolicy` gate keys on it.
        sender: SenderId::new(&msg.user_id),
        input: UserMsg::new(attributed),
        // DM-only transport: every user message turns the agent.
        addressed: true,
    }
}

/// Drive one account's inbound long-poll loop until the task is aborted. Each `getupdates` batch is
/// walked oldest-first: the peer's reply context-token is remembered, the message is gated through the
/// [`Ingestor`] under the trusted `internal` principal, and the opened session gets an outbound
/// delivery subscription. A poll error backs off (capped) and retries — a re-login on an expired
/// session is deferred to bring-up (the loop simply keeps failing + backing off until then).
pub async fn run_inbound(
    account: Arc<LiveAccount>,
    ingestor: Arc<Ingestor>,
    delivery: Arc<DeliveryManager>,
    transport: TransportId,
) {
    let base_url = account.session.base_url.clone();
    let token = account.session.token.clone();
    let mut cursor = String::new();
    let mut backoff = BASE_BACKOFF;

    tracing::info!(instance = %transport.as_str(), "wechat: long-poll loop started");
    loop {
        match account.client.get_updates(&base_url, &token, &cursor).await {
            Ok(updates) => {
                if !updates.get_updates_buf.is_empty() {
                    cursor = updates.get_updates_buf;
                }
                backoff = BASE_BACKOFF;
                for wire in &updates.msgs {
                    let Some(msg) = IncomingMessage::from_wire(wire) else {
                        continue;
                    };
                    account
                        .remember_context(&msg.user_id, msg.context_token())
                        .await;
                    let reception = build_reception(&transport, &msg);
                    let received = with_request_context(
                        RequestContext::internal(),
                        ingestor.receive(reception),
                    )
                    .await;
                    match received {
                        Ok(session) => delivery.ensure(session, transport.clone()),
                        Err(e) => tracing::warn!(error = %e, "wechat: ingest receive failed"),
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, instance = %transport.as_str(), "wechat: getupdates failed; backing off");
                sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use wechatbot::types::{
        MessageItemType, MessageState, MessageType, TextItem, WireMessage, WireMessageItem,
    };

    fn text_wire(from: &str, text: &str, ctx: &str, kind: MessageType) -> WireMessage {
        WireMessage {
            from_user_id: from.to_string(),
            to_user_id: "bot".to_string(),
            client_id: "c1".to_string(),
            create_time_ms: 1_700_000_000_000,
            message_type: kind,
            message_state: MessageState::Finish,
            context_token: ctx.to_string(),
            item_list: vec![WireMessageItem {
                item_type: MessageItemType::Text,
                text_item: Some(TextItem {
                    text: text.to_string(),
                }),
                image_item: None,
                voice_item: None,
                file_item: None,
                video_item: None,
                ref_msg: None,
            }],
        }
    }

    #[test]
    fn inbound_user_message_maps_to_dm_reception() {
        let wire = text_wire("peer-42", "hello agent", "ctx-1", MessageType::User);
        let msg = IncomingMessage::from_wire(&wire).expect("user message parses");
        let transport = TransportId::new("wechat/bot-self");

        let reception = build_reception(&transport, &msg);

        assert_eq!(reception.origin.transport.as_str(), "wechat/bot-self");
        assert_eq!(
            reception.origin.scope,
            OriginScope::Dm {
                user: "peer-42".to_string()
            }
        );
        assert_eq!(reception.sender.as_str(), "peer-42");
        assert!(reception.addressed, "DMs always address the agent");
        assert!(
            reception.input.text.contains("peer-42: hello agent"),
            "attribution rides in the body, got {:?}",
            reception.input.text
        );
    }

    #[test]
    fn bot_echoes_are_dropped_before_reception() {
        // The bot's own outbound message coming back on the poll is not user-originated.
        let wire = text_wire("bot", "my earlier reply", "ctx-1", MessageType::Bot);
        assert!(
            IncomingMessage::from_wire(&wire).is_none(),
            "bot echoes never become receptions (no self-loop)"
        );
    }
}
