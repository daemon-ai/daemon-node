// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `DemoAdapter` — the in-process "demo" messaging transport presented as a libpurple-style
//! [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! It seeds a deterministic roster + conversation tree (see [`crate::seed`]) and drives live
//! two-way chat entirely in-process: a [`ConvSend`](SupportsConversations::send) is reported through
//! the node's [`LifecycleSink`] (which journals it + emits `MessagesChanged`), then a scripted
//! contact reply is scheduled to arrive shortly after through the SAME seam — so a chat UI sees
//! genuine back-and-forth traffic against a real node with zero external network. The adapter holds
//! no `Arc<dyn NodeApi>` (only the node-owned sink handed to it at construction), so there is no
//! registry↔adapter reference cycle.
//!
//! ## Seed history choice
//!
//! Conversation history starts **empty** and grows from the first `ConvSend` (no seed burst on
//! connect). This keeps the journal a pure, deterministic function of the sends a client makes —
//! the property the conformance suite asserts against — rather than depending on connect timing.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use daemon_api::{
    AccountSettingsSchema, AccountSettingsValues, AdapterCapabilities, AdapterInfo, ApiError,
    AuthFieldKind, AuthParamField, ChatMessage, ConnectionState, ContactInfo, ConvSendArgs,
    ConversationInfo, ConversationOps, LifecycleSink, MessagingProtocol, NodeApi, Participant,
    PresenceState, RosterOps, SupportsConversations, SupportsRoster, TransportAdapter,
    TransportInstanceInfo,
};
use daemon_protocol::TransportId;

use crate::config::DemoConfig;
use crate::seed;

/// The transport family this adapter answers to (the management-addressable `transport`).
pub const FAMILY: &str = "demo";

/// The documented `validate_account` reject marker: setting ANY `account_schema` field to this
/// value fails validation (proves the N2 validate-rejection path). A normal demo never uses it.
pub const VALIDATE_REJECT_VALUE: &str = "reject-me";

/// A process-wide monotonic counter for synthetic message ids (`demo-msg-<n>`).
static MSG_SEQ: AtomicU64 = AtomicU64::new(1);

/// Unix seconds now — the node-side clock each journal record's `timestamp` is stamped with.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The next synthetic message id.
fn next_msg_id() -> String {
    format!("demo-msg-{}", MSG_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// The in-process demo transport adapter. Holds the resolved config + the node-owned lifecycle sink
/// (the wire-v38 chat-journal seam every message is reported through). All domain state is seeded
/// deterministically from [`crate::seed`]; nothing is persisted.
pub struct DemoAdapter {
    cfg: DemoConfig,
    sink: Option<Arc<dyn LifecycleSink>>,
}

impl DemoAdapter {
    /// Construct the adapter over the resolved `cfg` and the node's [`LifecycleSink`] (pass `None`
    /// where the node is not wired — unit tests). The returned `Arc` is what the host registry holds
    /// and what `serve` consumes.
    pub fn new(cfg: DemoConfig, sink: Option<Arc<dyn LifecycleSink>>) -> Arc<Self> {
        Arc::new(Self { cfg, sink })
    }

    /// The account-setup schema: a small settings form exercising the N2 configure path — a `Text`
    /// `display_name` with a default, a `Number` `reply_delay_ms`, and a `Choice` `mood`.
    fn account_schema() -> AccountSettingsSchema {
        AccountSettingsSchema {
            fields: vec![
                AuthParamField {
                    key: "display_name".into(),
                    label: "Display name".into(),
                    required: false,
                    kind: AuthFieldKind::Text,
                    default: Some("Demo Bot".into()),
                    placeholder: Some("Demo Bot".into()),
                    choices: Vec::new(),
                },
                AuthParamField {
                    key: "reply_delay_ms".into(),
                    label: "Scripted reply delay (ms)".into(),
                    required: false,
                    kind: AuthFieldKind::Number,
                    default: Some("40".into()),
                    placeholder: Some("40".into()),
                    choices: Vec::new(),
                },
                AuthParamField {
                    key: "mood".into(),
                    label: "Reply mood".into(),
                    required: false,
                    kind: AuthFieldKind::Choice,
                    default: Some("cheerful".into()),
                    placeholder: None,
                    choices: vec!["cheerful".into(), "neutral".into(), "grumpy".into()],
                },
            ],
        }
    }

    /// The scripted reply's author for `conv`: the first seeded member of the conversation (the DM
    /// peer for a DM), else the first roster contact — always a real roster [`ContactInfo`].
    fn reply_author(transport: &TransportId, conv: &str) -> Participant {
        let contact = seed::conversation(transport, conv)
            .and_then(|c| c.members.into_iter().next().map(|m| m.contact))
            .or_else(|| seed::roster().into_iter().next())
            .unwrap_or_else(|| ContactInfo {
                id: "u_ada".into(),
                ..ContactInfo::default()
            });
        Participant::Contact(contact)
    }
}

#[async_trait]
impl TransportAdapter for DemoAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Demo (in-process)".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: true,
                room_enumeration: true,
                file_transfer: false,
                interactive_auth: true,
            },
            account_schema: Self::account_schema(),
            // Per-verb ops (wire v33) are enriched centrally by the host from the feature-trait
            // `supported()` probes; the adapter leaves them at default here.
            ..Default::default()
        }
    }

    async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {
        if !self.cfg.enabled {
            return;
        }
        // The demo transport reaches nothing external; there is no connection loop to run. Park so
        // the supervisor owns the task — a reconfigure aborts it (disconnect) and re-enters `serve`
        // (reconnect), which is how apply-by-reconnect stays observable. Scripted replies are
        // spawned from `send` independently of this task.
        std::future::pending::<()>().await;
    }

    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        vec![TransportInstanceInfo {
            transport: seed::demo_transport(),
            family: FAMILY.to_string(),
            display_name: "Demo (in-process)".to_string(),
            connection: ConnectionState::Connected,
            presence: PresenceState::Available,
            bound_profile: None,
            reason: None,
            message: None,
            fatal: false,
            enabled: true,
            label: None,
        }]
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for DemoAdapter {
    async fn validate_account(&self, settings: &AccountSettingsValues) -> Result<(), ApiError> {
        // The documented marker: any field set to `reject-me` fails validation (the N2 reference
        // rejection). Everything else is accepted — the demo has no real account constraints.
        if settings.values.values().any(|v| v == VALIDATE_REJECT_VALUE) {
            return Err(ApiError::Other(format!(
                "validate_account: the demo rejects the marker value {VALIDATE_REJECT_VALUE:?}"
            )));
        }
        Ok(())
    }

    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn roster(self: Arc<Self>) -> Option<Arc<dyn SupportsRoster>> {
        Some(self)
    }
}

#[async_trait]
impl SupportsConversations for DemoAdapter {
    fn supported(&self) -> ConversationOps {
        // The demo exposes a live, seeded tree + sending; it does not create/join/leave/delete or
        // mutate metadata (the tree is fixed), so only `send` is advertised.
        ConversationOps {
            create: false,
            join_channel: false,
            leave: false,
            delete: false,
            send: true,
            set_topic: false,
            set_title: false,
            set_description: false,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        seed::conversations(&transport)
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        seed::conversation(&transport, &conv)
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from,
            message,
        } = args;

        // 1. Report the outbound message through the node's LifecycleSink: the node appends one
        //    verified `JournalRecordPayload::Chat` onto `conv:demo:<conv>` (the stream `ConvHistory`
        //    pages) and emits `MessagesChanged`. The author is exactly `ConvSendArgs.from`; the RAW
        //    text rides the body; a delivered send stamps `delivered_at`.
        let Some(sink) = &self.sink else {
            // No node wired (a unit-test construction): nothing to journal, but the send "succeeds".
            return Ok(());
        };
        let now = now_unix_secs();
        let mut outbound = ChatMessage::new(from, message.text.clone());
        outbound.id = Some(next_msg_id());
        outbound.timestamp = Some(now);
        outbound.set_delivered(true, now);
        sink.chat_message(transport.clone(), conv.clone(), outbound)
            .await;

        // 2. Schedule the scripted contact reply so the chat sees live two-way traffic. It arrives
        //    shortly after (the per-account `reply_delay_ms` cadence) through the SAME seam, from a
        //    real roster contact. Spawned detached: it needs no request context (the node's
        //    `chat_message` journaling does not gate on ownership).
        let sink = sink.clone();
        let delay = Duration::from_millis(self.cfg.reply_delay_ms);
        let author = Self::reply_author(&transport, &conv);
        let echoed = message.text;
        tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let now = now_unix_secs();
            let mut reply = ChatMessage::new(
                Some(author),
                format!("Thanks — the demo received: {echoed}"),
            );
            reply.id = Some(next_msg_id());
            reply.timestamp = Some(now);
            reply.set_delivered(true, now);
            sink.chat_message(transport, conv, reply).await;
        });
        Ok(())
    }
}

#[async_trait]
impl SupportsRoster for DemoAdapter {
    fn supported(&self) -> RosterOps {
        // The demo roster is a fixed, seeded contact list: enumeration only (no server-side
        // add/update/remove).
        RosterOps {
            list: true,
            add: false,
            update: false,
            remove: false,
        }
    }

    async fn list(&self, _transport: TransportId) -> Vec<ContactInfo> {
        // Adapter-ordered + unbounded; the host sorts by contact id and pages centrally.
        seed::roster()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::{ConversationType, PresencePrimitive};

    #[test]
    fn seed_tree_has_the_full_shape() {
        let t = seed::demo_transport();
        let convs = seed::conversations(&t);
        let space = convs
            .iter()
            .find(|c| c.kind == ConversationType::Space)
            .expect("a Space is seeded");
        assert!(space.parent.is_none(), "the Space is a tree root");
        // Both general/random name the Space as their parent.
        let children: Vec<_> = convs
            .iter()
            .filter(|c| c.parent.as_deref() == Some(space.id.as_str()))
            .collect();
        assert!(children.len() >= 2, "the Space has child channels");
        assert!(children.iter().all(|c| c.kind == ConversationType::Channel));
        // A standalone channel, DMs, and a group DM are all present.
        assert!(convs
            .iter()
            .any(|c| c.kind == ConversationType::Channel && c.parent.is_none()));
        assert!(convs.iter().any(|c| c.kind == ConversationType::Dm));
        assert!(convs.iter().any(|c| c.kind == ConversationType::GroupDm));
    }

    #[test]
    fn roster_has_varied_presence() {
        let roster = seed::roster();
        assert!(roster.len() >= 4, "a handful of contacts");
        let mut primitives: Vec<PresencePrimitive> = Vec::new();
        for c in &roster {
            if !primitives.contains(&c.presence.primitive) {
                primitives.push(c.presence.primitive);
            }
        }
        assert!(
            primitives.len() >= 3,
            "presence varies across the roster, got {primitives:?}"
        );
        // At least one contact carries avatar-ish decoration (a status message + emoji).
        assert!(roster
            .iter()
            .any(|c| c.presence.emoji.is_some() && c.presence.message.is_some()));
        // Offline + a live primitive both appear.
        assert!(primitives.contains(&PresencePrimitive::Offline));
        assert!(primitives.contains(&PresencePrimitive::Available));
    }
}
