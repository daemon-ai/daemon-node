// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Telegram primitive → `daemon-api` conversation/contact DTO projection.
//!
//! Pure, SDK-agnostic mappers: the confined grammers layer ([`crate::client`]) extracts primitive
//! values (a chat id `i64`, a display name, a role flag) from its SDK types and calls these to build
//! the wire DTOs. Keeping the projection here (rather than against grammers types) makes it unit
//! testable without a live client — the same split `daemon-matrix` uses for its `mapping` module.

use daemon_api::{ConversationInfo, ConversationMember, ConversationType};
use daemon_protocol::TransportId;

/// The daemon-opaque conversation id for a Telegram chat: the peer id rendered as a base-10 string.
/// Stable per chat and round-trippable back to the `i64` grammers needs (see [`parse_chat_id`]).
pub(crate) fn chat_conv_id(chat_id: i64) -> String {
    chat_id.to_string()
}

/// Parse a daemon-opaque conversation/contact id back into the `i64` peer id grammers indexes on.
pub(crate) fn parse_chat_id(conv: &str) -> Option<i64> {
    conv.trim().parse::<i64>().ok()
}

/// Project a Telegram chat into the wire [`ConversationInfo`]. `is_dm` selects `Dm` vs `Channel`
/// (Telegram groups + channels both project as `Channel`; a private chat is a `Dm`). Members are
/// supplied already-projected by the caller (the friendly grammers API enumerates participants
/// lazily; `list`/`get` project the summary without a full roster fetch).
pub(crate) fn conversation_from(
    transport: &TransportId,
    chat_id: i64,
    is_dm: bool,
    title: Option<String>,
    members: Vec<ConversationMember>,
) -> ConversationInfo {
    ConversationInfo {
        transport: transport.clone(),
        id: chat_conv_id(chat_id),
        kind: if is_dm {
            ConversationType::Dm
        } else {
            ConversationType::Channel
        },
        title,
        topic: None,
        description: None,
        members,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_id_roundtrips() {
        assert_eq!(chat_conv_id(-100123), "-100123");
        assert_eq!(parse_chat_id("-100123"), Some(-100123));
        assert_eq!(parse_chat_id("  42 "), Some(42));
        assert_eq!(parse_chat_id("@notnumeric"), None);
    }

    #[test]
    fn conversation_kind_tracks_dm_flag() {
        let t = TransportId::new("telegram/1");
        let dm = conversation_from(&t, 5, true, Some("Alice".into()), Vec::new());
        assert_eq!(dm.kind, ConversationType::Dm);
        assert_eq!(dm.id, "5");
        assert_eq!(dm.title.as_deref(), Some("Alice"));
        let chan = conversation_from(&t, -100, false, None, Vec::new());
        assert_eq!(chan.kind, ConversationType::Channel);
    }
}
