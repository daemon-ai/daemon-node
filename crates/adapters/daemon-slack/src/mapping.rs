// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ChannelSummary`] -> daemon-api conversation/contact DTO projection (the `list`/`get`/directory
//! read side). Pure, SDK-free functions over the in-crate [`ChannelSummary`], so they are unit-
//! testable without any Slack client.

use daemon_api::{ContactInfo, ContactPermission, ConversationInfo, ConversationType, Presence};
use daemon_protocol::TransportId;

use crate::conn::ChannelSummary;

/// The conversation kind for a Slack channel: `Dm` for an IM, else `Channel` (Slack has no native
/// group-DM-vs-channel distinction surfaced here).
pub(crate) fn conversation_type(is_im: bool) -> ConversationType {
    if is_im {
        ConversationType::Dm
    } else {
        ConversationType::Channel
    }
}

/// Project a [`ChannelSummary`] into the wire [`ConversationInfo`]. Members are not fetched on the
/// list projection (a separate `conversations.members` call); the occupant set is left empty.
pub(crate) fn channel_to_info(transport: &TransportId, c: &ChannelSummary) -> ConversationInfo {
    ConversationInfo {
        transport: transport.clone(),
        id: c.id.clone(),
        kind: conversation_type(c.is_im),
        title: c.name.clone(),
        topic: c.topic.clone(),
        description: None,
        members: Vec::new(),
        // Slack workspace hierarchy is not projected through this adapter (wire v38).
        parent: None,
    }
}

/// Project a [`ChannelSummary`] into a [`ContactInfo`] for the channel/room directory search
/// (`SupportsDirectory`, backed by `conversations.list`): the channel id is the opaque contact id and
/// the channel name is the display name.
pub(crate) fn channel_to_contact(c: &ChannelSummary) -> ContactInfo {
    ContactInfo {
        id: c.id.clone(),
        display_name: c.name.clone(),
        presence: Presence::default(),
        permission: ContactPermission::Unset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> ChannelSummary {
        ChannelSummary {
            id: "C123".into(),
            name: Some("secops".into()),
            topic: Some("incident response".into()),
            is_im: false,
            is_private: true,
        }
    }

    #[test]
    fn channel_projects_to_conversation_info() {
        let t = TransportId::new("slack/T1");
        let info = channel_to_info(&t, &summary());
        assert_eq!(info.id, "C123");
        assert_eq!(info.kind, ConversationType::Channel);
        assert_eq!(info.title.as_deref(), Some("secops"));
        assert_eq!(info.topic.as_deref(), Some("incident response"));
    }

    #[test]
    fn im_projects_as_dm() {
        assert_eq!(conversation_type(true), ConversationType::Dm);
        assert_eq!(conversation_type(false), ConversationType::Channel);
    }

    #[test]
    fn channel_projects_to_directory_contact() {
        let c = channel_to_contact(&summary());
        assert_eq!(c.id, "C123");
        assert_eq!(c.display_name.as_deref(), Some("secops"));
    }
}
