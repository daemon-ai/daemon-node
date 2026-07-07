// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! serenity `Channel`/`User` -> daemon-api conversation/contact DTO projection.
//!
//! The `list`/`get` projection: a guild text channel or a private (DM) channel becomes a
//! [`ConversationInfo`]; a Discord `User` becomes a [`ContactInfo`]. Occupant enumeration is left
//! empty (fetching a guild's full member list is heavyweight and gated behind a privileged intent),
//! so `list`/`get` report the channel shell without members — honest for a chat transport whose
//! primary job is send/receive.

use daemon_api::{ContactInfo, ContactPermission, ConversationInfo, ConversationType, Presence};
use daemon_protocol::TransportId;
use serenity_self::model::channel::{Channel, ChannelType, GuildChannel, PrivateChannel};
use serenity_self::model::user::User;

/// Whether a Discord [`ChannelType`] is a text-bearing conversation the adapter surfaces (text /
/// news / threads / DMs). Voice / category / stage / directory / forum containers are not chat
/// message targets and are filtered out of `list`.
pub(crate) fn is_text_conversation(kind: ChannelType) -> bool {
    matches!(
        kind,
        ChannelType::Text
            | ChannelType::News
            | ChannelType::NewsThread
            | ChannelType::PublicThread
            | ChannelType::PrivateThread
            | ChannelType::Private
            | ChannelType::GroupDm
    )
}

/// The [`ConversationType`] for a Discord channel kind (a DM/group-DM projects as `Dm`/`GroupDm`,
/// a thread as `Thread`, everything else guild-side as `Channel`).
pub(crate) fn conversation_type(kind: ChannelType) -> ConversationType {
    match kind {
        ChannelType::Private => ConversationType::Dm,
        ChannelType::GroupDm => ConversationType::GroupDm,
        ChannelType::NewsThread | ChannelType::PublicThread | ChannelType::PrivateThread => {
            ConversationType::Thread
        }
        _ => ConversationType::Channel,
    }
}

/// Project a synced guild [`GuildChannel`] into the wire [`ConversationInfo`] (members left empty).
pub(crate) fn guild_channel_to_info(
    transport: &TransportId,
    channel: &GuildChannel,
) -> ConversationInfo {
    ConversationInfo {
        transport: transport.clone(),
        id: channel.id.get().to_string(),
        kind: conversation_type(channel.kind),
        title: Some(channel.name.clone()),
        topic: channel.topic.clone(),
        description: None,
        members: Vec::new(),
    }
}

/// Project a [`PrivateChannel`] (a 1:1 DM) into the wire [`ConversationInfo`]; the title is the
/// recipient's name.
pub(crate) fn private_channel_to_info(
    transport: &TransportId,
    channel: &PrivateChannel,
) -> ConversationInfo {
    ConversationInfo {
        transport: transport.clone(),
        id: channel.id.get().to_string(),
        kind: ConversationType::Dm,
        title: Some(channel.recipient.name.clone()),
        topic: None,
        description: None,
        members: Vec::new(),
    }
}

/// Project a fetched [`Channel`] (`get_channel`) into the wire [`ConversationInfo`].
pub(crate) fn channel_to_info(transport: &TransportId, channel: &Channel) -> ConversationInfo {
    match channel {
        Channel::Guild(c) => guild_channel_to_info(transport, c),
        Channel::Private(c) => private_channel_to_info(transport, c),
        // `Channel` is `#[non_exhaustive]`; an unknown future variant projects as a bare channel.
        _ => ConversationInfo {
            transport: transport.clone(),
            id: channel.id().get().to_string(),
            kind: ConversationType::Unset,
            title: None,
            topic: None,
            description: None,
            members: Vec::new(),
        },
    }
}

/// Build a [`ContactInfo`] from a Discord [`User`] (the `get_profile` projection).
pub(crate) fn contact_from_user(user: &User) -> ContactInfo {
    ContactInfo {
        id: user.id.get().to_string(),
        display_name: Some(user.name.clone()),
        presence: Presence::default(),
        permission: ContactPermission::Unset,
    }
}

/// Render a Discord [`User`] as the human-readable profile text `get_profile` returns.
pub(crate) fn profile_text(user: &User) -> String {
    let mut lines = vec![format!("user_id: {}", user.id.get())];
    lines.push(format!("username: {}", user.name));
    if let Some(global) = &user.global_name {
        lines.push(format!("display_name: {global}"));
    }
    lines.push(format!("bot: {}", user.bot));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_conversation_filter() {
        assert!(is_text_conversation(ChannelType::Text));
        assert!(is_text_conversation(ChannelType::Private));
        assert!(is_text_conversation(ChannelType::PublicThread));
        assert!(!is_text_conversation(ChannelType::Voice));
        assert!(!is_text_conversation(ChannelType::Category));
    }

    #[test]
    fn conversation_type_mapping() {
        assert_eq!(
            conversation_type(ChannelType::Private),
            ConversationType::Dm
        );
        assert_eq!(
            conversation_type(ChannelType::GroupDm),
            ConversationType::GroupDm
        );
        assert_eq!(
            conversation_type(ChannelType::PublicThread),
            ConversationType::Thread
        );
        assert_eq!(
            conversation_type(ChannelType::Text),
            ConversationType::Channel
        );
        assert_eq!(
            conversation_type(ChannelType::News),
            ConversationType::Channel
        );
    }
}
