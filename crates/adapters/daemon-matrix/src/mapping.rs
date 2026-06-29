// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! matrix-sdk `Room`/`RoomMember` -> daemon-api conversation/contact DTO projection.
//!
//! The `list`/`get` projection (daemon-messaging-adapter-spec.md §10.2: "Matrix reads its synced
//! room list"; see also `daemon-matrix-bifrost-port-reference.md` §4.3): a synced `Room` becomes a
//! [`ConversationInfo`], its active occupants become [`ConversationMember`]s, and the Matrix power
//! level maps to the observed [`MemberRole`]. The inverse ([`role_to_power`]) backs
//! [`SupportsMembership::set_role`](daemon_api::SupportsMembership::set_role).

use daemon_api::{
    ContactInfo, ContactPermission, ConversationInfo, ConversationMember, ConversationType,
    MemberRole, Presence, TypingState,
};
use daemon_protocol::TransportId;
use matrix_sdk::room::{RoomMember, RoomMemberRole};
use matrix_sdk::{Room, RoomMemberships};

/// Project a synced Matrix `Room` into the wire [`ConversationInfo`]. Occupants are the active
/// membership set; `kind` is `Dm` for a direct room, else `Channel` (Matrix has no native
/// group-DM-vs-channel distinction, so non-DM rooms project as `Channel`).
pub(crate) async fn room_to_info(transport: &TransportId, room: &Room) -> ConversationInfo {
    let kind = if room.is_direct().await.unwrap_or(false) {
        ConversationType::Dm
    } else {
        ConversationType::Channel
    };
    let members = room
        .members(RoomMemberships::ACTIVE)
        .await
        .unwrap_or_default()
        .iter()
        .map(member_to_member)
        .collect();
    ConversationInfo {
        transport: transport.clone(),
        id: room.room_id().as_str().to_string(),
        kind,
        title: room.name(),
        topic: room.topic(),
        // Matrix has no native room "description" distinct from the topic.
        description: None,
        members,
    }
}

/// Project one Matrix `RoomMember` into the observed [`ConversationMember`]. `session` is always
/// `None`: Matrix occupants are humans, never daemon agent incarnations (the `session` binding is a
/// Rooms-only extension, daemon-messaging-adapter-spec.md §8).
pub(crate) fn member_to_member(m: &RoomMember) -> ConversationMember {
    ConversationMember {
        contact: ContactInfo {
            id: m.user_id().as_str().to_string(),
            display_name: m.display_name().map(str::to_string),
            presence: Presence::default(),
            permission: ContactPermission::Unset,
        },
        alias: None,
        nickname: None,
        typing: TypingState::None,
        role: role_from_matrix(m.suggested_role_for_power_level()),
        session: None,
    }
}

/// Map a Matrix power-level role to the observed [`MemberRole`]
/// (Creator/Administrator -> Founder, Moderator -> Op, regular User -> None).
fn role_from_matrix(role: RoomMemberRole) -> MemberRole {
    match role {
        RoomMemberRole::Creator | RoomMemberRole::Administrator => MemberRole::Founder,
        RoomMemberRole::Moderator => MemberRole::Op,
        RoomMemberRole::User => MemberRole::None,
    }
}

/// Build a [`ContactInfo`] from a Matrix user id + optional display name (the directory-search /
/// user-directory projection; `daemon-matrix-bifrost-port-reference.md` §4.3). Presence/permission
/// are unknown from a directory hit, so they default.
pub(crate) fn contact_from(id: String, display_name: Option<String>) -> ContactInfo {
    ContactInfo {
        id,
        display_name,
        presence: Presence::default(),
        permission: ContactPermission::Unset,
    }
}

/// Map an outbound [`MemberRole`] to a Matrix power level for `set_role`
/// (Founder=100, Op=50, HalfOp=25, Voice/None=0).
pub(crate) fn role_to_power(role: MemberRole) -> i32 {
    match role {
        MemberRole::Founder => 100,
        MemberRole::Op => 50,
        MemberRole::HalfOp => 25,
        MemberRole::Voice | MemberRole::None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_roundtrip_at_canonical_levels() {
        assert_eq!(role_to_power(MemberRole::Founder), 100);
        assert_eq!(role_to_power(MemberRole::Op), 50);
        assert_eq!(role_to_power(MemberRole::None), 0);
        assert_eq!(
            role_from_matrix(RoomMemberRole::Administrator),
            MemberRole::Founder
        );
        assert_eq!(role_from_matrix(RoomMemberRole::Moderator), MemberRole::Op);
        assert_eq!(role_from_matrix(RoomMemberRole::User), MemberRole::None);
    }
}
