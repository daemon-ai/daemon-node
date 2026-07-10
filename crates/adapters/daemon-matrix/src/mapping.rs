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
use futures::StreamExt;
use matrix_sdk::room::{ParentSpace, RoomMember, RoomMemberRole};
use matrix_sdk::{Room, RoomMemberships};

/// Project a synced Matrix `Room` into the wire [`ConversationInfo`]. Occupants are the active
/// membership set; `kind` is [`ConversationType::Space`] for an `m.space` room, else `Dm` for a
/// direct room, else `Channel` (Matrix has no native group-DM-vs-channel distinction, so non-DM
/// rooms project as `Channel`). `parent` (wire v38) is the containing space this room advertises
/// via its `m.space.parent` relations, if any (see [`select_parent`]).
pub(crate) async fn room_to_info(transport: &TransportId, room: &Room) -> ConversationInfo {
    // `is_space` short-circuits the DM heuristic: a space is a structural container, never a DM, so
    // we skip the (network-touching) `is_direct` probe for spaces.
    let is_space = room.is_space();
    let is_direct = !is_space && room.is_direct().await.unwrap_or(false);
    let kind = conversation_kind(is_space, is_direct);
    let parent = select_parent(parent_space_ids(room).await);
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
        parent,
    }
}

/// Pure projection of a Matrix room's structural flags to a wire [`ConversationType`] (wire v38).
/// `is_space` wins over `is_direct`: an `m.space` room is a structural container, never a message
/// DM. Kept pure (no `Room`) so the mapping is unit-testable with synthesized inputs.
pub(crate) fn conversation_kind(is_space: bool, is_direct: bool) -> ConversationType {
    if is_space {
        ConversationType::Space
    } else if is_direct {
        ConversationType::Dm
    } else {
        ConversationType::Channel
    }
}

/// Pick a single wire `parent` from the space ids a room advertises as parents (wire v38). Matrix
/// permits multiple `m.space.parent` relations, but the wire `parent` is one containing space, so we
/// pick the lexicographically-lowest id — mirroring the Matrix spec's canonical-parent tie-break
/// (lowest room id by Unicode code-point) — for a stable, deterministic projection. No parents ⟹
/// `None` (a root). Kept pure so it is unit-testable with synthesized inputs.
pub(crate) fn select_parent(mut parent_space_ids: Vec<String>) -> Option<String> {
    parent_space_ids.sort();
    parent_space_ids.into_iter().next()
}

/// Collect the room ids of the spaces a room names as its parents, draining the SDK's
/// [`Room::parent_spaces`] stream (the `m.space.parent` relation walk). We do NOT filter on the
/// verification tier — `Reciprocal` / `WithPowerlevel` / `Illegitimate` / `Unverifiable` all yield
/// the advertised parent id: the node emits what the protocol reports and leaves cycle/dangling
/// handling to the client. Errors and a missing relation both collapse to an empty list.
async fn parent_space_ids(room: &Room) -> Vec<String> {
    let Ok(stream) = room.parent_spaces().await else {
        return Vec::new();
    };
    stream
        .filter_map(|res| async move { res.ok().map(|p| parent_space_id(&p)) })
        .collect()
        .await
}

/// The advertised parent room id behind a [`ParentSpace`], regardless of its verification tier.
fn parent_space_id(parent: &ParentSpace) -> String {
    match parent {
        ParentSpace::Reciprocal(room)
        | ParentSpace::WithPowerlevel(room)
        | ParentSpace::Illegitimate(room) => room.room_id().as_str().to_string(),
        ParentSpace::Unverifiable(id) => id.as_str().to_string(),
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

    /// N4 (wire v38): the pure room-type projection. An `m.space` room is a structural
    /// [`ConversationType::Space`] container regardless of any DM heuristic; a direct room is a
    /// [`ConversationType::Dm`]; every other room is a [`ConversationType::Channel`].
    #[test]
    fn conversation_kind_projects_space_dm_channel() {
        assert_eq!(conversation_kind(true, false), ConversationType::Space);
        // `is_space` wins over `is_direct` — a space is a container, never a message DM.
        assert_eq!(conversation_kind(true, true), ConversationType::Space);
        assert_eq!(conversation_kind(false, true), ConversationType::Dm);
        assert_eq!(conversation_kind(false, false), ConversationType::Channel);
    }

    /// N4 (wire v38): `parent` is a single containing space, but Matrix permits multiple
    /// `m.space.parent` relations, so the projection picks the lexicographically-lowest space id
    /// (the spec's canonical-parent tie-break) for a deterministic result; no parents ⟹ `None`.
    #[test]
    fn parent_selection_is_deterministic() {
        assert_eq!(select_parent(vec![]), None);
        assert_eq!(
            select_parent(vec!["!only:hs".into()]),
            Some("!only:hs".into())
        );
        assert_eq!(
            select_parent(vec!["!b:hs".into(), "!a:hs".into(), "!c:hs".into()]),
            Some("!a:hs".into())
        );
    }
}
