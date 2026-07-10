// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The deterministic in-memory seed the [`DemoAdapter`](crate::DemoAdapter) presents: a roster of
//! contacts (stable ids/names, varied presence) and the full conversation tree shape a client's
//! account → spaces → rooms → DMs view exercises — a [`Space`](ConversationType::Space) ("Demo
//! Server") with child [`Channel`](ConversationType::Channel)s (each naming the space via
//! [`ConversationInfo::parent`]), a standalone channel, [`Dm`](ConversationType::Dm)s, and a
//! [`GroupDm`](ConversationType::GroupDm). Everything is a pure function of these constants, so two
//! calls (or two nodes) see byte-identical state — the property the conformance suite leans on.

use daemon_api::{
    ContactInfo, ContactPermission, ConversationInfo, ConversationMember, ConversationType,
    Presence, PresencePrimitive, TypingState,
};
use daemon_protocol::TransportId;

use crate::FAMILY;

/// The demo Space (server) id — the structural container the child channels name as their parent.
pub const SPACE_ID: &str = "space-demo";
/// The `#general` child channel of the demo Space.
pub const CHANNEL_GENERAL: &str = "chan-general";
/// The `#random` child channel of the demo Space.
pub const CHANNEL_RANDOM: &str = "chan-random";
/// A standalone (root, no parent) channel — proves a top-level channel renders beside the Space.
pub const CHANNEL_STANDALONE: &str = "chan-announcements";
/// A 1:1 DM with the roster contact `u_bravo`.
pub const DM_BRAVO: &str = "dm-bravo";
/// A 1:1 DM with the roster contact `u_cara`.
pub const DM_CARA: &str = "dm-cara";
/// A group DM with several roster contacts.
pub const GROUP_DM: &str = "gdm-weekend";

/// One seeded roster contact: `(id, display name, presence, status message, emoji)`.
struct SeedContact {
    id: &'static str,
    name: &'static str,
    primitive: PresencePrimitive,
    message: Option<&'static str>,
    emoji: Option<&'static str>,
}

/// The seeded roster: a handful of contacts with stable ids/names and varied presence. `u_ada`
/// carries "avatar-ish" decoration (a status message + a mood emoji on its [`Presence`]) since the
/// wire [`ContactInfo`] models no avatar field (see the crate docs).
const CONTACTS: &[SeedContact] = &[
    SeedContact {
        id: "u_ada",
        name: "Ada Lovelace",
        primitive: PresencePrimitive::Available,
        message: Some("Weaving algebraic patterns"),
        emoji: Some("🧮"),
    },
    SeedContact {
        id: "u_bravo",
        name: "Bravo Six",
        primitive: PresencePrimitive::Away,
        message: Some("Back in 10"),
        emoji: None,
    },
    SeedContact {
        id: "u_cara",
        name: "Cara Bell",
        primitive: PresencePrimitive::DoNotDisturb,
        message: Some("In a meeting"),
        emoji: None,
    },
    SeedContact {
        id: "u_dev",
        name: "Dev Null",
        primitive: PresencePrimitive::Offline,
        message: None,
        emoji: None,
    },
    SeedContact {
        id: "u_edda",
        name: "Edda North",
        primitive: PresencePrimitive::Idle,
        message: None,
        emoji: None,
    },
];

/// Build one wire [`ContactInfo`] from a seed row.
fn contact(seed: &SeedContact) -> ContactInfo {
    ContactInfo {
        id: seed.id.to_string(),
        display_name: Some(seed.name.to_string()),
        presence: Presence {
            primitive: seed.primitive,
            message: seed.message.map(str::to_string),
            emoji: seed.emoji.map(str::to_string),
            mobile: false,
            idle_since: None,
        },
        permission: ContactPermission::Allow,
    }
}

/// The seeded roster (adapter-ordered; the host sorts + pages it centrally).
pub fn roster() -> Vec<ContactInfo> {
    // RED stub: no seed yet (GREEN fills the roster).
    Vec::new()
}

/// The roster contact with `id`, if seeded.
pub fn contact_by_id(id: &str) -> Option<ContactInfo> {
    CONTACTS.iter().find(|c| c.id == id).map(contact)
}

/// The full seeded conversation tree for `transport` (the demo instance's id).
pub fn conversations(transport: &TransportId) -> Vec<ConversationInfo> {
    // RED stub: no seed tree yet (GREEN fills the Space/channels/DMs).
    if true {
        let _ = transport;
        return Vec::new();
    }
    let members: Vec<ConversationMember> = roster().into_iter().map(member).collect();
    let dm_members = |peer: &str| -> Vec<ConversationMember> {
        contact_by_id(peer).map(member).into_iter().collect()
    };
    let group_members: Vec<ConversationMember> = ["u_ada", "u_bravo", "u_edda"]
        .iter()
        .filter_map(|id| contact_by_id(id))
        .map(member)
        .collect();

    vec![
        // The Space (server): a structural container, no transcript, no parent (a tree root).
        ConversationInfo {
            transport: transport.clone(),
            id: SPACE_ID.to_string(),
            kind: ConversationType::Space,
            title: Some("Demo Server".to_string()),
            topic: Some("The in-process demo space".to_string()),
            description: Some("A structural container holding the demo channels.".to_string()),
            members: Vec::new(),
            parent: None,
        },
        // Child channels: each names the Space as its parent + lists the roster as members.
        ConversationInfo {
            transport: transport.clone(),
            id: CHANNEL_GENERAL.to_string(),
            kind: ConversationType::Channel,
            title: Some("general".to_string()),
            topic: Some("Company-wide chatter".to_string()),
            description: None,
            members: members.clone(),
            parent: Some(SPACE_ID.to_string()),
        },
        ConversationInfo {
            transport: transport.clone(),
            id: CHANNEL_RANDOM.to_string(),
            kind: ConversationType::Channel,
            title: Some("random".to_string()),
            topic: Some("Off-topic".to_string()),
            description: None,
            members,
            parent: Some(SPACE_ID.to_string()),
        },
        // A standalone (root) channel: a top-level channel with no parent.
        ConversationInfo {
            transport: transport.clone(),
            id: CHANNEL_STANDALONE.to_string(),
            kind: ConversationType::Channel,
            title: Some("announcements".to_string()),
            topic: Some("Read-only broadcasts".to_string()),
            description: None,
            members: Vec::new(),
            parent: None,
        },
        // Two 1:1 DMs with roster contacts.
        ConversationInfo {
            transport: transport.clone(),
            id: DM_BRAVO.to_string(),
            kind: ConversationType::Dm,
            title: Some("Bravo Six".to_string()),
            topic: None,
            description: None,
            members: dm_members("u_bravo"),
            parent: None,
        },
        ConversationInfo {
            transport: transport.clone(),
            id: DM_CARA.to_string(),
            kind: ConversationType::Dm,
            title: Some("Cara Bell".to_string()),
            topic: None,
            description: None,
            members: dm_members("u_cara"),
            parent: None,
        },
        // A group DM.
        ConversationInfo {
            transport: transport.clone(),
            id: GROUP_DM.to_string(),
            kind: ConversationType::GroupDm,
            title: Some("Weekend Plans".to_string()),
            topic: Some("Saturday hike?".to_string()),
            description: None,
            members: group_members,
            parent: None,
        },
    ]
}

/// The seeded conversation with id `conv` for `transport`, if any.
pub fn conversation(transport: &TransportId, conv: &str) -> Option<ConversationInfo> {
    conversations(transport).into_iter().find(|c| c.id == conv)
}

/// Wrap a contact as an observed conversation member (no alias/role/session — a plain occupant).
fn member(contact: ContactInfo) -> ConversationMember {
    ConversationMember {
        contact,
        alias: None,
        nickname: None,
        typing: TypingState::None,
        role: daemon_api::MemberRole::None,
        session: None,
    }
}

/// The transport id of the single seeded demo instance (`demo`).
pub fn demo_transport() -> TransportId {
    TransportId::new(FAMILY)
}
