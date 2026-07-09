// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Message state ([`ChatMessage`], [`MessageAttachment`]) ported from libpurple `PurpleMessage`
//! (`purplemessage.c`) + `PurpleAttachment` (`purpleattachment.c`), work package W2-E.
//!
//! A **wire type**: it becomes reachable from [`crate::ApiResponse::Journal`] through the additive
//! [`crate::JournalRecordPayload::Chat`] arm, so it enriches the conversation-history surface without
//! reshaping the existing `UserMsg`/`TranscriptBlock` types. Mirrored in `daemon-api.cddl`.
//!
//! `delivered`/`edited` are **derived** from their timestamps (as in libpurple): a message is
//! delivered iff `delivered_at` is set, and edited iff `edited_at` is set. The coupling setters keep
//! them in lockstep. Unlike libpurple's internal `g_date_time_new_now_utc()` clock, the "stamp now"
//! setters take a node-authoritative `now` (unix seconds), matching the DTO timestamp convention.

use crate::Participant;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// One message attachment (← `PurpleAttachment`). `wire vNEXT`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageAttachment {
    /// The adapter-opaque attachment id.
    pub id: String,
    /// The MIME content type, when known.
    #[serde(default)]
    pub content_type: Option<String>,
    /// Whether the attachment renders inline.
    #[serde(default)]
    pub is_inline: bool,
    /// A local (on-disk) URI, when materialized.
    #[serde(default)]
    pub local_uri: Option<String>,
    /// A remote URI, when the source is remote.
    #[serde(default)]
    pub remote_uri: Option<String>,
    /// The size in bytes, when known (`0` otherwise).
    #[serde(default)]
    pub size: u64,
}

/// A chat message (← `PurpleMessage`). `delivered`/`edited` are derived from
/// `delivered_at`/`edited_at`. `wire vNEXT`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The protocol-specific message id, when known.
    #[serde(default)]
    pub id: Option<String>,
    /// The author (`None` = a system/account-originated message).
    #[serde(default)]
    pub author: Option<Participant>,
    /// The id of the message this one replies to, when any.
    #[serde(default)]
    pub replying_to: Option<String>,
    /// The message body (`contents`).
    pub text: String,
    /// The attachments.
    #[serde(default)]
    pub attachments: Vec<MessageAttachment>,
    /// The message timestamp (unix seconds), when set.
    #[serde(default)]
    pub timestamp: Option<u64>,
    /// When the message was delivered (unix seconds); `Some` ⇔ delivered.
    #[serde(default)]
    pub delivered_at: Option<u64>,
    /// When the message was last edited (unix seconds); `Some` ⇔ edited.
    #[serde(default)]
    pub edited_at: Option<u64>,
    /// A delivery/redaction error message, when any (← `GError`, reduced to its text).
    #[serde(default)]
    pub error: Option<String>,
    /// A title for the message (used with `highlighted`).
    #[serde(default)]
    pub title: Option<String>,
    /// The highlight color, when set.
    #[serde(default)]
    pub highlight_color: Option<String>,
    /// Whether the message is an action (`/me`).
    #[serde(default)]
    pub action: bool,
    /// Whether the message is an event (topic change, join/leave, …).
    #[serde(default)]
    pub event: bool,
    /// Whether the message is a notice (do-not-auto-reply).
    #[serde(default)]
    pub notice: bool,
    /// Whether the message is a system message.
    #[serde(default)]
    pub system: bool,
    /// Whether the message should be highlighted.
    #[serde(default)]
    pub highlighted: bool,
}

impl ChatMessage {
    /// Construct a message from an author and body (`purple_message_new`).
    pub fn new(author: Option<Participant>, text: impl Into<String>) -> Self {
        Self {
            author,
            text: text.into(),
            ..Default::default()
        }
    }

    /// Whether the message was delivered (`purple_message_get_delivered`): `delivered_at` is set.
    pub fn delivered(&self) -> bool {
        self.delivered_at.is_some()
    }

    /// Set/clear delivery (`purple_message_set_delivered`): `true` stamps `delivered_at` with `now`,
    /// `false` clears it.
    pub fn set_delivered(&mut self, delivered: bool, now: u64) {
        self.set_delivered_at(delivered.then_some(now));
    }

    /// Set the delivered timestamp directly (`purple_message_set_delivered_at`); `None` clears it (and
    /// thus marks the message not-delivered).
    pub fn set_delivered_at(&mut self, at: Option<u64>) {
        self.delivered_at = at;
    }

    /// Whether the message was edited (`purple_message_get_edited`): `edited_at` is set.
    pub fn edited(&self) -> bool {
        self.edited_at.is_some()
    }

    /// Set/clear edited (`purple_message_set_edited`): `true` stamps `edited_at` with `now`, `false`
    /// clears it.
    pub fn set_edited(&mut self, edited: bool, now: u64) {
        self.set_edited_at(edited.then_some(now));
    }

    /// Set the edited timestamp directly (`purple_message_set_edited_at`); `None` clears it.
    pub fn set_edited_at(&mut self, at: Option<u64>) {
        self.edited_at = at;
    }

    /// Whether the message body is empty (`purple_message_is_empty`).
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Order two messages by timestamp (`purple_message_compare_timestamp`): an unset timestamp
    /// (`None`) sorts before a set one, else earlier before later (`birb_date_time_compare`).
    pub fn compare_timestamp(&self, other: &ChatMessage) -> Ordering {
        self.timestamp.cmp(&other.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactInfo, JournalRecordPayload};

    fn contact(id: &str) -> Participant {
        Participant::Contact(ContactInfo {
            id: id.to_string(),
            ..Default::default()
        })
    }

    #[test]
    fn message_properties_roundtrip() {
        // libpurple `/message/properties`.
        let author = contact("author");
        let mut m = ChatMessage::new(Some(author.clone()), "Now that is a big door");
        m.action = true;
        m.event = true;
        m.notice = true;
        m.system = true;
        m.highlighted = true;
        m.id = Some("id".into());
        m.title = Some("Titled".into());
        m.highlight_color = Some("#FF00FF".into());
        m.replying_to = Some("reply-guy".into());
        m.error = Some("delivery failed".into());
        m.timestamp = Some(911347200);
        // delivered/edited stamp their timestamps (checked via the derived getters).
        m.set_delivered(true, 1000);
        m.set_edited(true, 1000);

        assert_eq!(m.text, "Now that is a big door");
        assert_eq!(m.author, Some(author));
        assert!(m.action);
        assert!(m.event);
        assert!(m.notice);
        assert!(m.system);
        assert!(m.highlighted);
        assert_eq!(m.id.as_deref(), Some("id"));
        assert_eq!(m.title.as_deref(), Some("Titled"));
        assert_eq!(m.highlight_color.as_deref(), Some("#FF00FF"));
        assert_eq!(m.replying_to.as_deref(), Some("reply-guy"));
        assert_eq!(m.error.as_deref(), Some("delivery failed"));
        assert_eq!(m.timestamp, Some(911347200));
        assert!(m.delivered());
        assert!(m.delivered_at.is_some());
        assert!(m.edited());
        assert!(m.edited_at.is_some());
        assert!(m.attachments.is_empty());
    }

    #[test]
    fn message_set_delivered_stamps_delivered_at() {
        // libpurple `/message/delivered-sets-delivered-at`.
        let mut m = ChatMessage::default();
        assert!(!m.delivered());
        assert_eq!(m.delivered_at, None);

        m.set_delivered(true, 500);
        assert!(m.delivered());
        assert_eq!(m.delivered_at, Some(500));

        m.set_delivered(false, 999);
        assert!(!m.delivered());
        assert_eq!(m.delivered_at, None);
    }

    #[test]
    fn message_set_delivered_at_implies_delivered() {
        // libpurple `/message/delivered-at-sets-delivered`.
        let mut m = ChatMessage::default();
        m.set_delivered_at(Some(1234));
        assert!(m.delivered());
        assert_eq!(m.delivered_at, Some(1234));

        m.set_delivered_at(None);
        assert!(!m.delivered());
        assert_eq!(m.delivered_at, None);
    }

    #[test]
    fn message_set_edited_stamps_edited_at() {
        // libpurple `/message/edited-sets-edited-at`.
        let mut m = ChatMessage::default();
        assert!(!m.edited());
        assert_eq!(m.edited_at, None);

        m.set_edited(true, 500);
        assert!(m.edited());
        assert_eq!(m.edited_at, Some(500));

        m.set_edited(false, 999);
        assert!(!m.edited());
        assert_eq!(m.edited_at, None);
    }

    #[test]
    fn message_set_edited_at_implies_edited() {
        // libpurple `/message/edited-at-sets-edited`.
        let mut m = ChatMessage::default();
        m.set_edited_at(Some(1234));
        assert!(m.edited());
        assert_eq!(m.edited_at, Some(1234));

        m.set_edited_at(None);
        assert!(!m.edited());
        assert_eq!(m.edited_at, None);
    }

    #[test]
    fn message_is_empty() {
        assert!(ChatMessage::default().is_empty());
        assert!(ChatMessage::new(None, "").is_empty());
        assert!(!ChatMessage::new(None, "hi").is_empty());
    }

    #[test]
    fn message_compare_timestamp() {
        let mut a = ChatMessage::new(None, "a");
        a.timestamp = Some(10);
        let mut b = ChatMessage::new(None, "b");
        b.timestamp = Some(20);
        assert_eq!(a.compare_timestamp(&b), Ordering::Less);
        assert_eq!(b.compare_timestamp(&a), Ordering::Greater);
        assert_eq!(a.compare_timestamp(&a.clone()), Ordering::Equal);
        // An unset timestamp sorts before a set one.
        let none = ChatMessage::new(None, "n");
        assert_eq!(none.compare_timestamp(&a), Ordering::Less);
        assert_eq!(a.compare_timestamp(&none), Ordering::Greater);
    }

    #[test]
    fn chat_message_journal_payload_round_trips() {
        // The additive wire arm: ChatMessage rides the conversation-history surface.
        let msg = ChatMessage::new(Some(contact("u")), "hi");
        let payload = JournalRecordPayload::Chat {
            message: Box::new(msg.clone()),
        };
        let bytes = crate::to_cbor(&payload);
        let back: JournalRecordPayload = crate::from_cbor(&bytes).expect("decode");
        assert_eq!(
            back,
            JournalRecordPayload::Chat {
                message: Box::new(msg)
            }
        );
    }
}
