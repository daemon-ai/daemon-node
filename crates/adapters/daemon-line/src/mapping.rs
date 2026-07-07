// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! LINE id/source → daemon-protocol projection helpers.
//!
//! LINE conversation ids are prefix-typed: a user id starts with `U`, a group id with `C`, a room id
//! with `R`. This module maps that prefix to the daemon [`OriginScope`] (a user source is a
//! [`OriginScope::Dm`], a group/room is a [`OriginScope::Group`]) and to the operation LINE actually
//! supports on that id (a bot can `leave` a group/room but not a 1:1). Keeping these as small
//! functions makes them cheap to unit-test and reusable across the inbound (event → scope) and
//! adapter (verb → LINE call) paths. The only SDK type in scope is [`UserProfileResponse`], which the
//! profile renderer projects into text.

use daemon_protocol::OriginScope;
use line_bot_sdk_rust::line_messaging_api::models::UserProfileResponse;

/// The kind of a LINE conversation id, classified by its prefix character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// A 1:1 user (`U…`) — a direct conversation.
    User,
    /// A group (`C…`).
    Group,
    /// A multi-person room (`R…`).
    Room,
    /// Anything else (defensive; treated as non-leaveable / group-scoped).
    Unknown,
}

/// Classify a LINE conversation/user id by its prefix (`U`=user, `C`=group, `R`=room).
pub fn classify_target(id: &str) -> TargetKind {
    match id.as_bytes().first() {
        Some(b'U') => TargetKind::User,
        Some(b'C') => TargetKind::Group,
        Some(b'R') => TargetKind::Room,
        _ => TargetKind::Unknown,
    }
}

/// The daemon [`OriginScope`] for a LINE conversation id: a user id is a DM (reply target = the
/// user), a group/room/unknown is a group (reply target = the group/room id). The `route` the host
/// derives from this scope is exactly the LINE `to` id used for push (see [`crate::outbound`]).
pub fn scope_for(target: &str) -> OriginScope {
    match classify_target(target) {
        TargetKind::User => OriginScope::Dm {
            user: target.to_string(),
        },
        TargetKind::Group | TargetKind::Room | TargetKind::Unknown => OriginScope::Group {
            chat: target.to_string(),
            thread: None,
        },
    }
}

/// Render a LINE user profile into the newline-joined text
/// [`SupportsContacts::get_profile`](daemon_api::SupportsContacts::get_profile) returns.
pub fn profile_lines(profile: &UserProfileResponse) -> String {
    let mut lines = vec![
        format!("user_id: {}", profile.user_id),
        format!("display_name: {}", profile.display_name),
    ];
    if let Some(status) = &profile.status_message {
        if !status.is_empty() {
            lines.push(format!("status_message: {status}"));
        }
    }
    if let Some(picture) = &profile.picture_url {
        if !picture.is_empty() {
            lines.push(format!("picture_url: {picture}"));
        }
    }
    if let Some(language) = &profile.language {
        if !language.is_empty() {
            lines.push(format!("language: {language}"));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_by_prefix() {
        assert_eq!(classify_target("U1234"), TargetKind::User);
        assert_eq!(classify_target("C1234"), TargetKind::Group);
        assert_eq!(classify_target("R1234"), TargetKind::Room);
        assert_eq!(classify_target("x1234"), TargetKind::Unknown);
        assert_eq!(classify_target(""), TargetKind::Unknown);
    }

    #[test]
    fn scope_maps_user_to_dm_and_group_to_group() {
        assert_eq!(
            scope_for("Uabc"),
            OriginScope::Dm {
                user: "Uabc".to_string()
            }
        );
        assert_eq!(
            scope_for("Cabc"),
            OriginScope::Group {
                chat: "Cabc".to_string(),
                thread: None
            }
        );
        assert_eq!(
            scope_for("Rabc"),
            OriginScope::Group {
                chat: "Rabc".to_string(),
                thread: None
            }
        );
    }

    #[test]
    fn profile_renders_known_fields_only() {
        let profile = UserProfileResponse {
            display_name: "Ada".to_string(),
            user_id: "Uabc".to_string(),
            picture_url: None,
            status_message: Some("hi".to_string()),
            language: None,
        };
        let rendered = profile_lines(&profile);
        assert!(rendered.contains("user_id: Uabc"));
        assert!(rendered.contains("display_name: Ada"));
        assert!(rendered.contains("status_message: hi"));
        assert!(!rendered.contains("picture_url"));
        assert!(!rendered.contains("language"));
    }
}
