// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Conversation/account **DTO behavior logic** ported from libpurple (work package W1-B).
//!
//! This module adds methods/functions over the existing wire DTOs in [`crate`] — it introduces **no
//! wire-contract changes**. Everything here is a pure, node-authoritative decision the thin clients
//! consume rather than re-derive:
//!
//! - [`CreateConversationDetails::is_valid`] ← `purplecreateconversationdetails.c`
//! - [`ChannelJoinDetails::merge`] ← `purplechanneljoindetails.c`
//! - [`ConversationType`] predicates + tag derivation, and [`ConversationInfo`] title logic ←
//!   `purpleconversation.c`
//! - a typed account-settings model ([`AccountSettings`]) ← `purpleaccountsetting*.c`
//! - [`Presence`]/[`PresencePrimitive`] predicates + ordering ← `purplepresence.c`, and
//!   [`DisconnectReason::is_fatal`] (the node's reconnect policy).
//!
//! The sibling package owns name/display/matching helpers on `ContactInfo`/`ConversationMember`
//! (`src/matching.rs`); this module deliberately does not implement those. `generate_title`'s
//! per-member name uses only the DTO's `display_name`-else-`id`, matching the libpurple test
//! behavior; the fuller `name_for_display` precedence is the sibling's.

use crate::{AccountSettingsSchema, AccountSettingsValues, AuthParamField};
use crate::{
    ChannelJoinDetails, ConversationInfo, ConversationType, CreateConversationDetails,
    DisconnectReason, Presence, PresencePrimitive,
};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// CreateConversationDetails::is_valid  (← purplecreateconversationdetails.c)
// ---------------------------------------------------------------------------

/// Why a [`CreateConversationDetails`] failed validation (← `PurpleCreateConversationDetailsError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreateConversationDetailsError {
    /// No participants were provided (`*_ERROR_NO_PARTICIPANTS`).
    NoParticipants,
    /// More participants than the protocol's `max_participants` (`*_ERROR_TOO_MANY_PARTICIPANTS`).
    TooManyParticipants,
}

impl CreateConversationDetails {
    /// Validate the create request the way libpurple does
    /// (`purple_create_conversation_details_is_valid`): at least one participant is required, and —
    /// unless `max_participants == 0` (unlimited) — the participant count must not exceed it.
    pub fn is_valid(&self) -> Result<(), CreateConversationDetailsError> {
        // STUB
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChannelJoinDetails::merge  (← purplechanneljoindetails.c)
// ---------------------------------------------------------------------------

impl ChannelJoinDetails {
    /// Merge `source` into `self` (`self` is the destination), mirroring
    /// `purple_channel_join_details_merge(source, destination)`: copies `name`,
    /// `nickname_supported`, `nickname`, `password_supported`, and `password`. The `*_max_length`
    /// fields are intentionally left untouched.
    pub fn merge(&mut self, source: &ChannelJoinDetails) {
        // STUB
        let _ = source;
    }
}

// ---------------------------------------------------------------------------
// ConversationType predicates + tag derivation  (← purpleconversation.c)
// ---------------------------------------------------------------------------

impl ConversationType {
    /// Whether this is a 1:1 direct message (`purple_conversation_is_dm`).
    pub fn is_dm(self) -> bool {
        // STUB
        false
    }

    /// Whether this is a group direct message (`purple_conversation_is_group_dm`).
    pub fn is_group_dm(self) -> bool {
        // STUB
        false
    }

    /// Whether this is a multi-user channel (`purple_conversation_is_channel`).
    pub fn is_channel(self) -> bool {
        // STUB
        false
    }

    /// Whether this is a thread (`purple_conversation_is_thread`).
    pub fn is_thread(self) -> bool {
        // STUB
        false
    }

    /// The `"type"` tag value libpurple derives in `purple_conversation_set_conversation_type`:
    /// `Dm→"dm"`, `GroupDm→"group-dm"`, `Channel→"channel"`, `Thread→"thread"`, `Unset→None`.
    pub fn tag_value(self) -> Option<&'static str> {
        // STUB
        None
    }
}

/// The display title of a conversation, matching `purple_conversation_get_title_for_display`:
/// the first non-empty of `alias`, then `title`, then `id` (`id` is the final fallback even if
/// empty).
pub fn title_for_display(alias: Option<&str>, title: Option<&str>, id: &str) -> String {
    // STUB
    let _ = (alias, title);
    id.to_string()
}

impl ConversationInfo {
    /// Whether this conversation is a DM.
    pub fn is_dm(&self) -> bool {
        self.kind.is_dm()
    }

    /// Whether this conversation is a group DM.
    pub fn is_group_dm(&self) -> bool {
        self.kind.is_group_dm()
    }

    /// Whether this conversation is a channel.
    pub fn is_channel(&self) -> bool {
        self.kind.is_channel()
    }

    /// Whether this conversation is a thread.
    pub fn is_thread(&self) -> bool {
        self.kind.is_thread()
    }

    /// The display title (`purple_conversation_get_title_for_display`). `alias` is the
    /// conversation-local alias (not modeled on the wire DTO), passed in by the caller; falls back
    /// to `self.title` then `self.id`.
    pub fn title_for_display(&self, alias: Option<&str>) -> String {
        title_for_display(alias, self.title.as_deref(), &self.id)
    }

    /// Generate a title from members for a DM / group-DM (`purple_conversation_generate_title`):
    /// skip the account's own member (`self_id`), take each remaining member's display name
    /// (`display_name`-else-`id`, empty skipped), and join with `", "`. Returns `Some(title)` only
    /// when at least one name was found (else the title is left unchanged → `None`); returns `None`
    /// for non-DM/group-DM conversations.
    pub fn generate_title(&self, self_id: &str) -> Option<String> {
        // STUB
        let _ = self_id;
        None
    }
}

// ---------------------------------------------------------------------------
// Typed account settings  (← purpleaccountsetting*.c)
//
// A NON-WIRE, in-memory model faithful to libpurple's PurpleAccountSetting hierarchy. It carries
// the per-setting type so the "wrong type → fallback" semantics port exactly (the string-keyed wire
// `AccountSettingsValues` cannot express types). `AccountSettings::to_values` projects back onto the
// wire DTO.
// ---------------------------------------------------------------------------

/// One `id → label` option of a [`AccountSettingStringList`] (← `BirbLocalizedString`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalizedString {
    /// The stable option id.
    pub id: String,
    /// The human label.
    pub label: String,
}

/// The choice-list body of a string-list setting (← `PurpleAccountSettingStringList`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccountSettingStringList {
    /// The available options.
    pub items: Vec<LocalizedString>,
    /// The active option id, when one is selected.
    active: Option<String>,
}

impl AccountSettingStringList {
    /// Add an option, deduplicating by id (`purple_account_setting_string_list_add_item`). Returns
    /// `false` (and does nothing) if an option with `id` already exists.
    pub fn add_item(&mut self, id: &str, label: &str) -> bool {
        // STUB
        let _ = (id, label);
        false
    }

    /// The active option id, if any (`purple_account_setting_string_list_get_active_item`).
    pub fn active_item(&self) -> Option<&str> {
        // STUB
        None
    }

    /// Set (or, with `None`, clear) the active option
    /// (`purple_account_setting_string_list_set_active_item`). A non-`None` id is only accepted when
    /// it names an existing option (the libpurple production path; the g_test bypass is not ported).
    pub fn set_active_item(&mut self, id: Option<&str>) {
        // STUB
        let _ = id;
    }
}

/// A typed account-setting value (← the `PurpleAccountSetting{Boolean,Int,String,StringList}`
/// subclasses).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccountSettingValue {
    /// A boolean setting.
    Boolean(bool),
    /// An integer setting.
    Int(i64),
    /// A string setting (nullable, like the C `const char *`).
    Str(Option<String>),
    /// A choice-list setting.
    StringList(AccountSettingStringList),
}

/// One typed account setting (← `PurpleAccountSetting` + subclass value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountSetting {
    /// The stable setting id.
    pub id: String,
    /// The human label.
    pub label: String,
    /// The typed value.
    pub value: AccountSettingValue,
}

impl AccountSetting {
    /// A boolean setting (`purple_account_setting_boolean_new`).
    pub fn boolean(id: &str, label: &str, value: bool) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            value: AccountSettingValue::Boolean(value),
        }
    }

    /// An integer setting (`purple_account_setting_int_new`).
    pub fn int(id: &str, label: &str, value: i64) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            value: AccountSettingValue::Int(value),
        }
    }

    /// A string setting with an optional default (`purple_account_setting_string_new`).
    pub fn string(id: &str, label: &str, value: Option<&str>) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            value: AccountSettingValue::Str(value.map(str::to_string)),
        }
    }

    /// An empty string-list setting (`purple_account_setting_string_list_new`).
    pub fn string_list(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            value: AccountSettingValue::StringList(AccountSettingStringList::default()),
        }
    }

    /// The boolean value, if this is a boolean setting.
    pub fn as_bool(&self) -> Option<bool> {
        match &self.value {
            AccountSettingValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// The integer value, if this is an integer setting.
    pub fn as_int(&self) -> Option<i64> {
        match &self.value {
            AccountSettingValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The string value, if this is a string setting (inner `None` = a null-valued string setting).
    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            AccountSettingValue::Str(s) => s.as_deref(),
            _ => None,
        }
    }

    /// The choice-list body, if this is a string-list setting.
    pub fn as_string_list(&self) -> Option<&AccountSettingStringList> {
        match &self.value {
            AccountSettingValue::StringList(list) => Some(list),
            _ => None,
        }
    }

    /// The mutable choice-list body, if this is a string-list setting.
    pub fn as_string_list_mut(&mut self) -> Option<&mut AccountSettingStringList> {
        match &mut self.value {
            AccountSettingValue::StringList(list) => Some(list),
            _ => None,
        }
    }
}

/// A typed, ordered account-settings collection (← `PurpleAccountSettings`). The typed accessors
/// return a fallback unless a setting exists **with the matching type**, exactly matching
/// libpurple's `PURPLE_IS_ACCOUNT_SETTING_*` guards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccountSettings {
    settings: Vec<AccountSetting>,
}

impl AccountSettings {
    /// An empty collection (`purple_account_settings_new`).
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of settings.
    pub fn len(&self) -> usize {
        self.settings.len()
    }

    /// Whether there are no settings.
    pub fn is_empty(&self) -> bool {
        self.settings.is_empty()
    }

    /// Find a setting by id.
    pub fn find(&self, id: &str) -> Option<&AccountSetting> {
        // STUB
        let _ = id;
        None
    }

    /// Add a setting (`purple_account_settings_add_setting`). Returns `false` (and does nothing) if
    /// a setting with the same id already exists — libpurple treats a double-add as a programming
    /// error; here it is a rejected no-op.
    pub fn add_setting(&mut self, setting: AccountSetting) -> bool {
        // STUB
        let _ = setting;
        false
    }

    /// Remove a setting by id (`purple_account_settings_remove_setting`). Returns whether one was
    /// removed.
    pub fn remove_setting(&mut self, id: &str) -> bool {
        // STUB
        let _ = id;
        false
    }

    /// Remove every setting (`purple_account_settings_remove_all_settings`).
    pub fn remove_all_settings(&mut self) {
        // STUB
    }

    /// Get a boolean, or `fallback` if absent/wrong-type (`purple_account_settings_get_boolean`).
    pub fn get_boolean(&self, id: &str, fallback: bool) -> bool {
        // STUB
        let _ = id;
        fallback
    }

    /// Get an integer, or `fallback` if absent/wrong-type (`purple_account_settings_get_int`).
    pub fn get_int(&self, id: &str, fallback: i64) -> i64 {
        // STUB
        let _ = id;
        fallback
    }

    /// Get a string, or `fallback` if absent/wrong-type/null (`purple_account_settings_get_string`).
    pub fn get_string(&self, id: &str, fallback: &str) -> String {
        // STUB
        let _ = id;
        fallback.to_string()
    }

    /// Get the active id of a string-list, or `fallback` if absent/wrong-type/unset
    /// (`purple_account_settings_get_string_list`).
    pub fn get_string_list(&self, id: &str, fallback: Option<&str>) -> Option<String> {
        // STUB
        let _ = id;
        fallback.map(str::to_string)
    }

    /// Set a boolean, if a boolean setting with `id` exists
    /// (`purple_account_settings_set_boolean`).
    pub fn set_boolean(&mut self, id: &str, value: bool) {
        // STUB
        let _ = (id, value);
    }

    /// Set an integer, if an integer setting with `id` exists (`purple_account_settings_set_int`).
    pub fn set_int(&mut self, id: &str, value: i64) {
        // STUB
        let _ = (id, value);
    }

    /// Set a string, if a string setting with `id` exists (`purple_account_settings_set_string`).
    pub fn set_string(&mut self, id: &str, value: &str) {
        // STUB
        let _ = (id, value);
    }

    /// Set the active id of a string-list, if one with `id` exists
    /// (`purple_account_settings_set_string_list`).
    pub fn set_string_list(&mut self, id: &str, item: Option<&str>) {
        // STUB
        let _ = (id, item);
    }

    /// Apply `updates` (`purple_account_settings_update_settings`): copy same-type values onto
    /// existing settings, add settings that don't yet exist, and skip type-mismatched updates.
    pub fn update_settings(&mut self, updates: &AccountSettings) {
        // STUB
        let _ = updates;
    }

    /// Project onto the wire [`AccountSettingsValues`] (string-keyed): booleans render `"true"`/
    /// `"false"`, ints their decimal, strings their value (null strings are omitted), string-lists
    /// their active id (unset lists are omitted).
    pub fn to_values(&self) -> AccountSettingsValues {
        // STUB
        AccountSettingsValues::default()
    }

    /// Derive an [`AccountSettingsSchema`] (one [`AuthParamField`] per setting, not required).
    pub fn to_schema(&self) -> AccountSettingsSchema {
        // STUB
        AccountSettingsSchema::default()
    }
}

// ---------------------------------------------------------------------------
// Presence predicates + ordering  (← purplepresence.c)
// ---------------------------------------------------------------------------

impl PresencePrimitive {
    /// Whether this primitive counts as "online" (`purple_presence_is_online`'s switch):
    /// `Available/Idle/Invisible/Away/DoNotDisturb/Streaming` are online; `Offline` and
    /// `OutOfOffice` (the `default` arm) are not.
    pub fn is_online(self) -> bool {
        // STUB
        false
    }
}

impl Presence {
    /// Whether the peer is available (`purple_presence_is_available`).
    pub fn is_available(&self) -> bool {
        // STUB
        false
    }

    /// Whether the peer is online (`purple_presence_is_online`).
    pub fn is_online(&self) -> bool {
        // STUB
        false
    }

    /// Whether the peer is idle (`purple_presence_is_idle`): online and with an idle timestamp.
    pub fn is_idle(&self) -> bool {
        // STUB
        false
    }

    /// Sort order vs another presence (`purple_presence_compare`, non-null arms): a non-offline
    /// presence sorts before an offline one; otherwise compare idle timestamps (`None` before
    /// `Some`).
    pub fn compare(&self, other: &Presence) -> Ordering {
        // STUB
        let _ = other;
        Ordering::Equal
    }
}

/// Null-aware presence ordering (`purple_presence_compare`, full): `None` sorts after `Some`
/// (an absent presence is "less online").
pub fn presence_compare(a: Option<&Presence>, b: Option<&Presence>) -> Ordering {
    // STUB
    let _ = (a, b);
    Ordering::Equal
}

// ---------------------------------------------------------------------------
// DisconnectReason fatal policy  (node reconnect authority)
// ---------------------------------------------------------------------------

impl DisconnectReason {
    /// Whether a disconnect for this reason is fatal (stop retrying; offer re-auth). The node — not
    /// a thin client — owns this. Credential/settings/certificate failures only fail again on a
    /// blind retry, so they are terminal; transport-level and server-initiated drops are transient.
    /// Mirrors the node's existing `reason_is_fatal` policy.
    pub fn is_fatal(self) -> bool {
        // STUB
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactInfo, ConversationMember};

    // -- helpers ------------------------------------------------------------

    fn contact(id: &str, display_name: Option<&str>) -> ContactInfo {
        ContactInfo {
            id: id.to_string(),
            display_name: display_name.map(str::to_string),
            ..Default::default()
        }
    }

    fn member(id: &str, display_name: Option<&str>) -> ConversationMember {
        ConversationMember {
            contact: contact(id, display_name),
            alias: None,
            nickname: None,
            typing: Default::default(),
            role: Default::default(),
            session: None,
        }
    }

    fn conversation(kind: ConversationType, members: Vec<ConversationMember>) -> ConversationInfo {
        ConversationInfo {
            transport: "t".into(),
            id: "id1".to_string(),
            kind,
            title: None,
            topic: None,
            description: None,
            members,
        }
    }

    fn presence(primitive: PresencePrimitive, idle_since: Option<u64>) -> Presence {
        Presence {
            primitive,
            idle_since,
            ..Default::default()
        }
    }

    // -- CreateConversationDetails::is_valid --------------------------------

    fn ccd(max_participants: u32, n_participants: usize) -> CreateConversationDetails {
        CreateConversationDetails {
            max_participants,
            participants: (0..n_participants)
                .map(|i| contact(&format!("c{i}"), None))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn ccd_new_sets_max_participants() {
        // libpurple `/create-conversation-details/new`: constructed with a max.
        let details = ccd(9, 0);
        assert_eq!(details.max_participants, 9);
    }

    #[test]
    fn ccd_properties_roundtrip() {
        // libpurple `/create-conversation-details/properties`.
        let details = ccd(9, 0);
        assert_eq!(details.max_participants, 9);
        assert!(details.participants.is_empty());
    }

    #[test]
    fn ccd_is_valid_null_no_participants() {
        // `/is-valid/null`: no participants set -> NO_PARTICIPANTS.
        let details = ccd(1, 0);
        assert_eq!(
            details.is_valid(),
            Err(CreateConversationDetailsError::NoParticipants)
        );
    }

    #[test]
    fn ccd_is_valid_empty_no_participants() {
        // `/is-valid/empty`: empty participant list -> NO_PARTICIPANTS.
        let details = ccd(1, 0);
        assert_eq!(
            details.is_valid(),
            Err(CreateConversationDetailsError::NoParticipants)
        );
    }

    #[test]
    fn ccd_is_valid_too_many() {
        // `/is-valid/too-many`: max 1, 2 participants -> TOO_MANY_PARTICIPANTS.
        let details = ccd(1, 2);
        assert_eq!(
            details.is_valid(),
            Err(CreateConversationDetailsError::TooManyParticipants)
        );
    }

    #[test]
    fn ccd_is_valid_limited_ok() {
        // `/is-valid/limited`: max 1, 1 participant -> valid.
        let details = ccd(1, 1);
        assert_eq!(details.is_valid(), Ok(()));
    }

    #[test]
    fn ccd_is_valid_unlimited_ok() {
        // `/is-valid/unlimited`: max 0 (unlimited), 3 participants -> valid.
        let details = ccd(0, 3);
        assert_eq!(details.is_valid(), Ok(()));
    }

    // -- ChannelJoinDetails::merge -----------------------------------------

    #[test]
    fn cjd_new_defaults() {
        // libpurple `/channel-join-details/new`.
        let details = ChannelJoinDetails::default();
        assert_eq!(details.name, None);
        assert!(!details.nickname_supported);
        assert!(!details.password_supported);
    }

    #[test]
    fn cjd_properties_roundtrip() {
        // libpurple `/channel-join-details/properties`.
        let details = ChannelJoinDetails {
            name: Some("name".into()),
            name_max_length: 42,
            nickname: Some("nickname".into()),
            nickname_max_length: 1337,
            nickname_supported: true,
            password: Some("hunter2".into()),
            password_max_length: 8,
            password_supported: true,
            ..Default::default()
        };
        assert_eq!(details.name.as_deref(), Some("name"));
        assert_eq!(details.name_max_length, 42);
        assert_eq!(details.nickname.as_deref(), Some("nickname"));
        assert_eq!(details.nickname_max_length, 1337);
        assert!(details.nickname_supported);
        assert_eq!(details.password.as_deref(), Some("hunter2"));
        assert_eq!(details.password_max_length, 8);
        assert!(details.password_supported);
    }

    #[test]
    fn cjd_merge_copies_source_fields() {
        // libpurple `/channel-join-details/merge`: copy name/nickname(+supported)/password(+
        // supported) from source into destination; max-length fields untouched.
        let source = ChannelJoinDetails {
            name: Some("name".into()),
            name_max_length: 16,
            nickname: Some("nickname".into()),
            nickname_supported: true,
            nickname_max_length: 16,
            password: Some("password".into()),
            password_supported: true,
            ..Default::default()
        };
        let mut destination = ChannelJoinDetails::default();
        destination.merge(&source);

        assert_eq!(destination.name, source.name);
        assert_eq!(destination.nickname_supported, source.nickname_supported);
        assert_eq!(destination.nickname, source.nickname);
        assert_eq!(destination.password_supported, source.password_supported);
        assert_eq!(destination.password, source.password);
        // Max-length fields are NOT merged.
        assert_eq!(destination.name_max_length, 0);
        assert_eq!(destination.nickname_max_length, 0);
    }

    // -- ConversationType predicates ---------------------------------------

    #[test]
    fn conv_type_is_dm() {
        let c = conversation(ConversationType::Dm, vec![]);
        assert!(c.is_dm());
        assert!(!c.is_group_dm());
        assert!(!c.is_channel());
        assert!(!c.is_thread());
    }

    #[test]
    fn conv_type_is_group_dm() {
        let c = conversation(ConversationType::GroupDm, vec![]);
        assert!(!c.is_dm());
        assert!(c.is_group_dm());
        assert!(!c.is_channel());
        assert!(!c.is_thread());
    }

    #[test]
    fn conv_type_is_channel() {
        let c = conversation(ConversationType::Channel, vec![]);
        assert!(!c.is_dm());
        assert!(!c.is_group_dm());
        assert!(c.is_channel());
        assert!(!c.is_thread());
    }

    #[test]
    fn conv_type_is_thread() {
        let c = conversation(ConversationType::Thread, vec![]);
        assert!(!c.is_dm());
        assert!(!c.is_group_dm());
        assert!(!c.is_channel());
        assert!(c.is_thread());
    }

    // -- title_for_display -------------------------------------------------

    #[test]
    fn conv_title_for_display_precedence() {
        // libpurple `/conversation/title-for-display`: alias -> title -> id.
        assert_eq!(
            title_for_display(Some("alias1"), Some("title1"), "id1"),
            "alias1"
        );
        assert_eq!(title_for_display(None, Some("title1"), "id1"), "title1");
        assert_eq!(title_for_display(None, None, "id1"), "id1");
        // Empty strings are skipped like birb_str_is_empty.
        assert_eq!(title_for_display(Some(""), Some("title1"), "id1"), "title1");
        assert_eq!(title_for_display(Some(""), Some(""), "id1"), "id1");

        let mut c = conversation(ConversationType::Unset, vec![]);
        c.title = Some("title1".into());
        assert_eq!(c.title_for_display(Some("alias1")), "alias1");
        assert_eq!(c.title_for_display(None), "title1");
        c.title = None;
        assert_eq!(c.title_for_display(None), "id1");
    }

    // -- generate_title ----------------------------------------------------

    #[test]
    fn conv_generate_title_empty_none() {
        // `/generate-title/empty`: DM with only the account member -> no title generated.
        let c = conversation(ConversationType::Dm, vec![member("me", None)]);
        assert_eq!(c.generate_title("me"), None);
        // Non-DM/group-DM types never generate.
        let ch = conversation(
            ConversationType::Channel,
            vec![member("them", Some("Them"))],
        );
        assert_eq!(ch.generate_title("me"), None);
    }

    #[test]
    fn conv_generate_title_dm() {
        // `/generate-title/dm`: one other member -> that member's display name; tracks id changes.
        let mut c = conversation(
            ConversationType::Dm,
            vec![member("me", None), member("Alice", None)],
        );
        assert_eq!(c.generate_title("me").as_deref(), Some("Alice"));
        // display name resolves to id when no display_name; changing id changes the title.
        c.members[1].contact.id = "Alice!".into();
        assert_eq!(c.generate_title("me").as_deref(), Some("Alice!"));
    }

    #[test]
    fn conv_generate_title_group_dm() {
        // `/generate-title/group-dm`: join other members with ", ".
        let mut c = conversation(
            ConversationType::GroupDm,
            vec![
                member("me", None),
                member("Alice", None),
                member("Bob", None),
                member("Eve", None),
            ],
        );
        assert_eq!(c.generate_title("me").as_deref(), Some("Alice, Bob, Eve"));
        c.members[2].contact.id = "Robert".into();
        c.members[3].contact.id = "Evelyn".into();
        assert_eq!(
            c.generate_title("me").as_deref(),
            Some("Alice, Robert, Evelyn")
        );
    }

    // -- ConversationType tag derivation -----------------------------------

    #[test]
    fn conv_type_tag_unset_none() {
        assert_eq!(ConversationType::Unset.tag_value(), None);
    }

    #[test]
    fn conv_type_tag_dm() {
        assert_eq!(ConversationType::Dm.tag_value(), Some("dm"));
    }

    #[test]
    fn conv_type_tag_group_dm() {
        assert_eq!(ConversationType::GroupDm.tag_value(), Some("group-dm"));
    }

    #[test]
    fn conv_type_tag_channel() {
        assert_eq!(ConversationType::Channel.tag_value(), Some("channel"));
    }

    #[test]
    fn conv_type_tag_thread() {
        assert_eq!(ConversationType::Thread.tag_value(), Some("thread"));
    }

    // -- AccountSetting (singular) -----------------------------------------

    #[test]
    fn setting_boolean_get_set_value() {
        // libpurple `/account-setting/boolean`.
        let mut setting = AccountSetting::boolean("id", "Label", true);
        assert_eq!(setting.as_bool(), Some(true));
        setting.value = AccountSettingValue::Boolean(false);
        assert_eq!(setting.as_bool(), Some(false));
    }

    #[test]
    fn setting_int_get_set_value() {
        // libpurple `/account-setting/int`.
        let mut setting = AccountSetting::int("id", "Label", 1337);
        assert_eq!(setting.as_int(), Some(1337));
        setting.value = AccountSettingValue::Int(42);
        assert_eq!(setting.as_int(), Some(42));
    }

    #[test]
    fn setting_string_get_set_value() {
        // libpurple `/account-setting/string`.
        let mut setting = AccountSetting::string("id", "Label", Some("default"));
        assert_eq!(setting.as_str(), Some("default"));
        setting.value = AccountSettingValue::Str(Some("new-value".into()));
        assert_eq!(setting.as_str(), Some("new-value"));
    }

    #[test]
    fn setting_string_list_new_empty() {
        // libpurple `/account-setting/string-list/new`.
        let setting = AccountSetting::string_list("list1", "List 1");
        let list = setting.as_string_list().expect("string list");
        assert!(list.items.is_empty());
        assert_eq!(list.active_item(), None);
    }

    #[test]
    fn setting_string_list_add_dedup() {
        // libpurple `/account-setting/string-list/add`: add two, reject duplicate id.
        let mut setting = AccountSetting::string_list("list1", "List 1");
        let list = setting.as_string_list_mut().expect("string list");
        assert!(list.add_item("item1", "Item 1"));
        assert!(list.add_item("item2", "Item 2"));
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.items[0].id, "item1");
        assert_eq!(list.items[0].label, "Item 1");
        assert_eq!(list.items[1].id, "item2");
        // Duplicate id is rejected.
        assert!(!list.add_item("item1", "Item 3"));
        assert_eq!(list.items.len(), 2);
    }

    #[test]
    fn setting_string_list_set_active() {
        // libpurple `/account-setting/string-list/set-active`.
        let mut setting = AccountSetting::string_list("list1", "List 1");
        let list = setting.as_string_list_mut().expect("string list");
        assert_eq!(list.active_item(), None);
        list.add_item("item1", "Item 1");
        // Still nothing active after adding.
        assert_eq!(list.active_item(), None);
        list.set_active_item(Some("item1"));
        assert_eq!(list.active_item(), Some("item1"));
        list.set_active_item(None);
        assert_eq!(list.active_item(), None);
    }

    // -- AccountSettings (collection) --------------------------------------

    #[test]
    fn settings_new_empty() {
        let settings = AccountSettings::new();
        assert_eq!(settings.len(), 0);
        assert!(settings.is_empty());
    }

    #[test]
    fn settings_add_remove() {
        // libpurple `/account-settings/add-remove`.
        let mut settings = AccountSettings::new();
        assert!(settings.add_setting(AccountSetting::boolean("test-foo", "Foo", true)));
        assert_eq!(settings.len(), 1);
        assert!(settings.remove_setting("test-foo"));
        assert_eq!(settings.len(), 0);
        // Removing again fails.
        assert!(!settings.remove_setting("test-foo"));
        assert_eq!(settings.len(), 0);
    }

    #[test]
    fn settings_double_add_rejected() {
        // libpurple `/account-settings/double-add`: same id twice -> rejected no-op.
        let mut settings = AccountSettings::new();
        assert!(settings.add_setting(AccountSetting::boolean("test-foo", "Foo", true)));
        assert!(!settings.add_setting(AccountSetting::boolean("test-foo", "Foo", true)));
        assert_eq!(settings.len(), 1);
    }

    #[test]
    fn settings_add_again_wrong_type_rejected() {
        // libpurple `/account-settings/add-again`: same id, different type -> rejected.
        let mut settings = AccountSettings::new();
        assert!(settings.add_setting(AccountSetting::boolean("test-foo", "Foo", true)));
        assert!(!settings.add_setting(AccountSetting::string(
            "test-foo",
            "Foobar",
            Some("a string!")
        )));
        assert_eq!(settings.len(), 1);
    }

    #[test]
    fn settings_get_set_boolean() {
        // libpurple `/account-settings/get-set-boolean`.
        let mut settings = AccountSettings::new();
        assert!(!settings.get_boolean("test", false));
        settings.add_setting(AccountSetting::boolean("test", "Test", true));
        assert!(settings.get_boolean("test", false));
        settings.set_boolean("test", false);
        assert!(!settings.get_boolean("test", true));
        // Wrong type -> fallback.
        settings.add_setting(AccountSetting::int("wrong", "Wrong", 42));
        assert!(!settings.get_boolean("wrong", false));
    }

    #[test]
    fn settings_get_set_int() {
        // libpurple `/account-settings/get-set-int`.
        let mut settings = AccountSettings::new();
        assert_eq!(settings.get_int("test", 42), 42);
        settings.add_setting(AccountSetting::int("test", "Test", 1337));
        assert_eq!(settings.get_int("test", 42), 1337);
        settings.set_int("test", -1);
        assert_eq!(settings.get_int("test", 42), -1);
        // Wrong type -> fallback.
        settings.add_setting(AccountSetting::string("wrong", "Wrong", Some("bad-wrong")));
        assert_eq!(settings.get_int("wrong", 42), 42);
    }

    #[test]
    fn settings_get_set_string() {
        // libpurple `/account-settings/get-set-string`.
        let mut settings = AccountSettings::new();
        assert_eq!(settings.get_string("test", "fallback"), "fallback");
        settings.add_setting(AccountSetting::string("test", "Test", Some("the value")));
        assert_eq!(settings.get_string("test", "fallback"), "the value");
        settings.set_string("test", "the other value");
        assert_eq!(settings.get_string("test", "the value"), "the other value");
        // Wrong type -> fallback (a boolean's string form must NOT leak through).
        settings.add_setting(AccountSetting::boolean("wrong", "Wrong", true));
        assert_eq!(settings.get_string("wrong", "fallback"), "fallback");
    }

    #[test]
    fn settings_get_set_string_list() {
        // libpurple `/account-settings/get-set-string-list`.
        let mut settings = AccountSettings::new();
        assert_eq!(
            settings
                .get_string_list("test", Some("fallback"))
                .as_deref(),
            Some("fallback")
        );
        let mut setting = AccountSetting::string_list("test", "Test");
        let list = setting.as_string_list_mut().expect("string list");
        list.add_item("foo", "Foo");
        list.add_item("bar", "Bar");
        list.add_item("baz", "baz");
        settings.add_setting(setting);
        // Nothing active -> fallback.
        assert_eq!(
            settings
                .get_string_list("test", Some("fallback"))
                .as_deref(),
            Some("fallback")
        );
        settings.set_string_list("test", Some("foo"));
        assert_eq!(
            settings.get_string_list("test", None).as_deref(),
            Some("foo")
        );
        settings.set_string_list("test", Some("bar"));
        assert_eq!(
            settings.get_string_list("test", None).as_deref(),
            Some("bar")
        );
        settings.set_string_list("test", None);
        assert_eq!(
            settings
                .get_string_list("test", Some("fallback"))
                .as_deref(),
            Some("fallback")
        );
        // Wrong type -> fallback.
        settings.add_setting(AccountSetting::boolean("wrong", "Wrong", true));
        assert_eq!(
            settings
                .get_string_list("wrong", Some("fallback"))
                .as_deref(),
            Some("fallback")
        );
    }

    #[test]
    fn settings_remove_all() {
        // libpurple `/account-settings/remove-all`.
        let mut settings = AccountSettings::new();
        settings.remove_all_settings();
        assert_eq!(settings.len(), 0);
        settings.add_setting(AccountSetting::string("test1", "Test 1", None));
        settings.add_setting(AccountSetting::string("test2", "Test 2", None));
        settings.add_setting(AccountSetting::string("test3", "Test 3", None));
        assert_eq!(settings.len(), 3);
        settings.remove_all_settings();
        assert_eq!(settings.len(), 0);
    }

    #[test]
    fn settings_update() {
        // libpurple `/account-settings/update`.
        let mut settings = AccountSettings::new();
        settings.add_setting(AccountSetting::boolean("foo", "Foo", true));
        settings.add_setting(AccountSetting::int("bar", "Bar", 42));
        settings.add_setting(AccountSetting::string(
            "baz",
            "Baz",
            Some("You can't touch this"),
        ));
        let mut quux = AccountSetting::string_list("quux", "Quux");
        quux.as_string_list_mut().unwrap().add_item("abc", "123");
        settings.add_setting(quux);

        let mut updates = AccountSettings::new();
        updates.add_setting(AccountSetting::int("bar", "Bar", 1337));
        updates.add_setting(AccountSetting::boolean("baz", "Baz", false)); // wrong type
        updates.add_setting(AccountSetting::string("qux", "Qux", Some("woo"))); // new
        let mut quux_u = AccountSetting::string_list("quux", "Quux");
        {
            let l = quux_u.as_string_list_mut().unwrap();
            l.add_item("abc", "123");
            l.set_active_item(Some("abc"));
        }
        updates.add_setting(quux_u);

        settings.update_settings(&updates);

        assert_eq!(settings.len(), 5);
        assert!(settings.get_boolean("foo", false));
        assert_eq!(settings.get_int("bar", 0), 1337);
        // Type mismatch left the string untouched.
        assert_eq!(settings.get_string("baz", ""), "You can't touch this");
        assert_eq!(settings.get_string("qux", ""), "woo");
        assert_eq!(
            settings.get_string_list("quux", None).as_deref(),
            Some("abc")
        );
    }

    #[test]
    fn settings_to_values_projection() {
        let mut settings = AccountSettings::new();
        settings.add_setting(AccountSetting::boolean("b", "B", true));
        settings.add_setting(AccountSetting::int("i", "I", -5));
        settings.add_setting(AccountSetting::string("s", "S", Some("hi")));
        settings.add_setting(AccountSetting::string("null", "Null", None));
        let mut sl = AccountSetting::string_list("sl", "SL");
        {
            let l = sl.as_string_list_mut().unwrap();
            l.add_item("x", "X");
            l.set_active_item(Some("x"));
        }
        settings.add_setting(sl);

        let values = settings.to_values();
        assert_eq!(values.values.get("b").map(String::as_str), Some("true"));
        assert_eq!(values.values.get("i").map(String::as_str), Some("-5"));
        assert_eq!(values.values.get("s").map(String::as_str), Some("hi"));
        assert_eq!(values.values.get("sl").map(String::as_str), Some("x"));
        // Null string and unset lists are omitted.
        assert!(!values.values.contains_key("null"));
    }

    // -- Presence predicates + ordering ------------------------------------

    #[test]
    fn presence_properties_roundtrip() {
        // libpurple `/presence/properties` (the DTO-modeled subset).
        let p = Presence {
            primitive: PresencePrimitive::Available,
            message: Some("I'll be back!".into()),
            emoji: Some("🤖".into()),
            mobile: true,
            idle_since: Some(100),
        };
        assert_eq!(p.primitive, PresencePrimitive::Available);
        assert_eq!(p.message.as_deref(), Some("I'll be back!"));
        assert_eq!(p.emoji.as_deref(), Some("🤖"));
        assert!(p.mobile);
        assert_eq!(p.idle_since, Some(100));
    }

    #[test]
    fn presence_is_available() {
        assert!(presence(PresencePrimitive::Available, None).is_available());
        for p in [
            PresencePrimitive::Offline,
            PresencePrimitive::Idle,
            PresencePrimitive::Away,
            PresencePrimitive::DoNotDisturb,
            PresencePrimitive::Streaming,
            PresencePrimitive::Invisible,
            PresencePrimitive::OutOfOffice,
        ] {
            assert!(!presence(p, None).is_available());
        }
    }

    #[test]
    fn presence_is_online_all_primitives() {
        // Online set from purple_presence_is_online's switch.
        for p in [
            PresencePrimitive::Available,
            PresencePrimitive::Idle,
            PresencePrimitive::Invisible,
            PresencePrimitive::Away,
            PresencePrimitive::DoNotDisturb,
            PresencePrimitive::Streaming,
        ] {
            assert!(p.is_online(), "{p:?} should be online");
            assert!(presence(p, None).is_online());
        }
        // Offline and OutOfOffice (the default arm) are NOT online.
        assert!(!PresencePrimitive::Offline.is_online());
        assert!(!PresencePrimitive::OutOfOffice.is_online());
        assert!(!presence(PresencePrimitive::Offline, None).is_online());
        assert!(!presence(PresencePrimitive::OutOfOffice, None).is_online());
    }

    #[test]
    fn presence_is_idle() {
        // Online + idle timestamp -> idle.
        assert!(presence(PresencePrimitive::Available, Some(1)).is_idle());
        // Online but no idle timestamp -> not idle.
        assert!(!presence(PresencePrimitive::Available, None).is_idle());
        // Offline -> never idle even with a timestamp.
        assert!(!presence(PresencePrimitive::Offline, Some(1)).is_idle());
        // OutOfOffice is not online -> not idle.
        assert!(!presence(PresencePrimitive::OutOfOffice, Some(1)).is_idle());
    }

    #[test]
    fn presence_compare_ordering() {
        let online = presence(PresencePrimitive::Available, None);
        let offline = presence(PresencePrimitive::Offline, None);
        // Non-offline sorts before offline.
        assert_eq!(online.compare(&offline), Ordering::Less);
        assert_eq!(offline.compare(&online), Ordering::Greater);
        // Both offline -> equal idle (both None).
        assert_eq!(offline.compare(&offline.clone()), Ordering::Equal);
        // Both online: compare idle timestamps (None before Some; earlier before later).
        let idle_none = presence(PresencePrimitive::Available, None);
        let idle_early = presence(PresencePrimitive::Available, Some(10));
        let idle_late = presence(PresencePrimitive::Available, Some(20));
        assert_eq!(idle_none.compare(&idle_early), Ordering::Less);
        assert_eq!(idle_early.compare(&idle_late), Ordering::Less);
        assert_eq!(idle_late.compare(&idle_early), Ordering::Greater);
        assert_eq!(idle_early.compare(&idle_early.clone()), Ordering::Equal);
    }

    #[test]
    fn presence_compare_options() {
        let online = presence(PresencePrimitive::Available, None);
        // Two None -> equal.
        assert_eq!(presence_compare(None, None), Ordering::Equal);
        // Some sorts before None (present is "more online").
        assert_eq!(presence_compare(Some(&online), None), Ordering::Less);
        assert_eq!(presence_compare(None, Some(&online)), Ordering::Greater);
    }

    // -- DisconnectReason::is_fatal ----------------------------------------

    #[test]
    fn disconnect_reason_is_fatal_variants() {
        assert!(DisconnectReason::AuthenticationFailed.is_fatal());
        assert!(DisconnectReason::InvalidSettings.is_fatal());
        assert!(DisconnectReason::CertificateError.is_fatal());
        assert!(!DisconnectReason::UserRequested.is_fatal());
        assert!(!DisconnectReason::NetworkError.is_fatal());
        assert!(!DisconnectReason::ReplacedByOtherClient.is_fatal());
        assert!(!DisconnectReason::Other.is_fatal());
    }
}
