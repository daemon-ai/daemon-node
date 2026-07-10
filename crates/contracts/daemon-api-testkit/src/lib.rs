// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-api-testkit` — reference [`MessagingProtocol`] implementations plus the
//! ops-vs-behavior conformance invariant, ported from libpurple's protocol test fixtures.
//!
//! This dev-only crate is the executable spec for W1-A: the daemon analogue of libpurple's
//! `Test*Empty` / `Test*` protocol fixtures.
//!
//! - [`EmptyProtocol`] exposes **every** `Supports*` feature-trait handle but reports **no** verb
//!   supported and leaves every verb at its trait default (→ [`ApiError::Unsupported`]). It mirrors
//!   libpurple's "Empty" fixtures, whose interface is implemented with an empty `iface_init` so
//!   `purple_protocol_*_implements_*` is false and each verb warns/errors.
//! - [`FakeProtocol`] is an in-memory reference impl of every feature trait, with a per-verb
//!   failure switch ([`FailSwitches`]) mirroring libpurple's per-fixture `should_error` boolean.
//! - [`assert_ops_match_behavior`] is the cross-adapter invariant: advertised `supported()` and
//!   the actual verb behavior agree, keyed on each trait method's capability sentinel string.

use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AccountSettingsValues, ActionMenu, AdapterCapabilities, AdapterInfo,
    ApiError, ChannelJoinDetails, ContactInfo, ContactsOps, ConvSendArgs, ConversationInfo,
    ConversationOps, ConversationType, CreateConversationDetails, FileTransfer, FileTransferOps,
    MemberBanArgs, MemberInviteArgs, MemberRemoveArgs, MemberSetRoleArgs, MembershipOps,
    MessagingProtocol, NodeApi, RosterOps, SupportsContacts, SupportsConversations,
    SupportsDirectory, SupportsFileTransfer, SupportsMembership, SupportsRoster, TransportAdapter,
};
use daemon_protocol::TransportId;

/// The capability-sentinel string each trait default method carries. Kept in one place so both the
/// reference impls and [`assert_ops_match_behavior`] agree on the exact strings the daemon-api
/// trait defaults return (e.g. `SupportsConversations::send` → `Unsupported("conv_send")`).
pub mod sentinels {
    /// `SupportsConversations::create`.
    pub const CONV_CREATE: &str = "conv_create";
    /// `SupportsConversations::join_channel`.
    pub const CONV_JOIN: &str = "conv_join";
    /// `SupportsConversations::leave`.
    pub const CONV_LEAVE: &str = "conv_leave";
    /// `SupportsConversations::delete`.
    pub const CONV_DELETE: &str = "conv_delete";
    /// `SupportsConversations::send`.
    pub const CONV_SEND: &str = "conv_send";
    /// `SupportsConversations::set_topic`.
    pub const CONV_SET_TOPIC: &str = "conv_set_topic";
    /// `SupportsConversations::set_title`.
    pub const CONV_SET_TITLE: &str = "conv_set_title";
    /// `SupportsConversations::set_description`.
    pub const CONV_SET_DESCRIPTION: &str = "conv_set_description";
    /// `SupportsMembership::invite`.
    pub const MEMBER_INVITE: &str = "member_invite";
    /// `SupportsMembership::remove`.
    pub const MEMBER_REMOVE: &str = "member_remove";
    /// `SupportsMembership::ban`.
    pub const MEMBER_BAN: &str = "member_ban";
    /// `SupportsMembership::set_role`.
    pub const MEMBER_SET_ROLE: &str = "member_set_role";
    /// `SupportsContacts::get_profile`.
    pub const CONTACT_GET_PROFILE: &str = "contact_get_profile";
    /// `SupportsContacts::set_alias`.
    pub const CONTACT_SET_ALIAS: &str = "contact_set_alias";
    /// `SupportsRoster::add`.
    pub const ROSTER_ADD: &str = "roster_add";
    /// `SupportsRoster::update`.
    pub const ROSTER_UPDATE: &str = "roster_update";
    /// `SupportsRoster::remove`.
    pub const ROSTER_REMOVE: &str = "roster_remove";
    /// `SupportsDirectory::search_contacts`.
    pub const DIRECTORY_SEARCH: &str = "directory_search";
    /// `SupportsFileTransfer::send`.
    pub const FILE_TRANSFER_SEND: &str = "file_transfer_send";
    /// `SupportsFileTransfer::receive`.
    pub const FILE_TRANSFER_RECEIVE: &str = "file_transfer_receive";
}

// ---------------------------------------------------------------------------
// EmptyProtocol — every feature trait present, no verb supported.
// ---------------------------------------------------------------------------

/// A minimal [`MessagingProtocol`] that advertises **no** optional feature verb: every feature
/// trait handle is present ([`Some`]), every `supported()` is all-false, and every verb is left at
/// its trait default (→ [`ApiError::Unsupported`] with the capability sentinel). The daemon analogue
/// of libpurple's `Test*Empty` fixtures.
#[derive(Debug, Default)]
pub struct EmptyProtocol;

impl EmptyProtocol {
    /// Construct an [`EmptyProtocol`] behind an [`Arc`] (the shape adapters are held as).
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl TransportAdapter for EmptyProtocol {
    fn family(&self) -> &str {
        "empty"
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: "empty".to_string(),
            display_name: "Empty".to_string(),
            capabilities: AdapterCapabilities::default(),
            account_schema: AccountSettingsSchema::default(),
            ..Default::default()
        }
    }

    async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {}

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for EmptyProtocol {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }
    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }
    fn roster(self: Arc<Self>) -> Option<Arc<dyn SupportsRoster>> {
        Some(self)
    }
    fn contacts(self: Arc<Self>) -> Option<Arc<dyn SupportsContacts>> {
        Some(self)
    }
    fn directory(self: Arc<Self>) -> Option<Arc<dyn SupportsDirectory>> {
        Some(self)
    }
    fn file_transfer(self: Arc<Self>) -> Option<Arc<dyn SupportsFileTransfer>> {
        Some(self)
    }
}

// All feature traits: `supported()` is all-false; every verb keeps its trait default.
#[async_trait]
impl SupportsConversations for EmptyProtocol {
    fn supported(&self) -> ConversationOps {
        ConversationOps::default()
    }
}
#[async_trait]
impl SupportsMembership for EmptyProtocol {
    fn supported(&self) -> MembershipOps {
        MembershipOps::default()
    }
}
#[async_trait]
impl SupportsRoster for EmptyProtocol {
    fn supported(&self) -> RosterOps {
        RosterOps::default()
    }
}
#[async_trait]
impl SupportsContacts for EmptyProtocol {
    fn supported(&self) -> ContactsOps {
        ContactsOps::default()
    }
}
#[async_trait]
impl SupportsDirectory for EmptyProtocol {
    fn supported(&self) -> bool {
        false
    }
}
#[async_trait]
impl SupportsFileTransfer for EmptyProtocol {
    fn supported(&self) -> FileTransferOps {
        FileTransferOps::default()
    }
}

// ---------------------------------------------------------------------------
// FailSwitches — per-verb failure toggles (libpurple `should_error` analogue).
// ---------------------------------------------------------------------------

/// Per-verb failure switches for [`FakeProtocol`]. A verb whose sentinel key is present errors with
/// [`ApiError::Other`] (a *non-`Unsupported`* error, so it never collides with the capability
/// sentinel). Mirrors libpurple's per-fixture `should_error` boolean, generalized to per-verb.
#[derive(Clone, Debug, Default)]
pub struct FailSwitches {
    keys: std::collections::BTreeSet<&'static str>,
}

impl FailSwitches {
    /// No verb fails (the libpurple `should_error = FALSE` fixture).
    pub fn none() -> Self {
        Self::default()
    }

    /// Every verb fails (the libpurple `should_error = TRUE` fixture, applied globally).
    pub fn all() -> Self {
        use sentinels::*;
        Self {
            keys: [
                CONV_CREATE,
                CONV_JOIN,
                CONV_LEAVE,
                CONV_DELETE,
                CONV_SEND,
                CONV_SET_TOPIC,
                CONV_SET_TITLE,
                CONV_SET_DESCRIPTION,
                MEMBER_INVITE,
                MEMBER_REMOVE,
                MEMBER_BAN,
                MEMBER_SET_ROLE,
                CONTACT_GET_PROFILE,
                CONTACT_SET_ALIAS,
                ROSTER_ADD,
                ROSTER_UPDATE,
                ROSTER_REMOVE,
                DIRECTORY_SEARCH,
                FILE_TRANSFER_SEND,
                FILE_TRANSFER_RECEIVE,
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Fail only the named verbs (their capability sentinel keys, e.g. [`sentinels::CONV_SEND`]).
    pub fn only(keys: &[&'static str]) -> Self {
        Self {
            keys: keys.iter().copied().collect(),
        }
    }

    /// Whether the verb keyed by `key` is switched to fail.
    pub fn fails(&self, key: &str) -> bool {
        self.keys.contains(key)
    }
}

// ---------------------------------------------------------------------------
// FakeProtocol — in-memory reference impl of every feature trait.
// ---------------------------------------------------------------------------

/// An in-memory reference [`MessagingProtocol`] implementing **every** optional feature trait, with
/// per-verb [`FailSwitches`]. The daemon analogue of libpurple's non-empty `Test*` fixtures: it
/// advertises all verbs supported and each verb succeeds unless its switch is flipped. Wave-2
/// packages build feature tests on top of this.
#[derive(Debug, Default)]
pub struct FakeProtocol {
    fail: FailSwitches,
    /// Whether `validate_account` should reject (separate from the verb switches — it is a
    /// [`MessagingProtocol`]-level method, not a feature-trait verb).
    validate_fails: bool,
    /// The in-memory record of accepted [`SupportsFileTransfer::send`] transfers (the daemon
    /// analogue of the libpurple fixture "sending" the file); only successful sends are recorded.
    sent: std::sync::Mutex<Vec<FileTransfer>>,
    /// The in-memory record of accepted [`SupportsFileTransfer::receive`] transfers.
    received: std::sync::Mutex<Vec<FileTransfer>>,
}

impl FakeProtocol {
    /// The marker settings VALUE [`MessagingProtocol::validate_account`] rejects (N2): any
    /// settings entry whose value equals this marker fails validation with a non-`Unsupported`
    /// error, so host tests can prove the `transport_configure` op surfaces adapter validation
    /// failures. Orthogonal to [`FakeProtocol::failing`]'s global validate switch.
    pub const VALIDATE_REJECT_VALUE: &'static str = "reject-me";

    /// A fully-operable fake: every verb supported and succeeds.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A fake whose every verb fails (the global `should_error = TRUE` fixture) — including
    /// `validate_account`.
    pub fn failing() -> Arc<Self> {
        Arc::new(Self {
            fail: FailSwitches::all(),
            validate_fails: true,
            ..Default::default()
        })
    }

    /// A fake with a custom failure-switch set (verbs not listed still succeed).
    pub fn with_failures(fail: FailSwitches) -> Arc<Self> {
        Arc::new(Self {
            fail,
            ..Default::default()
        })
    }

    /// The transfers accepted by [`SupportsFileTransfer::send`] so far (in-memory reference state).
    pub fn sent_transfers(&self) -> Vec<FileTransfer> {
        self.sent.lock().unwrap().clone()
    }

    /// The transfers accepted by [`SupportsFileTransfer::receive`] so far.
    pub fn received_transfers(&self) -> Vec<FileTransfer> {
        self.received.lock().unwrap().clone()
    }

    /// Resolve a unit-returning verb keyed by its sentinel: `Ok(())` unless switched to fail.
    fn unit(&self, key: &'static str) -> Result<(), ApiError> {
        self.value(key, ())
    }

    /// Resolve a value-returning verb keyed by its sentinel: `Ok(ok)` unless switched to fail (then
    /// `Err(Other)` — never the capability sentinel).
    fn value<T>(&self, key: &'static str, ok: T) -> Result<T, ApiError> {
        if self.fail.fails(key) {
            Err(ApiError::Other(format!("fake:{key} error")))
        } else {
            Ok(ok)
        }
    }
}

#[async_trait]
impl TransportAdapter for FakeProtocol {
    fn family(&self) -> &str {
        "fake"
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: "fake".to_string(),
            display_name: "Fake".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: true,
                room_enumeration: true,
                file_transfer: true,
                interactive_auth: false,
            },
            account_schema: AccountSettingsSchema::default(),
            ..Default::default()
        }
    }

    async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {}

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for FakeProtocol {
    async fn validate_account(&self, settings: &AccountSettingsValues) -> Result<(), ApiError> {
        if self.validate_fails {
            return Err(ApiError::Other("fake:validate_account error".into()));
        }
        // The value-keyed rejection (N2): an otherwise-operable fake still rejects the marker,
        // so host tests can drive a REAL validation failure through `transport_configure`.
        if settings
            .values
            .values()
            .any(|v| v == Self::VALIDATE_REJECT_VALUE)
        {
            return Err(ApiError::Other(format!(
                "fake:validate_account rejects marker value `{}`",
                Self::VALIDATE_REJECT_VALUE
            )));
        }
        Ok(())
    }

    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }
    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }
    fn roster(self: Arc<Self>) -> Option<Arc<dyn SupportsRoster>> {
        Some(self)
    }
    fn contacts(self: Arc<Self>) -> Option<Arc<dyn SupportsContacts>> {
        Some(self)
    }
    fn directory(self: Arc<Self>) -> Option<Arc<dyn SupportsDirectory>> {
        Some(self)
    }
    fn file_transfer(self: Arc<Self>) -> Option<Arc<dyn SupportsFileTransfer>> {
        Some(self)
    }
}

/// A reference conversation the Fake's `create`/`join_channel` return.
fn fake_conversation(transport: TransportId, kind: ConversationType) -> ConversationInfo {
    ConversationInfo {
        transport,
        id: "fake-conv".to_string(),
        kind,
        title: None,
        topic: None,
        description: None,
        members: Vec::new(),
        parent: None,
    }
}

#[async_trait]
impl SupportsConversations for FakeProtocol {
    fn supported(&self) -> ConversationOps {
        ConversationOps {
            create: true,
            join_channel: true,
            leave: true,
            delete: true,
            send: true,
            set_topic: true,
            set_title: true,
            set_description: true,
        }
    }

    async fn create_details(&self, _transport: TransportId) -> CreateConversationDetails {
        // Mirrors libpurple's fixture `purple_create_conversation_details_new(10)`.
        CreateConversationDetails {
            max_participants: 10,
            ..Default::default()
        }
    }

    async fn create(
        &self,
        transport: TransportId,
        _details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        // libpurple's fixture returns a PURPLE_CONVERSATION_TYPE_UNSET conversation.
        self.value(
            sentinels::CONV_CREATE,
            fake_conversation(transport, ConversationType::Unset),
        )
    }

    async fn channel_join_details(&self, _transport: TransportId) -> ChannelJoinDetails {
        // Mirrors `purple_channel_join_details_new(16, TRUE, 16, TRUE, 0)`.
        ChannelJoinDetails {
            name_max_length: 16,
            nickname_supported: true,
            nickname_max_length: 16,
            password_supported: true,
            password_max_length: 0,
            ..Default::default()
        }
    }

    async fn join_channel(
        &self,
        transport: TransportId,
        _details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        self.value(
            sentinels::CONV_JOIN,
            fake_conversation(transport, ConversationType::Channel),
        )
    }

    async fn leave(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_LEAVE)
    }
    async fn delete(&self, _transport: TransportId, _conv: String) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_DELETE)
    }
    async fn send(&self, _args: ConvSendArgs) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_SEND)
    }
    async fn set_topic(
        &self,
        _transport: TransportId,
        _conv: String,
        _topic: Option<String>,
    ) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_SET_TOPIC)
    }
    async fn set_title(
        &self,
        _transport: TransportId,
        _conv: String,
        _title: Option<String>,
    ) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_SET_TITLE)
    }
    async fn set_description(
        &self,
        _transport: TransportId,
        _conv: String,
        _description: Option<String>,
    ) -> Result<(), ApiError> {
        self.unit(sentinels::CONV_SET_DESCRIPTION)
    }
}

#[async_trait]
impl SupportsMembership for FakeProtocol {
    fn supported(&self) -> MembershipOps {
        MembershipOps {
            invite: true,
            remove: true,
            ban: true,
            set_role: true,
        }
    }
    async fn invite(&self, _args: MemberInviteArgs) -> Result<(), ApiError> {
        self.unit(sentinels::MEMBER_INVITE)
    }
    async fn remove(&self, _args: MemberRemoveArgs) -> Result<(), ApiError> {
        self.unit(sentinels::MEMBER_REMOVE)
    }
    async fn ban(&self, _args: MemberBanArgs) -> Result<(), ApiError> {
        self.unit(sentinels::MEMBER_BAN)
    }
    async fn set_role(&self, _args: MemberSetRoleArgs) -> Result<(), ApiError> {
        self.unit(sentinels::MEMBER_SET_ROLE)
    }
}

#[async_trait]
impl SupportsRoster for FakeProtocol {
    fn supported(&self) -> RosterOps {
        RosterOps {
            list: true,
            add: true,
            update: true,
            remove: true,
        }
    }
    async fn add(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        self.unit(sentinels::ROSTER_ADD)
    }
    async fn update(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        self.unit(sentinels::ROSTER_UPDATE)
    }
    async fn remove(&self, _transport: TransportId, _contact: ContactInfo) -> Result<(), ApiError> {
        self.unit(sentinels::ROSTER_REMOVE)
    }
}

#[async_trait]
impl SupportsContacts for FakeProtocol {
    fn supported(&self) -> ContactsOps {
        ContactsOps {
            get_profile: true,
            action_menu: true,
            set_alias: true,
        }
    }
    async fn get_profile(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
    ) -> Result<String, ApiError> {
        // Mirrors the libpurple fixture returning `"profile data"`.
        self.value(sentinels::CONTACT_GET_PROFILE, "profile data".to_string())
    }
    fn action_menu(&self, _transport: TransportId, _contact: ContactInfo) -> Option<ActionMenu> {
        Some(ActionMenu::default())
    }
    async fn set_alias(
        &self,
        _transport: TransportId,
        _contact: ContactInfo,
        _alias: Option<String>,
    ) -> Result<(), ApiError> {
        self.unit(sentinels::CONTACT_SET_ALIAS)
    }
}

#[async_trait]
impl SupportsDirectory for FakeProtocol {
    fn supported(&self) -> bool {
        true
    }
    async fn search_contacts(
        &self,
        _transport: TransportId,
        _query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        // Mirrors the libpurple fixture returning an (empty) PurpleContacts container.
        self.value(sentinels::DIRECTORY_SEARCH, Vec::new())
    }
}

#[async_trait]
impl SupportsFileTransfer for FakeProtocol {
    fn supported(&self) -> FileTransferOps {
        FileTransferOps {
            send: true,
            receive: true,
        }
    }
    async fn send(&self, _transport: TransportId, transfer: FileTransfer) -> Result<(), ApiError> {
        self.unit(sentinels::FILE_TRANSFER_SEND)?;
        self.sent.lock().unwrap().push(transfer);
        Ok(())
    }
    async fn receive(
        &self,
        _transport: TransportId,
        transfer: FileTransfer,
    ) -> Result<(), ApiError> {
        self.unit(sentinels::FILE_TRANSFER_RECEIVE)?;
        self.received.lock().unwrap().push(transfer);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The invariant.
// ---------------------------------------------------------------------------

/// A stable [`TransportId`] the invariant probes the verbs with.
fn probe_transport() -> TransportId {
    TransportId::new("testkit")
}

/// Assert `res` is exactly `Err(ApiError::Unsupported(sentinel))` — the trait-default capability
/// error for an unsupported verb.
fn assert_is_sentinel<T: std::fmt::Debug>(res: &Result<T, ApiError>, sentinel: &str, verb: &str) {
    match res {
        Err(ApiError::Unsupported(s)) if s == sentinel => {}
        other => panic!(
            "verb `{verb}` is advertised unsupported, so it must return \
             Err(Unsupported({sentinel:?})); got {other:?}"
        ),
    }
}

/// Assert `res` is **not** the capability sentinel for `sentinel` — i.e. an advertised verb did not
/// fall through to its unsupported trait default. It may be `Ok`, or a *different* error (e.g. a
/// transient `Unsupported("… not connected")` from a real, unconnected adapter).
fn assert_not_sentinel<T: std::fmt::Debug>(res: &Result<T, ApiError>, sentinel: &str, verb: &str) {
    if let Err(ApiError::Unsupported(s)) = res {
        assert!(
            s != sentinel,
            "verb `{verb}` is advertised supported, but it returned the unsupported \
             capability sentinel Unsupported({sentinel:?}) — it must be overridden"
        );
    }
}

/// The cross-adapter conformance invariant: for every optional verb, advertised `supported()` and
/// actual behavior agree, keyed on the verb's capability sentinel string.
///
/// - Forward (universal, no I/O): `supported()==false` ⟹ the verb returns
///   `Err(ApiError::Unsupported(<sentinel>))` (non-`Result` accessors return their empty default —
///   `action_menu` → `None`). This only exercises trait-default bodies.
/// - Reverse (sentinel-keyed, safe against unconnected real adapters): `supported()==true` ⟹ the
///   verb does not return that sentinel (it is `Ok`, or a *different* error).
///
/// Run against [`EmptyProtocol`]/[`FakeProtocol`] (full biconditional) and against every real
/// adapter (the forward half is the load-bearing safety invariant for the thin-client model).
pub async fn assert_ops_match_behavior(proto: Arc<dyn MessagingProtocol>) {
    let t = probe_transport();

    if let Some(c) = proto.clone().conversations() {
        let ops = c.supported();
        check(
            &c.create(t.clone(), CreateConversationDetails::default())
                .await,
            ops.create,
            sentinels::CONV_CREATE,
            "conversations::create",
        );
        check(
            &c.join_channel(t.clone(), ChannelJoinDetails::default())
                .await,
            ops.join_channel,
            sentinels::CONV_JOIN,
            "conversations::join_channel",
        );
        check(
            &c.leave(t.clone(), "conv".to_string()).await,
            ops.leave,
            sentinels::CONV_LEAVE,
            "conversations::leave",
        );
        check(
            &c.delete(t.clone(), "conv".to_string()).await,
            ops.delete,
            sentinels::CONV_DELETE,
            "conversations::delete",
        );
        check(
            &c.send(ConvSendArgs {
                transport: t.clone(),
                conv: "conv".to_string(),
                from: None,
                message: daemon_protocol::UserMsg::new("hi"),
                op_id: None,
            })
            .await,
            ops.send,
            sentinels::CONV_SEND,
            "conversations::send",
        );
        check(
            &c.set_topic(t.clone(), "conv".to_string(), Some("topic".into()))
                .await,
            ops.set_topic,
            sentinels::CONV_SET_TOPIC,
            "conversations::set_topic",
        );
        check(
            &c.set_title(t.clone(), "conv".to_string(), Some("title".into()))
                .await,
            ops.set_title,
            sentinels::CONV_SET_TITLE,
            "conversations::set_title",
        );
        check(
            &c.set_description(t.clone(), "conv".to_string(), Some("desc".into()))
                .await,
            ops.set_description,
            sentinels::CONV_SET_DESCRIPTION,
            "conversations::set_description",
        );
    }

    if let Some(m) = proto.clone().membership() {
        let ops = m.supported();
        let who = daemon_api::Participant::Contact(ContactInfo::default());
        check(
            &m.invite(MemberInviteArgs {
                transport: t.clone(),
                conv: "conv".to_string(),
                who: who.clone(),
                message: None,
                op_id: None,
            })
            .await,
            ops.invite,
            sentinels::MEMBER_INVITE,
            "membership::invite",
        );
        check(
            &m.remove(MemberRemoveArgs {
                transport: t.clone(),
                conv: "conv".to_string(),
                who: who.clone(),
                reason: None,
                op_id: None,
            })
            .await,
            ops.remove,
            sentinels::MEMBER_REMOVE,
            "membership::remove",
        );
        check(
            &m.ban(MemberBanArgs {
                transport: t.clone(),
                conv: "conv".to_string(),
                who: who.clone(),
                reason: None,
                op_id: None,
            })
            .await,
            ops.ban,
            sentinels::MEMBER_BAN,
            "membership::ban",
        );
        check(
            &m.set_role(MemberSetRoleArgs {
                transport: t.clone(),
                conv: "conv".to_string(),
                who,
                role: daemon_api::MemberRole::default(),
                op_id: None,
            })
            .await,
            ops.set_role,
            sentinels::MEMBER_SET_ROLE,
            "membership::set_role",
        );
    }

    if let Some(c) = proto.clone().contacts() {
        let ops = c.supported();
        check(
            &c.get_profile(t.clone(), ContactInfo::default()).await,
            ops.get_profile,
            sentinels::CONTACT_GET_PROFILE,
            "contacts::get_profile",
        );
        check(
            &c.set_alias(t.clone(), ContactInfo::default(), Some("alias".into()))
                .await,
            ops.set_alias,
            sentinels::CONTACT_SET_ALIAS,
            "contacts::set_alias",
        );
        // `action_menu` is a non-Result accessor: unsupported ⟹ None.
        let menu = c.action_menu(t.clone(), ContactInfo::default());
        if !ops.action_menu {
            assert!(
                menu.is_none(),
                "contacts::action_menu advertised unsupported must return None; got {menu:?}"
            );
        }
    }

    if let Some(r) = proto.clone().roster() {
        let ops = r.supported();
        check(
            &r.add(t.clone(), ContactInfo::default()).await,
            ops.add,
            sentinels::ROSTER_ADD,
            "roster::add",
        );
        check(
            &r.update(t.clone(), ContactInfo::default()).await,
            ops.update,
            sentinels::ROSTER_UPDATE,
            "roster::update",
        );
        check(
            &r.remove(t.clone(), ContactInfo::default()).await,
            ops.remove,
            sentinels::ROSTER_REMOVE,
            "roster::remove",
        );
    }

    if let Some(d) = proto.clone().directory() {
        let supported = d.supported();
        check(
            &d.search_contacts(t.clone(), Some("q".into())).await,
            supported,
            sentinels::DIRECTORY_SEARCH,
            "directory::search_contacts",
        );
    }

    if let Some(ft) = proto.clone().file_transfer() {
        let ops = ft.supported();
        let transfer = FileTransfer {
            name: "f.bin".to_string(),
            blob: daemon_common::BlobRef::new(daemon_common::ContentHash::new([0u8; 32]), 0),
            ..Default::default()
        };
        check(
            &ft.send(t.clone(), transfer.clone()).await,
            ops.send,
            sentinels::FILE_TRANSFER_SEND,
            "file_transfer::send",
        );
        check(
            &ft.receive(t.clone(), transfer).await,
            ops.receive,
            sentinels::FILE_TRANSFER_RECEIVE,
            "file_transfer::receive",
        );
    }
}

/// Apply the biconditional to one verb: forward (unsupported ⟹ sentinel) or reverse (supported ⟹
/// not sentinel), keyed on `sentinel`.
fn check<T: std::fmt::Debug>(
    res: &Result<T, ApiError>,
    supported: bool,
    sentinel: &str,
    verb: &str,
) {
    if supported {
        assert_not_sentinel(res, sentinel, verb);
    } else {
        assert_is_sentinel(res, sentinel, verb);
    }
}
