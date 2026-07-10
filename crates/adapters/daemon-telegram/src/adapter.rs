// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `TelegramAdapter` — the Telegram transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! The twin of `daemon-matrix`: the feature-trait method bodies execute Telegram operations against
//! the live grammers clients that [`serve`](crate::client::serve) brings up. Because the trait
//! methods only get `&self`, the adapter holds a [`LiveClients`] registry (populated at bring-up)
//! that the methods read to resolve the per-account client. To keep grammers confined to
//! [`crate::client`], the registry holds `Arc<dyn TelegramClient>` — the SDK-agnostic verb seam this
//! module defines — never a grammers type.
//!
//! `supported()` is HONEST: it reports only the ops actually wired against grammers 0.10's friendly
//! API. Telegram's ocap peer model (a management op needs a cached [`grammers_session`] `PeerRef`,
//! not a bare id) means title/topic edits, chat creation, and member *invite* are not wired in this
//! phase (no friendly-API counterpart); they inherit the trait's `Unsupported` default and
//! `supported()` reports `false` for them, rather than pretending.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ChannelJoinDetails,
    ConnectionState, ContactInfo, ContactsOps, ConvSendArgs, ConversationInfo, ConversationOps,
    MemberBanArgs, MemberRemoveArgs, MembershipOps, MessagingProtocol, NodeApi, Participant,
    PresenceState, RosterOps, SupportsContacts, SupportsConversations, SupportsDirectory,
    SupportsMembership, SupportsRoster, TransportAdapter, TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use crate::config::TelegramConfig;
use crate::mapping::parse_chat_id;
use crate::{LiveClients, FAMILY};

/// The SDK-agnostic verb seam: the Telegram operations the adapter's feature-trait bodies execute,
/// implemented against grammers by [`crate::client`] (and by a mock in tests). Ids are the numeric
/// Telegram peer ids the daemon-opaque conversation/contact ids parse to. Only the ops the adapter
/// honestly reports as supported are declared here; unsupported ops are simply not part of the seam.
#[async_trait]
pub trait TelegramClient: Send + Sync {
    /// Post `text` to the chat `chat_id`.
    async fn send_text(&self, chat_id: i64, text: &str) -> Result<(), ApiError>;
    /// The account's known conversations (its cached dialogs), projected to wire DTOs.
    async fn list_conversations(&self, transport: &TransportId) -> Vec<ConversationInfo>;
    /// One known conversation by chat id, if the account has it cached.
    async fn get_conversation(
        &self,
        transport: &TransportId,
        chat_id: i64,
    ) -> Option<ConversationInfo>;
    /// Join a public group/channel by `@username` (or a public link handle).
    async fn join_channel(
        &self,
        transport: &TransportId,
        target: &str,
    ) -> Result<ConversationInfo, ApiError>;
    /// Leave the chat `chat_id` (delete the dialog / leave the group).
    async fn leave(&self, chat_id: i64) -> Result<(), ApiError>;
    /// Kick `user_id` from `chat_id` (removable; they may re-join).
    async fn remove(&self, chat_id: i64, user_id: i64) -> Result<(), ApiError>;
    /// Ban `user_id` from `chat_id` (revoke all rights).
    async fn ban(&self, chat_id: i64, user_id: i64) -> Result<(), ApiError>;
    /// Render a short profile card for `user_id`.
    async fn get_profile(&self, user_id: i64) -> Result<String, ApiError>;
    /// Resolve a `@username` search `query` into matching contacts (0 or 1 for a username lookup).
    async fn search_contacts(&self, query: &str) -> Result<Vec<ContactInfo>, ApiError>;
    /// The account's server-side contact roster (`contacts.getContacts`), projected to wire DTOs.
    async fn roster_list(&self, transport: &TransportId) -> Result<Vec<ContactInfo>, ApiError>;
    /// Add/upsert `user_id` to the roster with `first_name` (`contacts.addContact`; the same call
    /// backs both add and update, since addContact refreshes the name of an existing contact).
    async fn roster_add(&self, user_id: i64, first_name: &str) -> Result<(), ApiError>;
    /// Remove `user_id` from the roster (`contacts.deleteContacts`).
    async fn roster_remove(&self, user_id: i64) -> Result<(), ApiError>;
}

/// The Telegram transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the multi-account bring-up, plus the [`LiveClients`]
/// registry the verb bodies resolve their per-account client from.
pub struct TelegramAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: TelegramConfig,
    clients: LiveClients,
}

impl TelegramAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved Telegram `cfg`. The live
    /// client registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: TelegramConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            clients: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live `client` under its instance-qualified `transport` — the same registration
    /// bring-up performs. Public so tests can stage a mock client exactly the way bring-up would.
    pub async fn register_live_client(
        &self,
        transport: TransportId,
        client: Arc<dyn TelegramClient>,
    ) {
        self.clients.write().await.insert(transport, client);
    }

    /// Resolve the live client for an instance-qualified `transport`. `Unsupported` when the account
    /// is not (yet) connected.
    async fn client_for(
        &self,
        transport: &TransportId,
    ) -> Result<Arc<dyn TelegramClient>, ApiError> {
        self.clients
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "telegram account {} is not connected",
                    transport.as_str()
                ))
            })
    }
}

/// Parse a daemon-opaque conversation id into the numeric chat id grammers indexes on.
fn conv_chat_id(conv: &str) -> Result<i64, ApiError> {
    parse_chat_id(conv).ok_or_else(|| ApiError::Other(format!("invalid telegram chat id {conv}")))
}

/// Parse a roster `ContactInfo`'s opaque id into the numeric Telegram user id (same convention as
/// [`SupportsContacts::get_profile`]).
fn roster_user_id(contact: &ContactInfo) -> Result<i64, ApiError> {
    parse_chat_id(&contact.id)
        .ok_or_else(|| ApiError::Other(format!("invalid telegram user id {}", contact.id)))
}

/// Extract the target Telegram user id from a membership `Participant`. Telegram membership targets a
/// human user (`Participant::Contact`, `id` = numeric user id); the daemon `Agent` participant is a
/// Rooms-only extension and is unsupported here.
fn contact_user_id(who: &Participant) -> Result<i64, ApiError> {
    match who {
        Participant::Contact(c) => parse_chat_id(&c.id)
            .ok_or_else(|| ApiError::Other(format!("invalid telegram user id {}", c.id))),
        Participant::Agent { .. } => Err(ApiError::Unsupported(
            "telegram membership targets a Telegram user (Participant::Contact), not an agent"
                .into(),
        )),
    }
}

#[async_trait]
impl TransportAdapter for TelegramAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Telegram".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: false,
                room_enumeration: true,
                file_transfer: false,
                interactive_auth: true,
            },
            account_schema: AccountSettingsSchema::default(),
            policies: Vec::new(),
            // Per-verb ops (wire v33) are enriched centrally in the host `transport_adapters()` from
            // the feature-trait `supported()` probes; the adapter leaves them at default here.
            ..Default::default()
        }
    }

    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        self.provisioning
            .bound_accounts(FAMILY)
            .into_iter()
            .map(|acct| {
                let connection = if self
                    .provisioning
                    .account_credential(&acct.credential_ref)
                    .is_some()
                {
                    ConnectionState::Connected
                } else {
                    ConnectionState::Offline
                };
                TransportInstanceInfo {
                    display_name: acct.transport_instance.as_str().to_string(),
                    transport: acct.transport_instance,
                    family: FAMILY.to_string(),
                    connection,
                    presence: PresenceState::default(),
                    bound_profile: None,
                    reason: None,
                    message: None,
                    fatal: false,
                    // Wire v35: enabled/label are node-overlaid from the store; report inert default.
                    enabled: true,
                    label: None,
                }
            })
            .collect()
    }

    async fn serve(self: Arc<Self>, api: Arc<dyn NodeApi>) {
        crate::client::serve(
            api,
            self.provisioning.clone(),
            self.cfg.clone(),
            self.clients.clone(),
        )
        .await
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for TelegramAdapter {
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
}

#[async_trait]
impl SupportsConversations for TelegramAdapter {
    fn supported(&self) -> ConversationOps {
        // The honest subset wired against grammers 0.10's friendly API. Chat creation, delete, and
        // title/topic/description edits have no friendly-API counterpart (they need raw MTProto +
        // an ocap `PeerRef`) and are deferred — reported `false` rather than pretended.
        ConversationOps {
            create: false,
            join_channel: true,
            leave: true,
            delete: false,
            send: true,
            set_topic: false,
            set_title: false,
            set_description: false,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        match self.client_for(&transport).await {
            Ok(client) => client.list_conversations(&transport).await,
            Err(_) => Vec::new(),
        }
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        let client = self.client_for(&transport).await.ok()?;
        let chat_id = parse_chat_id(&conv)?;
        client.get_conversation(&transport, chat_id).await
    }

    async fn channel_join_details(&self, _transport: TransportId) -> ChannelJoinDetails {
        // Telegram joins a public group/channel by its `@username` — no per-channel nick or password.
        ChannelJoinDetails {
            nickname_supported: false,
            password_supported: false,
            ..ChannelJoinDetails::default()
        }
    }

    async fn join_channel(
        &self,
        transport: TransportId,
        details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let client = self.client_for(&transport).await?;
        let target = details
            .name
            .clone()
            .or_else(|| details.extras.values.get("username").cloned())
            .ok_or_else(|| ApiError::Other("telegram join requires a @username".into()))?;
        client.join_channel(&transport, &target).await
    }

    async fn leave(&self, transport: TransportId, conv: String) -> Result<(), ApiError> {
        let client = self.client_for(&transport).await?;
        client.leave(conv_chat_id(&conv)?).await
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
            op_id: _,
        } = args;
        // The Telegram account is always the sender; `from` attribution is not forwarded onto the
        // wire (grammers posts as the bound account), matching the outbound projector.
        let client = self.client_for(&transport).await?;
        client.send_text(conv_chat_id(&conv)?, &message.text).await
    }
}

#[async_trait]
impl SupportsMembership for TelegramAdapter {
    fn supported(&self) -> MembershipOps {
        // remove (kick) and ban map to friendly-API calls; invite and set_role have no friendly
        // counterpart in grammers 0.10 (raw MTProto), so they are deferred and reported `false`.
        MembershipOps {
            invite: false,
            remove: true,
            ban: true,
            set_role: false,
        }
    }

    async fn remove(&self, args: MemberRemoveArgs) -> Result<(), ApiError> {
        let MemberRemoveArgs {
            transport,
            conv,
            who,
            reason: _reason,
            op_id: _,
        } = args;
        let client = self.client_for(&transport).await?;
        client
            .remove(conv_chat_id(&conv)?, contact_user_id(&who)?)
            .await
    }

    async fn ban(&self, args: MemberBanArgs) -> Result<(), ApiError> {
        let MemberBanArgs {
            transport,
            conv,
            who,
            reason: _reason,
            op_id: _,
        } = args;
        let client = self.client_for(&transport).await?;
        client
            .ban(conv_chat_id(&conv)?, contact_user_id(&who)?)
            .await
    }
}

#[async_trait]
impl SupportsRoster for TelegramAdapter {
    fn supported(&self) -> RosterOps {
        // The full server-side contact roster is wired against the raw Telegram TL API
        // (`contacts.getContacts` / `addContact` / `deleteContacts`). `update` maps to the same
        // `addContact` upsert as `add` (it refreshes an existing contact's name), so it is honestly
        // supported. These ops are user-account-only; a bot session gets a clean `Unsupported` at
        // call time (`supported()` is per-adapter, so it advertises the user-account capability).
        RosterOps {
            list: true,
            add: true,
            update: true,
            remove: true,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ContactInfo> {
        // Mirrors `SupportsConversations::list`: an unconnected account (or a bot, which has no
        // roster) yields an empty list; the host sorts + pages it centrally.
        match self.client_for(&transport).await {
            Ok(client) => client.roster_list(&transport).await.unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    async fn add(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        let client = self.client_for(&transport).await?;
        client
            .roster_add(
                roster_user_id(&contact)?,
                contact.display_name.as_deref().unwrap_or_default(),
            )
            .await
    }

    async fn update(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        // Telegram's `contacts.addContact` upserts, so an update is the same call as an add: it
        // refreshes the contact's stored first name.
        let client = self.client_for(&transport).await?;
        client
            .roster_add(
                roster_user_id(&contact)?,
                contact.display_name.as_deref().unwrap_or_default(),
            )
            .await
    }

    async fn remove(&self, transport: TransportId, contact: ContactInfo) -> Result<(), ApiError> {
        let client = self.client_for(&transport).await?;
        client.roster_remove(roster_user_id(&contact)?).await
    }
}

#[async_trait]
impl SupportsContacts for TelegramAdapter {
    fn supported(&self) -> ContactsOps {
        // A remote profile fetch is wired; per-contact alias / action menu have no counterpart.
        ContactsOps {
            get_profile: true,
            action_menu: false,
            set_alias: false,
        }
    }

    async fn get_profile(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Result<String, ApiError> {
        let client = self.client_for(&transport).await?;
        let user_id = parse_chat_id(&contact.id)
            .ok_or_else(|| ApiError::Other(format!("invalid telegram user id {}", contact.id)))?;
        client.get_profile(user_id).await
    }
}

#[async_trait]
impl SupportsDirectory for TelegramAdapter {
    fn supported(&self) -> bool {
        true
    }

    async fn search_contacts(
        &self,
        transport: TransportId,
        query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        let client = self.client_for(&transport).await?;
        client.search_contacts(&query.unwrap_or_default()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use daemon_protocol::UserMsg;

    use crate::account::AccountMode;
    use daemon_host::ProvisionedAccount;

    /// A no-op provisioning seam: the verb tests resolve the live client from the seeded registry,
    /// never from provisioning, so empty answers suffice.
    struct MockProvisioning;
    impl AccountProvisioning for MockProvisioning {
        fn bound_accounts(&self, _family: &str) -> Vec<ProvisionedAccount> {
            Vec::new()
        }
        fn account_credential(&self, _credential_ref: &str) -> Option<String> {
            None
        }
        fn store_account_credential(&self, _r: &str, _b: &str) -> Result<(), ApiError> {
            Ok(())
        }
    }

    /// A mock verb seam that records the calls the adapter routes to it.
    #[derive(Default)]
    struct MockClient {
        sent: Mutex<Vec<(i64, String)>>,
        banned: Mutex<Vec<(i64, i64)>>,
        roster: Mutex<Vec<ContactInfo>>,
    }

    #[async_trait]
    impl TelegramClient for MockClient {
        async fn send_text(&self, chat_id: i64, text: &str) -> Result<(), ApiError> {
            self.sent.lock().unwrap().push((chat_id, text.to_string()));
            Ok(())
        }
        async fn list_conversations(&self, _t: &TransportId) -> Vec<ConversationInfo> {
            Vec::new()
        }
        async fn get_conversation(&self, _t: &TransportId, _c: i64) -> Option<ConversationInfo> {
            None
        }
        async fn join_channel(
            &self,
            _t: &TransportId,
            _target: &str,
        ) -> Result<ConversationInfo, ApiError> {
            Err(ApiError::Unsupported("test".into()))
        }
        async fn leave(&self, _chat_id: i64) -> Result<(), ApiError> {
            Ok(())
        }
        async fn remove(&self, _chat_id: i64, _user_id: i64) -> Result<(), ApiError> {
            Ok(())
        }
        async fn ban(&self, chat_id: i64, user_id: i64) -> Result<(), ApiError> {
            self.banned.lock().unwrap().push((chat_id, user_id));
            Ok(())
        }
        async fn get_profile(&self, user_id: i64) -> Result<String, ApiError> {
            Ok(format!("user_id: {user_id}"))
        }
        async fn search_contacts(&self, _query: &str) -> Result<Vec<ContactInfo>, ApiError> {
            Ok(Vec::new())
        }
        async fn roster_list(&self, _t: &TransportId) -> Result<Vec<ContactInfo>, ApiError> {
            Ok(self.roster.lock().unwrap().clone())
        }
        async fn roster_add(&self, user_id: i64, first_name: &str) -> Result<(), ApiError> {
            let mut roster = self.roster.lock().unwrap();
            let id = user_id.to_string();
            let contact = ContactInfo {
                id: id.clone(),
                display_name: Some(first_name.to_string()),
                ..ContactInfo::default()
            };
            match roster.iter_mut().find(|c| c.id == id) {
                Some(slot) => *slot = contact,
                None => roster.push(contact),
            }
            Ok(())
        }
        async fn roster_remove(&self, user_id: i64) -> Result<(), ApiError> {
            self.roster
                .lock()
                .unwrap()
                .retain(|c| c.id != user_id.to_string());
            Ok(())
        }
    }

    async fn adapter_with(
        transport: &TransportId,
        client: Arc<MockClient>,
    ) -> Arc<TelegramAdapter> {
        let adapter = TelegramAdapter::new(Arc::new(MockProvisioning), TelegramConfig::default());
        adapter
            .register_live_client(transport.clone(), client)
            .await;
        adapter
    }

    #[test]
    fn supported_reports_the_honest_telegram_subset() {
        let adapter = TelegramAdapter::new(Arc::new(MockProvisioning), TelegramConfig::default());
        let conv = SupportsConversations::supported(&*adapter);
        assert!(conv.send && conv.join_channel && conv.leave);
        assert!(
            !conv.create
                && !conv.delete
                && !conv.set_topic
                && !conv.set_title
                && !conv.set_description
        );
        let mem = SupportsMembership::supported(&*adapter);
        assert!(mem.remove && mem.ban);
        assert!(!mem.invite && !mem.set_role);
        let contacts = SupportsContacts::supported(&*adapter);
        assert!(contacts.get_profile && !contacts.action_menu && !contacts.set_alias);
        let roster = SupportsRoster::supported(&*adapter);
        assert!(roster.list && roster.add && roster.update && roster.remove);
        assert!(SupportsDirectory::supported(&*adapter));
    }

    #[test]
    fn info_advertises_interactive_auth_and_rooms() {
        let adapter = TelegramAdapter::new(Arc::new(MockProvisioning), TelegramConfig::default());
        let info = adapter.info();
        assert_eq!(info.family, "telegram");
        assert!(info.capabilities.interactive_auth);
        assert!(info.capabilities.rooms && info.capabilities.direct_messages);
    }

    #[tokio::test]
    async fn send_routes_to_the_resolved_client() {
        let transport = TransportId::new("telegram/555");
        let client = Arc::new(MockClient::default());
        let adapter = adapter_with(&transport, client.clone()).await;

        SupportsConversations::send(
            &*adapter,
            ConvSendArgs {
                transport: transport.clone(),
                conv: "-100".to_string(),
                from: None,
                message: UserMsg::new("hello"),
                op_id: None,
            },
        )
        .await
        .expect("send routes to the live client");
        assert_eq!(
            client.sent.lock().unwrap().as_slice(),
            &[(-100, "hello".to_string())]
        );
    }

    #[tokio::test]
    async fn send_on_unconnected_account_is_unsupported() {
        let adapter = TelegramAdapter::new(Arc::new(MockProvisioning), TelegramConfig::default());
        let err = SupportsConversations::send(
            &*adapter,
            ConvSendArgs {
                transport: TransportId::new("telegram/999"),
                conv: "1".to_string(),
                from: None,
                message: UserMsg::new("x"),
                op_id: None,
            },
        )
        .await
        .expect_err("an unconnected account cannot send");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn ban_rejects_agent_participants_and_accepts_contacts() {
        let transport = TransportId::new("telegram/555");
        let client = Arc::new(MockClient::default());
        let adapter = adapter_with(&transport, client.clone()).await;

        // A contact is the accepted membership target.
        SupportsMembership::ban(
            &*adapter,
            MemberBanArgs {
                transport: transport.clone(),
                conv: "-100".to_string(),
                who: Participant::Contact(ContactInfo {
                    id: "42".to_string(),
                    ..ContactInfo::default()
                }),
                reason: None,
                op_id: None,
            },
        )
        .await
        .expect("banning a contact routes to the client");
        assert_eq!(client.banned.lock().unwrap().as_slice(), &[(-100, 42)]);

        // An agent is not a Telegram membership target.
        let err = SupportsMembership::ban(
            &*adapter,
            MemberBanArgs {
                transport,
                conv: "-100".to_string(),
                who: Participant::Agent {
                    profile: daemon_common::ProfileRef::new("alpha"),
                    member: "@agent".to_string(),
                },
                reason: None,
                op_id: None,
            },
        )
        .await
        .expect_err("an agent is not a membership target");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn roster_add_update_list_remove_round_trip() {
        let transport = TransportId::new("telegram/555");
        let client = Arc::new(MockClient::default());
        let adapter = adapter_with(&transport, client.clone()).await;

        let contact = ContactInfo {
            id: "42".to_string(),
            display_name: Some("Alice".to_string()),
            ..ContactInfo::default()
        };
        SupportsRoster::add(&*adapter, transport.clone(), contact.clone())
            .await
            .expect("add routes to the client");
        let listed = SupportsRoster::list(&*adapter, transport.clone()).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "42");
        assert_eq!(listed[0].display_name.as_deref(), Some("Alice"));

        // update maps to the same addContact upsert (refreshes the name in place, no duplicate).
        let renamed = ContactInfo {
            display_name: Some("Alice B.".to_string()),
            ..contact.clone()
        };
        SupportsRoster::update(&*adapter, transport.clone(), renamed)
            .await
            .expect("update routes to the client");
        let listed = SupportsRoster::list(&*adapter, transport.clone()).await;
        assert_eq!(listed.len(), 1, "update upserts in place");
        assert_eq!(listed[0].display_name.as_deref(), Some("Alice B."));

        SupportsRoster::remove(&*adapter, transport.clone(), contact)
            .await
            .expect("remove routes to the client");
        assert!(SupportsRoster::list(&*adapter, transport).await.is_empty());
    }

    #[tokio::test]
    async fn roster_add_rejects_a_non_numeric_id() {
        let transport = TransportId::new("telegram/555");
        let client = Arc::new(MockClient::default());
        let adapter = adapter_with(&transport, client).await;
        let err = SupportsRoster::add(
            &*adapter,
            transport,
            ContactInfo {
                id: "@notnumeric".to_string(),
                ..ContactInfo::default()
            },
        )
        .await
        .expect_err("a non-numeric contact id is rejected");
        assert!(matches!(err, ApiError::Other(_)));
    }

    #[tokio::test]
    async fn roster_ops_on_unconnected_account_are_unsupported() {
        let adapter = TelegramAdapter::new(Arc::new(MockProvisioning), TelegramConfig::default());
        let transport = TransportId::new("telegram/999");
        // list is lenient (empty), matching conversation list.
        assert!(SupportsRoster::list(&*adapter, transport.clone())
            .await
            .is_empty());
        // mutations on an unconnected account surface Unsupported.
        let err = SupportsRoster::remove(
            &*adapter,
            transport,
            ContactInfo {
                id: "1".to_string(),
                ..ContactInfo::default()
            },
        )
        .await
        .expect_err("an unconnected account cannot mutate its roster");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }

    #[test]
    fn account_mode_is_re_exported_for_wiring() {
        // Compile-time check that the public surface the Phase 2 wiring needs is reachable.
        let _ = AccountMode::Bot;
    }
}
