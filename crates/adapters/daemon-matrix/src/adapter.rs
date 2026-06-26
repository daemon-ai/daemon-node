//! `MatrixAdapter` — the Matrix transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! The second reference implementor of the messaging-adapter interface (daemon-messaging-adapter-spec.md
//! §10.2; the port rationale + source mapping is `daemon-matrix-bifrost-port-reference.md`). It proves
//! the interface generalizes: a *different* `supported()` set than the Rooms adapter, on the same
//! traits, with **no host changes**.
//!
//! The feature-trait method bodies execute real Matrix client-server operations against the live
//! `matrix-sdk` [`Client`]s that [`serve`](crate::serve) brings up. Because the trait methods only get
//! `&self`, the adapter holds a [`LiveClients`] registry that `serve` populates at bring-up and the
//! methods read to resolve the per-account client (the architectural seam this adapter adds vs. the
//! Rooms adapter, which owns its runtime via a command channel). Unlike Rooms, no command channel is
//! needed: a `matrix_sdk::Client` is `Send + Sync` and async, so a verb body calls it directly.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ChannelJoinDetails,
    ConnectionState, ContactInfo, ContactsOps, ConversationInfo, ConversationOps,
    CreateConversationDetails, MemberRole, MembershipOps, MessagingProtocol, NodeApi, Participant,
    PresenceState, SupportsContacts, SupportsConversations, SupportsDirectory, SupportsMembership,
    TransportAdapter, TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::{TransportId, UserMsg};

use matrix_sdk::ruma::api::client::room::{create_room, Visibility};
use matrix_sdk::ruma::events::room::message::RoomMessageEventContent;
use matrix_sdk::ruma::{Int, OwnedUserId, RoomId, RoomOrAliasId, UserId};
use matrix_sdk::{Client, Room};

use crate::mapping::{contact_from, role_to_power, room_to_info};
use crate::{serve, LiveClients, MatrixConfig, FAMILY};

/// The page size for a user-directory search (`SupportsDirectory::search_contacts`).
const DIRECTORY_SEARCH_LIMIT: u64 = 50;

/// The Matrix transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the existing multi-account bring-up, plus the
/// [`LiveClients`] registry the management verb bodies resolve their per-account client from.
pub struct MatrixAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: MatrixConfig,
    clients: LiveClients,
}

impl MatrixAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved Matrix `cfg`. The live
    /// client registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: MatrixConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            clients: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Resolve the live `Client` for an instance-qualified `transport` (`matrix/@user:hs`, the same
    /// key `instances()` emits and `serve` registers). `Unsupported` when the account is not (yet)
    /// connected (e.g. before `serve` brought it up, or it has no stored session).
    async fn client_for(&self, transport: &TransportId) -> Result<Client, ApiError> {
        self.clients
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "matrix account {} is not connected",
                    transport.as_str()
                ))
            })
    }

    /// Resolve a known `Room` by its opaque `conv` (a Matrix room id) on `transport`.
    async fn room_for(&self, transport: &TransportId, conv: &str) -> Result<Room, ApiError> {
        let client = self.client_for(transport).await?;
        let room_id = RoomId::parse(conv)
            .map_err(|e| ApiError::Other(format!("invalid matrix room id {conv}: {e}")))?;
        client
            .get_room(&room_id)
            .ok_or_else(|| ApiError::Other(format!("matrix room {conv} not found")))
    }
}

/// Extract the target Matrix user id from a membership `Participant`. Matrix membership targets a
/// human Matrix user (`Participant::Contact`, `id` = MXID `@user:hs`); the daemon `Agent` participant
/// (a Rooms-only extension, §8) is unsupported here.
fn contact_mxid(who: &Participant) -> Result<OwnedUserId, ApiError> {
    match who {
        Participant::Contact(c) => UserId::parse(&c.id)
            .map_err(|e| ApiError::Other(format!("invalid matrix user id {}: {e}", c.id))),
        Participant::Agent { .. } => Err(ApiError::Unsupported(
            "matrix membership targets a Matrix user (Participant::Contact), not an agent".into(),
        )),
    }
}

#[async_trait]
impl TransportAdapter for MatrixAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Matrix".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: true,
                room_enumeration: true,
                file_transfer: true,
                interactive_auth: true,
            },
            account_schema: AccountSettingsSchema::default(),
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
                }
            })
            .collect()
    }

    async fn serve(self: Arc<Self>, api: Arc<dyn NodeApi>) {
        serve(
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
impl MessagingProtocol for MatrixAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
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
impl SupportsConversations for MatrixAdapter {
    fn supported(&self) -> ConversationOps {
        // Matrix's subset (vs. Rooms' full set): list/get/send/set_topic/set_title + create/
        // join_channel/leave are wired against matrix-sdk; set_description has no Matrix counterpart
        // and delete (room destroy) is not a Matrix operation (leaving is the closest, exposed as
        // `leave`). See daemon-matrix-bifrost-port-reference.md §4.1.
        ConversationOps {
            create: true,
            join_channel: true,
            leave: true,
            delete: false,
            send: true,
            set_topic: true,
            set_title: true,
            set_description: false,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        let Ok(client) = self.client_for(&transport).await else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for room in client.rooms() {
            out.push(room_to_info(&transport, &room).await);
        }
        out
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        let room = self.room_for(&transport, &conv).await.ok()?;
        Some(room_to_info(&transport, &room).await)
    }

    async fn create(
        &self,
        transport: TransportId,
        details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let client = self.client_for(&transport).await?;
        let v = &details.extras.values;

        let mut request = create_room::v3::Request::new();
        request.name = v.get("name").cloned();
        request.topic = v.get("topic").cloned();
        request.room_alias_name = v.get("alias").cloned();

        let mut invites = Vec::with_capacity(details.participants.len());
        for c in &details.participants {
            let user = UserId::parse(&c.id)
                .map_err(|e| ApiError::Other(format!("invalid matrix invitee {}: {e}", c.id)))?;
            invites.push(user);
        }
        request.invite = invites;

        request.is_direct = matches!(v.get("kind").map(String::as_str), Some("dm") | Some("Dm"));
        let public = matches!(v.get("visibility").map(String::as_str), Some("public"));
        request.visibility = if public {
            Visibility::Public
        } else {
            Visibility::Private
        };
        request.preset = Some(if public {
            create_room::v3::RoomPreset::PublicChat
        } else {
            create_room::v3::RoomPreset::PrivateChat
        });

        let room = client
            .create_room(request)
            .await
            .map_err(|e| ApiError::Other(format!("matrix create_room: {e}")))?;
        Ok(room_to_info(&transport, &room).await)
    }

    async fn channel_join_details(&self, _transport: TransportId) -> ChannelJoinDetails {
        // Matrix joins by room id/alias only — no per-channel nickname or password.
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
            .or_else(|| details.extras.values.get("id").cloned())
            .ok_or_else(|| ApiError::Other("matrix join requires a room id or alias".into()))?;
        let id = RoomOrAliasId::parse(&target)
            .map_err(|e| ApiError::Other(format!("invalid matrix room id/alias {target}: {e}")))?;
        let room = client
            .join_room_by_id_or_alias(&id, &[])
            .await
            .map_err(|e| ApiError::Other(format!("matrix join_room: {e}")))?;
        Ok(room_to_info(&transport, &room).await)
    }

    async fn leave(&self, transport: TransportId, conv: String) -> Result<(), ApiError> {
        let room = self.room_for(&transport, &conv).await?;
        room.leave()
            .await
            .map_err(|e| ApiError::Other(format!("matrix leave: {e}")))
    }

    async fn send(
        &self,
        transport: TransportId,
        conv: String,
        _from: Option<Participant>,
        message: UserMsg,
    ) -> Result<(), ApiError> {
        // The Matrix account user is always the sender; `from` attribution is not forwarded onto the
        // wire (matrix-sdk posts as the bound account). The outbound projector posts the same way.
        let room = self.room_for(&transport, &conv).await?;
        room.send(RoomMessageEventContent::text_plain(message.text))
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("matrix send: {e}")))
    }

    async fn set_topic(
        &self,
        transport: TransportId,
        conv: String,
        topic: Option<String>,
    ) -> Result<(), ApiError> {
        let room = self.room_for(&transport, &conv).await?;
        room.set_room_topic(topic.as_deref().unwrap_or(""))
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("matrix set_topic: {e}")))
    }

    async fn set_title(
        &self,
        transport: TransportId,
        conv: String,
        title: Option<String>,
    ) -> Result<(), ApiError> {
        let room = self.room_for(&transport, &conv).await?;
        room.set_name(title.unwrap_or_default())
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("matrix set_title: {e}")))
    }
}

#[async_trait]
impl SupportsMembership for MatrixAdapter {
    fn supported(&self) -> MembershipOps {
        // Matrix membership administration is richer than Rooms': invite/remove/ban map to
        // `m.room.member` invite/kick/ban; set_role maps to `m.room.power_levels`.
        MembershipOps {
            invite: true,
            remove: true,
            ban: true,
            set_role: true,
        }
    }

    async fn invite(
        &self,
        transport: TransportId,
        conv: String,
        who: Participant,
        _message: Option<String>,
    ) -> Result<(), ApiError> {
        let user = contact_mxid(&who)?;
        let room = self.room_for(&transport, &conv).await?;
        room.invite_user_by_id(&user)
            .await
            .map_err(|e| ApiError::Other(format!("matrix invite: {e}")))
    }

    async fn remove(
        &self,
        transport: TransportId,
        conv: String,
        who: Participant,
        reason: Option<String>,
    ) -> Result<(), ApiError> {
        let user = contact_mxid(&who)?;
        let room = self.room_for(&transport, &conv).await?;
        room.kick_user(&user, reason.as_deref())
            .await
            .map_err(|e| ApiError::Other(format!("matrix remove: {e}")))
    }

    async fn ban(
        &self,
        transport: TransportId,
        conv: String,
        who: Participant,
        reason: Option<String>,
    ) -> Result<(), ApiError> {
        let user = contact_mxid(&who)?;
        let room = self.room_for(&transport, &conv).await?;
        room.ban_user(&user, reason.as_deref())
            .await
            .map_err(|e| ApiError::Other(format!("matrix ban: {e}")))
    }

    async fn set_role(
        &self,
        transport: TransportId,
        conv: String,
        who: Participant,
        role: MemberRole,
    ) -> Result<(), ApiError> {
        let user = contact_mxid(&who)?;
        let room = self.room_for(&transport, &conv).await?;
        let level = Int::from(role_to_power(role));
        room.update_power_levels(vec![(&*user, level)])
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("matrix set_role: {e}")))
    }
}

#[async_trait]
impl SupportsContacts for MatrixAdapter {
    fn supported(&self) -> ContactsOps {
        // Matrix exposes a remote profile fetch (`/profile/{user}`); it has no native per-contact
        // alias or action menu, so those stay off (daemon-matrix-bifrost-port-reference.md §4).
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
        let user = UserId::parse(&contact.id)
            .map_err(|e| ApiError::Other(format!("invalid matrix user id {}: {e}", contact.id)))?;
        let resp = client
            .account()
            .fetch_user_profile_of(&user)
            .await
            .map_err(|e| ApiError::Other(format!("matrix get_profile: {e}")))?;
        let mut lines = vec![format!("user_id: {}", contact.id)];
        if let Some(name) = resp.get("displayname").and_then(|v| v.as_str()) {
            lines.push(format!("display_name: {name}"));
        }
        if let Some(avatar) = resp.get("avatar_url").and_then(|v| v.as_str()) {
            lines.push(format!("avatar_url: {avatar}"));
        }
        Ok(lines.join("\n"))
    }
}

#[async_trait]
impl SupportsDirectory for MatrixAdapter {
    fn supported(&self) -> bool {
        true
    }

    async fn search_contacts(
        &self,
        transport: TransportId,
        query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        let client = self.client_for(&transport).await?;
        let term = query.unwrap_or_default();
        let resp = client
            .search_users(&term, DIRECTORY_SEARCH_LIMIT)
            .await
            .map_err(|e| ApiError::Other(format!("matrix directory search: {e}")))?;
        Ok(resp
            .results
            .into_iter()
            .map(|u| contact_from(u.user_id.to_string(), u.display_name))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use daemon_api::ContactInfo;
    use daemon_common::ProfileRef;
    use daemon_host::ProvisionedAccount;

    use matrix_sdk::ruma::{event_id, room_id};
    use matrix_sdk::test_utils::mocks::MatrixMockServer;
    use matrix_sdk::Client;

    /// A no-op provisioning seam: the conversation/membership verb tests resolve the live client from
    /// the seeded registry, never from provisioning, so empty answers suffice.
    struct MockProvisioning;

    impl AccountProvisioning for MockProvisioning {
        fn bound_accounts(&self, _family: &str) -> Vec<ProvisionedAccount> {
            Vec::new()
        }
        fn account_credential(&self, _credential_ref: &str) -> Option<String> {
            None
        }
        fn store_account_credential(
            &self,
            _credential_ref: &str,
            _blob: &str,
        ) -> Result<(), ApiError> {
            Ok(())
        }
    }

    /// Build an adapter and seed its live-client registry with `client` under `transport` (the seam
    /// `serve` performs at bring-up), so the `&self` verb bodies can resolve it.
    async fn adapter_with(transport: &TransportId, client: Client) -> Arc<MatrixAdapter> {
        let adapter = MatrixAdapter::new(Arc::new(MockProvisioning), MatrixConfig::default());
        adapter
            .clients
            .write()
            .await
            .insert(transport.clone(), client);
        adapter
    }

    #[test]
    fn supported_reports_the_matrix_subset_plus_extras() {
        let adapter = MatrixAdapter::new(Arc::new(MockProvisioning), MatrixConfig::default());
        let conv = SupportsConversations::supported(&*adapter);
        assert!(conv.create && conv.join_channel && conv.leave && conv.send);
        assert!(conv.set_topic && conv.set_title);
        assert!(!conv.delete && !conv.set_description);
        let mem = SupportsMembership::supported(&*adapter);
        assert!(mem.invite && mem.remove && mem.ban && mem.set_role);
        let contacts = SupportsContacts::supported(&*adapter);
        assert!(contacts.get_profile && !contacts.action_menu && !contacts.set_alias);
        assert!(
            SupportsDirectory::supported(&*adapter),
            "directory search is on"
        );
    }

    #[tokio::test]
    async fn directory_search_maps_user_directory_results() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        server.mock_user_directory().ok().mock_once().mount().await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        let contacts =
            SupportsDirectory::search_contacts(&*adapter, transport, Some("test".into()))
                .await
                .expect("directory search succeeds against the mock");
        assert!(
            contacts
                .iter()
                .any(|c| c.id == "@test:example.me" && c.display_name.as_deref() == Some("Test")),
            "expected the mapped directory hit, got {contacts:?}"
        );
    }

    #[tokio::test]
    async fn get_profile_renders_the_remote_profile() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};

        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        // No prebuilt MatrixMockServer builder for the full `/profile/{user}` endpoint, so mount a
        // raw wiremock mock on the underlying server.
        Mock::given(method("GET"))
            .and(path("/_matrix/client/v3/profile/@alice:localhost"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "displayname": "Alice",
                "avatar_url": "mxc://localhost/avatar"
            })))
            .mount(server.server())
            .await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        let profile = SupportsContacts::get_profile(
            &*adapter,
            transport,
            ContactInfo {
                id: "@alice:localhost".to_string(),
                ..ContactInfo::default()
            },
        )
        .await
        .expect("get_profile succeeds against the mock");
        assert!(profile.contains("display_name: Alice"), "got: {profile}");
        assert!(
            profile.contains("avatar_url: mxc://localhost/avatar"),
            "got: {profile}"
        );
    }

    #[tokio::test]
    async fn list_and_get_project_a_synced_room() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        let room = room_id!("!room:localhost");
        server.sync_joined_room(&client, room).await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        let convs = SupportsConversations::list(&*adapter, transport.clone()).await;
        assert!(
            convs.iter().any(|c| c.id == room.as_str()),
            "list should project the synced room, got {convs:?}"
        );
        let got = SupportsConversations::get(&*adapter, transport, room.as_str().to_string()).await;
        assert_eq!(got.expect("room present").id, room.as_str());
    }

    #[tokio::test]
    async fn send_posts_to_the_room() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        server.mock_room_state_encryption().plain().mount().await;
        let room = room_id!("!room:localhost");
        server.sync_joined_room(&client, room).await;
        server
            .mock_room_send()
            .ok(event_id!("$evt:localhost"))
            .expect(1)
            .mount()
            .await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        SupportsConversations::send(
            &*adapter,
            transport,
            room.as_str().to_string(),
            None,
            UserMsg::new("hello".to_string()),
        )
        .await
        .expect("send succeeds against the mock");
    }

    #[tokio::test]
    async fn set_topic_issues_a_state_event() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        let room = room_id!("!room:localhost");
        server.sync_joined_room(&client, room).await;
        server
            .mock_room_send_state()
            .ok(event_id!("$evt:localhost"))
            .expect(1)
            .mount()
            .await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        SupportsConversations::set_topic(
            &*adapter,
            transport,
            room.as_str().to_string(),
            Some("the topic".to_string()),
        )
        .await
        .expect("set_topic succeeds against the mock");
    }

    #[tokio::test]
    async fn create_room_returns_the_new_conversation() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        server.mock_room_state_encryption().plain().mount().await;
        server.mock_create_room().ok().mount().await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        let mut details = CreateConversationDetails::default();
        details
            .extras
            .values
            .insert("name".to_string(), "secops".to_string());
        let info = SupportsConversations::create(&*adapter, transport, details)
            .await
            .expect("create succeeds against the mock");
        assert_eq!(info.id, "!room:example.org");
    }

    #[tokio::test]
    async fn membership_rejects_agent_participants() {
        let server = MatrixMockServer::new().await;
        let client = server.client_builder().build().await;
        let room = room_id!("!room:localhost");
        server.sync_joined_room(&client, room).await;

        let transport = TransportId::new("matrix/@bot:localhost");
        let adapter = adapter_with(&transport, client).await;

        let agent = Participant::Agent {
            profile: ProfileRef::new("alpha"),
            member: "@agent".to_string(),
        };
        let err = SupportsMembership::invite(
            &*adapter,
            transport,
            room.as_str().to_string(),
            agent,
            None,
        )
        .await
        .expect_err("an agent is not a Matrix membership target");
        assert!(matches!(err, ApiError::Unsupported(_)));

        // A contact MXID is the accepted target shape (compile/shape check of the happy branch).
        let _contact = Participant::Contact(ContactInfo {
            id: "@alice:localhost".to_string(),
            ..ContactInfo::default()
        });
    }
}
