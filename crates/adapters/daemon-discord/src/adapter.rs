// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `DiscordAdapter` — the Discord transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! A sibling of `daemon-matrix`: an in-process `NodeApi`-client adapter that isolates the
//! `serenity_self` Discord SDK and drives *our* engine. The feature-trait method bodies execute real
//! Discord REST operations against the per-account [`Http`] handles that [`serve`](crate::serve)
//! brings up. Because the trait methods only get `&self`, the adapter holds a [`LiveClients`] registry
//! that `serve` populates at bring-up and the methods read to resolve the per-account REST handle.
//!
//! The `supported()` sets are **honest** for Discord: conversations expose `send` + channel-edit
//! (`set_topic`/`set_title`); membership exposes `remove` (kick) + `ban` (both native `m`-level Guild
//! ops) but not `invite` (a bot cannot force-add an arbitrary user to a guild) nor `set_role` (the
//! abstract [`MemberRole`](daemon_api::MemberRole) has no faithful mapping to a per-guild Discord role
//! id); contacts expose `get_profile`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ConnectionState,
    ContactInfo, ContactsOps, ConvSendArgs, ConversationInfo, ConversationOps, MemberBanArgs,
    MemberRemoveArgs, MembershipOps, MessagingProtocol, NodeApi, Participant, PresenceState,
    SupportsContacts, SupportsConversations, SupportsMembership, TransportAdapter,
    TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use serenity_self::builder::EditChannel;
use serenity_self::http::Http;
use serenity_self::model::channel::Channel;
use serenity_self::model::id::{ChannelId, GuildId, UserId};

use crate::mapping::{
    channel_to_info, contact_from_user, guild_channel_to_info, is_text_conversation, profile_text,
};
use crate::{serve, DiscordConfig, FAMILY};

/// The shared registry of live, per-account REST handles keyed by their instance-qualified transport
/// id (`discord/1234`). Populated by [`serve`](crate::serve) at bring-up and read by the feature-trait
/// method bodies (which only hold `&self`). A `tokio::sync::RwLock` so the read in an `async` verb body
/// never blocks the runtime.
pub type LiveClients = Arc<tokio::sync::RwLock<HashMap<TransportId, Arc<Http>>>>;

/// A page size for the current-user guild enumeration backing `list`.
const GUILD_PAGE_LIMIT: u64 = 100;

/// The Discord transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the multi-account bring-up, plus the [`LiveClients`]
/// registry the verb bodies resolve their per-account REST handle from.
pub struct DiscordAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: DiscordConfig,
    clients: LiveClients,
}

impl DiscordAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved Discord `cfg`. The live
    /// client registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: DiscordConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            clients: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live REST `http` under its instance-qualified `transport` — the same registration
    /// [`serve`](TransportAdapter::serve) performs at bring-up. Public so tests can stage a handle.
    pub async fn register_live_client(&self, transport: TransportId, http: Arc<Http>) {
        self.clients.write().await.insert(transport, http);
    }

    /// Resolve the live REST handle for an instance-qualified `transport`. `Unsupported` when the
    /// account is not (yet) connected (e.g. before `serve` brought it up).
    async fn http_for(&self, transport: &TransportId) -> Result<Arc<Http>, ApiError> {
        self.clients
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "discord account {} is not connected",
                    transport.as_str()
                ))
            })
    }

    /// Parse an opaque conversation id into a Discord [`ChannelId`].
    fn channel_id(conv: &str) -> Result<ChannelId, ApiError> {
        conv.parse::<ChannelId>()
            .map_err(|e| ApiError::Other(format!("invalid discord channel id {conv}: {e}")))
    }

    /// Resolve the [`GuildId`] owning `conv` (a channel id) — the guild membership ops (`kick`/`ban`)
    /// operate on. A DM channel has no guild, so membership there is `Unsupported`.
    async fn guild_for(&self, transport: &TransportId, conv: &str) -> Result<GuildId, ApiError> {
        let http = self.http_for(transport).await?;
        let channel_id = Self::channel_id(conv)?;
        match http.get_channel(channel_id).await {
            Ok(Channel::Guild(gc)) => Ok(gc.guild_id),
            Ok(_) => Err(ApiError::Unsupported(
                "discord membership requires a guild channel (not a DM)".into(),
            )),
            Err(e) => Err(ApiError::Other(format!("discord get_channel {conv}: {e}"))),
        }
    }
}

/// Extract the target Discord user id from a membership `Participant`. Discord membership targets a
/// human Discord user (`Participant::Contact`, `id` = the numeric user id); the daemon `Agent`
/// participant (a Rooms-only extension) is unsupported here.
fn contact_uid(who: &Participant) -> Result<UserId, ApiError> {
    match who {
        Participant::Contact(c) => {
            c.id.parse::<UserId>()
                .map_err(|e| ApiError::Other(format!("invalid discord user id {}: {e}", c.id)))
        }
        Participant::Agent { .. } => Err(ApiError::Unsupported(
            "discord membership targets a Discord user (Participant::Contact), not an agent".into(),
        )),
    }
}

#[async_trait]
impl TransportAdapter for DiscordAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Discord".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                presence: false,
                room_enumeration: true,
                file_transfer: false,
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
impl MessagingProtocol for DiscordAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }

    fn contacts(self: Arc<Self>) -> Option<Arc<dyn SupportsContacts>> {
        Some(self)
    }
}

#[async_trait]
impl SupportsConversations for DiscordAdapter {
    fn supported(&self) -> ConversationOps {
        // Discord's honest subset: send + channel-edit (topic/title). Channel *creation* needs a guild
        // + type the abstract `create` shape doesn't carry; a bot does not "join"/"leave" a channel by
        // name (it is added to a guild via an OAuth invite); channel destroy is deliberately off.
        ConversationOps {
            create: false,
            join_channel: false,
            leave: false,
            delete: false,
            send: true,
            set_topic: true,
            set_title: true,
            set_description: false,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        let Ok(http) = self.http_for(&transport).await else {
            return Vec::new();
        };
        let guilds = match http.get_guilds(None, Some(GUILD_PAGE_LIMIT)).await {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!(error = %e, "discord: list guilds failed");
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        for guild in guilds {
            match http.get_channels(guild.id).await {
                Ok(channels) => {
                    for c in channels {
                        if is_text_conversation(c.kind) {
                            out.push(guild_channel_to_info(&transport, &c));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(guild = %guild.id.get(), error = %e, "discord: list channels failed")
                }
            }
        }
        out
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        let http = self.http_for(&transport).await.ok()?;
        let channel_id = Self::channel_id(&conv).ok()?;
        let channel = http.get_channel(channel_id).await.ok()?;
        Some(channel_to_info(&transport, &channel))
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
        } = args;
        // The Discord account is always the sender; `from` attribution is not forwarded onto the wire
        // (serenity posts as the bound account). The outbound projector posts the same way.
        let http = self.http_for(&transport).await?;
        let channel_id = Self::channel_id(&conv)?;
        channel_id
            .say(&http, message.text)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("discord send: {e}")))
    }

    async fn set_topic(
        &self,
        transport: TransportId,
        conv: String,
        topic: Option<String>,
    ) -> Result<(), ApiError> {
        let http = self.http_for(&transport).await?;
        let channel_id = Self::channel_id(&conv)?;
        let builder = EditChannel::new().topic(topic.unwrap_or_default());
        channel_id
            .edit(&http, builder)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("discord set_topic: {e}")))
    }

    async fn set_title(
        &self,
        transport: TransportId,
        conv: String,
        title: Option<String>,
    ) -> Result<(), ApiError> {
        let http = self.http_for(&transport).await?;
        let channel_id = Self::channel_id(&conv)?;
        let builder = EditChannel::new().name(title.unwrap_or_default());
        channel_id
            .edit(&http, builder)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("discord set_title: {e}")))
    }
}

#[async_trait]
impl SupportsMembership for DiscordAdapter {
    fn supported(&self) -> MembershipOps {
        // Native guild membership admin: kick (`remove`) + ban. `invite` is off — a bot cannot
        // force-add an arbitrary user to a guild (that needs the user's own OAuth2 `guilds.join`).
        // `set_role` is off — the abstract `MemberRole` has no faithful mapping to a per-guild Discord
        // role id (role assignment needs a concrete role id, which the wire `set_role` shape lacks).
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
            reason,
        } = args;
        let user = contact_uid(&who)?;
        let http = self.http_for(&transport).await?;
        let guild = self.guild_for(&transport, &conv).await?;
        match reason {
            Some(r) => guild.kick_with_reason(&http, user, &r).await,
            None => guild.kick(&http, user).await,
        }
        .map_err(|e| ApiError::Other(format!("discord remove (kick): {e}")))
    }

    async fn ban(&self, args: MemberBanArgs) -> Result<(), ApiError> {
        let MemberBanArgs {
            transport,
            conv,
            who,
            reason,
        } = args;
        let user = contact_uid(&who)?;
        let http = self.http_for(&transport).await?;
        let guild = self.guild_for(&transport, &conv).await?;
        // `dmd = 0`: do not retroactively delete the banned user's recent messages.
        match reason {
            Some(r) => guild.ban_with_reason(&http, user, 0, &r).await,
            None => guild.ban(&http, user, 0).await,
        }
        .map_err(|e| ApiError::Other(format!("discord ban: {e}")))
    }
}

#[async_trait]
impl SupportsContacts for DiscordAdapter {
    fn supported(&self) -> ContactsOps {
        // Discord exposes a public user fetch (`GET /users/{id}`); it has no native per-contact alias
        // or bot-facing action menu, so those stay off.
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
        let http = self.http_for(&transport).await?;
        let user_id = contact
            .id
            .parse::<UserId>()
            .map_err(|e| ApiError::Other(format!("invalid discord user id {}: {e}", contact.id)))?;
        let user = http
            .get_user(user_id)
            .await
            .map_err(|e| ApiError::Other(format!("discord get_profile: {e}")))?;
        // Touch the mapping projection so a caller-supplied contact and the fetched one stay aligned.
        let _ = contact_from_user(&user);
        Ok(profile_text(&user))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use daemon_host::ProvisionedAccount;

    /// A no-op provisioning seam: the `supported()` honesty test resolves nothing from provisioning.
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

    #[test]
    fn supported_reports_the_honest_discord_subset() {
        let adapter = DiscordAdapter::new(Arc::new(MockProvisioning), DiscordConfig::default());

        let conv = SupportsConversations::supported(&*adapter);
        assert!(conv.send && conv.set_topic && conv.set_title);
        assert!(!conv.create && !conv.join_channel && !conv.leave && !conv.delete);
        assert!(!conv.set_description);

        let mem = SupportsMembership::supported(&*adapter);
        assert!(mem.remove && mem.ban);
        assert!(!mem.invite && !mem.set_role);

        let contacts = SupportsContacts::supported(&*adapter);
        assert!(contacts.get_profile && !contacts.action_menu && !contacts.set_alias);
    }

    #[test]
    fn membership_rejects_agent_participants() {
        use daemon_common::ProfileRef;
        let agent = Participant::Agent {
            profile: ProfileRef::new("alpha"),
            member: "@agent".to_string(),
        };
        let err = contact_uid(&agent).expect_err("an agent is not a Discord membership target");
        assert!(matches!(err, ApiError::Unsupported(_)));

        let contact = Participant::Contact(ContactInfo {
            id: "1234".to_string(),
            ..ContactInfo::default()
        });
        assert_eq!(contact_uid(&contact).unwrap().get(), 1234);
    }
}
