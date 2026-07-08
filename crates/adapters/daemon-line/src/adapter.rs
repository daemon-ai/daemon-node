// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `LineAdapter` — the LINE transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! A sibling of the Matrix adapter, proving the interface generalizes to a **bot-only, webhook-push**
//! protocol with a *narrower, honest* capability set. The feature-trait method bodies run real LINE
//! Messaging API calls against the live bot clients [`serve`](crate::serve) brings up. Because the
//! trait methods only get `&self`, the adapter holds a [`LiveClients`] registry `serve` populates at
//! bring-up and the methods read to resolve the per-account client.
//!
//! ## Honest capability surface
//!
//! LINE bots are deliberately restricted by the platform, so this adapter reports **no**
//! [`SupportsMembership`](daemon_api::SupportsMembership) at all: a bot cannot invite, kick, ban, or
//! set roles. [`SupportsConversations`] exposes only what a bot can actually do — `send` (push) and
//! `leave` (a group/room; a bot cannot leave a 1:1). There is no conversation `create`/`join`, no
//! topic/title/description (LINE has no such concepts for a bot), and no live room enumeration
//! (`list`/`get` return empty). [`SupportsContacts::get_profile`] is supported (the profile endpoint).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ConnectionState,
    ContactInfo, ContactsOps, ConvSendArgs, ConversationOps, MessagingProtocol, NodeApi,
    PresenceState, SupportsContacts, SupportsConversations, TransportAdapter,
    TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use line_bot_sdk_rust::line_messaging_api::apis::MessagingApiApi;
use line_bot_sdk_rust::line_messaging_api::models::{Message, PushMessageRequest, TextMessage};

use crate::account::LineAccount;
use crate::mapping::{classify_target, profile_lines, TargetKind};
use crate::{serve, LineConfig, LiveClients, FAMILY};

/// The LINE transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the multi-account bring-up, plus the [`LiveClients`]
/// registry the verb bodies resolve their per-account bot client from.
pub struct LineAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: LineConfig,
    clients: LiveClients,
}

impl LineAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved LINE `cfg`. The live
    /// client registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: LineConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            clients: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live bot `account` under its instance-qualified `transport` — the same registration
    /// [`serve`](TransportAdapter::serve) performs at bring-up. Public so vertical tests can stage a
    /// client the way bring-up would.
    pub async fn register_live_client(&self, transport: TransportId, account: LineAccount) {
        self.clients.write().await.insert(transport, account);
    }

    /// Resolve the live bot account for an instance-qualified `transport`. `Unsupported` when the
    /// account is not (yet) connected (before `serve` brought it up, or no stored credential).
    async fn client_for(&self, transport: &TransportId) -> Result<LineAccount, ApiError> {
        self.clients
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "line account {} is not connected",
                    transport.as_str()
                ))
            })
    }
}

#[async_trait]
impl TransportAdapter for LineAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "LINE".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                // LINE bots do not receive peer presence, cannot enumerate their conversations, and
                // file transfer is not wired in v1 — report these honestly as off.
                presence: false,
                room_enumeration: false,
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
impl MessagingProtocol for LineAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn contacts(self: Arc<Self>) -> Option<Arc<dyn SupportsContacts>> {
        Some(self)
    }

    // membership()/roster()/directory()/file_transfer() intentionally use the trait defaults (None):
    // a LINE bot cannot administer membership, has no server-side roster, cannot search a directory,
    // and file transfer is not wired in v1.
}

#[async_trait]
impl SupportsConversations for LineAdapter {
    fn supported(&self) -> ConversationOps {
        // The honest LINE-bot subset: a bot can push a message (`send`) and leave a group/room
        // (`leave`). It cannot create/join conversations, set topic/title/description, or delete —
        // none of those are bot-available operations on LINE.
        ConversationOps {
            create: false,
            join_channel: false,
            leave: true,
            delete: false,
            send: true,
            set_topic: false,
            set_title: false,
            set_description: false,
        }
    }

    // list()/get() use the trait defaults (empty/None): LINE gives a bot no conversation enumeration.

    async fn leave(&self, transport: TransportId, conv: String) -> Result<(), ApiError> {
        let account = self.client_for(&transport).await?;
        match classify_target(&conv) {
            TargetKind::Group => account
                .line
                .messaging_api_client
                .leave_group(&conv)
                .await
                .map_err(|e| ApiError::Other(format!("line leave_group: {e:?}"))),
            TargetKind::Room => account
                .line
                .messaging_api_client
                .leave_room(&conv)
                .await
                .map_err(|e| ApiError::Other(format!("line leave_room: {e:?}"))),
            TargetKind::User | TargetKind::Unknown => Err(ApiError::Unsupported(
                "line: a bot can only leave a group or room, not a 1:1 conversation".into(),
            )),
        }
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
        } = args;
        let account = self.client_for(&transport).await?;
        let request = PushMessageRequest {
            to: conv,
            messages: vec![Message::TextMessage(TextMessage::new(message.text))],
            notification_disabled: Some(false),
            custom_aggregation_units: None,
        };
        account
            .line
            .messaging_api_client
            .push_message(request, None)
            .await
            .map(|_| ())
            .map_err(|e| ApiError::Other(format!("line send: {e:?}")))
    }
}

#[async_trait]
impl SupportsContacts for LineAdapter {
    fn supported(&self) -> ContactsOps {
        // LINE exposes a bot-visible user profile fetch; it has no per-contact alias or action menu.
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
        let account = self.client_for(&transport).await?;
        let profile = account
            .line
            .messaging_api_client
            .get_profile(&contact.id)
            .await
            .map_err(|e| ApiError::Other(format!("line get_profile: {e:?}")))?;
        Ok(profile_lines(&profile))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use daemon_host::ProvisionedAccount;

    /// A no-op provisioning seam: the capability tests never resolve a live client.
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

    fn adapter() -> Arc<LineAdapter> {
        LineAdapter::new(Arc::new(MockProvisioning), LineConfig::default())
    }

    #[test]
    fn info_reports_line_family_and_honest_capabilities() {
        let adapter = adapter();
        let info = TransportAdapter::info(&*adapter);
        assert_eq!(info.family, "line");
        assert_eq!(info.display_name, "LINE");
        assert!(info.capabilities.rooms && info.capabilities.direct_messages);
        assert!(info.capabilities.interactive_auth);
        assert!(!info.capabilities.presence);
        assert!(!info.capabilities.room_enumeration);
        assert!(!info.capabilities.file_transfer);
    }

    #[test]
    fn conversations_report_the_bot_subset() {
        let adapter = adapter();
        let ops = SupportsConversations::supported(&*adapter);
        assert!(ops.send && ops.leave);
        assert!(!ops.create && !ops.join_channel && !ops.delete);
        assert!(!ops.set_topic && !ops.set_title && !ops.set_description);
    }

    #[test]
    fn contacts_report_profile_only() {
        let adapter = adapter();
        let ops = SupportsContacts::supported(&*adapter);
        assert!(ops.get_profile);
        assert!(!ops.action_menu && !ops.set_alias);
    }

    #[test]
    fn membership_directory_roster_are_unsupported() {
        // The honest core: a LINE bot exposes conversations + contacts, but no membership admin,
        // roster, directory, or file transfer.
        assert!(MessagingProtocol::conversations(adapter()).is_some());
        assert!(MessagingProtocol::contacts(adapter()).is_some());
        assert!(MessagingProtocol::membership(adapter()).is_none());
        assert!(MessagingProtocol::roster(adapter()).is_none());
        assert!(MessagingProtocol::directory(adapter()).is_none());
        assert!(MessagingProtocol::file_transfer(adapter()).is_none());
    }

    #[tokio::test]
    async fn verbs_error_when_account_not_connected() {
        let adapter = adapter();
        let transport = TransportId::new("line/absent");
        let err = SupportsConversations::leave(&*adapter, transport, "Cgroup".into())
            .await
            .expect_err("no live client");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }
}
