// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `SlackAdapter` — the Slack transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! Modelled on `daemon-matrix`: the feature-trait method bodies only get `&self`, so the adapter
//! holds a [`LiveConns`] registry that [`serve`](crate::serve) populates at bring-up (one
//! [`SlackConn`] per account — a `slack-morphism` bot conn or a `slacko` stealth conn) and the
//! methods read to resolve the per-account conn. A `SlackConn` is `Send + Sync + async`, so a verb
//! body calls it directly (no command channel).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ConnectionState,
    ContactInfo, ConvSendArgs, ConversationInfo, ConversationOps, MemberInviteArgs,
    MemberRemoveArgs, MembershipOps, MessagingProtocol, NodeApi, Participant, PresenceState,
    SupportsConversations, SupportsDirectory, SupportsMembership, TransportAdapter,
    TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use crate::account::FAMILY;
use crate::conn::SlackConn;
use crate::mapping::{channel_to_contact, channel_to_info};
use crate::{serve, LiveConns, SlackConfig};

/// The Slack transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the multi-account bring-up, plus the [`LiveConns`]
/// registry the verb bodies resolve their per-account conn from.
pub struct SlackAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: SlackConfig,
    conns: LiveConns,
}

impl SlackAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved Slack `cfg`. The live
    /// conn registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: SlackConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            conns: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live conn under its instance-qualified `transport` — the same registration
    /// [`serve`](TransportAdapter::serve) performs at bring-up. Public so tests can stage a mock conn.
    pub async fn register_live_conn(&self, transport: TransportId, conn: Arc<dyn SlackConn>) {
        self.conns.write().await.insert(transport, conn);
    }

    /// Resolve the live conn for an instance-qualified `transport`. `Unsupported` when the account is
    /// not (yet) connected (before `serve` brought it up, or it has no stored credential).
    async fn conn_for(&self, transport: &TransportId) -> Result<Arc<dyn SlackConn>, ApiError> {
        self.conns
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "slack account {} is not connected",
                    transport.as_str()
                ))
            })
    }
}

/// Extract the target Slack user id from a membership `Participant`. Slack membership targets a Slack
/// user (`Participant::Contact`, `id` = `U…`); the daemon `Agent` participant is unsupported here.
fn contact_user(who: &Participant) -> Result<String, ApiError> {
    match who {
        Participant::Contact(c) => Ok(c.id.clone()),
        Participant::Agent { .. } => Err(ApiError::Unsupported(
            "slack membership targets a Slack user (Participant::Contact), not an agent".into(),
        )),
    }
}

#[async_trait]
impl TransportAdapter for SlackAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "Slack".to_string(),
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
            self.conns.clone(),
        )
        .await
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for SlackAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }

    fn directory(self: Arc<Self>) -> Option<Arc<dyn SupportsDirectory>> {
        Some(self)
    }
}

#[async_trait]
impl SupportsConversations for SlackAdapter {
    fn supported(&self) -> ConversationOps {
        // Slack v1 subset: `send` (chat.postMessage) plus the always-available `list`/`get` reads
        // (conversations.list). create/join/leave/delete/set_* are deferred, so they stay off.
        ConversationOps {
            create: false,
            join_channel: false,
            leave: false,
            delete: false,
            send: true,
            set_topic: false,
            set_title: false,
            set_description: false,
        }
    }

    async fn list(&self, transport: TransportId) -> Vec<ConversationInfo> {
        let Ok(conn) = self.conn_for(&transport).await else {
            return Vec::new();
        };
        match conn.list_channels().await {
            Ok(channels) => channels
                .iter()
                .map(|c| channel_to_info(&transport, c))
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "slack: conversations.list failed");
                Vec::new()
            }
        }
    }

    async fn get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        let conn = self.conn_for(&transport).await.ok()?;
        let channels = conn.list_channels().await.ok()?;
        channels
            .iter()
            .find(|c| c.id == conv)
            .map(|c| channel_to_info(&transport, c))
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
        } = args;
        // The Slack account is always the sender; `from` attribution is not forwarded onto the wire
        // (the SDK posts as the bound account). The outbound projector posts the same way.
        let conn = self.conn_for(&transport).await?;
        conn.post_message(&conv, &message.text).await
    }
}

#[async_trait]
impl SupportsMembership for SlackAdapter {
    fn supported(&self) -> MembershipOps {
        // invite/remove map to conversations.invite/kick; Slack has no per-conversation ban or role
        // administration via these methods, so ban/set_role stay off.
        MembershipOps {
            invite: true,
            remove: true,
            ban: false,
            set_role: false,
        }
    }

    async fn invite(&self, args: MemberInviteArgs) -> Result<(), ApiError> {
        let MemberInviteArgs {
            transport,
            conv,
            who,
            message: _message,
        } = args;
        let user = contact_user(&who)?;
        let conn = self.conn_for(&transport).await?;
        conn.invite(&conv, &user).await
    }

    async fn remove(&self, args: MemberRemoveArgs) -> Result<(), ApiError> {
        let MemberRemoveArgs {
            transport,
            conv,
            who,
            reason: _reason,
        } = args;
        let user = contact_user(&who)?;
        let conn = self.conn_for(&transport).await?;
        conn.kick(&conv, &user).await
    }
}

#[async_trait]
impl SupportsDirectory for SlackAdapter {
    fn supported(&self) -> bool {
        true
    }

    async fn search_contacts(
        &self,
        transport: TransportId,
        query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        // The channel/room directory (libpurple roomlist successor) over conversations.list: a match
        // is a substring hit on the channel name or id.
        let conn = self.conn_for(&transport).await?;
        let channels = conn.list_channels().await?;
        let needle = query.unwrap_or_default().to_lowercase();
        Ok(channels
            .iter()
            .filter(|c| {
                needle.is_empty()
                    || c.id.to_lowercase().contains(&needle)
                    || c.name
                        .as_deref()
                        .map(|n| n.to_lowercase().contains(&needle))
                        .unwrap_or(false)
            })
            .map(channel_to_contact)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    use daemon_common::ProfileRef;
    use daemon_host::ProvisionedAccount;
    use daemon_protocol::UserMsg;

    use crate::conn::ChannelSummary;

    /// A no-op provisioning seam: the verb tests resolve the live conn from the seeded registry.
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

    /// A recording mock conn: captures the last `post_message`, serves a fixed channel list.
    #[derive(Default)]
    struct MockConn {
        sent: Mutex<Vec<(String, String)>>,
    }

    #[async_trait]
    impl SlackConn for MockConn {
        async fn post_message(&self, channel: &str, text: &str) -> Result<(), ApiError> {
            self.sent
                .lock()
                .unwrap()
                .push((channel.to_string(), text.to_string()));
            Ok(())
        }
        async fn list_channels(&self) -> Result<Vec<ChannelSummary>, ApiError> {
            Ok(vec![ChannelSummary {
                id: "C123".into(),
                name: Some("secops".into()),
                topic: None,
                is_im: false,
                is_private: false,
            }])
        }
        async fn invite(&self, _channel: &str, _user: &str) -> Result<(), ApiError> {
            Ok(())
        }
        async fn kick(&self, _channel: &str, _user: &str) -> Result<(), ApiError> {
            Ok(())
        }
    }

    async fn adapter_with(transport: &TransportId, conn: Arc<MockConn>) -> Arc<SlackAdapter> {
        let adapter = SlackAdapter::new(Arc::new(MockProvisioning), SlackConfig::default());
        adapter.register_live_conn(transport.clone(), conn).await;
        adapter
    }

    #[test]
    fn supported_reports_the_honest_slack_subset() {
        let adapter = SlackAdapter::new(Arc::new(MockProvisioning), SlackConfig::default());
        let conv = SupportsConversations::supported(&*adapter);
        assert!(conv.send);
        assert!(!conv.create && !conv.join_channel && !conv.leave && !conv.delete);
        assert!(!conv.set_topic && !conv.set_title && !conv.set_description);
        let mem = SupportsMembership::supported(&*adapter);
        assert!(mem.invite && mem.remove);
        assert!(!mem.ban && !mem.set_role);
        assert!(SupportsDirectory::supported(&*adapter));
    }

    #[tokio::test]
    async fn list_maps_channels_and_send_posts() {
        let transport = TransportId::new("slack/T1");
        let conn = Arc::new(MockConn::default());
        let adapter = adapter_with(&transport, conn.clone()).await;

        let convs = SupportsConversations::list(&*adapter, transport.clone()).await;
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].id, "C123");
        assert_eq!(convs[0].title.as_deref(), Some("secops"));

        SupportsConversations::send(
            &*adapter,
            ConvSendArgs {
                transport: transport.clone(),
                conv: "C123".to_string(),
                from: None,
                message: UserMsg::new("hello".to_string()),
            },
        )
        .await
        .expect("send succeeds against the mock conn");
        assert_eq!(
            conn.sent.lock().unwrap().as_slice(),
            &[("C123".to_string(), "hello".to_string())]
        );
    }

    #[tokio::test]
    async fn directory_search_filters_channels() {
        let transport = TransportId::new("slack/T1");
        let adapter = adapter_with(&transport, Arc::new(MockConn::default())).await;
        let hits = SupportsDirectory::search_contacts(&*adapter, transport, Some("sec".into()))
            .await
            .unwrap();
        assert!(hits.iter().any(|c| c.id == "C123"));
    }

    #[tokio::test]
    async fn membership_rejects_agent_participants() {
        let transport = TransportId::new("slack/T1");
        let adapter = adapter_with(&transport, Arc::new(MockConn::default())).await;
        let agent = Participant::Agent {
            profile: ProfileRef::new("alpha"),
            member: "agent".to_string(),
        };
        let err = SupportsMembership::invite(
            &*adapter,
            MemberInviteArgs {
                transport,
                conv: "C123".to_string(),
                who: agent,
                message: None,
            },
        )
        .await
        .expect_err("an agent is not a Slack membership target");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }
}
