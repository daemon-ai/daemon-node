// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `WeChatAdapter` — the WeChat transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! Structurally the `daemon-matrix` shape: the feature-trait method bodies only get `&self`, so the
//! adapter holds a [`LiveClients`] registry that [`serve`](crate::serve) populates at bring-up and the
//! methods read to resolve the per-account [`LiveAccount`]. WeChat's iLink bot is a **DM-only** single
//! account (§ crate header): it exposes exactly [`SupportsConversations`] with `send` — there is no
//! group/room administration, no roster, no directory, and no contact-profile API in the iLink bot
//! surface, so those feature traits are (honestly) left unimplemented. `list`/`get` return empty:
//! iLink has no "enumerate my conversations" call, so the adapter cannot honestly project one.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ConnectionState,
    ConvSendArgs, ConversationOps, MessagingProtocol, NodeApi, PresenceState,
    SupportsConversations, TransportAdapter, TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;
use wechatbot::protocol::build_text_message;

use crate::account::LiveAccount;
use crate::{serve, LiveClients, WeChatConfig, FAMILY};

/// The WeChat transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the multi-account bring-up, plus the [`LiveClients`]
/// registry the `send` verb resolves its per-account client from.
pub struct WeChatAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: WeChatConfig,
    accounts: LiveClients,
}

impl WeChatAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved WeChat `cfg`. The live
    /// account registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: WeChatConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            accounts: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live account under its instance-qualified `transport` — the same registration
    /// [`serve`](TransportAdapter::serve) performs at bring-up. Public so tests can stage an account
    /// exactly the way bring-up would.
    pub async fn register_live_account(&self, transport: TransportId, account: Arc<LiveAccount>) {
        self.accounts.write().await.insert(transport, account);
    }

    /// Resolve the live [`LiveAccount`] for an instance-qualified `transport`. `Unsupported` when the
    /// account is not (yet) connected (before `serve` brought it up, or it has no stored session).
    async fn account_for(&self, transport: &TransportId) -> Result<Arc<LiveAccount>, ApiError> {
        self.accounts
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "wechat account {} is not connected",
                    transport.as_str()
                ))
            })
    }
}

#[async_trait]
impl TransportAdapter for WeChatAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "WeChat".to_string(),
            capabilities: AdapterCapabilities {
                rooms: false,
                direct_messages: true,
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
                    // Wire v35: enabled/label are node-overlaid from the store; report inert default.
                    enabled: true,
                    label: None,
                }
            })
            .collect()
    }

    async fn serve(self: Arc<Self>, api: Arc<dyn NodeApi>) {
        serve(
            api,
            self.provisioning.clone(),
            self.cfg.clone(),
            self.accounts.clone(),
        )
        .await
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for WeChatAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }
}

#[async_trait]
impl SupportsConversations for WeChatAdapter {
    fn supported(&self) -> ConversationOps {
        // WeChat iLink's bot surface is DM send only: no create/join/leave/delete, no topic/title/
        // description (a DM has none), no membership. Honest minimal set.
        ConversationOps {
            send: true,
            ..ConversationOps::default()
        }
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
        } = args;
        // The bound WeChat account is always the sender; `from` attribution is not forwarded onto the
        // wire (the outbound projector posts the same way). The conversation id is the peer user id.
        let account = self.account_for(&transport).await?;
        let context_token = account.context_for(&conv).await.ok_or_else(|| {
            ApiError::Other(format!(
                "wechat send: no reply context token for peer {conv} yet (needs a prior inbound \
                 message)"
            ))
        })?;
        let msg = build_text_message(&conv, &context_token, &message.text);
        account
            .client
            .send_message(&account.session.base_url, &account.session.token, &msg)
            .await
            .map_err(|e| ApiError::Other(format!("wechat send: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use daemon_host::ProvisionedAccount;
    use daemon_protocol::UserMsg;

    use crate::mapping::StoredSession;

    /// A no-op provisioning seam: the `send` guard test resolves the live account from the seeded
    /// registry, never from provisioning, so empty answers suffice.
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

    fn adapter() -> Arc<WeChatAdapter> {
        WeChatAdapter::new(Arc::new(MockProvisioning), WeChatConfig::default())
    }

    fn session() -> StoredSession {
        StoredSession {
            token: "tok".to_string(),
            base_url: "https://example.weixin.qq.com".to_string(),
            account_id: "bot".to_string(),
            user_id: "bot-self".to_string(),
        }
    }

    #[test]
    fn supported_reports_send_only() {
        let a = adapter();
        let ops = SupportsConversations::supported(&*a);
        assert!(ops.send);
        assert!(!ops.create && !ops.join_channel && !ops.leave && !ops.delete);
        assert!(!ops.set_topic && !ops.set_title && !ops.set_description);
    }

    #[test]
    fn info_advertises_dm_only_with_interactive_auth() {
        let a = adapter();
        let info = a.info();
        assert_eq!(info.family, "wechat");
        assert!(info.capabilities.direct_messages);
        assert!(info.capabilities.interactive_auth);
        assert!(!info.capabilities.rooms);
        assert!(!info.capabilities.room_enumeration);
        assert!(!info.capabilities.file_transfer);
    }

    #[tokio::test]
    async fn send_without_a_known_peer_context_errors_before_the_wire() {
        let a = adapter();
        let transport = TransportId::new("wechat/bot-self");
        a.register_live_account(transport.clone(), LiveAccount::new(session(), None))
            .await;

        // No inbound message has been seen for this peer, so there is no context token to reply with:
        // the verb must fail fast (never touching the network).
        let err = SupportsConversations::send(
            &*a,
            ConvSendArgs {
                transport,
                conv: "peer-unknown".to_string(),
                from: None,
                message: UserMsg::new("hi".to_string()),
            },
        )
        .await
        .expect_err("no context token => error before any send");
        assert!(matches!(err, ApiError::Other(_)));
    }

    #[tokio::test]
    async fn send_to_a_disconnected_account_is_unsupported() {
        let a = adapter();
        let err = SupportsConversations::send(
            &*a,
            ConvSendArgs {
                transport: TransportId::new("wechat/not-brought-up"),
                conv: "peer".to_string(),
                from: None,
                message: UserMsg::new("hi".to_string()),
            },
        )
        .await
        .expect_err("unknown account => unsupported");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }
}
