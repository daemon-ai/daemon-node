// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `WhatsappAdapter` — the WhatsApp transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! Like the Matrix adapter, the feature-trait method bodies only get `&self`, so the adapter holds a
//! [`LiveBackends`](crate::LiveBackends) registry that [`serve`](crate::serve) populates at bring-up;
//! the verb bodies resolve the per-account [`WaBackend`] from it. The `supported()` sets are honest
//! about what actually works across the two backends: outbound send (both modes) and group
//! participant invite/remove (WhatsApp Web / user mode). Conversation enumeration, topic/title, and
//! ban/role have no wired counterpart and report `false`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiError, ConnectionState,
    ConvSendArgs, ConversationInfo, ConversationOps, MemberInviteArgs, MemberRemoveArgs,
    MembershipOps, MessagingProtocol, NodeApi, Participant, PresenceState, SupportsConversations,
    SupportsMembership, TransportAdapter, TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;
use daemon_protocol::TransportId;

use crate::account::FAMILY;
use crate::backend::WaBackend;
use crate::{serve, LiveBackends, WhatsappConfig};

/// The WhatsApp transport adapter: the in-process provisioning seam + resolved config its
/// [`serve`](TransportAdapter::serve) drives, plus the live backend registry the management verb
/// bodies resolve their per-account backend from.
pub struct WhatsappAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: WhatsappConfig,
    backends: LiveBackends,
}

impl WhatsappAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved WhatsApp `cfg`. The live
    /// backend registry starts empty and is filled by [`serve`](TransportAdapter::serve).
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: WhatsappConfig) -> Arc<Self> {
        Arc::new(Self {
            provisioning,
            cfg,
            backends: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Register a live `backend` under its instance-qualified `transport` — the seam
    /// [`serve`](TransportAdapter::serve) performs at bring-up. Public so tests can stage a mock
    /// backend exactly the way bring-up would.
    pub async fn register_backend(&self, transport: TransportId, backend: Arc<dyn WaBackend>) {
        self.backends.write().await.insert(transport, backend);
    }

    /// Resolve the live backend for an instance-qualified `transport`. `Unsupported` when the account
    /// is not (yet) connected.
    async fn backend_for(&self, transport: &TransportId) -> Result<Arc<dyn WaBackend>, ApiError> {
        self.backends
            .read()
            .await
            .get(transport)
            .cloned()
            .ok_or_else(|| {
                ApiError::Unsupported(format!(
                    "whatsapp account {} is not connected",
                    transport.as_str()
                ))
            })
    }
}

/// Extract the target contact id from a membership `Participant`. WhatsApp membership targets a
/// contact (`Participant::Contact`, `id` = a JID / phone); the daemon `Agent` participant is unsupported.
fn contact_id(who: &Participant) -> Result<String, ApiError> {
    match who {
        Participant::Contact(c) => Ok(c.id.clone()),
        Participant::Agent { .. } => Err(ApiError::Unsupported(
            "whatsapp membership targets a contact (Participant::Contact), not an agent".into(),
        )),
    }
}

#[async_trait]
impl TransportAdapter for WhatsappAdapter {
    fn family(&self) -> &str {
        FAMILY
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: FAMILY.to_string(),
            display_name: "WhatsApp".to_string(),
            capabilities: AdapterCapabilities {
                rooms: true,
                direct_messages: true,
                // No presence reporting, no cheap conversation enumeration, no file transfer wired.
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
            self.backends.clone(),
        )
        .await
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait]
impl MessagingProtocol for WhatsappAdapter {
    fn conversations(self: Arc<Self>) -> Option<Arc<dyn SupportsConversations>> {
        Some(self)
    }

    fn membership(self: Arc<Self>) -> Option<Arc<dyn SupportsMembership>> {
        Some(self)
    }
}

#[async_trait]
impl SupportsConversations for WhatsappAdapter {
    fn supported(&self) -> ConversationOps {
        // Only outbound send is wired. WhatsApp has no cheap conversation enumeration via these SDKs,
        // and topic/title/create/join/leave have no wired counterpart here.
        ConversationOps {
            send: true,
            create: false,
            join_channel: false,
            leave: false,
            delete: false,
            set_topic: false,
            set_title: false,
            set_description: false,
        }
    }

    async fn list(&self, _transport: TransportId) -> Vec<ConversationInfo> {
        // No cheap enumeration surface (Cloud API can't list chats; WhatsApp Web sync is out of scope).
        Vec::new()
    }

    async fn get(&self, _transport: TransportId, _conv: String) -> Option<ConversationInfo> {
        None
    }

    async fn send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let ConvSendArgs {
            transport,
            conv,
            from: _from,
            message,
        } = args;
        // The account is always the sender; `from` attribution is not forwarded onto the wire.
        let backend = self.backend_for(&transport).await?;
        backend.send_text(&conv, &message.text).await
    }
}

#[async_trait]
impl SupportsMembership for WhatsappAdapter {
    fn supported(&self) -> MembershipOps {
        // WhatsApp Web (user mode) can add/remove group participants; the Cloud API (bot mode) cannot —
        // a bot account returns `Unsupported` at call time. Ban/role have no mapped counterpart.
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
        let who = contact_id(&who)?;
        let backend = self.backend_for(&transport).await?;
        backend.invite(&conv, &who).await
    }

    async fn remove(&self, args: MemberRemoveArgs) -> Result<(), ApiError> {
        let MemberRemoveArgs {
            transport,
            conv,
            who,
            reason: _reason,
        } = args;
        let who = contact_id(&who)?;
        let backend = self.backend_for(&transport).await?;
        backend.remove(&conv, &who).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use daemon_api::ContactInfo;
    use daemon_common::ProfileRef;
    use daemon_host::ProvisionedAccount;
    use daemon_protocol::UserMsg;

    use crate::backend::mock::MockBackend;

    /// A no-op provisioning seam: the verb tests resolve the live backend from the seeded registry.
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

    async fn adapter_with(
        transport: &TransportId,
        supports_membership: bool,
    ) -> (Arc<WhatsappAdapter>, Arc<MockBackend>) {
        let adapter = WhatsappAdapter::new(Arc::new(MockProvisioning), WhatsappConfig::default());
        let backend = Arc::new(MockBackend {
            supports_membership,
            ..MockBackend::default()
        });
        adapter
            .register_backend(transport.clone(), backend.clone())
            .await;
        (adapter, backend)
    }

    #[test]
    fn supported_reports_the_honest_whatsapp_subset() {
        let adapter = WhatsappAdapter::new(Arc::new(MockProvisioning), WhatsappConfig::default());
        let conv = SupportsConversations::supported(&*adapter);
        assert!(conv.send);
        assert!(!conv.create && !conv.join_channel && !conv.leave && !conv.delete);
        assert!(!conv.set_topic && !conv.set_title && !conv.set_description);
        let mem = SupportsMembership::supported(&*adapter);
        assert!(mem.invite && mem.remove);
        assert!(!mem.ban && !mem.set_role);
    }

    #[tokio::test]
    async fn send_dispatches_to_the_live_backend() {
        let transport = TransportId::new("whatsapp/15551234567");
        let (adapter, backend) = adapter_with(&transport, false).await;

        SupportsConversations::send(
            &*adapter,
            ConvSendArgs {
                transport,
                conv: "15559876543".to_string(),
                from: None,
                message: UserMsg::new("hello".to_string()),
            },
        )
        .await
        .expect("send dispatches to the backend");

        let sent = backend.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0], ("15559876543".to_string(), "hello".to_string()));
    }

    #[tokio::test]
    async fn send_to_unconnected_account_is_unsupported() {
        let adapter = WhatsappAdapter::new(Arc::new(MockProvisioning), WhatsappConfig::default());
        let err = SupportsConversations::send(
            &*adapter,
            ConvSendArgs {
                transport: TransportId::new("whatsapp/nobody"),
                conv: "1".to_string(),
                from: None,
                message: UserMsg::new("x".to_string()),
            },
        )
        .await
        .expect_err("no live backend");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn membership_invite_dispatches_and_rejects_agents() {
        let transport = TransportId::new("whatsapp/me");
        let (adapter, backend) = adapter_with(&transport, true).await;

        SupportsMembership::invite(
            &*adapter,
            MemberInviteArgs {
                transport: transport.clone(),
                conv: "120@g.us".to_string(),
                who: Participant::Contact(ContactInfo {
                    id: "15551112222@s.whatsapp.net".to_string(),
                    ..ContactInfo::default()
                }),
                message: None,
            },
        )
        .await
        .expect("invite dispatches to the backend");
        assert_eq!(backend.sent.lock().unwrap().len(), 1);

        // An agent participant is not a WhatsApp membership target.
        let err = SupportsMembership::invite(
            &*adapter,
            MemberInviteArgs {
                transport,
                conv: "120@g.us".to_string(),
                who: Participant::Agent {
                    profile: ProfileRef::new("alpha"),
                    member: "@agent".to_string(),
                },
                message: None,
            },
        )
        .await
        .expect_err("an agent is not a membership target");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }

    #[tokio::test]
    async fn membership_on_bot_backend_is_unsupported() {
        let transport = TransportId::new("whatsapp/bot");
        let (adapter, _backend) = adapter_with(&transport, false).await;
        let err = SupportsMembership::invite(
            &*adapter,
            MemberInviteArgs {
                transport,
                conv: "120@g.us".to_string(),
                who: Participant::Contact(ContactInfo {
                    id: "15551112222".to_string(),
                    ..ContactInfo::default()
                }),
                message: None,
            },
        )
        .await
        .expect_err("a bot backend cannot administer group membership");
        assert!(matches!(err, ApiError::Unsupported(_)));
    }
}
