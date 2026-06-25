//! `MatrixAdapter` — the Matrix transport presented as a [`MessagingProtocol`](daemon_api::MessagingProtocol).
//!
//! The second reference implementor of the messaging-adapter interface (daemon-messaging-adapter-spec.md
//! §10.2). Its purpose is to prove the interface generalizes: a *different* `supported()` set than the
//! Rooms adapter, on the same traits, with **no host changes**. It wraps the existing
//! [`serve`](crate::serve) bring-up (which additionally needs the in-process
//! [`AccountProvisioning`] seam the `TransportAdapter::serve(self, api)` signature does not carry, so
//! the adapter holds it) and enumerates bound accounts as transport instances.
//!
//! Real conversation/membership *execution* against `matrix-sdk` (send / set_topic / m.room.member
//! invite·leave·ban) is wired against the live client `serve` owns and is deferred (spec §12.1); the
//! feature methods inherit the trait defaults for now, but `supported()` already reports Matrix's
//! intended subset so the capability surface is correct end to end.

use std::sync::Arc;

use async_trait::async_trait;

use daemon_api::{
    AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ConnectionState, ConversationOps,
    MembershipOps, MessagingProtocol, NodeApi, PresenceState, SupportsConversations,
    SupportsMembership, TransportAdapter, TransportInstanceInfo,
};
use daemon_host::AccountProvisioning;

use crate::{serve, MatrixConfig, FAMILY};

/// The Matrix transport adapter: holds the in-process provisioning seam + resolved config so its
/// [`serve`](TransportAdapter::serve) can call the existing multi-account bring-up.
pub struct MatrixAdapter {
    provisioning: Arc<dyn AccountProvisioning>,
    cfg: MatrixConfig,
}

impl MatrixAdapter {
    /// Construct the adapter over the host `provisioning` seam and resolved Matrix `cfg`.
    pub fn new(provisioning: Arc<dyn AccountProvisioning>, cfg: MatrixConfig) -> Arc<Self> {
        Arc::new(Self { provisioning, cfg })
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
                let connection = if self.provisioning.account_credential(&acct.credential_ref).is_some() {
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
        serve(api, self.provisioning.clone(), self.cfg.clone()).await
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
}

#[async_trait]
impl SupportsConversations for MatrixAdapter {
    fn supported(&self) -> ConversationOps {
        // The subset Matrix exposes (vs. Rooms' full set): send + set_topic exist in the transport
        // today; create/join/leave/title/description are deferred (daemon-messaging-adapter-spec §10.2).
        ConversationOps {
            create: false,
            join_channel: false,
            leave: false,
            delete: false,
            send: true,
            set_topic: true,
            set_title: false,
            set_description: false,
        }
    }
}

#[async_trait]
impl SupportsMembership for MatrixAdapter {
    fn supported(&self) -> MembershipOps {
        // Matrix membership administration is richer than Rooms': invite/remove/ban map to
        // `m.room.member` invite/leave/ban; `set_role` (power levels) is deferred.
        MembershipOps {
            invite: true,
            remove: true,
            ban: true,
            set_role: false,
        }
    }
}
