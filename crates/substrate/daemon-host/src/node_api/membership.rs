// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The wire-v30 transport lifecycle + membership seams: the node's [`LifecycleSink`] impl (the one
//! seam an events-IO adapter reports disconnect causes + conversation/membership changes through),
//! plus the node-owned routing reconciliation on a self-removal. Kept out of the sibling-owned
//! `roster.rs` so the two streams merge cleanly.  [waveA:node-v30]

use super::*;
use daemon_api::{
    ConvChange, DisconnectReason, LifecycleSink, MembershipChange, NodeEvent, PresenceState,
};

/// Whether a disconnect reason is fatal (stop retrying; offer re-auth). The node — never a thin
/// client — owns this mapping. Credential/settings/certificate failures will only fail again on a
/// blind retry, so they are terminal; transport-level and server-initiated drops are transient.
fn reason_is_fatal(reason: DisconnectReason) -> bool {
    matches!(
        reason,
        DisconnectReason::AuthenticationFailed
            | DisconnectReason::InvalidSettings
            | DisconnectReason::CertificateError
    )
}

impl NodeApiImpl {
    /// The lifecycle sink handed to events-IO adapters by the assembling binary (the node itself).
    /// Keeps `NodeApi` clean: adapters report coarse lifecycle signals here rather than through a
    /// widened control surface.
    pub fn lifecycle_sink(self: &Arc<Self>) -> Arc<dyn LifecycleSink> {
        self.clone()
    }

    /// Emit a `TransportChanged` carrying a disconnect reason + node-decided `fatal` (item 2). Also
    /// records the fatal flag so the reconnect supervisor can short-circuit its backoff loop.
    pub(crate) fn emit_transport_disconnected(
        &self,
        transport: TransportId,
        reason: DisconnectReason,
        message: Option<String>,
    ) {
        let fatal = reason_is_fatal(reason);
        self.disconnect_fatal.insert(transport.clone(), fatal);
        if let Some(feed) = self.node_feed() {
            let connection = if fatal {
                daemon_api::ConnectionState::Error
            } else {
                daemon_api::ConnectionState::Disconnecting
            };
            feed.emit(NodeEvent::TransportChanged {
                transport,
                connection,
                presence: PresenceState::Offline,
                reason: Some(reason),
                message,
                fatal,
            });
        }
    }

    /// Emit a `ContactsChanged` for `transport` (wire v34) after a successful roster mutation
    /// (`roster_add`/`roster_update`/`roster_remove`) so clients refetch `RosterList` without
    /// polling. A payload-free-per-transport invalidation pointer, mirroring `conversations_changed`.
    pub(crate) fn emit_contacts_changed(&self, transport: TransportId) {
        if let Some(feed) = self.node_feed() {
            feed.emit(NodeEvent::ContactsChanged { transport });
        }
    }

    /// Reconcile the node's own routing on a self-removal (item 3): drop the now-dangling
    /// `ChatRoute` pin for the conversation's origin (matches libpurple teardown — a re-join re-pins
    /// on next inbound), then reload the live routing table. Called BEFORE the invalidation event is
    /// emitted so a client acting on the event sees a consistent routing registry.
    async fn reconcile_self_removal(&self, transport: &TransportId, conv: &str) {
        let origin = Origin::new(
            transport.clone(),
            OriginScope::Group {
                chat: conv.to_string(),
                thread: None,
            },
        );
        let key = crate::routing::origin_pin_key(&origin);
        if self.store.routing_get(&key).await.is_some() {
            let _ = self.store.routing_remove(&key).await;
            self.load_routing_pins().await;
        }
    }
}

#[async_trait]
impl LifecycleSink for NodeApiImpl {
    async fn transport_disconnected(
        &self,
        transport: TransportId,
        reason: DisconnectReason,
        message: Option<String>,
    ) {
        self.emit_transport_disconnected(transport, reason, message);
    }

    async fn conversations_changed(
        &self,
        transport: TransportId,
        conv: String,
        change: ConvChange,
    ) {
        if let Some(feed) = self.node_feed() {
            feed.emit(NodeEvent::ConversationsChanged {
                transport,
                conv,
                change,
            });
        }
    }

    async fn membership_changed(
        &self,
        transport: TransportId,
        conv: String,
        member: String,
        change: MembershipChange,
        actor: Option<String>,
        reason: Option<String>,
        is_self: bool,
    ) {
        // Node-owned consequence: a self departure drops the dangling routing pin BEFORE the event.
        if is_self
            && matches!(
                change,
                MembershipChange::Left | MembershipChange::Kicked | MembershipChange::Banned
            )
        {
            self.reconcile_self_removal(&transport, &conv).await;
        }
        if let Some(feed) = self.node_feed() {
            feed.emit(NodeEvent::MembershipChanged {
                transport,
                conv,
                member,
                change,
                actor,
                reason,
                is_self,
            });
        }
    }
}
