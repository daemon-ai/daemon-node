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
            // rung 1: bump the per-transport contact-roster rev (exactly once per emit) and stamp it
            // so `RosterList`'s echoed rev and this pointer agree on the reflected generation.
            let rev = feed.note_contacts_change(&transport);
            feed.emit(NodeEvent::ContactsChanged { transport, rev });
        }
    }

    /// Emit a payload-free `NotificationsChanged` pointer (wire v37) after a notification-manager
    /// mutation so clients re-list via `NotificationList`. Mirrors `emit_contacts_changed` /
    /// `CatalogChanged`: the whole list is cheap to refetch, so the event carries no detail.
    pub(crate) fn emit_notifications_changed(&self) {
        if let Some(feed) = self.node_feed() {
            // rung 1: bump the notifications rev (once per emit) and stamp it (echoed by `NotificationList`).
            let rev = feed.note_notifications_change();
            feed.emit(NodeEvent::NotificationsChanged { rev });
        }
    }

    /// A snapshot of the node's live notifications (newest first) — the [`ControlApi::notification_list`]
    /// backing (ported from libpurple's `PurpleNotificationManager`).
    pub(crate) fn notifications_snapshot(&self) -> Vec<daemon_api::NotificationInfo> {
        self.notifications
            .lock()
            .expect("notification manager mutex")
            .list()
    }

    /// Add a notification to the node manager and emit the `NotificationsChanged` pointer on a real
    /// add (a rejected double-add emits nothing). The producer seam adapters/tools use to raise a
    /// notification onto the node's list.
    pub fn notify_add(
        &self,
        notification: daemon_api::NotificationInfo,
    ) -> crate::notifications::AddOutcome {
        let outcome = self
            .notifications
            .lock()
            .expect("notification manager mutex")
            .add(notification);
        if outcome == crate::notifications::AddOutcome::Added {
            self.emit_notifications_changed();
        }
        outcome
    }

    /// Remove a notification from the node manager by id, emitting the `NotificationsChanged`
    /// pointer when one was removed.
    pub fn notify_remove(&self, id: &str) -> bool {
        let removed = self
            .notifications
            .lock()
            .expect("notification manager mutex")
            .remove(id);
        if removed {
            self.emit_notifications_changed();
        }
        removed
    }

    /// Emit a payload-free `PersonsChanged` pointer (wire v37) after a person-registry mutation
    /// so clients re-list via `PersonList`. Mirrors `emit_notifications_changed`: the whole list is
    /// cheap to refetch, so the event carries no detail.
    pub(crate) fn emit_persons_changed(&self) {
        if let Some(feed) = self.node_feed() {
            // rung 1: bump the persons rev (once per emit) and stamp it (echoed by `PersonList`).
            let rev = feed.note_persons_change();
            feed.emit(NodeEvent::PersonsChanged { rev });
        }
    }

    /// A snapshot of the node's person registry (insertion order) — the
    /// [`ControlApi::person_list`] backing (ported from the person half of libpurple's
    /// `PurpleContactManager`).
    pub(crate) fn persons_snapshot(&self) -> Vec<daemon_api::Person> {
        self.persons.lock().expect("person manager mutex").list()
    }

    /// Add a person to the node registry and emit the `PersonsChanged` pointer on a real add (a
    /// rejected double-add emits nothing). The producer seam adapters/tools use to create a person.
    pub fn person_add(&self, person: daemon_api::Person) -> crate::person::AddOutcome {
        let outcome = self
            .persons
            .lock()
            .expect("person manager mutex")
            .add_person(person);
        if outcome == crate::person::AddOutcome::Added {
            self.emit_persons_changed();
        }
        outcome
    }

    /// Remove a person from the node registry by id, emitting the `PersonsChanged` pointer when one
    /// was removed.
    pub fn person_remove(&self, id: &str, remove_endpoints: bool) -> bool {
        let removed = self
            .persons
            .lock()
            .expect("person manager mutex")
            .remove_person(id, remove_endpoints);
        if removed {
            self.emit_persons_changed();
        }
        removed
    }

    /// Associate a contact endpoint with a person, emitting the `PersonsChanged` pointer when the
    /// edge was created (an unknown person / duplicate edge emits nothing).
    pub fn person_associate(&self, person_id: &str, endpoint: daemon_api::PersonEndpoint) -> bool {
        let associated = self
            .persons
            .lock()
            .expect("person manager mutex")
            .associate(person_id, endpoint);
        if associated {
            self.emit_persons_changed();
        }
        associated
    }

    /// Dissociate a contact endpoint from a person, emitting the `PersonsChanged` pointer when the
    /// edge existed.
    pub fn person_dissociate(
        &self,
        person_id: &str,
        transport: &TransportId,
        contact_id: &str,
    ) -> bool {
        let dissociated = self
            .persons
            .lock()
            .expect("person manager mutex")
            .dissociate(person_id, transport, contact_id);
        if dissociated {
            self.emit_persons_changed();
        }
        dissociated
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
            // rung 1: bump the per-transport conversation-set rev (once per emit) and stamp it so
            // `ConvList`'s echoed rev and this pointer agree on the reflected generation.
            let rev = feed.note_conversations_change(&transport);
            feed.emit(NodeEvent::ConversationsChanged {
                transport,
                conv,
                change,
                rev,
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

    async fn chat_message(
        &self,
        transport: TransportId,
        conv: String,
        message: daemon_api::ChatMessage,
    ) {
        // Node-owned consequences of one adapter-reported message, at the single choke point every
        // messaging adapter shares: append the durable Chat record onto the conversation's journal
        // (the stream `ConvHistory` pages), THEN announce it. The pointer is emitted only for a
        // durably-recorded message, so a client acting on it always finds the record.
        if self.journal_chat_message(&transport, &conv, &message).await {
            if let Some(feed) = self.node_feed() {
                feed.emit(NodeEvent::MessagesChanged { transport, conv });
            }
        }
    }
}
