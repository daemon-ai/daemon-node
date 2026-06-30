// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Routing-table maintenance (daemon-event-io-spec §5.9 hot-reload): rebuild the live
//! [`RoutingRegistry`](crate::routing::RoutingRegistry) from the base + durable chat→session pins,
//! and the wire⇄store [`ChatRoute`] codec the `routing_*` control ops read/write through.

use super::*;

impl NodeApiImpl {
    /// Hot-swap the *base* routing table (live). Used by the rebuild hook and available to the binary
    /// for an explicit refresh; an in-flight `submit_routed` resolve (which clones the inner `Arc`) is
    /// unaffected. The swap re-layers the current chat→session pins on top of the new base.
    pub fn swap_routing(&self, routing: RoutingRegistry) {
        self.routing_base.store(Arc::new(routing));
        self.rebuild_routing();
    }

    /// Rebuild the live routing table: take the base (the rebuild hook's output when installed, else
    /// the static `routing_base`), layer the durable chat→session pins on top, and swap it in. A
    /// no-op-ish refresh when no builder is set, but always re-applies pins. Called after profile/auth
    /// mutations and after a pin reload so routing stays current without a restart.
    pub(crate) fn rebuild_routing(&self) {
        let mut reg = match &self.routing_builder {
            Some(builder) => builder(),
            None => (*self.routing_base.load_full()).clone(),
        };
        reg.set_pins(self.chat_pins.read().unwrap().clone());
        self.routing.store(Arc::new(reg));
    }

    /// Reload the durable chat→session routing pins (§5.9, I5) from the store into the in-memory pin
    /// cache and re-layer them onto the live registry. Called at boot (by the assembling binary) and
    /// after every `routing_*` mutation, riding the same hot-reload seam as profile/auth changes.
    pub async fn load_routing_pins(&self) {
        let routes = self.store.routing_list().await;
        let mut map = std::collections::HashMap::with_capacity(routes.len());
        for r in routes {
            map.insert(
                r.key.clone(),
                crate::routing::ChatPin {
                    session: r.session_id.clone(),
                    profile: r.profile.clone(),
                },
            );
        }
        *self.chat_pins.write().unwrap() = map;
        self.rebuild_routing();
    }
}

/// Encode a wire [`ChatRoute`] into the protocol-free store row (§5.9, I5): the canonical origin key
/// plus typed `session`/`profile` columns, with the full wire descriptor (origin + isolation)
/// carried as the opaque CBOR `descriptor` blob for faithful round-trip.
pub(crate) fn store_route_from_wire(route: &ChatRoute) -> daemon_store::ChatRoute {
    daemon_store::ChatRoute {
        key: crate::routing::origin_pin_key(&route.origin),
        session_id: route.session.clone(),
        profile: route.profile.clone(),
        descriptor: to_cbor(route),
    }
}

/// Decode a store row back to the wire [`ChatRoute`] from its opaque descriptor blob (`None` if the
/// blob fails to decode — a forward-compat/corruption guard).
pub(crate) fn wire_route_from_store(route: &daemon_store::ChatRoute) -> Option<ChatRoute> {
    from_cbor(&route.descriptor).ok()
}

/// Whether a stored route's transport instance belongs to the requested transport: an exact instance
/// match, or a family match (the requested id is the `family` segment before the first `/`). Lets
/// `transport_rooms("matrix")` enumerate rooms across every `matrix/@account` instance.
pub(crate) fn transport_family_matches(have: &TransportId, want: &TransportId) -> bool {
    have == want || have.as_str().split('/').next() == Some(want.as_str())
}

/// A human room/chat label for [`RoomInfo`], derived from an origin scope.
pub(crate) fn room_label(scope: &OriginScope) -> String {
    match scope {
        OriginScope::Dm { user } => user.clone(),
        OriginScope::Group { chat, .. } => chat.clone(),
        OriginScope::Api { key } => key.clone(),
        OriginScope::Internal => "internal".to_string(),
        other => format!("{other:?}"),
    }
}
