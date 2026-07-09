// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Host-level transport-adapter registry (daemon-transport-adapter-spec.md §3.4).
//!
//! The declarative counterpart of [`crate::routing::RoutingRegistry`]: where the routing registry
//! maps an inbound `Origin` to `(session, profile, delivery)`, this registry holds the node's set of
//! self-describing events-IO transport adapters ([`daemon_api::TransportAdapter`]) and answers the
//! read-only enumeration the GUI reads to render the "Add channel" picker and capability-gate UI
//! ([`daemon_api::ControlApi::transport_adapters`]).
//!
//! Status: the registry stores `Arc<dyn TransportAdapter>`, reports each adapter's [`AdapterInfo`] via
//! [`AdapterRegistry::infos`] / live instances via [`AdapterRegistry::instances`], and drives adapter
//! *lifecycle* via [`AdapterRegistry::spawn_all`] (called by `NodeApiImpl::spawn_adapters`, used from
//! `bins/daemon`, which registers `daemon-rooms`/`daemon-matrix`). A node that registers no adapters
//! yields an empty list, so the surface is inert by default — exactly like an empty `RoutingRegistry`.

use daemon_api::{AdapterInfo, NodeApi, TransportAdapter, TransportInstanceInfo};
use daemon_protocol::TransportId;
use std::sync::Arc;

/// The host's transport-adapter registry: an ordered set of registered adapters. Cheaply cloneable
/// (it holds `Arc`s), mirroring how [`crate::routing::RoutingRegistry`] is carried on `NodeApiImpl`.
#[derive(Clone, Default)]
pub struct AdapterRegistry {
    adapters: Vec<Arc<dyn TransportAdapter>>,
}

impl AdapterRegistry {
    /// An empty registry (a node with no events-IO transport adapters registered).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an adapter (builder form). First-registered is listed first.
    pub fn with_adapter(mut self, adapter: Arc<dyn TransportAdapter>) -> Self {
        self.adapters.push(adapter);
        self
    }

    /// Whether no adapters are registered.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// The descriptor of every registered adapter (the wire enumeration for
    /// [`daemon_api::ControlApi::transport_adapters`]).
    pub fn infos(&self) -> Vec<AdapterInfo> {
        self.adapters.iter().map(|a| a.info()).collect()
    }

    /// The registered adapters.
    pub fn adapters(&self) -> &[Arc<dyn TransportAdapter>] {
        &self.adapters
    }

    /// The adapter whose [`family`](TransportAdapter::family) equals `family`, if any.
    pub fn adapter_for_family(&self, family: &str) -> Option<Arc<dyn TransportAdapter>> {
        self.adapters.iter().find(|a| a.family() == family).cloned()
    }

    /// The adapter that owns `transport` — the one whose family is the transport id itself (the
    /// management-addressable form, e.g. `"room"` / `"matrix"`) or a `family/...` / `family:...`
    /// sub-instance of it (e.g. a Room's internal `room/<id>` loopback id).
    pub fn adapter_for_transport(
        &self,
        transport: &TransportId,
    ) -> Option<Arc<dyn TransportAdapter>> {
        let t = transport.as_str();
        self.adapters
            .iter()
            .find(|a| {
                let f = a.family();
                t == f || t.starts_with(&format!("{f}/")) || t.starts_with(&format!("{f}:"))
            })
            .cloned()
    }

    /// Every configured instance (account) across all adapters, with live connection/presence state
    /// (the wire enumeration for [`daemon_api::ControlApi::transport_instances`]).
    pub async fn instances(&self) -> Vec<TransportInstanceInfo> {
        let mut out = Vec::new();
        for adapter in &self.adapters {
            out.extend(adapter.instances().await);
        }
        out
    }

    /// Drive every adapter's [`serve`](TransportAdapter::serve) loop on the runtime, returning the
    /// spawned task handles (the host aborts them on shutdown). Each adapter wires its own
    /// `daemon-ingest`/`daemon-delivery` halves; a disabled adapter returns immediately.
    pub fn spawn_all(&self, api: Arc<dyn NodeApi>) -> Vec<tokio::task::JoinHandle<()>> {
        self.adapters
            .iter()
            .map(|adapter| {
                let adapter = adapter.clone();
                let api = api.clone();
                tokio::spawn(async move { adapter.serve(api).await })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::{
        AccountSettingsValues, ConnectionState, MessagingProtocol, PresenceState,
        TransportInstanceInfo,
    };

    /// A fake transport adapter with a configurable family + one account instance. `is_messaging`
    /// flips whether [`TransportAdapter::messaging`] returns `Some` (the daemon analogue of a
    /// `PurpleProtocol` vs a generic transport). `serve` is never driven by these registry tests.
    struct FakeAdapter {
        family: &'static str,
        display: &'static str,
        is_messaging: bool,
    }

    impl FakeAdapter {
        fn generic(family: &'static str, display: &'static str) -> Arc<Self> {
            Arc::new(Self {
                family,
                display,
                is_messaging: false,
            })
        }
        fn messaging(family: &'static str, display: &'static str) -> Arc<Self> {
            Arc::new(Self {
                family,
                display,
                is_messaging: true,
            })
        }
    }

    #[async_trait::async_trait]
    impl TransportAdapter for FakeAdapter {
        fn family(&self) -> &str {
            self.family
        }

        fn info(&self) -> AdapterInfo {
            AdapterInfo {
                family: self.family.to_string(),
                display_name: self.display.to_string(),
                ..Default::default()
            }
        }

        async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {}

        async fn instances(&self) -> Vec<TransportInstanceInfo> {
            vec![TransportInstanceInfo {
                transport: TransportId::new(format!("{}/acct", self.family)),
                family: self.family.to_string(),
                display_name: format!("{} account", self.display),
                connection: ConnectionState::Connected,
                presence: PresenceState::Unknown,
                bound_profile: None,
                reason: None,
                message: None,
                fatal: false,
                enabled: true,
                label: None,
            }]
        }

        fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
            if self.is_messaging {
                Some(self)
            } else {
                None
            }
        }
    }

    // A `FakeAdapter` flagged messaging is a `MessagingProtocol` that overrides nothing, so every
    // feature probe falls to its default (`None` / `Ok`).
    impl MessagingProtocol for FakeAdapter {}

    /// `test_protocol_manager.c` `/protocol-manager/new` + `/properties`: a fresh registry is a
    /// list with zero items — here, no adapters, so every enumeration is empty.
    #[tokio::test]
    async fn empty_registry_is_inert() {
        let reg = AdapterRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.infos().is_empty(), "n-items == 0 analogue");
        assert!(reg.instances().await.is_empty());
        assert!(reg.adapter_for_family("anything").is_none());
        // Default is the same empty registry.
        assert!(AdapterRegistry::default().is_empty());
    }

    /// Registering appends in order; `infos()` reports each adapter's descriptor, first-registered
    /// first (the wire enumeration the GUI renders).
    #[test]
    fn register_orders_and_enumerates_infos() {
        let reg = AdapterRegistry::new()
            .with_adapter(FakeAdapter::generic("room", "Rooms"))
            .with_adapter(FakeAdapter::messaging("matrix", "Matrix"));
        assert!(!reg.is_empty());
        let infos = reg.infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].family, "room", "first registered listed first");
        assert_eq!(infos[0].display_name, "Rooms");
        assert_eq!(infos[1].family, "matrix");
        assert_eq!(reg.adapters().len(), 2);
    }

    /// `instances()` concatenates every adapter's configured accounts (the account-manager
    /// analogue), preserving adapter order.
    #[tokio::test]
    async fn instances_aggregate_across_adapters() {
        let reg = AdapterRegistry::new()
            .with_adapter(FakeAdapter::generic("room", "Rooms"))
            .with_adapter(FakeAdapter::messaging("matrix", "Matrix"));
        let instances = reg.instances().await;
        assert_eq!(instances.len(), 2, "one account per adapter, aggregated");
        assert_eq!(instances[0].transport.as_str(), "room/acct");
        assert_eq!(instances[1].transport.as_str(), "matrix/acct");
    }

    /// The `messaging()` probe: a messaging adapter yields `Some(MessagingProtocol)`, a generic
    /// transport yields `None` (daemon-messaging-adapter-spec.md §3.1).
    #[test]
    fn messaging_probe_distinguishes_messaging_from_generic() {
        let reg = AdapterRegistry::new()
            .with_adapter(FakeAdapter::generic("http", "HTTP"))
            .with_adapter(FakeAdapter::messaging("matrix", "Matrix"));
        let http = reg.adapter_for_family("http").expect("http registered");
        assert!(
            http.messaging().is_none(),
            "a generic transport is not a messaging protocol"
        );
        let matrix = reg.adapter_for_family("matrix").expect("matrix registered");
        assert!(
            matrix.messaging().is_some(),
            "a messaging adapter probes as a MessagingProtocol"
        );
    }

    /// A `MessagingProtocol` that overrides no feature interface reports every optional capability
    /// as absent and accepts any account (the default-impl analogue of
    /// `test_credential_provider_empty.c`).
    #[tokio::test]
    async fn messaging_feature_probes_default_to_none() {
        let reg = AdapterRegistry::new().with_adapter(FakeAdapter::messaging("matrix", "Matrix"));
        let adapter = reg.adapter_for_family("matrix").expect("registered");
        let proto = adapter.messaging().expect("messaging protocol");
        assert!(proto.clone().conversations().is_none());
        assert!(proto.clone().membership().is_none());
        assert!(proto.clone().roster().is_none());
        assert!(proto.clone().contacts().is_none());
        assert!(proto.clone().directory().is_none());
        assert!(proto.clone().file_transfer().is_none());
        assert!(
            proto
                .validate_account(&AccountSettingsValues::default())
                .await
                .is_ok(),
            "the default validate_account accepts"
        );
    }

    /// Lookup by `family` and by `TransportId`: the owning adapter is found for the bare id, for a
    /// `family/…` sub-instance, and for a `family:…` form; an unrelated id/family misses.
    #[test]
    fn lookup_by_family_and_transport() {
        let reg = AdapterRegistry::new().with_adapter(FakeAdapter::generic("room", "Rooms"));
        assert!(reg.adapter_for_family("room").is_some());
        assert!(reg.adapter_for_family("matrix").is_none());

        assert!(reg
            .adapter_for_transport(&TransportId::new("room"))
            .is_some());
        assert!(
            reg.adapter_for_transport(&TransportId::new("room/loopback-1"))
                .is_some(),
            "a family/... sub-instance resolves to the family adapter"
        );
        assert!(
            reg.adapter_for_transport(&TransportId::new("room:internal"))
                .is_some(),
            "a family:... form resolves too"
        );
        assert!(
            reg.adapter_for_transport(&TransportId::new("matrix/@bot:hs"))
                .is_none(),
            "an unrelated family misses"
        );
        // A family that is only a prefix of the id's family-word must NOT match (`room` vs `rooms`).
        assert!(
            reg.adapter_for_transport(&TransportId::new("rooms/x"))
                .is_none(),
            "prefix-only family word is not a match"
        );
    }

    /// Divergence from libpurple's `PurpleProtocolManager` (which *rejects* a duplicate id): the
    /// daemon registry is an ordered `Vec` with no dedup — both adapters with the same family are
    /// retained in `infos()`, and `adapter_for_family` returns the first registrant.
    #[test]
    fn duplicate_family_both_retained_first_wins() {
        let reg = AdapterRegistry::new()
            .with_adapter(FakeAdapter::generic("dup", "First"))
            .with_adapter(FakeAdapter::generic("dup", "Second"));
        let infos = reg.infos();
        assert_eq!(infos.len(), 2, "no dedup: both retained");
        let found = reg.adapter_for_family("dup").expect("resolves");
        assert_eq!(
            found.info().display_name,
            "First",
            "first registrant wins the lookup"
        );
    }
}
