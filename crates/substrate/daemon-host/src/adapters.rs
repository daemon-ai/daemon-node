//! Host-level transport-adapter registry (daemon-transport-adapter-spec.md §3.4).
//!
//! The declarative counterpart of [`crate::routing::RoutingRegistry`]: where the routing registry
//! maps an inbound `Origin` to `(session, profile, delivery)`, this registry holds the node's set of
//! self-describing events-IO transport adapters ([`daemon_api::TransportAdapter`]) and answers the
//! read-only enumeration the GUI reads to render the "Add channel" picker and capability-gate UI
//! ([`daemon_api::ControlApi::transport_adapters`]).
//!
//! Skeleton status: the registry stores `Arc<dyn TransportAdapter>` and reports each adapter's
//! [`AdapterInfo`] via [`AdapterRegistry::infos`]. It does **not** yet drive adapter *lifecycle*
//! (the `serve` spawns still live in `bins/daemon`); retrofitting adapters onto the trait and moving
//! lifecycle here is deferred (spec §7 P1). A node that registers no adapters yields an empty list,
//! so the surface is inert by default — exactly like an empty `RoutingRegistry`.

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
        self.adapters
            .iter()
            .find(|a| a.family() == family)
            .cloned()
    }

    /// The adapter that owns `transport` — the one whose family is the transport id itself (the
    /// management-addressable form, e.g. `"room"` / `"matrix"`) or a `family/...` / `family:...`
    /// sub-instance of it (e.g. a Room's internal `room/<id>` loopback id).
    pub fn adapter_for_transport(&self, transport: &TransportId) -> Option<Arc<dyn TransportAdapter>> {
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
