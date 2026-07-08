// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Presence push (wire v29, B5): `NodeEvent::TransportChanged` rides the L3 node-event feed at the
//! coarse REAL transport transitions — the instance's reported state when its adapter's serve loop
//! starts, `Offline` at a clean teardown — carrying the full new state so a client updates its
//! channel/presence dots without re-polling `TransportInstances`. Driven through the real seams:
//! a registered `TransportAdapter`, `spawn_adapters`, and the `events_page` read.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{
    from_cbor, to_cbor, AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ConnectionState,
    ControlApi, NodeApi, NodeEvent, PresenceState, TransportAdapter, TransportInstanceInfo,
};
use daemon_common::{PartitionId, ProfileRef};
use daemon_core::{MockProvider, Provider, ProviderRegistry};
use daemon_host::{AdapterRegistry, HostConfig};
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_protocol::TransportId;
use daemon_store::InMemoryStore;

/// A scripted transport adapter: one configured instance reporting `Connected`, whose serve loop
/// parks until released (so the test controls the teardown transition).
struct MockTransport {
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl TransportAdapter for MockTransport {
    fn family(&self) -> &str {
        "mock"
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: "mock".to_string(),
            display_name: "Mock transport".to_string(),
            capabilities: AdapterCapabilities::default(),
            account_schema: AccountSettingsSchema::default(),
            policies: Vec::new(),
            ..Default::default()
        }
    }

    async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {
        self.release.notified().await;
    }

    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        vec![TransportInstanceInfo {
            transport: TransportId::new("mock/acct"),
            family: "mock".into(),
            display_name: "mock account".into(),
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
}

fn assemble_min() -> AssembledNode {
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>
    }));
    assemble(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x4a; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
    })
}

/// The wire shape round-trips (struct variant with the serde-default presence).
#[test]
fn transport_changed_round_trips() {
    let ev = NodeEvent::TransportChanged {
        transport: TransportId::new("mock/acct"),
        connection: ConnectionState::Connected,
        presence: PresenceState::Unknown,
        reason: None,
        message: None,
        fatal: false,
    };
    assert_eq!(ev, from_cbor::<NodeEvent>(&to_cbor(&ev)).unwrap());
}

/// The full push rail: serve start emits the instance's reported state (Connected), a clean
/// teardown emits Offline — both readable from the L3 `events_page` feed without any
/// `TransportInstances` poll.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_transitions_push_events() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        let release = Arc::new(tokio::sync::Notify::new());
        node.set_adapters(AdapterRegistry::new().with_adapter(Arc::new(MockTransport {
            release: release.clone(),
        })));
        let handles = node.spawn_adapters().await;

        // Serve start: the baseline push carries the instance's reported Connected state.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let connected = loop {
            let page = node.events_page(0, 0).await;
            let hit = page.events.iter().find_map(|e| match e {
                NodeEvent::TransportChanged {
                    transport,
                    connection,
                    ..
                } => Some((transport.clone(), *connection)),
                _ => None,
            });
            if let Some(hit) = hit {
                break hit;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the serve-start TransportChanged"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        };
        assert_eq!(connected.0.as_str(), "mock/acct");
        assert_eq!(connected.1, ConnectionState::Connected);

        // Clean teardown: releasing the serve loop pushes Offline for the same instance.
        release.notify_waiters();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let page = node.events_page(0, 0).await;
            let offline = page.events.iter().any(|e| {
                matches!(
                    e,
                    NodeEvent::TransportChanged {
                        transport,
                        connection: ConnectionState::Offline,
                        ..
                    } if transport.as_str() == "mock/acct"
                )
            });
            if offline {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the teardown TransportChanged"
            );
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        for h in handles {
            h.abort();
        }
    })
    .await;
}
