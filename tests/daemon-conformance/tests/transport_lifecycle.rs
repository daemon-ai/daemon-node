// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Transport lifecycle ops (wire v30, item 1): `TransportDisconnect`/`TransportRemove` round-trip
//! and route through the real `dispatch` fan-out. A node with no adapter owning the transport
//! returns an error (the intent is well-formed; there is nothing to act on).

use std::sync::Arc;

use daemon_api::{dispatch, from_cbor, to_cbor, ApiRequest, ApiResponse};
use daemon_common::{PartitionId, ProfileRef};
use daemon_core::{MockProvider, Provider, ProviderRegistry};
use daemon_host::HostConfig;
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_protocol::TransportId;
use daemon_store::InMemoryStore;

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
        journal_seed: Some([0x51; 32]),
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
        prompt: Default::default(),
    })
}

#[test]
fn transport_lifecycle_requests_round_trip() {
    for req in [
        ApiRequest::TransportDisconnect {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
        ApiRequest::TransportRemove {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    ] {
        assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_lifecycle_routes_through_dispatch() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        // No adapter owns the transport: a well-formed intent with nothing to act on -> error
        // (proves the routing fan-out reaches the op rather than hitting `unreachable!`).
        match dispatch(
            &*node,
            ApiRequest::TransportDisconnect {
                transport: TransportId::new("matrix/@bot:hs.org"),
            },
        )
        .await
        {
            ApiResponse::Error(_) => {}
            other => panic!("expected Error for an unowned transport, got {other:?}"),
        }
        match dispatch(
            &*node,
            ApiRequest::TransportRemove {
                transport: TransportId::new("matrix/@bot:hs.org"),
            },
        )
        .await
        {
            ApiResponse::Error(_) => {}
            other => panic!("expected Error for an unowned transport, got {other:?}"),
        }
    })
    .await;
}
