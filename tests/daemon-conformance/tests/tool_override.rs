// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool override overlay (wire v30, item 6): `ToolSetEnabled` persists a node-wide override that
//! `tool_list` overlays on the bound inventory. Force-disable is always honored; a force-enable can
//! never conjure a tool missing its build feature (a `requires` row stays disabled).

use std::sync::Arc;

use daemon_api::{dispatch, from_cbor, to_cbor, ApiRequest, ApiResponse, ToolInfo};
use daemon_common::{PartitionId, ProfileRef};
use daemon_core::{MockProvider, Provider, ProviderRegistry};
use daemon_host::HostConfig;
use daemon_node::{assemble, AssembledNode, NodeAssembly};
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
        journal_seed: Some([0x52; 32]),
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

#[test]
fn tool_set_enabled_round_trips() {
    let req = ApiRequest::ToolSetEnabled {
        tool: "browser".into(),
        enabled: false,
    };
    assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_override_overlays_tool_list() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        node.set_tool_inventory(vec![
            ToolInfo {
                name: "fs".into(),
                description: None,
                enabled: true,
                requires: None,
            },
            ToolInfo {
                name: "browser".into(),
                description: None,
                enabled: false,
                requires: Some("[browser].enable + a browser build feature".into()),
            },
        ]);

        // Force-disable an enabled tool.
        assert!(matches!(
            dispatch(
                &*node,
                ApiRequest::ToolSetEnabled {
                    tool: "fs".into(),
                    enabled: false,
                },
            )
            .await,
            ApiResponse::Ok
        ));
        // Force-enable a build-gated tool (must NOT re-enable it — decision E).
        assert!(matches!(
            dispatch(
                &*node,
                ApiRequest::ToolSetEnabled {
                    tool: "browser".into(),
                    enabled: true,
                },
            )
            .await,
            ApiResponse::Ok
        ));

        match dispatch(&*node, ApiRequest::ToolList).await {
            ApiResponse::Tools(tools) => {
                let fs = tools.iter().find(|t| t.name == "fs").unwrap();
                let browser = tools.iter().find(|t| t.name == "browser").unwrap();
                assert!(!fs.enabled, "force-disable is honored");
                assert!(
                    !browser.enabled,
                    "force-enable cannot conjure a build-gated tool"
                );
            }
            other => panic!("expected Tools, got {other:?}"),
        }
    })
    .await;
}
