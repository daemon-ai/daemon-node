// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool inventory (wire v29, D2) through the NODE API surface: the binary late-binds the
//! node-wide inventory (what registered + why each disabled config-gated surface did not), and
//! `ToolList` serves it over the real dispatch fan-out with the enriched `enabled`/`requires`
//! shape a client renders as "why is this tool unavailable". (The gate-mirroring itself is
//! unit-tested next to the build gates in `bins/daemon`.)

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
        journal_seed: Some([0x46; 32]),
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

/// The enriched `ToolInfo` round-trips, and an `enabled` encoding without `requires` decodes with
/// `requires: None` (the field is serde-default).
#[test]
fn tool_info_round_trips() {
    let rows = [
        ToolInfo {
            name: "shell".into(),
            description: None,
            enabled: true,
            requires: None,
        },
        ToolInfo {
            name: "web_search".into(),
            description: Some("web search".into()),
            enabled: false,
            requires: Some("[web].enable (+ a tavily credential)".into()),
        },
    ];
    for row in rows {
        assert_eq!(row, from_cbor::<ToolInfo>(&to_cbor(&row)).unwrap());
    }
}

/// `ToolList` serves the late-bound inventory through the real dispatch fan-out; a node without a
/// bound inventory serves the empty default.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_list_serves_the_bound_inventory() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();

        // Unbound: the trait default (empty) — a node whose binary bound no inventory.
        match dispatch(&*node, ApiRequest::ToolList).await {
            ApiResponse::Tools(tools) => assert!(tools.is_empty(), "unbound => empty: {tools:?}"),
            other => panic!("expected Tools, got {other:?}"),
        }

        // Bound: the binary's inventory (enabled + disabled-with-requires rows) serves verbatim.
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
                requires: Some(
                    "[browser].enable + a daemon built with the `browser` feature".into(),
                ),
            },
        ]);
        match dispatch(&*node, ApiRequest::ToolList).await {
            ApiResponse::Tools(tools) => {
                assert_eq!(tools.len(), 2, "{tools:?}");
                assert!(tools[0].enabled && tools[0].requires.is_none());
                assert!(!tools[1].enabled);
                assert!(
                    tools[1]
                        .requires
                        .as_deref()
                        .is_some_and(|r| r.contains("browser")),
                    "the disabled row names what it requires: {tools:?}"
                );
            }
            other => panic!("expected Tools, got {other:?}"),
        }
    })
    .await;
}
