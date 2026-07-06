// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Delegation guardrails (wire v29, F7) through the NODE API surface: the `[orchestrate]` caps
//! thread from assembly into the orchestrate tool, a capped spawn declines with the structured
//! `guardrail` tool detail (not just the bare `depth-limit:N` string), and the read-only `Caps`
//! op reports the EFFECTIVE ceilings (config policy min'd with the assembly recursion budget) so
//! a client renders them without probing.

use std::sync::Arc;

use daemon_api::{dispatch, from_cbor, to_cbor, ApiRequest, ApiResponse, CapsReport, ControlApi};
use daemon_common::{PartitionId, ProfileRef};
use daemon_core::{MockProvider, Provider, ProviderRegistry};
use daemon_host::HostConfig;
use daemon_node::{assemble, AssembledNode, NodeAssembly, OrchestrateCaps};
use daemon_store::InMemoryStore;

fn assemble_with_caps(nesting_depth: usize, caps: OrchestrateCaps) -> AssembledNode {
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
        journal_seed: Some([0x47; 32]),
        nesting_depth,
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
        orchestrate: caps,
    })
}

/// The wire shapes round-trip.
#[test]
fn caps_round_trips() {
    let req = ApiRequest::Caps;
    assert_eq!(req, from_cbor::<ApiRequest>(&to_cbor(&req)).unwrap());
    let report = CapsReport {
        orchestrate_max_depth: 3,
        orchestrate_max_fanout: 8,
    };
    assert_eq!(report, from_cbor::<CapsReport>(&to_cbor(&report)).unwrap());
}

/// `Caps` reports the EFFECTIVE ceilings: the `[orchestrate].max_depth` policy cap composed with
/// the assembly recursion budget (`min(max_depth, nesting_depth + 1)`), and the configured fanout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn caps_reports_effective_ceilings() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        // Defaults: policy cap 8, nesting 0 -> the pre-v29 effective depth guard of 1.
        let AssembledNode { node, .. } = assemble_with_caps(0, OrchestrateCaps::default());
        assert_eq!(
            node.caps().await,
            CapsReport {
                orchestrate_max_depth: 1,
                orchestrate_max_fanout: 8,
            }
        );

        // A deeper assembly budget with a NARROWER policy cap: policy wins (it may narrow, never
        // widen), and the configured fanout threads through. Served over the real dispatch too.
        let AssembledNode { node, .. } = assemble_with_caps(
            4,
            OrchestrateCaps {
                max_depth: 3,
                max_fanout: 2,
            },
        );
        match dispatch(&*node, ApiRequest::Caps).await {
            ApiResponse::Caps(report) => assert_eq!(
                report,
                CapsReport {
                    orchestrate_max_depth: 3,
                    orchestrate_max_fanout: 2,
                }
            ),
            other => panic!("expected Caps, got {other:?}"),
        }
    })
    .await;
}
