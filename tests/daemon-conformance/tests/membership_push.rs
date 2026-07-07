// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Membership push (wire v30, item 3): the two `NodeEvent` tiers + their enums round-trip, and the
//! node-owned routing reconciliation — on a self `Kicked` the node drops the dangling `ChatRoute`
//! pin BEFORE emitting, and surfaces both events on the L3 feed. A non-self departure leaves the
//! pin intact.

use std::sync::Arc;

use daemon_api::{
    from_cbor, to_cbor, ChatRoute, ControlApi, ConvChange, MembershipChange, NodeEvent,
};
use daemon_common::{PartitionId, ProfileRef, SessionId};
use daemon_core::{MockProvider, Provider, ProviderRegistry};
use daemon_host::HostConfig;
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_protocol::{IsolationPolicy, Origin, OriginScope, TransportId};
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
        journal_seed: Some([0x53; 32]),
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
    })
}

fn group_origin(transport: &str, conv: &str) -> Origin {
    Origin::new(
        TransportId::new(transport),
        OriginScope::Group {
            chat: conv.to_string(),
            thread: None,
        },
    )
}

#[test]
fn membership_events_round_trip() {
    let events = [
        NodeEvent::ConversationsChanged {
            transport: TransportId::new("matrix/@bot:hs.org"),
            conv: "!r:hs".into(),
            change: ConvChange::Added,
        },
        NodeEvent::MembershipChanged {
            transport: TransportId::new("matrix/@bot:hs.org"),
            conv: "!r:hs".into(),
            member: "@bot:hs.org".into(),
            change: MembershipChange::Kicked,
            actor: Some("@admin:hs".into()),
            reason: Some("cleanup".into()),
            is_self: true,
        },
    ];
    for ev in events {
        assert_eq!(ev, from_cbor::<NodeEvent>(&to_cbor(&ev)).unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn self_removal_drops_the_routing_pin_and_emits() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        let transport = TransportId::new("matrix/@bot:hs.org");
        let conv = "!room:hs.org";
        let origin = group_origin(transport.as_str(), conv);

        // Seed a routing pin for the conversation's origin.
        node.routing_set(ChatRoute {
            origin: origin.clone(),
            session: SessionId::new("s-pinned"),
            profile: None,
            isolation: IsolationPolicy::PerThread,
        })
        .await
        .expect("seed pin");
        assert!(
            node.routing_get(origin.clone()).await.is_some(),
            "pin seeded"
        );

        // A self kick reconciles routing (drops the pin) BEFORE emitting the event.
        let sink = node.lifecycle_sink();
        sink.membership_changed(
            transport.clone(),
            conv.to_string(),
            "@bot:hs.org".to_string(),
            MembershipChange::Kicked,
            Some("@admin:hs".to_string()),
            None,
            true,
        )
        .await;

        assert!(
            node.routing_get(origin.clone()).await.is_none(),
            "the dangling pin was dropped on self removal"
        );
        // The invalidation event is on the L3 feed.
        let page = node.events_page(0, 0).await;
        assert!(
            page.events.iter().any(|e| matches!(
                e,
                NodeEvent::MembershipChanged {
                    is_self: true,
                    change: MembershipChange::Kicked,
                    ..
                }
            )),
            "the membership event was emitted"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_self_departure_keeps_the_pin() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        let transport = TransportId::new("matrix/@bot:hs.org");
        let conv = "!room:hs.org";
        let origin = group_origin(transport.as_str(), conv);

        node.routing_set(ChatRoute {
            origin: origin.clone(),
            session: SessionId::new("s-pinned"),
            profile: None,
            isolation: IsolationPolicy::PerThread,
        })
        .await
        .expect("seed pin");

        let sink = node.lifecycle_sink();
        sink.membership_changed(
            transport.clone(),
            conv.to_string(),
            "@someone:hs.org".to_string(),
            MembershipChange::Left,
            None,
            None,
            false, // another member left — not us.
        )
        .await;

        assert!(
            node.routing_get(origin).await.is_some(),
            "a non-self departure leaves the pin untouched"
        );
    })
    .await;
}
