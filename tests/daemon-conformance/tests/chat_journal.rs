// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Conversation chat journal (wire v38): the `LifecycleSink::chat_message` seam appends a
//! `JournalRecordPayload::Chat` record onto the conversation's verifiable journal stream
//! (`conv:<transport>:<conv>` — the stream `ConvHistory` pages) and emits the granular
//! `NodeEvent::MessagesChanged` pointer, once per message. The seam is the single choke point every
//! messaging adapter reports through, so journaling + announcement are inherited, never re-derived
//! per adapter.

use std::sync::Arc;

use daemon_api::{
    from_cbor, to_cbor, ChatMessage, ContactInfo, ControlApi, ConvHistoryArgs,
    JournalRecordPayload, NodeEvent, Participant,
};
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
        journal_seed: Some([0x54; 32]),
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

/// The new node-event variant survives the shared CBOR codec (the additive wire v38 arm).
#[test]
fn messages_changed_round_trips() {
    let ev = NodeEvent::MessagesChanged {
        transport: TransportId::new("matrix/@bot:hs.org"),
        conv: "!room:hs.org".into(),
    };
    assert_eq!(ev, from_cbor::<NodeEvent>(&to_cbor(&ev)).unwrap());
}

/// One `chat_message` report through the node's lifecycle sink lands as ONE verified
/// `JournalRecordPayload::Chat` entry on `conv:<transport>:<conv>` (readable via `conv_history`,
/// with the `ChatMessage` intact) and raises ONE `MessagesChanged { transport, conv }` on the
/// node-wide feed. A second report appends after the first (stable, strictly-increasing cursors;
/// `after_cursor` pages past the first record).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_chat_message_journals_and_emits() {
    daemon_host::with_request_context(daemon_host::RequestContext::system(), async {
        let AssembledNode { node, .. } = assemble_min();
        let transport = TransportId::new("matrix/@bot:hs.org");
        let conv = "!room:hs.org";
        let sink = node.lifecycle_sink();

        let mut inbound = ChatMessage::new(
            Some(Participant::Contact(ContactInfo {
                id: "@alice:hs.org".into(),
                ..ContactInfo::default()
            })),
            "hello over the wire",
        );
        inbound.id = Some("$evt1:hs.org".into());
        inbound.timestamp = Some(1_720_000_000);
        sink.chat_message(transport.clone(), conv.to_string(), inbound.clone())
            .await;

        let history = |after_cursor: u64| {
            node.conv_history(ConvHistoryArgs {
                transport: transport.clone(),
                conv: conv.to_string(),
                after_cursor,
                before_cursor: None,
                max: 0,
            })
        };

        let page = history(0).await;
        assert_eq!(
            page.entries.len(),
            1,
            "one chat_message report = one journal record, got {page:?}"
        );
        let first = &page.entries[0];
        assert_eq!(first.kind, "chat.message");
        assert!(
            first.verified,
            "the per-message segment is sealed under the node signer and verifies"
        );
        match &first.payload {
            JournalRecordPayload::Chat { message } => assert_eq!(**message, inbound),
            other => panic!("expected JournalRecordPayload::Chat, got {other:?}"),
        }

        // The granular pointer is on the L3 feed, carrying the same (transport, conv).
        let events = node.events_page(0, 0).await;
        let pointers = events
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    NodeEvent::MessagesChanged { transport: t, conv: c }
                        if t == &transport && c == conv
                )
            })
            .count();
        assert_eq!(pointers, 1, "one append = one MessagesChanged pointer");

        // A second (outbound, account-authored) message appends AFTER the first.
        let outbound = ChatMessage::new(None, "reply from the account");
        sink.chat_message(transport.clone(), conv.to_string(), outbound.clone())
            .await;

        let page = history(0).await;
        assert_eq!(page.entries.len(), 2, "both directions on one stream");
        assert!(
            page.entries[0].cursor < page.entries[1].cursor,
            "cursors are stream-monotonic (append order)"
        );
        match &page.entries[1].payload {
            JournalRecordPayload::Chat { message } => assert_eq!(**message, outbound),
            other => panic!("expected JournalRecordPayload::Chat, got {other:?}"),
        }

        // `after_cursor` pages past the first record — stable, non-destructive.
        let tail = history(first.cursor).await;
        assert_eq!(tail.entries.len(), 1, "after_cursor skips the first record");
        assert_eq!(tail.entries[0].cursor, page.entries[1].cursor);

        let pointers = node
            .events_page(0, 0)
            .await
            .events
            .iter()
            .filter(|e| matches!(e, NodeEvent::MessagesChanged { .. }))
            .count();
        assert_eq!(pointers, 2, "MessagesChanged is emitted once per append");
    })
    .await;
}
