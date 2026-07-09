// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Completion-notice provenance (wire v29, F2): the turn the notice worker injects into a parent
//! when a detached (`spawn wait:false`) child completes carries structured chip-link fields —
//! `UserMsg.notice = { child, call_id }` — through the REAL rails: the spawn-time `call_id` rides
//! the durable completion-notice edge into the outbox (`mark_completed`, one transaction), the
//! notice worker builds the provenance-tagged `UserMsg`, and the live parent's merged session log
//! shows the injected `StartTurn` carrying it — exactly what a client tails to render the turn
//! with a chip back to the delegation card.

use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{from_cbor, to_cbor, SessionApi};
use daemon_common::{Epoch, PartitionId, ProfileRef, ReqId, SessionId};
use daemon_core::{MockProvider, Provider, ProviderRegistry, Snapshot};
use daemon_host::HostConfig;
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_protocol::{
    AgentCommand, CompletionNoticeRef, DelegationResult, SessionPayload, UserMsg,
};
use daemon_store::{Checkpoint, InMemoryStore, SessionStore};

fn assemble_min(store: Arc<dyn SessionStore>) -> AssembledNode {
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(MockProvider::completing("noted")) as Arc<dyn Provider>
    }));
    assemble(NodeAssembly {
        store,
        partition: PartitionId::DEFAULT,
        host_config: HostConfig::default(),
        providers,
        credentials: None,
        profile: ProfileRef::new("default"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x48; 32]),
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

/// The wire shape round-trips, and a pre-v29 encoding (no `notice` key) decodes as `None`.
#[test]
fn user_msg_notice_round_trips() {
    let msg = UserMsg::new("[subagent p/d1 completed] done").with_notice(CompletionNoticeRef {
        child: SessionId::new("p/d1"),
        call_id: Some("call-42".into()),
    });
    assert_eq!(msg, from_cbor::<UserMsg>(&to_cbor(&msg)).unwrap());
    let legacy = UserMsg::new("plain");
    let back: UserMsg = from_cbor(&to_cbor(&legacy)).unwrap();
    assert_eq!(back.notice, None);
}

/// The full provenance rail: spawn-time call_id on the edge -> terminal push -> notice worker ->
/// the LIVE parent's merged log shows the injected StartTurn carrying `notice { child, call_id }`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_notice_turn_carries_provenance() {
    daemon_host::with_request_context(
        daemon_host::RequestContext::system(),
        injected_notice_turn_carries_provenance_impl(),
    )
    .await;
}
async fn injected_notice_turn_carries_provenance_impl() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, .. } = assemble_min(store.clone());

    // A LIVE parent (the notice worker's live path submits a reactive StartTurn into it).
    let parent = node
        .session_create(Some(SessionId::new("notice-parent")), None)
        .await
        .expect("create the live parent");
    node.submit(
        parent.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("open the live parent");

    // The detached child, bound with the SPAWNING TOOL CALL's id (what the orchestrate tool
    // stamps at `spawn wait:false`), completing with a structured DelegationResult — the same
    // durable transaction the production path uses.
    let child = SessionId::new("notice-parent/d1");
    store
        .bind_completion_notice(&child, &parent, Some("call-42".into()))
        .await
        .expect("bind the completion-notice edge");
    let blob = Snapshot::fresh(child.clone()).encode().expect("encode");
    store
        .create_session(child.clone(), PartitionId::DEFAULT, blob)
        .await
        .expect("create the child");
    let fence = store.acquire_activation_lease(&child).await.expect("lease");
    store
        .mark_completed(
            Checkpoint::new(
                child.clone(),
                Epoch(1),
                Snapshot::fresh(child.clone()).encode().expect("encode"),
            )
            .with_completion_payload(Some(DelegationResult::summary("crunched it").encode())),
            fence,
        )
        .await
        .expect("complete the detached child");

    // The notice worker drains the outbox and injects the provenance-tagged turn.
    let delivered = daemon_node::fleet::NoticeWorker::new(store.clone(), node.clone())
        .drain_once()
        .await;
    assert_eq!(delivered, 1, "exactly one notice delivered");

    // The parent's merged log shows the injected StartTurn with the chip-link fields.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = node
            .log_after(parent.clone(), 0, 0)
            .await
            .expect("read the parent log");
        let injected = page.entries.iter().find_map(|e| match &e.payload {
            SessionPayload::Command(AgentCommand::StartTurn { input, .. }) => {
                input.notice.clone().map(|n| (n, input.text.clone()))
            }
            _ => None,
        });
        if let Some((notice, text)) = injected {
            assert_eq!(notice.child, child);
            assert_eq!(notice.call_id.as_deref(), Some("call-42"));
            assert!(
                text.contains("[subagent notice-parent/d1 completed] crunched it"),
                "the human-readable text stays: {text:?}"
            );
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the injected notice turn; log: {:?}",
            page.entries.len()
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
