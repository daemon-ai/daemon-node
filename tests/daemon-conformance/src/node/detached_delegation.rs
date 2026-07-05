// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE W9 DETACHED-DELEGATION GATE: the orchestrate `spawn wait:false` mode — a non-suspending
//! delegation whose child runs in the background and, on its terminal state, delivers a completion
//! *notice* to its parent as a fresh reactive turn (Cursor's `run_in_background: true` analogue).
//!
//! Unlike a joining delegation (which suspends the parent on the `waiting_for`/`completion_inbox`
//! rail), a detached child binds a completion-notice edge: `mark_completed` pushes a
//! `CompletionNotice`, drained by the node's notice worker and injected through the one
//! lifecycle-aware `inject_session_input` seam. This suite drives the REAL assembled node (durable
//! job worker + notice worker + inject seam) to prove:
//!
//! - a **live** parent takes the notice as a reactive `StartTurn` (a second `TurnFinished`);
//! - a **parked durable** parent gets the notice through the store seam (pending input + wake),
//!   folded into its rehydrated conversation;
//! - a detached fan-out materializes **distinct** children (`{parent}/dN`), each self-closing;
//! - a **settled** parent cleanly drops the notice (its owner is gone);
//! - a **failed** child still notifies its parent (a failed turn is a terminal `mark_completed`).

use super::harness::*;

use daemon_api::{ControlApi, Outbound, SessionApi};
use daemon_common::{JobId, ReqId};
use daemon_core::{
    Capabilities, Failure, ModelOutput, PendingApproval, Provider, ProviderRegistry, Request,
    Snapshot, ToolCall, ToolCallFormat,
};
use daemon_protocol::{AgentCommand, AgentEvent, ConvTurnView, DelegationInput, UserMsg};
use daemon_store::{ChildLifetime, JobCommand};

/// A provider that fans out `spawns` detached orchestrate calls on its first round (no tool result
/// yet, no completion-notice already folded in), then completes. Used as the durable orchestrator
/// provider so an assigned parent spawns a bounded, deterministic number of background children.
struct DetachedFanoutProvider {
    spawns: usize,
}

#[async_trait::async_trait]
impl Provider for DetachedFanoutProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let saw_notice = req
            .messages
            .iter()
            .any(|m| m.role == "user" && m.content.contains("[subagent"));
        if saw_notice || req.has_tool_result() {
            // The reactive notice turn, or the round after our spawn calls resolved: complete.
            return Ok(ModelOutput {
                text: "parent done".into(),
                ..Default::default()
            });
        }
        // First round: emit `spawns` detached orchestrate calls in one batch. A child (depth-guarded)
        // gets `depth-limit` results for these and then completes — no grandchildren.
        let tool_calls = (0..self.spawns)
            .map(|i| ToolCall {
                call_id: format!("spawn-{i}"),
                name: "orchestrate".into(),
                args: format!(r#"{{"verb":"spawn","wait":false,"task":"kid {i}"}}"#),
            })
            .collect();
        Ok(ModelOutput {
            text: String::new(),
            tool_calls,
            ..Default::default()
        })
    }
}

/// A provider that FAILS its turn when its seeded task carries the `PLEASE_FAIL` sentinel (a detached
/// child seeded to fail), and otherwise completes. A failed turn is a terminal `mark_completed`, so
/// the child still fires its completion notice (ratification #7).
struct FailingChildProvider;

#[async_trait::async_trait]
impl Provider for FailingChildProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let fail = req
            .messages
            .iter()
            .any(|m| m.role == "user" && m.content.contains("PLEASE_FAIL"));
        if fail && !req.has_tool_result() {
            return Err(Failure::Fatal("child asked to fail".into()));
        }
        Ok(ModelOutput {
            text: "child done".into(),
            ..Default::default()
        })
    }
}

/// Assemble a node over `store` whose durable **orchestrator** provider (and default) is
/// `orchestrator` — the recursive shape every durable session (top + child) is built from — so a
/// custom orchestrator provider drives detached fan-out / failure. A depth cap of `nesting_depth + 1`
/// keeps children from spawning grandchildren.
fn assemble_with_orchestrator(
    store: Arc<dyn SessionStore>,
    nesting_depth: usize,
    seed: [u8; 32],
    orchestrator: daemon_core::ProviderBuilder,
) -> AssembledNode {
    let mut providers = ProviderRegistry::new();
    providers.set_default(orchestrator.clone());
    providers.register("orchestrator", orchestrator.clone());
    providers.register("child", orchestrator);
    assemble_node(NodeAssembly {
        store,
        partition: PARTITION,
        host_config: fast_host_config(),
        providers,
        credentials: None,
        profile: ProfileRef::new("orchestrator"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some(seed),
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
    })
}

/// Enqueue a detached job for `parent` (as the orchestrate tool's `wait:false` path does), returning
/// the store-minted `{parent}/dN` child id. The node's job worker materializes + runs the child; its
/// terminal completion fires a notice the notice worker delivers.
async fn enqueue_detached(
    store: &Arc<dyn SessionStore>,
    parent: &SessionId,
    task: &str,
) -> SessionId {
    let payload = DelegationInput {
        task: task.into(),
        attachments: Vec::new(),
        lifetime: daemon_protocol::DelegationLifetime::Persistent,
        profile: None,
        detached: true,
    }
    .encode();
    store
        .enqueue_detached_job(JobCommand {
            job_id: JobId::new(format!("{parent}:detached")),
            session_id: parent.clone(),
            epoch: daemon_common::Epoch::ZERO,
            payload,
            lifetime: ChildLifetime::Persistent,
            child: None,
        })
        .await
        .expect("enqueue detached job")
}

/// Poll-drain the live session until `want` further `TurnFinished` events arrive (bounded).
async fn drain_finished(node: &Arc<NodeApiImpl>, session: &SessionId, want: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen = 0usize;
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            if matches!(&o, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                seen += 1;
            }
        }
        if seen >= want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("expected {want} TurnFinished event(s), saw {seen}");
}

/// Snapshot a live session's conversation turns (submits `Snapshot`, polls for the reply).
async fn snapshot_turns(node: &Arc<NodeApiImpl>, session: &SessionId) -> Vec<ConvTurnView> {
    node.submit(
        session.clone(),
        AgentCommand::Snapshot {
            request_id: ReqId(9),
        },
    )
    .await
    .expect("submit snapshot");
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            if let Outbound::Event(AgentEvent::Snapshot { view, .. }) = o {
                return view.turns;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("no snapshot reply");
}

/// A live (actor-resident) parent takes a detached child's completion as a fresh reactive turn: a
/// second `TurnFinished` runs with no user submit, and the notice text lands in the conversation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn detached_notice_drives_a_reactive_turn_on_a_live_parent() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x91; 32], fast_host_config());
    let parent = SessionId::new("detach-live");

    node.submit(
        parent.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit opens a live parent");
    drain_finished(&node, &parent, 1).await;

    // The detached child materializes, runs, and self-closes; its notice injects a reactive turn
    // (one further TurnFinished — `poll` drains, so this counts only the new reactive turn).
    let child = enqueue_detached(&store, &parent, "background work").await;
    assert_eq!(child.as_str(), "detach-live/d1");
    drain_finished(&node, &parent, 1).await;

    let turns = snapshot_turns(&node, &parent).await;
    assert!(
        turns
            .iter()
            .any(|t| t.text.contains("[subagent") && t.text.contains("detach-live/d1")),
        "the completion notice landed in the live parent's conversation: {turns:?}"
    );

    handle.shutdown().await;
}

/// A parked durable parent (dormant on an unanswered edit approval — not terminal) receives a
/// detached child's notice through the store seam: the pending input is drained at the wake's
/// hydrate and folded into the re-checkpointed conversation, while the parent stays parked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn detached_notice_reaches_a_parked_durable_parent() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode {
        node: _node,
        handle,
        ..
    } = assemble_over(store.clone(), 0, [0x92; 32], fast_host_config());
    let parent = SessionId::new("detach-parked");

    // Seed a session parked on an unanswered approval (the stable dormant durable state). The node's
    // recovery scanner activates the Ready row and the engine deterministically PARKS it.
    let job_id = JobId::new(format!("{parent}:1:approval:0"));
    let mut snapshot = Snapshot::fresh(parent.clone());
    snapshot.waiting_for = vec![job_id.clone()];
    snapshot.pending_approvals = vec![PendingApproval {
        job_id,
        call: ToolCall {
            call_id: "c1".into(),
            name: "fs".into(),
            args: r#"{"op":"write","path":"gated.txt","content":"hi"}"#.into(),
        },
        prompt: "approve write to gated.txt".into(),
        path: Some("gated.txt".into()),
    }];
    store
        .create_session(
            parent.clone(),
            PARTITION,
            snapshot.encode().expect("encode"),
        )
        .await
        .expect("create parked parent");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&parent).await,
        Some(daemon_store::SessionStatus::Suspended { .. })
    ) {
        assert!(Instant::now() < deadline, "the parent never parked");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The detached child materializes + self-closes; its notice injects (pending input + wake).
    let child = enqueue_detached(&store, &parent, "background work").await;
    assert_eq!(child.as_str(), "detach-parked/d1");

    // The notice reaches the parked parent's conversation. Re-nudge the wake each pass to recover
    // from a wake that a concurrent recovery-scanner activation (which hydrated before the input
    // landed) benignly absorbed — exactly the pattern the process-notify parked gate uses.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        store.enqueue_wake(parent.clone()).await;
        let has_marker = store
            .peek_snapshot(&parent)
            .await
            .and_then(|blob| Snapshot::decode(&blob).ok())
            .map(|s| {
                s.conversation.turns.iter().any(|t| {
                    matches!(t, daemon_core::Turn::User(msg) if msg.text.contains("[subagent") && msg.text.contains("detach-parked/d1"))
                })
            })
            .unwrap_or(false);
        if has_marker {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the detached notice never reached the parked parent"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The parent stayed parked (the notice did not fast-forward the approval).
    assert!(matches!(
        store.status(&parent).await,
        Some(daemon_store::SessionStatus::Suspended { .. } | daemon_store::SessionStatus::Active)
    ));

    handle.shutdown().await;
}

/// A detached fan-out from an orchestrator parent materializes 3 DISTINCT background children
/// (`{parent}/d1..d3`), each self-closing to `Completed` — no duplicate ids, no grandchildren.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn detached_fanout_materializes_distinct_children() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let orchestrator: daemon_core::ProviderBuilder =
        Arc::new(|| Arc::new(DetachedFanoutProvider { spawns: 3 }) as Arc<dyn Provider>);
    let AssembledNode { node, handle, .. } =
        assemble_with_orchestrator(store.clone(), 0, [0x93; 32], orchestrator);
    let parent = SessionId::new("detach-fan");

    node.assign(parent.clone()).await.expect("assign parent");

    // Wait for the 3 distinct children to materialize and complete.
    let expected: Vec<SessionId> = (1..=3)
        .map(|n| SessionId::new(format!("detach-fan/d{n}")))
        .collect();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let mut all_done = true;
        for c in &expected {
            if store.status(c).await != Some(daemon_store::SessionStatus::Completed) {
                all_done = false;
            }
        }
        if all_done {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the 3 detached children never all completed: {:?}",
            futures::future::join_all(expected.iter().map(|c| store.status(c))).await
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Exactly the 3 distinct children hang under the parent (no dupes, no d4+).
    let children = store.children_of(&parent).await;
    for c in &expected {
        assert!(
            children.contains(c),
            "missing detached child {c}: {children:?}"
        );
    }
    assert!(
        store
            .status(&SessionId::new("detach-fan/d4"))
            .await
            .is_none(),
        "no extra children beyond the fan-out"
    );

    handle.shutdown().await;
}

/// An injected notice for a **settled** parent is cleanly dropped: the detached child still completes
/// and fires its notice, but the notice worker drops it (the parent's owner is gone) rather than
/// resurrecting the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn notice_to_a_settled_parent_is_cleanly_dropped() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x94; 32], fast_host_config());
    let parent = SessionId::new("detach-settled");

    node.assign(parent.clone()).await.expect("assign");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(
        store.status(&parent).await,
        Some(daemon_store::SessionStatus::Completed)
    ) {
        assert!(Instant::now() < deadline, "the parent never settled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let child = enqueue_detached(&store, &parent, "background work").await;
    // Wait for the child to complete (so its notice has fired + been drained).
    let deadline = Instant::now() + Duration::from_secs(10);
    while store.status(&child).await != Some(daemon_store::SessionStatus::Completed) {
        assert!(
            Instant::now() < deadline,
            "the detached child never completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Give the notice worker time to drain + drop the notice.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The parent stayed settled and holds no pending input (the notice was dropped, not queued).
    assert!(matches!(
        store.status(&parent).await,
        Some(daemon_store::SessionStatus::Completed)
    ));
    assert!(
        store.take_session_inputs(&parent).await.is_empty(),
        "a settled parent queues nothing off a detached notice"
    );

    handle.shutdown().await;
}

/// Ratification #7: a FAILED detached child still notifies its parent — a failed turn is a terminal
/// `mark_completed`, so the completion-notice branch fires regardless of outcome. Uses a live parent
/// (deterministic reactive-turn delivery) so the assertion turns purely on the failed child's notice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_detached_child_still_notifies_its_parent() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let orchestrator: daemon_core::ProviderBuilder =
        Arc::new(|| Arc::new(FailingChildProvider) as Arc<dyn Provider>);
    let AssembledNode { node, handle, .. } =
        assemble_with_orchestrator(store.clone(), 0, [0x95; 32], orchestrator);
    let parent = SessionId::new("detach-failparent");

    // A live parent (its own turns never carry the fail sentinel, so it completes normally).
    node.submit(
        parent.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hello"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit opens a live parent");
    drain_finished(&node, &parent, 1).await;

    // The child is seeded to FAIL; its terminal (failed) completion must still fire the notice, which
    // drives a reactive turn on the live parent.
    let child = enqueue_detached(&store, &parent, "PLEASE_FAIL now").await;
    assert_eq!(child.as_str(), "detach-failparent/d1");
    let deadline = Instant::now() + Duration::from_secs(15);
    while store.status(&child).await != Some(daemon_store::SessionStatus::Completed) {
        assert!(
            Instant::now() < deadline,
            "the failed child never reached a terminal state"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drain_finished(&node, &parent, 1).await;

    let turns = snapshot_turns(&node, &parent).await;
    assert!(
        turns
            .iter()
            .any(|t| t.text.contains("[subagent") && t.text.contains("detach-failparent/d1")),
        "a failed detached child's notice must still reach the parent: {turns:?}"
    );

    handle.shutdown().await;
}
