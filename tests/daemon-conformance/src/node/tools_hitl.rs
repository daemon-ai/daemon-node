// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// A provider registry whose session default is a deterministic [`ScriptedProvider`] that drives
/// the real ReAct loop: write a file, read it back, run a command, then finish. Orchestrator /
/// child slots are completing mocks (unused by this leaf-work scenario, but the composition root
/// resolves them).
fn core_tools_providers() -> ProviderRegistry {
    use daemon_core::{ScriptStep, ScriptedProvider};
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(ScriptedProvider::new(
            vec![
                ScriptStep::Call {
                    name: "fs".into(),
                    args: r#"{"op":"write","path":"note.txt","content":"hello from daemon-core"}"#
                        .into(),
                },
                ScriptStep::Call {
                    name: "fs".into(),
                    args: r#"{"op":"read","path":"note.txt"}"#.into(),
                },
                ScriptStep::Call {
                    name: "shell".into(),
                    args: r#"{"command":"printf","args":["ran-%s","ok"]}"#.into(),
                },
            ],
            "work complete",
        )) as Arc<dyn Provider>
    }));
    providers.register(
        "orchestrator",
        Arc::new(|| Arc::new(MockProvider::completing("orchestrator done")) as Arc<dyn Provider>),
    );
    providers.register(
        "child",
        Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
    );
    providers
}

fn assemble_core_tools(store: Arc<dyn SessionStore>) -> AssembledNode {
    assemble_node(NodeAssembly {
        store,
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: core_tools_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        // A headless autonomous driver (no operator attached to answer a §12 edit-approval):
        // opt the interactive session into `AutoAllow` so its real fs/shell work runs without
        // parking. A GUI-attached session instead selects `Ask`/`AcceptEdits` via SetSessionMode.
        engine_config: daemon_core::Config {
            approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
            ..daemon_core::Config::default()
        },
        journal_seed: Some([0x33; 32]),
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
    })
}

/// THE BRAIN GATE: a `daemon-core` session does *real local work* in one turn through the node
/// surface — the in-turn ReAct loop (§4.2) runs the §13 fs + shell tools (write -> read -> exec)
/// against its contained workspace, and the tool I/O lands in the durable, verified
/// `session_history`. Asserted against both store backends.
async fn core_tools_session_does_real_work(store: Arc<dyn SessionStore>) {
    as_system(core_tools_session_does_real_work_impl(store)).await;
}
async fn core_tools_session_does_real_work_impl(store: Arc<dyn SessionStore>) {
    use daemon_api::{JournalRecordPayload, Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, AgentEvent, TranscriptBlock, UserMsg};

    let AssembledNode { node, handle, .. } = assemble_core_tools(store);
    let session = SessionId::new("core-tools-1");

    as_system(node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("do file work"),
            request_id: ReqId(1),
        },
    ))
    .await
    .expect("submit StartTurn");

    // Drain the live session until the turn finishes, collecting every outbound event.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut events = Vec::new();
    let mut finished = false;
    while Instant::now() < deadline {
        let drained = node.poll(session.clone(), 0).await.expect("poll");
        for o in drained {
            if matches!(&o, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                finished = true;
            }
            events.push(o);
        }
        if finished {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(finished, "the core-tools turn never reached TurnFinished");

    // The loop ran the real tools: the read returned the bytes the write produced, and the shell
    // command executed in the contained workspace.
    let tool_results: Vec<_> = events
        .iter()
        .filter_map(|o| match o {
            Outbound::Event(AgentEvent::ToolFinished { result, .. }) => Some(result.clone()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results
            .iter()
            .any(|r| r.ok && r.summary.contains("hello from daemon-core")),
        "the fs read should return the written content: {tool_results:?}"
    );
    assert!(
        tool_results
            .iter()
            .any(|r| r.ok && r.summary.contains("ran-ok")),
        "the shell command should run in the workspace: {tool_results:?}"
    );

    // The tool I/O is durable + verified: scroll back through session_history until the turn's
    // sealed tool blocks appear *and* the whole segment is committed (the seal lands just after
    // TurnFinished drains, and signature commit can lag the block append under load).
    let has_tool_result = |p: &daemon_api::JournalPageView| {
        p.entries.iter().any(|e| {
            matches!(
                &e.payload,
                JournalRecordPayload::Block {
                    block: TranscriptBlock::ToolResult { .. }
                }
            )
        })
    };
    let mut page = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let p = node.session_history(session.clone(), 0, 0).await;
        if has_tool_result(&p) && p.entries.iter().all(|e| e.verified) {
            page = Some(p);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let page = page.expect("durable history should carry the sealed, verified tool blocks");
    assert!(
        page.entries.iter().all(|e| e.verified),
        "every sealed entry must verify under the node key: {page:?}"
    );
    let call_names: Vec<_> = page
        .entries
        .iter()
        .filter_map(|e| match &e.payload {
            JournalRecordPayload::Block {
                block: TranscriptBlock::ToolCall { name, .. },
            } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        call_names.iter().any(|n| n == "fs"),
        "the fs tool calls should be journaled: {call_names:?}"
    );
    assert!(
        call_names.iter().any(|n| n == "shell"),
        "the shell tool call should be journaled: {call_names:?}"
    );

    handle.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_tools_session_does_real_work_in_memory() {
    core_tools_session_does_real_work(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn core_tools_session_does_real_work_sqlite() {
    core_tools_session_does_real_work(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite store"),
    ))
    .await;
}

/// A scripted provider that emits exactly one gated `fs` write (round 1), then completes with
/// final text once the write's tool result is in the conversation (round 2). Under the default
/// `Ask` policy that single write parks an in-stream approval the live session surfaces as a
/// `SessionPayload::Request` and resolves via `respond`.
fn core_approval_providers() -> ProviderRegistry {
    use daemon_core::{ScriptStep, ScriptedProvider};
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(|| {
        Arc::new(ScriptedProvider::new(
            vec![ScriptStep::Call {
                name: "fs".into(),
                args: r#"{"op":"write","path":"approved.txt","content":"hi"}"#.into(),
            }],
            "file written after approval",
        )) as Arc<dyn Provider>
    }));
    providers.register(
        "orchestrator",
        Arc::new(|| Arc::new(MockProvider::completing("orchestrator done")) as Arc<dyn Provider>),
    );
    providers.register(
        "child",
        Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
    );
    providers
}

/// As `assemble_core_tools` but leaves the engine on the default `Ask` approval policy, so a
/// gated tool parks for an in-stream operator decision instead of auto-allowing.
fn assemble_core_approval(store: Arc<dyn SessionStore>) -> AssembledNode {
    assemble_node(NodeAssembly {
        store,
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: core_approval_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(), // default = Ask
        journal_seed: Some([0x34; 32]),
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
    })
}

/// THE LIVE HITL GATE: a live (`submit`) session's gated fs write under `Ask` raises an in-stream
/// `SessionPayload::Request(Approval)` on the merged log - the exact entry a socket client sees
/// via `Subscribe`/`log_after`. `respond(Approved(allow))` resolves the parked oneshot (the live
/// `ParkingHandler` path, NOT the durable `ApprovalsPending` inbox), the turn resumes, and it
/// completes. This is the live counterpart to the durable `answer_approval` cycle and the
/// surfacing daemon-app's DaemonTurnEngine relies on.
async fn live_approval_park_then_respond(store: Arc<dyn SessionStore>, allow: bool) {
    as_system(live_approval_park_then_respond_impl(store, allow)).await;
}
async fn live_approval_park_then_respond_impl(store: Arc<dyn SessionStore>, allow: bool) {
    use daemon_api::{Outbound, SessionApi};
    use daemon_common::ReqId;
    use daemon_protocol::{
        AgentCommand, AgentEvent, HostRequestKind, HostResponse, HostResponseBody, SessionPayload,
        UserMsg,
    };

    let AssembledNode { node, handle, .. } = assemble_core_approval(store);
    let session = SessionId::new("live-approval-1");

    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("write the note"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit StartTurn");

    // The gated write parks an in-stream Approval the merged log surfaces (log_after is the same
    // non-destructive paging surface a socket `Subscribe` reads). Poll until it appears, and
    // assert the turn has NOT finished yet (the gate is holding the turn).
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut request_id = None;
    let mut finished_early = false;
    while Instant::now() < deadline {
        let page = node
            .log_after(session.clone(), 0, 0)
            .await
            .expect("log_after");
        for e in &page.entries {
            match &e.payload {
                SessionPayload::Request(req)
                    if matches!(req.kind, HostRequestKind::Approval { .. }) =>
                {
                    request_id = Some(req.request_id);
                }
                SessionPayload::Event(AgentEvent::TurnFinished { .. }) => {
                    finished_early = true;
                }
                _ => {}
            }
        }
        if request_id.is_some() || finished_early {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        !finished_early,
        "the gated turn finished without parking an in-stream approval"
    );
    let request_id =
        request_id.expect("a live Approval HostRequest should surface on the merged log");

    // Resolve the in-stream gate (the live ParkingHandler oneshot, via `respond`).
    node.respond(
        session.clone(),
        HostResponse {
            request_id,
            body: HostResponseBody::Approved {
                approved: allow,
                allow_permanent: false,
                reason: None,
            },
        },
    )
    .await
    .expect("respond to the parked approval");

    // The turn resumes and completes either way (a deny never strands the session).
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut events = Vec::new();
    let mut finished = false;
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            if matches!(&o, Outbound::Event(AgentEvent::TurnFinished { .. })) {
                finished = true;
            }
            events.push(o);
        }
        if finished {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        finished,
        "the turn never resumed to TurnFinished after respond"
    );

    let fs_results: Vec<_> = events
        .iter()
        .filter_map(|o| match o {
            Outbound::Event(AgentEvent::ToolFinished { result, .. }) => Some(result.clone()),
            _ => None,
        })
        .collect();
    if allow {
        assert!(
            fs_results.iter().any(|r| r.ok),
            "an approved gated write should run successfully: {fs_results:?}"
        );
    } else {
        assert!(
            fs_results.iter().all(|r| !r.ok),
            "a denied gated write must not succeed: {fs_results:?}"
        );
    }

    handle.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_approval_park_allow_resumes_in_memory() {
    live_approval_park_then_respond(Arc::new(InMemoryStore::new()), true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_approval_park_allow_resumes_sqlite() {
    live_approval_park_then_respond(
        Arc::new(SqliteStore::open_in_memory().expect("open sqlite store")),
        true,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_approval_park_deny_resumes_in_memory() {
    live_approval_park_then_respond(Arc::new(InMemoryStore::new()), false).await;
}
