// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W10 `semantic_search` conformance: the node assembled exactly as `bins/daemon` does injects the
//! tool via `extra_tools` over a MockEmbedder-backed [`WorkspaceIndex`], and a driven turn's tool
//! call returns ranked hits that reference the seeded file — scoped to the calling session's own
//! workspace subtree (cross-session files never leak). The negative case (no tool wired, mirroring
//! "no embedder ⇒ tool absent") surfaces an `unknown tool` result.

use std::io::Write;
use std::path::Path;

use super::harness::*;
use daemon_api::SessionApi;
use daemon_common::ReqId;
use daemon_core::{ScriptStep, ScriptedProvider, Tool};
use daemon_protocol::{AgentCommand, AgentEvent, Outbound, UserMsg};
use daemon_tool_semantic_search::SemanticSearchTool;
use daemon_workspace_index::{WorkspaceIndex, WorkspaceIndexConfig};
use tokio_util::sync::CancellationToken;

/// Write `contents` to `<root>/<rel>` (creating parent dirs).
fn seed(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::File::create(&path)
        .unwrap()
        .write_all(contents.as_bytes())
        .unwrap();
}

/// A provider registry whose session default drives one `semantic_search` call (with `args`) then
/// completes; orchestrator/child slots are completing mocks (unused here but resolved by the root).
fn semantic_providers(args: &'static str) -> ProviderRegistry {
    let mut providers = ProviderRegistry::new();
    providers.set_default(Arc::new(move || {
        Arc::new(ScriptedProvider::new(
            vec![ScriptStep::Call {
                name: "semantic_search".into(),
                args: args.into(),
            }],
            "search complete",
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

/// Assemble a node rooted at `workspace_root`, wiring `extra_tools` (the `semantic_search` tool, or
/// nothing for the negative case). Auto-allow so the scripted turn runs headless without parking.
fn assemble_semantic(
    workspace_root: &Path,
    extra_tools: Vec<Arc<dyn Tool>>,
    args: &'static str,
) -> AssembledNode {
    assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: semantic_providers(args),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config {
            approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
            ..daemon_core::Config::default()
        },
        journal_seed: Some([0x77; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools,
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
        workspace_root: Some(workspace_root.to_path_buf()),
        blob_root: None,
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
    })
}

/// Build + start a MockEmbedder-backed index over `root` and wait for its initial pass to complete.
async fn ready_index(root: &Path) -> (Arc<WorkspaceIndex>, CancellationToken) {
    let index = WorkspaceIndex::open(
        &root.join("index.sqlite"),
        root.to_path_buf(),
        Arc::new(daemon_core::MockEmbedder::new(64)),
        WorkspaceIndexConfig::default(),
    )
    .expect("open workspace index");
    let cancel = CancellationToken::new();
    index.spawn(cancel.clone());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !index.ready() {
        assert!(Instant::now() < deadline, "index never became ready");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (index, cancel)
}

/// Drive one turn on `session` and return every tool-result view it emitted.
async fn drive_tool_results(
    node: &Arc<NodeApiImpl>,
    session: &SessionId,
) -> Vec<daemon_protocol::ToolResultView> {
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("find the auth code"),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit StartTurn");

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut results = Vec::new();
    let mut finished = false;
    while Instant::now() < deadline {
        for o in node.poll(session.clone(), 0).await.expect("poll") {
            match &o {
                Outbound::Event(AgentEvent::ToolFinished { result, .. }) => {
                    results.push(result.clone());
                }
                Outbound::Event(AgentEvent::TurnFinished { .. }) => finished = true,
                _ => {}
            }
        }
        if finished {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        finished,
        "the semantic_search turn never reached TurnFinished"
    );
    results
}

/// THE W10 GATE: a session's `semantic_search` call returns ranked hits referencing the seeded file
/// under ITS OWN workspace subtree — and never a file from a sibling session's subtree.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn semantic_search_returns_ranked_hits_scoped_to_the_session() {
    as_system(semantic_search_returns_ranked_hits_scoped_to_the_session_impl()).await;
}
async fn semantic_search_returns_ranked_hits_scoped_to_the_session_impl() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    // The session id is all-alphanumeric so the sandbox segment == the id (no sanitization surprise).
    let session = SessionId::new("semidx1");
    seed(
        root,
        "semidx1/auth.rs",
        "fn authenticate_user_with_jwt() {\n    verify_token();\n}\n",
    );
    // A sibling session's file the caller must never see (cross-session containment).
    seed(
        root,
        "other1/secret.rs",
        "fn other_session_secret_authenticate() {}\n",
    );

    let (index, cancel) = ready_index(root).await;
    let AssembledNode { node, handle, .. } = assemble_semantic(
        root,
        vec![Arc::new(SemanticSearchTool::new(index)) as Arc<dyn Tool>],
        r#"{"query":"authenticate jwt token"}"#,
    );

    let results = drive_tool_results(&node, &session).await;
    let joined: String = results
        .iter()
        .map(|r| r.summary.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        results
            .iter()
            .any(|r| r.ok && r.summary.contains("semidx1/auth.rs")),
        "the seeded file should be ranked: {joined}"
    );
    assert!(
        !joined.contains("other1/secret.rs"),
        "a sibling session's file must never leak across the cwd-containment filter: {joined}"
    );

    cancel.cancel();
    handle.shutdown().await;
}

/// The negative case: with no `semantic_search` tool wired (the "no embedder ⇒ tool absent" path in
/// `bins/daemon`), the model's call resolves to an `unknown tool` result rather than any hits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn semantic_search_absent_without_the_tool() {
    as_system(semantic_search_absent_without_the_tool_impl()).await;
}
async fn semantic_search_absent_without_the_tool_impl() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let session = SessionId::new("semidx2");
    seed(
        root,
        "semidx2/auth.rs",
        "fn authenticate_user_with_jwt() {}\n",
    );

    let AssembledNode { node, handle, .. } =
        assemble_semantic(root, Vec::new(), r#"{"query":"authenticate jwt"}"#);

    let results = drive_tool_results(&node, &session).await;
    assert!(
        results
            .iter()
            .any(|r| !r.ok && r.summary.contains("unknown tool")),
        "an unwired semantic_search must surface an unknown-tool result: {results:?}"
    );
    assert!(
        !results.iter().any(|r| r.summary.contains("auth.rs")),
        "no ranked hits without the tool: {results:?}"
    );

    handle.shutdown().await;
}
