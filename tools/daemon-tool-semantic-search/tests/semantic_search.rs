// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `semantic_search` tool coverage over a real [`WorkspaceIndex`] (MockEmbedder-backed): the
//! unready message, ranked `path:span` output, the `num_results` clamp, `target_directories`
//! narrowing, and the MANDATORY session-cwd containment filter.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{MockEmbedder, Tool, ToolCall, ToolConcurrency, TurnCx};
use daemon_tool_semantic_search::SemanticSearchTool;
use daemon_workspace_index::{WorkspaceIndex, WorkspaceIndexConfig};
use tokio_util::sync::CancellationToken;

/// A host that must never be consulted (the tool is index-only).
struct NoHost;

#[async_trait::async_trait]
impl daemon_protocol::HostRequestHandler for NoHost {
    async fn request(&self, req: daemon_protocol::HostRequest) -> daemon_protocol::HostResponse {
        panic!("semantic_search must not raise host requests: {req:?}");
    }
}

/// Materialize a workspace fixture with the given `(relpath, contents)` files.
fn fixture(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (rel, contents) in files {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(contents.as_bytes())
            .unwrap();
    }
    dir
}

fn open_index(root: &Path) -> Arc<WorkspaceIndex> {
    WorkspaceIndex::open(
        &root.join("index.sqlite"),
        root.to_path_buf(),
        Arc::new(MockEmbedder::new(64)),
        WorkspaceIndexConfig::default(),
    )
    .unwrap()
}

async fn wait_ready(idx: &Arc<WorkspaceIndex>) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !idx.ready() {
        assert!(Instant::now() < deadline, "index never became ready");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Run the tool with `args`, with the session exec env rooted at `cwd`. Returns `(ok, content)`.
async fn run_tool(index: Arc<WorkspaceIndex>, cwd: &Path, args: &str) -> (bool, String) {
    let tool = SemanticSearchTool::new(index);
    assert_eq!(tool.concurrency(), ToolConcurrency::Parallel);
    assert!(!tool.mutates());
    let events = EventSink::discarding();
    let exec = LocalEnvironment::new(cwd);
    let host = NoHost;
    let cx = TurnCx {
        cancel: CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new("sem-test"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "semantic_search".into(),
        args: args.into(),
    };
    let out = tool.run(&call, &cx).await;
    (out.result.ok, out.result.content)
}

#[tokio::test]
async fn unready_index_reports_still_building_as_ok() {
    let dir = fixture(&[("a.rs", "fn alpha() {}\n")]);
    let idx = open_index(dir.path());
    // Not spawned ⇒ never ready.
    assert!(!idx.ready());
    let (ok, content) = run_tool(idx, dir.path(), r#"{"query":"alpha"}"#).await;
    assert!(ok, "unready is transport-ok, never an error");
    assert!(content.contains("still building"), "{content}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ranked_results_carry_path_span_and_snippet() {
    let dir = fixture(&[(
        "src/auth.rs",
        "fn authenticate_user_with_jwt() {\n    verify_token();\n}\n",
    )]);
    let idx = open_index(dir.path());
    let cancel = CancellationToken::new();
    let handle = idx.spawn(cancel.clone());
    wait_ready(&idx).await;

    let (ok, content) = run_tool(idx, dir.path(), r#"{"query":"authenticate jwt token"}"#).await;
    assert!(ok);
    assert!(
        content.contains("src/auth.rs:1-"),
        "path:span header: {content}"
    );
    assert!(
        content.contains("authenticate_user_with_jwt"),
        "chunk text: {content}"
    );

    cancel.cancel();
    handle.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn num_results_is_clamped() {
    // 20 single-line files ⇒ 20 chunks. num_results=100 clamps to 15.
    let files: Vec<(String, String)> = (0..20)
        .map(|i| {
            (
                format!("f{i}.rs"),
                format!("fn func{i}() {{ shared_token(); }}\n"),
            )
        })
        .collect();
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let dir = fixture(&refs);
    let idx = open_index(dir.path());
    let cancel = CancellationToken::new();
    let handle = idx.spawn(cancel.clone());
    wait_ready(&idx).await;

    let (_, content) = run_tool(
        idx.clone(),
        dir.path(),
        r#"{"query":"shared token","num_results":100}"#,
    )
    .await;
    assert!(
        content.starts_with("15 result(s)"),
        "clamped to 15: {}",
        &content[..40.min(content.len())]
    );

    // num_results=0 clamps up to 1.
    let (_, one) = run_tool(
        idx,
        dir.path(),
        r#"{"query":"shared token","num_results":0}"#,
    )
    .await;
    assert!(
        one.starts_with("1 result(s)"),
        "clamped to 1: {}",
        &one[..40.min(one.len())]
    );

    cancel.cancel();
    handle.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn target_directories_narrow_the_search() {
    let dir = fixture(&[
        ("area_a/one.rs", "fn shared_token_a() {}\n"),
        ("area_b/two.rs", "fn shared_token_b() {}\n"),
    ]);
    let idx = open_index(dir.path());
    let cancel = CancellationToken::new();
    let handle = idx.spawn(cancel.clone());
    wait_ready(&idx).await;

    let (_, content) = run_tool(
        idx,
        dir.path(),
        r#"{"query":"shared token","target_directories":["area_a"]}"#,
    )
    .await;
    assert!(content.contains("area_a/one.rs"), "{content}");
    assert!(
        !content.contains("area_b/two.rs"),
        "area_b excluded: {content}"
    );

    cancel.cancel();
    handle.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cwd_containment_hides_other_sessions() {
    // The index root is the PARENT of two session sandboxes. A session rooted at `ses1` must never
    // see `ses2`'s files, even without any target_directories filter.
    let dir = fixture(&[
        ("ses1/mine.rs", "fn shared_token_mine() {}\n"),
        ("ses2/other.rs", "fn shared_token_other() {}\n"),
    ]);
    let idx = open_index(dir.path());
    let cancel = CancellationToken::new();
    let handle = idx.spawn(cancel.clone());
    wait_ready(&idx).await;

    let cwd = dir.path().join("ses1");
    let (ok, content) = run_tool(idx, &cwd, r#"{"query":"shared token"}"#).await;
    assert!(ok);
    assert!(
        content.contains("ses1/mine.rs"),
        "own file present: {content}"
    );
    assert!(
        !content.contains("ses2/other.rs"),
        "cross-session file must never leak: {content}"
    );

    cancel.cancel();
    handle.await.unwrap();
}
