// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `todo` tool coverage: replace vs merge, per-session isolation, and the rendered checklist.

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_tool_todo::TodoTool;

struct NoopHost;
#[async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

async fn run(tool: &dyn Tool, session: &str, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("todo-test");
    let host = NoopHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new(session),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: "todo".into(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

#[tokio::test]
async fn replace_sets_the_list() {
    let tool = TodoTool::new();
    let out = run(
        &tool,
        "s1",
        r#"{"todos":[{"id":"a","content":"first","status":"pending"},{"id":"b","content":"second","status":"in_progress"}]}"#,
    )
    .await;
    assert!(out.result.ok);
    assert!(out.result.content.contains("0/2 complete"));
    assert!(out.result.content.contains("[ ] first"));
    assert!(out.result.content.contains("[~] second"));
}

#[tokio::test]
async fn merge_upserts_by_id() {
    let tool = TodoTool::new();
    run(
        &tool,
        "s1",
        r#"{"todos":[{"id":"a","content":"first","status":"pending"}]}"#,
    )
    .await;
    // Merge: update a's status and add c, keeping the list.
    let out = run(
        &tool,
        "s1",
        r#"{"merge":true,"todos":[{"id":"a","content":"first","status":"completed"},{"id":"c","content":"third","status":"pending"}]}"#,
    )
    .await;
    assert!(out.result.content.contains("1/2 complete"));
    assert!(out.result.content.contains("[x] first"));
    assert!(out.result.content.contains("[ ] third"));
}

#[tokio::test]
async fn lists_are_per_session() {
    let tool = TodoTool::new();
    run(
        &tool,
        "s1",
        r#"{"todos":[{"id":"a","content":"s1 task","status":"pending"}]}"#,
    )
    .await;
    let out = run(
        &tool,
        "s2",
        r#"{"todos":[{"id":"z","content":"s2 task","status":"pending"}]}"#,
    )
    .await;
    assert!(out.result.content.contains("s2 task"));
    assert!(
        !out.result.content.contains("s1 task"),
        "sessions must not bleed"
    );
}

#[tokio::test]
async fn attaches_todo_detail() {
    let tool = TodoTool::new();
    let out = run(
        &tool,
        "s1",
        r#"{"todos":[{"id":"a","content":"x","status":"pending"}]}"#,
    )
    .await;
    let detail = out.detail.expect("todo detail attached");
    assert_eq!(detail.kind, "todo");
    assert!(!detail.body.is_empty());
}

#[tokio::test]
async fn invalid_args_reported() {
    let tool = TodoTool::new();
    let out = run(&tool, "s1", "nonsense").await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("invalid arguments"));
}
