// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;
use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::ToolCall;
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};

const ARXIV: &str = "---\nname: arxiv\ndescription: Search arXiv papers.\nversion: 1.0.0\n---\n\n# arXiv\n\nBody.\n";

struct NoopHost;
#[async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
            },
        }
    }
}

/// Build a minimal turn context for invoking a tool directly.
async fn run(tool: &dyn Tool, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("skill-tool-test");
    let host = NoopHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new("s"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
        session_allow: &[],
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: tool.name().into(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

fn store() -> (tempfile::TempDir, Arc<SkillStore>) {
    let d = tempfile::tempdir().unwrap();
    let store = Arc::new(SkillStore::new(d.path().join("skills")));
    (d, store)
}

#[tokio::test]
async fn manage_create_then_list_and_view() {
    let (_d, store) = store();
    let manage = SkillManageTool::new(store.clone());
    let list = SkillsListTool::new(store.clone());
    let view = SkillViewTool::new(store.clone());

    let args = serde_json::json!({
        "action": "create",
        "name": "arxiv",
        "category": "research",
        "content": ARXIV,
    })
    .to_string();
    let out = run(&manage, &args).await;
    assert!(out.result.ok, "create succeeds: {:?}", out.result.content);

    let listed = run(&list, "{}").await;
    assert!(listed.result.content.contains("arxiv"));
    assert!(listed.result.content.contains("research"));

    let viewed = run(&view, &serde_json::json!({"name": "arxiv"}).to_string()).await;
    assert!(viewed.result.ok);
    assert!(viewed.result.content.contains("Body."));
}

#[tokio::test]
async fn manage_patch_and_write_file_and_delete() {
    let (_d, store) = store();
    let manage = SkillManageTool::new(store.clone());
    let view = SkillViewTool::new(store.clone());
    run(
        &manage,
        &serde_json::json!({"action":"create","name":"arxiv","content":ARXIV}).to_string(),
    )
    .await;

    let patched = run(
        &manage,
        &serde_json::json!({"action":"patch","name":"arxiv","old_string":"Body.","new_string":"Updated body."}).to_string(),
    )
    .await;
    assert!(patched.result.ok, "{:?}", patched.result.content);
    let viewed = run(&view, &serde_json::json!({"name":"arxiv"}).to_string()).await;
    assert!(viewed.result.content.contains("Updated body."));

    let wrote = run(
        &manage,
        &serde_json::json!({"action":"write_file","name":"arxiv","file_path":"references/api.md","file_content":"endpoints"}).to_string(),
    )
    .await;
    assert!(wrote.result.ok, "{:?}", wrote.result.content);
    let linked = run(
        &view,
        &serde_json::json!({"name":"arxiv","file_path":"references/api.md"}).to_string(),
    )
    .await;
    assert!(linked.result.content.contains("endpoints"));

    let deleted = run(
        &manage,
        &serde_json::json!({"action":"delete","name":"arxiv"}).to_string(),
    )
    .await;
    assert!(deleted.result.ok);
    assert!(store.discover().is_empty());
}

#[tokio::test]
async fn errors_are_reported_not_panicked() {
    let (_d, store) = store();
    let manage = SkillManageTool::new(store.clone());
    let view = SkillViewTool::new(store.clone());

    let missing = run(&view, &serde_json::json!({"name":"ghost"}).to_string()).await;
    assert!(!missing.result.ok);
    assert!(missing.result.content.contains("not found"));

    let bad_action = run(
        &manage,
        &serde_json::json!({"action":"nope","name":"x"}).to_string(),
    )
    .await;
    assert!(!bad_action.result.ok);

    let missing_content = run(
        &manage,
        &serde_json::json!({"action":"create","name":"x"}).to_string(),
    )
    .await;
    assert!(!missing_content.result.ok);
    assert!(missing_content.result.content.contains("content"));
}
