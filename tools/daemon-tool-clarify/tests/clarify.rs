// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `clarify` tool coverage: free-form `Input` ask, fixed-`options` `Choice` ask, and the
//! no-answer/declined path. A per-test host answers the §17 request.

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Tool, ToolCall, ToolOutcome, TurnCx};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody,
};
use daemon_tool_clarify::ClarifyTool;

/// A host that records the request kind and replies with a canned body.
struct ScriptedHost {
    reply: HostResponseBody,
    saw_choice: std::sync::Mutex<Option<Vec<String>>>,
}

impl ScriptedHost {
    fn new(reply: HostResponseBody) -> Self {
        Self {
            reply,
            saw_choice: std::sync::Mutex::new(None),
        }
    }
}

#[async_trait]
impl HostRequestHandler for ScriptedHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        if let HostRequestKind::Choice { options, .. } = &req.kind {
            *self.saw_choice.lock().unwrap() = Some(options.clone());
        }
        HostResponse {
            request_id: req.request_id,
            body: self.reply.clone(),
        }
    }
}

async fn run(tool: &dyn Tool, host: &dyn HostRequestHandler, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("clarify-test");
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host,
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
        name: "clarify".into(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

#[tokio::test]
async fn free_form_input_question() {
    let tool = ClarifyTool::new();
    let host = ScriptedHost::new(HostResponseBody::Input("blue".into()));
    let out = run(&tool, &host, r#"{"question":"favorite color?"}"#).await;
    assert!(out.result.ok);
    assert!(out.result.content.contains("user answer: blue"));
    assert!(
        host.saw_choice.lock().unwrap().is_none(),
        "no options => Input, not Choice"
    );
}

#[tokio::test]
async fn choice_question_resolves_option_text() {
    let tool = ClarifyTool::new();
    let host = ScriptedHost::new(HostResponseBody::Chosen(1));
    let out = run(
        &tool,
        &host,
        r#"{"question":"which db?","options":["postgres","sqlite"]}"#,
    )
    .await;
    assert!(out.result.ok);
    assert!(out.result.content.contains("user answer: sqlite"));
    assert_eq!(
        host.saw_choice.lock().unwrap().as_deref(),
        Some(["postgres".to_string(), "sqlite".to_string()].as_slice())
    );
}

#[tokio::test]
async fn declined_answer_is_not_ok() {
    let tool = ClarifyTool::new();
    let host = ScriptedHost::new(HostResponseBody::Approved {
        approved: false,
        allow_permanent: false,
        reason: None,
    });
    let out = run(&tool, &host, r#"{"question":"proceed?"}"#).await;
    assert!(!out.result.ok);
    assert!(out.result.content.contains("no answer"));
}

#[tokio::test]
async fn empty_question_rejected() {
    let tool = ClarifyTool::new();
    let host = ScriptedHost::new(HostResponseBody::Input("x".into()));
    let out = run(&tool, &host, r#"{"question":"   "}"#).await;
    assert!(!out.result.ok);
}

#[tokio::test]
async fn attaches_clarify_detail() {
    let tool = ClarifyTool::new();
    let host = ScriptedHost::new(HostResponseBody::Input("yes".into()));
    let out = run(&tool, &host, r#"{"question":"ok?"}"#).await;
    let detail = out.detail.expect("clarify detail");
    assert_eq!(detail.kind, "clarify");
}
