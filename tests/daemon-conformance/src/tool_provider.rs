// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE DYNAMIC-TOOL SEAM GATE: a `daemon_core::ToolProvider` (the boundary shared by the Python
//! worker `daemon-pytool-client` and any future MCP client) is queried through a trait object,
//! its discovered tools are registered on a real `ToolRegistry`, and the engine invokes one of
//! them through both the `run_tool` pipeline and a full ReAct turn — proving a runtime-discovered
//! tool is indistinguishable from a native one at the engine boundary. Hermetic: the provider is
//! an in-crate fake (the cross-process Python path is covered by `daemon-pytool`'s own tests).

use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::conversation::Turn;
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{
    run_tool, Config, Engine, ScriptStep, ScriptedProvider, SystemPrompt, Tool, ToolCall,
    ToolOutcome, ToolProvider, ToolProviderError, ToolRegistry, TurnControl, TurnCx, TurnOutcome,
    UserMsg,
};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};

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

/// A provider-supplied tool whose name/schema are only known after discovery — it echoes its
/// `text` argument, standing in for any out-of-process tool's proxy.
struct EchoTool;
#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "py_echo"
    }
    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"text":{"type":"string"}}}"#
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let text = serde_json::from_str::<serde_json::Value>(&call.args)
            .ok()
            .and_then(|v| {
                v.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| call.args.clone());
        ToolOutcome::text(call.call_id.clone(), true, text)
    }
}

/// A fake dynamic-tool source: discovery yields the `py_echo` tool, exactly as a real worker
/// client would after handshaking with its process.
struct EchoProvider {
    label: String,
}
#[async_trait]
impl ToolProvider for EchoProvider {
    fn label(&self) -> &str {
        &self.label
    }
    async fn discover(&self) -> Result<Vec<Arc<dyn Tool>>, ToolProviderError> {
        Ok(vec![Arc::new(EchoTool) as Arc<dyn Tool>])
    }
}

async fn dispatch(registry: &ToolRegistry, name: &str, args: &str) -> ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("tool-provider-conformance");
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
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: name.into(),
        args: args.into(),
    };
    run_tool(&call, registry, &cx).await
}

/// A tool obtained through the `ToolProvider` seam registers and dispatches through the real §12
/// `run_tool` pipeline like any native tool.
#[tokio::test]
async fn provider_discovered_tool_dispatches_through_pipeline() {
    let provider: Arc<dyn ToolProvider> = Arc::new(EchoProvider {
        label: "fake".into(),
    });
    assert_eq!(provider.label(), "fake");

    let mut registry = ToolRegistry::new();
    for tool in provider.discover().await.expect("discover provider tools") {
        registry.register(tool);
    }
    assert!(registry.get("py_echo").is_some());

    let out = dispatch(&registry, "py_echo", r#"{"text":"hello"}"#).await;
    assert!(out.result.ok);
    assert_eq!(out.result.content, "hello");
}

/// The engine's ReAct loop calls a provider-discovered tool and records its result in the durable
/// conversation — the seam is transparent end to end.
#[tokio::test]
async fn engine_invokes_provider_discovered_tool() {
    let provider: Arc<dyn ToolProvider> = Arc::new(EchoProvider {
        label: "fake".into(),
    });
    let mut registry = ToolRegistry::new();
    for tool in provider.discover().await.expect("discover provider tools") {
        registry.register(tool);
    }

    let model = Arc::new(ScriptedProvider::new(
        vec![ScriptStep::Call {
            name: "py_echo".into(),
            args: r#"{"text":"via the engine"}"#.into(),
        }],
        "done",
    ));
    let mut engine = Engine::fresh(
        SessionId::new("provider-e2e"),
        SystemPrompt::new("test"),
        model,
        Arc::new(registry),
    )
    .with_config(Config {
        max_iterations: 8,
        ..Config::default()
    });
    engine.push_user(UserMsg::new("go"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes");
    assert!(matches!(outcome, TurnOutcome::Completed(_)));

    let recorded = engine
        .snapshot()
        .conversation
        .turns
        .iter()
        .find_map(|turn| match turn {
            Turn::Tool(t) => t
                .calls
                .iter()
                .find(|(call, _)| call.name == "py_echo")
                .map(|(_, result)| result.clone()),
            _ => None,
        })
        .expect("a py_echo tool turn was recorded");
    assert!(recorded.ok);
    assert_eq!(recorded.content, "via the engine");
}
