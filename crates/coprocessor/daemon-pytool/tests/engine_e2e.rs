//! End-to-end engine gate: a Python tool, discovered over the supervised client and registered as
//! an ordinary [`Tool`], is invoked by the engine's ReAct loop and its result lands in the durable
//! conversation — exactly as a native Rust tool would. Uses the hermetic `fake-pytool-worker` (no
//! system Python required) behind a real [`PyToolProxy`], driven by a [`ScriptedProvider`].

use std::sync::Arc;
use std::time::Duration;

use daemon_common::SessionId;
use daemon_core::conversation::Turn;
use daemon_core::{
    Config, Engine, EventSink, ScriptStep, ScriptedProvider, SystemPrompt, ToolRegistry,
    TurnControl, TurnOutcome, UserMsg,
};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_pytool_client::{discover, PyToolConfig};

/// A host that approves everything — `py_echo` raises no host requests, so this is never exercised,
/// but `run_turn` requires a handler.
struct NoopHost;

#[async_trait::async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

#[tokio::test]
async fn engine_invokes_a_discovered_python_tool() {
    // Discover the worker's tools and register the `py_echo` proxy into a real ToolRegistry.
    let mut cfg = PyToolConfig::new(env!("CARGO_BIN_EXE_fake-pytool-worker"), Vec::new());
    cfg.op_timeout = Duration::from_secs(5);
    cfg.spawn_timeout = Duration::from_secs(5);
    let tools = discover(cfg).await.expect("discover python tools");
    assert!(tools.iter().any(|t| t.name() == "py_echo"));

    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(tool);
    }

    // The model calls `py_echo` once, then returns final text.
    let provider = Arc::new(ScriptedProvider::new(
        vec![ScriptStep::Call {
            name: "py_echo".into(),
            args: r#"{"text":"from the engine"}"#.into(),
        }],
        "done",
    ));

    let mut engine = Engine::fresh(
        SessionId::new("pytool-e2e"),
        SystemPrompt::new("test"),
        provider,
        Arc::new(registry),
    )
    .with_config(Config {
        max_iterations: 8,
        ..Config::default()
    });
    engine.push_user(UserMsg::new("echo something"));

    let outcome = engine
        .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
        .await
        .expect("turn completes");
    assert!(matches!(outcome, TurnOutcome::Completed(_)));

    // The Python tool's result is recorded in the durable conversation, routed back through the
    // engine exactly like a native tool.
    let snapshot = engine.snapshot();
    let echoed = snapshot
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
    assert!(echoed.ok);
    assert_eq!(echoed.content, "from the engine");
}
