// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared unit-test fixtures for the engine turn-loop tests.
//!
//! Consolidates the mock [`Provider`]/[`HostRequestHandler`]/[`Tool`] impls and the engine builders
//! that were duplicated across `engine::tests`, so the per-test boilerplate (identical capability
//! blocks, identical `Engine::fresh(..).with_config(..)` construction) lives in exactly one place.
//! Behavior-preserving: every fixture body is the verbatim move from the test module.

use super::Engine;
use crate::config::Config;
use crate::conversation::{SystemPrompt, ToolCall};
use crate::events::EventSink;
use crate::provider::{
    Capabilities, MockProvider, ModelOutput, Provider, Request, ScriptStep, ScriptedProvider,
    ToolCallFormat,
};
use crate::tools::{Tool, ToolConcurrency, ToolOutcome, ToolRegistry};
use crate::turn::TurnCx;
use crate::Failure;
use daemon_common::{SessionId, UsageDelta};
use daemon_protocol::{
    AgentEvent, HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The capabilities every deterministic test provider advertises (native tools, no streaming, an
/// 8K window). Hoisted so the identical block stops being copied into each mock `Provider`.
pub(super) fn test_caps() -> Capabilities {
    Capabilities {
        supports_native_tools: true,
        supports_streaming: false,
        tool_call_format: ToolCallFormat::Native,
        max_context: Some(8192),
    }
}

/// A plain final-text [`ModelOutput`] (no reasoning, no tool calls, default usage) — the shape every
/// "just complete" mock provider returns.
pub(super) fn ok_output(text: &str) -> ModelOutput {
    ModelOutput {
        text: text.into(),
        reasoning: None,
        tool_calls: Vec::new(),
        usage: UsageDelta::default(),
        ..Default::default()
    }
}

/// The over-budget [`Pressure`](crate::context::Pressure) reading the deterministic
/// `ContextEngine` test doubles return from `before_turn`: a huge `used_tokens` against the host
/// `budget` (defaulting to `1` when unset) so the §10 compaction path always fires. Hoisted so the
/// identical reading stops being copied into each test `ContextEngine`.
pub(super) fn over_budget_pressure(budget: Option<usize>) -> crate::context::Pressure {
    crate::context::Pressure {
        used_tokens: 1_000_000,
        budget_tokens: budget.or(Some(1)),
    }
}

/// A [`ScriptedProvider`] that re-issues the same single tool `name` (`{}` args) every round — the
/// looping model the iteration-budget / no-progress / cancel-mid-loop tests drive.
pub(super) fn looping_call_provider(name: &str) -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::looping(ScriptStep::Call {
        name: name.into(),
        args: "{}".into(),
    }))
}

/// Build a fresh engine for a session `id` over `provider` + `registry` with the shared `"test"`
/// system prompt — the construction the per-tool builders below all share.
pub(super) fn test_engine(id: &str, provider: Arc<dyn Provider>, registry: ToolRegistry) -> Engine {
    Engine::fresh(
        SessionId::new(id),
        SystemPrompt::new("test"),
        provider,
        Arc::new(registry),
    )
}

/// An event sink that records every emitted event for assertions.
pub(super) fn collecting() -> (EventSink, Arc<std::sync::Mutex<Vec<AgentEvent>>>) {
    let log = Arc::new(std::sync::Mutex::new(Vec::<AgentEvent>::new()));
    let l = log.clone();
    (EventSink::new(move |ev| l.lock().unwrap().push(ev)), log)
}

/// A host that approves every request (the default "no operator" host).
pub(super) struct NoopHost;

#[async_trait::async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

/// A provider that always completes with plain final text (no tool calls).
pub(super) struct TextProvider;

#[async_trait::async_trait]
impl Provider for TextProvider {
    fn capabilities(&self) -> Capabilities {
        test_caps()
    }
    async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
        Ok(ok_output("ok"))
    }
}

/// A host that records every spawn `kind` it is asked to materialize.
#[derive(Default)]
pub(super) struct SpawnRecordingHost {
    pub(super) spawns: std::sync::Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl HostRequestHandler for SpawnRecordingHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        if let HostRequestKind::Spawn { spec } = &req.kind {
            self.spawns.lock().unwrap().push(spec.kind.clone());
            return HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Spawned(SessionId::new("child")),
            };
        }
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

/// An engine wired to [`MockProvider::completing`] over an empty registry, for the boundary/steer
/// tests that just need a turn to complete.
pub(super) fn completing_engine(id: &str) -> Engine {
    test_engine(
        id,
        Arc::new(MockProvider::completing("hi")),
        ToolRegistry::new(),
    )
}

/// A trivial in-turn tool that records how many times it ran (shared counter) and returns a
/// fixed result — enough to exercise the model->tools->model loop without a real tool crate.
pub(super) struct CounterTool {
    pub(super) runs: Arc<AtomicU64>,
}

#[async_trait::async_trait]
impl Tool for CounterTool {
    fn name(&self) -> &str {
        "counter"
    }
    fn schema(&self) -> &str {
        "{}"
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let n = self.runs.fetch_add(1, Ordering::Relaxed);
        ToolOutcome::text(call.call_id.clone(), true, format!("counter:{n}"))
    }
}

/// An engine whose only tool is a [`CounterTool`], with a configurable iteration budget.
pub(super) fn looping_engine(
    provider: Arc<dyn Provider>,
    runs: Arc<AtomicU64>,
    max_iterations: u32,
) -> Engine {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CounterTool { runs }));
    test_engine("react", provider, registry).with_config(Config {
        max_iterations,
        ..Config::default()
    })
}

/// A tool that records the *peak* number of concurrent in-flight executions across a batch. By
/// holding a short overlap window in `run`, two genuinely concurrent calls observe `max_seen == 2`
/// while serialized calls only ever observe `1` — a deterministic probe for §12 batch concurrency.
pub(super) struct ProbeTool {
    pub(super) name: &'static str,
    pub(super) concurrency: ToolConcurrency,
    pub(super) active: Arc<AtomicU64>,
    pub(super) max_seen: Arc<AtomicU64>,
}

#[async_trait::async_trait]
impl Tool for ProbeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn schema(&self) -> &str {
        "{}"
    }
    fn concurrency(&self) -> ToolConcurrency {
        self.concurrency
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        let cur = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(cur, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        ToolOutcome::text(call.call_id.clone(), true, "ok")
    }
}

/// Construct a [`ProbeTool`] sharing the batch-wide `active`/`max_seen` counters — the per-tool
/// boilerplate the §12 batch-concurrency tests would otherwise repeat verbatim.
pub(super) fn probe_tool(
    name: &'static str,
    concurrency: ToolConcurrency,
    active: &Arc<AtomicU64>,
    max_seen: &Arc<AtomicU64>,
) -> Arc<ProbeTool> {
    Arc::new(ProbeTool {
        name,
        concurrency,
        active: active.clone(),
        max_seen: max_seen.clone(),
    })
}

/// An engine over an arbitrary set of [`Tool`]s (used to probe §12 batch concurrency).
pub(super) fn probe_engine(provider: Arc<dyn Provider>, tools: Vec<Arc<dyn Tool>>) -> Engine {
    let mut registry = ToolRegistry::new();
    for t in tools {
        registry.register(t);
    }
    test_engine("probe", provider, registry).with_config(Config {
        max_iterations: 8,
        ..Config::default()
    })
}

/// A tool that returns a *fixed* result every run (the inverse of [`CounterTool`], whose result
/// changes each round) — so repeated identical calls yield byte-identical rounds, exercising the
/// §4.2 no-progress guard.
pub(super) struct ConstantTool {
    pub(super) runs: Arc<AtomicU64>,
}

#[async_trait::async_trait]
impl Tool for ConstantTool {
    fn name(&self) -> &str {
        "constant"
    }
    fn schema(&self) -> &str {
        "{}"
    }
    async fn run(&self, call: &ToolCall, _cx: &TurnCx<'_>) -> ToolOutcome {
        self.runs.fetch_add(1, Ordering::Relaxed);
        ToolOutcome::text(call.call_id.clone(), true, "same".to_string())
    }
}

/// An engine whose only tool is a [`ConstantTool`], with configurable iteration + no-progress caps.
pub(super) fn constant_engine(
    provider: Arc<dyn Provider>,
    runs: Arc<AtomicU64>,
    max_iterations: u32,
    max_repeated_rounds: u32,
) -> Engine {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ConstantTool { runs }));
    test_engine("react", provider, registry).with_config(Config {
        max_iterations,
        max_repeated_rounds,
        ..Config::default()
    })
}
