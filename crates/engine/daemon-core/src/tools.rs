//! The tool trait, registry, and the one bundled tool (§12).
//!
//! A [`Tool`] is a capability the engine invokes during a turn. It runs against the [`TurnCx`] and
//! returns a [`ToolOutcome`] — a result slot plus the [`Effect`]s its execution produced. The
//! [`ToolRegistry`] resolves a call's name to its handler. Phase 3 ships exactly one real tool,
//! [`DelegateTool`], which exercises the full pipeline: it raises a blocking `HostRequest::Delegate`
//! and yields the durable [`Effect::Delegate`] the engine suspends on.

use crate::conversation::{ToolCall, ToolResult};
use crate::turn::{Effect, TurnCx};
use daemon_common::{JobId, ReqId};
use daemon_protocol::{HostRequest, HostRequestKind, HostResponseBody, ToolDetail};
use std::collections::HashMap;
use std::sync::Arc;

/// The outcome of running one tool: its result slot plus the effects it produced (§12).
pub struct ToolOutcome {
    /// The result slot to pair with the originating call.
    pub result: ToolResult,
    /// The effects the tool produced, applied by the single-owner applier.
    pub effects: Vec<Effect>,
    /// An optional structured payload for a rich consumer (the §17 `ToolResultView::detail`
    /// envelope): the tool's typed output (a diff, a command's exit/stdout, a file listing, ...),
    /// opaque to the daemon and rendered by the GUI per `kind`. `None` for plain-text tools.
    pub detail: Option<ToolDetail>,
    /// Whether the result content came from an external/untrusted source (web/MCP/browser fetch). The
    /// §12 pipeline fences such content with [`wrap_untrusted_tool_result`](crate::repair::wrap_untrusted_tool_result)
    /// before the byte-budget stage so the model reads it as inert data, never as instructions.
    pub untrusted: bool,
}

impl ToolOutcome {
    /// A plain text-only outcome with no effects or structured detail.
    pub fn text(call_id: impl Into<String>, ok: bool, content: impl Into<String>) -> Self {
        Self {
            result: ToolResult {
                call_id: call_id.into(),
                ok,
                content: content.into(),
            },
            effects: Vec::new(),
            detail: None,
            untrusted: false,
        }
    }

    /// A text-only outcome whose content is **untrusted** external data (web/MCP/browser): the §12
    /// pipeline fences it before budgeting. Use this for any tool result derived from a fetched page,
    /// search hit, or other source outside the agent's own trust boundary.
    pub fn untrusted_text(call_id: impl Into<String>, ok: bool, content: impl Into<String>) -> Self {
        let mut out = Self::text(call_id, ok, content);
        out.untrusted = true;
        out
    }

    /// Attach a structured detail envelope to this outcome.
    pub fn with_detail(mut self, detail: ToolDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Attach effects produced by the tool, applied by the single-owner applier (§4.3).
    pub fn with_effects(mut self, effects: Vec<Effect>) -> Self {
        self.effects = effects;
        self
    }

    /// Mark this outcome's content as untrusted external data (the §12 wrap-untrusted stage).
    pub fn mark_untrusted(mut self) -> Self {
        self.untrusted = true;
        self
    }
}

/// A tool's batch-concurrency class (§12). The engine may run a model-emitted tool batch
/// concurrently, but only when **every** call in the batch is [`Parallel`](ToolConcurrency::Parallel)
/// — an all-or-nothing rule that conservatively stands in for hermes' path-overlap analysis
/// (`agent/tool_executor.py`). Any [`Exclusive`](ToolConcurrency::Exclusive) call forces the whole
/// batch to run sequentially.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolConcurrency {
    /// Side-effect-free / read-only: safe to run alongside other parallel calls in the same batch
    /// (e.g. `web_search`, `web_extract`). The tool must not mutate shared state or block on a host
    /// request whose ordering matters.
    Parallel,
    /// Must run alone (the default): the tool mutates state, has ordered side effects, or blocks on
    /// the host (e.g. `shell`, an fs write, `delegate`, `clarify`). One such call serializes the batch.
    Exclusive,
}

/// A registry entry's static description (schemars-generated schema in the real engine).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDef {
    /// The tool's stable name.
    pub name: String,
    /// The tool's argument schema (placeholder JSON-schema string).
    pub schema: String,
}

/// A capability the engine can invoke during a turn (§12).
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The tool's stable name as exposed to the engine.
    fn name(&self) -> &str;
    /// The tool's argument schema.
    fn schema(&self) -> &str;
    /// Execute the call against the turn context.
    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome;
    /// The tool's batch-concurrency class (§12). Defaults to [`ToolConcurrency::Exclusive`] (the
    /// safe, sequential behaviour); a read-only tool overrides this to
    /// [`ToolConcurrency::Parallel`] to opt into concurrent batch execution.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }
}

/// A boxed, thread-safe error from a [`ToolProvider`] (kept opaque so `daemon-core` stays free of
/// any provider's concrete error/runtime types).
pub type ToolProviderError = Box<dyn std::error::Error + Send + Sync>;

/// A source of dynamically-discovered [`Tool`]s — an out-of-process worker, an MCP server, a plugin
/// host, etc. The single discovery seam the host queries at startup (and may re-query) to populate
/// the [`ToolRegistry`] with tools whose names/schemas are only known at runtime.
///
/// This is the shared boundary for every dynamic tool surface: the Python tool worker
/// (`daemon-pytool-client`) implements it today, and a future MCP client implements the same trait,
/// so the host wires `Vec<Arc<dyn ToolProvider>>` uniformly rather than special-casing each source.
/// The provider owns its own process/connection lifecycle (lazy spawn, crash-respawn); `discover`
/// just reports the tools currently on offer.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync {
    /// A short, stable label for diagnostics (e.g. `"python"`, `"mcp:github"`).
    fn label(&self) -> &str;
    /// Discover (or re-discover) the tools this provider currently offers.
    async fn discover(&self) -> Result<Vec<Arc<dyn Tool>>, ToolProviderError>;
}

/// The tool registry: resolves a call name to its handler (§12 `tools.rs`).
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool under its declared name.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_owned(), tool);
    }

    /// Resolve a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// The names of all registered tools (offered to the model each turn).
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// The static descriptions of all registered tools.
    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools
            .values()
            .map(|t| ToolDef {
                name: t.name().to_owned(),
                schema: t.schema().to_owned(),
            })
            .collect()
    }
}

/// The one bundled tool: delegate background work to the host.
///
/// Running it raises a blocking `HostRequest::Delegate`; the host answers with the durable `JobId`
/// the engine will wait on. The tool returns that id as its result and emits [`Effect::Delegate`],
/// which the engine's applier records into `waiting_for` before suspending (§16.2 / lifecycle §3.1).
pub struct DelegateTool {
    label: String,
}

impl DelegateTool {
    /// A delegate tool that labels its delegated work with `label`.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl Default for DelegateTool {
    fn default() -> Self {
        Self::new("background-work")
    }
}

#[async_trait::async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{}}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let req = HostRequest {
            request_id: ReqId(0),
            kind: HostRequestKind::Delegate {
                label: self.label.clone(),
                budget: cx.budget,
            },
        };
        let resp = cx.host.request(req).await;
        let job_id = match resp.body {
            HostResponseBody::Delegated(job) => job,
            _ => JobId::new(format!("{}:unresolved", cx.session_id)),
        };
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: true,
                content: format!("delegated:{job_id}"),
            },
            effects: vec![Effect::Delegate(job_id)],
            detail: None,
            untrusted: false,
        }
    }
}
