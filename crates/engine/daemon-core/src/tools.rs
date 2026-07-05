// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

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
    pub fn untrusted_text(
        call_id: impl Into<String>,
        ok: bool,
        content: impl Into<String>,
    ) -> Self {
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

    /// The **per-call** batch-concurrency class (§12): the argument-aware refinement of
    /// [`concurrency`](Tool::concurrency). A tool whose safety depends on its *arguments* — e.g. an
    /// `fs` tool that is [`Parallel`](ToolConcurrency::Parallel) for a `read`/`grep`/`glob` op but
    /// [`Exclusive`](ToolConcurrency::Exclusive) for a `write`/`edit` — overrides this. Defaults to
    /// the call-independent [`concurrency`](Tool::concurrency), so existing tools are unchanged.
    fn concurrency_for(&self, _call: &ToolCall) -> ToolConcurrency {
        self.concurrency()
    }

    /// The workspace paths this call reads/writes, for the batch path-overlap gate (hermes
    /// `_should_parallelize_tool_batch` parity). `None` = no declared path scope (a read-only tool
    /// like `web_search`: freely parallel). `Some(paths)` = path-scoped: the engine serializes the
    /// call against any other path-scoped call whose paths overlap (prefix-subtree). Defaults to
    /// `None`, so a tool that does not opt in is treated as unscoped (today's all-or-nothing rule).
    fn parallel_scope_paths(&self, _call: &ToolCall) -> Option<Vec<PathBuf>> {
        None
    }

    /// The **per-call** effective wall-clock timeout for the §12 pipeline's timeout stage. Receives
    /// the engine's configured `default` (`None` when the timeout is disabled) and returns the
    /// timeout to apply, or `None` to opt out entirely. A self-limiting tool (a `shell` foreground
    /// command, a long `execute_code` run) overrides this to `None` so it manages its own deadline.
    /// Defaults to the engine `default`, so no tool is affected until the host sets a default.
    fn call_timeout(&self, _call: &ToolCall, default: Option<Duration>) -> Option<Duration> {
        default
    }

    /// Whether this tool belongs to the **deferrable** (dynamic / long-tail) set rather than the
    /// always-offered core. Deferrable tools (MCP + Python proxies) are hidden behind the
    /// `tool_search` bridge once their summed schema exceeds the engine's threshold, so a large
    /// external tool surface never blows the prompt budget. Built-in tools default to `false` (core).
    fn deferrable(&self) -> bool {
        false
    }

    /// Whether this tool **mutates** the workspace (an fs write/edit or a shell command). The §12
    /// pipeline records a [`CheckpointStore`](crate::checkpoint::CheckpointStore) checkpoint before a
    /// mutating tool runs, so an operator can rewind. Defaults to `false` (read-only / pure tools);
    /// `fs` write/edit and `shell` opt in. Distinct from [`concurrency`](Tool::concurrency), which is
    /// about batch parallelism.
    fn mutates(&self) -> bool {
        false
    }

    /// The **per-call** mutation predicate: the argument-aware refinement of
    /// [`mutates`](Tool::mutates). An `fs` tool that mutates on `write`/`edit`/`delete` but not on
    /// `read`/`grep`/`glob` overrides this so the §12 checkpoint stage only fires for the mutating
    /// ops; the per-turn guardrail also uses it to classify a call as idempotent (read-only) vs
    /// mutating. Defaults to the call-independent [`mutates`](Tool::mutates), so behaviour is
    /// byte-identical until a tool opts in.
    fn mutates_for(&self, _call: &ToolCall) -> bool {
        self.mutates()
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

/// The `tool_search` bridge name: takes `{ "query": "..." }`, returns matching deferrable tools.
pub const TOOL_SEARCH: &str = "tool_search";
/// The `tool_describe` bridge name: takes `{ "name": "..." }`, returns that tool's full schema.
pub const TOOL_DESCRIBE: &str = "tool_describe";
/// The `tool_call` bridge name: takes `{ "name": "...", "arguments": { ... } }`, invokes a
/// deferrable tool by name (the indirection the model uses once the long tail is collapsed).
pub const TOOL_CALL: &str = "tool_call";

/// The tool registry: resolves a call name to its handler (§12 `tools.rs`).
///
/// Tools are split into a **core** set (always offered to the model) and a **deferrable** set (the
/// dynamic long tail — MCP + Python proxies, classified by [`Tool::deferrable`]). When the deferrable
/// schema is small it is offered inline; once it exceeds the engine's threshold the registry offers
/// the `tool_search`/`tool_describe`/`tool_call` bridge instead (progressive disclosure). Resolution
/// ([`get`](ToolRegistry::get)) always spans both sets, so a collapsed tool is still callable.
#[derive(Default)]
pub struct ToolRegistry {
    core: HashMap<String, Arc<dyn Tool>>,
    deferrable: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            core: HashMap::new(),
            deferrable: HashMap::new(),
        }
    }

    /// Register a tool under its declared name, routing it to the core or deferrable set per
    /// [`Tool::deferrable`]. Existing call sites are unaffected — a built-in lands in core as before.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        if tool.deferrable() {
            self.deferrable.insert(tool.name().to_owned(), tool);
        } else {
            self.core.insert(tool.name().to_owned(), tool);
        }
    }

    /// Force-register a tool into the deferrable set regardless of its [`Tool::deferrable`] hint.
    pub fn register_deferrable(&mut self, tool: Arc<dyn Tool>) {
        self.deferrable.insert(tool.name().to_owned(), tool);
    }

    /// Resolve a tool by name across both the core and deferrable sets.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.core
            .get(name)
            .or_else(|| self.deferrable.get(name))
            .cloned()
    }

    /// The names of all registered tools (core + deferrable).
    pub fn names(&self) -> Vec<String> {
        self.core
            .keys()
            .chain(self.deferrable.keys())
            .cloned()
            .collect()
    }

    /// The static descriptions of all registered tools (core + deferrable).
    pub fn defs(&self) -> Vec<ToolDef> {
        self.core
            .values()
            .chain(self.deferrable.values())
            .map(def_of)
            .collect()
    }

    /// The core (always-offered) tool descriptions.
    pub fn core_defs(&self) -> Vec<ToolDef> {
        self.core.values().map(def_of).collect()
    }

    /// The deferrable (long-tail) tool descriptions.
    pub fn deferrable_defs(&self) -> Vec<ToolDef> {
        self.deferrable.values().map(def_of).collect()
    }

    /// The summed byte length of every deferrable tool's schema — the quantity compared against the
    /// engine's `tool_search_threshold_bytes`.
    pub fn deferrable_schema_bytes(&self) -> usize {
        self.deferrable.values().map(|t| t.schema().len()).sum()
    }

    /// Whether there are any deferrable tools registered.
    pub fn has_deferrable(&self) -> bool {
        !self.deferrable.is_empty()
    }

    /// The static descriptions of the three bridge tools (offered when the long tail is collapsed).
    pub fn bridge_defs(&self) -> Vec<ToolDef> {
        vec![
            ToolDef {
                name: TOOL_SEARCH.to_owned(),
                schema: r#"{"type":"object","properties":{"query":{"type":"string","description":"keywords to find a tool by name or schema"}},"required":["query"]}"#.to_owned(),
            },
            ToolDef {
                name: TOOL_DESCRIBE.to_owned(),
                schema: r#"{"type":"object","properties":{"name":{"type":"string","description":"the exact tool name to describe"}},"required":["name"]}"#.to_owned(),
            },
            ToolDef {
                name: TOOL_CALL.to_owned(),
                schema: r#"{"type":"object","properties":{"name":{"type":"string","description":"the exact tool name to invoke"},"arguments":{"type":"object","description":"the tool's arguments"}},"required":["name"]}"#.to_owned(),
            },
        ]
    }

    /// The tool descriptions offered to the model this turn: always the core set, plus *either* every
    /// deferrable schema (when small / threshold disabled) *or* the three bridge tools (when the
    /// deferrable schema exceeds `threshold_bytes`). Progressive disclosure for tools.
    pub fn offered_defs(&self, threshold_bytes: usize) -> Vec<ToolDef> {
        let mut defs = self.core_defs();
        let collapse = threshold_bytes > 0
            && self.has_deferrable()
            && self.deferrable_schema_bytes() > threshold_bytes;
        if collapse {
            defs.extend(self.bridge_defs());
        } else {
            defs.extend(self.deferrable_defs());
        }
        defs
    }
}

/// The static description of a tool.
fn def_of(t: &Arc<dyn Tool>) -> ToolDef {
    ToolDef {
        name: t.name().to_owned(),
        schema: t.schema().to_owned(),
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
        // Carry the task as the structured job payload (no attachments from this core tool); the
        // node-side worker decodes it to seed the child. Replaces the historical fixed marker.
        let payload = daemon_protocol::DelegationInput::task(self.label.clone()).encode();
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: true,
                content: format!("delegated:{job_id}"),
            },
            effects: vec![Effect::Delegate {
                job: job_id,
                payload,
            }],
            detail: None,
            untrusted: false,
        }
    }
}
