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
use daemon_protocol::{HostRequest, HostRequestKind, HostResponseBody};
use std::collections::HashMap;
use std::sync::Arc;

/// The outcome of running one tool: its result slot plus the effects it produced (§12).
pub struct ToolOutcome {
    /// The result slot to pair with the originating call.
    pub result: ToolResult,
    /// The effects the tool produced, applied by the single-owner applier.
    pub effects: Vec<Effect>,
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
        }
    }
}
