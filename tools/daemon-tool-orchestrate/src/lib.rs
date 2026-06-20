//! `daemon-tool-orchestrate` — the agent veneer over the fleet runtime (layout §4: tool surface).
//!
//! Exposes orchestration to the engine as a single `daemon_core::Tool` so the brain can grow/steer a
//! fleet by policy. This is the explicit DOWN edge of the orchestration flow: the agent *calls* the
//! tool, the tool records the delegation intent through the §17 host port (yielding the durable
//! `JobId` the engine suspends on), and the fleet runtime — not the tool — spawns and drives the
//! child when the resulting job is processed. The `status`/`cancel` verbs read/poke live fleet state.
//!
//! The fleet machinery itself lives in `daemon-orchestration`; this crate is a thin handle onto it.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::{JobId, ReqId, UnitId};
use daemon_core::{Effect, Tool, ToolCall, ToolOutcome, ToolResult, TurnCx};
use daemon_orchestration::FleetRuntime;
use daemon_protocol::{HostRequest, HostRequestKind, HostResponseBody};

/// The verbs the agent can invoke through the orchestrate tool.
enum Verb {
    /// Grow the fleet: delegate background work to a new child (the default).
    Delegate,
    /// Observe live fleet state.
    Status,
    /// Cancel a registered child by id.
    Cancel(String),
}

/// Parse a verb from the tool-call args. Minimal by design (no JSON dep): a bare verb word, with
/// `cancel:<unit-id>` carrying its target. Anything unrecognized (including the empty `{}` the mock
/// provider emits) is a `delegate`.
fn parse_verb(args: &str) -> Verb {
    let trimmed = args.trim();
    if let Some(rest) = trimmed.strip_prefix("cancel:") {
        Verb::Cancel(rest.trim().to_owned())
    } else if trimmed == "status" {
        Verb::Status
    } else {
        Verb::Delegate
    }
}

/// The default ceiling on nested delegation depth before the tool stops delegating (and the engine
/// completes instead). The durable delegation graph is recursive — a child is itself an
/// orchestrator-capable engine that can delegate — so without a guard a model that always delegates
/// would spawn an unbounded chain of sessions. The depth is read from the session id (the durable
/// child minter encodes the tree path with `/`), so it needs no extra protocol field.
const DEFAULT_MAX_DEPTH: usize = 8;

/// The agent's handle onto a node's [`FleetRuntime`].
pub struct OrchestrateTool {
    fleet: FleetRuntime,
    label: String,
    max_depth: usize,
}

/// The nesting depth a durable session sits at, derived from its id: the top session is depth 0 and
/// each delegated child appends a `/c{epoch}` path segment, so the depth is the segment count.
fn session_depth(session_id: &str) -> usize {
    session_id.matches('/').count()
}

impl OrchestrateTool {
    /// A tool over `fleet`, labelling delegated work with a default label.
    pub fn new(fleet: FleetRuntime) -> Self {
        Self {
            fleet,
            label: "orchestrated-work".into(),
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// A tool that labels its delegated work with `label`.
    pub fn with_label(fleet: FleetRuntime, label: impl Into<String>) -> Self {
        Self {
            fleet,
            label: label.into(),
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Cap nested delegation at `max_depth` levels: a session already at or below the cap completes
    /// instead of delegating, so the recursive durable delegation chain terminates.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    fn ok(call: &ToolCall, content: String, effects: Vec<Effect>) -> ToolOutcome {
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: true,
                content,
            },
            effects,
        }
    }
}

#[async_trait]
impl Tool for OrchestrateTool {
    fn name(&self) -> &str {
        "orchestrate"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"verb":{"type":"string","enum":["delegate","status","cancel"]}}}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        match parse_verb(&call.args) {
            // Depth guard: at the cap, stop delegating and let the turn complete — this is what
            // terminates the recursive durable delegation chain (every child is itself
            // orchestrator-capable, so an always-delegating model would otherwise nest forever).
            Verb::Delegate if session_depth(cx.session_id.as_str()) >= self.max_depth => Self::ok(
                call,
                format!("depth-limit:{}", self.max_depth),
                Vec::new(),
            ),
            // DOWN edge: record the delegation through the host port; the fleet worker spawns the
            // child when the resulting durable job is processed. Emitting Effect::Delegate suspends
            // the parent until the child's completion wakes it (lifecycle §3.1).
            Verb::Delegate => {
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
                Self::ok(
                    call,
                    format!("delegated:{job_id}"),
                    vec![Effect::Delegate(job_id)],
                )
            }
            Verb::Status => {
                let children = self.fleet.children();
                let usage = self.fleet.fleet_usage();
                Self::ok(
                    call,
                    format!(
                        "fleet: {} children, {} api_calls",
                        children.len(),
                        usage.api_calls
                    ),
                    Vec::new(),
                )
            }
            Verb::Cancel(id) => {
                let cancelled = self.fleet.cancel_child(&UnitId::new(id.clone())).await;
                Self::ok(call, format!("cancel:{id}:{cancelled}"), Vec::new())
            }
        }
    }
}
