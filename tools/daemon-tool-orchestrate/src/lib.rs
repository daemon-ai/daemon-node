// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-orchestrate` — the agent veneer over the fleet runtime (layout §4: tool surface).
//!
//! Exposes orchestration to the engine as a single `daemon_core::Tool` so the brain can grow/steer a
//! fleet by policy. The verbs:
//!
//! - `spawn` (default; `delegate` is a back-compat alias): the explicit DOWN edge of the
//!   orchestration flow. The agent supplies the child's `task` (plus optional `lifetime`, `profile`,
//!   `attachments`); the tool records the delegation intent through the §17 host port (yielding the
//!   durable `JobId` the engine suspends on), and the fleet runtime — not the tool — spawns and
//!   drives the child when the resulting job is processed.
//! - `send`: deliver a follow-up message into an existing child session (parent-owns-subtree
//!   authorization): the text is queued on the durable pending-input seam and the child is woken,
//!   so the next incarnation folds it into the conversation (Steer-equivalent).
//! - `status`: per-child state lines sourced from the durable session graph (falling back to the
//!   legacy fleet-wide counts when no store is wired).
//! - `cancel`: cancel a registered child by id (the live fleet path).
//!
//! The fleet machinery itself lives in `daemon-orchestration`; this crate is a thin handle onto it.

#![forbid(unsafe_code)]

use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{JobId, ReqId, SessionId, UnitId};
use daemon_core::{Effect, Tool, ToolCall, ToolConcurrency, ToolOutcome, ToolResult, TurnCx};
use daemon_orchestration::FleetRuntime;
use daemon_protocol::{
    DelegationLifetime, HostRequest, HostRequestKind, HostResponseBody, ToolDetail, UserMsg,
};
use daemon_store::{ChildLifetime, SessionStatus};

/// Which guardrail declined a spawn (the structured `kind` inside the `guardrail` tool detail).
#[derive(Clone, Copy, Debug)]
enum GuardrailKind {
    /// The delegation-tree depth cap.
    Depth,
    /// The concurrent detached-children cap.
    Fanout,
}

/// Map the protocol-level [`DelegationLifetime`] onto the store's [`ChildLifetime`] (the source of
/// truth for the materialized child's [`SessionRole`](daemon_store::SessionRole)). Kept local so the
/// tool constructs the detached [`JobCommand`](daemon_store::JobCommand) directly (bypassing the host
/// port, which the joining path uses for the `JobId`).
fn map_lifetime(lifetime: DelegationLifetime) -> ChildLifetime {
    match lifetime {
        DelegationLifetime::Persistent => ChildLifetime::Persistent,
        DelegationLifetime::Ephemeral => ChildLifetime::Ephemeral,
    }
}

/// The verbs the agent can invoke through the orchestrate tool.
enum Verb {
    /// Grow the fleet: delegate background work to a new child (the default).
    Spawn {
        /// The child's task instruction (`None` falls back to the tool's static label — the
        /// back-compat path for the bare `{}` the mock provider emits).
        task: Option<String>,
        /// The declared child lifetime (managed vs transient subagent).
        lifetime: DelegationLifetime,
        /// The named profile the child's engine resolves from (`None` = the default engine shape).
        profile: Option<String>,
        /// Parent-workspace-relative paths to hand down into the child's `inbox/`.
        attachments: Vec<String>,
        /// Whether this turn **blocks** on the child (`true`, the default — the joining
        /// `Effect::Delegate` path that suspends the parent until the child finishes and returns its
        /// result inline) or runs the child **detached** in the background (`false` — the parent's
        /// turn continues and a completion notice arrives later as a fresh reactive turn).
        wait: bool,
    },
    /// Deliver a follow-up message into an existing child session (Steer-equivalent).
    Send {
        /// The child session/unit id to deliver into.
        target: String,
        /// The message text.
        text: String,
    },
    /// Observe per-child state (optionally filtered to one child).
    Status {
        /// A single child to report on (`None` = every direct child).
        target: Option<String>,
    },
    /// Cancel a registered child by id.
    Cancel(String),
}

/// Parse the tool-call args into a [`Verb`], or a human-readable validation error. Prefers the
/// JSON object shape a real model emits; falls back to the minimal word forms (`status`,
/// `cancel:<id>`) and treats anything else — including the bare `{}` the mock provider emits — as a
/// `spawn` with no explicit task (the label fallback).
fn parse_args(args: &str) -> Result<Verb, String> {
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(args) else {
        return Ok(parse_word(args));
    };
    let str_field = |key: &str| -> Option<String> {
        map.get(key)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    // An absent verb is a spawn; `delegate` is the pre-v2 alias.
    let verb = str_field("verb").unwrap_or_else(|| "spawn".to_owned());
    match verb.as_str() {
        "spawn" | "delegate" => {
            let lifetime = match str_field("lifetime").as_deref() {
                None | Some("persistent") => DelegationLifetime::Persistent,
                Some("ephemeral") => DelegationLifetime::Ephemeral,
                Some(other) => {
                    return Err(format!(
                        "unknown lifetime `{other}` (expected `persistent` or `ephemeral`)"
                    ))
                }
            };
            let attachments = map
                .get("attachments")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            // `wait` defaults to true (the joining, current behavior). Accept a JSON bool, or a
            // "true"/"false" string defensively (a model that stringifies the flag).
            let wait = match map.get("wait") {
                None => true,
                Some(serde_json::Value::Bool(b)) => *b,
                Some(serde_json::Value::String(s)) => match s.trim().to_lowercase().as_str() {
                    "false" => false,
                    "true" => true,
                    other => return Err(format!("wait must be a boolean (got `{other}`)")),
                },
                Some(other) => return Err(format!("wait must be a boolean (got `{other}`)")),
            };
            Ok(Verb::Spawn {
                task: str_field("task"),
                lifetime,
                profile: str_field("profile"),
                attachments,
                wait,
            })
        }
        "send" => {
            let target = str_field("target").ok_or("send requires a `target` child id")?;
            let text = str_field("task").ok_or("send requires the message text in `task`")?;
            Ok(Verb::Send { target, text })
        }
        "status" => Ok(Verb::Status {
            target: str_field("target"),
        }),
        "cancel" => Ok(Verb::Cancel(
            str_field("target").ok_or("cancel requires a `target` child id")?,
        )),
        other => Err(format!(
            "unknown verb `{other}` (expected spawn|send|status|cancel)"
        )),
    }
}

/// The minimal non-JSON word forms: `status`, `cancel:<unit-id>`, and everything else (including
/// the empty `{}` the mock provider emits, which fails JSON-object extraction upstream only for
/// non-objects — a bare word lands here) is a `spawn` with no explicit task.
fn parse_word(args: &str) -> Verb {
    let trimmed = args.trim();
    if let Some(rest) = trimmed.strip_prefix("cancel:") {
        Verb::Cancel(rest.trim().to_owned())
    } else if trimmed == "status" {
        Verb::Status { target: None }
    } else {
        Verb::Spawn {
            task: None,
            lifetime: DelegationLifetime::Persistent,
            profile: None,
            attachments: Vec::new(),
            wait: true,
        }
    }
}

/// The default ceiling on nested delegation depth before the tool stops delegating (and the engine
/// completes instead). The durable delegation graph is recursive — a child is itself an
/// orchestrator-capable engine that can delegate — so without a guard a model that always delegates
/// would spawn an unbounded chain of sessions. The depth is read from the session id (the durable
/// child minter encodes the tree path with `/`), so it needs no extra protocol field.
const DEFAULT_MAX_DEPTH: usize = 8;

/// The bound on the ancestor walk `send`'s subtree-authorization check performs (defense against a
/// pathological/cyclic parent chain in the meta rows; the id-prefix fast path covers the common
/// case without any walk).
const MAX_LINEAGE_WALK: usize = 16;

/// The default ceiling on a parent's concurrently-active **detached** children (`spawn wait:false`).
/// Detached spawns do not suspend the parent, so — unlike joining delegation, which is naturally
/// serialized by the parent's suspension — a fan-out loop could otherwise mint unbounded background
/// children. At the cap the tool declines with `fanout-limit:<n>` (mirroring the depth guard),
/// counting active children via the durable child index.
const DEFAULT_MAX_FANOUT: usize = 8;

/// The agent's handle onto a node's [`FleetRuntime`] (+ optionally the durable session graph).
pub struct OrchestrateTool {
    fleet: FleetRuntime,
    label: String,
    max_depth: usize,
    /// The ceiling on a parent's concurrently-active detached (`spawn wait:false`) children.
    max_fanout: usize,
    /// The durable session store backing the `send` + per-child `status` verbs. `None` (tests /
    /// legacy assemblies) keeps `status` on the fleet-wide counts and makes `send` unavailable.
    store: Option<Arc<dyn daemon_store::SessionStore>>,
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
            max_fanout: DEFAULT_MAX_FANOUT,
            store: None,
        }
    }

    /// A tool that labels its delegated work with `label`.
    pub fn with_label(fleet: FleetRuntime, label: impl Into<String>) -> Self {
        Self {
            fleet,
            label: label.into(),
            max_depth: DEFAULT_MAX_DEPTH,
            max_fanout: DEFAULT_MAX_FANOUT,
            store: None,
        }
    }

    /// Cap nested delegation at `max_depth` levels: a session already at or below the cap completes
    /// instead of delegating, so the recursive durable delegation chain terminates.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Cap a parent's concurrently-active detached (`spawn wait:false`) children at `max_fanout`.
    pub fn with_max_fanout(mut self, max_fanout: usize) -> Self {
        self.max_fanout = max_fanout;
        self
    }

    /// The count of a parent's currently-active children (any not-yet-`Completed` child in the
    /// durable child index) — the fan-out guard's measure. A just-enqueued detached child (no session
    /// row yet, so `status` is `None`) counts as active, so a within-turn fan-out loop is bounded; a
    /// `Completed` child does not, so a parent may spawn again once earlier children finish. `0` when
    /// no store is wired.
    async fn active_child_count(&self, parent: &SessionId) -> usize {
        let Some(store) = &self.store else {
            return 0;
        };
        let mut active = 0usize;
        for child in store.children_of(parent).await {
            if store.status(&child).await != Some(SessionStatus::Completed) {
                active += 1;
            }
        }
        active
    }

    /// Give the tool the durable session store so `status` reports per-child state from the durable
    /// graph and `send` can queue input into (and wake) a child session.
    pub fn with_store(mut self, store: Arc<dyn daemon_store::SessionStore>) -> Self {
        self.store = Some(store);
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
            detail: None,
            untrusted: false,
        }
    }

    /// A guardrail decline (wire v29): `ok: true` with the human-readable `content` (the turn
    /// keeps flowing — a decline is a normal outcome, not a tool failure), PLUS a structured
    /// [`ToolDetail`] (`kind = "guardrail"`, JSON `{ "kind": "depth"|"fanout", "limit": N,
    /// "reason": … }` body — the same JSON-body convention the `shell`/`todo` details use) so a
    /// rich client can render the cap without parsing the `depth-limit:N` / `fanout-limit:N`
    /// string.
    fn guardrail(call: &ToolCall, kind: GuardrailKind, limit: usize) -> ToolOutcome {
        let (tag, reason) = match kind {
            GuardrailKind::Depth => (
                "depth",
                format!(
                    "delegation-tree depth cap reached ({limit}); complete this branch instead \
                     of nesting deeper"
                ),
            ),
            GuardrailKind::Fanout => (
                "fanout",
                format!(
                    "concurrent detached-children cap reached ({limit}); wait for a running \
                     child to finish before spawning another"
                ),
            ),
        };
        let body = serde_json::to_vec(&serde_json::json!({
            "kind": tag,
            "limit": limit,
            "reason": reason,
        }))
        .unwrap_or_default();
        let content = match kind {
            GuardrailKind::Depth => format!("depth-limit:{limit}"),
            GuardrailKind::Fanout => format!("fanout-limit:{limit}"),
        };
        let mut outcome = Self::ok(call, content, Vec::new());
        outcome.detail = Some(ToolDetail::new("guardrail", body));
        outcome
    }

    fn err(call: &ToolCall, content: String) -> ToolOutcome {
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: false,
                content,
            },
            effects: Vec::new(),
            detail: None,
            untrusted: false,
        }
    }

    /// Whether `target` sits in the subtree `parent` owns. Fast path: the durable child minter
    /// encodes lineage in the id (`{parent}/c{epoch}[/c…]`), so a `{parent}/` prefix proves
    /// descent. Fallback: walk the durable `SessionMeta.parent` chain upward (bounded), so a child
    /// whose id does not embed the caller (e.g. a re-parented session) still authorizes.
    async fn owns_subtree(&self, parent: &SessionId, target: &SessionId) -> bool {
        if target
            .as_str()
            .starts_with(&format!("{}/", parent.as_str()))
        {
            return true;
        }
        let Some(store) = &self.store else {
            return false;
        };
        let mut cursor = target.clone();
        for _ in 0..MAX_LINEAGE_WALK {
            let Some(meta) = store.session_meta(&cursor).await else {
                return false;
            };
            match meta.parent {
                Some(p) if p == *parent => return true,
                Some(p) => cursor = p,
                None => return false,
            }
        }
        false
    }

    /// Render one per-child status line from the durable graph: id, role, state, resolved profile
    /// (when bound), and the child's title (its seeded task) as the work summary.
    async fn status_line(store: &Arc<dyn daemon_store::SessionStore>, child: &SessionId) -> String {
        let state = match store.status(child).await {
            Some(daemon_store::SessionStatus::Active) => "running",
            Some(daemon_store::SessionStatus::Suspended { .. }) => "suspended",
            Some(daemon_store::SessionStatus::Ready) => "ready",
            Some(daemon_store::SessionStatus::Completed) => "completed",
            None => "unknown",
        };
        let meta = store.session_meta(child).await.unwrap_or_default();
        let role = match meta.role {
            Some(daemon_store::SessionRole::EphemeralSubagent) => "ephemeral",
            Some(daemon_store::SessionRole::ManagedChild) => "managed",
            _ => "child",
        };
        let mut line = format!("{} [{role}] {state}", child.as_str());
        if let Some(profile) = &meta.bound_profile {
            line.push_str(&format!(" profile={}", profile.as_str()));
        }
        if meta.archived {
            line.push_str(" archived");
        }
        if let Some(title) = &meta.title {
            line.push_str(&format!(" — {title}"));
        }
        line
    }
}

#[async_trait]
impl Tool for OrchestrateTool {
    fn name(&self) -> &str {
        "orchestrate"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{"verb":{"type":"string","enum":["spawn","send","status","cancel"],"description":"spawn a child (default), send a follow-up message to one, report per-child status, or cancel one"},"task":{"type":"string","description":"spawn: the instruction seeding the child's first turn. send: the message text to deliver."},"lifetime":{"type":"string","enum":["persistent","ephemeral"],"description":"spawn: persistent = long-lived managed child (default); ephemeral = transient subagent, archived after it completes"},"profile":{"type":"string","description":"spawn: named profile the child runs under (omit for the default engine shape)"},"attachments":{"type":"array","items":{"type":"string"},"description":"spawn: parent-workspace-relative paths handed to the child's inbox/"},"wait":{"type":"boolean","description":"spawn: true (default) blocks this turn until the child finishes and returns its result inline; false runs the child in the background and delivers a completion notification later, letting this turn continue and fan out more subagents"},"target":{"type":"string","description":"send/cancel: the child id (required). status: optional filter to one child."}}}"#
    }

    /// Per-call batch-concurrency class: `status` is a read-only projection and may run alongside
    /// other parallel calls; `spawn`/`send`/`cancel` mutate orchestration state and serialize.
    fn concurrency_for(&self, call: &ToolCall) -> ToolConcurrency {
        match parse_args(&call.args) {
            Ok(Verb::Status { .. }) => ToolConcurrency::Parallel,
            _ => ToolConcurrency::Exclusive,
        }
    }

    /// Per-call mutation predicate: everything but `status` changes orchestration state.
    fn mutates_for(&self, call: &ToolCall) -> bool {
        !matches!(parse_args(&call.args), Ok(Verb::Status { .. }))
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let verb = match parse_args(&call.args) {
            Ok(verb) => verb,
            Err(reason) => return Self::err(call, format!("invalid orchestrate call: {reason}")),
        };
        match verb {
            // Depth guard: at the cap, stop delegating and let the turn complete — this is what
            // terminates the recursive durable delegation chain (every child is itself
            // orchestrator-capable, so an always-delegating model would otherwise nest forever).
            Verb::Spawn { .. } if session_depth(cx.session_id.as_str()) >= self.max_depth => {
                Self::guardrail(call, GuardrailKind::Depth, self.max_depth)
            }
            // DOWN edge (joining): record the delegation through the host port; the fleet worker
            // spawns the child when the resulting durable job is processed. Emitting Effect::Delegate
            // suspends the parent until the child's completion wakes it (lifecycle §3.1).
            Verb::Spawn {
                task,
                lifetime,
                profile,
                attachments,
                wait: true,
            } => {
                // The agent's task instruction; the tool's static label is the back-compat
                // fallback for callers that pass none (the mock provider's bare `{}`).
                let task = task.unwrap_or_else(|| self.label.clone());
                let req = HostRequest {
                    request_id: ReqId(0),
                    kind: HostRequestKind::Delegate {
                        label: task.clone(),
                        budget: cx.budget,
                    },
                };
                let resp = cx.host.request(req).await;
                let job_id = match resp.body {
                    HostResponseBody::Delegated(job) => job,
                    _ => JobId::new(format!("{}:unresolved", cx.session_id)),
                };
                // Carry the task + attachment paths + declared lifetime + profile as the structured
                // job payload; the node-side worker seeds the child from `task`, materializes
                // `attachments` into its inbox/, derives its role from `lifetime`, and binds
                // `profile` for the durable resolver.
                let payload = daemon_protocol::DelegationInput {
                    task,
                    attachments,
                    lifetime,
                    profile,
                    detached: false,
                }
                .encode();
                // A structured `delegation-spawn` detail (opaque kind+body, no wire bump) a rich
                // client renders as a spawn card. The joining child's session id is minted
                // node-side (`{parent}/c{epoch}`) only when the durable Delegate job is processed,
                // so it is not knowable here — the detail carries the `job` handle, correlatable to
                // the child later via the children/roster surfaces.
                let body = serde_json::to_vec(&serde_json::json!({
                    "job": job_id.as_str(),
                    "detached": false,
                }))
                .unwrap_or_default();
                let mut outcome = Self::ok(
                    call,
                    format!("spawned:{job_id}"),
                    vec![Effect::Delegate {
                        job: job_id,
                        payload,
                    }],
                );
                outcome.detail = Some(ToolDetail::new("delegation-spawn", body));
                outcome
            }
            // DETACHED (non-joining): enqueue the child directly onto the durable job outbox — no
            // Effect::Delegate, so the parent's turn does NOT suspend and keeps running. The child's
            // terminal completion arrives later as a fresh reactive turn (a completion notice). This
            // is Cursor's `run_in_background: true` subagent analogue.
            Verb::Spawn {
                task,
                lifetime,
                profile,
                attachments,
                wait: false,
            } => {
                let Some(store) = &self.store else {
                    return Self::err(
                        call,
                        "detached spawn requires a durable session store".into(),
                    );
                };
                // Fan-out cap: a detached spawn does not suspend the parent, so a fan-out loop could
                // mint unbounded background children. Decline at the cap (mirroring the depth guard).
                let active = self.active_child_count(&cx.session_id).await;
                if active >= self.max_fanout {
                    return Self::guardrail(call, GuardrailKind::Fanout, self.max_fanout);
                }
                let task = task.unwrap_or_else(|| self.label.clone());
                let payload = daemon_protocol::DelegationInput {
                    task,
                    attachments,
                    lifetime,
                    profile,
                    detached: true,
                }
                .encode();
                // The store mints the unique `{parent}/dN` child id, stamps it onto the bare job, and
                // enqueues it (no checkpoint/suspension). Then record the completion-notice edge so
                // the child's terminal `mark_completed` delivers a notice to this parent (and the
                // child shows up in `status`/tree immediately).
                let job = daemon_store::JobCommand {
                    job_id: JobId::new(format!("{}:detached", cx.session_id)),
                    session_id: cx.session_id.clone(),
                    epoch: daemon_common::Epoch::ZERO,
                    payload,
                    lifetime: map_lifetime(lifetime),
                    child: None,
                };
                let child = match store.enqueue_detached_job(job).await {
                    Ok(child) => child,
                    Err(e) => return Self::err(call, format!("detached spawn failed: {e}")),
                };
                // Stamp the spawning tool call onto the edge (wire v29): the eventual completion
                // notice carries it, so a client chip-links the injected turn to THIS call's card.
                if let Err(e) = store
                    .bind_completion_notice(&child, &cx.session_id, Some(call.call_id.clone()))
                    .await
                {
                    return Self::err(call, format!("detached spawn failed: {e}"));
                }
                // A structured `delegation-spawn` detail (opaque kind+body, no wire bump): the
                // detached child's session id is known here (the store minted it), so the detail
                // carries the concrete `child`.
                let body = serde_json::to_vec(&serde_json::json!({
                    "child": child.as_str(),
                    "detached": true,
                }))
                .unwrap_or_default();
                let mut outcome = Self::ok(call, format!("spawned-detached:{child}"), Vec::new());
                outcome.detail = Some(ToolDetail::new("delegation-spawn", body));
                outcome
            }
            Verb::Send { target, text } => {
                let Some(store) = &self.store else {
                    return Self::err(
                        call,
                        "send unavailable: no durable session store is wired".into(),
                    );
                };
                let child = SessionId::new(target.clone());
                // Parent-owns-subtree authorization: an agent may only message its own descendants.
                if !self.owns_subtree(&cx.session_id, &child).await {
                    return Self::err(
                        call,
                        format!("send denied: {target} is not a child of this session"),
                    );
                }
                if store.status(&child).await.is_none() {
                    return Self::err(call, format!("send failed: unknown child {target}"));
                }
                // Queue on the durable pending-input seam, then wake: the child's next incarnation
                // drains the message into its conversation before the turn runs.
                store
                    .enqueue_session_input(&child, UserMsg::new(text).encode())
                    .await;
                store.enqueue_wake(child).await;
                Self::ok(call, format!("sent:{target}"), Vec::new())
            }
            Verb::Status { target } => match &self.store {
                // Durable per-child projection (the same graph the GUI tree reads).
                Some(store) => {
                    let mut children = store.children_of(&cx.session_id).await;
                    if let Some(target) = &target {
                        children.retain(|c| c.as_str() == target);
                        if children.is_empty() {
                            return Self::err(
                                call,
                                format!("status: {target} is not a child of this session"),
                            );
                        }
                    }
                    if children.is_empty() {
                        return Self::ok(call, "no children".into(), Vec::new());
                    }
                    let mut lines = vec![format!("children: {}", children.len())];
                    for child in &children {
                        lines.push(Self::status_line(store, child).await);
                    }
                    Self::ok(call, lines.join("\n"), Vec::new())
                }
                // Legacy fleet-wide counts (no durable store wired).
                None => {
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
            },
            Verb::Cancel(id) => {
                let cancelled = self.fleet.cancel_child(&UnitId::new(id.clone())).await;
                Self::ok(call, format!("cancel:{id}:{cancelled}"), Vec::new())
            }
        }
    }
}
