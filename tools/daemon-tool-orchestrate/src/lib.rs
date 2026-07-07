// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-tool-orchestrate` — the agent veneer over the fleet runtime (layout §4: tool surface).
//!
//! Exposes orchestration to the engine as a single `daemon_core::Tool` so the brain can grow/steer a
//! fleet by policy. The verbs:
//!
//! - `spawn`: the explicit DOWN edge of the orchestration flow. The agent supplies the child's
//!   `task` (plus optional `lifetime`, `profile`, `attachments`, `wait`); the tool records the
//!   delegation intent through the §17 host port (yielding the durable `JobId` the engine suspends
//!   on), and the fleet runtime — not the tool — spawns and drives the child when the resulting job
//!   is processed.
//! - `send`: deliver a follow-up `message` into an existing child session (parent-owns-subtree
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
    ChildSource, DelegationInput, DelegationLifetime, HostRequest, HostRequestKind,
    HostResponseBody, ToolDetail, UserMsg,
};
use daemon_store::{ChildLifetime, SessionStatus};
use serde::Deserialize;

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

/// Build the CBOR [`DelegationInput`] payload both spawn paths hand the node-side worker: the
/// joining path carries it on `Effect::Delegate`, the detached path stamps it onto the bare
/// [`JobCommand`](daemon_store::JobCommand). `detached` is the only field that differs between the
/// two paths, so the caller passes it; everything else (task, attachments, lifetime, engine source)
/// is identical — this is the single payload-construction seam.
fn delegation_payload(
    task: String,
    attachments: Vec<String>,
    lifetime: DelegationLifetime,
    source: ChildSource,
    detached: bool,
) -> Vec<u8> {
    DelegationInput {
        task,
        attachments,
        lifetime,
        source,
        detached,
    }
    .encode()
}

/// The declared child lifetime in the tool's lowercase JSON vocabulary. A local mirror of
/// [`DelegationLifetime`] (a wire type whose serde form is PascalCase) so the tool contract stays
/// lowercase (`persistent`/`ephemeral`) without touching the wire enum. Converted at the payload
/// boundary via [`From`].
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Lifetime {
    /// A long-lived child the parent manages (the default): survives after completion.
    #[default]
    Persistent,
    /// A transient subagent the host may reap after it reaches a terminal state.
    Ephemeral,
}

impl From<Lifetime> for DelegationLifetime {
    fn from(lifetime: Lifetime) -> Self {
        match lifetime {
            Lifetime::Persistent => DelegationLifetime::Persistent,
            Lifetime::Ephemeral => DelegationLifetime::Ephemeral,
        }
    }
}

/// A joining spawn (`wait`) defaults to `true`: the parent suspends until the child finishes.
fn default_wait() -> bool {
    true
}

/// The verbs the agent can invoke through the orchestrate tool, deserialized directly from the
/// tool-call JSON. The `verb` tag selects the variant and `#[serde(deny_unknown_fields)]` rejects
/// typos and stray keys, so validation is the type's job — no hand-rolled matcher.
#[derive(Debug, Deserialize)]
#[serde(tag = "verb", rename_all = "lowercase", deny_unknown_fields)]
enum Verb {
    /// Grow the fleet: delegate work to a new child.
    Spawn {
        /// The child's task instruction (required).
        task: String,
        /// The declared child lifetime (managed vs transient subagent); defaults to `persistent`.
        #[serde(default)]
        lifetime: Lifetime,
        /// Where the child's engine comes from: omit for the node's default engine shape,
        /// `{"profile":"name"}` to delegate to a registered profile, or `{"inline":{…}}` for an
        /// ad-hoc sub-agent (Phase 1). Defaults to [`ChildSource::Default`].
        #[serde(default)]
        source: ChildSource,
        /// Parent-workspace-relative paths to hand down into the child's `inbox/`.
        #[serde(default)]
        attachments: Vec<String>,
        /// Whether this turn **blocks** on the child (`true`, the default — the joining
        /// `Effect::Delegate` path that suspends the parent until the child finishes and returns its
        /// result inline) or runs the child **detached** in the background (`false` — the parent's
        /// turn continues and a completion notice arrives later as a fresh reactive turn).
        #[serde(default = "default_wait")]
        wait: bool,
    },
    /// Deliver a follow-up message into an existing child session (Steer-equivalent).
    Send {
        /// The child session/unit id to deliver into.
        target: String,
        /// The message text to deliver.
        message: String,
    },
    /// Observe per-child state (optionally filtered to one child).
    Status {
        /// A single child to report on (`None` = every direct child).
        #[serde(default)]
        target: Option<String>,
    },
    /// Cancel a registered child by id.
    Cancel {
        /// The child id to cancel.
        target: String,
    },
}

/// Parse the tool-call args into a [`Verb`], surfacing serde's validation error (unknown verb,
/// missing/typo'd field, wrong value type) as a human-readable string.
fn parse_args(args: &str) -> Result<Verb, String> {
    serde_json::from_str::<Verb>(args).map_err(|e| e.to_string())
}

/// The default ceiling on nested delegation depth before the tool stops delegating (and the engine
/// completes instead). The durable delegation graph is recursive — a child is itself an
/// orchestrator-capable engine that can delegate — so without a guard a model that always delegates
/// would spawn an unbounded chain of sessions. The depth is read from the session id (the durable
/// child minter encodes the tree path with `/`), so it needs no extra protocol field.
const DEFAULT_MAX_DEPTH: usize = 8;

/// The default ceiling on a parent's concurrently-active **detached** children (`spawn wait:false`).
/// Detached spawns do not suspend the parent, so — unlike joining delegation, which is naturally
/// serialized by the parent's suspension — a fan-out loop could otherwise mint unbounded background
/// children. At the cap the tool declines with `fanout-limit:<n>` (mirroring the depth guard),
/// counting active children via the durable child index.
const DEFAULT_MAX_FANOUT: usize = 8;

/// The agent's handle onto a node's [`FleetRuntime`] (+ optionally the durable session graph).
pub struct OrchestrateTool {
    fleet: FleetRuntime,
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
    /// A tool over `fleet` (default depth/fanout caps, no durable store wired).
    pub fn new(fleet: FleetRuntime) -> Self {
        Self {
            fleet,
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
        r#"{"type":"object","required":["verb"],"properties":{"verb":{"type":"string","enum":["spawn","send","status","cancel"],"description":"spawn a child, send a follow-up message to one, report per-child status, or cancel one"},"task":{"type":"string","description":"spawn: the instruction seeding the child's first turn (required for spawn)"},"message":{"type":"string","description":"send: the message text to deliver into the child (required for send)"},"lifetime":{"type":"string","enum":["persistent","ephemeral"],"description":"spawn: persistent = long-lived managed child (default); ephemeral = transient subagent, archived after it completes"},"source":{"description":"spawn: where the child's engine comes from — omit for the node's default engine shape; {\"profile\":\"name\"} to delegate to a registered profile; or {\"inline\":{...}} for an ad-hoc sub-agent","oneOf":[{"type":"object","required":["profile"],"additionalProperties":false,"properties":{"profile":{"type":"string","description":"the registered profile name the child runs under"}}},{"type":"object","required":["inline"],"additionalProperties":false,"properties":{"inline":{"type":"object","additionalProperties":false,"description":"an ad-hoc, un-saved sub-agent config (no saved profile)","properties":{"system_prompt":{"type":"string","description":"the sub-agent's persona"},"tool_allowlist":{"type":"array","items":{"type":"string"},"description":"the tools the sub-agent may use (an explicit allowlist is required; omitting it requests the full toolset, which is operator-only and rejected)"},"model":{"type":"string","description":"the model id the sub-agent resolves (Core)"},"engine":{"description":"\"Core\" (default) or {\"Foreign\":{\"agent\":\"name\"}}"},"foreign_backend":{"description":"for a Foreign engine: how it sources its model backend"}}}}}]},"attachments":{"type":"array","items":{"type":"string"},"description":"spawn: parent-workspace-relative paths handed to the child's inbox/"},"wait":{"type":"boolean","description":"spawn: true (default) blocks this turn until the child finishes and returns its result inline; false runs the child in the background and delivers a completion notification later, letting this turn continue and fan out more subagents"},"target":{"type":"string","description":"send/cancel: the child id (required). status: optional filter to one child."}}}"#
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
            // The child's structured job payload is built ONCE from the agent's inputs (the shared
            // `delegation_payload` seam); `wait` decides only HOW it is dispatched — a joining
            // suspension (`Effect::Delegate`) vs a detached enqueue + completion-notice edge.
            Verb::Spawn {
                task,
                lifetime,
                source,
                attachments,
                wait,
            } => {
                // Security gate (Cluster E): an INLINE sub-agent spec that widens the security
                // posture (requests the full node toolset) is operator-tier. An in-turn agent has no
                // principal (never an operator), so the tool rejects a widening inline spec outright
                // — an inline sub-agent must run under an explicit least-privilege tool allowlist.
                if let ChildSource::Inline(spec) = &source {
                    if spec.widens_security_posture() {
                        return Self::err(
                            call,
                            "inline sub-agent denied: an inline spec may not request the full node \
                             toolset (set an explicit tool_allowlist); widening the tool surface is \
                             operator-only"
                                .into(),
                        );
                    }
                }
                let lifetime = DelegationLifetime::from(lifetime);
                if wait {
                    // DOWN edge (joining): record the delegation through the host port; the fleet
                    // worker spawns the child when the resulting durable job is processed. Emitting
                    // Effect::Delegate suspends the parent until the child's completion wakes it
                    // (lifecycle §3.1).
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
                    // The node-side worker seeds the child from `task`, materializes `attachments`
                    // into its inbox/, derives its role from `lifetime`, and resolves the engine from
                    // `source` (bound profile / inline spec / default) for the durable resolver.
                    let payload = delegation_payload(task, attachments, lifetime, source, false);
                    // A structured `delegation-spawn` detail (opaque kind+body, no wire bump) a rich
                    // client renders as a spawn card. The joining child's session id is minted
                    // node-side (`{parent}/c{epoch}`) only when the durable Delegate job is
                    // processed, so it is not knowable here — the detail carries the `job` handle,
                    // correlatable to the child later via the children/roster surfaces.
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
                } else {
                    // DETACHED (non-joining): enqueue the child directly onto the durable job outbox
                    // — no Effect::Delegate, so the parent's turn does NOT suspend and keeps
                    // running. The child's terminal completion arrives later as a fresh reactive
                    // turn (a completion notice). This is Cursor's `run_in_background: true`
                    // subagent analogue.
                    let Some(store) = &self.store else {
                        return Self::err(
                            call,
                            "detached spawn requires a durable session store".into(),
                        );
                    };
                    // Fan-out cap: a detached spawn does not suspend the parent, so a fan-out loop
                    // could mint unbounded background children. Decline at the cap (mirroring the
                    // depth guard).
                    let active = self.active_child_count(&cx.session_id).await;
                    if active >= self.max_fanout {
                        return Self::guardrail(call, GuardrailKind::Fanout, self.max_fanout);
                    }
                    let payload = delegation_payload(task, attachments, lifetime, source, true);
                    // The store mints the unique `{parent}/dN` child id, stamps it onto the bare job,
                    // and enqueues it (no checkpoint/suspension). Then record the completion-notice
                    // edge so the child's terminal `mark_completed` delivers a notice to this parent
                    // (and the child shows up in `status`/tree immediately).
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
                    // notice carries it, so a client chip-links the injected turn to THIS call's
                    // card.
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
                    let mut outcome =
                        Self::ok(call, format!("spawned-detached:{child}"), Vec::new());
                    outcome.detail = Some(ToolDetail::new("delegation-spawn", body));
                    outcome
                }
            }
            Verb::Send { target, message } => {
                let Some(store) = &self.store else {
                    return Self::err(
                        call,
                        "send unavailable: no durable session store is wired".into(),
                    );
                };
                let child = SessionId::new(target.clone());
                // Parent-owns-subtree authorization: an agent may only message its own descendants.
                if !daemon_store::owns_subtree(store.as_ref(), &cx.session_id, &child).await {
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
                    .enqueue_session_input(&child, UserMsg::new(message).encode())
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
            Verb::Cancel { target } => {
                let cancelled = self.fleet.cancel_child(&UnitId::new(target.clone())).await;
                Self::ok(call, format!("cancel:{target}:{cancelled}"), Vec::new())
            }
        }
    }
}
