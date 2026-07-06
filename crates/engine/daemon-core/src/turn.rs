// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Turn context and effects (§4.2 / §4.3).
//!
//! A turn is modelled as a near-pure function over the conversation that produces a stream of
//! [`Effect`]s; the single-owner applier (in [`crate::engine`]) orders and applies them. The
//! [`TurnCx`] carries the ambient handles a phase/tool needs — cooperative cancellation, the event
//! sink, and the host request channel for blocking human-in-the-loop / delegation requests (§17).

use crate::approval::{ApprovalPolicy, Decision};
use crate::conversation::{ToolCall, Turn};
use crate::events::EventSink;
use crate::exec::{CommandFingerprint, ExecutionEnvironment};
use daemon_common::ReqId;
use daemon_common::{Budget, JobId, ProfileRef, SessionId};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponseBody, SpawnSpec,
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// The ambient context handed to phases and tools during a turn (§4.2).
pub struct TurnCx<'a> {
    /// Cooperative cancellation, checked at phase boundaries and in streams.
    pub cancel: CancellationToken,
    /// The event sink to stream progress without owning the host.
    pub events: &'a EventSink,
    /// The host request channel for blocking requests (§17 human-in-the-loop / delegation).
    pub host: &'a dyn HostRequestHandler,
    /// The session this turn belongs to.
    pub session_id: SessionId,
    /// The owning *identity* profile (§5.9 routed profile) of the engine running this turn, or `None`
    /// for the node default. A profile-scoped tool (e.g. `lcm_*` / `mnemosyne_*`) resolves its bank by
    /// `(profile, session_id)` so two rooms routed to two profiles never share a context/memory store.
    pub profile: Option<ProfileRef>,
    /// The budget governing this turn's work.
    pub budget: Budget,
    /// The contained execution environment (§13) a tool reads/writes files and runs commands in.
    pub exec: &'a dyn ExecutionEnvironment,
    /// The per-tool result-byte budget: a tool result longer than this is truncated by the pipeline
    /// (the §12 sanitize+budget stage) so one tool cannot blow the model context.
    pub tool_result_budget: usize,
    /// The effective edit-approval policy (§12 session mode) for this turn — the engine resolves it
    /// from the session snapshot/config and threads it here so a gated tool (fs edit / dangerous
    /// shell command) consults it before acting.
    pub approval_policy: ApprovalPolicy,
    /// Set when the engine is **re-running** a gated tool call whose approval was already granted
    /// by an operator (the durable HITL resume): the tool skips its approval gate and performs the
    /// side effect directly. `false` on a normal turn.
    pub pre_approved: bool,
    /// The checkpoint store (§12 safety). When present, the pipeline records a workspace checkpoint
    /// before a [`mutates`](crate::tools::Tool::mutates) tool runs, so an operator can rewind. `None`
    /// disables checkpointing (the default for engines the host did not wire one into).
    pub checkpoints: Option<&'a dyn crate::checkpoint::CheckpointStore>,
    /// The default per-tool wall-clock timeout for the §12 pipeline timeout stage. `None` disables
    /// the stage (a tool runs to completion). The pipeline offers this to
    /// [`Tool::call_timeout`](crate::tools::Tool::call_timeout), which returns the effective per-call
    /// timeout (or `None` to opt out).
    pub tool_timeout: Option<Duration>,
    /// The per-session "allow permanently" allow-list (Cluster B / `allow_permanent`): command
    /// fingerprints the operator approved permanently this session. A read-only view seeded from the
    /// durable [`Snapshot`](crate::snapshot::Snapshot) each round; [`ask_host`] short-circuits the gate
    /// (auto-approves without contacting the host) when a gated command's fingerprint is a member.
    /// Empty (`&[]`) = feature-off / fail-safe (never short-circuits). The *write* side (remembering a
    /// new fingerprint) never happens through this borrow — it flows through the single-owner effect
    /// applier ([`Effect::RememberApproval`]) so the snapshot stays the sole source of truth.
    pub session_allow: &'a [CommandFingerprint],
}

impl<'a> TurnCx<'a> {
    /// Build a per-call context that shares every ambient handle but carries a **child** cancel
    /// token, plus that token. The §12 timeout stage runs one tool against the child cx so a
    /// per-tool timeout can abort just that tool (cancel the child) without cancelling the turn.
    pub(crate) fn child_for_call(&self) -> (TurnCx<'a>, CancellationToken) {
        let token = self.cancel.child_token();
        let cx = TurnCx {
            cancel: token.clone(),
            events: self.events,
            host: self.host,
            session_id: self.session_id.clone(),
            profile: self.profile.clone(),
            budget: self.budget,
            exec: self.exec,
            tool_result_budget: self.tool_result_budget,
            approval_policy: self.approval_policy,
            pre_approved: self.pre_approved,
            checkpoints: self.checkpoints,
            tool_timeout: self.tool_timeout,
            session_allow: self.session_allow,
        };
        (cx, token)
    }

    /// The configured default per-tool timeout offered to [`Tool::call_timeout`](crate::tools::Tool::call_timeout).
    pub(crate) fn default_tool_timeout(&self) -> Option<Duration> {
        self.tool_timeout
    }
}

/// An effect a turn phase or tool produces; the single-owner applier orders and applies them
/// (§4.3). Phase 3 carries the subset needed to drive durable suspension; `Checkpoint`/`MemoryWrite`
/// and payload externalization arrive with the later engine slices.
pub enum Effect {
    /// Append a turn to the conversation (durable record).
    Persist(Turn),
    /// The engine delegated background work and now waits on `job` — drives suspension. `payload` is
    /// the opaque job payload (a CBOR [`DelegationInput`](daemon_protocol::DelegationInput): task +
    /// attachment paths) the node-side worker decodes to seed the child + materialize its `inbox/`;
    /// the engine treats it as opaque bytes (it never reads the blob store).
    Delegate {
        /// The delegated job the parent now waits on.
        job: JobId,
        /// The opaque job payload handed to the background worker.
        payload: Vec<u8>,
    },
    /// Spawn an attached, non-joining, self-closing background child (§4.3): the applier issues a
    /// fire-and-forget [`HostRequestKind::Spawn`](daemon_protocol::HostRequestKind::Spawn) and keeps
    /// running — unlike [`Effect::Delegate`], it never enters `waiting_for` and never suspends the
    /// parent. The general post-turn self-improvement seam (background skill review / memory write).
    Spawn(SpawnSpec),
    /// A gated tool (fs edit / dangerous command) needs a **durable** operator decision (§12 HITL):
    /// the host parked the approval ([`HostResponseBody::Deferred`](daemon_protocol::HostResponseBody::Deferred))
    /// rather than answering inline, so the engine records the pending decision and suspends the
    /// turn. On the operator's answer the session wakes and re-runs `call` (allow) or injects a
    /// tool-error (deny). Unlike [`Effect::Delegate`], the wake comes from an operator, not a worker.
    AwaitDecision {
        /// The decision's correlation id (the suspension job id the operator answers by).
        job_id: JobId,
        /// The original tool call to re-run verbatim once approved.
        call: ToolCall,
        /// The human-readable approval prompt (diff summary / command) for the operator.
        prompt: String,
        /// The fs edit's target path (sensitive-path carve-out + display), if any.
        path: Option<String>,
    },
    /// Remember a command's [`CommandFingerprint`] on the session's `allow_permanent` allow-list
    /// (Cluster B / inline path): the operator answered an inline approval with "Allow permanently",
    /// so the single-owner applier records the fingerprint on the durable
    /// [`Snapshot`](crate::snapshot::Snapshot) and every later gate for that exact resolved command
    /// auto-approves for the rest of the session. Least-privilege: only the one approved fingerprint
    /// is trusted, never a blanket approval-mode change. The durable (`ApprovalDecide`) path records
    /// the fingerprint directly in `resolve_approvals`, so it does not emit this effect.
    RememberApproval(CommandFingerprint),
}

/// The verdict of an edit-approval gate (§12) for a gated tool action — what a tool should do after
/// consulting [`approve_path`] / [`approve_command`].
pub enum Gate {
    /// Perform the side effect. `permanent` is set when the operator answered an inline approval with
    /// "Allow permanently" AND the node offered it (a fingerprint exists to key on): the tool then
    /// emits an [`Effect::RememberApproval`] so the exact command auto-approves for the rest of the
    /// session. Always `false` on the auto-allow / policy fast paths and for non-command gates.
    Proceed { permanent: bool },
    /// Reject without acting; the tool returns this reason as a failed result.
    Reject(String),
    /// The host parked the decision durably (headless HITL): the tool must NOT act and instead
    /// suspend by returning an [`Effect::AwaitDecision`] carrying this `job_id` (and the
    /// `awaiting-approval:{job_id}` marker as its result content, so the engine can splice the
    /// resolved result on resume).
    Defer(JobId),
}

/// Run the §12 approval gate for a file-mutating action at `path` with a human-readable `prompt`.
/// Consults the turn's [`ApprovalPolicy`](TurnCx::approval_policy): auto-allow / deny outright, or
/// ask the host (which answers inline on the live path, or parks durably on the headless path).
/// A `pre_approved` re-run (operator already said yes) always proceeds.
pub async fn approve_path(cx: &TurnCx<'_>, path: &str, prompt: String) -> Gate {
    if cx.pre_approved {
        return Gate::Proceed { permanent: false };
    }
    match cx.approval_policy.decide_edit(path) {
        Decision::Allow => Gate::Proceed { permanent: false },
        Decision::Deny => Gate::Reject("denied by approval policy".to_string()),
        // An fs edit has no resolved-command fingerprint to key a permanent allow on, so it never
        // offers permanence (`None`) — a durable allow-permanent on an edit degrades to a single allow.
        Decision::Ask => ask_host(cx, prompt, None).await,
    }
}

/// Run the §12 approval gate for a non-path action (e.g. a dangerous shell command). `fingerprint` is
/// the resolved-command [`CommandFingerprint`] when the caller can key a per-session permanent allow on
/// it (a command surface) — the shell tool passes `Some`; callers without a fingerprint pass `None`
/// (no "allow permanently" offer; a permanent decision degrades to a single allow).
pub async fn approve_command(
    cx: &TurnCx<'_>,
    prompt: String,
    fingerprint: Option<&CommandFingerprint>,
) -> Gate {
    if cx.pre_approved {
        return Gate::Proceed { permanent: false };
    }
    match cx.approval_policy.decide_command() {
        Decision::Allow => Gate::Proceed { permanent: false },
        Decision::Deny => Gate::Reject("denied by approval policy".to_string()),
        Decision::Ask => ask_host(cx, prompt, fingerprint).await,
    }
}

/// Run the §12 approval gate for the **raw shell-string surface** (background `sh -c` / pty) — a
/// DISTINCT, higher-friction capability separate from ordinary foreground argv exec (Cluster B). Any
/// shell string is arbitrary code (pipes, redirects, subshells, `curl … | sh`) and is the persistence
/// / exfil vector in the OpenClaw CVE class, so it ALWAYS asks — it never rides the `AutoAllow` fast
/// path that benign foreground argv may. Only a hard `Deny` policy denies outright, and a
/// `pre_approved` re-run (operator already said yes) proceeds.
pub async fn approve_shell_command(
    cx: &TurnCx<'_>,
    prompt: String,
    fingerprint: Option<&CommandFingerprint>,
) -> Gate {
    if cx.pre_approved {
        return Gate::Proceed { permanent: false };
    }
    match cx.approval_policy {
        ApprovalPolicy::Deny => Gate::Reject("denied by approval policy".to_string()),
        // Ask / AcceptEdits / AutoAllow all ask for a raw shell string (no auto-allow).
        _ => ask_host(cx, prompt, fingerprint).await,
    }
}

/// Raise a blocking [`HostRequestKind::Approval`] and map the host's reply onto a [`Gate`]: the live
/// host answers inline ([`HostResponseBody::Approved`]); the headless/durable host parks it
/// ([`HostResponseBody::Deferred`]).
///
/// Cluster B / `allow_permanent` (wired here per the plan, so it covers BOTH surfaces): when
/// `fingerprint` is a member of the session's `allow_permanent` allow-list, the gate short-circuits to
/// [`Gate::Proceed`] WITHOUT contacting the host — this is what auto-approves an identical in-session
/// re-request and, on the durable path, avoids re-parking it (a park only happens if we ask). The
/// `allow_permanent_offered` flag is set only where a `fingerprint` exists to key the allow-list on;
/// a returned `permanent` is honored only if it was offered (defense in depth).
async fn ask_host(
    cx: &TurnCx<'_>,
    prompt: String,
    fingerprint: Option<&CommandFingerprint>,
) -> Gate {
    if let Some(fp) = fingerprint {
        if cx.session_allow.contains(fp) {
            return Gate::Proceed { permanent: false };
        }
    }
    let resp = cx
        .host
        .request(HostRequest {
            request_id: ReqId(0),
            kind: HostRequestKind::Approval {
                prompt,
                allow_permanent_offered: fingerprint.is_some(),
            },
        })
        .await;
    match resp.body {
        HostResponseBody::Approved {
            approved: true,
            allow_permanent,
        } => Gate::Proceed {
            permanent: allow_permanent && fingerprint.is_some(),
        },
        HostResponseBody::Approved {
            approved: false, ..
        } => Gate::Reject("denied by operator".to_string()),
        HostResponseBody::Deferred(job_id) => Gate::Defer(job_id),
        _ => Gate::Reject("approval not granted".to_string()),
    }
}
