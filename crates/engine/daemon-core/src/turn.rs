//! Turn context and effects (§4.2 / §4.3).
//!
//! A turn is modelled as a near-pure function over the conversation that produces a stream of
//! [`Effect`]s; the single-owner applier (in [`crate::engine`]) orders and applies them. The
//! [`TurnCx`] carries the ambient handles a phase/tool needs — cooperative cancellation, the event
//! sink, and the host request channel for blocking human-in-the-loop / delegation requests (§17).

use crate::approval::{ApprovalPolicy, Decision};
use crate::conversation::{ToolCall, Turn};
use crate::events::EventSink;
use crate::exec::ExecutionEnvironment;
use daemon_common::{Budget, JobId, ProfileRef, SessionId};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponseBody, SpawnSpec,
};
use daemon_common::ReqId;
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
}

/// The verdict of an edit-approval gate (§12) for a gated tool action — what a tool should do after
/// consulting [`approve_path`] / [`approve_command`].
pub enum Gate {
    /// Perform the side effect.
    Proceed,
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
        return Gate::Proceed;
    }
    match cx.approval_policy.decide_edit(path) {
        Decision::Allow => Gate::Proceed,
        Decision::Deny => Gate::Reject("denied by approval policy".to_string()),
        Decision::Ask => ask_host(cx, prompt).await,
    }
}

/// Run the §12 approval gate for a non-path action (e.g. a dangerous shell command).
pub async fn approve_command(cx: &TurnCx<'_>, prompt: String) -> Gate {
    if cx.pre_approved {
        return Gate::Proceed;
    }
    match cx.approval_policy.decide_command() {
        Decision::Allow => Gate::Proceed,
        Decision::Deny => Gate::Reject("denied by approval policy".to_string()),
        Decision::Ask => ask_host(cx, prompt).await,
    }
}

/// Raise a blocking [`HostRequestKind::Approval`] and map the host's reply onto a [`Gate`]: the live
/// host answers inline ([`HostResponseBody::Approved`]); the headless/durable host parks it
/// ([`HostResponseBody::Deferred`]).
async fn ask_host(cx: &TurnCx<'_>, prompt: String) -> Gate {
    let resp = cx
        .host
        .request(HostRequest {
            request_id: ReqId(0),
            kind: HostRequestKind::Approval { prompt },
        })
        .await;
    match resp.body {
        HostResponseBody::Approved(true) => Gate::Proceed,
        HostResponseBody::Approved(false) => Gate::Reject("denied by operator".to_string()),
        HostResponseBody::Deferred(job_id) => Gate::Defer(job_id),
        _ => Gate::Reject("approval not granted".to_string()),
    }
}
