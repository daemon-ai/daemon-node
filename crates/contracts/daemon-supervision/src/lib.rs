// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-supervision` — the generic management protocol.
//!
//! The shared, upward-facing contract spoken by **every managed unit** in the daemon tree (engine,
//! host, or orchestrator). It is `daemon-core`'s §17 host protocol *lifted one level*: §17 is this
//! protocol's `Engine` leaf profile (§4 mapping table). The four parts — [`ManageCommand`] down,
//! [`ManageEvent`] up (lossless-primary, monotonic `seq`), blocking correlated [`ManageRequest`]s
//! through [`ManageRequestHandler`], and the [`ManagedUnit`] interface a supervisor drives — are
//! identical at every level, which is what lets the unit tree nest without new wiring.
//!
//! Engine crates do **not** depend on this; the host adapts §17 ⇄ management on the engine's behalf
//! (spec §4 publication decision). Depends only on `daemon-common`.
//!
//! See `docs/specs/daemon-supervision-spec.md`.

#![forbid(unsafe_code)]

use daemon_common::{Budget, RateLimitSnapshot, ReqId, UnitId, UsageDelta};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Declare a simple string-backed newtype id (local to the management protocol).
macro_rules! str_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl $name {
            /// Construct from anything string-like.
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

str_newtype! {
    /// Identity of a unit of work handed to a unit via [`ManageCommand::Assign`].
    WorkId
}
str_newtype! {
    /// A host-owned OS process identity (a [`CompletionSource`]).
    ProcId
}
str_newtype! {
    /// Identity of an orchestration gate evaluated in a [`ProgressDelta::GateResult`].
    GateId
}
str_newtype! {
    /// A reference to a produced artifact recorded in an [`Outcome`].
    ArtifactRef
}
str_newtype! {
    /// A reference to externally-stored work content the unit resolves through its own tools/store.
    ContentRef
}

// ---------------------------------------------------------------------------
// 2.1 Commands (parent -> unit)
// ---------------------------------------------------------------------------

/// Commands a supervisor sends down to a [`ManagedUnit`] (spec §2.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ManageCommand {
    /// Hand the unit a unit of work. At the engine leaf, `work` resolves to a `UserMsg`.
    Assign {
        /// Correlation id for this assignment.
        request_id: ReqId,
        /// The opaque work reference (never a ticket).
        work: WorkRef,
        /// The caps the supervisor sets for this assignment.
        budget: Budget,
    },
    /// Stop scheduling new work; keep state. No-op at an engine leaf (`Ack::Unsupported`).
    Pause,
    /// Resume scheduling. No-op at an engine leaf (`Ack::Unsupported`).
    Resume,
    /// Cancel in-flight work.
    Cancel {
        /// Optional human-readable reason.
        reason: Option<String>,
    },
    /// Orchestrator: target child count. Engine leaf: no-op (`Ack::Unsupported`).
    Scale {
        /// The target concurrency.
        target: Concurrency,
    },
    /// Request a durable checkpoint.
    Snapshot {
        /// Correlation id for the snapshot request.
        request_id: ReqId,
    },
    /// Shut the unit down. `drain` = finish in-flight work first.
    Shutdown {
        /// Whether to drain in-flight work before stopping.
        drain: bool,
    },
}

/// An opaque work reference — *never a ticket* (synthesis §4.1): an id plus an optional inline
/// payload and a content reference the unit resolves through its own tools/store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkRef {
    /// The work identity.
    pub id: WorkId,
    /// An optional inline payload (small work delivered directly).
    pub payload: Option<WorkPayload>,
    /// An optional reference to externally-stored content.
    pub content: Option<ContentRef>,
}

impl WorkRef {
    /// A work reference carrying an inline text payload.
    pub fn inline(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: WorkId::new(id),
            payload: Some(WorkPayload::text(text)),
            content: None,
        }
    }
}

/// A small inline work payload (large content rides a [`ContentRef`] instead).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkPayload {
    /// The textual body of the work, at the engine leaf the `StartTurn` input.
    pub text: String,
}

impl WorkPayload {
    /// A text payload.
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A target concurrency for [`ManageCommand::Scale`] (meaningful only for orchestrators).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Concurrency(pub u32);

// ---------------------------------------------------------------------------
// 2.2 Events (unit -> parent)
// ---------------------------------------------------------------------------

/// Events a unit streams up to its supervisor (spec §2.2). Lossless-primary with a monotonic `seq`:
/// the in-process and durable paths apply backpressure rather than drop; a lossy live consumer must
/// resync from the last acked `seq` by reading durable state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ManageEvent {
    /// Work started.
    Started {
        /// Monotonic event sequence number.
        seq: u64,
        /// Why the unit started.
        trigger: StartTrigger,
    },
    /// Role-polymorphic progress (§3).
    Progress {
        /// Monotonic event sequence number.
        seq: u64,
        /// The progress increment.
        delta: ProgressDelta,
    },
    /// Incremental usage; identical at every level and aggregates up (invariant #4).
    Usage {
        /// Monotonic event sequence number.
        seq: u64,
        /// The usage increment.
        delta: UsageDelta,
    },
    /// A provider rate-limit window update.
    RateLimit {
        /// Monotonic event sequence number.
        seq: u64,
        /// The current window snapshot.
        snapshot: RateLimitSnapshot,
    },
    /// A health transition; drives the supervisor's restart decisions.
    Health {
        /// Monotonic event sequence number.
        seq: u64,
        /// The new health status.
        status: HealthStatus,
    },
    /// Work finished; `outcome` carries the [`EndReason`].
    Finished {
        /// Monotonic event sequence number.
        seq: u64,
        /// The terminal outcome.
        outcome: Outcome,
    },
    /// An error occurred.
    Error {
        /// Monotonic event sequence number.
        seq: u64,
        /// The failure view.
        failure: FailureView,
    },
}

impl ManageEvent {
    /// The monotonic sequence number this event carries.
    pub fn seq(&self) -> u64 {
        match self {
            ManageEvent::Started { seq, .. }
            | ManageEvent::Progress { seq, .. }
            | ManageEvent::Usage { seq, .. }
            | ManageEvent::RateLimit { seq, .. }
            | ManageEvent::Health { seq, .. }
            | ManageEvent::Finished { seq, .. }
            | ManageEvent::Error { seq, .. } => *seq,
        }
    }
}

/// Why a unit started a piece of work (spec §2.2). Superset of the §17 `TurnTrigger`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartTrigger {
    /// A supervisor assignment opened the work.
    Assigned(WorkId),
    /// The unit resumed from a durable checkpoint.
    Resumed,
    /// A completed background activity rehydrated the unit.
    BackgroundCompletion {
        /// What produced the completion.
        source: CompletionSource,
    },
    /// A child event drove the unit (orchestrator).
    ChildEvent,
}

/// The origin of a background completion (spec §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionSource {
    /// A delegated child engine / job.
    Delegation(UnitId),
    /// A host-owned OS process.
    Process(ProcId),
    /// A child unit.
    Child(UnitId),
}

/// Role-polymorphic progress — the one payload that genuinely differs by role (spec §3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProgressDelta {
    // Engine leaf — the §17 fine-grained turn stream:
    /// Assistant text fragment.
    Text(String),
    /// Assistant reasoning fragment (separate channel; never interleaved into [`ProgressDelta::Text`]).
    Reasoning(String),
    /// A tool invocation began.
    ToolStarted(ToolRef),
    /// A tool invocation finished.
    ToolFinished(ToolResultRef),
    // Orchestrator — fleet-shaped progress:
    /// A child unit started.
    ChildStarted(UnitId),
    /// A child unit finished with an outcome.
    ChildFinished {
        /// The child unit.
        unit: UnitId,
        /// The child's outcome.
        outcome: Outcome,
    },
    /// The unit's work queue depth.
    QueueDepth(u32),
    /// A gate evaluation result.
    GateResult {
        /// The gate.
        gate: GateId,
        /// Whether it passed.
        passed: bool,
    },
}

/// A compact view of a tool invocation (engine-leaf [`ProgressDelta`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRef {
    /// Correlates the start with its result.
    pub call_id: String,
    /// The tool's stable name.
    pub name: String,
}

/// A compact view of a tool result (engine-leaf [`ProgressDelta`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultRef {
    /// Correlates back to the originating [`ToolRef`].
    pub call_id: String,
    /// Whether the tool succeeded.
    pub ok: bool,
}

/// The terminal outcome of a unit's work (spec §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// Why the work ended.
    pub end_reason: EndReason,
    /// An optional human-readable summary.
    pub summary: Option<String>,
    /// References to any artifacts produced.
    pub artifacts: Vec<ArtifactRef>,
}

impl Outcome {
    /// An outcome that only records why the work ended.
    pub fn ended(end_reason: EndReason) -> Self {
        Self {
            end_reason,
            summary: None,
            artifacts: Vec::new(),
        }
    }
}

/// Why a unit's work ended (spec §2.2). At the engine leaf this is the §17.3 `EndReason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EndReason {
    /// Completed normally.
    Completed,
    /// Interrupted/cancelled.
    Interrupted,
    /// Ran out of assigned budget.
    BudgetExhausted,
    /// Failed with a classified cause.
    Failed(FailureClass),
}

/// A coarse classification of a failure, shared by [`EndReason`] and [`FailureView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FailureClass {
    /// Likely to succeed on retry.
    Transient,
    /// Will not succeed without intervention.
    Permanent,
    /// The unit was cancelled.
    Cancelled,
    /// A deadline elapsed.
    Timeout,
    /// An internal invariant failure.
    Internal,
}

/// A failure surfaced on the event stream (`#[non_exhaustive]` per spec §5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureView {
    /// The failure classification.
    pub class: FailureClass,
    /// A human-readable description.
    pub message: String,
}

impl FailureView {
    /// A failure view with a classification and message.
    pub fn new(class: FailureClass, message: impl Into<String>) -> Self {
        Self {
            class,
            message: message.into(),
        }
    }
}

/// A unit's health, driving supervision/restart decisions (spec §2.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy.
    Ok,
    /// Degraded but serving.
    Degraded {
        /// Why it is degraded.
        reason: String,
    },
    /// Unhealthy; a restart candidate.
    Unhealthy {
        /// Why it is unhealthy.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// 2.3 Requests (unit -> parent, blocking + correlated, escalating up)
// ---------------------------------------------------------------------------

/// A blocking, correlated request a unit raises to its supervisor (spec §2.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManageRequest {
    /// Correlation id.
    pub request_id: ReqId,
    /// The request payload.
    pub kind: ManageRequestKind,
}

/// The supervisor's correlated reply to a [`ManageRequest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManageResponse {
    /// Correlation id matching the originating request.
    pub request_id: ReqId,
    /// The typed reply body.
    pub body: ManageResponseBody,
}

/// The kinds of blocking request a unit can raise. Superset of the §17 `HostRequestKind`
/// (`{Approval, Input, Choice, Delegate}`); `Escalate`/`Resource` have no engine-leaf analog.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ManageRequestKind {
    /// "May I run this?" (leaf: §17 `HostRequest::Approval`).
    Approval(ApprovalReq),
    /// "I need input / clarification."
    Input(InputReq),
    /// "Pick one of N."
    Choice(ChoiceReq),
    /// "Attach me N child units" — grows the tree (§16.2).
    Delegate(Vec<DelegationSpec>),
    /// "I can't resolve this — raise to my supervisor." No engine-leaf analog.
    Escalate(EscalationReq),
    /// "I need budget / credentials / placement." No engine-leaf analog.
    Resource(ResourceReq),
}

/// An approval request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalReq {
    /// What is being approved.
    pub prompt: String,
}

/// A free-form input request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputReq {
    /// The input prompt.
    pub prompt: String,
}

/// A pick-one-of-N request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChoiceReq {
    /// The choice prompt.
    pub prompt: String,
    /// The available options.
    pub options: Vec<String>,
}

/// A request to grow the tree with a child unit. `DelegationSpec` carries the child's `WorkRef`, an
/// attenuated toolset/credential scope (never exceeding the parent's — §7), and a `Budget`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationSpec {
    /// The child's work reference.
    pub work: WorkRef,
    /// The budget allotted to the child.
    pub budget: Budget,
    /// The attenuated toolset granted to the child (subset of the parent's).
    pub toolset: Vec<String>,
}

/// An escalation request body (re-raised up the chain when a unit cannot resolve locally).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EscalationReq {
    /// What could not be resolved.
    pub reason: String,
}

/// A resource request body (budget / credentials / placement).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceReq {
    /// A description of the resource needed.
    pub description: String,
    /// The budget requested.
    pub budget: Budget,
}

/// The body of a [`ManageResponse`], typed per request kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ManageResponseBody {
    /// Approval decision.
    Approved(bool),
    /// Free-form input result.
    Input(String),
    /// The index of the chosen option.
    Chosen(usize),
    /// The ids of the attached child units.
    Delegated(Vec<UnitId>),
    /// Whether the escalation was handled upstream.
    Escalated(bool),
    /// A granted resource.
    Resource(ResourceGrant),
    /// The supervisor cannot honor this request kind.
    Unsupported,
    /// The request was cancelled (e.g. the unit was torn down).
    Cancelled,
}

/// A granted resource (the reply to a [`ResourceReq`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceGrant {
    /// The budget granted.
    pub budget: Budget,
}

/// The trait a supervisor implements so a unit can raise blocking requests (spec §2.3). Same shape
/// as §17.1 item 2 — request/response stays correlated and typed, escalating up the chain.
#[async_trait::async_trait]
pub trait ManageRequestHandler: Send + Sync {
    /// Answer (or escalate) a blocking management request.
    async fn request(&self, req: ManageRequest) -> ManageResponse;
}

// ---------------------------------------------------------------------------
// 2.4 The unit (as seen by its supervisor)
// ---------------------------------------------------------------------------

/// A lossless-primary, seq-resyncable stream of [`ManageEvent`]s from a unit to its supervisor.
///
/// The live face is an in-process broadcast; a lagging consumer reconciles gaps against durable
/// state by the monotonic `seq` (spec §2.2 / invariant #1).
pub struct EventStream<T> {
    rx: tokio::sync::broadcast::Receiver<T>,
}

/// The terminal status of an [`EventStream::recv`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamLagged {
    /// The producer was dropped; no more events will arrive.
    Closed,
    /// The consumer lagged and skipped `skipped` events; it must resync from durable state.
    Lagged {
        /// How many events were skipped.
        skipped: u64,
    },
}

impl<T: Clone> EventStream<T> {
    /// Wrap a broadcast receiver as an event stream.
    pub fn new(rx: tokio::sync::broadcast::Receiver<T>) -> Self {
        Self { rx }
    }

    /// Await the next event. `Err(Closed)` ends the stream; `Err(Lagged)` means resync is required.
    pub async fn recv(&mut self) -> Result<T, StreamLagged> {
        use tokio::sync::broadcast::error::RecvError;
        match self.rx.recv().await {
            Ok(ev) => Ok(ev),
            Err(RecvError::Closed) => Err(StreamLagged::Closed),
            Err(RecvError::Lagged(n)) => Err(StreamLagged::Lagged { skipped: n }),
        }
    }
}

/// What a unit is, as seen by its supervisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitKind {
    /// A single conversation (the §17 leaf).
    Engine,
    /// A sub-tree presented as one unit.
    Orchestrator,
}

/// A supervisor's synchronous acknowledgement of a [`ManageCommand`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Ack {
    /// Accepted and applied.
    Accepted,
    /// Accepted and queued behind in-flight work.
    Queued,
    /// The unit is busy and cannot accept the command now.
    Busy,
    /// The unit does not support this command (e.g. `Scale` at an engine leaf).
    Unsupported,
    /// Rejected with a reason.
    Rejected {
        /// Why it was rejected.
        reason: String,
    },
}

/// The uniform upward interface a supervisor drives — the recursion point of the unit tree (§2.4).
///
/// A supervisor cannot tell whether an `Engine` (a host presenting one engine) or an `Orchestrator`
/// (a whole sub-tree) sits behind it; that opacity is why nesting needs no new wiring. Supervisors
/// route by [`UnitId`], never a retained handle (invariant #5).
#[async_trait::async_trait]
pub trait ManagedUnit: Send + Sync {
    /// The durable routing key (generalizes the §17.3 `SessionId` activation key).
    fn id(&self) -> UnitId;
    /// Whether this unit is an engine leaf or an orchestrator sub-tree.
    fn kind(&self) -> UnitKind;
    /// Send a command down; returns a synchronous [`Ack`].
    async fn command(&self, cmd: ManageCommand) -> Ack;
    /// Subscribe to the unit's lossless-primary, seq-resyncable event stream.
    fn events(&self) -> EventStream<ManageEvent>;
    /// Install the handler the unit's upward requests flow through (set at attach time).
    fn install_request_handler(&self, handler: Arc<dyn ManageRequestHandler>);

    /// Drain up to `max` recent §17 [`Outbound`](daemon_protocol::Outbound) items (streamed events +
    /// raised host requests) retained for this unit — the rich, transcript-fidelity drill-down a
    /// consumer reads to render a full transcript for *this* unit (the node surface's
    /// `unit_outbound`). It is the opposite end from [`Self::events`]: that is the coarse,
    /// payload-agnostic management view a supervisor folds; this preserves the full §17 vocabulary
    /// (text, reasoning, tool I/O with opaque structured `detail`, opaque `ContentDelta`, usage,
    /// errors) plus blocking host requests, untouched. A destructive drain (like a poll): each call
    /// consumes what it returns; `max == 0` drains all buffered items.
    ///
    /// The default is empty: only a unit that retains a §17 leaf stream (a host's engine unit, ours
    /// or a foreign agent) overrides it; an orchestrator has no single §17 stream and returns
    /// nothing (its children are drained individually by id).
    fn drain_outbound(&self, _max: u32) -> Vec<daemon_protocol::Outbound> {
        Vec::new()
    }

    // -----------------------------------------------------------------------
    // Recursive tree projection / routing (the GUI's nested management surface)
    // -----------------------------------------------------------------------
    //
    // These let a supervisor project and address an *entire* subtree by `UnitId`, through the same
    // opacity that hides whether a unit is a leaf or a whole sub-fleet: an `Orchestrator` overrides
    // them to forward into its own runtime, so projection/routing recurse uniformly across any
    // nesting — and, eventually, across a placement cut (a remote proxy implements them over the
    // wire). The projection DTOs live in `daemon-protocol` (re-exported by `daemon-api`).
    //
    // Authority split: the fleet that holds a unit's record is the source of truth for *that* unit's
    // state/work/usage. So these methods carry only the recursion: a leaf returns empty/`None` (its
    // own node is built by its holding fleet from the record); an orchestrator returns its
    // descendants' nodes / routes id-addressed reads & commands into its sub-fleet.

    /// The **descendant** nodes of this unit's subtree, flat, with each node's `children` ids filled
    /// (the unit's *own* node is supplied by the fleet that holds its record). Empty for a leaf;
    /// an orchestrator overrides this to project its sub-fleet (which recurses further).
    fn project_subtree(&self) -> Vec<daemon_protocol::UnitNode> {
        Vec::new()
    }

    /// Resolve one node by `id` strictly *within* this unit's subtree (its descendants); `None` if
    /// `id` is not a descendant here. Default: `None` (a leaf has no descendants).
    fn locate_node(&self, _id: &UnitId) -> Option<daemon_protocol::UnitNode> {
        None
    }

    /// A bounded snapshot of recent management-event views for the descendant unit `id`; empty if
    /// `id` is not a descendant here. Default: empty.
    fn locate_events(&self, _id: &UnitId, _max: u32) -> Vec<daemon_protocol::ManageEventView> {
        Vec::new()
    }

    /// Drain up to `max` recent §17 outbound items for the descendant unit `id`; empty if `id` is
    /// not a descendant here. Default: empty.
    fn locate_outbound(&self, _id: &UnitId, _max: u32) -> Vec<daemon_protocol::Outbound> {
        Vec::new()
    }

    /// Route a lifecycle [`ManageCommand`] to the descendant unit `id`; `None` if `id` is not a
    /// descendant here (so a caller can distinguish "not found in this subtree" from a returned
    /// [`Ack`]). Default: `None`.
    async fn locate_command(&self, _id: &UnitId, _cmd: ManageCommand) -> Option<Ack> {
        None
    }
}
