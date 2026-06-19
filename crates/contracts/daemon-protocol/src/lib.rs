//! `daemon-protocol` — the §17 host wire protocol (the engine's upward face).
//!
//! The typed control/event/request surface the engine (`daemon-core`) speaks to its host: commands
//! down (`AgentCommand`), a lossless-primary event stream up (`AgentEvent`, each carrying a
//! monotonic `seq`), and blocking correlated host requests (`HostRequest` / `HostRequestHandler`).
//! Pure wire types only; no runtime logic. Depends only on `daemon-common`.
//!
//! The typed engine snapshot (§5 `Conversation`, references) lives in `daemon-core`, not here — the
//! durable substrate handles only the opaque [`SnapshotBlob`](daemon_common::SnapshotBlob). The host
//! adapts this §17 surface to the generic management protocol (`daemon-supervision` §4); the engine
//! crate stays free of that protocol.

#![forbid(unsafe_code)]

use daemon_common::{Budget, JobId, RateLimitSnapshot, ReqId, UsageDelta};
use serde::{Deserialize, Serialize};

/// A user-authored turn input (the `StartTurn` payload; the §5 message type proper lives in core).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMsg {
    /// The textual body of the message.
    pub text: String,
}

impl UserMsg {
    /// Construct a user message from text.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

// ---------------------------------------------------------------------------
// §17 control surface: host -> engine
// ---------------------------------------------------------------------------

/// Commands the host sends down to an engine (§17, host -> core).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AgentCommand {
    /// Begin a turn from a (user or resumed) trigger.
    StartTurn {
        /// The input that opens the turn.
        input: UserMsg,
        /// Correlation id for this turn request.
        request_id: ReqId,
    },
    /// Inject mid-turn steering text.
    Steer {
        /// The steering text.
        text: String,
    },
    /// Interrupt the current turn.
    Interrupt {
        /// Optional human-readable reason.
        reason: Option<String>,
    },
    /// Request a read-only snapshot view.
    Snapshot {
        /// Correlation id for the snapshot request.
        request_id: ReqId,
    },
    /// Drain and shut the engine down.
    Shutdown,
}

// ---------------------------------------------------------------------------
// §17 event surface: engine -> host
// ---------------------------------------------------------------------------

/// Why a turn started (§17). A background completion is the durable rehydration trigger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnTrigger {
    /// A user message opened the turn.
    User,
    /// Steering text opened the turn.
    Steer,
    /// A completed background activity rehydrated the engine.
    BackgroundCompletion {
        /// What produced the completion.
        source: CompletionSource,
    },
}

/// The origin of a background completion (§17).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionSource {
    /// A host-owned OS process.
    Process(JobId),
    /// A delegated child engine / job.
    Delegation(JobId),
}

/// How a turn ended (carried in [`TurnSummary`]; the §17.3 leaf form of the management `EndReason`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EndReason {
    /// The turn completed normally.
    Completed,
    /// The engine suspended at a phase boundary to await background work.
    Suspended,
    /// The turn was interrupted.
    Interrupted,
    /// The turn ran out of its assigned budget.
    BudgetExhausted,
    /// The turn failed.
    Failed,
}

/// Terminal turn outcome (§17).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSummary {
    /// Why the turn ended.
    pub end_reason: EndReason,
    /// Optional final assistant text.
    pub final_text: Option<String>,
    /// Usage accrued over the turn.
    pub usage: UsageDelta,
}

impl TurnSummary {
    /// A summary that only records why the turn ended.
    pub fn ended(end_reason: EndReason) -> Self {
        Self {
            end_reason,
            final_text: None,
            usage: UsageDelta::default(),
        }
    }
}

/// A compact view of a tool invocation, streamed on the event surface (the durable record lives in
/// the engine's `Conversation`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallView {
    /// Correlates the start with its result.
    pub call_id: String,
    /// The tool's stable name.
    pub name: String,
    /// A human-readable summary of the arguments (never the raw secret-bearing payload).
    pub args_summary: String,
}

/// A compact view of a tool result, streamed on the event surface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultView {
    /// Correlates back to the originating [`ToolCallView`].
    pub call_id: String,
    /// Whether the tool succeeded.
    pub ok: bool,
    /// A human-readable summary of the outcome.
    pub summary: String,
}

/// Events the engine streams up to the host (§17, core -> host). Each carries a monotonic `seq`;
/// the stream is lossless-primary, so a lossy live consumer resyncs from the last acked `seq`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AgentEvent {
    /// The turn began.
    TurnStarted {
        /// Monotonic event sequence number.
        seq: u64,
        /// Why the turn started.
        trigger: TurnTrigger,
    },
    /// A chunk of assistant text.
    TextDelta {
        /// Monotonic event sequence number.
        seq: u64,
        /// The text fragment.
        text: String,
    },
    /// A chunk of assistant reasoning. A deliberately separate channel from [`AgentEvent::TextDelta`]
    /// so a host never accidentally renders reasoning as output (§17.2 scrubbing).
    ReasoningDelta {
        /// Monotonic event sequence number.
        seq: u64,
        /// The reasoning fragment.
        text: String,
    },
    /// A tool invocation began.
    ToolStarted {
        /// Monotonic event sequence number.
        seq: u64,
        /// The invocation view.
        call: ToolCallView,
    },
    /// A tool invocation finished.
    ToolFinished {
        /// Monotonic event sequence number.
        seq: u64,
        /// The result view.
        result: ToolResultView,
    },
    /// Incremental usage; aggregates up the tree by construction (identical at every level).
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
    /// The turn finished.
    TurnFinished {
        /// Monotonic event sequence number.
        seq: u64,
        /// The terminal summary.
        summary: TurnSummary,
    },
    /// An error occurred during the turn.
    Error {
        /// Monotonic event sequence number.
        seq: u64,
        /// Human-readable failure description.
        failure: String,
    },
}

impl AgentEvent {
    /// The monotonic sequence number this event carries.
    pub fn seq(&self) -> u64 {
        match self {
            AgentEvent::TurnStarted { seq, .. }
            | AgentEvent::TextDelta { seq, .. }
            | AgentEvent::ReasoningDelta { seq, .. }
            | AgentEvent::ToolStarted { seq, .. }
            | AgentEvent::ToolFinished { seq, .. }
            | AgentEvent::Usage { seq, .. }
            | AgentEvent::RateLimit { seq, .. }
            | AgentEvent::TurnFinished { seq, .. }
            | AgentEvent::Error { seq, .. } => *seq,
        }
    }
}

// ---------------------------------------------------------------------------
// §17 blocking host requests (human-in-the-loop / delegation)
// ---------------------------------------------------------------------------

/// A blocking, correlated request the engine raises to the host (§17).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostRequest {
    /// Correlation id.
    pub request_id: ReqId,
    /// The request payload.
    pub kind: HostRequestKind,
}

/// The kinds of blocking host request the engine can raise (§17 = `{Approval, Input, Choice,
/// Delegate}`; the management protocol's `ManageRequestKind` is the superset that adds
/// `Escalate`/`Resource`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HostRequestKind {
    /// Ask the host to approve an action.
    Approval {
        /// What is being approved.
        prompt: String,
    },
    /// Ask the host for free-form input.
    Input {
        /// The input prompt.
        prompt: String,
    },
    /// Ask the host to pick one of N options.
    Choice {
        /// The choice prompt.
        prompt: String,
        /// The available options.
        options: Vec<String>,
    },
    /// Ask the host to delegate background work, yielding a [`JobId`].
    Delegate {
        /// A label describing the delegated work.
        label: String,
        /// The budget allotted to the delegated work.
        budget: Budget,
    },
}

/// The host's correlated reply to a [`HostRequest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostResponse {
    /// Correlation id matching the originating request.
    pub request_id: ReqId,
    /// The typed reply body.
    pub body: HostResponseBody,
}

/// The body of a [`HostResponse`], typed per request kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HostResponseBody {
    /// Approval decision.
    Approved(bool),
    /// Free-form input result.
    Input(String),
    /// The index of the chosen option.
    Chosen(usize),
    /// The id assigned to delegated work.
    Delegated(JobId),
}

/// The trait the host implements so an engine can raise blocking requests (§17).
#[async_trait::async_trait]
pub trait HostRequestHandler: Send + Sync {
    /// Answer a blocking host request.
    async fn request(&self, req: HostRequest) -> HostResponse;
}
