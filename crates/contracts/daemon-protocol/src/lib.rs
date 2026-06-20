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
    /// Inject mid-turn steering text (drained at the next phase boundary; opens a steer turn when
    /// the engine is idle).
    Steer {
        /// The steering text.
        text: String,
        /// Correlation id for this steer request (echoed on [`AgentEvent::Steered`]).
        request_id: ReqId,
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

/// A read-only projection of one conversation turn, carried in a [`ConvView`] (§17 snapshot reply).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvTurnView {
    /// The turn's role: `user`, `assistant`, or `tool`.
    pub role: String,
    /// The message / assistant text for the turn.
    pub text: String,
    /// The names of any tools invoked in this turn (empty for non-tool turns).
    pub tools: Vec<String>,
}

/// A read-only projection of the engine's conversation at a consistent phase boundary — the body of
/// an [`AgentEvent::Snapshot`] reply. Built by the engine from its durable `Snapshot`; never live
/// state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvView {
    /// The incarnation epoch the view was taken at.
    pub epoch: u64,
    /// The conversation turns, oldest first.
    pub turns: Vec<ConvTurnView>,
    /// Rendered ids of the background work the engine is currently waiting on.
    pub waiting_for: Vec<String>,
}

/// An opaque, structured payload a brain (or a foreign-agent adapter) attaches to a tool view or a
/// [`AgentEvent::ContentDelta`] so a rich consumer (a transcript GUI) can render it — a tool's
/// arguments object, a unified diff, a web-search result list, an image-generation output, a
/// terminal/PTY byte stream, etc.
///
/// The carrier is deliberately **opaque to the daemon**: the brain and the consuming GUI agree on
/// the schema; the host, orchestrator, and node surface pass it through untouched and never match on
/// it (so a foreign agent can ship payload shapes the daemon has never seen). `kind` is a stable
/// discriminator the GUI routes a renderer by (a tool name for tool I/O, or a reserved kind such as
/// `"ansi-stream"` / `"pty"` for terminal output); `body` is the encoded payload (CBOR by
/// convention) the GUI decodes per `kind`. Kept as raw bytes so the §17 wire types stay `Eq` and the
/// contract crate gains no codec dependency.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDetail {
    /// The stable renderer discriminator (e.g. a tool name, or `"ansi-stream"`/`"pty"`).
    pub kind: String,
    /// The opaque encoded payload (CBOR by convention), decoded by the consumer per `kind`.
    pub body: Vec<u8>,
}

impl ToolDetail {
    /// Construct a detail from a kind and its encoded body.
    pub fn new(kind: impl Into<String>, body: impl Into<Vec<u8>>) -> Self {
        Self {
            kind: kind.into(),
            body: body.into(),
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
    /// An optional opaque structured payload (e.g. the arguments object) for a rich consumer.
    /// Passed through the daemon untouched; absent when the brain has nothing structured to attach.
    #[serde(default)]
    pub detail: Option<ToolDetail>,
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
    /// An optional opaque structured payload (e.g. a diff, search results, an image) for a rich
    /// consumer. Passed through the daemon untouched; absent when there is nothing structured.
    #[serde(default)]
    pub detail: Option<ToolDetail>,
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
    /// A chunk of opaque structured stream content **not tied to a tool call** — a whole-agent
    /// terminal/PTY stream, a foreign agent's raw rendered output, or a future structured content
    /// type. Like [`ToolCallView::detail`] the payload is opaque to the daemon: the host,
    /// orchestrator, and node surface pass it through untouched; a rich consumer routes by `kind`
    /// (e.g. `"ansi-stream"` / `"pty"`) and decodes `body`. Reasoning and plain assistant text keep
    /// their dedicated typed channels ([`AgentEvent::ReasoningDelta`] / [`AgentEvent::TextDelta`]).
    ContentDelta {
        /// Monotonic event sequence number.
        seq: u64,
        /// The stable renderer discriminator (e.g. `"ansi-stream"` / `"pty"`).
        kind: String,
        /// The opaque encoded payload (CBOR by convention), decoded by the consumer per `kind`.
        body: Vec<u8>,
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
    /// A steer command was acknowledged (drained at a phase boundary or opened a steer turn).
    Steered {
        /// Monotonic event sequence number.
        seq: u64,
        /// Correlation id echoed from [`AgentCommand::Steer`].
        request_id: ReqId,
        /// Whether the steer was accepted into the conversation.
        accepted: bool,
    },
    /// A read-only snapshot reply (the §17 snapshot ride on the event stream).
    Snapshot {
        /// Monotonic event sequence number.
        seq: u64,
        /// Correlation id echoed from [`AgentCommand::Snapshot`].
        request_id: ReqId,
        /// The consistent conversation projection.
        view: ConvView,
    },
}

impl AgentEvent {
    /// The monotonic sequence number this event carries.
    pub fn seq(&self) -> u64 {
        match self {
            AgentEvent::TurnStarted { seq, .. }
            | AgentEvent::TextDelta { seq, .. }
            | AgentEvent::ReasoningDelta { seq, .. }
            | AgentEvent::ContentDelta { seq, .. }
            | AgentEvent::ToolStarted { seq, .. }
            | AgentEvent::ToolFinished { seq, .. }
            | AgentEvent::Usage { seq, .. }
            | AgentEvent::RateLimit { seq, .. }
            | AgentEvent::TurnFinished { seq, .. }
            | AgentEvent::Error { seq, .. }
            | AgentEvent::Steered { seq, .. }
            | AgentEvent::Snapshot { seq, .. } => *seq,
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

// ---------------------------------------------------------------------------
// §17 frame unions (engine-relative): `Outbound` (engine -> host) is the canonical
// up-union, also used as the node drain item; `Inbound` (host -> engine) is its partner.
// Both serialize as the CBOR-framed dialect spoken over a foreign-agent process cut.
// ---------------------------------------------------------------------------

/// A §17 frame delivered **to** an engine (host -> engine). Over a foreign-agent process cut these
/// arrive on the agent's stdin; a foreign brain that speaks §17 is driven by them, and the host
/// wraps the cut as an `Engine`-leaf managed unit. The reference in-process brain (`daemon-core`)
/// uses typed channels instead, but the dialect is the same §17.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Inbound {
    /// A §17 command for the engine to act on.
    Command(AgentCommand),
    /// The host's reply to a [`HostRequest`] the engine raised.
    Response(HostResponse),
}

/// A §17 frame emitted **from** an engine (engine -> host). This is the canonical "item coming up
/// from an engine" union: it doubles as the node drain item (`daemon-api` re-exports it as
/// `Outbound`) and as the up-frame over a foreign-agent process cut, written to the agent's stdout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Outbound {
    /// A streamed §17 event.
    Event(AgentEvent),
    /// A blocking §17 host request awaiting an [`Inbound::Response`].
    Request(HostRequest),
}
