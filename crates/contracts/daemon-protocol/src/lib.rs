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

use daemon_common::{Budget, JobId, RateLimitSnapshot, ReqId, SessionId, UnitId, UsageDelta};
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
#[non_exhaustive]
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
    /// A scheduled job fired (the daemon-schedule trigger source; an inbound, context-bearing
    /// event in the merged session log — see the event-io spec §5.5).
    Scheduled {
        /// The schedule/job that fired.
        job: JobId,
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

// ---------------------------------------------------------------------------
// Merged session event log (the event-io edge, wire v2)
// ---------------------------------------------------------------------------
//
// The §17 surface above is direction-asymmetric: `Outbound` is sequenced, observable, and journaled,
// while `Inbound` is mere engine intake (no `seq`, not broadcast, not recorded). The merged session
// event log closes that gap: both directions become first-class entries on ONE `seq`-stamped log,
// stamped by a single per-session sequencer (the generalisation of the outbound-only `EventSink`).
// Three consumers read the one log through different lenses — the engine (Context entries only),
// any attached surface (everything), and the verifiable journal (Context inbound + outbound +
// lifecycle). These types are the wire shapes for that log; the sequencer itself lives in
// `daemon-core` and the cursored subscribe surface in `daemon-api`.

/// The stable name of a surface/adapter that an item entered or left through ("telegram", "http",
/// "mcp", "slack", "socket", "ffi", "schedule", …). A plain interned string keeps the contract open:
/// adding an adapter needs no protocol change and the daemon never matches structurally on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransportId(pub String);

impl TransportId {
    /// Construct a transport id from its stable name.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The transport id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TransportId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for TransportId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// The conversational scope an inbound item belongs to — the single input (with the transport) to
/// deterministic session-id derivation, and the per-event attribution carried on the log. This is
/// the daemon analogue of hermes's `build_session_key` input, but carried explicitly (never via
/// env/`ContextVars`). Principal/route handles are opaque strings the originating adapter owns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OriginScope {
    /// A direct / 1:1 conversation with a single principal.
    Dm {
        /// The adapter-opaque principal handle (e.g. a Telegram user id).
        user: String,
    },
    /// A group / channel conversation, optionally threaded.
    Group {
        /// The adapter-opaque chat/channel handle.
        chat: String,
        /// The thread handle within the chat, when threaded.
        thread: Option<String>,
    },
    /// A programmatic API caller, keyed by credential/principal.
    Api {
        /// The adapter-opaque credential/principal handle.
        key: String,
    },
    /// A host-internal origin with no external principal — schedule ticks, background completions,
    /// and other daemon-raised triggers.
    Internal,
}

/// Where an inbound item came from (or an outbound item is attributed to). The single input to
/// session-id derivation, carried **per event, not just per session creation**, so the log and
/// journal can record "steered via the GUI by the owner" vs "message from the Telegram user" within
/// one shared conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    /// The surface/adapter the item entered or left through.
    pub transport: TransportId,
    /// The conversational scope (drives session-id derivation + attribution).
    pub scope: OriginScope,
}

impl Origin {
    /// Construct an origin from a transport and a scope.
    pub fn new(transport: impl Into<TransportId>, scope: OriginScope) -> Self {
        Self {
            transport: transport.into(),
            scope,
        }
    }

    /// A host-internal origin (schedule / background triggers) for the given transport.
    pub fn internal(transport: impl Into<TransportId>) -> Self {
        Self {
            transport: transport.into(),
            scope: OriginScope::Internal,
        }
    }
}

/// Which way an entry flows on the merged session log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// World → session: a surface message, steering, a tool result, a trigger, or a meta event.
    Inbound,
    /// Session → world: a streamed [`AgentEvent`] or a blocking [`HostRequest`].
    Outbound,
}

/// Whether a log entry enters the conversation/prompt or is observability-only.
///
/// Default is [`Disposition::Context`]: anything that arrives at the conversation is part of it
/// unless deliberately demoted. [`Disposition::Transport`] is the explicit lever for presence /
/// surface-attach / receipts — entries the engine's prompt never sees, making them **cache-safe by
/// construction** (the first-class form of hermes's "events describe transport, never context").
/// `Transport` entries are also never journaled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Disposition {
    /// Enters the conversation: a turn, a tool message, a rehydration/scheduled trigger.
    #[default]
    Context,
    /// Observability only: presence, surface-attach, receipts — never in the prompt or journal.
    Transport,
}

/// The body of a [`SessionLogEntry`]. Unifies the outbound §17 union ([`Outbound`] = event / request)
/// with the inbound intake ([`Inbound`] = command / response) and adds an inbound transport/meta
/// channel for observability-only events. Opaque `Meta` bodies ride through the daemon untouched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionPayload {
    /// An outbound streamed event (engine → world).
    Event(AgentEvent),
    /// An outbound blocking host request (engine → world).
    Request(HostRequest),
    /// An inbound command (world → engine).
    Command(AgentCommand),
    /// An inbound host response to a prior [`HostRequest`] (world → engine).
    Response(HostResponse),
    /// An inbound transport/meta event with no engine intake — observability only by default
    /// (presence, surface-attach, delivery receipts). Routed by `kind`, decoded by the consumer.
    Meta {
        /// The stable renderer/router discriminator (e.g. `"presence"` / `"attach"` / `"receipt"`).
        kind: String,
        /// The opaque encoded payload (CBOR by convention), decoded by the consumer per `kind`.
        body: Vec<u8>,
    },
}

impl SessionPayload {
    /// The direction implied by the payload variant (outbound for events/requests, inbound for
    /// commands/responses/meta). The canonical source for a [`SessionLogEntry::direction`] field.
    pub fn direction(&self) -> Direction {
        match self {
            SessionPayload::Event(_) | SessionPayload::Request(_) => Direction::Outbound,
            SessionPayload::Command(_)
            | SessionPayload::Response(_)
            | SessionPayload::Meta { .. } => Direction::Inbound,
        }
    }
}

/// One entry on the merged, bidirectional session event log. Carries four orthogonal axes plus the
/// payload: a single monotonic `seq` across **both** directions (global per-session ordering and the
/// subscribe cursor), the `direction`, the per-event `origin` (attribution), and the `disposition`
/// (context vs transport-only).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionLogEntry {
    /// One monotonic sequence across both directions — global ordering and the subscribe cursor.
    pub seq: u64,
    /// Which way this entry flows.
    pub direction: Direction,
    /// Which surface / trigger produced it (per-event attribution).
    pub origin: Origin,
    /// Whether it enters the conversation (`Context`) or is observability-only (`Transport`).
    #[serde(default)]
    pub disposition: Disposition,
    /// The typed payload.
    pub payload: SessionPayload,
}

impl SessionLogEntry {
    /// Build an entry, deriving `direction` from the payload and defaulting `disposition` to
    /// [`Disposition::Context`].
    pub fn new(seq: u64, origin: Origin, payload: SessionPayload) -> Self {
        Self {
            seq,
            direction: payload.direction(),
            origin,
            disposition: Disposition::default(),
            payload,
        }
    }

    /// Demote this entry to [`Disposition::Transport`] (observability-only; never prompt/journal).
    pub fn transport_only(mut self) -> Self {
        self.disposition = Disposition::Transport;
        self
    }
}

// ---------------------------------------------------------------------------
// Inbound normalisation: deterministic origin -> SessionId derivation
// ---------------------------------------------------------------------------
//
// The daemon analogue of hermes's `build_session_key`: one place owns the origin -> session mapping
// and the per-user/per-thread isolation rules (hermes scattered these across the gateway). The result
// is an ordinary `SessionId` the rest of daemon already understands; attribution stays explicit on the
// `Origin` (never via env/`ContextVars`). The first consumer is a transport that has an `Origin` but
// no caller-supplied `SessionId` (a chat adapter, or the HTTP surface deriving an id from a key).

/// How finely an [`Origin`] is split into distinct sessions — the isolation policy a transport
/// applies when deriving a [`SessionId`]. Coarser policies collapse principals/threads onto one
/// shared conversation; finer ones fork a session per principal or per thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IsolationPolicy {
    /// One session per principal (the DM user / API key). Groups fall back to per-chat (no principal).
    PerUser,
    /// One session per chat/channel (threads collapse onto the chat).
    PerChat,
    /// One session per thread within a chat (the finest grouping; falls back to per-chat when the
    /// scope is unthreaded).
    PerThread,
    /// One session per transport+scope-kind — principals and threads all share a single conversation.
    Shared,
}

/// Deterministically map an [`Origin`] to a stable [`SessionId`] under an [`IsolationPolicy`]. The
/// single source of truth for origin -> session derivation and isolation (the daemon analogue of
/// `build_session_key`); the same origin+policy always yields the same id, and distinct
/// principals/threads diverge exactly as the policy dictates.
pub fn session_id_for(origin: &Origin, policy: IsolationPolicy) -> SessionId {
    let t = origin.transport.as_str();
    let key = match &origin.scope {
        OriginScope::Dm { user } => match policy {
            IsolationPolicy::Shared => format!("{t}:dm"),
            _ => format!("{t}:dm:{user}"),
        },
        OriginScope::Group { chat, thread } => match policy {
            IsolationPolicy::PerThread => match thread {
                Some(th) => format!("{t}:group:{chat}:{th}"),
                None => format!("{t}:group:{chat}"),
            },
            IsolationPolicy::Shared => format!("{t}:group"),
            // PerUser/PerChat: chat-level (a group has no single principal; threads collapse).
            _ => format!("{t}:group:{chat}"),
        },
        OriginScope::Api { key } => match policy {
            IsolationPolicy::Shared => format!("{t}:api"),
            _ => format!("{t}:api:{key}"),
        },
        OriginScope::Internal => format!("{t}:internal"),
    };
    SessionId::new(key)
}

// ---------------------------------------------------------------------------
// Outbound delivery targets (where a session's replies are posted) + handover
// ---------------------------------------------------------------------------
//
// §5.4 demotes "primary handover" from an organising concept to an attribute: where an *outbound*
// reply must be posted (e.g. the Telegram message send) is a property of the session, populated from
// the opening `Origin`. A surface attaching is, by default, an observer+submitter (a `Spectator`);
// "handover" is the single explicit op that re-points the `Primary` target. Note: actually *posting*
// to a Primary needs a chat transport (deferred); these types + the host state they back are the
// contract that makes handover expressible now.

/// An adapter-opaque outbound route handle within a transport (e.g. a Telegram chat id, an HTTP
/// connection id). The originating adapter owns its meaning; the daemon never matches on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RouteAddr(pub String);

impl RouteAddr {
    /// Construct a route address from its opaque handle.
    pub fn new(addr: impl Into<String>) -> Self {
        Self(addr.into())
    }

    /// The route address as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Whether an outbound delivery target is the session's authoritative reply sink (`Primary`) or a
/// passive observer (`Spectator`). Exactly one `Primary` is in force at a time; handover re-points it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SinkKind {
    /// The authoritative reply sink — where outbound replies post.
    Primary,
    /// A passive observer — receives the stream but is not the reply sink.
    Spectator,
}

/// Where a session's outbound replies are delivered. A property of the session (populated from the
/// opening [`Origin`]), not caller state — the daemon analogue of hermes's `DeliveryRouter`, but
/// owned by the session rather than threaded through the caller.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeliveryTarget {
    /// The surface/adapter replies post through.
    pub transport: TransportId,
    /// The opaque route within that transport.
    pub route: RouteAddr,
    /// Whether this is the authoritative reply sink or a passive observer.
    pub kind: SinkKind,
}

impl DeliveryTarget {
    /// Construct a delivery target.
    pub fn new(
        transport: impl Into<TransportId>,
        route: impl Into<String>,
        kind: SinkKind,
    ) -> Self {
        Self {
            transport: transport.into(),
            route: RouteAddr::new(route),
            kind,
        }
    }
}

impl Origin {
    /// The default `Primary` [`DeliveryTarget`] implied by this origin — the same transport, routed
    /// to the scope's principal/chat handle. The host seeds a session's reply sink from this when the
    /// session opens.
    pub fn primary_target(&self) -> DeliveryTarget {
        let route = match &self.scope {
            OriginScope::Dm { user } => user.clone(),
            OriginScope::Group { chat, thread } => match thread {
                Some(th) => format!("{chat}:{th}"),
                None => chat.clone(),
            },
            OriginScope::Api { key } => key.clone(),
            OriginScope::Internal => "internal".to_string(),
        };
        DeliveryTarget::new(self.transport.clone(), route, SinkKind::Primary)
    }
}

// ---------------------------------------------------------------------------
// Durable transcript blocks (the verifiable journal's chat-entry payload)
// ---------------------------------------------------------------------------

/// Who authored a transcript message block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TranscriptRole {
    /// The agent/assistant.
    Assistant,
    /// The user / driver (e.g. an injected steer).
    User,
    /// The system / harness.
    System,
}

/// One *finished* block of an agent transcript — the coalesced unit that graduates into durable,
/// signed history. The host's coalescer folds the fine-grained §17 stream (streaming text/reasoning
/// deltas, which are *not* individually journaled) into these at turn/tool boundaries; the verifiable
/// journal stores each as one entry, and a consuming GUI replays them for scroll-back. Opaque tool
/// `detail` / content `body` ride through untouched (the daemon never matches on them).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TranscriptBlock {
    /// A finished message (assistant text assembled from its deltas, or an injected user/system msg).
    Message {
        /// The author of the message.
        role: TranscriptRole,
        /// The assembled message text.
        text: String,
    },
    /// A tool invocation that was issued (the call as it entered history).
    ToolCall {
        /// Correlates the call with its result.
        call_id: String,
        /// The tool's stable name.
        name: String,
        /// A human-readable summary of the arguments (never the raw secret-bearing payload).
        args_summary: String,
        /// An optional opaque structured payload (e.g. the arguments object) for a rich consumer.
        detail: Option<ToolDetail>,
    },
    /// A tool result that was produced.
    ToolResult {
        /// Correlates back to the originating [`TranscriptBlock::ToolCall`].
        call_id: String,
        /// Whether the tool succeeded.
        ok: bool,
        /// A human-readable summary of the outcome.
        summary: String,
        /// An optional opaque structured payload (e.g. a diff, search results) for a rich consumer.
        detail: Option<ToolDetail>,
    },
    /// A blocking host request that was raised (the prompt as it entered history).
    Request {
        /// Correlation id of the originating [`HostRequest`].
        request_id: ReqId,
        /// The request payload.
        kind: HostRequestKind,
    },
    /// A finished chunk of opaque structured content not tied to a tool call (e.g. a coalesced
    /// terminal/PTY block from a foreign agent). Routed by `kind`, decoded by the consumer.
    Content {
        /// The stable renderer discriminator (e.g. `"ansi-stream"` / `"pty"`).
        kind: String,
        /// The opaque encoded payload (CBOR by convention).
        body: Vec<u8>,
    },
}

impl TranscriptBlock {
    /// A stable kind label for the journal entry envelope subject (`block.*`).
    pub fn kind_label(&self) -> &'static str {
        match self {
            TranscriptBlock::Message { .. } => "block.message",
            TranscriptBlock::ToolCall { .. } => "block.tool_call",
            TranscriptBlock::ToolResult { .. } => "block.tool_result",
            TranscriptBlock::Request { .. } => "block.request",
            TranscriptBlock::Content { .. } => "block.content",
        }
    }
}

// ---------------------------------------------------------------------------
// Management-tree projection DTOs (the GUI/TUI surface)
// ---------------------------------------------------------------------------
//
// These transport-stable mirrors of the orchestration tree live here (next to `Outbound`) rather
// than in `daemon-api` so the management contract (`daemon-supervision`'s `ManagedUnit`) can carry
// the recursive projection/routing seam without an edge to the consumer-surface crate. `daemon-api`
// re-exports them, so the wire mirror (`daemon-api.cddl`) and every existing call site are
// unchanged — the same resolution already used for `Outbound`.

/// What kind of unit a tree node is (a transport-stable mirror of the supervision `UnitKind`). A
/// foreign agent and a `daemon-core` engine are both `Engine` — the GUI cannot, and need not, tell
/// them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitKind {
    /// A leaf brain (a `daemon-core` engine or a foreign agent over a §17 cut).
    Engine,
    /// A host running a unit.
    Host,
    /// An orchestrator running a sub-fleet.
    Orchestrator,
}

/// A tree node's lifecycle state (decoupled from the orchestration runtime's `ChildStatus`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitState {
    /// Attached, no terminal outcome yet (working or idle).
    Running,
    /// Reached a terminal outcome (`end_reason` is the supervision end reason, rendered).
    Finished {
        /// The terminal end reason (e.g. `Completed`, `Interrupted`, `Failed`).
        end_reason: String,
    },
    /// State could not be resolved.
    Unknown,
}

/// One node in the orchestration tree projection (the GUI's per-unit view). The tree is a flat node
/// list plus per-node `children` ids, so deeper / cross-node nesting can fill in later without a DTO
/// change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitNode {
    /// The unit id.
    pub id: UnitId,
    /// What kind of unit this is.
    pub kind: UnitKind,
    /// Its lifecycle state.
    pub state: UnitState,
    /// A short description of the unit's current work, when known.
    pub work: Option<String>,
    /// The unit's folded usage.
    pub usage: UsageDelta,
    /// The ids of this unit's direct children.
    pub children: Vec<UnitId>,
}

/// The orchestration tree as the GUI/TUI sees it: a flat node list rooted at `root`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeReport {
    /// The root unit id (the node itself), when there is one.
    pub root: Option<UnitId>,
    /// Every node in the tree.
    pub nodes: Vec<UnitNode>,
}

/// A transport-stable projection of a unit's management event, for GUI drill-down (decoupled from
/// the supervision `ManageEvent`). Mirrors the per-session poll model: a bounded drain of recent
/// events for one unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManageEventView {
    /// The unit started a unit of work.
    Started {
        /// Monotonic per-unit sequence.
        seq: u64,
    },
    /// Streamed progress (text/reasoning/tool activity rendered to a line).
    Progress {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// A rendered progress line, when textual.
        text: Option<String>,
    },
    /// A usage delta the unit reported.
    Usage {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// The reported delta.
        delta: UsageDelta,
    },
    /// The unit reached a terminal outcome.
    Finished {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// The terminal end reason, rendered.
        end_reason: String,
        /// A final summary, when present.
        summary: Option<String>,
    },
    /// The unit raised an error.
    Error {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// A rendered error message.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cbor_round_trip<T>(value: &T) -> T
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let mut buf = Vec::new();
        ciborium::into_writer(value, &mut buf).expect("cbor encode");
        ciborium::from_reader(buf.as_slice()).expect("cbor decode")
    }

    fn json_round_trip<T>(value: &T) -> T
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let s = serde_json::to_string(value).expect("json encode");
        serde_json::from_str(&s).expect("json decode")
    }

    #[test]
    fn scheduled_trigger_round_trips() {
        let trigger = TurnTrigger::Scheduled {
            job: JobId::from("nightly-digest"),
        };
        assert_eq!(trigger, cbor_round_trip(&trigger));
        assert_eq!(trigger, json_round_trip(&trigger));
    }

    #[test]
    fn disposition_defaults_to_context() {
        assert_eq!(Disposition::default(), Disposition::Context);
    }

    #[test]
    fn payload_direction_matches_variant() {
        let event = SessionPayload::Event(AgentEvent::TurnStarted {
            seq: 0,
            trigger: TurnTrigger::User,
        });
        assert_eq!(event.direction(), Direction::Outbound);

        let request = SessionPayload::Request(HostRequest {
            request_id: ReqId(1),
            kind: HostRequestKind::Input {
                prompt: "name?".into(),
            },
        });
        assert_eq!(request.direction(), Direction::Outbound);

        let command = SessionPayload::Command(AgentCommand::Shutdown);
        assert_eq!(command.direction(), Direction::Inbound);

        let response = SessionPayload::Response(HostResponse {
            request_id: ReqId(1),
            body: HostResponseBody::Input("ada".into()),
        });
        assert_eq!(response.direction(), Direction::Inbound);

        let meta = SessionPayload::Meta {
            kind: "presence".into(),
            body: vec![1, 2, 3],
        };
        assert_eq!(meta.direction(), Direction::Inbound);
    }

    #[test]
    fn session_log_entry_derives_direction_and_default_disposition() {
        let origin = Origin::new(
            "telegram",
            OriginScope::Group {
                chat: "c1".into(),
                thread: Some("t1".into()),
            },
        );
        let entry = SessionLogEntry::new(
            7,
            origin,
            SessionPayload::Command(AgentCommand::StartTurn {
                input: UserMsg::new("hello"),
                request_id: ReqId(2),
            }),
        );
        assert_eq!(entry.direction, Direction::Inbound);
        assert_eq!(entry.disposition, Disposition::Context);

        let round = cbor_round_trip(&entry);
        assert_eq!(entry, round);
        assert_eq!(entry, json_round_trip(&entry));
    }

    #[test]
    fn transport_only_demotes_disposition() {
        let entry = SessionLogEntry::new(
            1,
            Origin::internal("schedule"),
            SessionPayload::Meta {
                kind: "attach".into(),
                body: Vec::new(),
            },
        )
        .transport_only();
        assert_eq!(entry.disposition, Disposition::Transport);
        assert_eq!(entry, cbor_round_trip(&entry));
    }

    #[test]
    fn origin_round_trips_across_scopes() {
        for origin in [
            Origin::new("http", OriginScope::Api { key: "k1".into() }),
            Origin::new("telegram", OriginScope::Dm { user: "u1".into() }),
            Origin::internal("schedule"),
        ] {
            assert_eq!(origin, cbor_round_trip(&origin));
            assert_eq!(origin, json_round_trip(&origin));
        }
    }

    #[test]
    fn session_id_for_is_deterministic() {
        let origin = Origin::new("telegram", OriginScope::Dm { user: "u1".into() });
        assert_eq!(
            session_id_for(&origin, IsolationPolicy::PerUser),
            session_id_for(&origin, IsolationPolicy::PerUser),
        );
        assert_eq!(
            session_id_for(&origin, IsolationPolicy::PerUser).as_str(),
            "telegram:dm:u1",
        );
    }

    #[test]
    fn session_id_for_isolates_principals_unless_shared() {
        let u1 = Origin::new("telegram", OriginScope::Dm { user: "u1".into() });
        let u2 = Origin::new("telegram", OriginScope::Dm { user: "u2".into() });
        // Per-user forks a session per principal...
        assert_ne!(
            session_id_for(&u1, IsolationPolicy::PerUser),
            session_id_for(&u2, IsolationPolicy::PerUser),
        );
        // ...while Shared collapses them onto one conversation.
        assert_eq!(
            session_id_for(&u1, IsolationPolicy::Shared),
            session_id_for(&u2, IsolationPolicy::Shared),
        );
    }

    #[test]
    fn session_id_for_threads_collapse_under_per_chat() {
        let t1 = Origin::new(
            "slack",
            OriginScope::Group {
                chat: "c1".into(),
                thread: Some("t1".into()),
            },
        );
        let t2 = Origin::new(
            "slack",
            OriginScope::Group {
                chat: "c1".into(),
                thread: Some("t2".into()),
            },
        );
        // PerThread keeps threads distinct...
        assert_ne!(
            session_id_for(&t1, IsolationPolicy::PerThread),
            session_id_for(&t2, IsolationPolicy::PerThread),
        );
        assert_eq!(
            session_id_for(&t1, IsolationPolicy::PerThread).as_str(),
            "slack:group:c1:t1",
        );
        // ...PerChat folds them onto the chat.
        assert_eq!(
            session_id_for(&t1, IsolationPolicy::PerChat),
            session_id_for(&t2, IsolationPolicy::PerChat),
        );
        assert_eq!(
            session_id_for(&t1, IsolationPolicy::PerChat).as_str(),
            "slack:group:c1",
        );
    }

    #[test]
    fn session_id_for_internal_scope() {
        let origin = Origin::internal("schedule");
        assert_eq!(
            session_id_for(&origin, IsolationPolicy::Shared).as_str(),
            "schedule:internal",
        );
    }

    #[test]
    fn primary_target_derives_from_origin_and_round_trips() {
        let origin = Origin::new(
            "telegram",
            OriginScope::Group {
                chat: "c1".into(),
                thread: Some("t1".into()),
            },
        );
        let target = origin.primary_target();
        assert_eq!(target.transport, TransportId::new("telegram"));
        assert_eq!(target.route.as_str(), "c1:t1");
        assert_eq!(target.kind, SinkKind::Primary);
        assert_eq!(target, cbor_round_trip(&target));
        assert_eq!(target, json_round_trip(&target));
    }

    #[test]
    fn isolation_policy_round_trips() {
        let policy = IsolationPolicy::PerThread;
        assert_eq!(policy, cbor_round_trip(&policy));
        assert_eq!(policy, json_round_trip(&policy));
    }
}
