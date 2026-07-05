// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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

use daemon_common::{
    BlobRef, Budget, JobId, ProfileRef, RateLimitSnapshot, ReqId, SessionId, UnitId, UsageDelta,
};
use serde::{Deserialize, Serialize};

/// A user-authored turn input (the `StartTurn` payload; the §5 message type proper lives in core).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMsg {
    /// The textual body of the message.
    pub text: String,
    /// Content-addressed attachments accompanying the message (daemon-content-transfer-spec.md
    /// Phase 2b). The node materializes these into the session workspace before the turn; the engine
    /// sees the on-disk files (plus a note in `text`) and ignores this field.
    #[serde(default)]
    pub attachments: Vec<BlobRef>,
}

impl UserMsg {
    /// Construct a user message from text (no attachments).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            attachments: Vec::new(),
        }
    }

    /// Attach content-addressed blobs to this message.
    pub fn with_attachments(mut self, attachments: Vec<BlobRef>) -> Self {
        self.attachments = attachments;
        self
    }

    /// CBOR-encode for an opaque byte channel (e.g. the durable pending-input queue the
    /// orchestrate tool's `send` verb feeds). The wire shape is unchanged — this is the same serde
    /// form the §17 command surface carries, just framed as standalone bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("encode UserMsg");
        buf
    }

    /// Decode bytes from an opaque byte channel, falling back to treating raw bytes as plain UTF-8
    /// text (so a producer that queues bare text — or a legacy payload — still resolves to a
    /// message rather than an error).
    pub fn decode(bytes: &[u8]) -> Self {
        ciborium::from_reader(bytes)
            .unwrap_or_else(|_| Self::new(String::from_utf8_lossy(bytes).into_owned()))
    }
}

/// The lifetime a parent declares for a delegated child — the protocol-level mirror of the store's
/// `ChildLifetime`, carried inside the opaque delegation payload so the contract crates stay
/// decoupled (`daemon-protocol` depends only on `daemon-common`; the store enum lives in
/// `daemon-store`). The host maps this onto the durable `JobCommand.lifetime` at the suspension
/// boundary, which in turn derives the child's roster/tree `SessionRole`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelegationLifetime {
    /// A long-lived child the parent manages (the default): survives after completion.
    #[default]
    Persistent,
    /// A transient subagent spun up for a bounded task: the host's reaper may archive it after it
    /// reaches a terminal state.
    Ephemeral,
}

/// The structured input a parent hands a delegated child (daemon-content-transfer-spec.md Phase 2a):
/// the task text plus parent-workspace-relative paths to hand down. Carried as the opaque
/// `JobCommand.payload`. The node (which holds the workspace roots + blob store) resolves the paths
/// to the child's `inbox/`; the engine only ever produces/consumes the opaque bytes.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationInput {
    /// The task instruction seeded as the child's first message.
    pub task: String,
    /// Parent-workspace-relative paths to materialize into the child's `inbox/`.
    #[serde(default)]
    pub attachments: Vec<String>,
    /// The lifetime the parent declares for the child: a long-lived managed child (default) vs a
    /// transient subagent the host may reap after completion. `serde(default)` keeps pre-upgrade
    /// payloads decoding as `Persistent`.
    #[serde(default)]
    pub lifetime: DelegationLifetime,
    /// The named profile the child's engine resolves from (`None` = the node's default engine
    /// shape). The node-side worker binds it as the child's `bound_profile`; an unknown name falls
    /// back to the default shape at resolve time.
    #[serde(default)]
    pub profile: Option<String>,
    /// Whether this is a **detached** (non-suspending) delegation — the orchestrate `spawn wait:false`
    /// mode. `false` (the default) is the ordinary joining delegation: the parent suspends and the
    /// child's terminal completion wakes it. `true` runs the child in the background: the parent's
    /// turn continues, and the child's terminal completion is delivered as a fresh reactive turn (a
    /// completion *notice*), never a job completion — so the node-side worker binds a completion-notice
    /// edge instead of a delegation edge. `serde(default)` keeps pre-upgrade payloads decoding as
    /// joining delegations.
    #[serde(default)]
    pub detached: bool,
}

impl DelegationInput {
    /// A bare delegation with no attachments (default lifetime + profile).
    pub fn task(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            attachments: Vec::new(),
            lifetime: DelegationLifetime::default(),
            profile: None,
            detached: false,
        }
    }

    /// CBOR-encode for the job payload.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("encode DelegationInput");
        buf
    }

    /// Decode a job payload, falling back to treating raw bytes as a legacy plain-text task (e.g.
    /// the historical `b"delegated-work"` marker), so jobs enqueued before the upgrade still resolve.
    pub fn decode(bytes: &[u8]) -> Self {
        ciborium::from_reader(bytes).unwrap_or_else(|_| Self {
            task: String::from_utf8_lossy(bytes).into_owned(),
            attachments: Vec::new(),
            lifetime: DelegationLifetime::default(),
            profile: None,
            detached: false,
        })
    }
}

/// The structured result a child returns to its parent (daemon-content-transfer-spec.md Phase 2a):
/// a summary plus content-addressed artifacts (captured by the node from the child's `outbox/`).
/// Carried as the opaque completion payload.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationResult {
    /// A short human-readable summary of the child's outcome.
    pub summary: String,
    /// Content-addressed artifacts the child produced.
    #[serde(default)]
    pub artifacts: Vec<BlobRef>,
}

impl DelegationResult {
    /// A result with a summary and no artifacts.
    pub fn summary(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            artifacts: Vec::new(),
        }
    }

    /// CBOR-encode for the completion payload.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("encode DelegationResult");
        buf
    }

    /// Decode a completion payload, falling back to treating raw bytes as a legacy plain-text summary
    /// (e.g. the historical `"child:{id}"` marker), so completions written before the upgrade still
    /// resolve.
    pub fn decode(bytes: &[u8]) -> Self {
        ciborium::from_reader(bytes).unwrap_or_else(|_| Self {
            summary: String::from_utf8_lossy(bytes).into_owned(),
            artifacts: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// §17 control surface: host -> engine
// ---------------------------------------------------------------------------

/// Commands the host sends down to an engine (§17, host -> core).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// Append context-only input to the conversation **without** opening a turn (the multi-party
    /// accumulation seam, event-io §5.9): in a shared room the host feeds chatter the agent should
    /// see on its next mention-gated turn, but that must not itself trigger the engine. Folds into
    /// the conversation when idle, and lands in the following turn when busy (drained at the phase
    /// boundary). Attribution (who spoke) rides inside the [`UserMsg`] text, adapter-formatted.
    Observe {
        /// The context-only input to append (no turn is started).
        input: UserMsg,
        /// Correlation id for this observe request.
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
    /// Rewind the conversation to a prior point, sealing/truncating everything after `anchor` and
    /// reconstructing the engine state for that point so a subsequent `StartTurn` replays from there
    /// (conversation-rewind spec). Interrupt-first: if a turn is live the engine interrupts it before
    /// truncating, then bumps the [`Epoch`](daemon_common::Epoch) to fence late arrivals from the
    /// abandoned turn. Only supported by `daemon-core`-backed sessions; foreign agents (ACP) reject
    /// it (their conversation state is not daemon-owned and ACP has no truncate-at-anchor primitive).
    RewindTo {
        /// Where to rewind to.
        anchor: RewindAnchor,
        /// Correlation id (echoed on [`AgentEvent::Rewound`]).
        request_id: ReqId,
    },
    /// Drain and shut the engine down.
    Shutdown,
}

/// A durable, replay-stable address of a conversation-rewind point (conversation-rewind spec §2).
///
/// `ordinal` variants index into the live conversation turns — the same 0-based index a client sees
/// in [`ConvView::turns`] — so a client addresses what it can see. `Cursor` addresses by the durable
/// journal cursor (the §17 `session_history` cursor) for reconnect-stable addressing.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewindAnchor {
    /// Seal off the user turn at `ordinal` and everything after it (the restore/edit case): the
    /// truncation keeps turns `[0, ordinal)`, so the next `StartTurn` re-runs from that user turn.
    UserTurn {
        /// 0-based index of the user turn (a [`ConvView::turns`] index).
        ordinal: u64,
    },
    /// Keep the user turn at `ordinal` but seal off the assistant reply that followed it (the
    /// regenerate case): the truncation keeps turns `[0, ordinal]`.
    ReplyAfter {
        /// 0-based index of the user turn whose reply is being regenerated.
        ordinal: u64,
    },
    /// A raw durable journal cursor (the §17 `session_history` cursor) for clients that address by
    /// cursor rather than ordinal.
    Cursor {
        /// The journal cursor to truncate to (everything after it is sealed off).
        seq: u64,
    },
}

// ---------------------------------------------------------------------------
// §17 event surface: engine -> host
// ---------------------------------------------------------------------------

/// Why a turn started (§17). A background completion is the durable rehydration trigger.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionSource {
    /// A host-owned OS process.
    Process(JobId),
    /// A delegated child engine / job.
    Delegation(JobId),
}

/// How a turn ended (carried in [`TurnSummary`]; the §17.3 leaf form of the management `EndReason`).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// The turn was stopped early because the model stopped making progress — it kept re-issuing the
    /// same tool calls and getting the same results without converging (the §4.2 no-progress guard,
    /// distinct from exhausting the full iteration budget).
    NoProgress,
    /// The turn failed.
    Failed,
}

/// Terminal turn outcome (§17).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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

/// A point-in-time context-window status (the §10 context engine's view), carried on
/// [`AgentEvent::Context`]. `max_tokens` is the model window (the HUD denominator) when the provider
/// declares one; `budget_tokens` is the configured soft compaction budget.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextStatus {
    /// The estimated tokens the assembled context currently uses.
    pub used_tokens: u64,
    /// The model's context window in tokens, when known (the fill denominator).
    pub max_tokens: Option<u64>,
    /// The configured soft compaction budget in tokens, when set.
    pub budget_tokens: Option<u64>,
    /// Whether a compaction just occurred (drops/summarization reduced the context).
    pub compacted: bool,
    /// How many conversation turns the compaction dropped (`0` when none / no compaction).
    pub dropped_turns: u32,
}

/// A read-only projection of one conversation turn, carried in a [`ConvView`] (§17 snapshot reply).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDetail {
    /// The stable renderer discriminator (e.g. a tool name, or `"ansi-stream"`/`"pty"`).
    pub kind: String,
    /// The opaque encoded payload (CBOR by convention), decoded by the consumer per `kind`.
    #[serde(with = "serde_bytes")]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
        #[serde(with = "serde_bytes")]
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
    /// A context-window status update (the §10 context engine's fill + compaction signal) — the
    /// data a GUI renders as a context-fill HUD ("128k / 200k") and a "compacted" toast.
    Context {
        /// Monotonic event sequence number.
        seq: u64,
        /// The current context status.
        status: ContextStatus,
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
    /// The conversation was rewound (conversation-rewind spec §3). A live client drops every turn it
    /// holds with ordinal `>= to_cursor` the moment this arrives, so the UI matches the engine before
    /// the replayed `TurnStarted { trigger: User }` streams in; a reconnecting client reconciles via
    /// `session_history` (which stops at / flags the durable seal — see `JournalPageView::sealed_after`).
    Rewound {
        /// Monotonic event sequence number.
        seq: u64,
        /// Correlation id echoed from [`AgentCommand::RewindTo`].
        request_id: ReqId,
        /// The retained conversation length in turns — the new tail ordinal. The truncation keeps
        /// turns `[0, to_cursor)`; a live client drops every turn it holds with ordinal `>= to_cursor`.
        /// (The engine addresses turns by ordinal; the durable journal seal cursor is surfaced
        /// separately by `session_history` as `JournalPageView::sealed_after`.)
        to_cursor: u64,
        /// The new incarnation epoch fencing stale commits/events from the abandoned turn.
        epoch: u64,
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
            | AgentEvent::Context { seq, .. }
            | AgentEvent::RateLimit { seq, .. }
            | AgentEvent::TurnFinished { seq, .. }
            | AgentEvent::Error { seq, .. }
            | AgentEvent::Steered { seq, .. }
            | AgentEvent::Snapshot { seq, .. }
            | AgentEvent::Rewound { seq, .. } => *seq,
        }
    }
}

// ---------------------------------------------------------------------------
// §17 blocking host requests (human-in-the-loop / delegation)
// ---------------------------------------------------------------------------

/// A blocking, correlated request the engine raises to the host (§17).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// Ask the host to spawn an **attached, non-joining, self-closing** background child (§4.3):
    /// the child is recorded under the parent in the durable tree for audit, but binds no parent
    /// job, so the parent neither suspends nor is woken — the child runs bounded turns against a
    /// constrained background profile and reaches a terminal state on its own. This is the general
    /// post-turn self-improvement seam (background skill review / memory write). Fire-and-forget:
    /// the host returns a [`HostResponseBody::Spawned`] child id immediately.
    Spawn {
        /// The spawn request: which background profile to run and how to seed it.
        spec: SpawnSpec,
    },
}

/// A request to spawn an attached, non-joining background child ([`HostRequestKind::Spawn`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnSpec {
    /// The background-profile kind the host materializes the child from (e.g. `"skill_review"`,
    /// `"memory_review"`). The host owns the kind -> constrained-toolset + review-prompt mapping; an
    /// unknown kind is a no-op (the engine stays free of the side-store/tool specifics).
    pub kind: String,
    /// How the child's conversation is seeded.
    pub seed: SpawnSeed,
}

/// How a spawned background child's conversation is seeded ([`SpawnSpec::seed`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SpawnSeed {
    /// Seed the child with a read-only copy of the parent's conversation at spawn time, so the
    /// review agent sees exactly what just happened. The host reads the parent's durable snapshot.
    #[default]
    FromConversation,
}

/// The host's correlated reply to a [`HostRequest`].
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostResponse {
    /// Correlation id matching the originating request.
    pub request_id: ReqId,
    /// The typed reply body.
    pub body: HostResponseBody,
}

/// The body of a [`HostResponse`], typed per request kind.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// The id of the attached background child a [`HostRequestKind::Spawn`] materialized. Purely
    /// informational (audit/reference): the parent does not wait on it.
    Spawned(SessionId),
    /// An [`Approval`](HostRequestKind::Approval) the host parked **durably** for an operator
    /// (the headless/durable HITL path): the engine must suspend the turn and resume on the
    /// operator's decision (delivered as the wake completion keyed by this id), rather than
    /// proceeding inline. The live path never returns this — it parks for a synchronous human
    /// answer and returns [`Approved`](Self::Approved) instead.
    Deferred(JobId),
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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

/// An **immutable, platform-assigned sender identity** — a Matrix MXID (`@user:hs`), a Telegram user
/// id, etc. NEVER a display name or any user/operator-mutable text.
///
/// Sender allow-listing (the ingest `SenderPolicy`) and attribution key on this, so it must be the
/// stable identifier the platform guarantees, **supplied by the adapter** — never re-derived from a
/// message body or display text (the OpenClaw display-name `allowFrom` substrate). It is deliberately
/// off the wire in this iteration (enforced at the ingest boundary via `Reception`); carrying it onto
/// `Origin`/the log is a separate follow-on.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SenderId(pub String);

impl SenderId {
    /// Construct a sender id from its stable, immutable platform identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The sender id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The reserved identity for a node-internal **loopback** origin that has no external sender
    /// (e.g. a Rooms operator post). A typed, documented constant so no ingest path re-derives a
    /// sender from free text — the whole point of the newtype.
    pub fn local_loopback() -> Self {
        Self("local:loopback".to_string())
    }
}

impl From<&str> for SenderId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for SenderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// The conversational scope an inbound item belongs to — the single input (with the transport) to
/// deterministic session-id derivation, and the per-event attribution carried on the log. This is
/// the daemon analogue of hermes's `build_session_key` input, but carried explicitly (never via
/// env/`ContextVars`). Principal/route handles are opaque strings the originating adapter owns.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
        #[serde(with = "serde_bytes")]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
// Rooms: the internal loopback transport primitive (daemon-rooms-spec.md)
// ---------------------------------------------------------------------------
//
// A Room is an N-participant conversation backed by an *internal loopback transport* — structurally a
// chat transport (the `daemon-matrix` shape) whose "homeserver" is the daemon itself. Its identity is
// `TransportId("room/<RoomId>")` + `OriginScope::Group { chat: <RoomId> }`, so every routing /
// `session_id_for` / `DeliveryTarget` primitive above applies unchanged; the only novel logic is the
// floor-control policy (whose turn it is). DM/session-to-session is a 2-participant Room, a group chat
// is an N-participant Room, and the user observes as a `Spectator`. See `daemon-rooms-spec.md`.

/// Stable identity of a [`Room`-backed](crate) N-participant conversation. The loopback transport
/// instance a Room presents as is `TransportId("room/<RoomId>")`; its routing scope is
/// `OriginScope::Group { chat: <RoomId> }` (mirrors `TransportId`'s hand-rolled newtype shape).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RoomId(pub String);

impl RoomId {
    /// Construct a room id from its stable handle.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The room id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The loopback transport instance this room presents as (`room/<id>`), the `TransportId` the
    /// RoomRouter fans inbound posts out under and the outbound `Projector` re-injects through.
    pub fn transport(&self) -> TransportId {
        TransportId::new(format!("room/{}", self.0))
    }
}

impl From<&str> for RoomId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for RoomId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// One participant of a [`RoomId`]-keyed Room: an agent (or the user) bound to a profile and a
/// resolved per-member session. The membership table maps `(room, member) -> (ProfileRef,
/// SessionId)`; the RoomRouter fans an inbound post out to each member's session via `submit_from`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomMember {
    /// The adapter-opaque member handle within the room (the speaker label / `@name`).
    pub member: String,
    /// The profile this member's session runs under (`None` = the registry's default precedence).
    pub profile: Option<ProfileRef>,
    /// The resolved per-member session id (the engine incarnation this participant drives).
    pub session: SessionId,
}

impl RoomMember {
    /// Construct a room member binding.
    pub fn new(member: impl Into<String>, profile: Option<ProfileRef>, session: SessionId) -> Self {
        Self {
            member: member.into(),
            profile,
            session,
        }
    }
}

/// The floor-control / turn policy of a Room — the single genuinely novel piece of logic the
/// RoomRouter applies before fanning a post out (echo-storm prevention is a `max_turns` budget the
/// router enforces orthogonally to this choice). Each variant decides *whose* `TurnFinished` is
/// re-injected as the next inbound post. The stub set; the policy engine lives in `daemon-rooms`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum RoomPolicy {
    /// Members take the floor in a fixed rotation (the default group-chat shape).
    RoundRobin,
    /// Only an explicitly addressed (mentioned) member opens a turn; others stay observers.
    #[default]
    AddressedOnly,
    /// One moderator member arbitrates who may speak next.
    Moderator {
        /// The member handle holding the floor-granting role.
        profile: String,
    },
    /// No arbitration — every member turns on every post (bounded only by the turn budget).
    FreeForAll,
}

// ---------------------------------------------------------------------------
// Durable transcript blocks (the verifiable journal's chat-entry payload)
// ---------------------------------------------------------------------------

/// Who authored a transcript message block.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
        #[serde(with = "serde_bytes")]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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

/// A unit/session's hierarchy role — the shared roster/tree taxonomy a GUI uses to keep the inbox
/// (`Primary` only) separate from drill-down children, and to keep long-lived managed children
/// stable while coalescing transient-subagent churn. The transport-stable mirror of the store's
/// `SessionRole`.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionRole {
    /// A top-level conversation (the only role in the `TopLevel` roster scope / fleet root).
    #[default]
    Primary,
    /// A long-lived child an agent owns/manages: stable, low churn; always projected into the tree.
    ManagedChild,
    /// A transient/temporary subagent: in the tree but high churn (rapidly created/destroyed), so
    /// consumers may coalesce or filter it.
    EphemeralSubagent,
}

impl SessionRole {
    /// Whether this role is a transient subagent (the churn source consumers may collapse).
    pub fn is_ephemeral(self) -> bool {
        matches!(self, SessionRole::EphemeralSubagent)
    }
}

/// One node in the orchestration tree projection (the GUI's per-unit view). The tree is a flat node
/// list plus per-node `children` ids, so deeper / cross-node nesting can fill in later without a DTO
/// change.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// The profile this unit's engine runs under, when known (GUI agent identity).
    #[serde(default)]
    pub profile: Option<ProfileRef>,
    /// The session id backing this unit, when it maps to one (so a client can join to the roster).
    #[serde(default)]
    pub session: Option<SessionId>,
    /// A human-readable title for this unit/conversation, when known.
    #[serde(default)]
    pub title: Option<String>,
    /// This unit's hierarchy role: a top-level conversation, a long-lived managed child, or a
    /// transient subagent. Lets a client keep stable nodes pinned and collapse ephemeral churn.
    /// `None` on legacy payloads => treat as `Primary`.
    #[serde(default)]
    pub role: Option<SessionRole>,
}

/// The orchestration tree as the GUI/TUI sees it: a flat node list rooted at `root`. `nodes` is
/// served in wire-bounded pages (unit-id order); `next` is the resume cursor to pass back as the
/// tree request's `after` (`None` => last page). `root` rides every page; the id-linked structure
/// reassembles client-side regardless of page boundaries.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeReport {
    /// The root unit id (the node itself), when there is one.
    pub root: Option<UnitId>,
    /// The nodes in this page (at most the wire page bound).
    pub nodes: Vec<UnitNode>,
    /// The resume cursor when more nodes remain (`None` => last page).
    #[serde(default)]
    pub next: Option<String>,
}

/// A transport-stable projection of a unit's management event, for GUI drill-down (decoupled from
/// the supervision `ManageEvent`). Mirrors the per-session poll model: a bounded drain of recent
/// events for one unit.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    /// A delegation/subagent lifecycle signal — the coalescable churn-control event a GUI uses to
    /// render the subtree without polling `tree()`. Ephemeral-subagent spawn/finish is the churn
    /// source, so this carries the child id + role + a running active-child count rather than
    /// per-tick noise; clients filter by `role` (e.g. ignore `EphemeralSubagent`).
    Subagent {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// The child unit this signal is about.
        child: UnitId,
        /// The child's hierarchy role (managed vs ephemeral).
        role: SessionRole,
        /// The child lifecycle phase this signal reports.
        phase: SubagentPhase,
        /// The parent's count of currently-active children after this transition (for a stable
        /// "N running" badge even when individual ephemeral spawns are coalesced).
        active_children: u32,
    },
}

/// A delegation/subagent lifecycle phase (the [`ManageEventView::Subagent`] discriminator).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubagentPhase {
    /// The child was spawned/attached.
    Spawned,
    /// The child reached a terminal outcome and detached.
    Finished,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegation_payloads_round_trip_and_fall_back() {
        // Structured round-trip, including the v2 lifetime + profile fields.
        let input = DelegationInput {
            task: "do the thing".into(),
            attachments: vec!["src/a.rs".into(), "notes.md".into()],
            lifetime: DelegationLifetime::Ephemeral,
            profile: Some("opus".into()),
            detached: true,
        };
        assert_eq!(DelegationInput::decode(&input.encode()), input);
        assert!(DelegationInput::decode(&input.encode()).detached);

        let hash = daemon_common::ContentHash::new([3u8; 32]);
        let result = DelegationResult {
            summary: "done".into(),
            artifacts: vec![daemon_common::BlobRef::new(hash, 9)],
        };
        assert_eq!(DelegationResult::decode(&result.encode()), result);

        // Legacy plain-text payloads (pre-upgrade) decode via the fallback path with defaults.
        let legacy_in = DelegationInput::decode(b"delegated-work");
        assert_eq!(legacy_in.task, "delegated-work");
        assert!(legacy_in.attachments.is_empty());
        assert_eq!(legacy_in.lifetime, DelegationLifetime::Persistent);
        assert!(legacy_in.profile.is_none());
        assert!(!legacy_in.detached, "legacy payloads decode as joining");

        let legacy_out = DelegationResult::decode(b"child:parent/c1");
        assert_eq!(legacy_out.summary, "child:parent/c1");
        assert!(legacy_out.artifacts.is_empty());
    }

    #[test]
    fn delegation_input_pre_upgrade_cbor_defaults_new_fields() {
        // A payload encoded by the pre-v2 shape (task + attachments only) must decode with the new
        // fields at their defaults — jobs enqueued before the upgrade still resolve.
        #[derive(Serialize)]
        struct V1 {
            task: String,
            attachments: Vec<String>,
        }
        let mut buf = Vec::new();
        ciborium::into_writer(
            &V1 {
                task: "old job".into(),
                attachments: vec!["a.txt".into()],
            },
            &mut buf,
        )
        .unwrap();
        let decoded = DelegationInput::decode(&buf);
        assert_eq!(decoded.task, "old job");
        assert_eq!(decoded.attachments, vec!["a.txt".to_string()]);
        assert_eq!(decoded.lifetime, DelegationLifetime::Persistent);
        assert!(decoded.profile.is_none());
        assert!(!decoded.detached, "pre-upgrade payloads default to joining");
    }

    #[test]
    fn user_msg_bytes_round_trip_and_fall_back() {
        // CBOR round-trip through the opaque pending-input channel.
        let msg = UserMsg::new("status update please");
        assert_eq!(UserMsg::decode(&msg.encode()), msg);
        // A producer that queued bare text still resolves via the fallback.
        let text = UserMsg::decode(b"plain text input");
        assert_eq!(text.text, "plain text input");
        assert!(text.attachments.is_empty());
    }

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
