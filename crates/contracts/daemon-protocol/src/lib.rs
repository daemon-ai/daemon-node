//! `daemon-protocol` — the §17 host wire protocol and the typed engine snapshot.
//!
//! Phase 1 carries a *minimal* slice of the §17 surface — enough for `daemon-stub-engine` to speak
//! it and for the durable activation core to be proven — plus the typed [`Snapshot`] the engine
//! produces. The persisted form is CBOR via `ciborium`, emitted as a [`SnapshotBlob`]; the durable
//! layer never sees these typed structs. Depends only on `daemon-common`.
//!
//! The full §17 surface (streaming deltas, tool views, usage/rate-limit telemetry) and the real
//! `daemon-core` §5 `Conversation` arrive in phase 3.

#![forbid(unsafe_code)]

use daemon_common::{Budget, Epoch, JobId, SessionId, SnapshotBlob};
use serde::{Deserialize, Serialize};

/// Correlation id for a request/response pair on the §17 surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReqId(pub u64);

/// A user-authored turn input (placeholder for the richer §5 message type).
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

/// How a turn ended (carried in [`TurnSummary`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndReason {
    /// The turn completed normally.
    Completed,
    /// The engine suspended at a phase boundary to await background work.
    Suspended,
    /// The turn was interrupted.
    Interrupted,
}

/// Terminal turn outcome (§17).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSummary {
    /// Why the turn ended.
    pub end_reason: EndReason,
}

/// Events the engine streams up to the host (§17, core -> host). Each carries a monotonic `seq`.
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

/// The kinds of blocking host request the engine can raise (minimal phase-1 subset).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
pub enum HostResponseBody {
    /// Approval decision.
    Approved(bool),
    /// Free-form input result.
    Input(String),
    /// The id assigned to delegated work.
    Delegated(JobId),
}

/// The trait the host implements so an engine can raise blocking requests (§17). Thin in phase 1.
#[async_trait::async_trait]
pub trait HostRequestHandler: Send + Sync {
    /// Answer a blocking host request.
    async fn request(&self, req: HostRequest) -> HostResponse;
}

// ---------------------------------------------------------------------------
// The typed engine snapshot (lifecycle §2)
// ---------------------------------------------------------------------------

/// A single conversational turn (minimal placeholder for the §5 `Conversation` body).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Role label (e.g. `user`, `assistant`, `system`).
    pub role: String,
    /// The turn text.
    pub text: String,
}

/// The typed conversation body. Phase-1 placeholder; replaced by the `daemon-core` §5 type later.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    /// Ordered turns.
    pub turns: Vec<Turn>,
}

impl Conversation {
    /// Append a turn.
    pub fn push(&mut self, role: impl Into<String>, text: impl Into<String>) {
        self.turns.push(Turn {
            role: role.into(),
            text: text.into(),
        });
    }

    /// Number of turns recorded so far.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

/// A host-owned OS process handle, re-attached by the host on rehydration (lifecycle §2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcHandle {
    /// Opaque host-assigned process key.
    pub key: String,
}

/// A tool identity plus the key the tool uses to reload its own external state (lifecycle §1.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolBinding {
    /// The tool's stable name.
    pub name: String,
    /// The key the tool reloads its own state from.
    pub state_key: String,
}

/// Handles to re-establish on rehydration — never live resources (lifecycle §2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct References {
    /// Delegated child engines, by id (recursive composition).
    pub children: Vec<SessionId>,
    /// Host-owned OS processes, re-attached by the host.
    pub processes: Vec<ProcHandle>,
    /// Tool identities + the keys tools use to reload their own state.
    pub tools: Vec<ToolBinding>,
}

/// The complete, serializable state of one engine incarnation. Nothing else is durable
/// (lifecycle §2). Tool working state and live resources are never included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Stable logical identity (not a live task handle).
    pub session_id: SessionId,
    /// Monotonic epoch; bumped on every suspension; fences stale incarnations.
    pub epoch: Epoch,
    /// The typed conversation body (source of truth).
    pub conversation: Conversation,
    /// Handles to re-establish on rehydration.
    pub references: References,
    /// Outstanding background work this incarnation suspended for.
    pub waiting_for: Vec<JobId>,
}

impl Snapshot {
    /// A fresh snapshot for a newly created session at epoch 0.
    pub fn fresh(session_id: SessionId) -> Self {
        Self {
            session_id,
            epoch: Epoch::ZERO,
            conversation: Conversation::default(),
            references: References::default(),
            waiting_for: Vec::new(),
        }
    }

    /// Encode to the opaque persisted CBOR form.
    pub fn encode(&self) -> Result<SnapshotBlob, daemon_common::DaemonError> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|e| daemon_common::DaemonError::Codec(e.to_string()))?;
        Ok(SnapshotBlob::new(bytes))
    }

    /// Decode from the opaque persisted CBOR form.
    pub fn decode(blob: &SnapshotBlob) -> Result<Self, daemon_common::DaemonError> {
        ciborium::from_reader(blob.as_bytes())
            .map_err(|e| daemon_common::DaemonError::Codec(e.to_string()))
    }
}
