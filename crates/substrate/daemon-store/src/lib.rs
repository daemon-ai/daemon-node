// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-store` — durable persistence primitives for the activation core.
//!
//! The [`SessionStore`] trait is the *sole authority* for durable session state (lifecycle §4
//! invariant #1): snapshots, the completion inbox (idempotent via `UNIQUE(session_id, epoch,
//! job_id)`), the wake/job outboxes, and the monotonic activation lease that fences stale
//! incarnations. Two backends implement it with identical semantics (proven by the same conformance
//! acceptance suite run against both): the default in-memory [`InMemoryStore`] and, behind the
//! `sqlite` feature, the durable [`SqliteStore`] (WAL-mode `rusqlite`, including the trace journal).
//! Depends only on `daemon-common`.
//!
//! Snapshots are handled here only as opaque CBOR [`SnapshotBlob`]s — the typed `Snapshot` lives in
//! `daemon-protocol`, keeping this crate protocol-free (lifecycle §2; layout §3 DAG).
//!
//! See `docs/specs/daemon-lifecycle-persistence.md`.

#![forbid(unsafe_code)]
// Phase 4: test code may use raw fs/reqwest/Command; the --lib pass still guards production.
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::disallowed_types))]

use async_trait::async_trait;
use daemon_common::{
    ContentHash, DaemonError, Epoch, FenceToken, JobId, JournalStreamId, MerkleRoot, PartitionId,
    ProfileRef, SessionId, SnapshotBlob, UsageDelta,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStore;

/// The durable status of a session record (lifecycle §5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    /// A live incarnation is (or was) running; recoverable from the last snapshot.
    Active,
    /// Suspended at a phase boundary awaiting a background job.
    Suspended {
        /// The job this session is waiting on.
        job_id: JobId,
    },
    /// A completion is durably recorded; the session is resumable.
    Ready,
    /// The session reached a terminal state.
    Completed,
}

/// One durable session row (lifecycle §5).
#[derive(Clone, Debug)]
pub struct SessionRecord {
    /// Stable logical identity.
    pub session_id: SessionId,
    /// Owning partition.
    pub partition: PartitionId,
    /// Monotonic incarnation epoch.
    pub epoch: Epoch,
    /// Durable status.
    pub status: SessionStatus,
    /// The last persisted snapshot (opaque CBOR).
    pub snapshot: SnapshotBlob,
    /// The current (highest) fencing token granted for this session.
    pub fence: FenceToken,
}

/// Host-level per-session metadata kept beside the snapshot: which profile the session resolves its
/// engine from (`bound_profile`) and an opaque per-session overlay blob (the host's CBOR-encoded
/// `SessionOverlay` — model/provider/tools/approval overrides). The store treats the overlay as
/// opaque bytes (it never parses the protocol), so this stays protocol-free. The resolver reads it
/// at engine construction, so a live override is **restored on rehydration** rather than lost on
/// restart.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMeta {
    /// The profile this session binds its engine to (`None` = the node's active default).
    pub bound_profile: Option<ProfileRef>,
    /// Opaque CBOR of the host's `SessionOverlay` (empty = no overlay recorded).
    pub overlay: Vec<u8>,
    /// A human-readable conversation title (`None` until set/generated). Surfaced on the wire
    /// `SessionInfo` for the GUI roster; generation is deferred (the field is the foundation).
    #[serde(default)]
    pub title: Option<String>,
    /// Unix-millis of the last inbound/outbound activity on this session, for roster sort
    /// (`None` until first stamped). Stamped by the host on submit/append.
    #[serde(default)]
    pub last_activity_ms: Option<u64>,
    /// This session's hierarchy role relative to its parent: a top-level conversation, a long-lived
    /// managed child, or a transient subagent. Drives the GUI roster scope (`Primary` only in the
    /// inbox) and tree churn handling. `None` on legacy rows => treated as `Primary`.
    #[serde(default)]
    pub role: Option<SessionRole>,
    /// The parent session id, when this is a child/subagent (`None` for a `Primary`).
    #[serde(default)]
    pub parent: Option<SessionId>,
    /// Whether the operator pinned this conversation to the top of the roster (GUI session action).
    /// Pinned conversations sort ahead of the activity order; `false` on legacy rows.
    #[serde(default)]
    pub pinned: bool,
    /// Whether the operator archived this conversation (GUI session action). Archived conversations
    /// drop out of the default (`TopLevel`/per-agent) roster scopes and surface only under the
    /// explicit archived scope; `false` on legacy rows.
    #[serde(default)]
    pub archived: bool,
    /// The cron job that fired this session, when it is a scheduled-job run (I15). The host stamps
    /// this on the isolated `cron_{id}_{ts}` session the cron worker materializes; the incarnation
    /// reads it to set `TurnTrigger::Scheduled { job }` before the first turn. `None` for every
    /// non-cron session (and legacy rows).
    #[serde(default)]
    pub scheduled_job: Option<JobId>,
    /// The session-activation generation (L2 resync). The host reads this in `ensure()` to stamp the
    /// fresh in-memory `MergedLog` and persists `+1`, so each (re)activation - including after a
    /// daemon restart, since this sidecar is durable while the live log is not - yields a strictly
    /// greater epoch. Clients track `(epoch, seq)` to detect a generation change and re-baseline from
    /// the durable journal. `0` for the first activation / legacy rows.
    #[serde(default)]
    pub activation_epoch: u64,
    /// The `user_id` of the principal that owns this session (Auth 4 ownership). Stamped at every
    /// creation path (interactive submit / durable assign from the request principal; delegation,
    /// background, and cron children inherit their parent/job owner). `None` on legacy rows and on
    /// system/unattributed sessions — visible only to a `SessionSeeAll` holder, never to a peer
    /// user. The store treats it as an opaque key (the host enforces the ownership policy).
    #[serde(default)]
    pub owner: Option<String>,
    /// Unix-millis this session reached a terminal state, stamped by
    /// [`SessionStore::mark_completed`] in the same transaction as the status flip (re-stamped if a
    /// resumed session completes again). The ephemeral-subagent reaper's grace clock. `None` for
    /// non-terminal sessions and legacy rows (which are therefore never reaped — forward-looking).
    #[serde(default)]
    pub terminal_ms: Option<u64>,
}

/// A session's hierarchy role (the GUI roster/tree taxonomy). `Primary` conversations are the inbox;
/// child roles are reached only by walking the tree. The `ManagedChild` vs `EphemeralSubagent` split
/// lets clients keep long-lived children stable while coalescing transient-subagent churn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionRole {
    /// A top-level conversation (the only role listed in the `TopLevel` roster scope).
    #[default]
    Primary,
    /// A long-lived child an agent owns/manages; stable, low churn; always in the tree.
    ManagedChild,
    /// A transient/temporary subagent; in the tree but high churn (rapidly created/destroyed).
    EphemeralSubagent,
}

/// The lifetime an agent declares when delegating a child: a long-lived managed child vs a transient
/// subagent. The source of truth for the [`SessionRole`] child distinction, recorded at the
/// delegation seam (today every child is created identically, with no managed-vs-ephemeral marker).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChildLifetime {
    /// A long-lived child the parent manages (becomes [`SessionRole::ManagedChild`]).
    #[default]
    Persistent,
    /// A transient subagent spun up for a bounded task (becomes [`SessionRole::EphemeralSubagent`]).
    Ephemeral,
}

impl ChildLifetime {
    /// The hierarchy [`SessionRole`] a child created under this lifetime takes: a managed (persistent)
    /// child is a [`SessionRole::ManagedChild`]; a transient one is a [`SessionRole::EphemeralSubagent`].
    /// This is the seam that derives the child's roster/tree role from the parent's delegation intent.
    pub fn role(self) -> SessionRole {
        match self {
            ChildLifetime::Persistent => SessionRole::ManagedChild,
            ChildLifetime::Ephemeral => SessionRole::EphemeralSubagent,
        }
    }
}

/// A durable chat→session routing pin (daemon-event-io-spec §5.9): binds a canonical inbound-origin
/// `key` to an explicit `session_id` (+ optional `profile`), overriding the deterministic
/// `session_id_for` derivation in the host's routing registry. The store stays protocol-free, so the
/// full protocol descriptor (the `Origin` + isolation policy) rides through as the opaque
/// host-encoded `descriptor` blob (the host round-trips it back to a GUI); `key`/`session_id`/
/// `profile` are the typed columns the host indexes and builds the live pin map from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRoute {
    /// The canonical origin key (host-computed; the primary key for upsert/lookup/delete).
    pub key: String,
    /// The session this origin is pinned to.
    pub session_id: SessionId,
    /// An explicit profile to run the pinned session under (`None` = fall through to the registry's
    /// deterministic profile precedence).
    pub profile: Option<ProfileRef>,
    /// The opaque host descriptor (CBOR of the protocol `Origin` + isolation) for round-trip.
    pub descriptor: Vec<u8>,
}

/// A durable Room/Chat row (daemon-rooms-spec.md): a first-class N-participant conversation backed by
/// the internal loopback transport. Like [`ChatRoute`] the store stays protocol-free — the typed
/// floor-control policy and any extra metadata ride as the opaque host-encoded `descriptor` blob (the
/// CBOR of the wire `Room`); `id` / `name` / `policy` are the typed columns the host indexes and
/// lists. Membership lives in the companion [`RoomMember`] rows (the `room_members` table).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Room {
    /// The room id (primary key for upsert/lookup/delete).
    pub id: String,
    /// A human-readable room name, when set.
    pub name: Option<String>,
    /// The floor-control policy tag (mirrored from the descriptor for column-level listing; the host
    /// treats `descriptor` as authoritative).
    pub policy: String,
    /// The opaque host descriptor (CBOR of the wire `Room` metadata) for round-trip.
    pub descriptor: Vec<u8>,
}

/// A durable Room membership row (daemon-rooms-spec.md): one participant of a [`Room`], binding a
/// `member` handle to a `profile` + per-member `session_id`. Keyed by `(room_id, member)`, mirroring
/// the typed-columns shape of [`ChatRoute`] (the store stays protocol-free).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomMember {
    /// The room this membership belongs to (part of the `(room_id, member)` primary key).
    pub room_id: String,
    /// The adapter-opaque member handle within the room (part of the primary key).
    pub member: String,
    /// The profile this member's session runs under (`None` = registry default precedence).
    pub profile: Option<ProfileRef>,
    /// The resolved per-member session id.
    pub session_id: SessionId,
}

/// A durable manually-registered ACP agent catalog entry (I7): the operator-persisted half of the
/// ACP discovery catalog (auto-discovered builtins are re-probed each scan and need no persistence).
/// `entry` is the opaque host-encoded CBOR of the wire `AcpAgentEntry`; the store stays protocol-free.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpEntry {
    /// The agent catalog key (display name; the primary key for upsert/lookup/delete).
    pub name: String,
    /// The opaque host descriptor (CBOR of the wire `AcpAgentEntry`).
    pub entry: Vec<u8>,
}

/// Bounded retention for cron run history: the most recent N runs kept per job (both backends).
pub const CRON_RUN_RETENTION: usize = 50;

/// A durable scheduled-job row (I15). The store stays protocol-free: the typed schedule policy
/// (overlap/catch-up) and the full spec ride as the opaque host-encoded `spec` CBOR blob (the wire
/// `CronSpec`), while the columns the scheduler indexes on — `id`, `next_fire_unix` (the due-query
/// key), `paused` (the due filter), and the run bookkeeping — are typed. `schedule` is duplicated
/// out of the spec as a column purely so a backend could re-derive next-fire without decoding the
/// blob; the host treats `spec` as authoritative.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredCronJob {
    /// The opaque job id (primary key for upsert/lookup/delete).
    pub id: String,
    /// The human schedule expression (mirrored from the spec for column-level queries).
    pub schedule: String,
    /// The opaque host descriptor (CBOR of the wire `CronSpec`).
    pub spec: Vec<u8>,
    /// Unix seconds of the next scheduled fire (`None` = not yet computed / one-shot exhausted). The
    /// `cron_due` query keys on this.
    pub next_fire_unix: Option<u64>,
    /// Whether the job is paused (excluded from `cron_due`).
    pub paused: bool,
    /// Unix seconds the job last fired, when it has.
    pub last_run_unix: Option<u64>,
    /// Whether the last completed run succeeded, when one has completed.
    pub last_ok: Option<bool>,
    /// A rendered detail of the last run (error text or summary), when present.
    pub last_detail: Option<String>,
    /// How many times the job has fired (for `repeat` accounting / auto-delete).
    pub fire_count: u32,
    /// Unix seconds the job was created.
    pub created_unix: u64,
    /// The `user_id` of the principal that created this scheduled job (Auth 4 ownership). The cron
    /// worker stamps it onto each `cron_{id}_{ts}` session it materializes, so a scheduled run is
    /// owned by (and visible to) its creator. `None` on legacy rows / system jobs.
    #[serde(default)]
    pub owner: Option<String>,
}

/// One durable recorded run of a scheduled job (I15). Keyed by `job_id` (the wire `CronRun` omits it
/// — the store indexes runs under their job). Append-only with bounded retention.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredCronRun {
    /// The job this run belongs to.
    pub job_id: String,
    /// Unix seconds the run started.
    pub started_unix: u64,
    /// Unix seconds the run finished, when it has completed.
    pub finished_unix: Option<u64>,
    /// Whether the run succeeded.
    pub ok: bool,
    /// A rendered outcome detail, when present.
    pub detail: Option<String>,
    /// The isolated `cron_{id}_{ts}` session the run fired, when an agent turn was materialized.
    pub session: Option<SessionId>,
    /// Whether the run was an explicit `cron_trigger` ("run now") rather than a scheduled fire.
    pub manual: bool,
}

/// A durable consent-first cron suggestion (I15): a catalog starter or filled blueprint awaiting an
/// operator decision. `spec` is the opaque host-encoded CBOR of the wire `CronSpec` that
/// `cron_create` runs on accept. `dedup_key` is unique — once accepted/dismissed, a suggestion with
/// the same key is never re-offered. `status` is the host-encoded `SuggestionStatus`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredCronSuggestion {
    /// The opaque suggestion id (primary key).
    pub id: String,
    /// A short title for the proposal.
    pub title: String,
    /// A human description of what the job does.
    pub description: String,
    /// Where the suggestion came from (e.g. `"catalog"`, `"blueprint"`).
    pub source: String,
    /// The opaque host descriptor (CBOR of the wire `CronSpec`) to create on accept.
    pub spec: Vec<u8>,
    /// A stable key; once accepted/dismissed, the same key is never re-offered (unique).
    pub dedup_key: String,
    /// The host-encoded lifecycle status (`"pending"` / `"accepted"` / `"dismissed"`).
    pub status: String,
    /// Unix seconds the suggestion was created.
    pub created_unix: u64,
}

/// A background-job command enqueued on the durable job outbox (lifecycle §5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobCommand {
    /// Stable job identity (deterministic per `(session, epoch)` so re-enqueues dedupe).
    pub job_id: JobId,
    /// The session that delegated the work.
    pub session_id: SessionId,
    /// The epoch at which the work was delegated (part of the idempotency key).
    pub epoch: Epoch,
    /// Opaque job payload.
    pub payload: Vec<u8>,
    /// The lifetime the delegating parent declared for the child this job materializes (managed vs
    /// transient subagent). The source of truth for the child's [`SessionRole`]. Defaults to
    /// `Persistent` for legacy jobs and the current orchestrate path (which spawns long-lived
    /// managed children); the ephemeral-subagent producer is forward-looking.
    #[serde(default)]
    pub lifetime: ChildLifetime,
    /// The pre-minted child session id for a **detached** (`enqueue_detached_job`) job, so the fleet
    /// worker materializes the child at a store-chosen unique `{parent}/d{n}` id rather than deriving
    /// `{parent}/c{epoch}`. `None` for an ordinary joining delegation (the worker derives the id).
    #[serde(default)]
    pub child: Option<SessionId>,
}

/// A durable background-job completion, applied idempotently per `(session, epoch, job)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobCompletion {
    /// The session the completion is for.
    pub session_id: SessionId,
    /// The epoch the originating job was delegated at.
    pub epoch: Epoch,
    /// The job that completed.
    pub job_id: JobId,
    /// Opaque completion payload.
    pub payload: Vec<u8>,
}

/// A durable **completion notice** for a detached (`enqueue_detached_job`) child: unlike a
/// [`JobCompletion`] it never fulfills a parent job (there is no `waiting_for`/`completion_inbox`
/// entry to satisfy). It is drained off the notice outbox by the node's notice worker, which decodes
/// the opaque `payload` (a CBOR [`DelegationResult`](daemon_protocol) — the child's summary + any
/// artifacts) and injects a `[subagent {child} completed] {summary}` reactive turn into the parent.
/// Pushed by [`SessionStore::mark_completed`] in the terminal transaction when the child carries a
/// completion-notice edge ([`SessionStore::bind_completion_notice`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionNotice {
    /// The parent session the notice is delivered to (as a fresh reactive turn).
    pub parent: SessionId,
    /// The detached child that reached a terminal state.
    pub child: SessionId,
    /// The opaque completion payload (a CBOR `DelegationResult`; the legacy `child:{id}` marker for a
    /// child that produced no structured result).
    pub payload: Vec<u8>,
}

/// What a session activation loads: snapshot + unapplied completions, under a fencing token
/// (lifecycle §5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Activation {
    /// The last persisted snapshot (opaque CBOR).
    pub snapshot: SnapshotBlob,
    /// Completions recorded since the snapshot, not yet applied.
    pub unapplied: Vec<JobCompletion>,
    /// The fencing token the activation must commit under.
    pub fence: FenceToken,
}

/// A checkpoint write: the new snapshot for a session at a bumped epoch (lifecycle §5).
///
/// The store sees only ids + opaque bytes, never the typed `Snapshot`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The session being checkpointed.
    pub session_id: SessionId,
    /// The epoch the snapshot was taken at (post-bump).
    pub epoch: Epoch,
    /// The serialized snapshot.
    pub snapshot: SnapshotBlob,
    /// The opaque completion payload to record when this checkpoint marks a delegated child's
    /// terminal completion (daemon-content-transfer-spec.md Phase 2a: a CBOR `DelegationResult` -
    /// summary + artifact refs). `None` falls back to the legacy `child:{id}` marker.
    #[serde(default)]
    pub completion_payload: Option<Vec<u8>>,
}

impl Checkpoint {
    /// A checkpoint with no completion payload (the common case: a suspension/park checkpoint, or a
    /// completion that carries no structured result).
    pub fn new(session_id: SessionId, epoch: Epoch, snapshot: SnapshotBlob) -> Self {
        Self {
            session_id,
            epoch,
            snapshot,
            completion_payload: None,
        }
    }

    /// Attach a structured completion payload (used when a delegated child completes).
    pub fn with_completion_payload(mut self, payload: Option<Vec<u8>>) -> Self {
        self.completion_payload = payload;
        self
    }
}

/// A durable parked edit-approval request (§12 HITL): a gated tool action (an fs edit, a dangerous
/// shell command) that a headless/dormant session suspended on, awaiting an operator decision. It
/// is the store-side mirror of the engine's `Snapshot::pending_approvals` entry, kept as its own
/// durable row so the operator can *list* what is pending ([`SessionStore::pending_approvals_of`])
/// and *answer* it ([`SessionStore::answer_approval`]) across restarts. Analogous to a `delegations`
/// edge, but its completion is supplied by an operator decision, not a child's terminal state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParkedApproval {
    /// The session that parked the request.
    pub session_id: SessionId,
    /// The request id (matches the engine's `PendingApproval.job_id`; the completion fulfills it).
    pub job_id: JobId,
    /// The epoch the session suspended at (the completion's idempotency epoch).
    pub epoch: Epoch,
    /// A human-readable summary of the proposed action (the approval prompt).
    pub prompt: String,
    /// The target path, when the action is a file edit (`None` for a non-path action).
    pub path: Option<String>,
    /// The operator's decision once answered (`None` while still pending; `Some(true)` = allow).
    pub decision: Option<bool>,
}

/// Errors surfaced by a [`SessionStore`].
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A stale incarnation attempted to commit (lost the lease).
    #[error("fenced: holder token {have} is stale (current is {current})")]
    Fenced {
        /// The token the caller presented.
        have: u64,
        /// The current (highest) token.
        current: u64,
    },
    /// The session does not exist.
    #[error("session not found: {0}")]
    NotFound(SessionId),
    /// A test-injected crash boundary fired.
    #[error("injected fault at {0:?}")]
    Fault(FaultPoint),
    /// Wraps the shared base error.
    #[error(transparent)]
    Common(#[from] DaemonError),
}

/// A serializable form of [`StoreError`] for crossing a placement cut (phase 5).
///
/// [`StoreError`] is not `Serialize` (it carries a `thiserror` source and the test-only
/// [`FaultPoint`]). When the parent's store is brokered to an out-of-process child, the store's
/// verdict — crucially [`StoreError::Fenced`] — must round-trip across the wire so the child sees
/// the same fencing decision it would in-process. `daemon-host` (de)serializes this on the cut.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoreErrorWire {
    /// A stale incarnation attempted to commit (lost the lease).
    Fenced {
        /// The token the caller presented.
        have: u64,
        /// The current (highest) token.
        current: u64,
    },
    /// The session does not exist.
    NotFound(SessionId),
    /// A fault boundary fired (test-only crash simulation), rendered as text.
    Fault(String),
    /// Any other failure, rendered as text.
    Other(String),
}

impl From<&StoreError> for StoreErrorWire {
    fn from(e: &StoreError) -> Self {
        match e {
            StoreError::Fenced { have, current } => StoreErrorWire::Fenced {
                have: *have,
                current: *current,
            },
            StoreError::NotFound(id) => StoreErrorWire::NotFound(id.clone()),
            StoreError::Fault(point) => StoreErrorWire::Fault(format!("{point:?}")),
            StoreError::Common(inner) => StoreErrorWire::Other(inner.to_string()),
        }
    }
}

impl StoreErrorWire {
    /// Reconstruct a [`StoreError`] from its wire form on the far side of a cut.
    pub fn into_store_error(self) -> StoreError {
        match self {
            StoreErrorWire::Fenced { have, current } => StoreError::Fenced { have, current },
            StoreErrorWire::NotFound(id) => StoreError::NotFound(id),
            StoreErrorWire::Fault(msg) => StoreError::Common(DaemonError::Fault(msg)),
            StoreErrorWire::Other(msg) => StoreError::Common(DaemonError::Other(msg)),
        }
    }
}

/// A point-in-time view of durable queue depths and session count, for the host's Metrics/health
/// resident service and test assertions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreStats {
    /// Pending background jobs on the durable job outbox.
    pub pending_jobs: usize,
    /// Pending wake hints on the durable wake outbox.
    pub pending_wakes: usize,
    /// Total durable session records.
    pub sessions: usize,
}

/// One hit from [`SessionStore::search_sessions`]: the matching session, its indexed title, and a
/// highlighted snippet of the matching body text (matched terms wrapped in `[`…`]`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSearchHit {
    /// The session that matched.
    pub session_id: SessionId,
    /// The session's indexed title (empty when none was indexed).
    pub title: String,
    /// A highlighted excerpt of the matching body text.
    pub snippet: String,
}

/// Unix-millis now — the store's terminal-state clock ([`SessionMeta::terminal_ms`]).
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build a highlighted excerpt of `body` around the first occurrence of `needle` (lowercased), with
/// the match wrapped in `[`…`]` and `…` elision — the in-memory analogue of SQLite FTS5 `snippet()`.
fn snippet_around(body: &str, needle: &str) -> String {
    let lower = body.to_lowercase();
    let Some(pos) = lower.find(needle) else {
        return body.chars().take(64).collect();
    };
    const PAD: usize = 24;
    let start = body[..pos]
        .char_indices()
        .rev()
        .nth(PAD)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let match_end = pos + needle.len();
    let tail_len = body[match_end..]
        .char_indices()
        .nth(PAD)
        .map(|(i, _)| i)
        .unwrap_or(body.len() - match_end);
    let end = match_end + tail_len;
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(&body[start..pos]);
    out.push('[');
    out.push_str(&body[pos..match_end]);
    out.push(']');
    out.push_str(&body[match_end..end]);
    if end < body.len() {
        out.push('…');
    }
    out
}

/// One durable, append-only journal entry, keyed `(stream, segment, seq)`.
///
/// The store sees only opaque bytes — a deterministically-encoded (dCBOR) Gordian Envelope built by
/// `daemon-telemetry` — plus its [`ContentHash`]. This keeps `daemon-store` free of the crypto
/// stack (layout §3 DAG). The entry's payload is either a coarse management record or a coalesced
/// finished chat block; the store never distinguishes them (the envelope `kind` does).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Monotonic per-`(stream, segment)` sequence number.
    pub seq: u64,
    /// Opaque deterministic-CBOR bytes of the entry's Gordian Envelope.
    pub bytes: Vec<u8>,
    /// The content hash of `bytes` (the envelope's digest).
    pub content_hash: ContentHash,
}

/// The committed root of a journal segment: the per-`(stream, segment)` Merkle root and signature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedRoot {
    /// The digest-tree root folding every entry plus the prior segment's root (rolling chain).
    pub root: MerkleRoot,
    /// An opaque detached signature over the root (ed25519, produced by `daemon-telemetry`).
    pub signature: Vec<u8>,
}

/// A loaded journal segment: its append-only entries plus the committed root, if the segment has
/// been sealed at its turn/incarnation boundary. The seal-recompute path loads exactly one segment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSegment {
    /// The stream this segment belongs to.
    pub stream: JournalStreamId,
    /// The monotonic segment index this covers (a turn for streaming units, an incarnation for the
    /// durable path).
    pub segment: u64,
    /// The append-only entries, in `seq` order.
    pub entries: Vec<TraceEntry>,
    /// The committed root + signature, once sealed; `None` while the segment is still open.
    pub committed: Option<CommittedRoot>,
}

/// One entry as returned by the cursor-paged journal read: the stream-monotonic `cursor` (the
/// pagination key), the `segment` it belongs to, and the opaque entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntry {
    /// The stream-monotonic cursor; `load_journal` returns entries with `cursor > after_cursor`.
    pub cursor: u64,
    /// The segment this entry belongs to.
    pub segment: u64,
    /// The opaque journal entry.
    pub entry: TraceEntry,
}

/// A page of the verifiable journal for one stream: entries past a cursor, the sealed roots of the
/// segments they cover (for verification), and the pagination cursors. Non-destructive — repeated
/// reads from the same `after_cursor` return the same page (unlike the live drain `poll`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalPage {
    /// The entries in cursor order.
    pub entries: Vec<JournalEntry>,
    /// The committed roots of the segments covered by `entries`, as `(segment, root)`.
    pub segment_roots: Vec<(u64, CommittedRoot)>,
    /// The cursor to pass as `after_cursor` on the next read (the last entry's cursor, or the
    /// input `after_cursor` when the page is empty).
    pub next_cursor: u64,
    /// The highest cursor currently stored for the stream (so a reader knows how far it can scroll).
    pub head_cursor: u64,
}

/// An append-only conversation-rewind seal recorded against a journal stream (conversation-rewind
/// spec §6). The journal stays a complete audit log; the seal marks that a rewind occurred at
/// `seal_cursor` (the stream head at rewind time) retaining `retained_turns` conversation turns, so
/// `session_history` can surface the boundary (`JournalPageView::sealed_after`) and a reconnecting
/// client reconciles against the engine's truncated conversation (the authoritative `Snapshot`/
/// `ConvView`). The latest seal for a stream is the active one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalSeal {
    /// The stream head cursor at the moment of the rewind: everything already journaled belongs to
    /// the pre-rewind audit history.
    pub seal_cursor: u64,
    /// The number of conversation turns the engine retained (turns `[0, retained_turns)` survive).
    pub retained_turns: u64,
    /// The incarnation epoch the rewind bumped to (fences stale commits/events).
    pub epoch: u64,
    /// Unix seconds when the seal was recorded.
    pub recorded_unix: u64,
}

/// A crash boundary the in-memory store can be armed to fail at, for acceptance test #2.
///
/// These model the durable boundaries enumerated in `rust-substrate-evaluation.md` §6 test #2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaultPoint {
    /// Abort the checkpoint transaction before any snapshot is written.
    BeforeSnapshot,
    /// Crash after the snapshot is durable but before the job outbox is written.
    AfterSnapshot,
    /// Crash after the job is enqueued but before the activation task exits.
    AfterJobOutbox,
    /// Crash after the completion is durably inserted but before the wake is published.
    BeforeWakePublish,
}

/// The durable session store — the sole authority for activation state (lifecycle §4–§5).
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Create a fresh session row in `Ready` state with an initial snapshot.
    async fn create_session(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError>;

    /// Acquire/renew the activation lease for a session; returns a fresh monotonic fencing token
    /// and marks the session `Active` (lifecycle §5).
    async fn acquire_activation_lease(&self, id: &SessionId) -> Result<FenceToken, StoreError>;

    /// Load the snapshot + unapplied completions for activation, under a fencing token
    /// (lifecycle §5).
    async fn load_for_activation(
        &self,
        id: &SessionId,
        fence: FenceToken,
    ) -> Result<Activation, StoreError>;

    /// Atomically write the snapshot and enqueue the background job, bumping the epoch and marking
    /// the session `Suspended`. Fenced: only the highest token may commit (lifecycle §5).
    async fn checkpoint_and_enqueue(
        &self,
        checkpoint: Checkpoint,
        job: JobCommand,
        fence: FenceToken,
    ) -> Result<(), StoreError>;

    /// Persist a terminal snapshot and mark the session `Completed`. Fenced.
    async fn mark_completed(
        &self,
        checkpoint: Checkpoint,
        fence: FenceToken,
    ) -> Result<(), StoreError>;

    /// Record a completion durably and enqueue a `Wake` (one transaction). Idempotent per
    /// `(session, epoch, job)` (lifecycle §5; invariants #2, #3).
    async fn record_completion_and_wake(&self, c: &JobCompletion) -> Result<(), StoreError>;

    /// Bind a delegated child session to the parent `job` whose completion it fulfills: when the
    /// child reaches a terminal state ([`Self::mark_completed`]), the store records a completion for
    /// `job` and wakes the parent — in the *same* durable transaction, so a crash between the two
    /// cannot orphan the parent. This is the durable tree edge that makes nested delegation
    /// recursive and recovery-safe at any depth. Default: a no-op (a non-authoritative proxy store).
    async fn bind_delegation(&self, _child: SessionId, _job: JobCommand) -> Result<(), StoreError> {
        Ok(())
    }

    /// Enqueue a **detached** background job onto the durable job outbox *without* a checkpoint or a
    /// suspension — the seam behind the orchestrate `spawn wait:false` mode. Unlike
    /// [`checkpoint_and_enqueue`](Self::checkpoint_and_enqueue) the delegating parent is neither
    /// snapshotted nor moved to `Suspended`: its turn keeps running. The store mints a **unique**
    /// child id `{parent}/d{n}` via a per-parent monotonic sequence (so a turn-retry re-enqueue
    /// produces a distinct child rather than colliding), stamps it onto the job's
    /// [`JobCommand::child`], enqueues the job, and returns the minted id. Pair with
    /// [`bind_completion_notice`](Self::bind_completion_notice) so the child's terminal completion is
    /// delivered to the parent as a notice. Default: a no-op returning `job.session_id` (a
    /// non-authoritative proxy store).
    async fn enqueue_detached_job(&self, job: JobCommand) -> Result<SessionId, StoreError> {
        Ok(job.session_id)
    }

    /// Record a **completion-notice** edge: `child` is a detached background child of `parent` whose
    /// terminal completion must be delivered to the parent as a fresh reactive turn (a notice), NOT
    /// as a job completion. This ALSO records the child under `parent` in the tree/child index (so
    /// `status`/tree see it) but — unlike [`bind_delegation`](Self::bind_delegation) — binds no
    /// parent job, so [`mark_completed`](Self::mark_completed) never records a `completion_inbox`
    /// entry or wakes the parent through the `waiting_for` rail; it pushes a [`CompletionNotice`]
    /// instead. Idempotent. Default: a no-op (a non-authoritative proxy store).
    async fn bind_completion_notice(
        &self,
        _child: &SessionId,
        _parent: &SessionId,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Pop the next pending [`CompletionNotice`], if any (the node's notice-worker side). Default:
    /// `None` (a non-authoritative proxy store / a store without the notice seam).
    async fn dequeue_completion_notice(&self) -> Option<CompletionNotice> {
        None
    }

    /// Record an **attached, non-joining** parent->child edge for audit (§4.3): the child appears
    /// under `parent` in the tree projection labeled `work_label`, but — unlike [`bind_delegation`]
    /// — binds *no* parent job. So when the child reaches a terminal state ([`Self::mark_completed`])
    /// the store finds no delegation to fulfill and never wakes the parent: the child self-closes.
    /// This is the durable edge behind the engine-native background spawn (skill/memory review).
    /// Default: a no-op (a non-authoritative proxy store).
    ///
    /// [`bind_delegation`]: Self::bind_delegation
    async fn record_child_edge(
        &self,
        _parent: SessionId,
        _child: SessionId,
        _work_label: String,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// The child sessions `parent` delegated, in delegation order — the durable parent->child edge
    /// the management-tree projection walks. Default: empty (a non-authoritative proxy store).
    async fn children_of(&self, _parent: &SessionId) -> Vec<SessionId> {
        Vec::new()
    }

    /// Enqueue a bare wake hint for `id` (no completion) so the wake-outbox dispatcher activates it.
    /// Used to kick a freshly-created durable child session into its first turn. Default: no-op (a
    /// non-authoritative proxy store relies on its authoritative peer's dispatcher).
    async fn enqueue_wake(&self, _id: SessionId) {}

    /// Append a durable **pending input** for `id` — an opaque payload (a CBOR `UserMsg`) the host
    /// wants folded into the session's conversation the next time it activates. This is the durable
    /// inbound-input seam for sessions on the *activation* lifecycle (which `SessionApi::submit`
    /// cannot reach): a background process-exit notification, a message to a delegated child (the
    /// orchestrate tool's `send` verb), etc. The durable incarnation drains these at hydrate
    /// ([`take_session_inputs`]); pair with [`enqueue_wake`](Self::enqueue_wake) so the wake
    /// dispatcher runs the turn. FIFO per session. Default: no-op (a non-authoritative proxy
    /// store).
    ///
    /// [`take_session_inputs`]: Self::take_session_inputs
    async fn enqueue_session_input(&self, _id: &SessionId, _input: Vec<u8>) {}

    /// Drain (return and delete) every pending input for `id`, in enqueue (FIFO) order. Called by
    /// the durable incarnation at hydrate, before the turn runs, so a queued message lands in
    /// exactly one incarnation. Default: empty (a store without the pending-input seam / a
    /// non-authoritative proxy store).
    async fn take_session_inputs(&self, _id: &SessionId) -> Vec<Vec<u8>> {
        Vec::new()
    }

    /// Atomically checkpoint a session suspended on a §12 edit-approval decision and durably record
    /// its parked approval row(s) — **without** enqueuing a runnable background job (unlike
    /// [`checkpoint_and_enqueue`]). The session goes `Suspended` on the first approval's `job_id` and
    /// stays dormant until an operator [`answer_approval`](Self::answer_approval) wakes it. Fenced.
    /// Default: a no-op (a non-authoritative proxy store).
    ///
    /// [`checkpoint_and_enqueue`]: Self::checkpoint_and_enqueue
    async fn park_approval(
        &self,
        _checkpoint: Checkpoint,
        _approvals: Vec<ParkedApproval>,
        _fence: FenceToken,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Record an operator's decision for a parked approval and wake the session in one transaction:
    /// stamp the parked row's `decision`, record a [`JobCompletion`] for its `job_id` (payload
    /// `allow`/`deny`) so the rehydrated engine resolves the gated tool call, and publish a wake.
    /// Idempotent per `(session, epoch, job)` (a redelivered answer is a no-op). Returns `true` if a
    /// matching pending approval was found and answered. Default: `false` (no such row).
    async fn answer_approval(
        &self,
        _session: &SessionId,
        _job_id: &JobId,
        _allow: bool,
    ) -> Result<bool, StoreError> {
        Ok(false)
    }

    /// List the still-pending (unanswered) parked approvals — for one `session` when given, else
    /// across all sessions — backing the operator-facing `ApprovalsPending` surface. Default: empty.
    async fn pending_approvals_of(&self, _session: Option<&SessionId>) -> Vec<ParkedApproval> {
        Vec::new()
    }

    /// The work label `child` was delegated with (the parent job's payload as text), for the tree
    /// projection's per-node `work`. `None` for a top (parentless) session. Default: `None`.
    async fn delegation_work(&self, _child: &SessionId) -> Option<String> {
        None
    }

    /// Fold `delta` into a session's durable usage total — the per-session usage surface the tree
    /// projection reads (replacing the in-memory fleet fan-in for durable sessions). Recorded by the
    /// activation path as each turn runs. Default: no-op.
    async fn record_usage(&self, _id: &SessionId, _delta: UsageDelta) {}

    /// A session's folded durable usage total. Default: zero.
    async fn usage_of(&self, _id: &SessionId) -> UsageDelta {
        UsageDelta::default()
    }

    /// Index (or re-index) searchable text for a session — an optional `title` plus a `body` blob
    /// (e.g. coalesced turn text / a generated recap) — feeding the durable full-text session search
    /// surface. The store is handed already-extracted text by the host; it never parses snapshots,
    /// so this stays protocol-free. Replaces any prior index row for the session. Default: no-op (a
    /// backend without a text index).
    async fn index_session_text(&self, _id: &SessionId, _title: Option<String>, _body: &str) {}

    /// Full-text search over the indexed session text, most-relevant first, capped at `limit`
    /// (`0` => a sensible default). Returns per-session hits with a highlighted snippet. Default:
    /// empty (a backend without a text index).
    async fn search_sessions(&self, _query: &str, _limit: u32) -> Vec<SessionSearchHit> {
        Vec::new()
    }

    /// Scan for sessions in a resumable (`Ready`/`Active`) state for the recovery scanner
    /// (lifecycle §5; invariant #7).
    async fn scan_resumable(&self, partition: PartitionId) -> Result<Vec<SessionId>, StoreError>;

    /// Pop the next pending durable job, if any (job-outbox dispatcher / worker side).
    async fn dequeue_job(&self) -> Option<JobCommand>;

    /// Pop the next pending durable wake hint, if any (wake-outbox dispatcher).
    async fn dequeue_wake(&self) -> Option<SessionId>;

    /// Read the current durable status of a session (test/observability helper).
    async fn status(&self, id: &SessionId) -> Option<SessionStatus>;

    /// A non-fencing read of a session's last persisted snapshot blob (`None` if unknown). Used to
    /// seed an attached background child from its parent's conversation (§4.3 `SpawnSeed`) without
    /// acquiring an activation lease — a read-only audit/seed peek, not an activation. Default:
    /// `None` (a non-authoritative proxy store).
    async fn peek_snapshot(&self, _id: &SessionId) -> Option<SnapshotBlob> {
        None
    }

    /// Upsert a session's host-level [`SessionMeta`] (bound profile + opaque overlay blob). Called
    /// when a session's profile binding is first established and whenever its overlay changes. The
    /// resolver reads it back at engine construction so a live override survives restart. Default:
    /// no-op (a non-authoritative proxy store).
    async fn set_session_meta(
        &self,
        _id: &SessionId,
        _meta: SessionMeta,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Read a session's host-level [`SessionMeta`] (`None` if none recorded). Default: `None`.
    async fn session_meta(&self, _id: &SessionId) -> Option<SessionMeta> {
        None
    }

    /// List every session's host-level [`SessionMeta`] row (unordered) — the enumeration behind the
    /// recent-sessions browse of the `session_search` agent tool, which covers live-only sessions a
    /// `session_record`-based listing would miss. Default: empty (a non-authoritative proxy store).
    async fn session_meta_list(&self) -> Vec<(SessionId, SessionMeta)> {
        Vec::new()
    }

    /// List every durable chat→session routing pin (§5.9). The host loads these into the live routing
    /// registry's resolve-first pin map (via the hot-reload rebuild hook). Default: none (a store
    /// without durable routing — pins are then in-memory only for the process lifetime).
    async fn routing_list(&self) -> Vec<ChatRoute> {
        Vec::new()
    }

    /// Read one routing pin by its canonical key (`None` if unpinned). Default: `None`.
    async fn routing_get(&self, _key: &str) -> Option<ChatRoute> {
        None
    }

    /// Upsert a chat→session routing pin (keyed by [`ChatRoute::key`]). Default: no-op.
    async fn routing_set(&self, _route: ChatRoute) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a routing pin by key (idempotent). Default: no-op.
    async fn routing_remove(&self, _key: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// List every durable Room (daemon-rooms-spec.md). The Rooms adapter loads these at bring-up to
    /// reconstruct the loopback transports. Default: none (a store without durable rooms — rooms are
    /// then in-memory only for the process lifetime, mirroring the `routing_*` default).
    async fn room_list(&self) -> Vec<Room> {
        Vec::new()
    }

    /// Read one Room by id (`None` if absent). Default: `None`.
    async fn room_get(&self, _id: &str) -> Option<Room> {
        None
    }

    /// Upsert a Room (keyed by [`Room::id`]). Default: no-op.
    async fn room_set(&self, _room: Room) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a Room by id (idempotent; its membership rows cascade). Default: no-op.
    async fn room_remove(&self, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// List a Room's members (the membership table the RoomRouter fans posts out to). Default: none.
    async fn room_members(&self, _room_id: &str) -> Vec<RoomMember> {
        Vec::new()
    }

    /// Upsert a Room member (keyed by `(room_id, member)`). Default: no-op.
    async fn room_member_set(&self, _member: RoomMember) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a Room member by `(room_id, member)` (idempotent). Default: no-op.
    async fn room_member_remove(&self, _room_id: &str, _member: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// List every durable scheduled job (I15). The cron scheduler loads these to compute next-fire.
    /// Default: none (a store without durable cron — jobs are then process-lifetime only).
    async fn cron_list(&self) -> Vec<StoredCronJob> {
        Vec::new()
    }

    /// Read one scheduled job by id (`None` if absent). Default: `None`.
    async fn cron_get(&self, _id: &str) -> Option<StoredCronJob> {
        None
    }

    /// Upsert a scheduled job (keyed by [`StoredCronJob::id`]). Default: no-op.
    async fn cron_set(&self, _job: StoredCronJob) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a scheduled job by id (idempotent). Default: no-op.
    async fn cron_remove(&self, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// The jobs due at `now_unix`: enabled (`!paused`) jobs whose `next_fire_unix <= now`. The
    /// scheduler's tick query. Default: none.
    async fn cron_due(&self, _now_unix: u64) -> Vec<StoredCronJob> {
        Vec::new()
    }

    /// List a job's most recent runs (newest first, capped at `max`). Default: none.
    async fn cron_runs_list(&self, _id: &str, _max: usize) -> Vec<StoredCronRun> {
        Vec::new()
    }

    /// Append a run record (bounded retention per job). Default: no-op.
    async fn cron_run_append(&self, _run: StoredCronRun) -> Result<(), StoreError> {
        Ok(())
    }

    /// List the durable cron suggestions (I15). Default: none.
    async fn cron_suggestions_list(&self) -> Vec<StoredCronSuggestion> {
        Vec::new()
    }

    /// Read one suggestion by id (`None` if absent). Default: `None`.
    async fn cron_suggestion_get(&self, _id: &str) -> Option<StoredCronSuggestion> {
        None
    }

    /// Upsert a suggestion (keyed by [`StoredCronSuggestion::id`]; `dedup_key` is unique). Default:
    /// no-op.
    async fn cron_suggestion_set(
        &self,
        _suggestion: StoredCronSuggestion,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a suggestion by id (idempotent). Default: no-op.
    async fn cron_suggestion_remove(&self, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// List the durable manually-registered ACP agent catalog entries (I7). Default: none.
    async fn acp_list(&self) -> Vec<AcpEntry> {
        Vec::new()
    }

    /// Upsert a manually-registered ACP catalog entry (keyed by [`AcpEntry::name`]). Default: no-op.
    async fn acp_set(&self, _entry: AcpEntry) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a manually-registered ACP catalog entry by name (idempotent). Default: no-op.
    async fn acp_remove(&self, _name: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// List every durable session id with its current status (the node control surface's
    /// `sessions` projection). Defaults to empty so a non-authoritative store (the brokered child
    /// proxy) need not implement it; an authoritative backend overrides it.
    async fn list_sessions(&self) -> Vec<(SessionId, SessionStatus)> {
        Vec::new()
    }

    /// Snapshot durable queue depths + session count (Metrics/health resident service).
    async fn stats(&self) -> StoreStats;

    // -- verifiable journal (phase 6b; unified management + transcript) -----------------------
    //
    // One hash-linked, per-segment-signed chain per stream carries typed entries: coarse management
    // records and coalesced finished chat blocks. Keyed `(stream, segment, seq)` — decoupled from
    // the durable `(session, epoch)` identity so non-durable units (live/fleet/foreign) journal too.
    // Default impls report "unsupported" / empty so a non-authoritative store (the brokered child
    // proxy) need not implement them; an authoritative backend overrides them.

    /// Append one entry to the open `(stream, segment)` segment. Idempotent per `seq`.
    async fn append_trace(
        &self,
        _stream: &JournalStreamId,
        _segment: u64,
        _entry: TraceEntry,
    ) -> Result<(), StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "verifiable journal not supported by this store".into(),
        )))
    }

    /// Seal the `(stream, segment)` segment with its signed Merkle root. `fence` is `Some` on the
    /// durable path (only the highest token for the session may commit, binding the root to the
    /// durable incarnation) and `None` for non-durable streams (the ed25519 signature is the
    /// integrity primitive; there is no competing incarnation to fence).
    async fn commit_trace_segment(
        &self,
        _stream: &JournalStreamId,
        _segment: u64,
        _root: MerkleRoot,
        _signature: Vec<u8>,
        _fence: Option<FenceToken>,
    ) -> Result<(), StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "verifiable journal not supported by this store".into(),
        )))
    }

    /// Load one `(stream, segment)` segment (entries + committed root, if sealed) — the
    /// seal-recompute path.
    async fn load_trace_segment(
        &self,
        _stream: &JournalStreamId,
        _segment: u64,
    ) -> Option<TraceSegment> {
        None
    }

    /// Cursor-paged read of a stream's journal for reconnect/scroll-back: up to `max` entries with
    /// `cursor > after_cursor`, plus the sealed roots of the segments they cover. Non-destructive.
    async fn load_journal(
        &self,
        _stream: &JournalStreamId,
        _after_cursor: u64,
        _max: u32,
    ) -> JournalPage {
        JournalPage::default()
    }

    /// Record an append-only conversation-rewind seal against `stream` (conversation-rewind spec §6).
    /// Default no-op for stores without a verifiable journal.
    async fn record_journal_seal(
        &self,
        _stream: &JournalStreamId,
        _seal: JournalSeal,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// The active (latest) conversation-rewind seal for `stream`, if any. Surfaced by
    /// `session_history` as `JournalPageView::sealed_after`. Default `None`.
    async fn active_journal_seal(&self, _stream: &JournalStreamId) -> Option<JournalSeal> {
        None
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Inner {
    sessions: HashMap<SessionId, SessionRecord>,
    /// Idempotency key set: applied/recorded completions `(session, epoch, job)`.
    inbox_keys: HashSet<(SessionId, Epoch, JobId)>,
    /// Unapplied completions, keyed by session, in arrival order.
    unapplied: HashMap<SessionId, Vec<JobCompletion>>,
    job_outbox: VecDeque<JobCommand>,
    /// Job ids already enqueued, to dedupe re-enqueues from idempotent re-activation.
    enqueued_jobs: HashSet<JobId>,
    wake_outbox: VecDeque<SessionId>,
    /// child session -> the parent job its terminal completion fulfills (the durable tree edge).
    delegations: HashMap<SessionId, JobCommand>,
    /// Detached child session -> its parent (the completion-notice edge). Persistent (drives tree
    /// visibility via `child_index` + the notice firing in `mark_completed`); the in-memory analogue
    /// of the SQLite `completion_notices` table.
    completion_notices: HashMap<SessionId, SessionId>,
    /// Detached children whose terminal notice has already been pushed onto `notice_outbox`, so a
    /// re-completion (a resumed detached child) fires the notice at most once (idempotency).
    notices_fired: HashSet<SessionId>,
    /// Pending completion notices for detached children, drained by the node's notice worker (the
    /// in-memory analogue of the SQLite `completion_notice_outbox` table).
    notice_outbox: VecDeque<CompletionNotice>,
    /// Per-parent monotonic counter minting the unique `{parent}/d{n}` detached child ids.
    detached_seq: HashMap<SessionId, u64>,
    /// Per-session pending inbound inputs (opaque bytes), FIFO — the durable `send` seam drained by
    /// the next activation's hydrate (the in-memory analogue of the SQLite `pending_session_input`
    /// table).
    pending_inputs: HashMap<SessionId, VecDeque<Vec<u8>>>,
    /// Per-session parked §12 edit-approval requests, in park order. An unanswered row keeps the
    /// session dormant; [`SessionStore::answer_approval`] stamps its decision and wakes the session.
    pending_approvals: HashMap<SessionId, Vec<ParkedApproval>>,
    /// parent session -> its delegated children in order (reverse index for the tree projection).
    child_index: HashMap<SessionId, Vec<SessionId>>,
    /// child session -> its attached non-joining edge label (§4.3 background spawn). Recorded by
    /// [`SessionStore::record_child_edge`] *without* a `delegations` entry, so the child self-closes
    /// (no parent wake); surfaces as the node's `work` label in the tree projection.
    background_edges: HashMap<SessionId, String>,
    /// Per-session folded usage total (the durable usage surface the tree projection reads).
    usage: HashMap<SessionId, UsageDelta>,
    /// Per-session indexed search text `(title, body)` — the in-memory analogue of the SQLite
    /// `session_fts` index, searched by case-insensitive substring.
    session_text: HashMap<SessionId, (String, String)>,
    /// Per-session host-level metadata: bound profile + opaque overlay blob (the in-memory analogue
    /// of the SQLite `session_meta` table).
    session_meta: HashMap<SessionId, SessionMeta>,
    /// Durable chat→session routing pins, keyed by canonical origin key (§5.9; the in-memory analogue
    /// of the SQLite `chat_routes` table).
    chat_routes: HashMap<String, ChatRoute>,
    /// Durable manually-registered ACP catalog entries, keyed by name (I7; the in-memory analogue of
    /// the SQLite `acp_catalog` table).
    acp_catalog: HashMap<String, AcpEntry>,
    /// Durable scheduled jobs, keyed by id (I15; the in-memory analogue of the SQLite `cron_jobs`
    /// table).
    cron_jobs: HashMap<String, StoredCronJob>,
    /// Durable cron run history, keyed by job id, newest last (I15; analogue of `cron_runs`).
    cron_runs: HashMap<String, Vec<StoredCronRun>>,
    /// Durable cron suggestions, keyed by id (I15; analogue of `cron_suggestions`).
    cron_suggestions: HashMap<String, StoredCronSuggestion>,
    fault: Option<FaultPoint>,
    /// Append-only journal entries per stream, in append (cursor) order across all segments.
    journal_entries: HashMap<JournalStreamId, Vec<JournalEntry>>,
    /// Sealed segment roots per `(stream, segment)`.
    journal_roots: HashMap<(JournalStreamId, u64), CommittedRoot>,
    /// Stream-monotonic cursor allocator (the pagination key for `load_journal`).
    journal_cursor: u64,
    /// Append-only conversation-rewind seals per stream, in record order (the latest is active).
    journal_seals: HashMap<JournalStreamId, Vec<JournalSeal>>,
}

/// In-memory [`SessionStore`] backend. The default backend for phase 1 and the conformance harness.
///
/// All durable mutations happen under a single lock, so multi-step operations
/// ([`SessionStore::checkpoint_and_enqueue`], [`SessionStore::record_completion_and_wake`]) are
/// atomic. A shared `Arc<InMemoryStore>` can back two activation managers to simulate dual-node
/// ownership (acceptance tests #4/#6).
#[derive(Default)]
pub struct InMemoryStore {
    inner: Mutex<Inner>,
}

impl InMemoryStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm the store to fail at a given durable boundary (acceptance test #2). `None` disarms.
    pub fn set_fault(&self, fault: Option<FaultPoint>) {
        self.inner.lock().unwrap().fault = fault;
    }

    /// Whether a fault is currently armed at `point`, clearing it (one-shot) if so.
    fn take_fault(inner: &mut Inner, point: FaultPoint) -> Result<(), StoreError> {
        if inner.fault == Some(point) {
            inner.fault = None;
            return Err(StoreError::Fault(point));
        }
        Ok(())
    }

    fn check_fence(rec: &SessionRecord, fence: FenceToken) -> Result<(), StoreError> {
        if fence < rec.fence {
            return Err(StoreError::Fenced {
                have: fence.0,
                current: rec.fence.0,
            });
        }
        Ok(())
    }

    /// Apply a completion under the held lock: idempotent per `(session, epoch, job)`, push it onto
    /// the parent's unapplied queue and mark the parent `Ready`. Returns `true` if it was fresh (the
    /// caller then publishes the wake). The parent must exist. Shared by the explicit
    /// `record_completion_and_wake` and the delegation fulfillment inside `mark_completed`.
    fn apply_completion_locked(inner: &mut Inner, c: &JobCompletion) -> bool {
        let key = (c.session_id.clone(), c.epoch, c.job_id.clone());
        if !inner.inbox_keys.insert(key) {
            return false;
        }
        inner
            .unapplied
            .entry(c.session_id.clone())
            .or_default()
            .push(c.clone());
        if let Some(rec) = inner.sessions.get_mut(&c.session_id) {
            rec.status = SessionStatus::Ready;
        }
        true
    }
}

#[async_trait]
impl SessionStore for InMemoryStore {
    async fn create_session(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.sessions.insert(
            id.clone(),
            SessionRecord {
                session_id: id,
                partition,
                epoch: Epoch::ZERO,
                status: SessionStatus::Ready,
                snapshot,
                fence: FenceToken::ZERO,
            },
        );
        Ok(())
    }

    async fn acquire_activation_lease(&self, id: &SessionId) -> Result<FenceToken, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get_mut(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        rec.fence = rec.fence.next();
        rec.status = SessionStatus::Active;
        Ok(rec.fence)
    }

    async fn load_for_activation(
        &self,
        id: &SessionId,
        fence: FenceToken,
    ) -> Result<Activation, StoreError> {
        let inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        let unapplied = inner.unapplied.get(id).cloned().unwrap_or_default();
        Ok(Activation {
            snapshot: rec.snapshot.clone(),
            unapplied,
            fence,
        })
    }

    async fn checkpoint_and_enqueue(
        &self,
        checkpoint: Checkpoint,
        job: JobCommand,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        {
            let rec = inner
                .sessions
                .get(&checkpoint.session_id)
                .ok_or_else(|| StoreError::NotFound(checkpoint.session_id.clone()))?;
            Self::check_fence(rec, fence)?;
        }
        // Boundary: abort the whole transaction before anything is written.
        Self::take_fault(&mut inner, FaultPoint::BeforeSnapshot)?;

        // Atomic commit: snapshot, epoch, status, and job-outbox enqueue land together.
        let rec = inner.sessions.get_mut(&checkpoint.session_id).unwrap();
        rec.snapshot = checkpoint.snapshot;
        rec.epoch = checkpoint.epoch;
        rec.status = SessionStatus::Suspended {
            job_id: job.job_id.clone(),
        };
        if inner.enqueued_jobs.insert(job.job_id.clone()) {
            inner.job_outbox.push_back(job);
        }

        // Post-commit crash boundaries: the durable state is already complete and consistent;
        // these model the process/task dying after the transaction committed but before it freed.
        // Recovery drains the durable job outbox regardless.
        Self::take_fault(&mut inner, FaultPoint::AfterSnapshot)?;
        Self::take_fault(&mut inner, FaultPoint::AfterJobOutbox)?;
        Ok(())
    }

    async fn mark_completed(
        &self,
        checkpoint: Checkpoint,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get_mut(&checkpoint.session_id)
            .ok_or_else(|| StoreError::NotFound(checkpoint.session_id.clone()))?;
        Self::check_fence(rec, fence)?;
        rec.snapshot = checkpoint.snapshot;
        rec.epoch = checkpoint.epoch;
        rec.status = SessionStatus::Completed;
        // Stamp the terminal clock on the session's host meta (the reaper's grace timer). Same
        // transaction (the held lock); re-stamped if a resumed session completes again.
        inner
            .session_meta
            .entry(checkpoint.session_id.clone())
            .or_default()
            .terminal_ms = Some(now_ms());
        // If this session was delegated by a parent, fulfill that parent's job and wake it in the
        // *same* transaction (under the held lock). The binding is durable, so this is recovery-safe:
        // a child marked terminal always wakes its delegator, at any nesting depth.
        if let Some(job) = inner.delegations.get(&checkpoint.session_id).cloned() {
            let completion = JobCompletion {
                session_id: job.session_id.clone(),
                epoch: job.epoch,
                job_id: job.job_id.clone(),
                payload: checkpoint
                    .completion_payload
                    .clone()
                    .unwrap_or_else(|| format!("child:{}", checkpoint.session_id).into_bytes()),
            };
            if inner.sessions.contains_key(&completion.session_id)
                && Self::apply_completion_locked(&mut inner, &completion)
            {
                inner.wake_outbox.push_back(completion.session_id);
            }
        }
        // If this session is a detached child with a completion-notice edge, push a CompletionNotice
        // (delivered to the parent as a fresh reactive turn) in the SAME transaction as the terminal
        // flip — NEVER a `completion_inbox` entry or a `wake_outbox` wake (there is no parent job to
        // fulfill). Idempotent per child (a resumed child that completes again fires once).
        if let Some(parent) = inner
            .completion_notices
            .get(&checkpoint.session_id)
            .cloned()
        {
            if inner.notices_fired.insert(checkpoint.session_id.clone()) {
                let payload = checkpoint
                    .completion_payload
                    .clone()
                    .unwrap_or_else(|| format!("child:{}", checkpoint.session_id).into_bytes());
                inner.notice_outbox.push_back(CompletionNotice {
                    parent,
                    child: checkpoint.session_id.clone(),
                    payload,
                });
            }
        }
        Ok(())
    }

    async fn record_completion_and_wake(&self, c: &JobCompletion) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.sessions.contains_key(&c.session_id) {
            return Err(StoreError::NotFound(c.session_id.clone()));
        }
        // Idempotent: a redelivered completion is a no-op (invariant #2/#3).
        if !Self::apply_completion_locked(&mut inner, c) {
            return Ok(());
        }
        // Boundary: completion durable + session Ready; crash before publishing the wake.
        // Recovery scan must still re-activate the Ready session (invariant #7).
        Self::take_fault(&mut inner, FaultPoint::BeforeWakePublish)?;
        inner.wake_outbox.push_back(c.session_id.clone());
        Ok(())
    }

    async fn bind_delegation(&self, child: SessionId, job: JobCommand) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .child_index
            .entry(job.session_id.clone())
            .or_default()
            .push(child.clone());
        inner.delegations.insert(child, job);
        Ok(())
    }

    async fn enqueue_detached_job(&self, mut job: JobCommand) -> Result<SessionId, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let parent = job.session_id.clone();
        let seq = inner.detached_seq.entry(parent.clone()).or_insert(0);
        *seq += 1;
        let child = SessionId::new(format!("{}/d{}", parent, *seq));
        // A detached job is bare (no checkpoint, no suspension, no `enqueued_jobs` dedupe): the parent
        // keeps running. The pre-minted child id rides on the job so the fleet worker materializes the
        // child at exactly this id.
        job.child = Some(child.clone());
        inner.job_outbox.push_back(job);
        Ok(child)
    }

    async fn bind_completion_notice(
        &self,
        child: &SessionId,
        parent: &SessionId,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Idempotent tree edge: record the child under the parent for `children_of`/tree, but not in
        // `delegations` (so `mark_completed` binds no job — the child self-closes with a notice).
        let siblings = inner.child_index.entry(parent.clone()).or_default();
        if !siblings.contains(child) {
            siblings.push(child.clone());
        }
        inner
            .completion_notices
            .insert(child.clone(), parent.clone());
        Ok(())
    }

    async fn dequeue_completion_notice(&self) -> Option<CompletionNotice> {
        self.inner.lock().unwrap().notice_outbox.pop_front()
    }

    async fn record_child_edge(
        &self,
        parent: SessionId,
        child: SessionId,
        work_label: String,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // The reverse index drives the tree projection (audit), but we deliberately do *not* write a
        // `delegations` entry: `mark_completed` finds no job, so the child self-closes (no wake).
        inner
            .child_index
            .entry(parent)
            .or_default()
            .push(child.clone());
        inner.background_edges.insert(child, work_label);
        Ok(())
    }

    async fn children_of(&self, parent: &SessionId) -> Vec<SessionId> {
        self.inner
            .lock()
            .unwrap()
            .child_index
            .get(parent)
            .cloned()
            .unwrap_or_default()
    }

    async fn enqueue_wake(&self, id: SessionId) {
        self.inner.lock().unwrap().wake_outbox.push_back(id);
    }

    async fn enqueue_session_input(&self, id: &SessionId, input: Vec<u8>) {
        self.inner
            .lock()
            .unwrap()
            .pending_inputs
            .entry(id.clone())
            .or_default()
            .push_back(input);
    }

    async fn take_session_inputs(&self, id: &SessionId) -> Vec<Vec<u8>> {
        self.inner
            .lock()
            .unwrap()
            .pending_inputs
            .remove(id)
            .map(|q| q.into_iter().collect())
            .unwrap_or_default()
    }

    async fn park_approval(
        &self,
        checkpoint: Checkpoint,
        approvals: Vec<ParkedApproval>,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        {
            let rec = inner
                .sessions
                .get(&checkpoint.session_id)
                .ok_or_else(|| StoreError::NotFound(checkpoint.session_id.clone()))?;
            Self::check_fence(rec, fence)?;
        }
        Self::take_fault(&mut inner, FaultPoint::BeforeSnapshot)?;
        // Atomic commit: snapshot + epoch + Suspended status + parked rows land together. No job is
        // enqueued — the session stays dormant until an operator decision wakes it.
        let suspend_job = approvals.first().map(|a| a.job_id.clone());
        let rec = inner.sessions.get_mut(&checkpoint.session_id).unwrap();
        rec.snapshot = checkpoint.snapshot;
        rec.epoch = checkpoint.epoch;
        if let Some(job_id) = suspend_job {
            rec.status = SessionStatus::Suspended { job_id };
        }
        let rows = inner
            .pending_approvals
            .entry(checkpoint.session_id.clone())
            .or_default();
        for approval in approvals {
            // Dedupe a re-parked row on deterministic recovery (same session + job).
            if !rows.iter().any(|r| r.job_id == approval.job_id) {
                rows.push(approval);
            }
        }
        Ok(())
    }

    async fn answer_approval(
        &self,
        session: &SessionId,
        job_id: &JobId,
        allow: bool,
    ) -> Result<bool, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let epoch = match inner.pending_approvals.get_mut(session) {
            Some(rows) => match rows.iter_mut().find(|r| &r.job_id == job_id) {
                // Already answered: idempotent no-op (a redelivered decision).
                Some(row) if row.decision.is_some() => return Ok(true),
                Some(row) => {
                    row.decision = Some(allow);
                    row.epoch
                }
                None => return Ok(false),
            },
            None => return Ok(false),
        };
        let completion = JobCompletion {
            session_id: session.clone(),
            epoch,
            job_id: job_id.clone(),
            payload: if allow {
                b"allow".to_vec()
            } else {
                b"deny".to_vec()
            },
        };
        // Completion durable + session Ready, then publish the wake (one transaction).
        if Self::apply_completion_locked(&mut inner, &completion) {
            inner.wake_outbox.push_back(session.clone());
        }
        Ok(true)
    }

    async fn pending_approvals_of(&self, session: Option<&SessionId>) -> Vec<ParkedApproval> {
        let inner = self.inner.lock().unwrap();
        let unanswered = |rows: &Vec<ParkedApproval>| -> Vec<ParkedApproval> {
            rows.iter()
                .filter(|r| r.decision.is_none())
                .cloned()
                .collect()
        };
        match session {
            Some(id) => inner
                .pending_approvals
                .get(id)
                .map(unanswered)
                .unwrap_or_default(),
            None => inner
                .pending_approvals
                .values()
                .flat_map(unanswered)
                .collect(),
        }
    }

    async fn delegation_work(&self, child: &SessionId) -> Option<String> {
        let inner = self.inner.lock().unwrap();
        inner
            .delegations
            .get(child)
            .map(|job| String::from_utf8_lossy(&job.payload).into_owned())
            // Fall back to the attached non-joining edge label (§4.3 background spawn).
            .or_else(|| inner.background_edges.get(child).cloned())
    }

    async fn record_usage(&self, id: &SessionId, delta: UsageDelta) {
        self.inner
            .lock()
            .unwrap()
            .usage
            .entry(id.clone())
            .or_default()
            .add(&delta);
    }

    async fn usage_of(&self, id: &SessionId) -> UsageDelta {
        self.inner
            .lock()
            .unwrap()
            .usage
            .get(id)
            .copied()
            .unwrap_or_default()
    }

    async fn index_session_text(&self, id: &SessionId, title: Option<String>, body: &str) {
        self.inner
            .lock()
            .unwrap()
            .session_text
            .insert(id.clone(), (title.unwrap_or_default(), body.to_string()));
    }

    async fn search_sessions(&self, query: &str, limit: u32) -> Vec<SessionSearchHit> {
        let limit = if limit == 0 { 50 } else { limit } as usize;
        let needle = query.trim().to_lowercase();
        if needle.is_empty() {
            return Vec::new();
        }
        let inner = self.inner.lock().unwrap();
        inner
            .session_text
            .iter()
            .filter(|(_, (title, body))| {
                title.to_lowercase().contains(&needle) || body.to_lowercase().contains(&needle)
            })
            .take(limit)
            .map(|(id, (title, body))| SessionSearchHit {
                session_id: id.clone(),
                title: title.clone(),
                snippet: snippet_around(body, &needle),
            })
            .collect()
    }

    async fn scan_resumable(&self, partition: PartitionId) -> Result<Vec<SessionId>, StoreError> {
        let inner = self.inner.lock().unwrap();
        Ok(inner
            .sessions
            .values()
            .filter(|r| {
                r.partition == partition
                    && matches!(r.status, SessionStatus::Ready | SessionStatus::Active)
            })
            .map(|r| r.session_id.clone())
            .collect())
    }

    async fn dequeue_job(&self) -> Option<JobCommand> {
        self.inner.lock().unwrap().job_outbox.pop_front()
    }

    async fn dequeue_wake(&self) -> Option<SessionId> {
        self.inner.lock().unwrap().wake_outbox.pop_front()
    }

    async fn peek_snapshot(&self, id: &SessionId) -> Option<SnapshotBlob> {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .get(id)
            .map(|rec| rec.snapshot.clone())
    }

    async fn set_session_meta(&self, id: &SessionId, meta: SessionMeta) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .session_meta
            .insert(id.clone(), meta);
        Ok(())
    }

    async fn session_meta(&self, id: &SessionId) -> Option<SessionMeta> {
        self.inner.lock().unwrap().session_meta.get(id).cloned()
    }

    async fn session_meta_list(&self) -> Vec<(SessionId, SessionMeta)> {
        self.inner
            .lock()
            .unwrap()
            .session_meta
            .iter()
            .map(|(id, meta)| (id.clone(), meta.clone()))
            .collect()
    }

    async fn routing_list(&self) -> Vec<ChatRoute> {
        self.inner
            .lock()
            .unwrap()
            .chat_routes
            .values()
            .cloned()
            .collect()
    }

    async fn routing_get(&self, key: &str) -> Option<ChatRoute> {
        self.inner.lock().unwrap().chat_routes.get(key).cloned()
    }

    async fn routing_set(&self, route: ChatRoute) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .chat_routes
            .insert(route.key.clone(), route);
        Ok(())
    }

    async fn routing_remove(&self, key: &str) -> Result<(), StoreError> {
        self.inner.lock().unwrap().chat_routes.remove(key);
        Ok(())
    }

    async fn acp_list(&self) -> Vec<AcpEntry> {
        self.inner
            .lock()
            .unwrap()
            .acp_catalog
            .values()
            .cloned()
            .collect()
    }

    async fn acp_set(&self, entry: AcpEntry) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .acp_catalog
            .insert(entry.name.clone(), entry);
        Ok(())
    }

    async fn acp_remove(&self, name: &str) -> Result<(), StoreError> {
        self.inner.lock().unwrap().acp_catalog.remove(name);
        Ok(())
    }

    async fn cron_list(&self) -> Vec<StoredCronJob> {
        self.inner
            .lock()
            .unwrap()
            .cron_jobs
            .values()
            .cloned()
            .collect()
    }

    async fn cron_get(&self, id: &str) -> Option<StoredCronJob> {
        self.inner.lock().unwrap().cron_jobs.get(id).cloned()
    }

    async fn cron_set(&self, job: StoredCronJob) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .cron_jobs
            .insert(job.id.clone(), job);
        Ok(())
    }

    async fn cron_remove(&self, id: &str) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        inner.cron_jobs.remove(id);
        inner.cron_runs.remove(id);
        Ok(())
    }

    async fn cron_due(&self, now_unix: u64) -> Vec<StoredCronJob> {
        let mut due: Vec<StoredCronJob> = self
            .inner
            .lock()
            .unwrap()
            .cron_jobs
            .values()
            .filter(|j| !j.paused && j.next_fire_unix.is_some_and(|t| t <= now_unix))
            .cloned()
            .collect();
        // Earliest-due first, mirroring the SqliteStore `ORDER BY next_fire_unix`.
        due.sort_by(|a, b| {
            a.next_fire_unix
                .cmp(&b.next_fire_unix)
                .then(a.id.cmp(&b.id))
        });
        due
    }

    async fn cron_runs_list(&self, id: &str, max: usize) -> Vec<StoredCronRun> {
        self.inner
            .lock()
            .unwrap()
            .cron_runs
            .get(id)
            .map(|runs| runs.iter().rev().take(max).cloned().collect())
            .unwrap_or_default()
    }

    async fn cron_run_append(&self, run: StoredCronRun) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let runs = inner.cron_runs.entry(run.job_id.clone()).or_default();
        runs.push(run);
        // Bounded retention: keep the most recent CRON_RUN_RETENTION rows per job.
        let len = runs.len();
        if len > CRON_RUN_RETENTION {
            runs.drain(0..len - CRON_RUN_RETENTION);
        }
        Ok(())
    }

    async fn cron_suggestions_list(&self) -> Vec<StoredCronSuggestion> {
        self.inner
            .lock()
            .unwrap()
            .cron_suggestions
            .values()
            .cloned()
            .collect()
    }

    async fn cron_suggestion_get(&self, id: &str) -> Option<StoredCronSuggestion> {
        self.inner.lock().unwrap().cron_suggestions.get(id).cloned()
    }

    async fn cron_suggestion_set(
        &self,
        suggestion: StoredCronSuggestion,
    ) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .cron_suggestions
            .insert(suggestion.id.clone(), suggestion);
        Ok(())
    }

    async fn cron_suggestion_remove(&self, id: &str) -> Result<(), StoreError> {
        self.inner.lock().unwrap().cron_suggestions.remove(id);
        Ok(())
    }

    async fn status(&self, id: &SessionId) -> Option<SessionStatus> {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .get(id)
            .map(|r| r.status.clone())
    }

    async fn list_sessions(&self) -> Vec<(SessionId, SessionStatus)> {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .values()
            .map(|r| (r.session_id.clone(), r.status.clone()))
            .collect()
    }

    async fn stats(&self) -> StoreStats {
        let inner = self.inner.lock().unwrap();
        StoreStats {
            pending_jobs: inner.job_outbox.len(),
            pending_wakes: inner.wake_outbox.len(),
            sessions: inner.sessions.len(),
        }
    }

    async fn append_trace(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        entry: TraceEntry,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Append-only + idempotent per `(segment, seq)`: a redelivered entry is a no-op.
        let log = inner.journal_entries.entry(stream.clone()).or_default();
        if log
            .iter()
            .any(|e| e.segment == segment && e.entry.seq == entry.seq)
        {
            return Ok(());
        }
        // 1-based, matching the SQLite backend's `AUTOINCREMENT` cursor: `after_cursor = 0` (strict
        // `>`) yields the first entry, so the two backends paginate identically.
        inner.journal_cursor += 1;
        let cursor = inner.journal_cursor;
        inner
            .journal_entries
            .get_mut(stream)
            .unwrap()
            .push(JournalEntry {
                cursor,
                segment,
                entry,
            });
        Ok(())
    }

    async fn commit_trace_segment(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        root: MerkleRoot,
        signature: Vec<u8>,
        fence: Option<FenceToken>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Durable path: fenced exactly like a checkpoint — a stale incarnation cannot seal a root.
        // Non-durable streams pass `None`: no competing incarnation, the signature is the integrity
        // primitive.
        if let Some(fence) = fence {
            let id = SessionId::new(stream.as_str());
            let rec = inner
                .sessions
                .get(&id)
                .ok_or_else(|| StoreError::NotFound(id.clone()))?;
            Self::check_fence(rec, fence)?;
        }
        inner
            .journal_roots
            .insert((stream.clone(), segment), CommittedRoot { root, signature });
        Ok(())
    }

    async fn load_trace_segment(
        &self,
        stream: &JournalStreamId,
        segment: u64,
    ) -> Option<TraceSegment> {
        let inner = self.inner.lock().unwrap();
        let mut entries: Vec<TraceEntry> = inner
            .journal_entries
            .get(stream)
            .map(|log| {
                log.iter()
                    .filter(|e| e.segment == segment)
                    .map(|e| e.entry.clone())
                    .collect()
            })
            .unwrap_or_default();
        entries.sort_by_key(|e| e.seq);
        let committed = inner.journal_roots.get(&(stream.clone(), segment)).cloned();
        if entries.is_empty() && committed.is_none() {
            return None;
        }
        Some(TraceSegment {
            stream: stream.clone(),
            segment,
            entries,
            committed,
        })
    }

    async fn load_journal(
        &self,
        stream: &JournalStreamId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPage {
        let inner = self.inner.lock().unwrap();
        let Some(log) = inner.journal_entries.get(stream) else {
            return JournalPage::default();
        };
        let head_cursor = log.iter().map(|e| e.cursor).max().unwrap_or(0);
        let mut entries: Vec<JournalEntry> = log
            .iter()
            .filter(|e| e.cursor > after_cursor)
            .cloned()
            .collect();
        entries.sort_by_key(|e| e.cursor);
        if max > 0 {
            entries.truncate(max as usize);
        }
        let next_cursor = entries.last().map(|e| e.cursor).unwrap_or(after_cursor);
        // The sealed roots of the segments this page covers, for verification.
        let mut segments: Vec<u64> = entries.iter().map(|e| e.segment).collect();
        segments.sort_unstable();
        segments.dedup();
        let segment_roots = segments
            .into_iter()
            .filter_map(|seg| {
                inner
                    .journal_roots
                    .get(&(stream.clone(), seg))
                    .cloned()
                    .map(|root| (seg, root))
            })
            .collect();
        JournalPage {
            entries,
            segment_roots,
            next_cursor,
            head_cursor,
        }
    }

    async fn record_journal_seal(
        &self,
        stream: &JournalStreamId,
        seal: JournalSeal,
    ) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .journal_seals
            .entry(stream.clone())
            .or_default()
            .push(seal);
        Ok(())
    }

    async fn active_journal_seal(&self, stream: &JournalStreamId) -> Option<JournalSeal> {
        self.inner
            .lock()
            .unwrap()
            .journal_seals
            .get(stream)
            .and_then(|seals| seals.last().copied())
    }
}

#[cfg(test)]
mod journal_tests {
    //! Trace-journal conformance against the in-memory backend: append-only ordering + idempotency,
    //! the committed-root round-trip, and the fence guarding a segment seal (phase 6b store layer).

    use super::*;

    fn entry(seq: u64, byte: u8) -> TraceEntry {
        TraceEntry {
            seq,
            bytes: vec![byte; 4],
            content_hash: ContentHash::new([byte; 32]),
        }
    }

    async fn seeded() -> (InMemoryStore, SessionId, FenceToken) {
        let store = InMemoryStore::new();
        let id = SessionId::new("journaled");
        store
            .create_session(id.clone(), PartitionId::DEFAULT, SnapshotBlob::default())
            .await
            .unwrap();
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        (store, id, fence)
    }

    /// Conversation-rewind seals are append-only; the latest seal for a stream is the active one and
    /// other streams are unaffected (conversation-rewind spec §6).
    #[tokio::test]
    async fn journal_seal_round_trips_latest_active() {
        let store = InMemoryStore::new();
        let stream = JournalStreamId::session(&SessionId::new("rw"));
        assert!(store.active_journal_seal(&stream).await.is_none());

        for (cursor, retained, epoch, ts) in [(10u64, 2u64, 1u64, 100u64), (25, 1, 2, 200)] {
            store
                .record_journal_seal(
                    &stream,
                    JournalSeal {
                        seal_cursor: cursor,
                        retained_turns: retained,
                        epoch,
                        recorded_unix: ts,
                    },
                )
                .await
                .unwrap();
        }

        let active = store.active_journal_seal(&stream).await.expect("seal");
        assert_eq!(active.seal_cursor, 25);
        assert_eq!(active.retained_turns, 1);
        assert_eq!(active.epoch, 2);
        let other = JournalStreamId::session(&SessionId::new("other"));
        assert!(store.active_journal_seal(&other).await.is_none());
    }

    #[tokio::test]
    async fn append_is_ordered_and_idempotent() {
        let (store, id, _f) = seeded().await;
        let stream = JournalStreamId::session(&id);
        // Append out of order; load_trace_segment returns them sorted by seq.
        store
            .append_trace(&stream, 0, entry(2, 0x22))
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(0, 0x00))
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(1, 0x11))
            .await
            .unwrap();
        // Redelivered seq is a no-op (append-only, idempotent).
        store
            .append_trace(&stream, 0, entry(1, 0xFF))
            .await
            .unwrap();

        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        assert_eq!(
            seg.entries.iter().map(|e| e.seq).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        // The first writer of seq=1 wins; the duplicate did not overwrite.
        assert_eq!(seg.entries[1].bytes, vec![0x11; 4]);
        assert!(seg.committed.is_none(), "segment is still open");
    }

    #[tokio::test]
    async fn non_durable_stream_journals_without_a_session() {
        // A unit stream has no session record; the journal accepts it (keyed by stream, not session).
        let store = InMemoryStore::new();
        let stream = JournalStreamId::unit(&daemon_common::UnitId::new("fleet-child"));
        store.append_trace(&stream, 0, entry(0, 1)).await.unwrap();
        store.append_trace(&stream, 0, entry(1, 2)).await.unwrap();
        // Unfenced seal (None) succeeds for a non-durable stream.
        store
            .commit_trace_segment(&stream, 0, MerkleRoot::new([5; 32]), vec![9], None)
            .await
            .unwrap();
        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        assert_eq!(seg.entries.len(), 2);
        assert_eq!(seg.committed.unwrap().root, MerkleRoot::new([5; 32]));
    }

    #[tokio::test]
    async fn cursor_paging_walks_segments_in_order() {
        let (store, id, _f) = seeded().await;
        let stream = JournalStreamId::session(&id);
        // Segment 0 then segment 1, each with two entries.
        store
            .append_trace(&stream, 0, entry(0, 0xA0))
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(1, 0xA1))
            .await
            .unwrap();
        store
            .append_trace(&stream, 1, entry(0, 0xB0))
            .await
            .unwrap();
        store
            .append_trace(&stream, 1, entry(1, 0xB1))
            .await
            .unwrap();

        let page = store.load_journal(&stream, 0, 3).await;
        assert_eq!(page.entries.len(), 3, "max caps the page");
        assert_eq!(
            page.entries[0].segment, 0,
            "from the start (after_cursor 0 is inclusive)"
        );
        assert_eq!(page.head_cursor, 4, "four entries -> 1-based cursors 1..=4");
        // Walk the rest from the returned cursor.
        let rest = store.load_journal(&stream, page.next_cursor, 0).await;
        assert_eq!(rest.entries.len(), 1);
        assert_eq!(rest.entries[0].segment, 1);
    }

    #[tokio::test]
    async fn commit_root_round_trips() {
        let (store, id, fence) = seeded().await;
        let stream = JournalStreamId::session(&id);
        store.append_trace(&stream, 0, entry(0, 7)).await.unwrap();
        let root = MerkleRoot::new([9u8; 32]);
        store
            .commit_trace_segment(&stream, 0, root, vec![1, 2, 3], Some(fence))
            .await
            .unwrap();

        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        let committed = seg.committed.expect("segment sealed");
        assert_eq!(committed.root, root);
        assert_eq!(committed.signature, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn stale_fence_cannot_seal_a_segment() {
        let (store, id, stale) = seeded().await;
        let stream = JournalStreamId::session(&id);
        // A newer owner supersedes the fence we hold.
        let _current = store.acquire_activation_lease(&id).await.unwrap();
        let r = store
            .commit_trace_segment(&stream, 0, MerkleRoot::new([0; 32]), vec![], Some(stale))
            .await;
        assert!(
            matches!(r, Err(StoreError::Fenced { .. })),
            "a stale incarnation must not seal a segment root, got {r:?}"
        );
        // And nothing was committed.
        assert!(store.load_trace_segment(&stream, 0).await.is_none());
    }
}

#[cfg(test)]
mod session_meta_tests {
    //! Host-level [`SessionMeta`] persistence (bound profile + opaque overlay blob), proven against
    //! both backends so a per-session override is restored on rehydration regardless of store.

    use super::*;

    fn sample() -> SessionMeta {
        SessionMeta {
            bound_profile: Some(ProfileRef::new("opus")),
            overlay: vec![0xCB, 0x01, 0x02, 0x03],
            title: Some("a chat".into()),
            last_activity_ms: Some(1_700_000_000_000),
            role: Some(SessionRole::ManagedChild),
            parent: Some(SessionId::new("p1")),
            pinned: true,
            archived: false,
            scheduled_job: Some(JobId::from("cron-7")),
            activation_epoch: 3,
            owner: Some("user-alice".into()),
            terminal_ms: Some(1_700_000_000_500),
        }
    }

    /// `mark_completed` stamps the terminal clock ([`SessionMeta::terminal_ms`]) in the same
    /// transaction as the status flip — the reaper's grace timer, proven on both backends.
    async fn terminal_stamp_behaviour(store: &dyn SessionStore) {
        let id = SessionId::new("stamped");
        store
            .create_session(id.clone(), PartitionId::DEFAULT, SnapshotBlob::default())
            .await
            .unwrap();
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        assert!(
            store
                .session_meta(&id)
                .await
                .is_none_or(|m| m.terminal_ms.is_none()),
            "no terminal stamp before completion"
        );
        store
            .mark_completed(
                Checkpoint::new(id.clone(), Epoch(1), SnapshotBlob::default()),
                fence,
            )
            .await
            .unwrap();
        let meta = store.session_meta(&id).await.expect("meta after terminal");
        assert!(
            meta.terminal_ms.is_some_and(|t| t > 0),
            "mark_completed stamps terminal_ms"
        );
    }

    #[tokio::test]
    async fn in_memory_mark_completed_stamps_terminal_ms() {
        terminal_stamp_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_mark_completed_stamps_terminal_ms() {
        terminal_stamp_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_meta_round_trips_and_upserts() {
        let store = InMemoryStore::new();
        let id = SessionId::new("s1");
        // Absent until written.
        assert!(store.session_meta(&id).await.is_none());
        store.set_session_meta(&id, sample()).await.unwrap();
        assert_eq!(store.session_meta(&id).await.unwrap(), sample());
        // Upsert overwrites (e.g. an overlay change preserving the bound profile).
        let updated = SessionMeta {
            bound_profile: Some(ProfileRef::new("opus")),
            overlay: vec![0xFF],
            ..SessionMeta::default()
        };
        store.set_session_meta(&id, updated.clone()).await.unwrap();
        assert_eq!(store.session_meta(&id).await.unwrap(), updated);
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_meta_round_trips_and_upserts() {
        let store = SqliteStore::open_in_memory().unwrap();
        let id = SessionId::new("s1");
        assert!(store.session_meta(&id).await.is_none());
        store.set_session_meta(&id, sample()).await.unwrap();
        assert_eq!(store.session_meta(&id).await.unwrap(), sample());
        let updated = SessionMeta {
            bound_profile: None,
            overlay: Vec::new(),
            ..SessionMeta::default()
        };
        store.set_session_meta(&id, updated.clone()).await.unwrap();
        assert_eq!(store.session_meta(&id).await.unwrap(), updated);
    }

    /// `session_meta_list` enumerates every recorded meta row with full field fidelity — the
    /// browse surface behind the `session_search` tool (covers live-only sessions that have no
    /// `session_record`). Proven against both backends.
    async fn meta_list_behaviour(store: &dyn SessionStore) {
        assert!(store.session_meta_list().await.is_empty());
        store
            .set_session_meta(&SessionId::new("m1"), sample())
            .await
            .unwrap();
        store
            .set_session_meta(&SessionId::new("m2"), SessionMeta::default())
            .await
            .unwrap();
        let mut rows = store.session_meta_list().await;
        rows.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, SessionId::new("m1"));
        assert_eq!(rows[0].1, sample());
        assert_eq!(rows[1].1, SessionMeta::default());
    }

    #[tokio::test]
    async fn in_memory_meta_list_round_trips() {
        meta_list_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_meta_list_round_trips() {
        meta_list_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }

    fn sample_job(id: &str, next_fire: Option<u64>) -> StoredCronJob {
        StoredCronJob {
            id: id.into(),
            schedule: "0 9 * * *".into(),
            spec: vec![0xCB, 0xA1, 0x02],
            next_fire_unix: next_fire,
            paused: false,
            last_run_unix: None,
            last_ok: None,
            last_detail: None,
            fire_count: 0,
            created_unix: 1_700_000_000,
            owner: None,
        }
    }

    async fn cron_store_behaviour(store: &dyn SessionStore) {
        // Upsert + get round-trip.
        store.cron_set(sample_job("j1", Some(100))).await.unwrap();
        store.cron_set(sample_job("j2", Some(300))).await.unwrap();
        // A paused job is never due.
        let mut paused = sample_job("j3", Some(50));
        paused.paused = true;
        store.cron_set(paused).await.unwrap();
        assert_eq!(store.cron_get("j1").await.unwrap().schedule, "0 9 * * *");
        assert_eq!(store.cron_list().await.len(), 3);

        // The Auth 4 `owner` column round-trips (the cron worker stamps the spawned session's owner
        // from it); a legacy job (sample_job) carries `None`.
        let mut owned = sample_job("j-owned", Some(400));
        owned.owner = Some("user-bob".into());
        store.cron_set(owned).await.unwrap();
        assert_eq!(
            store.cron_get("j-owned").await.unwrap().owner.as_deref(),
            Some("user-bob")
        );
        assert!(store.cron_get("j1").await.unwrap().owner.is_none());
        store.cron_remove("j-owned").await.unwrap();

        // cron_due: only enabled jobs with next_fire <= now.
        let due: Vec<String> = store
            .cron_due(200)
            .await
            .into_iter()
            .map(|j| j.id)
            .collect();
        assert_eq!(due, vec!["j1".to_string()]); // j2 is future, j3 is paused
        let due_all: Vec<String> = store
            .cron_due(1000)
            .await
            .into_iter()
            .map(|j| j.id)
            .collect();
        assert_eq!(due_all, vec!["j1".to_string(), "j2".to_string()]);

        // Runs append + bounded retrieval (newest first).
        for i in 0..3 {
            store
                .cron_run_append(StoredCronRun {
                    job_id: "j1".into(),
                    started_unix: 100 + i,
                    finished_unix: Some(101 + i),
                    ok: i % 2 == 0,
                    detail: Some(format!("run-{i}")),
                    session: Some(SessionId::new(format!("cron_j1_{i}"))),
                    manual: false,
                })
                .await
                .unwrap();
        }
        let runs = store.cron_runs_list("j1", 2).await;
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].started_unix, 102); // newest first

        // Remove also clears runs.
        store.cron_remove("j1").await.unwrap();
        assert!(store.cron_get("j1").await.is_none());
        assert!(store.cron_runs_list("j1", 10).await.is_empty());

        // Suggestions round-trip.
        store
            .cron_suggestion_set(StoredCronSuggestion {
                id: "s1".into(),
                title: "Daily".into(),
                description: "d".into(),
                source: "catalog".into(),
                spec: vec![1, 2, 3],
                dedup_key: "catalog:daily".into(),
                status: "pending".into(),
                created_unix: 1_700_000_000,
            })
            .await
            .unwrap();
        assert_eq!(store.cron_suggestions_list().await.len(), 1);
        assert_eq!(
            store.cron_suggestion_get("s1").await.unwrap().title,
            "Daily"
        );
        store.cron_suggestion_remove("s1").await.unwrap();
        assert!(store.cron_suggestions_list().await.is_empty());
    }

    #[tokio::test]
    async fn in_memory_cron_round_trips() {
        cron_store_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_cron_round_trips() {
        cron_store_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }

    /// The durable pending-input seam: enqueue is FIFO per session, take drains-and-deletes (a
    /// second take is empty), and sessions are isolated from each other.
    async fn pending_input_behaviour(store: &dyn SessionStore) {
        let a = SessionId::new("in-a");
        let b = SessionId::new("in-b");
        assert!(store.take_session_inputs(&a).await.is_empty());
        store.enqueue_session_input(&a, vec![1]).await;
        store.enqueue_session_input(&a, vec![2, 2]).await;
        store.enqueue_session_input(&b, vec![9]).await;
        assert_eq!(
            store.take_session_inputs(&a).await,
            vec![vec![1], vec![2, 2]],
            "FIFO order per session"
        );
        assert!(
            store.take_session_inputs(&a).await.is_empty(),
            "take drains: a second take is empty"
        );
        assert_eq!(store.take_session_inputs(&b).await, vec![vec![9]]);
    }

    #[tokio::test]
    async fn in_memory_pending_inputs_round_trip() {
        pending_input_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_pending_inputs_round_trip() {
        pending_input_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }
}

#[cfg(test)]
mod pending_input_tests {
    //! The pending-input seam (the durable `send` path), proven against both backends: FIFO order,
    //! destructive drain-once, and per-session isolation.

    use super::*;

    async fn pending_input_behaviour(store: &dyn SessionStore) {
        let a = SessionId::new("s-a");
        let b = SessionId::new("s-b");
        // Empty until enqueued.
        assert!(store.take_session_inputs(&a).await.is_empty());

        store.enqueue_session_input(&a, b"first".to_vec()).await;
        store.enqueue_session_input(&a, b"second".to_vec()).await;
        store.enqueue_session_input(&b, b"other".to_vec()).await;

        // FIFO, scoped to the session.
        let drained = store.take_session_inputs(&a).await;
        assert_eq!(drained, vec![b"first".to_vec(), b"second".to_vec()]);
        // Destructive: a second drain is empty; the sibling queue is untouched.
        assert!(store.take_session_inputs(&a).await.is_empty());
        assert_eq!(store.take_session_inputs(&b).await, vec![b"other".to_vec()]);
    }

    #[tokio::test]
    async fn in_memory_pending_inputs_drain_fifo_once() {
        pending_input_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_pending_inputs_drain_fifo_once() {
        pending_input_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }
}

#[cfg(test)]
mod detached_delegation_tests {
    //! The detached-delegation (W9 `spawn wait:false`) store seam, proven against both backends:
    //! `enqueue_detached_job` mints unique `{parent}/dN` children and stamps them onto the bare job;
    //! `bind_completion_notice` makes the child tree-visible without a delegation edge; and a detached
    //! child's terminal `mark_completed` pushes exactly one `CompletionNotice` (idempotent), stamps
    //! `terminal_ms`, and NEVER touches the `completion_inbox`/`wake_outbox` rails.

    use super::*;

    fn detached_job(parent: &SessionId) -> JobCommand {
        JobCommand {
            job_id: JobId::new(format!("{parent}:detached")),
            session_id: parent.clone(),
            epoch: Epoch::ZERO,
            payload: Vec::new(),
            lifetime: ChildLifetime::Persistent,
            child: None,
        }
    }

    /// `enqueue_detached_job` mints a unique `{parent}/dN` id per call (monotonic), stamps it onto the
    /// enqueued job, and isolates the sequence per parent.
    async fn fanout_mint_behaviour(store: &dyn SessionStore) {
        let a = SessionId::new("pa");
        let b = SessionId::new("pb");
        let a1 = store.enqueue_detached_job(detached_job(&a)).await.unwrap();
        let a2 = store.enqueue_detached_job(detached_job(&a)).await.unwrap();
        let b1 = store.enqueue_detached_job(detached_job(&b)).await.unwrap();
        assert_eq!(a1.as_str(), "pa/d1");
        assert_eq!(a2.as_str(), "pa/d2");
        assert_eq!(b1.as_str(), "pb/d1", "the sequence is per-parent");
        assert_ne!(a1, a2);

        // Each job carries its pre-minted child id (FIFO order).
        let j1 = store.dequeue_job().await.expect("job 1");
        assert_eq!(j1.child, Some(a1));
        let j2 = store.dequeue_job().await.expect("job 2");
        assert_eq!(j2.child, Some(a2));
        let j3 = store.dequeue_job().await.expect("job 3");
        assert_eq!(j3.child, Some(b1));
    }

    #[tokio::test]
    async fn in_memory_fanout_mints_unique_children() {
        fanout_mint_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_fanout_mints_unique_children() {
        fanout_mint_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }

    /// A detached child's terminal completion pushes exactly one `CompletionNotice` (idempotent on
    /// re-completion), stamps `terminal_ms`, keeps the child tree-visible, and touches neither the
    /// parent's `completion_inbox` (unapplied) nor the `wake_outbox`.
    async fn notice_branch_behaviour(store: &dyn SessionStore) {
        let parent = SessionId::new("parent");
        let child = SessionId::new("parent/d1");
        // A parent row so we can assert its completion inbox stays empty (no job was fulfilled).
        store
            .create_session(
                parent.clone(),
                PartitionId::DEFAULT,
                SnapshotBlob::default(),
            )
            .await
            .unwrap();
        let parent_fence = store.acquire_activation_lease(&parent).await.unwrap();

        store.bind_completion_notice(&child, &parent).await.unwrap();
        // The child is tree-visible under the parent even before it materializes.
        assert!(
            store.children_of(&parent).await.contains(&child),
            "the detached child shows up in the parent's tree/child index"
        );

        store
            .create_session(child.clone(), PartitionId::DEFAULT, SnapshotBlob::default())
            .await
            .unwrap();
        let fence = store.acquire_activation_lease(&child).await.unwrap();
        store
            .mark_completed(
                Checkpoint::new(child.clone(), Epoch(1), SnapshotBlob::default())
                    .with_completion_payload(Some(b"did the thing".to_vec())),
                fence,
            )
            .await
            .unwrap();

        // Exactly one notice, carrying the structured payload, addressed parent<-child.
        let notice = store
            .dequeue_completion_notice()
            .await
            .expect("one completion notice");
        assert_eq!(notice.parent, parent);
        assert_eq!(notice.child, child);
        assert_eq!(notice.payload, b"did the thing".to_vec());
        assert!(
            store.dequeue_completion_notice().await.is_none(),
            "exactly one notice per terminal child"
        );

        // The notice branch never touches the job-completion rails: no wake, no parent completion.
        assert!(
            store.dequeue_wake().await.is_none(),
            "a detached child never wakes its parent through the wake outbox"
        );
        let parent_activation = store
            .load_for_activation(&parent, parent_fence)
            .await
            .unwrap();
        assert!(
            parent_activation.unapplied.is_empty(),
            "a detached child records no completion_inbox entry for the parent"
        );

        // terminal_ms stamped (same transaction as the flip).
        assert!(
            store
                .session_meta(&child)
                .await
                .and_then(|m| m.terminal_ms)
                .is_some(),
            "mark_completed stamps the terminal clock on a detached child too"
        );

        // Re-completion (a resumed child) fires the notice at most once (idempotent).
        let fence2 = store.acquire_activation_lease(&child).await.unwrap();
        store
            .mark_completed(
                Checkpoint::new(child.clone(), Epoch(2), SnapshotBlob::default()),
                fence2,
            )
            .await
            .unwrap();
        assert!(
            store.dequeue_completion_notice().await.is_none(),
            "a re-completed detached child does not fire a second notice"
        );
    }

    #[tokio::test]
    async fn in_memory_notice_branch_fires_once() {
        notice_branch_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_notice_branch_fires_once() {
        notice_branch_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }
}
