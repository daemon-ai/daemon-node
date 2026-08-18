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
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Mutex;

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteStore;

/// The durable status of a session record (lifecycle §5; session-unification §2).
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
    /// The session exists but has no runnable work (session-unification §2): a blank interactive
    /// creation, or (stage 3) a committed interactive turn with an empty inbox. NEVER selected by
    /// the recovery scanner and never woken — the status rule is `Ready` iff unconsumed work
    /// exists, `Idle` iff none does. Every transition that creates work flips `Idle -> Ready` in
    /// the same transaction that records the work.
    Idle,
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
    /// The number of durably committed turns (session-unification §5). The in-flight turn's
    /// identity — and its journal segment index — is this value; `commit_turn` (or a terminal
    /// `mark_completed`) increments it. Orthogonal to `epoch`, which fences suspensions.
    pub turn_seq: u64,
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
    /// The ad-hoc inline engine spec an [`orchestrate spawn { source: Inline }`] child was
    /// materialized from (Phase 1): the opaque CBOR of the host's `ProfileSpec` (the store stays
    /// protocol-free, mirroring `overlay`). The resolver decodes it at hydrate to build the
    /// sub-agent's engine when `bound_profile` is `None`. Empty for every non-inline session
    /// (bound-profile / default) and legacy rows.
    #[serde(default)]
    pub inline_profile: Vec<u8>,
}

/// The bound on the ancestor walk [`owns_subtree`] performs — defense against a pathological/cyclic
/// parent chain in the meta rows. The id-prefix fast path covers the common case without any walk.
pub const MAX_LINEAGE_WALK: usize = 16;

/// Whether `target` sits in the subtree `parent` owns — the shared subtree-authorization check
/// reused by the orchestrate tool (`send`/`cancel`) and the `profile_manage` tool (view/edit/delete
/// scoping). Fast path: the durable child minter encodes lineage in the id (`{parent}/c{epoch}[/c…]`),
/// so a `{parent}/` prefix proves descent without any store read. Fallback: walk the durable
/// [`SessionMeta::parent`] chain upward (bounded by [`MAX_LINEAGE_WALK`]), so a child whose id does
/// not embed the caller (e.g. a re-parented session) still authorizes. NOT reflexive: `parent ==
/// target` is not a descent (a caller manages its OWN artifacts by an explicit equality check).
pub async fn owns_subtree(
    store: &dyn SessionStore,
    parent: &SessionId,
    target: &SessionId,
) -> bool {
    if target
        .as_str()
        .starts_with(&format!("{}/", parent.as_str()))
    {
        return true;
    }
    let mut cursor = target.clone();
    for _ in 0..MAX_LINEAGE_WALK {
        let Some(meta) = store.session_meta(&cursor).await else {
            return false;
        };
        match meta.parent {
            Some(p) if p == *parent => return true,
            Some(p) => cursor = p,
            None => return false,
        }
    }
    false
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

/// A durable per-transport-instance preference row (wire v35): the operator's desired
/// enabled/disabled state plus an optional human label (rename) for a transport instance
/// (account), keyed by the instance-qualified transport id string (e.g. `"matrix/@bot:hs.org"`,
/// `"room"`). The node consults `enabled` at boot/spawn and reconnect, and overlays `label` onto
/// the adapter-reported `TransportInstanceInfo` in `transport_instances()` — so the store stays
/// protocol-free (plain strings, no wire types) exactly like [`ChatRoute`]/[`Room`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportPref {
    /// The instance-qualified transport id (primary key for upsert/lookup).
    pub transport: String,
    /// The operator's desired enabled state (`false` = disconnected now + skipped at spawn).
    pub enabled: bool,
    /// The operator-set human label/rename (`None` = no custom label).
    pub label: Option<String>,
    /// The persisted per-instance NON-SECRET account-settings values (wire v38), keyed by the
    /// adapter's `account_schema` field keys. Plain strings, so the store stays protocol-free.
    /// SECURITY INVARIANT: secrets never land here — they go to the credential store via the
    /// auth flows; this map holds only non-secret configuration.
    pub settings: BTreeMap<String, String>,
}

/// A durable manually-registered foreign-agent catalog entry (I7): the operator-persisted half of
/// the agent discovery catalog (auto-discovered builtins are re-probed each scan and need no
/// persistence). `entry` is the opaque host-encoded CBOR of the wire `AgentEntry`; the store stays
/// protocol-free. (The type + table keep their historical `acp` names — the rows are opaque, so
/// the wire-v29 `AcpAgentEntry` -> `AgentEntry` rename needed no store migration.)
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcpEntry {
    /// The agent catalog key (display name; the primary key for upsert/lookup/delete).
    pub name: String,
    /// The opaque host descriptor (CBOR of the wire `AgentEntry`).
    pub entry: Vec<u8>,
}

/// A durable user-defined custom provider entry (the "generalized Daemon Cloud" write model): the
/// persisted half of the provider catalog. `entry` is the opaque host-encoded CBOR of the wire
/// `CustomProvider`; the store stays protocol-free (mirrors [`AcpEntry`]).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomProviderRecord {
    /// The custom-provider id (the primary key for upsert/lookup/delete).
    pub id: String,
    /// The opaque host descriptor (CBOR of the wire `CustomProvider`).
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

/// A durable saved-presence row. The store stays protocol-free: `payload` is the opaque
/// host-encoded CBOR of the wire `SavedPresence` (mirroring `cron_jobs.spec`), while `id` is the
/// typed primary key for upsert/lookup/delete. Rows are insertion-ordered by the backend so the
/// host-side `PresenceManager` list (and its active index) is stable across reloads.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSavedPresence {
    /// The opaque saved-presence id (primary key).
    pub id: String,
    /// The opaque host descriptor (CBOR of the wire `SavedPresence`).
    pub payload: Vec<u8>,
}

/// A durable user-feedback record on the feedback outbox (N1: "user feedback over OpenTelemetry").
///
/// The store stays protocol-free: every field is a primitive/string the host mapped from the wire
/// `FeedbackSubmit` (`kind`/`rating`/`consent` are stable lowercase strings). SQLite persists the
/// whole record as an opaque CBOR blob plus indexed `id`/`created_at_ms`/`delivered` columns; the
/// exporter (a sibling workstream, wired in the integration phase) drains it via
/// [`SessionStore::feedback_pending`] and marks each delivered with
/// [`SessionStore::feedback_mark_delivered`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackRecord {
    /// The opaque feedback id (primary key; host-minted, e.g. `fb-<hex>`).
    pub id: String,
    /// Unix milliseconds the feedback was accepted by the node.
    pub created_at_ms: i64,
    /// The feedback flavor (`"response"` / `"app"`).
    pub kind: String,
    /// The thumbs rating (`"up"` / `"down"`), when given.
    pub rating: Option<String>,
    /// The free-form comment, when given (already length-validated by the host).
    pub comment: Option<String>,
    /// Whether the submitter consented to including the rated response content in the export.
    pub include_content: bool,
    /// The rated response's session, for response feedback (`None` for app feedback).
    pub session: Option<String>,
    /// The rated response's durable journal cursor, for response feedback.
    pub cursor: Option<u64>,
    /// The rated turn's trace-context id, when the client supplied it.
    pub trace: Option<u64>,
    /// The UI surface the feedback came from (free-form label).
    pub surface: String,
    /// The submitting app's version string, when supplied.
    pub app_version: Option<String>,
    /// The submitting app's OS/platform string, when supplied.
    pub os: Option<String>,
    /// Consent provenance: `"opted-in"` if the global telemetry toggle was on at submit time, else
    /// `"explicit-one-shot"` (explicit feedback is per-event consent, queued even when the toggle
    /// is off).
    pub consent: String,
    /// The node version (`daemon_common::VERSION`) that accepted the feedback.
    pub node_version: String,
    /// The rated turn's model, resolved best-effort at submit time (the session's bound model).
    /// Rendered as `gen_ai.request.model`. `None` for app feedback or when unresolvable.
    #[serde(default)]
    pub model: Option<String>,
    /// The rated turn's provider, when resolvable (best-effort). Rendered as `gen_ai.provider.name`.
    #[serde(default)]
    pub provider: Option<String>,
    /// The rated turn's stop/finish reason, when resolvable. Rendered as
    /// `gen_ai.response.finish_reasons`. (Per-turn end_reason is currently only journaled as a
    /// `mgmt.turn_finished` debug string, so this stays `None` pending a structured per-turn summary.)
    #[serde(default)]
    pub end_reason: Option<String>,
    /// The rated turn's prompt tokens, when resolvable. Rendered as `gen_ai.usage.input_tokens`.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// The rated turn's completion tokens, when resolvable. Rendered as `gen_ai.usage.output_tokens`.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// The rated response text, captured (size-capped) at submit time ONLY when the submitter set
    /// `include_content` (per-event consent). Rendered as `daemon.feedback.content`. This is what
    /// makes a response thumb self-describing rather than a bare `(session, cursor)` anchor.
    #[serde(default)]
    pub response_content: Option<String>,
    /// Whether the exporter has drained + delivered this record (set by `feedback_mark_delivered`).
    pub delivered: bool,
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
    /// The parent's spawning tool `call_id` (wire v29), recorded at
    /// [`bind_completion_notice`](SessionStore::bind_completion_notice) so the injected notice
    /// turn can chip-link back to the delegation card. `None` for pre-v29 edges.
    #[serde(default)]
    pub call_id: Option<String>,
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
    /// The durable inbox splices this activation claimed (session-unification §4.2): every
    /// unconsumed splice, CAS-flipped `Pending`/stale-`Claimed` → `Claimed { fence }` inside the
    /// same load transaction. The incarnation folds them and stamps the consumed cursor onto its
    /// commit ([`Checkpoint::consumed_splices`]); a crash before that commit leaves them
    /// `Claimed`, reclaimable exactly once by the next (newer-fenced) activation.
    /// `#[serde(default)]` keeps the brokered-store wire (StoreCall) compatible.
    #[serde(default)]
    pub splices: Vec<InboxSplice>,
    /// The persisted [`ExecutionPolicy`] stamped at creation (session-unification §3), read in the
    /// SAME load transaction so the incarnation's turn-boundary decision (terminal vs a
    /// non-terminal turn commit) can never race a policy it didn't run under. `None` on legacy
    /// rows — the incarnation then keeps today's terminal semantics.
    #[serde(default)]
    pub policy: Option<ExecutionPolicy>,
    /// The in-flight turn's identity (session-unification §5): the session's committed-turn count
    /// at load time, which is also the journal segment index this activation appends into. A
    /// resumed suspension re-loads the same value (the turn is still in flight), so its appends
    /// continue the same open segment.
    #[serde(default)]
    pub turn_seq: u64,
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
    /// The highest inbox `splice_seq` this checkpoint's snapshot has folded (session-unification
    /// §4.2/§5): the store flips every splice at or below it to `Consumed` inside the SAME fenced
    /// commit transaction — consumption is never written separately. `None` = no splice statement
    /// (legacy writers; nothing is consumed).
    #[serde(default)]
    pub consumed_splices: Option<u64>,
    /// The completion-inbox rows this checkpoint's snapshot has APPLIED (the `(epoch, job)` keys
    /// the activation load delivered as `Activation::unapplied` and the engine folded): the store
    /// deletes exactly these rows inside the SAME commit transaction (session-unification §5 —
    /// the completion sibling of `consumed_splices`). Without this, `commit_turn`'s Idle-iff-no-
    /// work rule would see the already-folded completions as forever-pending work and livelock the
    /// session on `Ready` + self-wake. A completion that raced in AFTER the load is not listed,
    /// stays durable, and correctly forces `Ready`.
    #[serde(default)]
    pub applied_completions: Vec<(Epoch, JobId)>,
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
            consumed_splices: None,
            applied_completions: Vec::new(),
        }
    }

    /// Attach a structured completion payload (used when a delegated child completes).
    pub fn with_completion_payload(mut self, payload: Option<Vec<u8>>) -> Self {
        self.completion_payload = payload;
        self
    }

    /// Stamp the consumed-splice cursor (session-unification §4.2): every inbox splice at or
    /// below `seq` is flipped `Consumed` inside this checkpoint's commit transaction.
    pub fn with_consumed_splices(mut self, seq: Option<u64>) -> Self {
        self.consumed_splices = seq;
        self
    }

    /// Stamp the applied completion keys (the `Activation::unapplied` rows this snapshot folded):
    /// the store deletes exactly these completion-inbox rows inside this checkpoint's commit
    /// transaction, so an applied completion never counts as pending work again.
    pub fn with_applied_completions(mut self, keys: Vec<(Epoch, JobId)>) -> Self {
        self.applied_completions = keys;
        self
    }
}

/// The journal-root seal a [`commit_turn`](SessionStore::commit_turn) writes inside its own
/// transaction (session-unification §5 item 3): the committed turn's segment — identified by the
/// session's `turn_seq` — sealed to a signed Merkle root. Computed by the incarnation's journal
/// sink over the already-durable entry rows; the store only persists the root row, atomically with
/// the snapshot and splice consumption. `None` on a commit for a session that isn't journaling.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSeal {
    /// The segment being sealed. MUST equal the session's in-flight `turn_seq` (the value the
    /// activation loaded); a mismatch is an incarnation bug and fails the commit.
    pub segment: u64,
    /// The recomputed Merkle root over the segment's entries.
    pub root: MerkleRoot,
    /// The ed25519 signature over the root.
    pub signature: Vec<u8>,
}

/// What a fenced [`commit_turn`](SessionStore::commit_turn) decided (session-unification §5): the
/// committed turn's identity and the durable status the session landed in — `Idle` iff no
/// unconsumed work (pending splices / unapplied completions) remained inside the commit
/// transaction, else `Ready` (with a self-wake enqueued so raced-in input is never stranded).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCommit {
    /// The committed turn's identity (= its journal segment index; the session's `turn_seq` was
    /// advanced past it in the same transaction).
    pub turn_seq: u64,
    /// The post-commit durable status: [`SessionStatus::Idle`] or [`SessionStatus::Ready`].
    pub status: SessionStatus,
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
    /// The §12 exec-approval command fingerprint (wire v28): the lowercase-hex sha256 of the resolved
    /// command tuple, mirrored from the engine's `PendingApproval.fingerprint` so the operator
    /// surface ([`ApprovalInfo`](daemon_api::ApprovalInfo)) can display it structurally. `None` for
    /// non-command approvals and pre-v28 rows (`#[serde(default)]`). Display-only — the durable
    /// re-run enforcement remains keyed on the engine's typed fingerprint.
    #[serde(default)]
    pub fingerprint: Option<String>,
    /// The operator's decision once answered (`None` while still pending; `Some(true)` = allow).
    pub decision: Option<bool>,
}

/// Encode an operator approval decision as the [`JobCompletion`] payload the engine's
/// `resolve_approvals` decodes (shared by both store backends). The sentinels are stable:
/// `allow_permanent` / `allow` / `deny` — permanence still starts with "allow" (the engine's
/// allow/deny split is unchanged) and a deny `reason` (wire v29) rides as `deny:{reason}` so the
/// engine can inject the operator's own words as the gated tool's error content. A reasonless deny
/// keeps the bare legacy `deny` sentinel.
pub fn approval_completion_payload(
    allow: bool,
    allow_permanent: bool,
    reason: Option<&str>,
) -> Vec<u8> {
    match (allow, allow_permanent, reason.map(str::trim)) {
        (true, true, _) => b"allow_permanent".to_vec(),
        (true, false, _) => b"allow".to_vec(),
        (false, _, Some(reason)) if !reason.is_empty() => format!("deny:{reason}").into_bytes(),
        (false, _, _) => b"deny".to_vec(),
    }
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
    /// A creation op found the session already present (session-unification §3): creation is
    /// insert-if-absent, never a silent state reset (`INSERT OR REPLACE` recreated the scanner
    /// race by resetting epoch/status/fence under a concurrent incarnation).
    #[error("session already exists: {0}")]
    AlreadyExists(SessionId),
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
            StoreError::AlreadyExists(id) => {
                StoreErrorWire::Other(format!("session already exists: {id}"))
            }
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
    /// Crash inside [`SessionStore::create_runnable`] mid-construction — after the session row,
    /// before the meta/edge/input bind. The commit is one transaction, so the observable outcome
    /// must be NOTHING persisted (session-unification §3: a scan racing construction sees either
    /// nothing runnable or the complete session).
    MidRunnableConstruction,
}

/// Rung 3 (api/39): the time-to-live of a `command_dedup` row — 24h. The retry window that
/// matters spans a node restart (06 open-Q5), so the guarantee is durable, not an in-memory LRU;
/// the TTL + a bounded key set keep the table from growing without bound. A read past the TTL
/// re-executes the op and re-caches (see [`SessionStore::command_dedup_get`]).
pub const COMMAND_DEDUP_TTL_MS: u64 = 24 * 60 * 60 * 1000;

/// The kind of a durable inbox splice (session-unification §4.1): the three durable input
/// intents. `Observe` IS spliced (§4.3) — it mutates conversation context and must survive
/// restart — but never triggers a model turn by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpliceKind {
    /// A user message that starts (or queues) a turn.
    StartTurn,
    /// Mid-turn steering input.
    Steer,
    /// Context-only input: folds into the conversation without triggering a turn.
    Observe,
}

/// The claim lifecycle of one inbox splice (session-unification §4.2). `Claimed` is the
/// crash-recovery midpoint: an incarnation took the splice into a turn that has not yet reached a
/// durable commit; a newer fence reclaims it exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpliceClaim {
    /// Appended (and therefore acknowledged), not yet taken into a turn.
    Pending,
    /// Taken into a turn by the incarnation holding `fence`; reverts to claimable if that turn
    /// never commits (a newer fence may reclaim).
    Claimed {
        /// The claiming activation's fence token value.
        fence: u64,
    },
    /// A durable commit captured this splice's effect; replay skips it. Written only inside the
    /// commit transaction ([`Checkpoint::consumed_splices`]), never separately.
    Consumed {
        /// The durable turn marker of the consuming commit (stage 3 promotes this to the
        /// monotonic `turn_seq`; until then it records the committing epoch).
        turn_seq: u64,
    },
}

/// One durable inbox splice (session-unification §4.1): a typed session input that is
/// acknowledged only after it is durable (splice-before-ack). The store sees the payload as
/// opaque CBOR (a `UserMsg`) — the typed envelope lives in the protocol crate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboxSplice {
    /// The session this input belongs to.
    pub session_id: SessionId,
    /// Store-assigned per-session monotonic sequence — never reused, never renumbered.
    pub splice_seq: u64,
    /// The input intent.
    pub kind: SpliceKind,
    /// The opaque CBOR `UserMsg` payload (typed; kind/request_id/origin preserved — the F4
    /// bare-bytes collapse is retired).
    pub payload: Vec<u8>,
    /// The dedupe identity (§4.2): `UNIQUE(session_id, origin_op)` within the retention window —
    /// a producer retry after a crash-before-ack returns the original `splice_seq`.
    pub origin_op: String,
    /// Producer provenance (wire client, notice worker, factory, ...).
    pub origin: String,
    /// Wall-clock provenance (unix millis).
    pub received_at_ms: u64,
    /// The claim state.
    pub claim: SpliceClaim,
}

/// The producer-facing input to [`SessionStore::append_splice`] (everything but the
/// store-assigned `splice_seq`/`received_at_ms`/`claim`).
#[derive(Clone, Debug)]
pub struct NewSplice {
    /// The session to append to.
    pub session_id: SessionId,
    /// The input intent.
    pub kind: SpliceKind,
    /// The opaque CBOR `UserMsg` payload.
    pub payload: Vec<u8>,
    /// The dedupe identity (see [`InboxSplice::origin_op`]).
    pub origin_op: String,
    /// Producer provenance.
    pub origin: String,
}

/// The retention window for consumed splices (session-unification §4.2): consumed rows older
/// than this are prunable, and the append dedupe guarantee is scoped to it.
pub const SPLICE_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// The persisted per-session execution policy (session-unification §3): written at creation,
/// driving terminal-vs-idle at the turn boundary and role-aware failure (stage 3). NEVER inferred
/// from the presence of one binding. The engine backend (Core vs Foreign) is an orthogonal
/// dimension — a foreign child may itself be joining or detached — so backend kind is NOT a
/// policy value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPolicy {
    /// A user-driven conversation: turns commit back to `Idle`/`Ready`, never terminal; a failed
    /// turn stays retryable (retry = a new user action).
    InteractiveRoot,
    /// A delegated child whose terminal completion fulfills its parent's job (wakes the suspended
    /// parent) — success or failure.
    JoiningChild,
    /// A detached child whose terminal completion delivers a notice to its parent (no parent job).
    DetachedChild,
    /// An attached, non-joining background child (skill/memory review): self-closes terminal.
    BackgroundChild,
    /// A one-shot scheduled run: terminal at its first turn boundary, closing the cron run.
    CronRun,
}

/// The durable edge a seeded child is created with, committed atomically inside
/// [`SessionStore::create_runnable`] so a recovery scan firing mid-construction can never run a
/// child whose parent linkage is missing (session-unification §3).
#[derive(Clone, Debug)]
pub enum RunnableEdge {
    /// A joining delegation: the child's terminal completion fulfills this parent job
    /// (see [`SessionStore::bind_delegation`]).
    Delegation(JobCommand),
    /// A detached child: terminal completion pushes a [`CompletionNotice`] to `parent`
    /// (see [`SessionStore::bind_completion_notice`]).
    CompletionNotice {
        /// The parent session the notice is delivered to.
        parent: SessionId,
        /// The parent's spawning tool call, for chip-link provenance (wire v29).
        call_id: Option<String>,
    },
    /// An attached, non-joining background edge: tree-visible, self-closing
    /// (see [`SessionStore::record_child_edge`]).
    ChildEdge {
        /// The parent session the child appears under.
        parent: SessionId,
        /// The tree projection's `work` label.
        work_label: String,
    },
}

/// Everything a seeded producer publishes for a runnable session, committed in ONE transaction by
/// [`SessionStore::create_runnable`] with the `Ready` status landing only as part of that commit
/// (session-unification §3). A scan firing at any point during construction therefore sees either
/// nothing runnable or the complete session.
#[derive(Clone, Debug)]
pub struct RunnableSession {
    /// Stable logical identity.
    pub id: SessionId,
    /// Owning partition.
    pub partition: PartitionId,
    /// The seeded initial snapshot (opaque CBOR).
    pub snapshot: SnapshotBlob,
    /// The persisted execution policy (stage 3 consumes it at the turn boundary).
    pub policy: ExecutionPolicy,
    /// Host-level meta to stamp at creation (owner/role/parent/profile/…), if any.
    pub meta: Option<SessionMeta>,
    /// The durable parent edge, if this is a child (joining / detached / background).
    pub edge: Option<RunnableEdge>,
    /// A first pending input (an opaque CBOR `UserMsg`) for engines that drain the
    /// pending-input seam at hydrate (today: the Foreign-child task; stage 2 migrates this to
    /// the typed inbox).
    pub first_input: Option<Vec<u8>>,
}

/// The durable session store — the sole authority for activation state (lifecycle §4–§5).
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Create a fresh session row in `Ready` state with an initial snapshot. Insert-if-absent:
    /// an existing row is NEVER reset (that was the `INSERT OR REPLACE` clobber);
    /// [`StoreError::AlreadyExists`] surfaces a duplicate create.
    async fn create_session(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError>;

    /// Create a blank interactive session in `Idle` state (session-unification §2/§3): a durable
    /// row + [`ExecutionPolicy::InteractiveRoot`], NEVER scanner work (the scanner selects only
    /// `Ready`/`Active`). Replaces the `session_create` path's `Ready` write — the incident's
    /// root cause was a non-runnable blank presented as activation work. Insert-if-absent
    /// ([`StoreError::AlreadyExists`] on a duplicate). Default: unsupported (a non-authoritative
    /// proxy store never creates sessions).
    async fn create_idle(
        &self,
        _id: SessionId,
        _partition: PartitionId,
        _snapshot: SnapshotBlob,
    ) -> Result<(), StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "create_idle: not supported by this store".into(),
        )))
    }

    /// The seeded-producer factory (session-unification §3): snapshot + meta + parent edge +
    /// first pending input + execution policy committed in ONE transaction, with the `Ready`
    /// status landing only as part of that commit — a recovery scan racing construction sees
    /// either nothing runnable or the complete session. Returns `true` if the session was
    /// created, `false` if it already existed (nothing written — the caller's idempotent
    /// re-bind/re-wake path applies). Default: unsupported (a non-authoritative proxy store).
    async fn create_runnable(&self, _spec: RunnableSession) -> Result<bool, StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "create_runnable: not supported by this store".into(),
        )))
    }

    /// The persisted [`ExecutionPolicy`] stamped at creation (`None` for legacy rows and stores
    /// without the seam — stage 3 falls back to edge inspection for those). Default: `None`.
    async fn execution_policy(&self, _id: &SessionId) -> Option<ExecutionPolicy> {
        None
    }

    /// Append a durable inbox splice (session-unification §4.2). ONE transaction:
    /// append-or-return-existing on `UNIQUE(session_id, origin_op)` + the `Idle → Ready` status
    /// transition + the assigned `splice_seq` returned. The caller acknowledges the producer only
    /// after this returns (splice-before-ack) and pairs it with
    /// [`enqueue_wake`](Self::enqueue_wake) when the input should drive a turn. Default:
    /// unsupported (a non-authoritative proxy store).
    async fn append_splice(&self, _splice: NewSplice) -> Result<u64, StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "append_splice: not supported by this store".into(),
        )))
    }

    /// Reopen a SETTLED session for further input (session-unification §8): flip `Completed` back
    /// to `Ready` so an explicit client `StartTurn`/`Steer` splice runs a fresh turn over the
    /// retained durable transcript (an operator re-driving a settled delegated child). Returns
    /// `true` when the flip happened, `false` when the session was not `Completed` (no-op).
    /// Deliberately NOT part of `append_splice`: host-originated injections (notifications) must
    /// keep dropping input at settled sessions — only the wire's addressed submit resurrects.
    /// Default: `false` (a non-authoritative proxy store).
    async fn reopen_if_settled(&self, _id: &SessionId) -> Result<bool, StoreError> {
        Ok(false)
    }

    /// Every unconsumed splice (`Pending`/`Claimed`) with `splice_seq > after_seq`, ordered by
    /// sequence — the replay/projection read. Default: empty.
    async fn splices_after(&self, _id: &SessionId, _after_seq: u64) -> Vec<InboxSplice> {
        Vec::new()
    }

    /// Fenced claim CAS (session-unification §4.2): flip every `Pending` splice — and every
    /// `Claimed { old }` where `old < fence.0` (the exactly-once crash reclaim) — to
    /// `Claimed { fence }`, returning the full claimed set (including rows already claimed by
    /// this same fence, so a re-load under one activation is idempotent), ordered by sequence.
    /// Fails `Fenced` for a fence below the session's current lease. Default: unsupported.
    async fn claim_splices(
        &self,
        _id: &SessionId,
        _fence: FenceToken,
    ) -> Result<Vec<InboxSplice>, StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "claim_splices: not supported by this store".into(),
        )))
    }

    /// Prune `Consumed` splices received before the absolute `cutoff_ms` unix-millis instant
    /// (retention, §4.2 — callers pass `now - SPLICE_RETENTION_MS`): the dedupe guarantee is
    /// scoped to the retention window. Unconsumed splices are NEVER pruned. Returns the number
    /// pruned. Default: 0.
    async fn prune_consumed_splices(&self, _cutoff_ms: u64) -> u64 {
        0
    }

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

    /// The durable turn boundary (session-unification §5), fenced, in ONE transaction:
    ///
    /// 1. the snapshot blob, advancing the session's monotonic `turn_seq` past the committed turn;
    /// 2. the consumed-splice cursor ([`Checkpoint::consumed_splices`]) — every folded splice
    ///    flips `Consumed { turn_seq }` with the committed turn's identity;
    /// 3. the turn's journal segment root (`seal`), promoted atomically with the state it covers;
    /// 4. the next status: `Idle` iff no unconsumed work (pending splices / unapplied completions)
    ///    remains, else `Ready` **with a self-wake enqueued** — input that raced in mid-turn is
    ///    never stranded on an `Idle` session.
    ///
    /// The non-terminal sibling of [`mark_completed`](Self::mark_completed): the persisted
    /// [`ExecutionPolicy`] decides which of the two a turn ends in (`Step::TurnCommitted` vs
    /// `Step::Completed` at the activation layer). Default: unsupported (a non-authoritative
    /// proxy store).
    async fn commit_turn(
        &self,
        _checkpoint: Checkpoint,
        _seal: Option<TurnSeal>,
        _fence: FenceToken,
    ) -> Result<TurnCommit, StoreError> {
        Err(StoreError::Common(DaemonError::Other(
            "commit_turn: not supported by this store".into(),
        )))
    }

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
    /// instead. `call_id` (wire v29) is the parent's spawning tool call, stamped onto the edge so
    /// the eventual notice carries chip-link provenance. Idempotent. Default: a no-op (a
    /// non-authoritative proxy store).
    async fn bind_completion_notice(
        &self,
        _child: &SessionId,
        _parent: &SessionId,
        _call_id: Option<String>,
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

    /// Compare-and-swap a DORMANT session's snapshot blob: atomically replace it with `new` only
    /// when the session is NOT `Active` and its current blob equals `expected`. The host-mediated
    /// snapshot-edit seam (wire v29 `FingerprintRevoke`): an operator mutation of durable engine
    /// state must never race a running incarnation (which would overwrite the edit at its next
    /// checkpoint), so an `Active` session — or a blob that changed since the caller's read —
    /// refuses with `Ok(false)` and the caller retries. The store keeps treating the blob as
    /// opaque bytes (the typed decode/encode lives host-side). Default: `false` (a
    /// non-authoritative proxy store).
    async fn swap_snapshot_if_dormant(
        &self,
        _id: &SessionId,
        _expected: &SnapshotBlob,
        _new: SnapshotBlob,
    ) -> Result<bool, StoreError> {
        Ok(false)
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
    /// `allow`/`allow_permanent`/`deny`/`deny:{reason}`) so the rehydrated engine resolves the gated
    /// tool call, and publish a wake. `allow_permanent` (Cluster B) carries the operator's "Allow
    /// permanently" choice through to the engine via the completion payload (`allow_permanent`),
    /// where `resolve_approvals` remembers the verified command fingerprint for the session; it is
    /// meaningful only when `allow`. `reason` (wire v29) is the operator's optional deny
    /// justification, riding the payload as `deny:{reason}` so the engine injects it as the gated
    /// tool's error content; it is meaningful only when NOT `allow`. Idempotent per `(session,
    /// epoch, job)` (a redelivered answer is a no-op). Returns `true` if a matching pending approval
    /// was found and answered. Default: `false` (no such row).
    async fn answer_approval(
        &self,
        _session: &SessionId,
        _job_id: &JobId,
        _allow: bool,
        _allow_permanent: bool,
        _reason: Option<String>,
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

    /// Scan for sessions with resumable work for the recovery scanner (lifecycle §5;
    /// invariant #7): `Ready`/`Active`, plus `Suspended` sessions holding an unapplied
    /// completion. The last case is the safety net for an absorbed completion wake — a wake that
    /// raced the suspending cycle's slot release is a lost HINT, and a `Suspended` session is
    /// reachable by no other rail (never `Ready`, so a status-only scan would strand it forever).
    /// A `Suspended` session with NO recorded completion stays un-scanned (still waiting on its
    /// job), as does a blank `Idle` one (the §2 incident pin).
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

    /// List the node-wide tool enable/disable overrides (wire v30; `ToolSetEnabled`) as
    /// `(tool, enabled)` pairs. `tool_list` overlays these on the bound inventory and per-session
    /// tool wiring consults them. Default: none (a store without the override table).
    async fn tool_overrides(&self) -> Vec<(String, bool)> {
        Vec::new()
    }

    /// Upsert a node-wide tool enable/disable override (wire v30). Default: no-op.
    async fn set_tool_override(&self, _tool: &str, _enabled: bool) -> Result<(), StoreError> {
        Ok(())
    }

    /// List every persisted per-transport-instance preference (wire v35): the desired
    /// enabled/disabled state + optional human label, keyed by transport id. The node consults
    /// these at boot/spawn (skip a fully-disabled family) and overlays `label` +
    /// `enabled` onto `transport_instances()`. Default: none (a store without the prefs table —
    /// every instance is then enabled with no custom label).
    async fn transport_prefs(&self) -> Vec<TransportPref> {
        Vec::new()
    }

    /// Upsert a transport instance's desired enabled state (wire v35), preserving any existing
    /// label. Default: no-op.
    async fn set_transport_enabled(
        &self,
        _transport: &str,
        _enabled: bool,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Set (or clear, with `None`) a transport instance's human label (wire v35), preserving its
    /// enabled state (a new row defaults to enabled). Default: no-op.
    async fn set_transport_label(
        &self,
        _transport: &str,
        _label: Option<String>,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Upsert a transport instance's persisted NON-SECRET account-settings values (wire v38),
    /// preserving its enabled state and label. The caller (the node's `transport_configure`)
    /// passes the already-MERGED effective map, so this is a whole-map replace of the `settings`
    /// column on the same prefs row the enabled/label ops use. Secrets never ride this call —
    /// they belong to the credential store. Default: no-op.
    async fn set_transport_settings(
        &self,
        _transport: &str,
        _settings: &BTreeMap<String, String>,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// List every persisted credential/account human label (wire v35) as `(profile, label)` pairs.
    /// The node overlays these onto `credential_list()`. Default: none (a store without the labels
    /// table — every credential then renders with no custom label). Backs the app's AccountsPage
    /// rename.
    async fn credential_labels(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Set (or clear, with `None`) a credential/account's human label (wire v35). Default: no-op.
    async fn set_credential_label(
        &self,
        _profile: &str,
        _label: Option<String>,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Enqueue a user-feedback record onto the durable feedback outbox (N1). Idempotent by `id`
    /// (a re-enqueue of the same id is a no-op). Default: no-op (a store without the outbox).
    async fn feedback_enqueue(&self, _record: FeedbackRecord) -> Result<(), StoreError> {
        Ok(())
    }

    /// The oldest undelivered feedback records (by `created_at_ms`), capped at `limit` (`0` = all).
    ///
    /// This is the exporter's DRAIN seam, now wired: `daemon-host`'s feedback drain
    /// (`node_api::feedback`) polls this after each `FeedbackSubmit` enqueue and once at node
    /// startup, maps each record to a `daemon_telemetry::feedback::FeedbackEvent`, ships it to the
    /// configured `telemetry.feedback_endpoint` (reusing a `FeedbackExporter` for `"opted-in"`
    /// records, `emit_one_shot` for `"explicit-one-shot"`), then calls
    /// [`Self::feedback_mark_delivered`] on success — leaving a record queued on failure or when
    /// export is inert (no endpoint / the `otel` feature off). Default: none (a store without the
    /// outbox).
    async fn feedback_pending(&self, _limit: usize) -> Vec<FeedbackRecord> {
        Vec::new()
    }

    /// Mark a feedback record delivered (idempotent) so [`Self::feedback_pending`] stops returning
    /// it. Called by the exporter after a successful ship. Default: no-op.
    async fn feedback_mark_delivered(&self, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// Read the node-owned global telemetry consent toggle (N1). Default OFF (opt-in): a store
    /// without the setting reports `false`.
    async fn telemetry_consent_get(&self) -> bool {
        false
    }

    /// Set the node-owned global telemetry consent toggle (N1). Default: no-op.
    async fn telemetry_consent_set(&self, _enabled: bool) -> Result<(), StoreError> {
        Ok(())
    }

    /// Read the node-owned crash-reporting consent toggle (wire v41). Default OFF (opt-in): a store
    /// without the setting reports `false`. Distinct from the telemetry consent above.
    async fn crash_consent_get(&self) -> bool {
        false
    }

    /// Set the node-owned crash-reporting consent toggle (wire v41). Default: no-op.
    async fn crash_consent_set(&self, _enabled: bool) -> Result<(), StoreError> {
        Ok(())
    }

    /// Rung 3 (api/39) op-id idempotent dedup: look up a prior result for `(principal, op_id)`,
    /// returning the stored CBOR bytes iff a row exists AND is unexpired at `now_ms` (within
    /// [`COMMAND_DEDUP_TTL_MS`]). An expired row is not served (and is lazily evicted) so the op
    /// re-executes. Default: `None` (a store with no dedup table — the v38-era re-execute behavior).
    async fn command_dedup_get(
        &self,
        _principal: &str,
        _op_id: &str,
        _now_ms: u64,
    ) -> Option<Vec<u8>> {
        None
    }

    /// Rung 3 (api/39): record `result` for `(principal, op_id)` stamped at `at_ms`. FIRST-
    /// writer-wins — a duplicate key does not overwrite the stored value, so a retry always sees
    /// the ORIGINAL result (an expired row, already evicted by [`Self::command_dedup_get`], is
    /// re-insertable). Default: no-op (never dedups).
    async fn command_dedup_put(
        &self,
        _principal: &str,
        _op_id: &str,
        _result: Vec<u8>,
        _at_ms: u64,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    /// Read the node-owned OpenAI-gateway runtime override (`GatewaySet`) as `(enabled, addr)`, if
    /// one was ever set. The node resolves the effective gateway config as this override layered on
    /// top of the boot `[gateway]` config, so a runtime enable/rebind survives a restart. Default:
    /// `None` (a store without the override — the boot config stands).
    async fn gateway_override(&self) -> Option<(bool, Option<String>)> {
        None
    }

    /// Persist the node-owned gateway runtime override (`enabled` + optional bind `addr`). A
    /// single-row setting (mirroring `telemetry_consent`). Default: no-op.
    async fn set_gateway_override(
        &self,
        _enabled: bool,
        _addr: Option<&str>,
    ) -> Result<(), StoreError> {
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

    // -- saved presences: the durable backing for the host `PresenceManager` --------------------

    /// List every durable saved presence in insertion order. The host `PresenceManager`
    /// loads these at startup (and seeds its default offline/available presences if absent).
    /// Default: none (a store without the saved-presence table — presences are then in-memory only).
    async fn saved_presence_list(&self) -> Vec<StoredSavedPresence> {
        Vec::new()
    }

    /// Upsert a saved presence (keyed by [`StoredSavedPresence::id`]); an existing row keeps its
    /// position, a new row is appended. Default: no-op.
    async fn saved_presence_set(&self, _presence: StoredSavedPresence) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a saved presence by id (idempotent). Default: no-op.
    async fn saved_presence_remove(&self, _id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    /// The id of the active saved presence, if one has been set. Default: `None`.
    async fn saved_presence_active_get(&self) -> Option<String> {
        None
    }

    /// Set the active saved-presence id (single-row setting, mirroring `telemetry_consent`).
    /// Default: no-op.
    async fn saved_presence_active_set(&self, _id: &str) -> Result<(), StoreError> {
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

    /// List the durable user-defined custom provider entries. Default: none.
    async fn custom_provider_list(&self) -> Vec<CustomProviderRecord> {
        Vec::new()
    }

    /// Upsert a custom provider entry (keyed by [`CustomProviderRecord::id`]). Default: no-op.
    async fn custom_provider_set(&self, _entry: CustomProviderRecord) -> Result<(), StoreError> {
        Ok(())
    }

    /// Remove a custom provider entry by id (idempotent). Default: no-op.
    async fn custom_provider_remove(&self, _id: &str) -> Result<(), StoreError> {
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

    /// Append one entry to the open `(stream, segment)` segment. Idempotent per `seq`. `fence` is
    /// `Some` on the durable path (session-unification §5: the fence rides EVERY append, so a
    /// stale incarnation can neither append into nor seal the winning segment — the seal CAS alone
    /// is insufficient) and `None` for non-durable streams.
    async fn append_trace(
        &self,
        _stream: &JournalStreamId,
        _segment: u64,
        _entry: TraceEntry,
        _fence: Option<FenceToken>,
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

    /// Backward cursor-paged read of a stream's journal (rung 2): the `max` NEWEST entries with
    /// `cursor < before_cursor` (the window ending just below `before_cursor`), returned in
    /// ASCENDING cursor order like [`Self::load_journal`]. `max == 0` = no cap. The page's
    /// `next_cursor` is the backward continuation — the OLDEST returned cursor (pass it as the
    /// next `before_cursor`), or `before_cursor` itself when the page is empty; `head_cursor` is
    /// the stream head as on a forward read. Anchoring is stable by construction: appends land
    /// above every already-served backward anchor, so an interleaved write never skips or
    /// duplicates entries across a backward page walk. Non-destructive. Default: empty (a store
    /// without a verifiable journal).
    async fn load_journal_before(
        &self,
        _stream: &JournalStreamId,
        _before_cursor: u64,
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
    /// Detached child session -> its parent + the spawning tool call (the completion-notice edge).
    /// Persistent (drives tree visibility via `child_index` + the notice firing in
    /// `mark_completed`); the in-memory analogue of the SQLite `completion_notices` table.
    completion_notices: HashMap<SessionId, (SessionId, Option<String>)>,
    /// Detached children whose terminal notice has already been pushed onto `notice_outbox`, so a
    /// re-completion (a resumed detached child) fires the notice at most once (idempotency).
    notices_fired: HashSet<SessionId>,
    /// Pending completion notices for detached children, drained by the node's notice worker (the
    /// in-memory analogue of the SQLite `completion_notice_outbox` table).
    notice_outbox: VecDeque<CompletionNotice>,
    /// Per-parent monotonic counter minting the unique `{parent}/d{n}` detached child ids.
    detached_seq: HashMap<SessionId, u64>,
    /// Per-session durable inbox splices in sequence order (session-unification §4; the in-memory
    /// analogue of the SQLite `inbox_splice` table). The per-session monotonic `splice_seq` mint
    /// rides `splice_seq` below — never reused even after pruning.
    splices: HashMap<SessionId, Vec<InboxSplice>>,
    /// Per-session highest `splice_seq` ever assigned (survives pruning: sequences never renumber).
    splice_seq: HashMap<SessionId, u64>,
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
    /// The persisted per-session execution policy, stamped at creation (session-unification §3;
    /// the in-memory analogue of the SQLite `session_record.execution_policy` column).
    execution_policy: HashMap<SessionId, ExecutionPolicy>,
    /// Durable chat→session routing pins, keyed by canonical origin key (§5.9; the in-memory analogue
    /// of the SQLite `chat_routes` table).
    chat_routes: HashMap<String, ChatRoute>,
    /// Node-wide tool enable/disable overrides (wire v30), keyed by tool name (the in-memory
    /// analogue of the SQLite `tool_overrides` table).
    tool_overrides: HashMap<String, bool>,
    /// Per-transport-instance preferences (wire v35 + v38): desired enabled state + optional
    /// label + non-secret settings values, keyed by transport id (the in-memory analogue of the
    /// SQLite `transport_prefs` table).
    #[allow(clippy::type_complexity)]
    transport_prefs: HashMap<String, (bool, Option<String>, BTreeMap<String, String>)>,
    /// Per-credential/account human labels (wire v35), keyed by profile (the in-memory analogue of
    /// the SQLite `credential_labels` table).
    credential_labels: HashMap<String, String>,
    /// Durable user-feedback outbox (N1), in enqueue order (the in-memory analogue of the SQLite
    /// `feedback_outbox` table). Keyed-dedup by `id`.
    feedback_outbox: Vec<FeedbackRecord>,
    /// The node-owned global telemetry consent toggle (N1; default OFF / opt-in). In-memory analogue
    /// of the SQLite `telemetry_consent` single-row setting.
    telemetry_consent: bool,
    /// The node-owned crash-reporting consent toggle (wire v41; default OFF / opt-in; distinct from
    /// `telemetry_consent`). In-memory analogue of the SQLite `crash_consent` single-row setting.
    crash_consent: bool,
    /// The node-owned gateway runtime override (`GatewaySet`) as `(enabled, addr)`, if ever set.
    /// In-memory analogue of the SQLite `gateway_config` single-row setting.
    gateway_override: Option<(bool, Option<String>)>,
    /// Durable manually-registered ACP catalog entries, keyed by name (I7; the in-memory analogue of
    /// the SQLite `acp_catalog` table).
    acp_catalog: HashMap<String, AcpEntry>,
    /// Durable user-defined custom providers, keyed by id (the in-memory analogue of the SQLite
    /// `custom_providers` table).
    custom_providers: HashMap<String, CustomProviderRecord>,
    /// Durable scheduled jobs, keyed by id (I15; the in-memory analogue of the SQLite `cron_jobs`
    /// table).
    cron_jobs: HashMap<String, StoredCronJob>,
    /// Durable cron run history, keyed by job id, newest last (I15; analogue of `cron_runs`).
    cron_runs: HashMap<String, Vec<StoredCronRun>>,
    /// Durable cron suggestions, keyed by id (I15; analogue of `cron_suggestions`).
    cron_suggestions: HashMap<String, StoredCronSuggestion>,
    /// Durable saved presences in insertion order (analogue of the SQLite `saved_presences`
    /// table). A `Vec` (not a `HashMap`) so the list order — which the `PresenceManager` active
    /// index depends on — is stable across reloads.
    saved_presences: Vec<StoredSavedPresence>,
    /// The active saved-presence id, if set (analogue of `saved_presence_active`).
    saved_presence_active: Option<String>,
    fault: Option<FaultPoint>,
    /// Append-only journal entries per stream, in append (cursor) order across all segments.
    journal_entries: HashMap<JournalStreamId, Vec<JournalEntry>>,
    /// Sealed segment roots per `(stream, segment)`.
    journal_roots: HashMap<(JournalStreamId, u64), CommittedRoot>,
    /// Stream-monotonic cursor allocator (the pagination key for `load_journal`).
    journal_cursor: u64,
    /// Append-only conversation-rewind seals per stream, in record order (the latest is active).
    journal_seals: HashMap<JournalStreamId, Vec<JournalSeal>>,
    /// Rung 3 (api/39) op-id dedup: `(principal, op_id) -> (result bytes, at_ms)` (the in-memory
    /// analogue of the SQLite `command_dedup` table). Bounded by the 24h TTL + lazy eviction on
    /// access. Not durable across a restart on this backend (the accepted caveat; the durable
    /// guarantee is the SQLite backend's).
    command_dedup: HashMap<(String, String), (Vec<u8>, u64)>,
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

    /// The §4.2 claim CAS under the held lock: flip `Pending` and stale-`Claimed` rows to
    /// `Claimed { fence }`, returning the full set claimed by `fence` (including rows it already
    /// held, so a re-load under one activation is idempotent), in sequence order.
    fn claim_splices_locked(
        inner: &mut Inner,
        session: &SessionId,
        fence: FenceToken,
    ) -> Vec<InboxSplice> {
        let mut claimed = Vec::new();
        if let Some(rows) = inner.splices.get_mut(session) {
            for row in rows.iter_mut() {
                match row.claim {
                    SpliceClaim::Pending => {
                        row.claim = SpliceClaim::Claimed { fence: fence.0 };
                        claimed.push(row.clone());
                    }
                    SpliceClaim::Claimed { fence: old } if old < fence.0 => {
                        row.claim = SpliceClaim::Claimed { fence: fence.0 };
                        claimed.push(row.clone());
                    }
                    SpliceClaim::Claimed { fence: same } if same == fence.0 => {
                        claimed.push(row.clone());
                    }
                    _ => {}
                }
            }
        }
        claimed
    }

    /// Flip every splice at or below `up_to` to `Consumed { turn_seq }` — called only from inside
    /// the fenced commit ops under the held lock (session-unification §4.2: consumption is written
    /// transactionally with the snapshot, never separately).
    fn consume_splices_locked(inner: &mut Inner, session: &SessionId, up_to: u64, turn_seq: u64) {
        if let Some(rows) = inner.splices.get_mut(session) {
            for row in rows.iter_mut().filter(|r| r.splice_seq <= up_to) {
                if !matches!(row.claim, SpliceClaim::Consumed { .. }) {
                    row.claim = SpliceClaim::Consumed { turn_seq };
                }
            }
        }
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

    /// Insert a fresh session row under the held lock — insert-if-absent (session-unification §3):
    /// an existing row is never reset; a duplicate create surfaces [`StoreError::AlreadyExists`].
    fn insert_fresh(
        inner: &mut Inner,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
        status: SessionStatus,
    ) -> Result<(), StoreError> {
        if inner.sessions.contains_key(&id) {
            return Err(StoreError::AlreadyExists(id));
        }
        inner.sessions.insert(
            id.clone(),
            SessionRecord {
                session_id: id,
                partition,
                epoch: Epoch::ZERO,
                status,
                snapshot,
                fence: FenceToken::ZERO,
                turn_seq: 0,
            },
        );
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
        Self::insert_fresh(&mut inner, id, partition, snapshot, SessionStatus::Ready)
    }

    async fn create_idle(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        Self::insert_fresh(
            &mut inner,
            id.clone(),
            partition,
            snapshot,
            SessionStatus::Idle,
        )?;
        inner
            .execution_policy
            .insert(id, ExecutionPolicy::InteractiveRoot);
        Ok(())
    }

    async fn create_runnable(&self, spec: RunnableSession) -> Result<bool, StoreError> {
        // One lock hold = one transaction: row + policy + meta + edge + first input land together,
        // so a scan interleaving construction sees either nothing runnable or the complete session.
        let mut inner = self.inner.lock().unwrap();
        if inner.sessions.contains_key(&spec.id) {
            return Ok(false);
        }
        // The construction-boundary crash: the SQLite backend rolls its transaction back; the
        // in-memory model faults before mutating, observing the same all-or-nothing contract.
        Self::take_fault(&mut inner, FaultPoint::MidRunnableConstruction)?;
        Self::insert_fresh(
            &mut inner,
            spec.id.clone(),
            spec.partition,
            spec.snapshot,
            SessionStatus::Ready,
        )?;
        inner.execution_policy.insert(spec.id.clone(), spec.policy);
        if let Some(meta) = spec.meta {
            inner.session_meta.insert(spec.id.clone(), meta);
        }
        // Each arm mirrors its standalone bind op exactly (`bind_delegation` /
        // `bind_completion_notice` / `record_child_edge`): dedupe the child-index row and keep
        // first-writer-wins on the notice edge — a spawn-time bind (which carries the call_id)
        // may already have run before the factory materializes the child.
        match spec.edge {
            Some(RunnableEdge::Delegation(job)) => {
                let siblings = inner.child_index.entry(job.session_id.clone()).or_default();
                if !siblings.contains(&spec.id) {
                    siblings.push(spec.id.clone());
                }
                inner.delegations.insert(spec.id.clone(), job);
            }
            Some(RunnableEdge::CompletionNotice { parent, call_id }) => {
                let siblings = inner.child_index.entry(parent.clone()).or_default();
                if !siblings.contains(&spec.id) {
                    siblings.push(spec.id.clone());
                }
                inner
                    .completion_notices
                    .entry(spec.id.clone())
                    .or_insert((parent, call_id));
            }
            Some(RunnableEdge::ChildEdge { parent, work_label }) => {
                let siblings = inner.child_index.entry(parent).or_default();
                if !siblings.contains(&spec.id) {
                    siblings.push(spec.id.clone());
                }
                inner.background_edges.insert(spec.id.clone(), work_label);
            }
            None => {}
        }
        if let Some(input) = spec.first_input {
            // The seeded first input rides the durable inbox (session-unification §4; stage 2
            // migrated it off `pending_session_input`), inside the same creation transaction.
            let seq = {
                let s = inner.splice_seq.entry(spec.id.clone()).or_insert(0);
                *s += 1;
                *s
            };
            inner
                .splices
                .entry(spec.id.clone())
                .or_default()
                .push(InboxSplice {
                    session_id: spec.id.clone(),
                    splice_seq: seq,
                    kind: SpliceKind::StartTurn,
                    payload: input,
                    // Deterministic op id: creation is already insert-if-absent, and a retried
                    // factory run that lost the create race dedupes here by construction.
                    origin_op: "first-input".into(),
                    origin: "factory".into(),
                    received_at_ms: now_ms(),
                    claim: SpliceClaim::Pending,
                });
        }
        Ok(true)
    }

    async fn execution_policy(&self, id: &SessionId) -> Option<ExecutionPolicy> {
        self.inner.lock().unwrap().execution_policy.get(id).copied()
    }

    async fn reopen_if_settled(&self, id: &SessionId) -> Result<bool, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get_mut(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        if matches!(rec.status, SessionStatus::Completed) {
            rec.status = SessionStatus::Ready;
            return Ok(true);
        }
        Ok(false)
    }

    async fn append_splice(&self, splice: NewSplice) -> Result<u64, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.sessions.contains_key(&splice.session_id) {
            return Err(StoreError::NotFound(splice.session_id.clone()));
        }
        // Append-or-return-existing on (session, origin_op): a producer retry after a
        // crash-before-ack returns the original seq instead of duplicating (§4.2).
        if let Some(existing) = inner.splices.get(&splice.session_id).and_then(|rows| {
            rows.iter()
                .find(|r| r.origin_op == splice.origin_op)
                .map(|r| r.splice_seq)
        }) {
            return Ok(existing);
        }
        let seq = {
            let s = inner
                .splice_seq
                .entry(splice.session_id.clone())
                .or_insert(0);
            *s += 1;
            *s
        };
        let row = InboxSplice {
            session_id: splice.session_id.clone(),
            splice_seq: seq,
            kind: splice.kind,
            payload: splice.payload,
            origin_op: splice.origin_op,
            origin: splice.origin,
            received_at_ms: now_ms(),
            claim: SpliceClaim::Pending,
        };
        inner
            .splices
            .entry(splice.session_id.clone())
            .or_default()
            .push(row);
        // Same transaction (the held lock): durable input on an Idle session makes it Ready —
        // the §2 status rule ("Ready iff unconsumed work exists").
        let rec = inner.sessions.get_mut(&splice.session_id).unwrap();
        if rec.status == SessionStatus::Idle {
            rec.status = SessionStatus::Ready;
        }
        Ok(seq)
    }

    async fn splices_after(&self, id: &SessionId, after_seq: u64) -> Vec<InboxSplice> {
        let inner = self.inner.lock().unwrap();
        inner
            .splices
            .get(id)
            .map(|rows| {
                rows.iter()
                    .filter(|r| {
                        r.splice_seq > after_seq && !matches!(r.claim, SpliceClaim::Consumed { .. })
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn claim_splices(
        &self,
        id: &SessionId,
        fence: FenceToken,
    ) -> Result<Vec<InboxSplice>, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        Self::check_fence(rec, fence)?;
        Ok(Self::claim_splices_locked(&mut inner, id, fence))
    }

    async fn prune_consumed_splices(&self, cutoff_ms: u64) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        let mut pruned = 0u64;
        for rows in inner.splices.values_mut() {
            let before = rows.len();
            rows.retain(|r| {
                !(matches!(r.claim, SpliceClaim::Consumed { .. }) && r.received_at_ms < cutoff_ms)
            });
            pruned += (before - rows.len()) as u64;
        }
        pruned
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
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        let snapshot = rec.snapshot.clone();
        let turn_seq = rec.turn_seq;
        let unapplied = inner.unapplied.get(id).cloned().unwrap_or_default();
        let policy = inner.execution_policy.get(id).cloned();
        // Claim the durable inbox in the same load transaction (session-unification §4.2): the
        // incarnation folds these and stamps `Checkpoint::consumed_splices`; a crash before that
        // commit leaves them `Claimed { fence }`, reclaimable by the next (newer) fence.
        let splices = Self::claim_splices_locked(&mut inner, id, fence);
        Ok(Activation {
            snapshot,
            unapplied,
            fence,
            splices,
            policy,
            turn_seq,
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
        if let Some(up_to) = checkpoint.consumed_splices {
            // The turn is still in flight (a suspension, not a boundary): stamp the in-flight
            // turn's identity, without advancing it.
            let turn = inner
                .sessions
                .get(&checkpoint.session_id)
                .map(|r| r.turn_seq)
                .unwrap_or(0);
            Self::consume_splices_locked(&mut inner, &checkpoint.session_id, up_to, turn);
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
        // A terminal commit IS this turn's boundary (session-unification §5 item 5): the turn
        // commits under its identity and the counter advances past it, exactly like `commit_turn`.
        let committed_turn = rec.turn_seq;
        rec.turn_seq += 1;
        if let Some(up_to) = checkpoint.consumed_splices {
            Self::consume_splices_locked(&mut inner, &checkpoint.session_id, up_to, committed_turn);
        }
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
        if let Some((parent, call_id)) = inner
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
                    call_id,
                    payload,
                });
            }
        }
        Ok(())
    }

    async fn commit_turn(
        &self,
        checkpoint: Checkpoint,
        seal: Option<TurnSeal>,
        fence: FenceToken,
    ) -> Result<TurnCommit, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        {
            let rec = inner
                .sessions
                .get(&checkpoint.session_id)
                .ok_or_else(|| StoreError::NotFound(checkpoint.session_id.clone()))?;
            Self::check_fence(rec, fence)?;
            if let Some(seal) = &seal {
                if seal.segment != rec.turn_seq {
                    return Err(StoreError::Common(DaemonError::Other(format!(
                        "commit_turn: seal segment {} != in-flight turn {} (incarnation bug)",
                        seal.segment, rec.turn_seq
                    ))));
                }
            }
        }
        // Boundary: abort the whole transaction before anything is written.
        Self::take_fault(&mut inner, FaultPoint::BeforeSnapshot)?;

        // Atomic commit (session-unification §5): snapshot + turn_seq + consumed splices +
        // journal-root seal + next status land together under the held lock.
        let rec = inner.sessions.get_mut(&checkpoint.session_id).unwrap();
        let committed_turn = rec.turn_seq;
        rec.snapshot = checkpoint.snapshot;
        rec.epoch = checkpoint.epoch;
        rec.turn_seq += 1;
        if let Some(up_to) = checkpoint.consumed_splices {
            Self::consume_splices_locked(&mut inner, &checkpoint.session_id, up_to, committed_turn);
        }
        // Consume the completions this turn's snapshot folded (the load-delivered `unapplied`
        // set): delete exactly those rows, so an applied completion never re-counts as pending
        // work. A completion that raced in after the load survives and forces `Ready` below.
        if !checkpoint.applied_completions.is_empty() {
            if let Some(rows) = inner.unapplied.get_mut(&checkpoint.session_id) {
                rows.retain(|c| {
                    !checkpoint
                        .applied_completions
                        .iter()
                        .any(|(e, j)| c.epoch == *e && c.job_id == *j)
                });
            }
        }
        if let Some(seal) = seal {
            let stream = JournalStreamId::session(&checkpoint.session_id);
            inner.journal_roots.insert(
                (stream, seal.segment),
                CommittedRoot {
                    root: seal.root,
                    signature: seal.signature,
                },
            );
        }
        // Idle iff no unconsumed work remains INSIDE this transaction, else Ready + self-wake —
        // a splice or completion that raced in mid-turn is never stranded on an Idle session.
        let pending_splices = inner
            .splices
            .get(&checkpoint.session_id)
            .map(|rows| {
                rows.iter()
                    .any(|r| matches!(r.claim, SpliceClaim::Pending | SpliceClaim::Claimed { .. }))
            })
            .unwrap_or(false);
        let unapplied = inner
            .unapplied
            .get(&checkpoint.session_id)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let status = if pending_splices || unapplied {
            SessionStatus::Ready
        } else {
            SessionStatus::Idle
        };
        inner
            .sessions
            .get_mut(&checkpoint.session_id)
            .unwrap()
            .status = status.clone();
        if matches!(status, SessionStatus::Ready) {
            inner.wake_outbox.push_back(checkpoint.session_id.clone());
        }
        Ok(TurnCommit {
            turn_seq: committed_turn,
            status,
        })
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
        call_id: Option<String>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Idempotent tree edge: record the child under the parent for `children_of`/tree, but not in
        // `delegations` (so `mark_completed` binds no job — the child self-closes with a notice).
        let siblings = inner.child_index.entry(parent.clone()).or_default();
        if !siblings.contains(child) {
            siblings.push(child.clone());
        }
        // First-writer-wins (mirrors the SQLite `INSERT OR IGNORE`): the spawn-time bind carries
        // the call_id; a later idempotent re-bind (the fleet worker's materialize path) must not
        // clobber it with `None`.
        inner
            .completion_notices
            .entry(child.clone())
            .or_insert((parent.clone(), call_id));
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
        // An approval park suspends mid-turn: stamp the in-flight turn without advancing it.
        let turn = rec.turn_seq;
        if let Some(up_to) = checkpoint.consumed_splices {
            Self::consume_splices_locked(&mut inner, &checkpoint.session_id, up_to, turn);
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
        allow_permanent: bool,
        reason: Option<String>,
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
            payload: approval_completion_payload(allow, allow_permanent, reason.as_deref()),
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
                    && (matches!(r.status, SessionStatus::Ready | SessionStatus::Active)
                        || (matches!(r.status, SessionStatus::Suspended { .. })
                            && inner
                                .unapplied
                                .get(&r.session_id)
                                .is_some_and(|c| !c.is_empty())))
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

    async fn swap_snapshot_if_dormant(
        &self,
        id: &SessionId,
        expected: &SnapshotBlob,
        new: SnapshotBlob,
    ) -> Result<bool, StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let rec = inner
            .sessions
            .get_mut(id)
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        if matches!(rec.status, SessionStatus::Active) || &rec.snapshot != expected {
            return Ok(false);
        }
        rec.snapshot = new;
        Ok(true)
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

    async fn tool_overrides(&self) -> Vec<(String, bool)> {
        self.inner
            .lock()
            .unwrap()
            .tool_overrides
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    async fn set_tool_override(&self, tool: &str, enabled: bool) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .tool_overrides
            .insert(tool.to_string(), enabled);
        Ok(())
    }

    async fn transport_prefs(&self) -> Vec<TransportPref> {
        self.inner
            .lock()
            .unwrap()
            .transport_prefs
            .iter()
            .map(|(transport, (enabled, label, settings))| TransportPref {
                transport: transport.clone(),
                enabled: *enabled,
                label: label.clone(),
                settings: settings.clone(),
            })
            .collect()
    }

    async fn set_transport_enabled(
        &self,
        transport: &str,
        enabled: bool,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let entry = inner
            .transport_prefs
            .entry(transport.to_string())
            .or_insert((true, None, BTreeMap::new()));
        entry.0 = enabled;
        Ok(())
    }

    async fn set_transport_label(
        &self,
        transport: &str,
        label: Option<String>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let entry = inner
            .transport_prefs
            .entry(transport.to_string())
            .or_insert((true, None, BTreeMap::new()));
        entry.1 = label;
        Ok(())
    }

    async fn set_transport_settings(
        &self,
        transport: &str,
        settings: &BTreeMap<String, String>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        let entry = inner
            .transport_prefs
            .entry(transport.to_string())
            .or_insert((true, None, BTreeMap::new()));
        entry.2 = settings.clone();
        Ok(())
    }

    async fn credential_labels(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .unwrap()
            .credential_labels
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    async fn set_credential_label(
        &self,
        profile: &str,
        label: Option<String>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        match label {
            Some(l) => {
                inner.credential_labels.insert(profile.to_string(), l);
            }
            None => {
                inner.credential_labels.remove(profile);
            }
        }
        Ok(())
    }

    async fn feedback_enqueue(&self, record: FeedbackRecord) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Idempotent by id: a re-enqueue of the same id is a no-op (mirrors the SQLite PK upsert).
        if inner.feedback_outbox.iter().any(|r| r.id == record.id) {
            return Ok(());
        }
        inner.feedback_outbox.push(record);
        Ok(())
    }

    async fn feedback_pending(&self, limit: usize) -> Vec<FeedbackRecord> {
        let inner = self.inner.lock().unwrap();
        let mut pending: Vec<FeedbackRecord> = inner
            .feedback_outbox
            .iter()
            .filter(|r| !r.delivered)
            .cloned()
            .collect();
        // Oldest first (mirrors the SQLite `ORDER BY created_at_ms, id`).
        pending.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms).then(a.id.cmp(&b.id)));
        if limit != 0 {
            pending.truncate(limit);
        }
        pending
    }

    async fn feedback_mark_delivered(&self, id: &str) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(rec) = inner.feedback_outbox.iter_mut().find(|r| r.id == id) {
            rec.delivered = true;
        }
        Ok(())
    }

    async fn telemetry_consent_get(&self) -> bool {
        self.inner.lock().unwrap().telemetry_consent
    }

    async fn telemetry_consent_set(&self, enabled: bool) -> Result<(), StoreError> {
        self.inner.lock().unwrap().telemetry_consent = enabled;
        Ok(())
    }

    async fn crash_consent_get(&self) -> bool {
        self.inner.lock().unwrap().crash_consent
    }

    async fn crash_consent_set(&self, enabled: bool) -> Result<(), StoreError> {
        self.inner.lock().unwrap().crash_consent = enabled;
        Ok(())
    }

    async fn command_dedup_get(
        &self,
        principal: &str,
        op_id: &str,
        now_ms: u64,
    ) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        let key = (principal.to_string(), op_id.to_string());
        match inner.command_dedup.get(&key) {
            // Fresh: serve the ORIGINAL result.
            Some((result, at_ms)) if now_ms.saturating_sub(*at_ms) < COMMAND_DEDUP_TTL_MS => {
                Some(result.clone())
            }
            // Expired: evict lazily on access so a re-execution can re-cache.
            Some(_) => {
                inner.command_dedup.remove(&key);
                None
            }
            None => None,
        }
    }

    async fn command_dedup_put(
        &self,
        principal: &str,
        op_id: &str,
        result: Vec<u8>,
        at_ms: u64,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // First-writer-wins: a duplicate key keeps the ORIGINAL (an expired row was already
        // evicted by `command_dedup_get`, so this re-inserts fresh after a TTL re-execution).
        inner
            .command_dedup
            .entry((principal.to_string(), op_id.to_string()))
            .or_insert((result, at_ms));
        Ok(())
    }

    async fn gateway_override(&self) -> Option<(bool, Option<String>)> {
        self.inner.lock().unwrap().gateway_override.clone()
    }

    async fn set_gateway_override(
        &self,
        enabled: bool,
        addr: Option<&str>,
    ) -> Result<(), StoreError> {
        self.inner.lock().unwrap().gateway_override = Some((enabled, addr.map(str::to_string)));
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

    async fn custom_provider_list(&self) -> Vec<CustomProviderRecord> {
        self.inner
            .lock()
            .unwrap()
            .custom_providers
            .values()
            .cloned()
            .collect()
    }

    async fn custom_provider_set(&self, entry: CustomProviderRecord) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .custom_providers
            .insert(entry.id.clone(), entry);
        Ok(())
    }

    async fn custom_provider_remove(&self, id: &str) -> Result<(), StoreError> {
        self.inner.lock().unwrap().custom_providers.remove(id);
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

    async fn saved_presence_list(&self) -> Vec<StoredSavedPresence> {
        self.inner.lock().unwrap().saved_presences.clone()
    }

    async fn saved_presence_set(&self, presence: StoredSavedPresence) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Upsert in place to preserve insertion order (an existing id keeps its position).
        match inner
            .saved_presences
            .iter_mut()
            .find(|p| p.id == presence.id)
        {
            Some(existing) => *existing = presence,
            None => inner.saved_presences.push(presence),
        }
        Ok(())
    }

    async fn saved_presence_remove(&self, id: &str) -> Result<(), StoreError> {
        self.inner
            .lock()
            .unwrap()
            .saved_presences
            .retain(|p| p.id != id);
        Ok(())
    }

    async fn saved_presence_active_get(&self) -> Option<String> {
        self.inner.lock().unwrap().saved_presence_active.clone()
    }

    async fn saved_presence_active_set(&self, id: &str) -> Result<(), StoreError> {
        self.inner.lock().unwrap().saved_presence_active = Some(id.to_string());
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
        fence: Option<FenceToken>,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // Durable path (session-unification §5): the fence rides every append, so a stale
        // incarnation cannot write into the winning turn's segment. Non-durable streams pass
        // `None` (no competing incarnation; the signature is the integrity primitive).
        if let Some(fence) = fence {
            let id = SessionId::new(stream.as_str());
            let rec = inner
                .sessions
                .get(&id)
                .ok_or_else(|| StoreError::NotFound(id.clone()))?;
            Self::check_fence(rec, fence)?;
        }
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

    async fn load_journal_before(
        &self,
        stream: &JournalStreamId,
        before_cursor: u64,
        max: u32,
    ) -> JournalPage {
        let inner = self.inner.lock().unwrap();
        let Some(log) = inner.journal_entries.get(stream) else {
            return JournalPage::default();
        };
        let head_cursor = log.iter().map(|e| e.cursor).max().unwrap_or(0);
        // The backward sibling of `load_journal` (rung 2): the `max` NEWEST entries strictly
        // below the anchor, served ascending (sort, then keep the tail below `before_cursor`).
        let mut entries: Vec<JournalEntry> = log
            .iter()
            .filter(|e| e.cursor < before_cursor)
            .cloned()
            .collect();
        entries.sort_by_key(|e| e.cursor);
        if max > 0 && entries.len() > max as usize {
            entries.drain(..entries.len() - max as usize);
        }
        // The backward continuation: the OLDEST returned cursor, or the anchor when empty.
        let next_cursor = entries.first().map(|e| e.cursor).unwrap_or(before_cursor);
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
            .append_trace(&stream, 0, entry(2, 0x22), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(0, 0x00), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(1, 0x11), None)
            .await
            .unwrap();
        // Redelivered seq is a no-op (append-only, idempotent).
        store
            .append_trace(&stream, 0, entry(1, 0xFF), None)
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
        store
            .append_trace(&stream, 0, entry(0, 1), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(1, 2), None)
            .await
            .unwrap();
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
            .append_trace(&stream, 0, entry(0, 0xA0), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 0, entry(1, 0xA1), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 1, entry(0, 0xB0), None)
            .await
            .unwrap();
        store
            .append_trace(&stream, 1, entry(1, 0xB1), None)
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
        store
            .append_trace(&stream, 0, entry(0, 7), None)
            .await
            .unwrap();
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
            inline_profile: vec![0xAA, 0xBB, 0xCC],
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
}

#[cfg(test)]
mod backward_journal_tests {
    //! The rung-2 backward journal read (`load_journal_before`), proven against both backends:
    //! newest-anchored windows in ascending order, strict upper bound, stable anchoring under
    //! interleaved appends (no skips/dupes across a backward page walk), and non-destructive reads.

    use super::*;

    /// Append `n` entries to `stream` (one segment, seq = 0..n) and return their cursors in
    /// append order (read back through the forward read, so the tests never assume the global
    /// cursor's starting value).
    async fn seed(store: &dyn SessionStore, stream: &JournalStreamId, n: u8) -> Vec<u64> {
        for seq in 0..n {
            store
                .append_trace(
                    stream,
                    0,
                    TraceEntry {
                        seq: seq as u64,
                        bytes: vec![seq],
                        content_hash: ContentHash::new([seq; 32]),
                    },
                    None,
                )
                .await
                .expect("append");
        }
        store
            .load_journal(stream, 0, 0)
            .await
            .entries
            .iter()
            .map(|e| e.cursor)
            .collect()
    }

    async fn backward_window_behaviour(store: &dyn SessionStore) {
        let stream = JournalStreamId::unit(&daemon_common::UnitId::new("bwd-1"));
        let cursors = seed(store, &stream, 7).await;
        assert_eq!(cursors.len(), 7, "seed sanity");
        let head = *cursors.last().unwrap();

        // Latest window in one round-trip: before = u64::MAX anchors at the head.
        let page = store.load_journal_before(&stream, u64::MAX, 3).await;
        let got: Vec<u64> = page.entries.iter().map(|e| e.cursor).collect();
        assert_eq!(got, cursors[4..7], "the 3 newest, ascending");
        assert_eq!(page.head_cursor, head);
        assert_eq!(
            page.next_cursor, cursors[4],
            "next_cursor = the OLDEST returned cursor (the backward continuation)"
        );

        // Continue backward: contiguous, no dupes, no skips.
        let page2 = store
            .load_journal_before(&stream, page.next_cursor, 3)
            .await;
        let got2: Vec<u64> = page2.entries.iter().map(|e| e.cursor).collect();
        assert_eq!(got2, cursors[1..4]);
        let page3 = store
            .load_journal_before(&stream, page2.next_cursor, 3)
            .await;
        let got3: Vec<u64> = page3.entries.iter().map(|e| e.cursor).collect();
        assert_eq!(got3, cursors[0..1], "the final, short page");

        // Past the oldest entry: empty, next_cursor echoes the input anchor.
        let done = store
            .load_journal_before(&stream, page3.next_cursor, 3)
            .await;
        assert!(done.entries.is_empty());
        assert_eq!(done.next_cursor, page3.next_cursor);

        // Strict upper bound: before = smallest cursor excludes it; +1 yields exactly it.
        let below = store.load_journal_before(&stream, cursors[0], 8).await;
        assert!(below.entries.is_empty(), "cursor < before is strict");
        let exactly = store.load_journal_before(&stream, cursors[0] + 1, 8).await;
        assert_eq!(
            exactly.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            cursors[0..1]
        );

        // max == 0 = no cap (mirrors the forward read's store contract).
        let all = store.load_journal_before(&stream, u64::MAX, 0).await;
        assert_eq!(
            all.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            cursors,
            "max 0 returns the whole stream below the anchor, ascending"
        );

        // Non-destructive: a repeat read returns the identical page.
        let again = store.load_journal_before(&stream, u64::MAX, 3).await;
        assert_eq!(again.entries, page.entries);
    }

    /// Writes landing between backward pages never disturb the walk: pages below an already
    /// -served anchor are byte-identical, and the union has no dupes or skips — new records are
    /// picked up by a forward read from the old head, exactly where the client expects them.
    async fn backward_window_stable_under_interleaved_appends(store: &dyn SessionStore) {
        let stream = JournalStreamId::unit(&daemon_common::UnitId::new("bwd-2"));
        let cursors = seed(store, &stream, 5).await;
        let head = *cursors.last().unwrap();

        // First backward page (newest 2), anchoring the walk.
        let first = store.load_journal_before(&stream, u64::MAX, 2).await;
        assert_eq!(
            first.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            cursors[3..5]
        );

        // Two new records land mid-walk.
        for seq in 5..7u8 {
            store
                .append_trace(
                    &stream,
                    0,
                    TraceEntry {
                        seq: seq as u64,
                        bytes: vec![seq],
                        content_hash: ContentHash::new([seq; 32]),
                    },
                    None,
                )
                .await
                .expect("append");
        }

        // The continuation below the anchor is untouched by the appends.
        let second = store
            .load_journal_before(&stream, first.next_cursor, 2)
            .await;
        assert_eq!(
            second.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            cursors[1..3],
            "pages below a served anchor must not shift under appends"
        );
        let third = store
            .load_journal_before(&stream, second.next_cursor, 2)
            .await;
        assert_eq!(
            third.entries.iter().map(|e| e.cursor).collect::<Vec<_>>(),
            cursors[0..1]
        );

        // No dupes / no skips across the whole walk; the interleaved records sit above the old
        // head, served by the forward read from it.
        let forward = store.load_journal(&stream, head, 0).await;
        assert_eq!(forward.entries.len(), 2, "the two interleaved appends");
        let mut union: Vec<u64> = first
            .entries
            .iter()
            .chain(&second.entries)
            .chain(&third.entries)
            .map(|e| e.cursor)
            .collect();
        union.sort_unstable();
        union.dedup();
        assert_eq!(union, cursors, "backward union = the pre-append stream");
    }

    #[tokio::test]
    async fn in_memory_backward_windows_are_anchored_and_ordered() {
        backward_window_behaviour(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_backward_windows_are_anchored_and_ordered() {
        backward_window_behaviour(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_backward_windows_survive_interleaved_appends() {
        backward_window_stable_under_interleaved_appends(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_backward_windows_survive_interleaved_appends() {
        backward_window_stable_under_interleaved_appends(&SqliteStore::open_in_memory().unwrap())
            .await;
    }
}

#[cfg(test)]
mod inbox_splice_tests {
    //! The durable typed inbox (session-unification §4), proven against both backends: the
    //! append-or-return-existing dedupe on `(session_id, origin_op)`, the atomic `Idle → Ready`
    //! flip, the fenced claim CAS (idempotent re-claim under one fence, exactly-once reclaim by a
    //! newer fence, `Fenced` for a stale one), transactional consumption at the commit ops, and
    //! retention pruning that touches only consumed rows.

    use super::*;

    const PARTITION: PartitionId = PartitionId(0);

    fn splice(session: &SessionId, kind: SpliceKind, payload: &[u8], op: &str) -> NewSplice {
        NewSplice {
            session_id: session.clone(),
            kind,
            payload: payload.to_vec(),
            origin_op: op.into(),
            origin: "test".into(),
        }
    }

    /// Append: NotFound on an absent session; monotonic per-session sequences; a duplicate
    /// `origin_op` returns the ORIGINAL seq without inserting; the append flips an `Idle`
    /// session `Ready` in the same transaction and leaves other statuses alone.
    async fn append_dedupe_and_idle_flip(store: &dyn SessionStore) {
        let a = SessionId::new("splice-a");
        assert!(
            matches!(
                store
                    .append_splice(splice(&a, SpliceKind::StartTurn, b"x", "op-0"))
                    .await,
                Err(StoreError::NotFound(_))
            ),
            "append to an absent session is NotFound (no orphan inbox rows)"
        );

        store
            .create_idle(a.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        assert_eq!(store.status(&a).await, Some(SessionStatus::Idle));

        let s1 = store
            .append_splice(splice(&a, SpliceKind::StartTurn, b"first", "op-1"))
            .await
            .unwrap();
        assert_eq!(s1, 1);
        assert_eq!(
            store.status(&a).await,
            Some(SessionStatus::Ready),
            "durable input flips Idle -> Ready atomically with the append (§2 status rule)"
        );

        // A producer retry (crash-before-ack) returns the original seq; nothing is duplicated.
        let retry = store
            .append_splice(splice(&a, SpliceKind::StartTurn, b"first", "op-1"))
            .await
            .unwrap();
        assert_eq!(retry, s1, "duplicate origin_op returns the original seq");

        let s2 = store
            .append_splice(splice(&a, SpliceKind::Observe, b"ctx", "op-2"))
            .await
            .unwrap();
        assert_eq!(s2, 2, "sequences are per-session monotonic");

        let rows = store.splices_after(&a, 0).await;
        assert_eq!(
            rows.iter().map(|r| r.splice_seq).collect::<Vec<_>>(),
            vec![1, 2],
            "splices_after returns unconsumed rows in sequence order, no duplicates"
        );
        assert_eq!(rows[0].kind, SpliceKind::StartTurn);
        assert_eq!(rows[0].payload, b"first");
        assert_eq!(rows[1].kind, SpliceKind::Observe);

        // Sessions are isolated.
        let b = SessionId::new("splice-b");
        store
            .create_idle(b.clone(), PARTITION, SnapshotBlob::new(vec![2]))
            .await
            .unwrap();
        // The same origin_op namespace on ANOTHER session dedupes independently.
        let sb = store
            .append_splice(splice(&b, SpliceKind::Steer, b"other", "op-1"))
            .await
            .unwrap();
        assert_eq!(sb, 1, "the dedupe key is scoped per session");
        assert_eq!(store.splices_after(&a, 0).await.len(), 2);
        assert_eq!(store.splices_after(&b, 0).await.len(), 1);
    }

    /// Claim CAS: a claim takes every pending row under the current fence; re-claiming under the
    /// SAME fence is idempotent (returns the held set); a STALE fence gets `Fenced`; a NEWER
    /// fence reclaims un-consumed claims exactly once (the crash-recovery path).
    async fn claim_cas_fencing(store: &dyn SessionStore) {
        let id = SessionId::new("splice-claim");
        store
            .create_session(id.clone(), PARTITION, SnapshotBlob::new(vec![0]))
            .await
            .unwrap();
        store
            .append_splice(splice(&id, SpliceKind::StartTurn, b"one", "c-1"))
            .await
            .unwrap();
        store
            .append_splice(splice(&id, SpliceKind::Steer, b"two", "c-2"))
            .await
            .unwrap();

        let f1 = store.acquire_activation_lease(&id).await.unwrap();
        let claimed = store.claim_splices(&id, f1).await.unwrap();
        assert_eq!(claimed.len(), 2);
        assert!(claimed
            .iter()
            .all(|r| r.claim == SpliceClaim::Claimed { fence: f1.0 }));

        // Idempotent under the same fence: a re-load sees the same set.
        let again = store.claim_splices(&id, f1).await.unwrap();
        assert_eq!(
            again.iter().map(|r| r.splice_seq).collect::<Vec<_>>(),
            claimed.iter().map(|r| r.splice_seq).collect::<Vec<_>>(),
        );

        // The next lease fences the old one out and reclaims the un-consumed rows exactly once.
        let f2 = store.acquire_activation_lease(&id).await.unwrap();
        assert!(
            matches!(
                store.claim_splices(&id, f1).await,
                Err(StoreError::Fenced { .. })
            ),
            "a stale fence cannot claim"
        );
        let reclaimed = store.claim_splices(&id, f2).await.unwrap();
        assert_eq!(
            reclaimed.len(),
            2,
            "a newer fence reclaims the stale claims"
        );
        assert!(reclaimed
            .iter()
            .all(|r| r.claim == SpliceClaim::Claimed { fence: f2.0 }));
    }

    /// Consumption rides the commit transaction: `load_for_activation` claims in the load,
    /// the checkpoint's `consumed_splices` cursor flips rows at/below it to `Consumed`, replay
    /// (`splices_after`) skips them, and retention pruning deletes ONLY consumed rows.
    async fn consume_at_commit_and_prune(store: &dyn SessionStore) {
        let id = SessionId::new("splice-consume");
        store
            .create_session(id.clone(), PARTITION, SnapshotBlob::new(vec![0]))
            .await
            .unwrap();
        for (i, op) in ["k-1", "k-2", "k-3"].iter().enumerate() {
            store
                .append_splice(splice(&id, SpliceKind::StartTurn, &[i as u8], op))
                .await
                .unwrap();
        }

        let fence = store.acquire_activation_lease(&id).await.unwrap();
        let activation = store.load_for_activation(&id, fence).await.unwrap();
        assert_eq!(
            activation
                .splices
                .iter()
                .map(|r| r.splice_seq)
                .collect::<Vec<_>>(),
            vec![1, 2, 3],
            "load_for_activation claims the pending inbox in the load transaction"
        );

        // The turn folded splices 1-2; the commit consumes them transactionally.
        store
            .mark_completed(
                Checkpoint::new(id.clone(), Epoch(1), SnapshotBlob::new(vec![9]))
                    .with_consumed_splices(Some(2)),
                fence,
            )
            .await
            .unwrap();
        let rest = store.splices_after(&id, 0).await;
        assert_eq!(
            rest.iter().map(|r| r.splice_seq).collect::<Vec<_>>(),
            vec![3],
            "replay skips consumed rows; the un-folded splice survives"
        );

        // Retention: a future cutoff prunes exactly the consumed rows, never the pending one.
        let pruned = store.prune_consumed_splices(now_ms() + 1).await;
        assert_eq!(pruned, 2, "prune deletes only consumed rows");
        assert_eq!(
            store
                .splices_after(&id, 0)
                .await
                .iter()
                .map(|r| r.splice_seq)
                .collect::<Vec<_>>(),
            vec![3],
            "unconsumed splices are NEVER pruned"
        );
        // Sequences never renumber: the next append continues past the pruned rows.
        let next = store
            .append_splice(splice(&id, SpliceKind::Steer, b"after-prune", "k-4"))
            .await
            .unwrap();
        assert_eq!(next, 4, "splice_seq is never reused after pruning");
    }

    #[tokio::test]
    async fn in_memory_append_dedupe_and_idle_flip() {
        append_dedupe_and_idle_flip(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_append_dedupe_and_idle_flip() {
        append_dedupe_and_idle_flip(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_claim_cas_fencing() {
        claim_cas_fencing(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_claim_cas_fencing() {
        claim_cas_fencing(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_consume_at_commit_and_prune() {
        consume_at_commit_and_prune(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_consume_at_commit_and_prune() {
        consume_at_commit_and_prune(&SqliteStore::open_in_memory().unwrap()).await;
    }
}

#[cfg(test)]
mod turn_commit_tests {
    //! The fenced durable turn boundary (session-unification §5), proven against both backends:
    //! one `commit_turn` transaction carries the snapshot, the monotonic `turn_seq` advance, the
    //! consumed-splice flip stamped with the committed turn's identity, the turn's journal-root
    //! seal, and the Idle-iff-no-work-else-Ready status selection (+ self-wake). Plus: stale
    //! fences are rejected, a seal for the wrong segment is an incarnation bug, terminal
    //! `mark_completed` advances the same counter, and fenced `append_trace` refuses a stale
    //! incarnation's writes.

    use super::*;

    const PARTITION: PartitionId = PartitionId(0);

    fn splice(session: &SessionId, payload: &[u8], op: &str) -> NewSplice {
        NewSplice {
            session_id: session.clone(),
            kind: SpliceKind::Steer,
            payload: payload.to_vec(),
            origin_op: op.into(),
            origin: "test".into(),
        }
    }

    /// Drain the wake outbox into a list (order-preserving) so tests can assert the self-wake.
    async fn drain_wakes(store: &dyn SessionStore) -> Vec<SessionId> {
        let mut wakes = Vec::new();
        while let Some(id) = store.dequeue_wake().await {
            wakes.push(id);
        }
        wakes
    }

    /// A quiet turn commits to `Idle` (no self-wake); a commit with a raced-in splice lands
    /// `Ready` WITH a self-wake; `turn_seq` advances monotonically through both; and the
    /// consumed splices carry the committed turn's identity, not the epoch.
    async fn commit_turn_idle_vs_ready(store: &dyn SessionStore) {
        let id = SessionId::new("turn-a");
        store
            .create_idle(id.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        store
            .append_splice(splice(&id, b"start", "op-1"))
            .await
            .unwrap();
        drain_wakes(store).await;

        // Turn 0: load claims the splice; the commit consumes it and finds no remaining work.
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        let activation = store.load_for_activation(&id, fence).await.unwrap();
        assert_eq!(activation.turn_seq, 0, "the first turn's identity is 0");
        assert_eq!(activation.splices.len(), 1);
        let commit = store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![2]))
                    .with_consumed_splices(Some(1)),
                None,
                fence,
            )
            .await
            .unwrap();
        assert_eq!(commit.turn_seq, 0, "the commit reports the committed turn");
        assert_eq!(
            commit.status,
            SessionStatus::Idle,
            "no unconsumed work -> Idle"
        );
        assert_eq!(store.status(&id).await, Some(SessionStatus::Idle));
        assert!(
            drain_wakes(store).await.is_empty(),
            "an Idle commit publishes no self-wake"
        );
        let consumed = store.splices_after(&id, 0).await;
        assert!(
            consumed.is_empty(),
            "the folded splice is consumed, not re-deliverable"
        );

        // Turn 1: a splice races in mid-turn (after load, before commit) and is NOT folded.
        store
            .append_splice(splice(&id, b"turn-two", "op-2"))
            .await
            .unwrap();
        drain_wakes(store).await;
        let fence2 = store.acquire_activation_lease(&id).await.unwrap();
        let activation2 = store.load_for_activation(&id, fence2).await.unwrap();
        assert_eq!(
            activation2.turn_seq, 1,
            "the next turn's identity advanced past the committed one"
        );
        store
            .append_splice(splice(&id, b"raced-in", "op-3"))
            .await
            .unwrap();
        drain_wakes(store).await;
        let commit2 = store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![3]))
                    .with_consumed_splices(Some(2)),
                None,
                fence2,
            )
            .await
            .unwrap();
        assert_eq!(commit2.turn_seq, 1);
        assert_eq!(
            commit2.status,
            SessionStatus::Ready,
            "unconsumed raced-in work -> Ready, never stranded on Idle"
        );
        assert_eq!(
            drain_wakes(store).await,
            vec![id.clone()],
            "the Ready commit publishes exactly one self-wake"
        );
        let remaining = store.splices_after(&id, 0).await;
        assert_eq!(
            remaining.iter().map(|r| r.splice_seq).collect::<Vec<_>>(),
            vec![3],
            "only the raced-in splice remains deliverable"
        );
    }

    /// A stale fence cannot commit a turn; the winning fence still can. And the seal's segment
    /// must equal the in-flight turn — a mismatch is an incarnation bug that fails the commit
    /// without writing anything.
    async fn commit_turn_fencing_and_seal_guard(store: &dyn SessionStore) {
        let id = SessionId::new("turn-b");
        store
            .create_idle(id.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        let stale = store.acquire_activation_lease(&id).await.unwrap();
        let winner = store.acquire_activation_lease(&id).await.unwrap();
        assert!(
            matches!(
                store
                    .commit_turn(
                        Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![9])),
                        None,
                        stale,
                    )
                    .await,
                Err(StoreError::Fenced { .. })
            ),
            "a stale incarnation cannot commit a turn"
        );

        // A seal for the wrong segment is refused (nothing written: turn_seq unmoved).
        let bad_seal = TurnSeal {
            segment: 7,
            root: MerkleRoot::new([1; 32]),
            signature: vec![1],
        };
        assert!(
            store
                .commit_turn(
                    Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![2])),
                    Some(bad_seal),
                    winner,
                )
                .await
                .is_err(),
            "a seal segment != the in-flight turn is an incarnation bug"
        );
        let activation = store.load_for_activation(&id, winner).await.unwrap();
        assert_eq!(activation.turn_seq, 0, "the failed commit wrote nothing");

        // The correct seal commits atomically with the turn: the root is readable afterwards.
        let stream = JournalStreamId::session(&id);
        store
            .append_trace(
                &stream,
                0,
                TraceEntry {
                    seq: 0,
                    bytes: vec![0xAB],
                    content_hash: ContentHash::new([3; 32]),
                },
                Some(winner),
            )
            .await
            .unwrap();
        let commit = store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![3])),
                Some(TurnSeal {
                    segment: 0,
                    root: MerkleRoot::new([4; 32]),
                    signature: vec![9],
                }),
                winner,
            )
            .await
            .unwrap();
        assert_eq!(commit.turn_seq, 0);
        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        assert_eq!(
            seg.committed
                .expect("sealed in the commit transaction")
                .root,
            MerkleRoot::new([4; 32])
        );
    }

    /// The fence rides every durable journal append (§5): a stale incarnation's `append_trace`
    /// is refused, so it can never write into the winning turn's segment.
    async fn append_trace_is_fenced(store: &dyn SessionStore) {
        let id = SessionId::new("turn-c");
        store
            .create_idle(id.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        let stale = store.acquire_activation_lease(&id).await.unwrap();
        let winner = store.acquire_activation_lease(&id).await.unwrap();
        let stream = JournalStreamId::session(&id);
        let entry = |seq: u64, byte: u8| TraceEntry {
            seq,
            bytes: vec![byte],
            content_hash: ContentHash::new([byte; 32]),
        };
        assert!(
            matches!(
                store
                    .append_trace(&stream, 0, entry(0, 0x01), Some(stale))
                    .await,
                Err(StoreError::Fenced { .. })
            ),
            "a stale incarnation cannot append into the winning segment"
        );
        store
            .append_trace(&stream, 0, entry(0, 0x02), Some(winner))
            .await
            .unwrap();
        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        assert_eq!(seg.entries.len(), 1);
        assert_eq!(seg.entries[0].bytes, vec![0x02], "only the winner's entry");
    }

    /// A terminal `mark_completed` IS that turn's boundary: it advances the same `turn_seq`
    /// counter and stamps its consumed splices with the committed turn's identity.
    async fn terminal_commit_advances_turn_seq(store: &dyn SessionStore) {
        let id = SessionId::new("turn-d");
        store
            .create_idle(id.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        store
            .append_splice(splice(&id, b"only", "op-1"))
            .await
            .unwrap();
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        let activation = store.load_for_activation(&id, fence).await.unwrap();
        assert_eq!(activation.turn_seq, 0);

        // Commit turn 0 non-terminally, then complete on turn 1.
        store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![2]))
                    .with_consumed_splices(Some(1)),
                None,
                fence,
            )
            .await
            .unwrap();
        let fence2 = store.acquire_activation_lease(&id).await.unwrap();
        let activation2 = store.load_for_activation(&id, fence2).await.unwrap();
        assert_eq!(activation2.turn_seq, 1);
        store
            .mark_completed(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![3])),
                fence2,
            )
            .await
            .unwrap();
        assert_eq!(store.status(&id).await, Some(SessionStatus::Completed));
        // The counter advanced past the terminal turn too (visible to any post-mortem reader).
        let fence3 = store.acquire_activation_lease(&id).await.unwrap();
        let post = store.load_for_activation(&id, fence3).await.unwrap();
        assert_eq!(
            post.turn_seq, 2,
            "the terminal commit advanced the committed-turn counter"
        );
    }

    /// The completion sibling of splice consumption: a turn commit that lists the completions it
    /// folded (`Checkpoint::applied_completions`) deletes exactly those inbox rows, so the session
    /// lands `Idle` and the next load re-delivers nothing. A commit that does NOT list a delivered
    /// completion (the raced-in shape) leaves it durable and lands `Ready` + self-wake — the
    /// livelock regression: without deletion, one applied completion kept every future turn commit
    /// on `Ready`, self-waking a policy-committing session in a hot loop forever.
    async fn commit_turn_consumes_applied_completions(store: &dyn SessionStore) {
        let id = SessionId::new("turn-e");
        store
            .create_idle(id.clone(), PARTITION, SnapshotBlob::new(vec![1]))
            .await
            .unwrap();
        let completion = JobCompletion {
            session_id: id.clone(),
            epoch: Epoch(0),
            job_id: JobId::new("job-1"),
            payload: b"child done".to_vec(),
        };
        store.record_completion_and_wake(&completion).await.unwrap();
        drain_wakes(store).await;

        // Turn 0 folds the completion but does NOT list it as applied (a legacy/raced-in shape):
        // the row survives, the session stays Ready and self-wakes.
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        let activation = store.load_for_activation(&id, fence).await.unwrap();
        assert_eq!(
            activation.unapplied.len(),
            1,
            "the load delivers the completion"
        );
        let commit = store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![2])),
                None,
                fence,
            )
            .await
            .unwrap();
        assert_eq!(
            commit.status,
            SessionStatus::Ready,
            "an unlisted completion still counts as pending work"
        );
        assert_eq!(drain_wakes(store).await, vec![id.clone()]);

        // Turn 1 lists it: the row is deleted in the same transaction, the session lands Idle
        // with no self-wake, and the next load delivers nothing (no livelock, no re-fold).
        let fence2 = store.acquire_activation_lease(&id).await.unwrap();
        let activation2 = store.load_for_activation(&id, fence2).await.unwrap();
        assert_eq!(
            activation2.unapplied.len(),
            1,
            "still delivered until consumed"
        );
        let commit2 = store
            .commit_turn(
                Checkpoint::new(id.clone(), Epoch(0), SnapshotBlob::new(vec![3]))
                    .with_applied_completions(
                        activation2
                            .unapplied
                            .iter()
                            .map(|c| (c.epoch, c.job_id.clone()))
                            .collect(),
                    ),
                None,
                fence2,
            )
            .await
            .unwrap();
        assert_eq!(
            commit2.status,
            SessionStatus::Idle,
            "consuming the folded completion leaves no pending work"
        );
        assert!(drain_wakes(store).await.is_empty(), "no self-wake on Idle");
        let fence3 = store.acquire_activation_lease(&id).await.unwrap();
        let activation3 = store.load_for_activation(&id, fence3).await.unwrap();
        assert!(
            activation3.unapplied.is_empty(),
            "a consumed completion is never re-delivered"
        );
    }

    #[tokio::test]
    async fn in_memory_commit_turn_idle_vs_ready() {
        commit_turn_idle_vs_ready(&InMemoryStore::new()).await;
    }

    #[tokio::test]
    async fn in_memory_commit_turn_consumes_applied_completions() {
        commit_turn_consumes_applied_completions(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_commit_turn_consumes_applied_completions() {
        commit_turn_consumes_applied_completions(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_commit_turn_idle_vs_ready() {
        commit_turn_idle_vs_ready(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_commit_turn_fencing_and_seal_guard() {
        commit_turn_fencing_and_seal_guard(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_commit_turn_fencing_and_seal_guard() {
        commit_turn_fencing_and_seal_guard(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_append_trace_is_fenced() {
        append_trace_is_fenced(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_append_trace_is_fenced() {
        append_trace_is_fenced(&SqliteStore::open_in_memory().unwrap()).await;
    }

    #[tokio::test]
    async fn in_memory_terminal_commit_advances_turn_seq() {
        terminal_commit_advances_turn_seq(&InMemoryStore::new()).await;
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_terminal_commit_advances_turn_seq() {
        terminal_commit_advances_turn_seq(&SqliteStore::open_in_memory().unwrap()).await;
    }
}

#[cfg(test)]
mod detached_delegation_tests {
    //! The detached-delegation (`spawn wait:false`) store seam, proven against both backends:
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

        store
            .bind_completion_notice(&child, &parent, Some("call-provenance".into()))
            .await
            .unwrap();
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

        // Exactly one notice, carrying the structured payload, addressed parent<-child, with the
        // spawn-time call_id provenance (wire v29) surviving the edge -> outbox round-trip.
        let notice = store
            .dequeue_completion_notice()
            .await
            .expect("one completion notice");
        assert_eq!(notice.parent, parent);
        assert_eq!(notice.child, child);
        assert_eq!(notice.payload, b"did the thing".to_vec());
        assert_eq!(
            notice.call_id.as_deref(),
            Some("call-provenance"),
            "the spawning tool call_id rides the notice"
        );
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
