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

use async_trait::async_trait;
use daemon_common::{
    ContentHash, DaemonError, Epoch, FenceToken, JobId, JournalStreamId, MerkleRoot, PartitionId,
    SessionId, SnapshotBlob, UsageDelta,
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

/// Build a highlighted excerpt of `body` around the first occurrence of `needle` (lowercased), with
/// the match wrapped in `[`…`]` and `…` elision — the in-memory analogue of SQLite FTS5 `snippet()`.
fn snippet_around(body: &str, needle: &str) -> String {
    let lower = body.to_lowercase();
    let Some(pos) = lower.find(needle) else {
        return body.chars().take(64).collect();
    };
    const PAD: usize = 24;
    let start = body[..pos].char_indices().rev().nth(PAD).map(|(i, _)| i).unwrap_or(0);
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
    fault: Option<FaultPoint>,
    /// Append-only journal entries per stream, in append (cursor) order across all segments.
    journal_entries: HashMap<JournalStreamId, Vec<JournalEntry>>,
    /// Sealed segment roots per `(stream, segment)`.
    journal_roots: HashMap<(JournalStreamId, u64), CommittedRoot>,
    /// Stream-monotonic cursor allocator (the pagination key for `load_journal`).
    journal_cursor: u64,
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
        // If this session was delegated by a parent, fulfill that parent's job and wake it in the
        // *same* transaction (under the held lock). The binding is durable, so this is recovery-safe:
        // a child marked terminal always wakes its delegator, at any nesting depth.
        if let Some(job) = inner.delegations.get(&checkpoint.session_id).cloned() {
            let completion = JobCompletion {
                session_id: job.session_id.clone(),
                epoch: job.epoch,
                job_id: job.job_id.clone(),
                payload: format!("child:{}", checkpoint.session_id).into_bytes(),
            };
            if inner.sessions.contains_key(&completion.session_id)
                && Self::apply_completion_locked(&mut inner, &completion)
            {
                inner.wake_outbox.push_back(completion.session_id);
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

    async fn record_child_edge(
        &self,
        parent: SessionId,
        child: SessionId,
        work_label: String,
    ) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        // The reverse index drives the tree projection (audit), but we deliberately do *not* write a
        // `delegations` entry: `mark_completed` finds no job, so the child self-closes (no wake).
        inner.child_index.entry(parent).or_default().push(child.clone());
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
