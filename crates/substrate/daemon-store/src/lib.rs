//! `daemon-store` — durable persistence primitives for the activation core.
//!
//! The [`SessionStore`] trait is the *sole authority* for durable session state (lifecycle §4
//! invariant #1): snapshots, the completion inbox (idempotent via `UNIQUE(session_id, epoch,
//! job_id)`), the wake/job outboxes, and the monotonic activation lease that fences stale
//! incarnations. Phase 1 ships the in-memory backend ([`InMemoryStore`]); the `sqlite` feature is a
//! deferred stub. Depends only on `daemon-common`.
//!
//! Snapshots are handled here only as opaque CBOR [`SnapshotBlob`]s — the typed `Snapshot` lives in
//! `daemon-protocol`, keeping this crate protocol-free (lifecycle §2; layout §3 DAG).
//!
//! See `docs/specs/daemon-lifecycle-persistence.md`.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::{DaemonError, Epoch, FenceToken, JobId, PartitionId, SessionId, SnapshotBlob};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

/// The durable status of a session record (lifecycle §5).
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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

    /// Scan for sessions in a resumable (`Ready`/`Active`) state for the recovery scanner
    /// (lifecycle §5; invariant #7).
    async fn scan_resumable(&self, partition: PartitionId) -> Result<Vec<SessionId>, StoreError>;

    /// Pop the next pending durable job, if any (job-outbox dispatcher / worker side).
    async fn dequeue_job(&self) -> Option<JobCommand>;

    /// Pop the next pending durable wake hint, if any (wake-outbox dispatcher).
    async fn dequeue_wake(&self) -> Option<SessionId>;

    /// Read the current durable status of a session (test/observability helper).
    async fn status(&self, id: &SessionId) -> Option<SessionStatus>;
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
    fault: Option<FaultPoint>,
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
        Ok(())
    }

    async fn record_completion_and_wake(&self, c: &JobCompletion) -> Result<(), StoreError> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.sessions.contains_key(&c.session_id) {
            return Err(StoreError::NotFound(c.session_id.clone()));
        }
        let key = (c.session_id.clone(), c.epoch, c.job_id.clone());
        // Idempotent: a redelivered completion is a no-op (invariant #2/#3).
        if !inner.inbox_keys.insert(key) {
            return Ok(());
        }
        inner
            .unapplied
            .entry(c.session_id.clone())
            .or_default()
            .push(c.clone());
        if let Some(rec) = inner.sessions.get_mut(&c.session_id) {
            rec.status = SessionStatus::Ready;
        }
        // Boundary: completion durable + session Ready; crash before publishing the wake.
        // Recovery scan must still re-activate the Ready session (invariant #7).
        Self::take_fault(&mut inner, FaultPoint::BeforeWakePublish)?;
        inner.wake_outbox.push_back(c.session_id.clone());
        Ok(())
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

    async fn status(&self, id: &SessionId) -> Option<SessionStatus> {
        self.inner
            .lock()
            .unwrap()
            .sessions
            .get(id)
            .map(|r| r.status.clone())
    }
}
