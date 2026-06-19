//! `daemon-activation` — the durable activation / virtual-entity core.
//!
//! The correctness-critical layer with no upstream reference implementation (the build-first
//! milestone). It owns the active-only directory, the monotonic lease/fence, the wake/job-outbox
//! dispatchers, the completion consumer, and the recovery scanner — proving lifecycle §4 invariants
//! #1, #5, #6, #7, #8. It drives engines through a protocol-agnostic seam ([`Incarnation`] /
//! [`EngineFactory`]) so the durable core can be exercised by `daemon-stub-engine` with no
//! dependency on `daemon-core` or `daemon-host`. Depends on `daemon-store` + `daemon-common`.
//!
//! The `elfo` feature (off by default) is reserved for an optional elfo-backed mailbox experiment.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::{DaemonError, Epoch, FenceToken, PartitionId, SessionId};
use daemon_store::{
    Checkpoint, JobCommand, JobCompletion, SessionStatus, SessionStore, StoreError,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio_util::task::TaskTracker;

// Re-export so downstream crates need only depend on `daemon-activation` for the seam.
pub use daemon_common::SnapshotBlob;

/// The outcome of running one activation of an engine incarnation.
pub enum Step {
    /// The engine reached a terminal state this activation.
    Completed,
    /// The engine suspended at a phase boundary, delegating background work.
    Suspended {
        /// The durable job to enqueue on the outbox.
        job: JobCommand,
    },
}

/// Errors raised by an engine incarnation through the seam.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// A generic engine failure.
    #[error("engine: {0}")]
    Other(String),
    /// Wraps the shared base error (e.g. snapshot codec failures).
    #[error(transparent)]
    Common(#[from] DaemonError),
}

/// One live engine incarnation, driven by the activation layer (the phase-1 stand-in for the
/// `daemon-host` session task). Protocol-agnostic: it deals in opaque [`SnapshotBlob`]s and the
/// durable job/completion types, never §17 messages directly.
#[async_trait]
pub trait Incarnation: Send {
    /// Reconstruct from the last snapshot and apply unapplied completions idempotently *before*
    /// running new work (lifecycle §3.1, invariant #2).
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<JobCompletion>,
    ) -> Result<(), EngineError>;

    /// Process available work, returning whether the engine completed or suspended.
    async fn run(&mut self) -> Result<Step, EngineError>;

    /// Produce the snapshot to persist at the current phase boundary.
    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError>;

    /// The current incarnation epoch (post-bump at suspension).
    fn epoch(&self) -> Epoch;
}

/// Constructs fresh [`Incarnation`]s for the activation layer to hydrate.
pub trait EngineFactory: Send + Sync {
    /// Create a new, un-hydrated incarnation.
    fn create(&self) -> Box<dyn Incarnation>;
}

/// A message deliverable to an activation (phase-1 minimal surface).
#[derive(Clone, Debug)]
pub enum SessionMsg {
    /// A wake hint: ensure the session is activated (the store is authoritative; this is only a
    /// hint — lifecycle §4 invariant #1).
    Wake,
}

/// Errors raised by the activation substrate.
#[derive(Debug, thiserror::Error)]
pub enum SubErr {
    /// A durable store operation failed (including [`StoreError::Fenced`] for stale incarnations).
    #[error(transparent)]
    Store(#[from] StoreError),
    /// The engine seam failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The session already has a live incarnation in this process (single-activation guard).
    #[error("busy: {0}")]
    Busy(SessionId),
    /// The activation task could not be joined.
    #[error("activation task join error: {0}")]
    Join(String),
}

/// The plain-Tokio durable activation substrate (host-spec §3).
#[async_trait]
pub trait ActivationSubstrate: Send + Sync {
    /// Ensure exactly one live, hydrated incarnation for `id`, under the given fencing token.
    async fn activate(&self, id: SessionId, fence: FenceToken) -> Result<(), SubErr>;
    /// Drop the in-memory incarnation (durability already committed).
    async fn passivate(&self, id: &SessionId);
    /// Deliver a message to the active incarnation (activating it if absent).
    async fn deliver(&self, id: &SessionId, msg: SessionMsg) -> Result<(), SubErr>;
}

struct ManagerInner {
    store: Arc<dyn SessionStore>,
    factory: Arc<dyn EngineFactory>,
    partition: PartitionId,
    /// The active-only directory: currently running sessions. Returns to baseline after passivation
    /// (invariant #8) — this is what the churn acceptance test asserts.
    directory: DashMap<SessionId, ()>,
    /// Tracks live activation tasks so their memory is released on completion (invariant #8).
    tracker: TaskTracker,
}

impl ManagerInner {
    async fn run_cycle(&self, id: &SessionId, fence: FenceToken) -> Result<(), SubErr> {
        let activation = self.store.load_for_activation(id, fence).await?;
        let mut inc = self.factory.create();
        inc.hydrate(activation.snapshot, activation.unapplied).await?;
        match inc.run().await? {
            Step::Suspended { job } => {
                let snapshot = inc.checkpoint()?;
                let checkpoint = Checkpoint {
                    session_id: id.clone(),
                    epoch: inc.epoch(),
                    snapshot,
                };
                self.store
                    .checkpoint_and_enqueue(checkpoint, job, fence)
                    .await?;
            }
            Step::Completed => {
                let snapshot = inc.checkpoint()?;
                let checkpoint = Checkpoint {
                    session_id: id.clone(),
                    epoch: inc.epoch(),
                    snapshot,
                };
                self.store.mark_completed(checkpoint, fence).await?;
            }
        }
        Ok(())
    }
}

/// The plain-Tokio [`ActivationSubstrate`] implementation and the home of the resident dispatchers,
/// completion consumer, and recovery scanner.
#[derive(Clone)]
pub struct ActivationManager {
    inner: Arc<ManagerInner>,
}

impl ActivationManager {
    /// Construct a manager over a shared store and engine factory, owning `partition`.
    pub fn new(
        store: Arc<dyn SessionStore>,
        factory: Arc<dyn EngineFactory>,
        partition: PartitionId,
    ) -> Self {
        Self {
            inner: Arc::new(ManagerInner {
                store,
                factory,
                partition,
                directory: DashMap::new(),
                tracker: TaskTracker::new(),
            }),
        }
    }

    /// The number of currently active incarnations in this process (acceptance test #1 baseline).
    pub fn active_count(&self) -> usize {
        self.inner.directory.len()
    }

    /// Acquire a fresh lease and activate `id`, guarding against re-entry and completed sessions.
    /// This is the convenience wake path; the store remains authoritative (invariant #1).
    pub async fn wake(&self, id: SessionId) -> Result<(), SubErr> {
        match self.inner.store.status(&id).await {
            Some(SessionStatus::Completed) | None => return Ok(()),
            _ => {}
        }
        let fence = self.inner.store.acquire_activation_lease(&id).await?;
        match self.activate(id, fence).await {
            Err(SubErr::Busy(_)) => Ok(()),
            other => other,
        }
    }

    /// Drain the durable job outbox, producing a completion per job (the worker side). Completions
    /// are recorded idempotently and a wake enqueued (lifecycle §3.1, §5).
    pub async fn run_workers(&self) -> Result<usize, SubErr> {
        let mut processed = 0usize;
        while let Some(job) = self.inner.store.dequeue_job().await {
            let completion = JobCompletion {
                session_id: job.session_id.clone(),
                epoch: job.epoch,
                job_id: job.job_id.clone(),
                payload: job.payload.clone(),
            };
            self.inner
                .store
                .record_completion_and_wake(&completion)
                .await?;
            processed += 1;
        }
        Ok(processed)
    }

    /// Drain the durable wake outbox, activating each hinted session.
    pub async fn dispatch_wakes(&self) -> Result<usize, SubErr> {
        let mut dispatched = 0usize;
        while let Some(id) = self.inner.store.dequeue_wake().await {
            self.wake(id).await?;
            dispatched += 1;
        }
        Ok(dispatched)
    }

    /// The recovery scanner: rebuild from the store alone (in-memory directories are gone).
    /// Drains durable work, dispatches pending wakes, then re-activates any session left in a
    /// resumable state whose wake never arrived (lifecycle §3.1; invariants #5, #7). Loops until
    /// the world is quiescent so a multi-step cycle (suspend -> work -> resume) fully drains.
    pub async fn recover(&self) -> Result<(), SubErr> {
        loop {
            let jobs = self.run_workers().await?;
            let wakes = self.dispatch_wakes().await?;
            let mut scanned = 0usize;
            for id in self.inner.store.scan_resumable(self.inner.partition).await? {
                self.wake(id).await?;
                scanned += 1;
            }
            if jobs == 0 && wakes == 0 && scanned == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Gracefully close the task tracker and wait for in-flight activations to drain.
    pub async fn shutdown(&self) {
        self.inner.tracker.close();
        self.inner.tracker.wait().await;
    }
}

#[async_trait]
impl ActivationSubstrate for ActivationManager {
    async fn activate(&self, id: SessionId, fence: FenceToken) -> Result<(), SubErr> {
        // Single-activation guard for this process (invariant #6). Cluster-wide single-activation is
        // enforced durably by the fence: a stale incarnation cannot commit (invariant #5).
        if self.inner.directory.contains_key(&id) {
            return Err(SubErr::Busy(id));
        }
        self.inner.directory.insert(id.clone(), ());

        let inner = self.inner.clone();
        let task_id = id.clone();
        let handle = self.inner.tracker.spawn(async move {
            let result = inner.run_cycle(&task_id, fence).await;
            // Passivate: drop the directory entry so memory returns to baseline (invariant #8).
            inner.directory.remove(&task_id);
            result
        });
        match handle.await {
            Ok(result) => result,
            Err(join_err) => {
                self.inner.directory.remove(&id);
                Err(SubErr::Join(join_err.to_string()))
            }
        }
    }

    async fn passivate(&self, id: &SessionId) {
        self.inner.directory.remove(id);
    }

    async fn deliver(&self, id: &SessionId, msg: SessionMsg) -> Result<(), SubErr> {
        match msg {
            SessionMsg::Wake => self.wake(id.clone()).await,
        }
    }
}
