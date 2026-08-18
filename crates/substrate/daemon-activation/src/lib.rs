// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-activation` — the durable activation / virtual-entity core.
//!
//! The correctness-critical layer with no upstream reference implementation (the build-first
//! milestone). It owns the active-only directory, the monotonic lease/fence, the wake/job-outbox
//! dispatchers, the completion consumer, and the recovery scanner — proving lifecycle §4 invariants
//! #1, #5, #6, #7, #8. It drives engines through a protocol-agnostic seam ([`Incarnation`] /
//! [`EngineFactory`]) so the durable core remains independent of `daemon-core` and `daemon-host`.
//! Depends on `daemon-store` + `daemon-common`.
//!
//! The `elfo` feature (off by default) is reserved for an optional elfo-backed mailbox experiment.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::{DaemonError, Epoch, FenceToken, PartitionId, SessionId};
use daemon_store::{
    Checkpoint, ExecutionPolicy, InboxSplice, JobCommand, JobCompletion, ParkedApproval,
    SessionStatus, SessionStore, StoreError, TurnSeal,
};
use daemon_telemetry::{current_trace, ingress_trace, with_trace};
use dashmap::DashMap;
use std::sync::Arc;
use tokio_util::task::TaskTracker;
use tracing::Instrument;

// Re-export so downstream crates need only depend on `daemon-activation` for the seam.
pub use daemon_common::SnapshotBlob;

/// The outcome of running one activation of an engine incarnation.
pub enum Step {
    /// The engine reached a terminal state this activation.
    Completed,
    /// The engine finished a turn WITHOUT terminating the session (session-unification §5): the
    /// persisted [`ExecutionPolicy`] said the turn boundary commits back to `Idle`/`Ready`
    /// (interactive-root), never `Completed`. The activation layer routes this through the fenced
    /// [`SessionStore::commit_turn`] — snapshot, `turn_seq`, splice consumption, the turn's
    /// journal seal, and the next status in ONE transaction.
    TurnCommitted,
    /// The engine suspended at a phase boundary, delegating background work.
    Suspended {
        /// The durable job to enqueue on the outbox.
        job: JobCommand,
    },
    /// The engine suspended on a §12 edit-approval decision (HITL): the session parks dormant until
    /// an operator answers. Unlike [`Suspended`](Step::Suspended) **no** runnable job is enqueued —
    /// the wake comes from `answer_approval`, not a background worker.
    ParkApproval {
        /// The parked approval request(s) recorded this suspension (the snapshot already holds the
        /// typed `PendingApproval`s; these are the store-side rows for the operator-facing surface).
        approvals: Vec<ParkedApproval>,
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

/// The per-activation turn context the manager hands to [`Incarnation::hydrate`]
/// (session-unification §5): everything the incarnation needs to decide and journal the turn
/// boundary, read from the SAME load transaction as the snapshot it runs.
#[derive(Clone, Copy, Debug)]
pub struct TurnCtx {
    /// The persisted execution policy driving terminal-vs-idle at the turn boundary (`None` on
    /// legacy rows — the incarnation keeps terminal semantics).
    pub policy: Option<ExecutionPolicy>,
    /// The in-flight turn's identity = its journal segment index (the session's committed-turn
    /// count at load; a resumed suspension re-loads the same value and continues the segment).
    pub turn_seq: u64,
    /// The activation fence — rides every durable journal append (§5: a stale incarnation can
    /// neither append into nor seal the winning segment).
    pub fence: FenceToken,
}

/// One live engine incarnation, driven by the activation layer (the phase-1 stand-in for the
/// `daemon-host` session task). Protocol-agnostic: it deals in opaque [`SnapshotBlob`]s and the
/// durable job/completion types, never §17 messages directly.
#[async_trait]
pub trait Incarnation: Send {
    /// Reconstruct from the last snapshot, apply unapplied completions idempotently, and fold the
    /// claimed durable-inbox splices (session-unification §4.2; already CAS-claimed under this
    /// activation's fence by `load_for_activation`) — all *before* running new work (lifecycle
    /// §3.1, invariant #2). Splices at or below the snapshot's consumed cursor were captured by an
    /// earlier commit and must be skipped, never re-folded.
    ///
    /// Commit-then-linger (§8): the manager may hydrate the SAME instance again after its own
    /// non-terminal turn commit, guaranteeing `snapshot` is byte-identical to the blob that commit
    /// persisted (a diverged store discards the instance instead). An implementation may therefore
    /// keep its engine resident across hydrates and fold the new work in place; rebuilding from
    /// `snapshot` is always a correct fallback.
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<JobCompletion>,
        splices: Vec<InboxSplice>,
        ctx: TurnCtx,
    ) -> Result<(), EngineError>;

    /// Process available work, returning whether the engine completed or suspended.
    async fn run(&mut self) -> Result<Step, EngineError>;

    /// Produce the snapshot to persist at the current phase boundary.
    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError>;

    /// The current incarnation epoch (post-bump at suspension).
    fn epoch(&self) -> Epoch;

    /// The highest durable-inbox `splice_seq` this incarnation's snapshot has folded, stamped onto
    /// every commit ([`Checkpoint::consumed_splices`]) so the store flips exactly that prefix to
    /// `Consumed` inside the commit transaction (§4.2: consumption is never written separately).
    /// `None` = this incarnation makes no splice statement (nothing is consumed).
    fn consumed_splices(&self) -> Option<u64> {
        None
    }

    /// The structured completion payload to record when this incarnation reaches `Step::Completed`
    /// (daemon-content-transfer-spec.md Phase 2a: a CBOR `DelegationResult` capturing the child's
    /// summary + artifact refs). Default `None` (legacy `child:{id}` marker). A `daemon-core`
    /// incarnation with a content store + workspace roots overrides it to capture the child's
    /// `outbox/`.
    fn completion_payload(&self) -> Option<Vec<u8>> {
        None
    }

    /// The committed turn's journal segment seal, taken by the manager when this incarnation
    /// returns [`Step::TurnCommitted`] and written INSIDE the `commit_turn` transaction
    /// (session-unification §5 item 3: the root is promoted atomically with the state it covers).
    /// Default `None` (a non-journaling incarnation commits the turn without a seal).
    fn take_turn_seal(&mut self) -> Option<TurnSeal> {
        None
    }
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
    /// The active-only directory: currently running sessions, each holding its reservation
    /// generation. Returns to baseline after passivation (invariant #8) — this is what the churn
    /// acceptance test asserts. Reservation is an atomic insert-if-vacant acquired BEFORE the
    /// lease (session-unification §6): the old lease-then-check order let concurrent wakes bump
    /// the fence past an in-flight incarnation of the same session (self-fencing).
    directory: DashMap<SessionId, u64>,
    /// Monotonic reservation-generation mint, so a guard's release can never remove a newer
    /// incarnation's reservation (the double-remove hazard on racing exits).
    generations: std::sync::atomic::AtomicU64,
    /// Tracks live activation tasks so their memory is released on completion (invariant #8).
    tracker: TaskTracker,
    /// Commit-then-linger residency (session-unification §8): after a non-terminal turn commit
    /// the incarnation stays hydrated this long awaiting the next wake (no rehydrate cost per
    /// message); the timeout only passivates the ALREADY-COMMITTED incarnation — no commit is
    /// ever owed at passivation. `None` disables lingering (every commit passivates immediately).
    linger: Option<std::time::Duration>,
    /// The lingering incarnations' wake mailboxes: a wake that finds the slot occupied hands the
    /// hint to the lingerer here instead of dropping it (the sole in-process wake seam a resident
    /// incarnation has — durable wakes stay authoritative via the store + recovery scanner).
    lingers: DashMap<SessionId, Arc<tokio::sync::Notify>>,
    /// Cancelled at shutdown so lingering incarnations exit immediately instead of holding the
    /// task tracker open for a full idle timeout.
    shutdown: tokio_util::sync::CancellationToken,
}

/// An owned directory reservation (session-unification §6): acquired before the activation lease,
/// released on EVERY exit path — lease failure, spawn failure, cancellation, panic, shutdown —
/// by `Drop`, and generation-checked so it never removes a newer incarnation's reservation.
struct SlotGuard {
    inner: Arc<ManagerInner>,
    id: SessionId,
    generation: u64,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.inner
            .directory
            .remove_if(&self.id, |_, gen| *gen == self.generation);
    }
}

impl ManagerInner {
    /// Atomically reserve the session's slot (insert-if-vacant). `None` = a live incarnation
    /// already holds it in this process.
    fn try_reserve(self: &Arc<Self>, id: &SessionId) -> Option<SlotGuard> {
        let generation = self
            .generations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match self.directory.entry(id.clone()) {
            dashmap::mapref::entry::Entry::Occupied(_) => None,
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert(generation);
                Some(SlotGuard {
                    inner: self.clone(),
                    id: id.clone(),
                    generation,
                })
            }
        }
    }
}

/// One turn's outcome inside an activation cycle: the non-terminal `commit_turn` result paired
/// with the committed snapshot blob (the linger decision reads the status; the blob guards
/// resident-engine reuse), or `None` for the terminal/suspension/park commits that end the cycle
/// unconditionally.
type TurnResult = Result<Option<(daemon_store::TurnCommit, SnapshotBlob)>, SubErr>;

impl ManagerInner {
    /// Drive one activation cycle (session-unification §8): a loop of fenced
    /// load→hydrate→run→commit turns on one slot reservation, with a fresh lease per follow-on
    /// turn so splices are never claimed under a stale fence.
    ///
    /// Reporting discipline: the first turn's outcome goes to `first_done` — but ONLY when the
    /// cycle decides to keep running (resident drain / linger), where releasing the caller early
    /// is safe because the resident incarnation itself claims any wake absorbed against its held
    /// slot. On every EXIT path the sender is left untouched and the result is returned, so the
    /// caller reports it strictly AFTER the slot drops — a wake dispatched right behind this
    /// cycle's end must find the slot free, or its work would strand `Ready` until a scanner pass.
    async fn run_cycle(
        &self,
        id: &SessionId,
        first_fence: FenceToken,
        first_done: &mut Option<tokio::sync::oneshot::Sender<Result<(), SubErr>>>,
    ) -> Result<(), SubErr> {
        let span = tracing::info_span!(
            "activation.run_cycle",
            trace_id = %current_trace(),
            session = %id,
            fence = first_fence.0
        );
        async {
            let mut inc = self.factory.create();
            let mut fence = first_fence;
            // The blob the incarnation's last commit persisted: an incarnation may keep its engine
            // resident across iterations ONLY while the stored snapshot is byte-identical to it —
            // any divergence (e.g. a rewind CAS landed while lingering) discards the resident
            // incarnation, because the store is the authority.
            let mut last_committed: Option<SnapshotBlob> = None;
            loop {
                // Terminal/suspension/park commits (`None`) and errors passivate unconditionally.
                let Some((commit, snapshot)) = self
                    .run_turn(id, fence, &mut inc, last_committed.take())
                    .await?
                else {
                    return Ok(());
                };
                let Some(linger) = self.linger else {
                    return Ok(());
                };
                match commit.status {
                    // Work already queued (the commit enqueued its own self-wake, which a later
                    // dispatch absorbs benignly): release the caller and drain it on the resident
                    // engine now.
                    SessionStatus::Ready => {
                        if let Some(tx) = first_done.take() {
                            let _ = tx.send(Ok(()));
                        }
                    }
                    // Nothing queued: release the caller and linger hydrated awaiting the next
                    // wake. Timeout / shutdown passivates the already-committed incarnation — no
                    // commit is owed.
                    SessionStatus::Idle => {
                        if let Some(tx) = first_done.take() {
                            let _ = tx.send(Ok(()));
                        }
                        if !self.linger_wait(id, linger).await {
                            return Ok(());
                        }
                        // Only claimable work re-enters the loop; a spurious notify with nothing
                        // queued would otherwise open a BLANK turn (the incident shape stage 1
                        // buried).
                        if !matches!(self.store.status(id).await, Some(SessionStatus::Ready)) {
                            return Ok(());
                        }
                    }
                    _ => return Ok(()),
                }
                last_committed = Some(snapshot);
                // A fresh lease per follow-on turn: the next load claims splices under the LATEST
                // fence — exactly what a fresh wake would hold — so a lingering incarnation can
                // never claim work under a superseded fence.
                fence = self.store.acquire_activation_lease(id).await?;
            }
        }
        .instrument(span)
        .await
    }

    /// One fenced load→hydrate→run→commit turn. Returns the non-terminal turn commit (the caller
    /// decides whether to linger on its status), `None` after a terminal/suspension/park commit.
    async fn run_turn(
        &self,
        id: &SessionId,
        fence: FenceToken,
        inc: &mut Box<dyn Incarnation>,
        last_committed: Option<SnapshotBlob>,
    ) -> TurnResult {
        let activation = self.store.load_for_activation(id, fence).await?;
        // §8 resident-reuse guard: the incarnation may only keep its engine when the stored
        // snapshot is exactly the blob its own commit persisted.
        if let Some(committed) = last_committed {
            if committed != activation.snapshot {
                *inc = self.factory.create();
            }
        }
        tracing::debug!(
            trace_id = %current_trace(),
            session = %id,
            unapplied_jobs = activation.unapplied.len(),
            splices = activation.splices.len(),
            "activation.hydrate"
        );
        let ctx = TurnCtx {
            policy: activation.policy,
            turn_seq: activation.turn_seq,
            fence,
        };
        // The completion keys this load delivered: the incarnation folds them at hydrate, so
        // a non-terminal turn commit deletes exactly these rows (`applied_completions`) —
        // otherwise the folded completions would count as pending work forever and
        // `commit_turn`'s Idle-iff-no-work rule would livelock the session on Ready+self-wake.
        let applied_completions: Vec<_> = activation
            .unapplied
            .iter()
            .map(|c| (c.epoch, c.job_id.clone()))
            .collect();
        inc.hydrate(
            activation.snapshot,
            activation.unapplied,
            activation.splices,
            ctx,
        )
        .await?;
        match inc.run().await? {
            Step::Suspended { job } => {
                let snapshot = inc.checkpoint()?;
                let checkpoint = Checkpoint::new(id.clone(), inc.epoch(), snapshot)
                    .with_consumed_splices(inc.consumed_splices());
                tracing::info!(
                    trace_id = %current_trace(),
                    session = %id,
                    epoch = inc.epoch().0,
                    step = "Suspended",
                    job_id = %job.job_id,
                    "activation.commit"
                );
                self.store
                    .checkpoint_and_enqueue(checkpoint, job, fence)
                    .await?;
                Ok(None)
            }
            Step::ParkApproval { approvals } => {
                // §12 HITL park: checkpoint the suspended snapshot + record the parked approval rows
                // in one transaction, but enqueue *no* runnable job — the session stays dormant until
                // an operator `answer_approval` wakes it (recovery re-park dedupes on the unique row).
                let snapshot = inc.checkpoint()?;
                let checkpoint = Checkpoint::new(id.clone(), inc.epoch(), snapshot)
                    .with_consumed_splices(inc.consumed_splices());
                tracing::info!(
                    trace_id = %current_trace(),
                    session = %id,
                    epoch = inc.epoch().0,
                    step = "ParkApproval",
                    approvals = approvals.len(),
                    "activation.commit"
                );
                self.store
                    .park_approval(checkpoint, approvals, fence)
                    .await?;
                Ok(None)
            }
            Step::TurnCommitted => {
                // The non-terminal turn boundary (session-unification §5): commit the
                // snapshot + turn_seq + consumed splices + the turn's journal seal + the
                // Idle/Ready selection in ONE fenced transaction — IMMEDIATELY, before any
                // lingering (commit first, then linger: crash-of-resident loses nothing).
                let snapshot = inc.checkpoint()?;
                let committed = snapshot.clone();
                let checkpoint = Checkpoint::new(id.clone(), inc.epoch(), snapshot)
                    .with_consumed_splices(inc.consumed_splices())
                    .with_applied_completions(applied_completions);
                let seal = inc.take_turn_seal();
                let commit = self.store.commit_turn(checkpoint, seal, fence).await?;
                tracing::info!(
                    trace_id = %current_trace(),
                    session = %id,
                    epoch = inc.epoch().0,
                    step = "TurnCommitted",
                    turn_seq = commit.turn_seq,
                    status = ?commit.status,
                    "activation.commit"
                );
                Ok(Some((commit, committed)))
            }
            Step::Completed => {
                let snapshot = inc.checkpoint()?;
                // A delegated child carries its structured result (DelegationResult: summary +
                // artifact refs) on the completion payload; the incarnation captured it at terminal.
                let checkpoint = Checkpoint::new(id.clone(), inc.epoch(), snapshot)
                    .with_completion_payload(inc.completion_payload())
                    .with_consumed_splices(inc.consumed_splices());
                tracing::info!(
                    trace_id = %current_trace(),
                    session = %id,
                    epoch = inc.epoch().0,
                    step = "Completed",
                    "activation.commit"
                );
                self.store.mark_completed(checkpoint, fence).await?;
                Ok(None)
            }
        }
    }

    /// Park this cycle between turns (commit-then-linger §8): register the session's wake mailbox,
    /// then wait for a handed-off wake, the idle `timeout`, or shutdown. Returns `true` when a wake
    /// arrived (the caller re-checks the store before running — the store stays authoritative).
    /// A wake absorbed in the unregister→slot-release window is recovered by the scanner, the same
    /// safety net every absorbed wake already rides.
    async fn linger_wait(&self, id: &SessionId, timeout: std::time::Duration) -> bool {
        let notify = Arc::new(tokio::sync::Notify::new());
        self.lingers.insert(id.clone(), notify.clone());
        // A splice that landed between the commit and this registration already flipped `Ready`
        // (its wake found the slot occupied with no mailbox yet): don't wait on it.
        let woke = if matches!(self.store.status(id).await, Some(SessionStatus::Ready)) {
            true
        } else {
            tokio::select! {
                _ = notify.notified() => true,
                _ = tokio::time::sleep(timeout) => false,
                _ = self.shutdown.cancelled() => false,
            }
        };
        self.lingers.remove(id);
        woke
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
    /// Lingering is disabled: every commit passivates immediately (the pre-§8 behavior).
    pub fn new(
        store: Arc<dyn SessionStore>,
        factory: Arc<dyn EngineFactory>,
        partition: PartitionId,
    ) -> Self {
        Self::with_linger(store, factory, partition, None)
    }

    /// As [`Self::new`], with commit-then-linger residency (session-unification §8): after a
    /// non-terminal turn commit the incarnation stays hydrated for up to `linger` awaiting the
    /// next wake (no rehydrate cost per message); the timeout only passivates the
    /// already-committed incarnation. `None` disables lingering.
    pub fn with_linger(
        store: Arc<dyn SessionStore>,
        factory: Arc<dyn EngineFactory>,
        partition: PartitionId,
        linger: Option<std::time::Duration>,
    ) -> Self {
        Self {
            inner: Arc::new(ManagerInner {
                store,
                factory,
                partition,
                directory: DashMap::new(),
                generations: std::sync::atomic::AtomicU64::new(0),
                tracker: TaskTracker::new(),
                linger,
                lingers: DashMap::new(),
                shutdown: tokio_util::sync::CancellationToken::new(),
            }),
        }
    }

    /// The number of currently active incarnations in this process (acceptance test #1 baseline).
    pub fn active_count(&self) -> usize {
        self.inner.directory.len()
    }

    /// Acquire a fresh lease and activate `id`, guarding against re-entry, completed sessions,
    /// and idle (no-runnable-work) sessions. This is the convenience wake path; the store remains
    /// authoritative (invariant #1).
    ///
    /// Ordering (session-unification §6): the in-process slot is reserved BEFORE the lease is
    /// acquired. The old lease-then-check order meant every concurrent wake of a busy session
    /// bumped the fence — self-fencing the in-flight incarnation's eventual commit.
    pub async fn wake(&self, id: SessionId) -> Result<(), SubErr> {
        match self.inner.store.status(&id).await {
            // `Idle` = exists, no runnable work (never scanner/wake work); the splice/completion
            // that creates work flips it `Ready` in the same transaction.
            Some(SessionStatus::Completed) | Some(SessionStatus::Idle) | None => return Ok(()),
            _ => {}
        }
        // A live incarnation already holds the slot: wake's contract ("ensure it is progressing")
        // is satisfied without touching the lease. If that incarnation is LINGERING between turns
        // (§8), hand it the hint so it drains the new work on its resident engine.
        let Some(slot) = self.inner.try_reserve(&id) else {
            if let Some(mailbox) = self.inner.lingers.get(&id) {
                mailbox.notify_one();
            }
            return Ok(());
        };
        let fence = self.inner.store.acquire_activation_lease(&id).await?;
        match self.run_reserved(id, fence, slot).await {
            // A superseding lease fenced our incarnation mid-run (e.g. a cross-process writer):
            // the winner is driving the session, so the wake is satisfied.
            Err(SubErr::Store(StoreError::Fenced { .. })) => Ok(()),
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

    /// One recovery-scan pass: re-activate every session the store reports as resumable
    /// (`Ready`/`Active`) whose wake may have been lost (invariant #7). This is the per-tick body
    /// the host's `RecoveryScanner` resident service runs on an interval.
    pub async fn scan_once(&self) -> Result<usize, SubErr> {
        let mut scanned = 0usize;
        for id in self
            .inner
            .store
            .scan_resumable(self.inner.partition)
            .await?
        {
            self.wake(id).await?;
            scanned += 1;
        }
        Ok(scanned)
    }

    /// The recovery scanner: rebuild from the store alone (in-memory directories are gone).
    /// Drains durable work, dispatches pending wakes, then re-activates any session left in a
    /// resumable state whose wake never arrived (lifecycle §3.1; invariants #5, #7). Loops until
    /// the world is quiescent so a multi-step cycle (suspend -> work -> resume) fully drains.
    pub async fn recover(&self) -> Result<(), SubErr> {
        loop {
            let jobs = self.run_workers().await?;
            let wakes = self.dispatch_wakes().await?;
            let scanned = self.scan_once().await?;
            if jobs == 0 && wakes == 0 && scanned == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Gracefully close the task tracker and wait for in-flight activations to drain. Lingering
    /// incarnations are released immediately (they owe no commit — §8 commits before lingering).
    pub async fn shutdown(&self) {
        self.inner.shutdown.cancel();
        self.inner.tracker.close();
        self.inner.tracker.wait().await;
    }
}

impl ActivationManager {
    /// Run one activation cycle under an already-held slot reservation: spawn the tracked task
    /// with the guard moved into it, so the slot is released on every exit path — completion,
    /// error, cancellation, or panic (the guard's `Drop`; invariant #8).
    ///
    /// The caller awaits only the FIRST turn's outcome (reported through a oneshot), not the
    /// task's completion: a lingering cycle (§8) may hold its slot long after the first commit,
    /// and awaiting it would stall the wake dispatcher for the whole linger window.
    async fn run_reserved(
        &self,
        id: SessionId,
        fence: FenceToken,
        slot: SlotGuard,
    ) -> Result<(), SubErr> {
        let trace = ingress_trace(Some(current_trace()));
        tracing::debug!(
            trace_id = %trace,
            session = %id,
            fence = fence.0,
            "activation.wake"
        );
        let inner = self.inner.clone();
        let (first_done, first_result) = tokio::sync::oneshot::channel();
        self.inner.tracker.spawn(with_trace(trace, async move {
            let mut first_done = Some(first_done);
            let result = inner.run_cycle(&slot.id, fence, &mut first_done).await;
            // Passivate: the guard drops here, releasing the directory entry so memory returns to
            // baseline (invariant #8). Generation-checked, so a racing newer reservation survives.
            // The caller is released strictly AFTER the drop (unless a linger continuation
            // already released it), so a wake dispatched behind this result finds the slot free.
            drop(slot);
            match first_done.take() {
                Some(tx) => {
                    let _ = tx.send(result);
                }
                // The cycle already released its caller (it continued past its first commit):
                // report follow-on failures to the log instead. A fence loss is benign — a newer
                // lease owns the session; the durable state is already committed either way and
                // the scanner re-drives `Ready`.
                None => match result {
                    Ok(()) => {}
                    Err(SubErr::Store(StoreError::Fenced { .. })) => {
                        tracing::debug!(
                            trace_id = %current_trace(),
                            "linger turn fenced; a newer activation owns the session"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            trace_id = %current_trace(),
                            error = %e,
                            "linger turn failed; passivating"
                        );
                    }
                },
            }
        }));
        match first_result.await {
            Ok(result) => result,
            // The sender dropped without reporting: the cycle task panicked before its first
            // commit (a completed first turn always sends after releasing the slot).
            Err(_) => Err(SubErr::Join(format!(
                "activation cycle for {id} aborted before its first turn commit"
            ))),
        }
    }
}

#[async_trait]
impl ActivationSubstrate for ActivationManager {
    async fn activate(&self, id: SessionId, fence: FenceToken) -> Result<(), SubErr> {
        // Single-activation guard for this process (invariant #6): atomic insert-if-vacant, so two
        // racing activations can never both pass a check-then-insert. Cluster-wide
        // single-activation is enforced durably by the fence: a stale incarnation cannot commit
        // (invariant #5). NOTE: callers of this seam bring their own fence (acquired elsewhere);
        // the reserve-before-lease ordering lives in [`ActivationManager::wake`].
        let Some(slot) = self.inner.try_reserve(&id) else {
            return Err(SubErr::Busy(id));
        };
        self.run_reserved(id, fence, slot).await
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
