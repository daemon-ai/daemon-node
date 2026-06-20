//! `daemon-conformance` — the substrate + translation conformance harness.
//!
//! The executable acceptance gate for the build-first milestones: the seven substrate acceptance
//! tests from [`rust-substrate-evaluation.md`](../../../docs/specs/rust-substrate-evaluation.md) §6,
//! run against the in-memory [`daemon_store::InMemoryStore`] driven through [`daemon_activation`].
//! From phase 3 the engine under test is the *real* `daemon-core`, driven via the host's
//! [`CoreEngineFactory`](daemon_host::CoreEngineFactory) (the stub engine is retired): the substrate
//! invariants are now proven against the real engine's deterministic delegate→suspend→resume cycle.
//!
//! Coverage map (acceptance test -> lifecycle §4 invariant):
//! 1 churn/baseline (#8), 2 crash-after-every-boundary (#2/#3/#7), 3 idempotency (#2/#3),
//! 4 dual-node fencing (#5/#6), 5 empty-mailbox kill (#1/#7), 6 ownership-transfer (#5/#6),
//! 7 lost-wake recovery (#1/#7).
//!
//! `mod supervision` (phase 2): the resident-service supervisor (restart/backoff/meltdown) and the
//! running `daemon_host::Host` driving sessions to completion under churn and service crashes.
//! `mod translation` (phase 3 gate): the §17 ⇄ management protocol round-trip — the host presents
//! the real engine as a `ManagedUnit` and the supervision §4 mapping table is exercised end to end.
//! `mod orchestration` (phase 4 gate): one engine delegates to a child via the `daemon-orchestration`
//! fleet runtime + `daemon-tool-orchestrate` veneer — events fan in, the child's completion wakes the
//! parent, and a child request is answered/escalated (layout §7 phase-4 gate).

#![forbid(unsafe_code)]

#[cfg(test)]
mod harness {
    use daemon_activation::ActivationManager;
    use daemon_common::{PartitionId, SessionId};
    use daemon_core::Snapshot;
    use daemon_host::CoreEngineFactory;
    use daemon_store::{InMemoryStore, JobCompletion, SessionStatus, SessionStore};
    use std::sync::Arc;

    pub const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// A fresh store + a single activation manager owning the default partition.
    pub fn new_world() -> (Arc<InMemoryStore>, ActivationManager) {
        let store = Arc::new(InMemoryStore::new());
        let mgr = manager(store.clone());
        (store, mgr)
    }

    /// An activation manager over an existing (possibly shared) store, driving the real engine.
    pub fn manager(store: Arc<InMemoryStore>) -> ActivationManager {
        ActivationManager::new(store, Arc::new(CoreEngineFactory::delegating()), PARTITION)
    }

    /// Create a fresh `Ready` session with an encoded empty snapshot.
    pub async fn seed(store: &InMemoryStore, id: &SessionId) {
        let blob = Snapshot::fresh(id.clone())
            .encode()
            .expect("encode fresh snapshot");
        store
            .create_session(id.clone(), PARTITION, blob)
            .await
            .expect("create session");
    }

    pub async fn status(store: &InMemoryStore, id: &SessionId) -> Option<SessionStatus> {
        store.status(id).await
    }

    pub async fn assert_completed(store: &InMemoryStore, id: &SessionId) {
        assert_eq!(
            status(store, id).await,
            Some(SessionStatus::Completed),
            "session {id} should be Completed"
        );
    }

    /// Build a completion for whatever job is sitting on the durable outbox.
    pub async fn completion_for_next_job(store: &InMemoryStore) -> JobCompletion {
        let job = store.dequeue_job().await.expect("a job on the outbox");
        JobCompletion {
            session_id: job.session_id,
            epoch: job.epoch,
            job_id: job.job_id,
            payload: job.payload,
        }
    }
}

#[cfg(test)]
mod acceptance {
    use super::harness::*;
    use daemon_activation::{ActivationSubstrate, SubErr};
    use daemon_common::SessionId;
    use daemon_store::{FaultPoint, SessionStatus, SessionStore, StoreError};

    /// Default churn size for CI. The full 1,000,000-session run lives in the `#[ignore]`d test
    /// below (acceptance test #1 specifies >= 1e6).
    const CHURN_SESSIONS: usize = 2_000;

    /// #1 — churn / memory baseline. Activate and passivate many unique sessions; the in-memory
    /// active directory returns to a stable baseline (no per-incarnation leak — invariant #8).
    async fn run_churn(n: usize) {
        let (store, mgr) = new_world();
        for i in 0..n {
            let id = SessionId::new(format!("churn-{i}"));
            seed(&store, &id).await;
            // Activate (first turn suspends) then passivate; directory must not retain the entry.
            mgr.wake(id).await.expect("wake");
            assert_eq!(mgr.active_count(), 0, "directory leaked after session {i}");
        }
        assert_eq!(mgr.active_count(), 0, "active directory above baseline");
    }

    #[tokio::test]
    async fn test_1_churn_baseline() {
        run_churn(CHURN_SESSIONS).await;
    }

    #[tokio::test]
    #[ignore = "heavy: 1,000,000 sessions; run with --ignored"]
    async fn test_1_churn_one_million() {
        run_churn(1_000_000).await;
    }

    /// #2 — crash-after-every-boundary. Inject a crash at each durable boundary of the
    /// delegate -> complete -> resume cycle and recover correctly each time.
    #[tokio::test]
    async fn test_2_crash_after_every_boundary() {
        // (a) before snapshot: the checkpoint transaction aborts; the session stays activatable.
        {
            let (store, mgr) = new_world();
            let id = SessionId::new("crash-before-snapshot");
            seed(&store, &id).await;
            let f = store.acquire_activation_lease(&id).await.unwrap();
            store.set_fault(Some(FaultPoint::BeforeSnapshot));
            let r = mgr.activate(id.clone(), f).await;
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            mgr.recover().await.unwrap();
            assert_completed(&store, &id).await;
        }

        // (b) after snapshot and (c) after job outbox: the checkpoint committed atomically, then
        // the task died. Recovery drains the durable job outbox and finishes the cycle.
        for fault in [FaultPoint::AfterSnapshot, FaultPoint::AfterJobOutbox] {
            let (store, mgr) = new_world();
            let id = SessionId::new(format!("crash-{fault:?}"));
            seed(&store, &id).await;
            let f = store.acquire_activation_lease(&id).await.unwrap();
            store.set_fault(Some(fault));
            let r = mgr.activate(id.clone(), f).await;
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            // Durable state: Suspended with the job on the outbox.
            assert!(matches!(
                status(&store, &id).await,
                Some(SessionStatus::Suspended { .. })
            ));
            mgr.recover().await.unwrap();
            assert_completed(&store, &id).await;
        }

        // (d) before task exit: the checkpoint committed; the *process* is killed (drop the
        // manager) and a fresh manager recovers solely from the store + durable queues.
        {
            let (store, mgr1) = new_world();
            let id = SessionId::new("crash-before-task-exit");
            seed(&store, &id).await;
            mgr1.wake(id.clone()).await.unwrap(); // suspends; job durable
            assert!(matches!(
                status(&store, &id).await,
                Some(SessionStatus::Suspended { .. })
            ));
            drop(mgr1);
            let mgr2 = manager(store.clone());
            mgr2.recover().await.unwrap();
            assert_completed(&store, &id).await;
        }

        // (e) after completion insert: completion durable + wake published; crash before the wake
        // is consumed. Recovery dispatches the pending wake.
        {
            let (store, mgr) = new_world();
            let id = SessionId::new("crash-after-completion-insert");
            seed(&store, &id).await;
            mgr.wake(id.clone()).await.unwrap();
            mgr.run_workers().await.unwrap(); // completion recorded + wake enqueued
            assert_eq!(status(&store, &id).await, Some(SessionStatus::Ready));
            mgr.recover().await.unwrap(); // dispatches the pending wake
            assert_completed(&store, &id).await;
        }

        // (f) before wake publication: completion durable + Ready, but the wake was never
        // published. The recovery *scan* (not the wake) must re-activate the Ready session.
        {
            let (store, mgr) = new_world();
            let id = SessionId::new("crash-before-wake-publish");
            seed(&store, &id).await;
            mgr.wake(id.clone()).await.unwrap();
            store.set_fault(Some(FaultPoint::BeforeWakePublish));
            let r = mgr.run_workers().await; // completion commits; wake publication faults
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            assert_eq!(status(&store, &id).await, Some(SessionStatus::Ready));
            assert!(store.dequeue_wake().await.is_none(), "wake should be lost");
            mgr.recover().await.unwrap(); // scan_resumable rescues the Ready session
            assert_completed(&store, &id).await;
        }
    }

    /// #3 — wake/completion idempotency. A completion delivered repeatedly is applied at most once
    /// (`UNIQUE(session_id, epoch, job_id)`), and yields at most one wake (invariants #2, #3).
    #[tokio::test]
    async fn test_3_wake_completion_idempotency() {
        let (store, mgr) = new_world();
        let id = SessionId::new("idempotent");
        seed(&store, &id).await;
        mgr.wake(id.clone()).await.unwrap(); // suspends; one job enqueued

        let completion = completion_for_next_job(&store).await;
        // Deliver the same completion several times.
        for _ in 0..5 {
            store.record_completion_and_wake(&completion).await.unwrap();
        }
        // Only the first delivery enqueued a wake.
        assert_eq!(store.dequeue_wake().await.as_ref(), Some(&id));
        assert!(
            store.dequeue_wake().await.is_none(),
            "duplicate completions must not enqueue extra wakes"
        );

        // Resume applies the single completion and finishes.
        mgr.wake(id.clone()).await.unwrap();
        assert_completed(&store, &id).await;

        // A late duplicate after completion is still a no-op (no new wake, stays Completed).
        store.record_completion_and_wake(&completion).await.unwrap();
        assert!(store.dequeue_wake().await.is_none());
        assert_completed(&store, &id).await;
    }

    /// #4 — dual-node fencing. Two managers over one shared store concurrently activate the same
    /// session; only the holder of the highest fencing token may commit (invariant #5).
    #[tokio::test]
    async fn test_4_dual_node_fencing() {
        let (store, mgr_a) = new_world();
        let mgr_b = manager(store.clone());
        let id = SessionId::new("dual-node");
        seed(&store, &id).await;

        // Node A acquires the lease first, then node B steals it (higher token).
        let fa = store.acquire_activation_lease(&id).await.unwrap();
        let fb = store.acquire_activation_lease(&id).await.unwrap();
        assert!(fb > fa);

        // The stale node A cannot commit its checkpoint.
        let ra = mgr_a.activate(id.clone(), fa).await;
        assert!(
            matches!(ra, Err(SubErr::Store(StoreError::Fenced { .. }))),
            "stale node committed: {ra:?}"
        );

        // The current node B commits successfully.
        let rb = mgr_b.activate(id.clone(), fb).await;
        assert!(rb.is_ok(), "current node should commit: {rb:?}");
    }

    /// #5 — empty-mailbox process kill. Kill the whole process while all in-memory mailboxes are
    /// empty; recover solely from the store + durable queues (invariants #1, #7).
    #[tokio::test]
    async fn test_5_empty_mailbox_process_kill() {
        let store = {
            let (store, mgr1) = new_world();
            let id = SessionId::new("process-kill");
            seed(&store, &id).await;
            mgr1.wake(id.clone()).await.unwrap(); // suspends; job durable
            drop(mgr1); // "kill the process": all in-memory state gone
            store
        };

        // A fresh process: new manager, empty directory, recovers from durable state alone.
        let mgr2 = manager(store.clone());
        mgr2.recover().await.unwrap();
        assert_completed(&store, &SessionId::new("process-kill")).await;
        assert_eq!(mgr2.active_count(), 0);
    }

    /// #6 — ownership-transfer stale-write rejection. Pause an old owner, transfer ownership,
    /// resume the old owner; its writes are rejected (invariants #5, #6).
    #[tokio::test]
    async fn test_6_ownership_transfer_stale_write() {
        let store = std::sync::Arc::new(daemon_store::InMemoryStore::new());
        let id = SessionId::new("ownership-transfer");
        seed(&store, &id).await;

        // Old owner acquires the lease (pause it before it commits).
        let f_old = store.acquire_activation_lease(&id).await.unwrap();
        // Ownership transfers to a new owner (higher token).
        let f_new = store.acquire_activation_lease(&id).await.unwrap();

        // Resume the old owner: its checkpoint commit is rejected by the fence.
        let old_owner = manager(store.clone());
        let r_old = old_owner.activate(id.clone(), f_old).await;
        assert!(
            matches!(r_old, Err(SubErr::Store(StoreError::Fenced { .. }))),
            "stale write was not rejected: {r_old:?}"
        );

        // The new owner makes progress.
        let new_owner = manager(store.clone());
        let r_new = new_owner.activate(id.clone(), f_new).await;
        assert!(r_new.is_ok(), "new owner should commit: {r_new:?}");
    }

    /// #7 — lost-wake recovery. Drop a wake notification entirely; the recovery scan eventually
    /// re-activates the `Ready` session (invariant #7).
    #[tokio::test]
    async fn test_7_lost_wake_recovery() {
        let (store, mgr) = new_world();
        let id = SessionId::new("lost-wake");
        seed(&store, &id).await;
        mgr.wake(id.clone()).await.unwrap(); // suspends; job enqueued
        mgr.run_workers().await.unwrap(); // completion recorded; wake enqueued + Ready

        // Drop the wake hint entirely — a naive dispatcher would now strand the session.
        let dropped = store.dequeue_wake().await;
        assert_eq!(dropped.as_ref(), Some(&id));
        assert!(store.dequeue_wake().await.is_none());

        // The recovery scan rescues the Ready session despite the lost wake.
        mgr.recover().await.unwrap();
        assert_completed(&store, &id).await;
    }
}

#[cfg(test)]
mod supervision {
    //! Phase-2 gate: resident services restart/backoff/meltdown, and the running host drives the
    //! full lifecycle under churn + injected service crashes (`daemon-host-spec.md` §5;
    //! `daemon-workspace-layout.md` §7 phase-2 gate).

    use super::harness::{seed, status};
    use daemon_activation::ActivationManager;
    use daemon_common::{PartitionId, SessionId};
    use daemon_host::supervisor::ServiceFactory;
    use daemon_host::services::{interval_child, job_tick, scan_tick, wake_tick, TickFn};
    use daemon_host::{
        Backoff, ChildSpec, HealthStatus, Host, HostConfig, MeltdownPolicy, RestartPolicy,
        ServiceError, Supervisor,
    };
    use daemon_host::CoreEngineFactory;
    use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio_util::sync::CancellationToken;

    fn fast_backoff() -> Backoff {
        Backoff {
            initial: Duration::from_millis(1),
            max: Duration::from_millis(3),
            factor: 1.0,
        }
    }

    fn lenient_meltdown() -> MeltdownPolicy {
        MeltdownPolicy {
            max_restarts: 50,
            window: Duration::from_secs(30),
        }
    }

    /// A factory that returns immediately with the given result (no internal loop).
    fn immediate(ok: bool) -> ServiceFactory {
        Arc::new(move |_cancel| {
            Box::pin(async move {
                if ok {
                    Ok(())
                } else {
                    Err(ServiceError::new("boom"))
                }
            })
        })
    }

    /// Poll until every session is `Completed`, or the timeout elapses.
    async fn poll_until_completed(
        store: &InMemoryStore,
        ids: &[SessionId],
        timeout: Duration,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            let mut all = true;
            for id in ids {
                if store.status(id).await != Some(SessionStatus::Completed) {
                    all = false;
                    break;
                }
            }
            if all {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Poll a predicate until true or timeout.
    async fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if cond() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    /// A tick wrapper that panics on its first `budget` invocations (bounded chaos), then delegates.
    fn bounded_panic_tick(inner: TickFn, budget: Arc<AtomicUsize>) -> TickFn {
        Arc::new(move || {
            let inner = inner.clone();
            let budget = budget.clone();
            Box::pin(async move {
                if budget.load(Ordering::SeqCst) > 0 {
                    budget.fetch_sub(1, Ordering::SeqCst);
                    panic!("injected dispatcher crash");
                }
                inner().await
            })
        })
    }

    /// Restart policy: `Permanent` restarts always, `Temporary` never, `Transient` only on abnormal
    /// termination.
    #[tokio::test]
    async fn transient_vs_permanent() {
        let cancel = CancellationToken::new();
        let handle = Supervisor::new(lenient_meltdown())
            .child(
                ChildSpec::permanent("perm", immediate(true)).with_backoff(fast_backoff()),
            )
            .child(
                ChildSpec::permanent("temp", immediate(true))
                    .with_policy(RestartPolicy::Temporary)
                    .with_backoff(fast_backoff()),
            )
            .child(
                ChildSpec::permanent("trans_ok", immediate(true))
                    .with_policy(RestartPolicy::Transient)
                    .with_backoff(fast_backoff()),
            )
            .child(
                ChildSpec::permanent("trans_err", immediate(false))
                    .with_policy(RestartPolicy::Transient)
                    .with_backoff(fast_backoff()),
            )
            .start(cancel.clone());

        // Give the permanent/transient-err children time to restart several times.
        assert!(
            wait_until(Duration::from_secs(2), || handle.restarts("perm").unwrap_or(0) >= 2).await,
            "permanent child should keep restarting after clean exits"
        );

        assert!(handle.restarts("temp").unwrap() == 0, "temporary must not restart");
        assert!(
            handle.restarts("trans_ok").unwrap() == 0,
            "transient + clean exit must not restart"
        );
        assert!(
            handle.restarts("trans_err").unwrap() >= 2,
            "transient + error must restart"
        );

        handle.shutdown().await;
    }

    /// A child that panics a few times is restarted with backoff and then runs healthily.
    #[tokio::test]
    async fn restart_with_backoff() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factory: ServiceFactory = {
            let counter = counter.clone();
            Arc::new(move |cancel| {
                let counter = counter.clone();
                Box::pin(async move {
                    let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    if n <= 3 {
                        panic!("boom {n}");
                    }
                    // Recovered: run until shut down (so no further restarts occur).
                    cancel.cancelled().await;
                    Ok(())
                })
            })
        };

        let backoff = Backoff {
            initial: Duration::from_millis(5),
            max: Duration::from_millis(40),
            factor: 2.0,
        };
        let cancel = CancellationToken::new();
        let started = Instant::now();
        let handle = Supervisor::new(lenient_meltdown())
            .child(ChildSpec::permanent("svc", factory).with_backoff(backoff))
            .start(cancel.clone());

        // Wait until the service recovers past its 3 panics (the 4th invocation).
        assert!(
            wait_until(Duration::from_secs(2), || counter.load(Ordering::SeqCst) >= 4).await,
            "service never recovered past its panics"
        );
        let elapsed = started.elapsed();

        // Exactly three restarts, and backoff (5 + 10 + 20 ms) was applied between them.
        assert_eq!(handle.restarts("svc"), Some(3));
        assert!(
            elapsed >= Duration::from_millis(30),
            "backoff not applied: recovered in {elapsed:?}"
        );
        assert_ne!(
            handle.health("svc"),
            Some(HealthStatus::Unhealthy {
                reason: String::new()
            })
        );
        assert!(matches!(
            handle.health("svc"),
            Some(HealthStatus::Ok) | Some(HealthStatus::Degraded { .. })
        ));

        handle.shutdown().await;
    }

    /// A child that crashes faster than the meltdown threshold is stopped and marked `Unhealthy`.
    #[tokio::test]
    async fn meltdown() {
        let factory: ServiceFactory = Arc::new(|_cancel| Box::pin(async { panic!("always") }));
        let cancel = CancellationToken::new();
        let handle = Supervisor::new(MeltdownPolicy {
            max_restarts: 3,
            window: Duration::from_secs(60),
        })
        .child(ChildSpec::permanent("doomed", factory).with_backoff(fast_backoff()))
        .start(cancel.clone());

        assert!(
            wait_until(Duration::from_secs(2), || matches!(
                handle.health("doomed"),
                Some(HealthStatus::Unhealthy { .. })
            ))
            .await,
            "meltdown did not trip"
        );

        // It stopped restarting after meltdown (count stays put).
        let after = handle.restarts("doomed").unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(handle.restarts("doomed"), Some(after), "child kept restarting after meltdown");

        handle.shutdown().await;
    }

    /// The running host drives seeded sessions through the full lifecycle with no manual pumping.
    #[tokio::test]
    async fn running_host_drives_lifecycle() {
        let store = Arc::new(InMemoryStore::new());
        let host = Host::new(
            store.clone(),
            Arc::new(CoreEngineFactory::delegating()),
            HostConfig::default(),
        );

        let ids: Vec<SessionId> = (0..25).map(|i| SessionId::new(format!("host-{i}"))).collect();
        for id in &ids {
            seed(&store, id).await;
        }

        let handle = host.start();
        let completed = poll_until_completed(&store, &ids, Duration::from_secs(5)).await;
        assert!(completed, "running host failed to complete all sessions");

        handle.shutdown().await;
        assert_eq!(host.manager().active_count(), 0, "active directory above baseline");
    }

    /// THE PHASE-2 GATE: resident dispatchers crash repeatedly during churn, yet the supervisor
    /// restarts them (no meltdown) and every session still drains to completion (durable queues
    /// lose nothing), with the active directory back to baseline.
    #[tokio::test]
    async fn residents_survive_churn() {
        let store = Arc::new(InMemoryStore::new());
        let manager = ActivationManager::new(
            store.clone(),
            Arc::new(CoreEngineFactory::delegating()),
            PartitionId::DEFAULT,
        );

        let wake_budget = Arc::new(AtomicUsize::new(3));
        let job_budget = Arc::new(AtomicUsize::new(2));
        let wake = bounded_panic_tick(wake_tick(manager.clone()), wake_budget.clone());
        let job = bounded_panic_tick(job_tick(manager.clone()), job_budget.clone());

        let cancel = CancellationToken::new();
        let handle = Supervisor::new(MeltdownPolicy {
            max_restarts: 20,
            window: Duration::from_secs(30),
        })
        .child(interval_child(
            "WakeOutboxDispatcher",
            Duration::from_millis(1),
            RestartPolicy::Permanent,
            fast_backoff(),
            wake,
        ))
        .child(interval_child(
            "JobOutboxDispatcher",
            Duration::from_millis(1),
            RestartPolicy::Permanent,
            fast_backoff(),
            job,
        ))
        .child(interval_child(
            "RecoveryScanner",
            Duration::from_millis(5),
            RestartPolicy::Permanent,
            fast_backoff(),
            scan_tick(manager.clone()),
        ))
        .start(cancel.clone());

        let ids: Vec<SessionId> = (0..30).map(|i| SessionId::new(format!("churn-{i}"))).collect();
        for id in &ids {
            seed(&store, id).await;
        }

        let completed = poll_until_completed(&store, &ids, Duration::from_secs(10)).await;
        assert!(completed, "sessions did not complete despite supervised restarts");

        // The injected crashes were consumed and caused real restarts...
        assert_eq!(wake_budget.load(Ordering::SeqCst), 0);
        assert_eq!(job_budget.load(Ordering::SeqCst), 0);
        assert!(handle.restarts("WakeOutboxDispatcher").unwrap() >= 3);
        assert!(handle.restarts("JobOutboxDispatcher").unwrap() >= 2);

        // ...but no service melted down.
        for svc in ["WakeOutboxDispatcher", "JobOutboxDispatcher", "RecoveryScanner"] {
            assert!(
                !matches!(handle.health(svc), Some(HealthStatus::Unhealthy { .. })),
                "{svc} melted down"
            );
        }

        handle.shutdown().await;
        assert_eq!(manager.active_count(), 0, "active directory above baseline");

        // ensure status() import is exercised (sanity on one session)
        assert_eq!(status(&store, &ids[0]).await, Some(SessionStatus::Completed));
    }
}

#[cfg(test)]
mod translation {
    //! THE PHASE-3 GATE: §17 ⇄ management protocol round-trips (`daemon-workspace-layout.md` §7
    //! phase-3 gate). The host presents a real `daemon-core` engine as a `UnitKind::Engine`
    //! [`ManagedUnit`]; driving it with `ManageCommand`s and observing `ManageEvent`s / a
    //! `ManageRequest` exercises the supervision §4 mapping table end to end (host-spec §9,
    //! supervision invariant #7).

    use async_trait::async_trait;
    use daemon_common::{Budget, ReqId, SessionId, UnitId};
    use daemon_core::{DelegateTool, Engine, MockProvider, Provider, SystemPrompt, ToolRegistry};
    use daemon_host::EngineUnit;
    use daemon_supervision::{
        Ack, Concurrency, EndReason, ManageCommand, ManageEvent, ManageRequest,
        ManageRequestHandler, ManageRequestKind, ManageResponse, ManageResponseBody, ManagedUnit,
        ProgressDelta, StartTrigger, UnitKind, WorkRef,
    };
    use std::sync::Arc;
    use std::time::Duration;

    /// Build a managed unit over a real engine driven by `provider`.
    fn engine_unit(provider: Arc<dyn Provider>) -> daemon_host::AgentUnit {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelegateTool::new("background-work")));
        let engine = Engine::fresh(
            SessionId::new("u1"),
            SystemPrompt::new("translation gate engine"),
            provider,
            Arc::new(registry),
        );
        EngineUnit::spawn(UnitId::new("u1"), engine)
    }

    /// `Assign` drives a turn whose §17 events surface as `Started → Progress → Finished` upward.
    #[tokio::test]
    async fn assign_round_trips_to_finished() {
        let unit = engine_unit(Arc::new(MockProvider::completing("all done")));
        assert_eq!(unit.kind(), UnitKind::Engine);
        assert_eq!(unit.id(), UnitId::new("u1"));

        let mut events = unit.events();
        let ack = unit
            .command(ManageCommand::Assign {
                request_id: ReqId(1),
                work: WorkRef::inline("w1", "do the thing"),
                budget: Budget::unlimited(),
            })
            .await;
        assert_eq!(ack, Ack::Accepted);

        let mut saw_started = false;
        let mut saw_progress = false;
        let outcome = loop {
            match tokio::time::timeout(Duration::from_secs(2), events.recv()).await {
                Ok(Ok(ManageEvent::Started {
                    trigger: StartTrigger::Assigned(_),
                    ..
                })) => saw_started = true,
                Ok(Ok(ManageEvent::Progress {
                    delta: ProgressDelta::Text(_),
                    ..
                })) => saw_progress = true,
                Ok(Ok(ManageEvent::Finished { outcome, .. })) => break outcome,
                Ok(Ok(_)) => {}
                Ok(Err(_)) => panic!("event stream closed before Finished"),
                Err(_) => panic!("timed out waiting for ManageEvent::Finished"),
            }
        };

        assert!(saw_started, "no Started{{Assigned}} mapped from TurnStarted");
        assert!(saw_progress, "no Progress{{Text}} mapped from TextDelta");
        assert_eq!(outcome.end_reason, EndReason::Completed);
    }

    /// `Pause`/`Resume`/`Scale` are no-ops at an engine leaf — the partial-downward `Ack::Unsupported`.
    #[tokio::test]
    async fn pause_resume_scale_are_unsupported() {
        let unit = engine_unit(Arc::new(MockProvider::completing("done")));
        assert_eq!(unit.command(ManageCommand::Pause).await, Ack::Unsupported);
        assert_eq!(unit.command(ManageCommand::Resume).await, Ack::Unsupported);
        assert_eq!(
            unit.command(ManageCommand::Scale {
                target: Concurrency(4)
            })
            .await,
            Ack::Unsupported
        );
    }

    /// A blocking §17 `HostRequest` raised inside a turn surfaces upward as a correlated
    /// `ManageRequest` through the installed handler (supervision §2.3 / §4).
    #[tokio::test]
    async fn host_request_maps_to_manage_request() {
        struct Recorder {
            tx: tokio::sync::mpsc::UnboundedSender<ManageRequest>,
        }

        #[async_trait]
        impl ManageRequestHandler for Recorder {
            async fn request(&self, req: ManageRequest) -> ManageResponse {
                let request_id = req.request_id;
                let _ = self.tx.send(req);
                ManageResponse {
                    request_id,
                    body: ManageResponseBody::Delegated(vec![UnitId::new("child-1")]),
                }
            }
        }

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let unit = engine_unit(Arc::new(MockProvider::delegating("delegate", "done")));
        unit.install_request_handler(Arc::new(Recorder { tx }));

        let ack = unit
            .command(ManageCommand::Assign {
                request_id: ReqId(7),
                work: WorkRef::inline("w1", "needs background work"),
                budget: Budget::unlimited(),
            })
            .await;
        assert_eq!(ack, Ack::Accepted);

        let got = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for the escalated ManageRequest")
            .expect("a ManageRequest");
        assert!(
            matches!(got.kind, ManageRequestKind::Delegate(_)),
            "the §17 HostRequest::Delegate did not map to ManageRequestKind::Delegate"
        );
    }
}

#[cfg(test)]
mod journal {
    //! THE PHASE-6b GATE: a verifiable, signed, hash-linked trace journal bound to the durable
    //! incarnation (`daemon-workspace-layout.md` §7 phase-6: "trace-in-envelope"). A unit's
    //! `ManageEvent`s are journaled as Gordian Envelopes, the per-`(session, epoch)` Merkle root is
    //! signed and committed under the checkpoint fence, and verification recomputes the root from
    //! the durable bytes — detecting tampering, proving the cross-epoch chain, and rejecting a stale
    //! incarnation's seal.

    use daemon_common::{ContentHash, JournalStreamId, MerkleRoot, SessionId};
    use daemon_common::{PartitionId, SnapshotBlob, UsageDelta};
    use daemon_host::JournalSink;
    use daemon_store::{InMemoryStore, SessionStore, StoreError, TraceSegment};
    use daemon_supervision::{EndReason, ManageEvent, Outcome, StartTrigger};
    use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, VerifyError, GENESIS_ROOT};
    use std::sync::Arc;

    const PARTITION: PartitionId = PartitionId::DEFAULT;

    fn turn_events() -> Vec<ManageEvent> {
        vec![
            ManageEvent::Started {
                seq: 0,
                trigger: StartTrigger::Resumed,
            },
            ManageEvent::Usage {
                seq: 1,
                delta: UsageDelta {
                    input_tokens: 12,
                    output_tokens: 8,
                    api_calls: 1,
                },
            },
            ManageEvent::Finished {
                seq: 2,
                outcome: Outcome::ended(EndReason::Completed),
            },
        ]
    }

    async fn seed(store: &InMemoryStore, id: &SessionId) {
        store
            .create_session(id.clone(), PARTITION, SnapshotBlob::default())
            .await
            .unwrap();
    }

    fn input_from<'a>(
        stream: &'a JournalStreamId,
        segment: u64,
        prior: MerkleRoot,
        entries: &'a [(u64, Vec<u8>, ContentHash)],
    ) -> SegmentInput<'a> {
        SegmentInput {
            stream,
            segment,
            prior,
            entries,
        }
    }

    fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
        seg.entries
            .iter()
            .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
            .collect()
    }

    /// Record a turn's events, seal the signed root under the fence, and verify it from the store.
    /// Then prove tampering with any persisted entry is detected.
    #[tokio::test]
    async fn record_commit_verify_and_tamper_detect() {
        let store = Arc::new(InMemoryStore::new());
        let id = SessionId::new("journal-happy");
        seed(&store, &id).await;
        let fence = store.acquire_activation_lease(&id).await.unwrap();

        let signer = Arc::new(TraceSigner::generate());
        let stream = JournalStreamId::session(&id);
        let sink = JournalSink::for_incarnation(
            store.clone() as Arc<dyn SessionStore>,
            signer.clone(),
            stream.clone(),
            fence,
            0,
        )
        .await;

        for ev in turn_events() {
            sink.record(&ev).await.unwrap();
        }
        let root = sink.seal().await.unwrap();

        // Load the sealed segment from the store and verify it end to end.
        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        let committed = seg.committed.clone().expect("segment sealed");
        assert_eq!(committed.root, root);
        let entries = loaded_entries(&seg);
        verify_segment(
            &input_from(&stream, 0, GENESIS_ROOT, &entries),
            &committed.root,
            &committed.signature,
            &signer.verifying_key(),
        )
        .expect("a faithfully sealed segment verifies from the store");

        // Tamper: mutate a persisted entry's bytes; verification must fail.
        let mut tampered = entries.clone();
        tampered[1].1[0] ^= 0xFF;
        let err = verify_segment(
            &input_from(&stream, 0, GENESIS_ROOT, &tampered),
            &committed.root,
            &committed.signature,
            &signer.verifying_key(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                VerifyError::Decode
                    | VerifyError::ContentHashMismatch
                    | VerifyError::RootMismatch
            ),
            "tampering must be detected, got {err:?}"
        );
    }

    /// Epoch N+1's root chains onto epoch N's: a sink opened `chained` onto the prior root produces
    /// a root that only verifies with that prior (a broken link is a root mismatch).
    #[tokio::test]
    async fn cross_epoch_chain() {
        let store = Arc::new(InMemoryStore::new());
        let id = SessionId::new("journal-chain");
        seed(&store, &id).await;

        // Epoch 0.
        let f0 = store.acquire_activation_lease(&id).await.unwrap();
        let signer = Arc::new(TraceSigner::generate());
        let stream = JournalStreamId::session(&id);
        let sink0 = JournalSink::for_incarnation(
            store.clone() as Arc<dyn SessionStore>,
            signer.clone(),
            stream.clone(),
            f0,
            0,
        )
        .await;
        for ev in turn_events() {
            sink0.record(&ev).await.unwrap();
        }
        let root0 = sink0.seal().await.unwrap();

        // Epoch 1 chains onto epoch 0's sealed root (loaded from the store), under a fresh
        // (superseding) fence.
        let f1 = store.acquire_activation_lease(&id).await.unwrap();
        let sink1 = JournalSink::for_incarnation(
            store.clone() as Arc<dyn SessionStore>,
            signer.clone(),
            stream.clone(),
            f1,
            1,
        )
        .await;
        for ev in turn_events() {
            sink1.record(&ev).await.unwrap();
        }
        let root1 = sink1.seal().await.unwrap();

        let seg1 = store.load_trace_segment(&stream, 1).await.unwrap();
        let committed1 = seg1.committed.clone().unwrap();
        let entries1 = loaded_entries(&seg1);

        // Verifies with the true prior (root0)...
        verify_segment(
            &input_from(&stream, 1, root0, &entries1),
            &committed1.root,
            &committed1.signature,
            &signer.verifying_key(),
        )
        .expect("the chained segment verifies with the correct prior");
        assert_eq!(committed1.root, root1);

        // ...and a broken link (wrong prior) is rejected.
        assert_eq!(
            verify_segment(
                &input_from(&stream, 1, GENESIS_ROOT, &entries1),
                &committed1.root,
                &committed1.signature,
                &signer.verifying_key(),
            )
            .unwrap_err(),
            VerifyError::RootMismatch
        );
    }

    /// A stale incarnation cannot seal a segment root: the commit is fenced exactly as a checkpoint
    /// would be (reuses acceptance tests #4/#6, now for the trace root).
    #[tokio::test]
    async fn stale_fence_cannot_seal() {
        let store = Arc::new(InMemoryStore::new());
        let id = SessionId::new("journal-fenced");
        seed(&store, &id).await;

        let stale = store.acquire_activation_lease(&id).await.unwrap();
        // Ownership transfers to a newer incarnation.
        let _current = store.acquire_activation_lease(&id).await.unwrap();

        let signer = Arc::new(TraceSigner::generate());
        let stream = JournalStreamId::session(&id);
        let sink = JournalSink::for_incarnation(
            store.clone() as Arc<dyn SessionStore>,
            signer,
            stream.clone(),
            stale,
            0,
        )
        .await;
        for ev in turn_events() {
            // Appends are not fenced (the open log), but the seal is.
            sink.record(&ev).await.unwrap();
        }
        let r = sink.seal().await;
        assert!(
            matches!(r, Err(StoreError::Fenced { .. })),
            "a stale incarnation must not seal a segment root, got {r:?}"
        );
        // No root was committed.
        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        assert!(seg.committed.is_none());
    }
}

#[cfg(test)]
mod orchestration {
    //! THE PHASE-4 GATE: orchestration (`daemon-workspace-layout.md` §7 phase-4 gate). One engine
    //! delegates to a child through the fleet runtime; the child's `ManageEvent`s fan in to fleet
    //! state; the child's completion wakes the parent as a `BackgroundCompletion`; and a child's
    //! `ManageRequest` is answered by policy or escalated upward (synthesis §3.1; layout §4).
    //!
    //! The runtime is core-free: it drives children only through `daemon_supervision::ManagedUnit`
    //! and the durable store. Child construction is the injected `ChildSpawner` — here the
    //! engine-backed spawner wiring `daemon-core` + `daemon_host::EngineUnit`, exactly as
    //! `bins/daemon` will. The fleet worker replaces the substrate's placeholder echo worker, so the
    //! cycle is driven explicitly (never via `run_workers`/`recover`).

    use async_trait::async_trait;
    use daemon_activation::ActivationManager;
    use daemon_common::{PartitionId, ReqId, SessionId, UnitId};
    use daemon_core::{Engine, MockProvider, Provider, Snapshot, SystemPrompt, ToolRegistry};
    use daemon_host::{CoreEngineFactory, EngineUnit};
    use daemon_orchestration::{ChildSpawner, ChildStatus, DefaultAnswerPolicy, FleetRuntime};
    use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
    use daemon_supervision::{
        ApprovalReq, DelegationSpec, EndReason, EscalationReq, ManageRequest, ManageRequestHandler,
        ManageRequestKind, ManageResponse, ManageResponseBody, ManagedUnit,
    };
    use daemon_tool_orchestrate::OrchestrateTool;
    use std::sync::Arc;

    const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// The injected placement seam: materialize a child as an engine-backed `ManagedUnit`. A
    /// completing provider finishes the child in one turn (no further delegation).
    struct EngineChildSpawner;

    #[async_trait]
    impl ChildSpawner for EngineChildSpawner {
        async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
            let engine = Engine::fresh(
                SessionId::new(id.as_str()),
                SystemPrompt::new("fleet child"),
                Arc::new(MockProvider::completing("child done")),
                Arc::new(ToolRegistry::new()),
            );
            Arc::new(EngineUnit::spawn(id, engine))
        }
    }

    /// Build a fleet runtime over `store` at the default partition, with an optional supervisor.
    fn fleet_runtime(
        store: Arc<InMemoryStore>,
        parent: Option<Arc<dyn ManageRequestHandler>>,
    ) -> FleetRuntime {
        FleetRuntime::new(
            store,
            PARTITION,
            Arc::new(EngineChildSpawner),
            Arc::new(DefaultAnswerPolicy),
            parent,
        )
    }

    /// An orchestrating parent: a `CoreEngineFactory` whose engine offers the orchestrate tool and a
    /// provider that delegates through it once, then completes.
    fn orchestrating_manager(store: Arc<InMemoryStore>, fleet: FleetRuntime) -> ActivationManager {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(OrchestrateTool::new(fleet)));
        let factory = CoreEngineFactory::with_provider(
            Arc::new(|| {
                Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>
            }),
            Arc::new(registry),
            SystemPrompt::new("parent orchestrator"),
        );
        ActivationManager::new(store, Arc::new(factory), PARTITION)
    }

    async fn seed(store: &InMemoryStore, id: &SessionId) {
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode snapshot");
        store
            .create_session(id.clone(), PARTITION, blob)
            .await
            .expect("create session");
    }

    /// One engine delegates to a child; the fleet spawns + drives it, folds its events, and the
    /// child's completion wakes the parent to completion.
    #[tokio::test]
    async fn engine_delegates_child_completes_via_fleet() {
        let store = Arc::new(InMemoryStore::new());
        let fleet = fleet_runtime(store.clone(), None);
        let mgr = orchestrating_manager(store.clone(), fleet.clone());

        // The parent's first turn delegates through the orchestrate tool and suspends.
        let parent = SessionId::new("parent");
        seed(&store, &parent).await;
        mgr.wake(parent.clone()).await.expect("wake parent");
        assert!(
            matches!(
                store.status(&parent).await,
                Some(SessionStatus::Suspended { .. })
            ),
            "parent should suspend on the delegation job"
        );

        // The fleet worker — not the substrate echo — spawns and drives the child.
        let processed = fleet.process_jobs_once().await.expect("process jobs");
        assert_eq!(processed, 1, "exactly one delegation job processed");

        // Fan-in: one child, Finished/Completed, usage folded, recorded as a real durable session.
        let children = fleet.children();
        assert_eq!(children.len(), 1, "one child spawned");
        let child = &children[0];
        assert_eq!(fleet.child_status(child), Some(ChildStatus::Finished));
        assert_eq!(
            fleet.child_outcome(child).expect("outcome").end_reason,
            EndReason::Completed
        );
        assert!(
            fleet.fleet_usage().api_calls > 0,
            "child usage did not fan in to fleet state"
        );
        assert_eq!(
            store.status(&SessionId::new(child.as_str())).await,
            Some(SessionStatus::Completed),
            "the child should be a real Completed session in the store"
        );

        // The recorded completion woke the parent: dispatch it and the parent resumes to completion.
        mgr.dispatch_wakes().await.expect("dispatch wakes");
        assert_eq!(
            store.status(&parent).await,
            Some(SessionStatus::Completed),
            "the child's completion should wake the parent to completion"
        );
    }

    /// A child's blocking request is answered by policy, and an unresolvable one escalates — to the
    /// runtime's supervisor when present, else unhandled at the root (the answer-authority chain).
    #[tokio::test]
    async fn child_request_is_answered_and_escalated() {
        let store = Arc::new(InMemoryStore::new());
        let fleet = fleet_runtime(store.clone(), None);
        let handler = fleet.request_handler();

        // Answered by policy: an approval is granted.
        let resp = handler
            .request(ManageRequest {
                request_id: ReqId(1),
                kind: ManageRequestKind::Approval(ApprovalReq {
                    prompt: "run it?".into(),
                }),
            })
            .await;
        assert_eq!(resp.body, ManageResponseBody::Approved(true));
        assert_eq!(fleet.request_log().len(), 1, "the child request was logged");

        // Escalated at the root: no supervisor to re-raise to.
        let resp = handler
            .request(ManageRequest {
                request_id: ReqId(2),
                kind: ManageRequestKind::Escalate(EscalationReq {
                    reason: "cannot resolve locally".into(),
                }),
            })
            .await;
        assert_eq!(resp.body, ManageResponseBody::Escalated(false));

        // Escalated upward: with a supervisor installed, the request is re-raised and answered there.
        struct Recorder {
            tx: tokio::sync::mpsc::UnboundedSender<ManageRequest>,
        }
        #[async_trait]
        impl ManageRequestHandler for Recorder {
            async fn request(&self, req: ManageRequest) -> ManageResponse {
                let request_id = req.request_id;
                let _ = self.tx.send(req);
                ManageResponse {
                    request_id,
                    body: ManageResponseBody::Escalated(true),
                }
            }
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let escalating = fleet_runtime(store.clone(), Some(Arc::new(Recorder { tx })));
        let resp = escalating
            .request_handler()
            .request(ManageRequest {
                request_id: ReqId(3),
                kind: ManageRequestKind::Escalate(EscalationReq {
                    reason: "raise me".into(),
                }),
            })
            .await;
        assert_eq!(resp.body, ManageResponseBody::Escalated(true));
        assert!(
            rx.recv().await.is_some(),
            "the escalation should reach the installed supervisor"
        );
    }
}

#[cfg(test)]
mod credentials {
    //! THE PHASE-7 GATE: the credential authority (`daemon-workspace-layout.md` §7 phase-7 gate).
    //! A placed child several cuts down runs under only a brokered, attenuated, short-lived
    //! capability lease — no raw secret crosses the cut. The authority (owner) mints signed,
    //! scoped, TTL-bounded `CapabilityLease`s; intermediate hosts re-broker upward, intersecting
    //! scope at each hop (least privilege); the three modes (`Native`/`Bearer`/`Proxied`) trade
    //! isolation against cost; a stale incarnation cannot acquire; an expired/edited capability is
    //! refused; every lifecycle step is journaled into the phase-6 verifiable trace and verifies;
    //! and a cost ceiling feeds back into `Budget`.

    use daemon_common::{
        ContentHash, CredError, CredMode, CredScope, FenceToken, JournalStreamId, PartitionId,
        ProfileRef, SessionId, SnapshotBlob, UnitId,
    };
    use daemon_credentials::{
        CapabilitySigner, CredAuditKind, CredentialAuthority, StubCredentialSource,
    };
    use daemon_host::{
        serve_credentials, CredentialBroker, FenceGuard, JournalSink, OwnerBroker, RelayBroker,
        RemoteCredentialClient,
    };
    use daemon_provision::CutChannel;
    use daemon_store::{InMemoryStore, SessionStore, TraceSegment};
    use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, GENESIS_ROOT};
    use std::sync::{Arc, Mutex};

    const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// A connected pair of in-process cut channels (parent end, child end) over a duplex pipe — a
    /// cut without spawning a process, so the broker chain is exercised over the real frame codec.
    fn cut_pair() -> (CutChannel, CutChannel) {
        let (a, b) = tokio::io::duplex(1 << 16);
        let (ar, aw) = tokio::io::split(a);
        let (br, bw) = tokio::io::split(b);
        (
            CutChannel::from_parts(Box::new(ar), Box::new(aw)),
            CutChannel::from_parts(Box::new(br), Box::new(bw)),
        )
    }

    /// Build the 2-hop chain A -> B -> C over two real cuts. A is the owner (mints); B is a relay
    /// granting at most `grant_b` (optionally fenced); C gets the descendant-side client. Returns
    /// `(client_at_C, authority_A)`.
    fn build_chain(
        mode: CredMode,
        grant_a: CredScope,
        grant_b: CredScope,
        fence_b: Option<FenceGuard>,
    ) -> (Arc<RemoteCredentialClient>, Arc<CredentialAuthority>) {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::minting("openai", "sk-configured"));
        let authority = Arc::new(CredentialAuthority::new(grant_a, mode, 60_000, signer, source));

        // Cut A<->B: A serves as the owner.
        let (a_parent, a_child) = cut_pair();
        let owner = Arc::new(OwnerBroker::new(authority.clone())) as Arc<dyn CredentialBroker>;
        tokio::spawn(serve_credentials(a_parent, owner));
        let client_to_a = RemoteCredentialClient::connect(a_child); // lives at B

        // Cut B<->C: B re-brokers upward as a relay (narrowing by grant_b).
        let mut relay = RelayBroker::new(client_to_a as Arc<dyn CredentialBroker>, grant_b);
        if let Some(f) = fence_b {
            relay = relay.with_fence(f);
        }
        let relay = Arc::new(relay) as Arc<dyn CredentialBroker>;
        let (b_parent, b_child) = cut_pair();
        tokio::spawn(serve_credentials(b_parent, relay));
        let client_to_b = RemoteCredentialClient::connect(b_child); // lives at C

        (client_to_b, authority)
    }

    fn unit_c() -> Option<UnitId> {
        Some(UnitId::new("unit-C"))
    }

    fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
        seg.entries
            .iter()
            .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
            .collect()
    }

    /// (1) + (3): re-brokering composes across two cuts, and the effective scope is the
    /// intersection along the whole path — a descendant can never exceed `grant_A ∩ grant_B`.
    #[tokio::test]
    async fn two_hop_chain_composes_and_attenuates() {
        let grant_a = CredScope::new(["openai"], ["chat", "embed"], Some(1_000));
        let grant_b = CredScope::new(["openai"], ["chat"], Some(500));
        let (c, _auth) = build_chain(CredMode::Native, grant_a, grant_b, None);

        // C asks for *more* than either hop grants: extra profile/actions, a bigger ceiling.
        let broad = CredScope::new(["openai", "anthropic"], ["chat", "embed", "admin"], Some(10_000));
        let lease = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &broad)
            .await
            .expect("the chain mints a capability for C");

        // Effective scope = grant_A ∩ grant_B ∩ request.
        assert!(lease.scope.profiles.contains("openai"));
        assert!(!lease.scope.profiles.contains("anthropic"));
        assert!(lease.scope.actions.contains("chat"));
        assert!(!lease.scope.actions.contains("embed"), "embed is not in grant_B");
        assert!(!lease.scope.actions.contains("admin"), "admin is in no grant");
        assert_eq!(lease.scope.max_tokens, Some(500), "ceiling clamps to the tightest hop");
        assert!(lease.secret.is_some(), "Native carries a short-lived token");

        // A request with no overlap is denied at the narrowing hop (never forwarded to the owner).
        let off_grant = CredScope::new(["ghost"], ["chat"], None);
        let err = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &off_grant)
            .await
            .unwrap_err();
        assert_eq!(err, CredError::ScopeDenied);
    }

    /// (2 Proxied) + (1 Proxied): the lease C holds is a handle (no secret); resolution must
    /// round-trip to the owner A, which returns only a *result* — the raw key never crosses to B/C.
    #[tokio::test]
    async fn proxied_use_round_trips_to_owner_without_leaking_key() {
        let grant = CredScope::new(["openai"], ["chat"], None);
        let (c, auth) = build_chain(CredMode::Proxied, grant.clone(), grant.clone(), None);

        let lease = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .unwrap();
        assert!(lease.secret.is_none(), "Proxied hands C only a handle");

        let result = c.use_capability(unit_c(), &lease).await.unwrap();
        assert_ne!(result.expose(), "sk-configured", "the raw key must never reach C/B");
        assert!(result.expose().starts_with("proxied-result:"), "owner returns a result");

        // The owner recorded the use (the round-trip reached A).
        assert!(
            auth.audit_log().iter().any(|e| e.kind == CredAuditKind::Use),
            "the proxied use must be audited at the owner"
        );
    }

    /// (2 Bearer): a long-lived-key profile hands over a usable key; the compensating control is
    /// the mandatory audit record. With a minting source the key is fresh per-grant.
    #[tokio::test]
    async fn bearer_hands_over_key_and_is_audited() {
        let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
        let (c, auth) = build_chain(CredMode::Bearer, grant.clone(), grant.clone(), None);

        let lease = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .unwrap();
        let key = lease.secret.as_ref().expect("Bearer carries a usable key").expose();
        assert!(key.starts_with("sk-fresh-"), "a minting source issues a fresh per-grant key");

        let granted = auth
            .audit_log()
            .into_iter()
            .find(|e| e.kind == CredAuditKind::Grant)
            .expect("the issuance is audited");
        assert_eq!(granted.requester, unit_c(), "the audit answers *who* was issued the key");
    }

    /// (4): a stale incarnation cannot acquire — the superseded hop (here the relay B) rejects with
    /// `Fenced`, exactly as the dual-ownership store fence does across a cut.
    #[tokio::test]
    async fn stale_fence_acquire_is_rejected() {
        let grant = CredScope::new(["openai"], ["chat"], None);
        let live = Arc::new(Mutex::new(FenceToken(1)));
        let guard = FenceGuard::new(FenceToken(1), live.clone());
        let (c, _auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), Some(guard));

        // While B's incarnation is current, acquire succeeds.
        c.acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .expect("current incarnation acquires");

        // A newer activation supersedes B.
        *live.lock().unwrap() = FenceToken(2);
        let err = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .unwrap_err();
        assert_eq!(err, CredError::Fenced, "the superseded hop must reject the acquire");
    }

    /// (5): a capability whose signed fields were edited fails verification (signature), and a
    /// zero-TTL capability fails verification (expiry).
    #[tokio::test]
    async fn edited_and_expired_capabilities_are_refused() {
        let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
        let (c, auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), None);
        let mut lease = c
            .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .unwrap();
        auth.verify(&lease).expect("a freshly minted capability verifies");

        // Tamper with a signed field: verification fails.
        lease.scope.max_tokens = Some(999_999);
        assert_eq!(auth.verify(&lease).unwrap_err(), CredError::BadSignature);

        // A zero-TTL authority mints an already-expired capability.
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::new("openai", "sk"));
        let auth0 = CredentialAuthority::new(grant.clone(), CredMode::Native, 0, signer, source);
        let ctx = daemon_credentials::AcquireCtx::default();
        let lease0 = auth0
            .acquire(&ctx, &ProfileRef::new("openai"), &grant)
            .unwrap();
        assert_eq!(auth0.verify(&lease0).unwrap_err(), CredError::Expired);
    }

    /// (6): the credential audit trail is journaled into the phase-6 verifiable trace and verifies
    /// end-to-end — the sealed, signed segment is the tamper-evident answer to "who requested which
    /// credential, when."
    #[tokio::test]
    async fn audit_trail_is_journaled_and_verifies() {
        let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
        let (c, auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), None);
        c.acquire(unit_c(), &ProfileRef::new("openai"), &grant)
            .await
            .unwrap();

        // Journal A's credential audit log into a sealed, signed trace segment.
        let store = Arc::new(InMemoryStore::new());
        let id = SessionId::new("cred-audit");
        store
            .create_session(id.clone(), PARTITION, SnapshotBlob::default())
            .await
            .unwrap();
        let fence = store.acquire_activation_lease(&id).await.unwrap();
        let tsigner = Arc::new(TraceSigner::generate());
        let stream = JournalStreamId::session(&id);
        let sink = JournalSink::for_incarnation(
            store.clone() as Arc<dyn SessionStore>,
            tsigner.clone(),
            stream.clone(),
            fence,
            0,
        )
        .await;

        let events = auth.audit_log();
        assert!(events.iter().any(|e| e.kind == CredAuditKind::Request));
        assert!(events.iter().any(|e| e.kind == CredAuditKind::Grant));
        for ev in &events {
            sink.record_credential(ev).await.unwrap();
        }
        let root = sink.seal().await.unwrap();

        let seg = store.load_trace_segment(&stream, 0).await.unwrap();
        let committed = seg.committed.clone().expect("segment sealed");
        assert_eq!(committed.root, root);
        let entries = loaded_entries(&seg);
        verify_segment(
            &SegmentInput {
                stream: &stream,
                segment: 0,
                prior: GENESIS_ROOT,
                entries: &entries,
            },
            &committed.root,
            &committed.signature,
            &tsigner.verifying_key(),
        )
        .expect("the sealed credential-audit segment verifies end to end");

        // The journaled detail carries the requester (the "who").
        let grant_ev = events
            .iter()
            .find(|e| e.kind == CredAuditKind::Grant)
            .unwrap();
        assert!(grant_ev.summary().contains("unit-C"));
    }

    /// (7): a fleet cost ceiling feeds back into `Budget` — under the ceiling there is headroom,
    /// once reached the budget is throttled to zero (which a supervisor enforces as a cap).
    #[tokio::test]
    async fn cost_ceiling_feeds_budget() {
        let signer = Arc::new(CapabilitySigner::generate());
        let source = Arc::new(StubCredentialSource::new("openai", "sk"));
        let auth = CredentialAuthority::new(
            CredScope::new(["openai"], ["chat"], Some(1_000)),
            CredMode::Native,
            60_000,
            signer,
            source,
        )
        .with_cost_ceiling(100);

        assert_eq!(auth.charge(60).tokens, Some(40), "headroom remains under the ceiling");
        assert_eq!(auth.charge(60).tokens, Some(0), "throttled once the ceiling is reached");
        assert_eq!(auth.spent_tokens(), 120);
    }
}

#[cfg(test)]
mod node_interface {
    //! THE PHASE-8 CONTROL-SURFACE GATE (`daemon-workspace-layout.md` §7 phase-8 gate). The node is
    //! assembled exactly as `bins/daemon` does (durable substrate + fleet-as-job-worker + the live
    //! session surface) and driven through the one [`daemon_api`] surface over two transports: the
    //! in-process trait call and the Unix socket. The gate proves the surface is transport-agnostic:
    //! a session assigned over the socket is driven to `Completed` by the real `FleetRuntime` job
    //! worker, the fleet usage folds in, and the in-process and socket reads agree.
    //!
    //! The session sub-surface's cross-language twin (the C FFI driving `StartTurn -> TurnFinished`)
    //! is proven by the `bindings/daemon-core-ffi` C harness, not here.

    use daemon_api::{ApiRequest, ApiResponse, ControlApi, SessionState};
    use daemon_common::{PartitionId, ProfileRef, SessionId};
    use daemon_core::{MockProvider, Provider, ProviderRegistry};
    use daemon_host::{serve_api_unix, ApiClient, HostConfig, NodeApiImpl};
    use daemon_node::{assemble as assemble_node, AssembledNode, NodeAssembly};
    use daemon_store::InMemoryStore;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::net::UnixListener;

    const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// Assemble a node through the shared composition root ([`daemon_node::assemble`]) — exactly as
    /// `bins/daemon`'s host role does — with the gate's mock providers (an orchestrator that
    /// delegates once, completing children, and a completing session default). Returns the in-process
    /// surface and the started resident-service handle.
    fn assemble() -> (Arc<NodeApiImpl>, daemon_host::SupervisorHandle) {
        let mut providers = ProviderRegistry::new();
        providers.set_default(Arc::new(|| {
            Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
        }));
        providers.register(
            "orchestrator",
            Arc::new(|| {
                Arc::new(MockProvider::delegating("orchestrate", "fleet done")) as Arc<dyn Provider>
            }),
        );
        providers.register(
            "child",
            Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
        );

        let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
            store: Arc::new(InMemoryStore::new()),
            partition: PARTITION,
            host_config: HostConfig {
                partition: PARTITION,
                ..HostConfig::default()
            },
            providers,
            credentials: None,
            profile: ProfileRef::new("openai"),
            engine_config: daemon_core::Config::default(),
            journal_seed: Some([0x11; 32]),
        });
        (node, handle)
    }

    fn temp_socket() -> std::path::PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("daemon-api-gate-{}-{}.sock", std::process::id(), n))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_surface_is_transport_agnostic_and_drives_a_session_to_completion() {
        let (node, handle) = assemble();

        // Serve the same surface over a Unix socket.
        let path = temp_socket();
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind api socket");
        let server = tokio::spawn(serve_api_unix(listener, node.clone()));
        let client = ApiClient::new(path.clone());

        // Health over the socket: the four resident services are present.
        let health = match client.call(ApiRequest::Health).await.unwrap() {
            ApiResponse::Health(h) => h,
            other => panic!("expected Health, got {other:?}"),
        };
        assert!(
            health.services.len() >= 4,
            "expected the resident-service tree, got {:?}",
            health.services
        );

        // Assign a durable session over the socket and drive it to Completed via the real fleet
        // job worker (the resident JobOutboxDispatcher), polling the control surface.
        let session = SessionId::new("op-session");
        assert!(matches!(
            client
                .call(ApiRequest::Assign {
                    session: session.clone()
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let resp = client.call(ApiRequest::Sessions).await.unwrap();
            if let ApiResponse::Sessions(list) = resp {
                if list
                    .iter()
                    .any(|i| i.session == session && i.state == SessionState::Completed)
                {
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "the assigned session never reached Completed"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // A delegation child ran: fleet usage folded in (the §7 fan-in).
        let fleet = match client.call(ApiRequest::Fleet).await.unwrap() {
            ApiResponse::Fleet(f) => f,
            other => panic!("expected Fleet, got {other:?}"),
        };
        assert!(
            fleet.usage.api_calls > 0 && !fleet.children.is_empty(),
            "expected a delegation child to have run and folded usage, got {fleet:?}"
        );

        // Transport parity: the in-process trait call and the socket round-trip agree.
        let inproc_health = node.health().await;
        let socket_health = match client.call(ApiRequest::Health).await.unwrap() {
            ApiResponse::Health(h) => h,
            other => panic!("expected Health, got {other:?}"),
        };
        assert_eq!(
            inproc_health.all_ok, socket_health.all_ok,
            "health all_ok must agree across transports"
        );
        assert_eq!(
            sorted_names(&inproc_health),
            sorted_names(&socket_health),
            "the service set must agree across transports"
        );

        let inproc_sessions = node.sessions().await;
        let socket_sessions = match client.call(ApiRequest::Sessions).await.unwrap() {
            ApiResponse::Sessions(list) => list,
            other => panic!("expected Sessions, got {other:?}"),
        };
        assert!(
            inproc_sessions
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed)
                && socket_sessions
                    .iter()
                    .any(|i| i.session == session && i.state == SessionState::Completed),
            "both transports must observe the completed session"
        );

        server.abort();
        handle.shutdown().await;
        let _ = std::fs::remove_file(&path);
    }

    fn sorted_names(h: &daemon_api::HealthReport) -> Vec<String> {
        let mut names: Vec<String> = h.services.iter().map(|s| s.name.clone()).collect();
        names.sort();
        names
    }

    /// The tree-aware control surface (the GUI's real surface) is transport-agnostic: `tree`/`unit`/
    /// `unit_events` and the lifecycle ops agree in-process and over the socket.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tree_surface_is_transport_agnostic() {
        use daemon_api::ApiError;

        let (node, handle) = assemble();
        let path = temp_socket();
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind api socket");
        let server = tokio::spawn(serve_api_unix(listener, node.clone()));
        let client = ApiClient::new(path.clone());

        // Drive a delegation child to completion so the tree has a unit to project.
        let session = SessionId::new("tree-op");
        assert!(matches!(
            client
                .call(ApiRequest::Assign {
                    session: session.clone()
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let ApiResponse::Sessions(list) = client.call(ApiRequest::Sessions).await.unwrap() {
                if list
                    .iter()
                    .any(|i| i.session == session && i.state == SessionState::Completed)
                {
                    break;
                }
            }
            assert!(Instant::now() < deadline, "the assigned session never completed");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // tree() parity + the fleet child presents as an Engine leaf.
        let inproc_tree = node.tree().await;
        let socket_tree = match client.call(ApiRequest::Tree).await.unwrap() {
            ApiResponse::Tree(t) => t,
            other => panic!("expected Tree, got {other:?}"),
        };
        assert_eq!(inproc_tree, socket_tree, "tree must agree across transports");
        assert!(
            !socket_tree.nodes.is_empty(),
            "expected at least one unit in the tree"
        );
        let child = socket_tree.nodes[0].clone();
        assert_eq!(
            child.kind,
            daemon_api::UnitKind::Engine,
            "a fleet child is an Engine leaf"
        );

        // unit() parity.
        let inproc_unit = node.unit(child.id.clone()).await;
        let socket_unit = match client
            .call(ApiRequest::Unit {
                unit: child.id.clone(),
            })
            .await
            .unwrap()
        {
            ApiResponse::Unit(u) => u,
            other => panic!("expected Unit, got {other:?}"),
        };
        assert_eq!(inproc_unit, socket_unit, "unit view must agree across transports");
        assert!(socket_unit.is_some(), "the child unit should resolve");

        // unit_events() parity: the child emitted at least Started + Finished views.
        let inproc_events = node.unit_events(child.id.clone(), 0).await;
        let socket_events = match client
            .call(ApiRequest::UnitEvents {
                unit: child.id.clone(),
                max: 0,
            })
            .await
            .unwrap()
        {
            ApiResponse::UnitEvents(e) => e,
            other => panic!("expected UnitEvents, got {other:?}"),
        };
        assert_eq!(
            inproc_events, socket_events,
            "unit events must agree across transports"
        );
        assert!(
            !socket_events.is_empty(),
            "expected buffered drill-down events for the child"
        );

        // Lifecycle parity: an engine leaf does not support pause/resume/scale — identically on both
        // transports (the surface is meaningful for orchestrator sub-fleets).
        for (req, label) in [
            (
                ApiRequest::Pause {
                    unit: child.id.clone(),
                },
                "pause",
            ),
            (
                ApiRequest::Resume {
                    unit: child.id.clone(),
                },
                "resume",
            ),
            (
                ApiRequest::Scale {
                    unit: child.id.clone(),
                    n: 2,
                },
                "scale",
            ),
        ] {
            let socket = client.call(req).await.unwrap();
            assert!(
                matches!(socket, ApiResponse::Error(ApiError::Unsupported(_))),
                "{label} should be Unsupported over the socket, got {socket:?}"
            );
        }
        assert!(node.pause(child.id.clone()).await.is_err());
        assert!(node.resume(child.id.clone()).await.is_err());
        assert!(node.scale(child.id.clone(), 2).await.is_err());

        server.abort();
        handle.shutdown().await;
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn session_surface_runs_an_interactive_turn_to_finished() {
        use daemon_api::{Outbound, SessionApi};
        use daemon_common::ReqId;
        use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};

        let (node, handle) = assemble();
        let session = SessionId::new("live-1");

        // Open + run a turn on the live session sub-surface (the same surface the FFI wraps).
        node.submit(
            session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hello"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("submit StartTurn");

        // Drain events until TurnFinished arrives.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finished = false;
        while Instant::now() < deadline {
            let drained = node.poll(session.clone(), 0).await.expect("poll");
            if drained.iter().any(|o| {
                matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. }))
            }) {
                finished = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(finished, "the interactive turn never reached TurnFinished");

        handle.shutdown().await;
    }

    /// THE THREAD-C GATE: reconnect + scroll-back through durable, verified history. After an
    /// interactive turn seals into the unified verifiable journal, the session's history is read
    /// back through the (non-destructive) `session_history` surface — independent of the live drain,
    /// exactly as a reconnecting client sees it. The coalesced assistant message is present, the
    /// whole sealed chain verifies under the node's published verifying key, and the read is
    /// non-destructive (a second read returns the same page).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconnect_reads_back_verified_session_history() {
        use daemon_api::{ControlApi, JournalRecordPayload, Outbound, SessionApi};
        use daemon_common::ReqId;
        use daemon_protocol::{AgentCommand, AgentEvent, TranscriptBlock, UserMsg};

        let (node, handle) = assemble();
        let session = SessionId::new("history-1");

        // Drive an interactive turn to TurnFinished (the live path journals + seals per turn).
        node.submit(
            session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hello"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("submit StartTurn");
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut finished = false;
        while Instant::now() < deadline {
            let drained = node.poll(session.clone(), 0).await.expect("poll");
            if drained
                .iter()
                .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
            {
                finished = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(finished, "the interactive turn never reached TurnFinished");

        // Scroll back through durable history — non-destructive and independent of the live drain
        // (the seal may land just after TurnFinished drains, so retry until the page appears).
        let mut page = None;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let p = node.session_history(session.clone(), 0, 0).await;
            if !p.entries.is_empty() {
                page = Some(p);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let page = page.expect("durable history should appear after the turn seals");

        // The whole sealed chain verifies, and the coalesced assistant message is present.
        assert!(
            page.entries.iter().all(|e| e.verified),
            "every sealed entry must verify under the node key: {page:?}"
        );
        assert!(
            page.entries.iter().any(|e| matches!(
                &e.payload,
                JournalRecordPayload::Block {
                    block: TranscriptBlock::Message { .. }
                }
            )),
            "expected a coalesced assistant message block, got {page:?}"
        );

        // Non-destructive: a repeat read from the same cursor returns the same entries.
        let again = node.session_history(session.clone(), 0, 0).await;
        assert_eq!(again.entries, page.entries, "history read must be non-destructive");

        // The node publishes its verifying key so an auditor can verify the chain offline.
        let key = node.verifying_key().await;
        assert!(
            key.map(|k| !k.is_empty()).unwrap_or(false),
            "the node must publish a journal verifying key"
        );

        handle.shutdown().await;
    }

    /// Steer / Snapshot / Interrupt drive over the Unix socket, and the snapshot projection agrees
    /// with the in-process transport (the phase-9 control-surface parity gate).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn steer_snapshot_interrupt_drive_over_socket_with_parity() {
        use daemon_api::{Outbound, SessionApi};
        use daemon_common::ReqId;
        use daemon_protocol::{AgentCommand, AgentEvent, ConvView, UserMsg};

        let (node, handle) = assemble();
        let path = temp_socket();
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind api socket");
        let server = tokio::spawn(serve_api_unix(listener, node.clone()));
        let client = ApiClient::new(path.clone());

        // Drain the socket until `pred` matches one of the outbound items; returns all drained.
        async fn drain_socket_until(
            client: &ApiClient,
            session: &SessionId,
            pred: impl Fn(&Outbound) -> bool,
        ) -> Vec<Outbound> {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut seen = Vec::new();
            loop {
                match client
                    .call(ApiRequest::Poll {
                        session: session.clone(),
                        max: 0,
                    })
                    .await
                    .unwrap()
                {
                    ApiResponse::Drained(v) => {
                        let hit = v.iter().any(&pred);
                        seen.extend(v);
                        if hit {
                            return seen;
                        }
                    }
                    other => panic!("expected Drained, got {other:?}"),
                }
                assert!(Instant::now() < deadline, "socket drain never matched");
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }

        fn find_snapshot(items: &[Outbound], request_id: ReqId) -> Option<ConvView> {
            items.iter().find_map(|o| match o {
                Outbound::Event(AgentEvent::Snapshot {
                    request_id: id,
                    view,
                    ..
                }) if *id == request_id => Some(view.clone()),
                _ => None,
            })
        }

        // --- socket transport: StartTurn -> Snapshot -> Steer -> Interrupt ---
        let socket_session = SessionId::new("socket-live");
        assert!(matches!(
            client
                .call(ApiRequest::Submit {
                    session: socket_session.clone(),
                    command: AgentCommand::StartTurn {
                        input: UserMsg::new("hello there"),
                        request_id: ReqId(1),
                    },
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
        drain_socket_until(&client, &socket_session, |o| {
            matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. }))
        })
        .await;

        // Snapshot over the socket.
        client
            .call(ApiRequest::Submit {
                session: socket_session.clone(),
                command: AgentCommand::Snapshot {
                    request_id: ReqId(2),
                },
            })
            .await
            .unwrap();
        let socket_items = drain_socket_until(&client, &socket_session, |o| {
            matches!(o, Outbound::Event(AgentEvent::Snapshot { request_id, .. }) if *request_id == ReqId(2))
        })
        .await;
        let socket_view = find_snapshot(&socket_items, ReqId(2)).expect("a snapshot view");
        assert!(socket_view
            .turns
            .iter()
            .any(|t| t.role == "user" && t.text == "hello there"));

        // Steer over the socket: acked via a Steered event.
        client
            .call(ApiRequest::Submit {
                session: socket_session.clone(),
                command: AgentCommand::Steer {
                    text: "stay focused".into(),
                    request_id: ReqId(3),
                },
            })
            .await
            .unwrap();
        drain_socket_until(&client, &socket_session, |o| {
            matches!(o, Outbound::Event(AgentEvent::Steered { request_id, accepted, .. }) if *request_id == ReqId(3) && *accepted)
        })
        .await;

        // Interrupt over the socket flows through and is accepted.
        assert!(matches!(
            client
                .call(ApiRequest::Submit {
                    session: socket_session.clone(),
                    command: AgentCommand::Interrupt {
                        reason: Some("stop".into()),
                    },
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));

        // --- in-process parity: the same StartTurn + Snapshot yields the same view shape ---
        let inproc_session = SessionId::new("inproc-live");
        node.submit(
            inproc_session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hello there"),
                request_id: ReqId(1),
            },
        )
        .await
        .unwrap();
        let inproc_deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let drained = node.poll(inproc_session.clone(), 0).await.unwrap();
            if drained
                .iter()
                .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
            {
                break;
            }
            assert!(Instant::now() < inproc_deadline, "in-proc turn never finished");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        node.submit(
            inproc_session.clone(),
            AgentCommand::Snapshot {
                request_id: ReqId(2),
            },
        )
        .await
        .unwrap();
        let mut inproc_items = Vec::new();
        let snap_deadline = Instant::now() + Duration::from_secs(10);
        let inproc_view = loop {
            inproc_items.extend(node.poll(inproc_session.clone(), 0).await.unwrap());
            if let Some(view) = find_snapshot(&inproc_items, ReqId(2)) {
                break view;
            }
            assert!(Instant::now() < snap_deadline, "in-proc snapshot never arrived");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        assert_eq!(
            socket_view.turns, inproc_view.turns,
            "the snapshot projection must agree across transports"
        );

        server.abort();
        handle.shutdown().await;
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod store_backends {
    //! Cross-backend store conformance (phase 9): the substrate acceptance invariants run
    //! *identically* against both the in-memory backend and the durable SQLite backend, proving
    //! `SqliteStore` is a faithful drop-in. This is the impl-agnostic acceptance harness,
    //! parameterized by the store backend + a small fault-injection seam.

    use daemon_activation::{ActivationManager, ActivationSubstrate, SubErr};
    use daemon_common::{PartitionId, SessionId};
    use daemon_core::Snapshot;
    use daemon_host::CoreEngineFactory;
    use daemon_store::{
        FaultPoint, InMemoryStore, JobCompletion, SessionStatus, SessionStore, SqliteStore,
        StoreError,
    };
    use std::sync::Arc;

    const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// A store backend that can also arm a one-shot crash boundary (acceptance test #2).
    trait FaultStore: SessionStore {
        fn arm(&self, fault: Option<FaultPoint>);
    }
    impl FaultStore for InMemoryStore {
        fn arm(&self, fault: Option<FaultPoint>) {
            self.set_fault(fault);
        }
    }
    impl FaultStore for SqliteStore {
        fn arm(&self, fault: Option<FaultPoint>) {
            self.set_fault(fault);
        }
    }

    fn manager<S: FaultStore + 'static>(store: Arc<S>) -> ActivationManager {
        ActivationManager::new(store, Arc::new(CoreEngineFactory::delegating()), PARTITION)
    }

    async fn seed<S: SessionStore>(store: &S, id: &SessionId) {
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode snapshot");
        store
            .create_session(id.clone(), PARTITION, blob)
            .await
            .expect("create session");
    }

    async fn assert_completed<S: SessionStore>(store: &S, id: &SessionId) {
        assert_eq!(
            store.status(id).await,
            Some(SessionStatus::Completed),
            "session {id} should be Completed"
        );
    }

    /// Run the substrate acceptance invariants against a freshly built backend.
    async fn run_suite<S: FaultStore + 'static>(make: impl Fn() -> Arc<S>) {
        // #1 churn / baseline: the active directory returns to baseline after each session.
        {
            let store = make();
            let mgr = manager(store.clone());
            for i in 0..200 {
                let id = SessionId::new(format!("churn-{i}"));
                seed(&*store, &id).await;
                mgr.wake(id).await.expect("wake");
                assert_eq!(mgr.active_count(), 0, "directory leaked after session {i}");
            }
        }

        // #2 crash-after-every-boundary.
        {
            let store = make();
            let mgr = manager(store.clone());
            let id = SessionId::new("crash-before-snapshot");
            seed(&*store, &id).await;
            let f = store.acquire_activation_lease(&id).await.unwrap();
            store.arm(Some(FaultPoint::BeforeSnapshot));
            let r = mgr.activate(id.clone(), f).await;
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            mgr.recover().await.unwrap();
            assert_completed(&*store, &id).await;
        }
        for fault in [FaultPoint::AfterSnapshot, FaultPoint::AfterJobOutbox] {
            let store = make();
            let mgr = manager(store.clone());
            let id = SessionId::new(format!("crash-{fault:?}"));
            seed(&*store, &id).await;
            let f = store.acquire_activation_lease(&id).await.unwrap();
            store.arm(Some(fault));
            let r = mgr.activate(id.clone(), f).await;
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            assert!(matches!(
                store.status(&id).await,
                Some(SessionStatus::Suspended { .. })
            ));
            mgr.recover().await.unwrap();
            assert_completed(&*store, &id).await;
        }
        {
            // (f) completion durable + Ready, but the wake was lost; the scan must rescue it.
            let store = make();
            let mgr = manager(store.clone());
            let id = SessionId::new("crash-before-wake-publish");
            seed(&*store, &id).await;
            mgr.wake(id.clone()).await.unwrap();
            store.arm(Some(FaultPoint::BeforeWakePublish));
            let r = mgr.run_workers().await;
            assert!(matches!(r, Err(SubErr::Store(StoreError::Fault(_)))));
            assert_eq!(store.status(&id).await, Some(SessionStatus::Ready));
            assert!(store.dequeue_wake().await.is_none(), "wake should be lost");
            mgr.recover().await.unwrap();
            assert_completed(&*store, &id).await;
        }

        // #3 wake/completion idempotency.
        {
            let store = make();
            let mgr = manager(store.clone());
            let id = SessionId::new("idempotent");
            seed(&*store, &id).await;
            mgr.wake(id.clone()).await.unwrap();
            let job = store.dequeue_job().await.expect("a job on the outbox");
            let completion = JobCompletion {
                session_id: job.session_id,
                epoch: job.epoch,
                job_id: job.job_id,
                payload: job.payload,
            };
            for _ in 0..5 {
                store.record_completion_and_wake(&completion).await.unwrap();
            }
            assert_eq!(store.dequeue_wake().await.as_ref(), Some(&id));
            assert!(
                store.dequeue_wake().await.is_none(),
                "duplicate completions must not enqueue extra wakes"
            );
            mgr.wake(id.clone()).await.unwrap();
            assert_completed(&*store, &id).await;
        }

        // #4 dual-node fencing: only the highest-token holder commits.
        {
            let store = make();
            let mgr_a = manager(store.clone());
            let mgr_b = manager(store.clone());
            let id = SessionId::new("dual-node");
            seed(&*store, &id).await;
            let fa = store.acquire_activation_lease(&id).await.unwrap();
            let fb = store.acquire_activation_lease(&id).await.unwrap();
            assert!(fb > fa);
            let ra = mgr_a.activate(id.clone(), fa).await;
            assert!(matches!(ra, Err(SubErr::Store(StoreError::Fenced { .. }))));
            let rb = mgr_b.activate(id.clone(), fb).await;
            assert!(rb.is_ok(), "current node should commit: {rb:?}");
        }

        // #5 empty-mailbox process kill: recover solely from durable state.
        {
            let store = make();
            {
                let mgr1 = manager(store.clone());
                let id = SessionId::new("process-kill");
                seed(&*store, &id).await;
                mgr1.wake(id.clone()).await.unwrap();
            }
            let mgr2 = manager(store.clone());
            mgr2.recover().await.unwrap();
            assert_completed(&*store, &SessionId::new("process-kill")).await;
            assert_eq!(mgr2.active_count(), 0);
        }

        // #7 lost-wake recovery.
        {
            let store = make();
            let mgr = manager(store.clone());
            let id = SessionId::new("lost-wake");
            seed(&*store, &id).await;
            mgr.wake(id.clone()).await.unwrap();
            mgr.run_workers().await.unwrap();
            assert_eq!(store.dequeue_wake().await.as_ref(), Some(&id));
            assert!(store.dequeue_wake().await.is_none());
            mgr.recover().await.unwrap();
            assert_completed(&*store, &id).await;
        }
    }

    #[tokio::test]
    async fn in_memory_backend_acceptance() {
        run_suite(|| Arc::new(InMemoryStore::new())).await;
    }

    #[tokio::test]
    async fn sqlite_backend_acceptance() {
        run_suite(|| Arc::new(SqliteStore::open_in_memory().expect("open sqlite"))).await;
    }

    #[tokio::test]
    async fn sqlite_file_backend_round_trips() {
        // A temp DB *file* (WAL on disk): the on-disk path drives a session to completion and the
        // durable trace journal round-trips.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "daemon-conformance-{}-{}.sqlite",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&path);

        let store = Arc::new(SqliteStore::open(&path).expect("open sqlite file"));
        let mgr = manager(store.clone());
        let id = SessionId::new("file-backed");
        seed(&*store, &id).await;
        mgr.wake(id.clone()).await.unwrap();
        mgr.recover().await.unwrap();
        assert_completed(&*store, &id).await;
        drop(mgr);
        drop(store);

        for ext in ["sqlite", "sqlite-wal", "sqlite-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }
}
