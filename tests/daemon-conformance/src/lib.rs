//! `daemon-conformance` — the substrate conformance harness.
//!
//! The executable acceptance gate for the build-first milestone: the seven acceptance tests from
//! [`rust-substrate-evaluation.md`](../../../docs/specs/rust-substrate-evaluation.md) §6, run against
//! the in-memory [`daemon_store::InMemoryStore`] driven through [`daemon_activation`] with the
//! [`daemon_stub_engine`] standing in for the real `daemon-core`. No dependency on `daemon-host`.
//!
//! Coverage map (acceptance test -> lifecycle §4 invariant):
//! 1 churn/baseline (#8), 2 crash-after-every-boundary (#2/#3/#7), 3 idempotency (#2/#3),
//! 4 dual-node fencing (#5/#6), 5 empty-mailbox kill (#1/#7), 6 ownership-transfer (#5/#6),
//! 7 lost-wake recovery (#1/#7).

#![forbid(unsafe_code)]

#[cfg(test)]
mod harness {
    use daemon_activation::ActivationManager;
    use daemon_common::{PartitionId, SessionId};
    use daemon_protocol::Snapshot;
    use daemon_store::{InMemoryStore, JobCompletion, SessionStatus, SessionStore};
    use daemon_stub_engine::StubEngineFactory;
    use std::sync::Arc;

    pub const PARTITION: PartitionId = PartitionId::DEFAULT;

    /// A fresh store + a single activation manager owning the default partition.
    pub fn new_world() -> (Arc<InMemoryStore>, ActivationManager) {
        let store = Arc::new(InMemoryStore::new());
        let mgr = manager(store.clone());
        (store, mgr)
    }

    /// An activation manager over an existing (possibly shared) store.
    pub fn manager(store: Arc<InMemoryStore>) -> ActivationManager {
        ActivationManager::new(store, Arc::new(StubEngineFactory::new()), PARTITION)
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
