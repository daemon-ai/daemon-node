// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Cross-backend store conformance (phase 9): the substrate acceptance invariants run
//! *identically* against both the in-memory backend and the durable SQLite backend, proving
//! `SqliteStore` is a faithful drop-in. This is the impl-agnostic acceptance harness,
//! parameterized by the store backend + a small fault-injection seam.

use daemon_activation::{ActivationManager, ActivationSubstrate, SubErr};
use daemon_common::{PartitionId, SessionId};
use daemon_core::Snapshot;
use daemon_host::CoreEngineFactory;
use daemon_store::{
    FaultPoint, FeedbackRecord, InMemoryStore, JobCompletion, SessionStatus, SessionStore,
    SqliteStore, StoreError,
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
    let blob = Snapshot::fresh(id.clone())
        .encode()
        .expect("encode snapshot");
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

/// Both backends round-trip the full enriched `UsageDelta` (cache/reasoning/cost) additively and
/// answer full-text session search identically — the P1 persistence surface (token columns + FTS).
async fn usage_and_search_suite<S: SessionStore>(store: Arc<S>) {
    use daemon_common::UsageDelta;

    let s = SessionId::new("acct");
    let delta = UsageDelta {
        input_tokens: 100,
        output_tokens: 40,
        api_calls: 1,
        cache_read_tokens: 60,
        cache_write_tokens: 20,
        reasoning_tokens: 10,
        cost_micros: 1234,
    };
    store.record_usage(&s, delta).await;
    store.record_usage(&s, delta).await;
    let total = store.usage_of(&s).await;
    assert_eq!(total.input_tokens, 200);
    assert_eq!(total.cache_read_tokens, 120);
    assert_eq!(total.cache_write_tokens, 40);
    assert_eq!(total.reasoning_tokens, 20);
    assert_eq!(total.cost_micros, 2468);

    store
        .index_session_text(
            &s,
            Some("Parser work".into()),
            "refactored the parser pipeline today",
        )
        .await;
    store
        .index_session_text(
            &SessionId::new("other"),
            Some("Renderer".into()),
            "fixed a crash in the gpu renderer",
        )
        .await;

    let hits = store.search_sessions("parser", 10).await;
    assert_eq!(hits.len(), 1, "exactly one session mentions the parser");
    assert_eq!(hits[0].session_id, s);
    assert!(hits[0].snippet.to_lowercase().contains("parser"));
    assert!(store
        .search_sessions("nonexistent-term", 10)
        .await
        .is_empty());
}

#[tokio::test]
async fn in_memory_usage_and_search() {
    usage_and_search_suite(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_usage_and_search() {
    usage_and_search_suite(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite"),
    ))
    .await;
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

/// §4.3 attached, non-joining edge: `record_child_edge` makes the child tree-visible under the
/// parent (audit) and labels it, but binds *no* delegation — so the child's terminal
/// `mark_completed` self-closes without enqueueing a parent wake. Contrast with a delegated child
/// (`bind_delegation`), whose completion *does* wake the parent.
async fn child_edge_suite<S: SessionStore>(store: Arc<S>) {
    use daemon_store::Checkpoint;

    let parent = SessionId::new("bg-parent");
    let child = SessionId::new("bg-child");
    seed(&*store, &parent).await;
    seed(&*store, &child).await;

    store
        .record_child_edge(parent.clone(), child.clone(), "skill_review".into())
        .await
        .expect("record attached edge");

    // tree-visible + labeled (audit), without a delegation binding.
    assert_eq!(
        store.children_of(&parent).await,
        vec![child.clone()],
        "background child must appear under the parent for audit"
    );
    assert_eq!(
        store.delegation_work(&child).await.as_deref(),
        Some("skill_review"),
        "background edge surfaces its work label"
    );

    // Drain any stray wakes first, then drive the child to terminal: it must NOT wake the parent.
    while store.dequeue_wake().await.is_some() {}
    let fence = store
        .acquire_activation_lease(&child)
        .await
        .expect("lease child");
    let snapshot = Snapshot::fresh(child.clone()).encode().expect("encode");
    store
        .mark_completed(
            Checkpoint::new(child.clone(), daemon_common::Epoch::ZERO, snapshot),
            fence,
        )
        .await
        .expect("child self-closes");

    assert_completed(&*store, &child).await;
    assert!(
        store.dequeue_wake().await.is_none(),
        "an attached non-joining child must never wake its parent"
    );
}

#[tokio::test]
async fn in_memory_child_edge() {
    child_edge_suite(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_child_edge() {
    child_edge_suite(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite"),
    ))
    .await;
}

/// N1: the durable feedback outbox + node-owned telemetry consent behave identically on both
/// backends — enqueue -> pending (oldest first) -> mark_delivered removes from pending, enqueue is
/// idempotent by id, and consent defaults OFF then round-trips through get/set.
async fn feedback_suite<S: SessionStore>(store: Arc<S>) {
    fn rec(id: &str, created_at_ms: i64) -> FeedbackRecord {
        FeedbackRecord {
            id: id.into(),
            created_at_ms,
            kind: "app".into(),
            rating: Some("up".into()),
            comment: Some("nice".into()),
            include_content: false,
            session: None,
            cursor: None,
            trace: None,
            surface: "settings".into(),
            app_version: Some("1.0.0".into()),
            os: Some("linux".into()),
            consent: "explicit-one-shot".into(),
            node_version: "test".into(),
            model: None,
            provider: None,
            end_reason: None,
            input_tokens: None,
            output_tokens: None,
            response_content: None,
            delivered: false,
        }
    }

    // Consent defaults OFF (opt-in) and round-trips.
    assert!(!store.telemetry_consent_get().await, "consent defaults OFF");
    store
        .telemetry_consent_set(true)
        .await
        .expect("set consent");
    assert!(store.telemetry_consent_get().await, "consent persisted on");
    store
        .telemetry_consent_set(false)
        .await
        .expect("clear consent");
    assert!(!store.telemetry_consent_get().await, "consent cleared");

    // Crash-reporting consent (wire v41) is a separate toggle: defaults OFF, round-trips, and is
    // independent of the telemetry consent above.
    assert!(
        !store.crash_consent_get().await,
        "crash consent defaults OFF"
    );
    store.crash_consent_set(true).await.expect("set crash");
    assert!(
        store.crash_consent_get().await,
        "crash consent persisted on"
    );
    assert!(
        !store.telemetry_consent_get().await,
        "crash consent does not affect telemetry consent"
    );
    store.crash_consent_set(false).await.expect("clear crash");
    assert!(!store.crash_consent_get().await, "crash consent cleared");

    // Enqueue two records out of created order; pending returns oldest first.
    store
        .feedback_enqueue(rec("fb-b", 200))
        .await
        .expect("enq b");
    store
        .feedback_enqueue(rec("fb-a", 100))
        .await
        .expect("enq a");
    let pending = store.feedback_pending(0).await;
    assert_eq!(
        pending.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        vec!["fb-a", "fb-b"],
        "pending is oldest-first"
    );
    // The record round-trips through the opaque CBOR blob faithfully.
    assert_eq!(pending[0].rating.as_deref(), Some("up"));
    assert_eq!(pending[0].surface, "settings");
    assert_eq!(pending[0].consent, "explicit-one-shot");

    // Idempotent by id: a re-enqueue of fb-a does not duplicate.
    store
        .feedback_enqueue(rec("fb-a", 100))
        .await
        .expect("re-enq");
    assert_eq!(store.feedback_pending(0).await.len(), 2, "no duplicate id");

    // limit caps the page.
    assert_eq!(store.feedback_pending(1).await.len(), 1);

    // mark_delivered removes it from the pending drain (idempotent).
    store
        .feedback_mark_delivered("fb-a")
        .await
        .expect("deliver a");
    store
        .feedback_mark_delivered("fb-a")
        .await
        .expect("deliver a again");
    let pending = store.feedback_pending(0).await;
    assert_eq!(
        pending.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        vec!["fb-b"],
        "delivered records drop out of pending"
    );
}

#[tokio::test]
async fn in_memory_feedback_outbox() {
    feedback_suite(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_feedback_outbox() {
    feedback_suite(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite"),
    ))
    .await;
}

/// Rung 3: the durable `command_dedup` table behaves identically on both backends — a
/// `(principal, op_id)` result round-trips within the TTL, different principals with the same
/// op_id are independent, a duplicate put preserves the ORIGINAL result (first-writer-wins), and
/// a read past the 24h TTL returns nothing (the op re-executes) and clears the stale row so a
/// fresh put re-caches.
async fn command_dedup_suite<S: SessionStore>(store: Arc<S>) {
    use daemon_store::COMMAND_DEDUP_TTL_MS;

    // A miss before anything is stored.
    assert!(store
        .command_dedup_get("alice", "op-1", 1_000)
        .await
        .is_none());

    // Store, then a fresh get within the TTL returns the ORIGINAL bytes.
    store
        .command_dedup_put("alice", "op-1", b"RESULT".to_vec(), 1_000)
        .await
        .expect("put dedup row");
    assert_eq!(
        store.command_dedup_get("alice", "op-1", 1_500).await,
        Some(b"RESULT".to_vec()),
        "a stored result is returned within the TTL"
    );

    // Different principals with the same op_id are independent.
    assert!(
        store
            .command_dedup_get("bob", "op-1", 1_500)
            .await
            .is_none(),
        "dedup is keyed on (principal, op_id): a different principal is independent"
    );

    // A duplicate put keeps the FIRST result (the ORIGINAL is what a retry must see).
    store
        .command_dedup_put("alice", "op-1", b"SECOND".to_vec(), 1_600)
        .await
        .expect("duplicate put is a no-op on the value");
    assert_eq!(
        store.command_dedup_get("alice", "op-1", 1_700).await,
        Some(b"RESULT".to_vec()),
        "first-writer-wins: the original result is preserved"
    );

    // A read past the TTL returns nothing (the op re-executes) and clears the stale row.
    let expired = 1_000 + COMMAND_DEDUP_TTL_MS + 1;
    assert!(
        store
            .command_dedup_get("alice", "op-1", expired)
            .await
            .is_none(),
        "an expired row is not served (the op re-executes)"
    );
    // After expiry a fresh put re-caches (the cleared row does not mask it).
    store
        .command_dedup_put("alice", "op-1", b"THIRD".to_vec(), expired)
        .await
        .expect("re-cache after expiry");
    assert_eq!(
        store.command_dedup_get("alice", "op-1", expired + 1).await,
        Some(b"THIRD".to_vec()),
        "a post-expiry put re-caches the re-executed result"
    );
}

#[tokio::test]
async fn in_memory_command_dedup() {
    command_dedup_suite(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_command_dedup() {
    command_dedup_suite(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite"),
    ))
    .await;
}

/// Rung 3: a `command_dedup` row survives a node restart on the durable sqlite backend — the
/// retry window that matters spans a restart (06 open-Q5), so the guarantee must be durable, not
/// an in-memory LRU.
#[tokio::test]
async fn sqlite_command_dedup_survives_restart() {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "daemon-dedup-restart-{}-{}.sqlite",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_file(&path);

    {
        let store = SqliteStore::open(&path).expect("open sqlite file");
        store
            .command_dedup_put("alice", "op-durable", b"KEPT".to_vec(), 1_000)
            .await
            .expect("put dedup row");
    }
    {
        // A fresh process (reopened store) still deduplicates the retried op.
        let store = SqliteStore::open(&path).expect("reopen sqlite file");
        assert_eq!(
            store.command_dedup_get("alice", "op-durable", 2_000).await,
            Some(b"KEPT".to_vec()),
            "the dedup row is durable across a restart"
        );
    }

    for ext in ["sqlite", "sqlite-wal", "sqlite-shm"] {
        let _ = std::fs::remove_file(path.with_extension(ext));
    }
}
