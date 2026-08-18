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

/// Session-unification Stage 1 (spec §3, §5): identical containment semantics on both backends —
/// a blank interactive session is durably inert (`Idle`), creation never resurrects an existing
/// row, and the seeded factory (`create_runnable`) is atomic, idempotent, and crash-safe at the
/// construction boundary.
async fn unification_stage1_suite<S: FaultStore + 'static>(make: impl Fn() -> Arc<S>) {
    use daemon_store::{ExecutionPolicy, RunnableEdge, RunnableSession, SessionMeta, SessionRole};

    // (a) The incident pin: `create_idle` publishes a blank interactive session as `Idle` —
    // invisible to the recovery scan and inert to wakes — so no blank zero-epoch activation can
    // ever race the interactive rail (was: `create_session` published it 'ready'; the scanner ran
    // a blank turn whose failed commit became the authoritative snapshot).
    {
        let store = make();
        let mgr = manager(store.clone());
        let id = SessionId::new("blank-interactive");
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .create_idle(id.clone(), PARTITION, blob)
            .await
            .expect("create idle");
        assert_eq!(store.status(&id).await, Some(SessionStatus::Idle));
        assert_eq!(
            store.execution_policy(&id).await,
            Some(ExecutionPolicy::InteractiveRoot)
        );
        assert!(
            store
                .scan_resumable(PARTITION)
                .await
                .expect("scan")
                .is_empty(),
            "Idle must be invisible to the recovery scan"
        );
        // Even a stray durable wake hint must not run a turn on it.
        store.enqueue_wake(id.clone()).await;
        mgr.recover().await.expect("recover");
        assert_eq!(
            store.status(&id).await,
            Some(SessionStatus::Idle),
            "recovery must never run a turn on an Idle session"
        );
        assert_eq!(mgr.active_count(), 0);
    }

    // (b) Creation is insert-if-absent: a duplicate create surfaces `AlreadyExists` and resets
    // NOTHING (was: `INSERT OR REPLACE` silently resurrected the row at a blank epoch 0).
    {
        let store = make();
        let mgr = manager(store.clone());
        let id = SessionId::new("no-resurrect");
        seed(&*store, &id).await;
        mgr.wake(id.clone()).await.expect("wake");
        mgr.recover().await.expect("recover");
        assert_completed(&*store, &id).await;
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
        assert!(matches!(
            store
                .create_session(id.clone(), PARTITION, blob.clone())
                .await,
            Err(StoreError::AlreadyExists(_))
        ));
        assert!(matches!(
            store.create_idle(id.clone(), PARTITION, blob).await,
            Err(StoreError::AlreadyExists(_))
        ));
        assert_completed(&*store, &id).await;
    }

    // (c) The seeded factory: the construction-boundary crash rolls back to NOTHING; the retry
    // commits session + policy + meta + delegation edge + first input as one unit; a duplicate is
    // a no-op `false`; and the child's terminal completion fulfills the parent's job exactly like
    // a standalone `bind_delegation`.
    {
        let store = make();
        let mgr = manager(store.clone());
        let parent = SessionId::new("factory-parent");
        seed(&*store, &parent).await;
        mgr.wake(parent.clone()).await.expect("wake parent");
        let job = store
            .dequeue_job()
            .await
            .expect("the parent's delegated job");
        assert!(matches!(
            store.status(&parent).await,
            Some(SessionStatus::Suspended { .. })
        ));

        let child = SessionId::new("factory-parent/c1");
        let spec = RunnableSession {
            id: child.clone(),
            partition: PARTITION,
            snapshot: Snapshot::fresh(child.clone()).encode().expect("encode"),
            policy: ExecutionPolicy::JoiningChild,
            meta: Some(SessionMeta {
                parent: Some(parent.clone()),
                role: Some(SessionRole::ManagedChild),
                ..Default::default()
            }),
            edge: Some(RunnableEdge::Delegation(job)),
            first_input: Some(b"seeded-first-input".to_vec()),
        };

        store.arm(Some(FaultPoint::MidRunnableConstruction));
        assert!(matches!(
            store.create_runnable(spec.clone()).await,
            Err(StoreError::Fault(_))
        ));
        assert_eq!(
            store.status(&child).await,
            None,
            "rolled back: no session row"
        );
        assert!(
            store.children_of(&parent).await.is_empty(),
            "rolled back: no tree edge"
        );
        assert!(
            store.splices_after(&child, 0).await.is_empty(),
            "rolled back: no seeded inbox splice"
        );

        assert!(store
            .create_runnable(spec.clone())
            .await
            .expect("create runnable"));
        assert_eq!(store.status(&child).await, Some(SessionStatus::Ready));
        assert_eq!(
            store.execution_policy(&child).await,
            Some(ExecutionPolicy::JoiningChild)
        );
        assert_eq!(store.children_of(&parent).await, vec![child.clone()]);
        assert_eq!(
            store
                .session_meta(&child)
                .await
                .expect("meta stamped")
                .parent
                .as_ref(),
            Some(&parent)
        );
        let seeded = store.splices_after(&child, 0).await;
        assert_eq!(seeded.len(), 1, "the first input rides the durable inbox");
        assert_eq!(seeded[0].kind, daemon_store::SpliceKind::StartTurn);
        assert_eq!(seeded[0].payload, b"seeded-first-input");

        // Idempotent duplicate: `false` and NOTHING re-written — no duplicate tree edge, no
        // duplicated seeded splice.
        assert!(!store.create_runnable(spec).await.expect("idempotent"));
        assert_eq!(store.children_of(&parent).await, vec![child.clone()]);
        assert_eq!(store.splices_after(&child, 0).await.len(), 1);

        // The child runs to terminal; its completion fulfills the parent's job (the parent wakes
        // and completes) — the factory's edge behaves exactly like `bind_delegation`.
        mgr.recover().await.expect("recover");
        assert_completed(&*store, &child).await;
        assert_completed(&*store, &parent).await;
    }

    // (d) A detached child's notice edge: the spawn-time `bind_completion_notice` (which carries
    // the tool-call provenance) may land BEFORE the factory materializes the child — the factory
    // must dedupe the tree row and keep first-writer-wins on the recorded `call_id`.
    {
        let store = make();
        let mgr = manager(store.clone());
        let parent = SessionId::new("notice-parent");
        seed(&*store, &parent).await;
        let child = SessionId::new("notice-parent/d1");
        store
            .bind_completion_notice(&child, &parent, Some("call-7".into()))
            .await
            .expect("spawn-time bind");
        assert!(store
            .create_runnable(RunnableSession {
                id: child.clone(),
                partition: PARTITION,
                snapshot: Snapshot::fresh(child.clone()).encode().expect("encode"),
                policy: ExecutionPolicy::DetachedChild,
                meta: None,
                edge: Some(RunnableEdge::CompletionNotice {
                    parent: parent.clone(),
                    call_id: None,
                }),
                first_input: None,
            })
            .await
            .expect("create runnable"));
        assert_eq!(
            store.children_of(&parent).await,
            vec![child.clone()],
            "the factory edge dedupes against the spawn-time bind"
        );

        mgr.wake(child.clone()).await.expect("wake child");
        mgr.recover().await.expect("recover");
        assert_completed(&*store, &child).await;
        let notice = store
            .dequeue_completion_notice()
            .await
            .expect("terminal notice");
        assert_eq!(notice.parent, parent);
        assert_eq!(notice.child, child);
        assert_eq!(
            notice.call_id.as_deref(),
            Some("call-7"),
            "the spawn-time call_id survives the factory's None (first-writer-wins)"
        );
    }
}

#[tokio::test]
async fn in_memory_unification_stage1() {
    unification_stage1_suite(|| Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_unification_stage1() {
    unification_stage1_suite(|| Arc::new(SqliteStore::open_in_memory().expect("open sqlite")))
        .await;
}

/// Session-unification Stage 2 (spec §4, acceptance #4/#5): the durable typed inbox behaves
/// identically on both backends across the crash/replay boundaries — a producer retry after a
/// crash-before-ack lands the ORIGINAL splice, a claim orphaned by a crashed incarnation is
/// reclaimed by the next fence exactly once, consumption rides the commit transaction, and input
/// racing a turn's commit is never stranded.
async fn unification_stage2_suite<S: FaultStore + 'static>(make: impl Fn() -> Arc<S>) {
    use daemon_store::{Checkpoint, NewSplice, SpliceKind};

    fn splice(session: &SessionId, payload: &[u8], op: &str) -> NewSplice {
        NewSplice {
            session_id: session.clone(),
            kind: SpliceKind::StartTurn,
            payload: payload.to_vec(),
            origin_op: op.into(),
            origin: "conformance".into(),
        }
    }

    // (a) Crash after splice commit / before ack (acceptance #4a): the producer's retry — same
    // `origin_op` — returns the original `splice_seq` and inserts nothing; the append that landed
    // flipped the Idle session Ready atomically, so the input is never stranded invisible.
    {
        let store = make();
        let id = SessionId::new("splice-retry");
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .create_idle(id.clone(), PARTITION, blob)
            .await
            .expect("create idle");
        let seq = store
            .append_splice(splice(&id, b"turn please", "req-1"))
            .await
            .expect("append");
        assert_eq!(store.status(&id).await, Some(SessionStatus::Ready));
        let retried = store
            .append_splice(splice(&id, b"turn please", "req-1"))
            .await
            .expect("retry append");
        assert_eq!(retried, seq, "the retry lands on the original splice");
        assert_eq!(store.splices_after(&id, 0).await.len(), 1);
    }

    // (b) Crash after claim (acceptance #4b): an incarnation claims the inbox in its load and
    // dies before committing. The stale fence can no longer claim; the next activation's fence
    // reclaims the orphaned splices exactly once; its commit consumes them; a third activation
    // sees an empty inbox.
    {
        let store = make();
        let id = SessionId::new("splice-reclaim");
        seed(&*store, &id).await;
        store
            .append_splice(splice(&id, b"orphaned work", "req-2"))
            .await
            .expect("append");

        // First incarnation: claims in the load, then "crashes" (no commit ever happens).
        let f1 = store.acquire_activation_lease(&id).await.expect("lease 1");
        let crashed = store.load_for_activation(&id, f1).await.expect("load 1");
        assert_eq!(crashed.splices.len(), 1, "the load claims the inbox");

        // Recovery: the next fence reclaims the orphan; the stale fence is locked out.
        let f2 = store.acquire_activation_lease(&id).await.expect("lease 2");
        assert!(
            matches!(
                store.claim_splices(&id, f1).await,
                Err(StoreError::Fenced { .. })
            ),
            "the crashed incarnation's fence must not reclaim"
        );
        let recovered = store.load_for_activation(&id, f2).await.expect("load 2");
        assert_eq!(
            recovered.splices.len(),
            1,
            "the newer fence reclaims the orphaned splice exactly once"
        );

        // The recovered incarnation folds and commits: consumption rides the same transaction.
        let folded = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .mark_completed(
                Checkpoint::new(id.clone(), daemon_common::Epoch(1), folded)
                    .with_consumed_splices(Some(recovered.splices[0].splice_seq)),
                f2,
            )
            .await
            .expect("commit");
        let f3 = store.acquire_activation_lease(&id).await.expect("lease 3");
        let after = store.load_for_activation(&id, f3).await.expect("load 3");
        assert!(
            after.splices.is_empty(),
            "a consumed splice is never redelivered"
        );
    }

    // (c) Input racing the commit (acceptance #5): a splice appended AFTER the incarnation's load
    // but BEFORE its commit is untouched by that commit's consumption cursor — it stays pending
    // and the next activation claims it. Nothing raced-in is ever silently consumed or stranded.
    {
        let store = make();
        let id = SessionId::new("splice-race");
        seed(&*store, &id).await;
        store
            .append_splice(splice(&id, b"in the turn", "req-3"))
            .await
            .expect("append pre-load");
        let fence = store.acquire_activation_lease(&id).await.expect("lease");
        let activation = store.load_for_activation(&id, fence).await.expect("load");
        let claimed_seq = activation.splices[0].splice_seq;

        // Mid-turn: new input races in (a wire submit / an operator send).
        let raced_seq = store
            .append_splice(splice(&id, b"raced in mid-turn", "req-4"))
            .await
            .expect("append mid-turn");
        assert!(raced_seq > claimed_seq);

        // The commit consumes only what the turn folded (the claimed prefix).
        let folded = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .mark_completed(
                Checkpoint::new(id.clone(), daemon_common::Epoch(1), folded)
                    .with_consumed_splices(Some(claimed_seq)),
                fence,
            )
            .await
            .expect("commit");
        let remaining = store.splices_after(&id, 0).await;
        assert_eq!(
            remaining.iter().map(|s| s.splice_seq).collect::<Vec<_>>(),
            vec![raced_seq],
            "the raced-in splice survives the commit, unconsumed"
        );
        let f2 = store.acquire_activation_lease(&id).await.expect("lease 2");
        let next = store.load_for_activation(&id, f2).await.expect("load 2");
        assert_eq!(
            next.splices
                .iter()
                .map(|s| s.splice_seq)
                .collect::<Vec<_>>(),
            vec![raced_seq],
            "the next activation claims exactly the raced-in input"
        );
    }

    // (d) End-to-end through the real activation loop: a spliced input on a Ready session is
    // folded by the woken incarnation and consumed by its terminal commit — the inbox is empty
    // afterwards and the session completed (the fold happened exactly once, inside the substrate).
    {
        let store = make();
        let mgr = manager(store.clone());
        let id = SessionId::new("splice-end-to-end");
        seed(&*store, &id).await;
        store
            .append_splice(splice(&id, b"delegated task", "req-5"))
            .await
            .expect("append");
        mgr.wake(id.clone()).await.expect("wake");
        // The delegating test engine suspends on a job after its first activation; drive the
        // job + resume through the workers (the same path the resident dispatchers run).
        mgr.recover().await.expect("recover");
        assert_completed(&*store, &id).await;
        assert!(
            store.splices_after(&id, 0).await.is_empty(),
            "the terminal commit consumed the folded splice"
        );
    }
}

#[tokio::test]
async fn in_memory_unification_stage2() {
    unification_stage2_suite(|| Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_unification_stage2() {
    unification_stage2_suite(|| Arc::new(SqliteStore::open_in_memory().expect("open sqlite")))
        .await;
}

/// Session-unification stage 3 (§5): the persisted execution policy drives the turn boundary
/// through the REAL activation loop. An interactive root commits each turn back to `Idle` via the
/// fenced `commit_turn` — snapshot + turn_seq + splice consumption + the turn's journal seal in
/// one transaction — and survives a process restart into turn two (entered via the stage-2
/// internal append rail), with the journal re-keyed by turn (segment 0, then segment 1, each
/// sealed). A failed interactive turn stays retryable (never `Completed`). Every other policy
/// stays terminal at its first turn boundary — success and failure both.
async fn unification_stage3_suite<S: FaultStore + 'static>(make: impl Fn() -> Arc<S>) {
    use daemon_core::{
        Capabilities, EngineProfile, Failure, MockProvider, ModelOutput, Provider, Request,
        SystemPrompt, ToolCallFormat, ToolRegistry,
    };
    use daemon_store::{ExecutionPolicy, NewSplice, RunnableSession, SpliceKind};
    use daemon_telemetry::TraceSigner;

    fn splice(session: &SessionId, payload: &[u8], op: &str) -> NewSplice {
        NewSplice {
            session_id: session.clone(),
            kind: SpliceKind::StartTurn,
            payload: payload.to_vec(),
            origin_op: op.into(),
            origin: "conformance".into(),
        }
    }

    /// A provider whose every call fails non-retryably (a scripted HTTP 400), driving the
    /// engine's turn to `Completed(Failed)` — the policy then decides terminal vs retryable.
    struct FailingProvider;
    #[async_trait::async_trait]
    impl Provider for FailingProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            Err(Failure::InvalidRequest("scripted 400".into()))
        }
    }

    fn factory<S: SessionStore + 'static>(
        store: &Arc<S>,
        signer: &Arc<TraceSigner>,
        failing: bool,
    ) -> Arc<CoreEngineFactory> {
        let provider: Arc<dyn Fn() -> Arc<dyn Provider> + Send + Sync> = if failing {
            Arc::new(|| Arc::new(FailingProvider) as Arc<dyn Provider>)
        } else {
            Arc::new(|| Arc::new(MockProvider::completing("turn done")) as Arc<dyn Provider>)
        };
        let profile = EngineProfile::new(
            provider,
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("stage-3 conformance"),
        );
        Arc::new(
            CoreEngineFactory::from_profile(profile)
                .with_journal(store.clone() as Arc<dyn SessionStore>, signer.clone()),
        )
    }

    // (a) The interactive-root lifecycle across a restart. Turn one: a spliced input on an Idle
    // session wakes the activation loop, the turn commits back to Idle (never Completed), the
    // splice is consumed, and journal segment 0 (= turn 0) is sealed atomically with the commit.
    // "Restart": a fresh manager over the same store. Turn two enters via the stage-2 internal
    // append rail and lands in segment 1, itself sealed; the committed-turn counter reads 2.
    {
        let store = make();
        let signer = Arc::new(TraceSigner::generate());
        let id = SessionId::new("stage3-interactive");
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .create_idle(id.clone(), PARTITION, blob)
            .await
            .expect("create idle");

        // Turn one.
        {
            let mgr =
                ActivationManager::new(store.clone(), factory(&store, &signer, false), PARTITION);
            store
                .append_splice(splice(&id, b"first question", "turn-1"))
                .await
                .expect("append turn one");
            assert_eq!(store.status(&id).await, Some(SessionStatus::Ready));
            mgr.recover().await.expect("drive turn one");
            assert_eq!(
                store.status(&id).await,
                Some(SessionStatus::Idle),
                "an interactive turn commits back to Idle, never Completed"
            );
            assert!(
                store.splices_after(&id, 0).await.is_empty(),
                "the turn's commit consumed the folded splice"
            );
        }

        // The turn's journal segment (= turn 0) exists and was sealed by the commit transaction.
        let stream = daemon_common::JournalStreamId::session(&id);
        let seg0 = store
            .load_trace_segment(&stream, 0)
            .await
            .expect("segment 0 exists");
        assert!(
            !seg0.entries.is_empty(),
            "turn one journaled into segment 0"
        );
        assert!(
            seg0.committed.is_some(),
            "segment 0 sealed atomically with commit_turn"
        );

        // Restart: a fresh manager (fresh factory, fresh directory) over the same store. Turn two
        // enters via the internal append rail (stage 2) exactly like turn one did.
        {
            let mgr =
                ActivationManager::new(store.clone(), factory(&store, &signer, false), PARTITION);
            store
                .append_splice(splice(&id, b"second question", "turn-2"))
                .await
                .expect("append turn two");
            mgr.recover().await.expect("drive turn two");
        }
        assert_eq!(
            store.status(&id).await,
            Some(SessionStatus::Idle),
            "turn two also commits back to Idle"
        );
        let seg1 = store
            .load_trace_segment(&stream, 1)
            .await
            .expect("segment 1 exists");
        assert!(
            !seg1.entries.is_empty(),
            "turn two journaled into its OWN segment (journal re-keyed by turn, not epoch)"
        );
        assert!(seg1.committed.is_some(), "segment 1 sealed");
        let fence = store.acquire_activation_lease(&id).await.expect("lease");
        let post = store
            .load_for_activation(&id, fence)
            .await
            .expect("post-restart load");
        assert_eq!(post.turn_seq, 2, "two committed turns");
        assert_eq!(
            post.policy,
            Some(ExecutionPolicy::InteractiveRoot),
            "the persisted policy rides every activation load"
        );
        // Both user inputs live in the durable snapshot: turn two resumed the same conversation.
        let snap = Snapshot::decode(&post.snapshot).expect("decode snapshot");
        let users: Vec<&str> = snap
            .conversation
            .turns
            .iter()
            .filter_map(|t| match t {
                daemon_core::Turn::User(m) => Some(m.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            users,
            vec!["first question", "second question"],
            "the restarted turn folded into the SAME conversation"
        );
    }

    // (b) A FAILED interactive turn is not terminal: the session stays retryable (Idle), the
    // turn still commits (turn_seq advances, the splice is consumed) — retry = a new user action.
    {
        let store = make();
        let signer = Arc::new(TraceSigner::generate());
        let id = SessionId::new("stage3-interactive-fail");
        let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
        store
            .create_idle(id.clone(), PARTITION, blob)
            .await
            .expect("create idle");
        let mgr = ActivationManager::new(store.clone(), factory(&store, &signer, true), PARTITION);
        store
            .append_splice(splice(&id, b"doomed question", "turn-1"))
            .await
            .expect("append");
        mgr.recover().await.expect("drive the failing turn");
        assert_eq!(
            store.status(&id).await,
            Some(SessionStatus::Idle),
            "a failed interactive turn stays retryable — never Completed"
        );
        assert!(
            store.splices_after(&id, 0).await.is_empty(),
            "the failed turn still consumed its input (no poison replay loop)"
        );
        let fence = store.acquire_activation_lease(&id).await.expect("lease");
        let post = store.load_for_activation(&id, fence).await.expect("load");
        assert_eq!(post.turn_seq, 1, "the failed turn committed as a turn");
    }

    // (c) Every non-interactive policy is terminal at its turn boundary — on success AND on
    // failure (a one-shot run must close, fulfilling/notifying its parent seam; retryability is
    // the interactive contract only).
    for policy in [
        ExecutionPolicy::JoiningChild,
        ExecutionPolicy::DetachedChild,
        ExecutionPolicy::BackgroundChild,
        ExecutionPolicy::CronRun,
    ] {
        for failing in [false, true] {
            let store = make();
            let signer = Arc::new(TraceSigner::generate());
            let id = SessionId::new(format!(
                "stage3-{policy:?}-{}",
                if failing { "fail" } else { "ok" }
            ));
            let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
            store
                .create_runnable(RunnableSession {
                    id: id.clone(),
                    partition: PARTITION,
                    snapshot: blob,
                    policy,
                    meta: None,
                    edge: None,
                    first_input: None,
                })
                .await
                .expect("create runnable");
            store
                .append_splice(splice(&id, b"one-shot task", "task-1"))
                .await
                .expect("append task");
            let mgr =
                ActivationManager::new(store.clone(), factory(&store, &signer, failing), PARTITION);
            mgr.recover().await.expect("drive to terminal");
            assert_eq!(
                store.status(&id).await,
                Some(SessionStatus::Completed),
                "policy {policy:?} (failing={failing}) is terminal at the turn boundary"
            );
        }
    }
}

#[tokio::test]
async fn in_memory_unification_stage3() {
    unification_stage3_suite(|| Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_unification_stage3() {
    unification_stage3_suite(|| Arc::new(SqliteStore::open_in_memory().expect("open sqlite")))
        .await;
}

/// Session-unification stage 4 (§7): activation-vs-live parity through the REAL activation loop.
/// A durable interactive session with an attached [`daemon_host::AttachmentHub`] serves the live
/// rail's client surface — engine events stream onto the hub's merged log (non-destructive
/// `log_after`/`subscribe`, N consumers) and its destructive `poll` drain, each append badges
/// `SessionAdvanced` on the node feed, the in-flight turn's control occupies the hub's slot so a
/// MID-TURN steer is delivered INTO the resident turn (acked by `Steered`, folded into the same
/// turn — never deferred to the next activation), and the boundary releases the slot. A session
/// without a hub is untouched (the registry is pay-for-what-you-attach).
async fn unification_stage4_suite<S: FaultStore + 'static>(make: impl Fn() -> Arc<S>) {
    use daemon_api::{LogStreamItem, NodeEvent};
    use daemon_common::ReqId;
    use daemon_core::{
        Capabilities, EngineProfile, Failure, ModelOutput, Provider, Request, SystemPrompt,
        ToolCallFormat, ToolRegistry,
    };
    use daemon_host::{AttachmentHubs, NodeEventFeed};
    use daemon_protocol::{AgentCommand, AgentEvent, Direction, Outbound, SessionPayload};
    use daemon_store::{NewSplice, SpliceKind};
    use daemon_telemetry::TraceSigner;
    use futures::StreamExt;
    use tokio::sync::Notify;

    /// A provider that opens a mid-turn window: the FIRST call parks until the test releases it
    /// (entered/release notify pair), so the test can act on the turn while it is provably in
    /// flight; every call returns a plain completion (a steer folded at the phase boundary simply
    /// drives one more call).
    struct GatedProvider {
        entered: Arc<Notify>,
        release: Arc<Notify>,
        first_done: std::sync::atomic::AtomicBool,
    }
    #[async_trait::async_trait]
    impl Provider for GatedProvider {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_native_tools: true,
                supports_streaming: false,
                tool_call_format: ToolCallFormat::Native,
                max_context: Some(8192),
            }
        }
        async fn chat(&self, _req: Request) -> Result<ModelOutput, Failure> {
            if !self
                .first_done
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                self.entered.notify_one();
                self.release.notified().await;
            }
            Ok(ModelOutput {
                text: "gated turn done".into(),
                reasoning: None,
                tool_calls: vec![],
                usage: Default::default(),
                meta: None,
            })
        }
    }

    let store = make();
    let signer = Arc::new(TraceSigner::generate());
    let feed = NodeEventFeed::new(64);
    let hubs = Arc::new(AttachmentHubs::new(Some(feed.clone())));
    let id = SessionId::new("stage4-attached");
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
    store
        .create_idle(id.clone(), PARTITION, blob)
        .await
        .expect("create idle");

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let provider = {
        let (entered, release) = (entered.clone(), release.clone());
        let gate: Arc<dyn Provider> = Arc::new(GatedProvider {
            entered,
            release,
            first_done: std::sync::atomic::AtomicBool::new(false),
        });
        Arc::new(move || gate.clone())
    };
    let profile = EngineProfile::new(
        provider,
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("stage-4 conformance"),
    );
    let factory = Arc::new(
        CoreEngineFactory::from_profile(profile)
            .with_journal(store.clone() as Arc<dyn SessionStore>, signer.clone())
            .with_attachments(hubs.clone()),
    );

    // Attach BEFORE the turn (an already-observing client), take a push subscription from seq 0,
    // and prove the slot is empty pre-turn: a steer with no resident turn is NOT claimed — the
    // caller must route it durably (splice + wake) instead.
    let hub = hubs.attach(&id, 0);
    let mut sub = hub.subscribe(0);
    assert!(!hub.occupied(), "no turn resident before activation");
    assert!(
        !hub.deliver_steer(ReqId(41), "too early".into()),
        "a steer with no resident turn is not claimed by the hub"
    );

    // Open the turn through the durable rail (stage 2/3: splice + activation), holding the engine
    // mid-model-call so the turn is provably in flight.
    store
        .append_splice(NewSplice {
            session_id: id.clone(),
            kind: SpliceKind::StartTurn,
            payload: b"start the gated turn".to_vec(),
            origin_op: "stage4-turn".into(),
            origin: "conformance".into(),
        })
        .await
        .expect("append splice");
    let mgr = ActivationManager::new(store.clone(), factory, PARTITION);
    let drive = tokio::spawn(async move { mgr.recover().await });

    // Mid-turn: the incarnation registered its TurnControl on the hub, so the slot is occupied and
    // a steer is delivered INTO the resident turn (live parity — not parked for the next turn).
    entered.notified().await;
    assert!(hub.occupied(), "the in-flight turn occupies the hub's slot");
    assert!(
        hub.deliver_steer(ReqId(42), "mid-turn steer".into()),
        "a mid-turn steer is claimed into the resident turn"
    );
    release.notify_one();
    drive.await.expect("join").expect("drive the gated turn");

    // Boundary parity: the turn committed back to Idle (stage 3) and released the hub's slot.
    assert_eq!(store.status(&id).await, Some(SessionStatus::Idle));
    assert!(!hub.occupied(), "the turn boundary releases the slot");

    // The steer reached the SAME turn: the engine acked it (`Steered`, accepted) and the ack
    // streamed onto the hub — the activation path emits events as they happen, not post hoc.
    let polled = hub.poll(0);
    assert!(
        polled.iter().any(|f| matches!(
            f,
            Outbound::Event(AgentEvent::Steered { request_id, accepted, .. })
                if *request_id == ReqId(42) && *accepted
        )),
        "the mid-turn steer was folded and acked by the resident turn"
    );
    assert!(
        polled
            .iter()
            .any(|f| matches!(f, Outbound::Event(AgentEvent::TurnFinished { .. }))),
        "the turn's terminal event streamed onto the drain"
    );
    assert!(hub.poll(0).is_empty(), "poll is destructive (live parity)");

    // The merged log is NON-destructive and carries both directions on one seq timeline: the
    // outbound engine events survive the poll, and the claimed steer was recorded as the inbound
    // command it was.
    let page = hub.log_after(0, 0);
    assert!(
        page.entries.iter().any(|e| matches!(
            (&e.direction, &e.payload),
            (
                Direction::Inbound,
                SessionPayload::Command(AgentCommand::Steer { request_id, .. })
            ) if *request_id == ReqId(42)
        )),
        "the claimed steer is on the merged log as an inbound command"
    );
    assert!(
        page.entries.iter().any(|e| matches!(
            (&e.direction, &e.payload),
            (
                Direction::Outbound,
                SessionPayload::Event(AgentEvent::TurnFinished { .. })
            )
        )),
        "the merged log retains outbound events after the destructive poll"
    );
    // The pre-turn push subscription replays the same timeline (backfill + live on one stream).
    let mut streamed = 0usize;
    while streamed < page.entries.len() {
        match sub.next().await {
            Some(LogStreamItem::Entry(e)) => {
                assert_eq!(e.seq, page.entries[streamed].seq, "same seq timeline");
                streamed += 1;
            }
            other => panic!("subscription ended early: {other:?}"),
        }
    }

    // Each hub append coalesced a `SessionAdvanced` onto the node feed (live-advance parity).
    let advanced = feed.page(0, 0).events.into_iter().any(|e| {
        matches!(&e, NodeEvent::SessionAdvanced { session, head_seq, .. }
            if *session == id && *head_seq >= page.entries.len() as u64)
    });
    assert!(
        advanced,
        "hub appends badge SessionAdvanced on the node feed"
    );

    // Pay-for-what-you-attach: a session with NO hub runs exactly the stage-3 path under the same
    // attachment-carrying factory (nothing streams, nothing occupies, the turn still commits).
    let bare = SessionId::new("stage4-bare");
    let blob = Snapshot::fresh(bare.clone()).encode().expect("encode");
    store
        .create_idle(bare.clone(), PARTITION, blob)
        .await
        .expect("create idle");
    store
        .append_splice(NewSplice {
            session_id: bare.clone(),
            kind: SpliceKind::StartTurn,
            payload: b"unattached turn".to_vec(),
            origin_op: "stage4-bare".into(),
            origin: "conformance".into(),
        })
        .await
        .expect("append splice");
    let gate: Arc<dyn Provider> = Arc::new(daemon_core::MockProvider::completing("done"));
    let profile = EngineProfile::new(
        Arc::new(move || gate.clone()),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("stage-4 bare"),
    );
    let factory = Arc::new(
        CoreEngineFactory::from_profile(profile)
            .with_journal(store.clone() as Arc<dyn SessionStore>, signer.clone())
            .with_attachments(hubs.clone()),
    );
    ActivationManager::new(store.clone(), factory, PARTITION)
        .recover()
        .await
        .expect("drive the unattached turn");
    assert_eq!(store.status(&bare).await, Some(SessionStatus::Idle));
    assert!(hubs.get(&bare).is_none(), "no hub was implicitly created");
}

#[tokio::test]
async fn in_memory_unification_stage4() {
    unification_stage4_suite(|| Arc::new(InMemoryStore::new())).await;
}

#[tokio::test]
async fn sqlite_unification_stage4() {
    unification_stage4_suite(|| Arc::new(SqliteStore::open_in_memory().expect("open sqlite")))
        .await;
}

/// Session-unification §6 regression: a concurrent wake of a BUSY session must not bump the fence
/// past the in-flight incarnation. The in-process slot is reserved before the lease is acquired,
/// so a busy session's wake returns satisfied without touching the lease; under the old
/// lease-then-check order every such wake self-fenced the running incarnation's eventual commit
/// (this test then fails: the commit is Fenced and the session never completes).
#[tokio::test]
async fn concurrent_wake_does_not_self_fence() {
    use daemon_activation::{EngineError, EngineFactory, Incarnation, Step};
    use daemon_common::Epoch;
    use tokio::sync::Notify;

    /// An engine that parks mid-run until released, so the test can interleave wakes while the
    /// incarnation verifiably holds the slot.
    struct GateEngine {
        entered: Arc<Notify>,
        release: Arc<Notify>,
        snapshot: Option<daemon_activation::SnapshotBlob>,
    }
    #[async_trait::async_trait]
    impl Incarnation for GateEngine {
        async fn hydrate(
            &mut self,
            snapshot: daemon_activation::SnapshotBlob,
            _unapplied: Vec<JobCompletion>,
            _splices: Vec<daemon_store::InboxSplice>,
            _ctx: daemon_activation::TurnCtx,
        ) -> Result<(), EngineError> {
            self.snapshot = Some(snapshot);
            Ok(())
        }
        async fn run(&mut self) -> Result<Step, EngineError> {
            self.entered.notify_one();
            self.release.notified().await;
            Ok(Step::Completed)
        }
        fn checkpoint(&self) -> Result<daemon_activation::SnapshotBlob, EngineError> {
            Ok(self.snapshot.clone().expect("hydrated"))
        }
        fn epoch(&self) -> Epoch {
            Epoch::ZERO
        }
    }
    struct GateFactory {
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }
    impl EngineFactory for GateFactory {
        fn create(&self) -> Box<dyn Incarnation> {
            Box::new(GateEngine {
                entered: self.entered.clone(),
                release: self.release.clone(),
                snapshot: None,
            })
        }
    }

    let store = Arc::new(InMemoryStore::new());
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mgr = ActivationManager::new(
        store.clone(),
        Arc::new(GateFactory {
            entered: entered.clone(),
            release: release.clone(),
        }),
        PARTITION,
    );
    let id = SessionId::new("busy-no-self-fence");
    seed(&*store, &id).await;

    let first = {
        let mgr = mgr.clone();
        let id = id.clone();
        tokio::spawn(async move { mgr.wake(id).await })
    };
    entered.notified().await; // the incarnation is mid-run, holding the slot

    for _ in 0..8 {
        mgr.wake(id.clone()).await.expect("busy wake is satisfied");
    }

    release.notify_one();
    first
        .await
        .expect("join")
        .expect("the in-flight incarnation's commit must not be fenced by busy wakes");
    assert_completed(&*store, &id).await;
}

/// Session-unification §8 regression: commit-then-linger runs consecutive turns on ONE resident
/// incarnation. After a non-terminal `commit_turn` the incarnation holds its slot awaiting the
/// next wake — the second turn hydrates the SAME instance (no factory re-create), each turn was
/// committed at its boundary (the store showed `Idle` before the second input existed), and the
/// first wake's caller was released at the first commit rather than being held for the linger
/// window. Shutdown passivates the lingering incarnation promptly (cancellation, not timeout).
#[tokio::test]
async fn commit_then_linger_runs_consecutive_turns_on_one_incarnation() {
    use daemon_activation::{EngineError, EngineFactory, Incarnation, Step};
    use daemon_common::Epoch;
    use daemon_store::{NewSplice, SpliceKind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A minimal turn-committing engine: every hydrate+run consumes the delivered splices and
    /// commits a turn, counting hydrates so residency is observable.
    struct LingerEngine {
        hydrates: Arc<AtomicUsize>,
        consumed: Option<u64>,
        snapshot: Option<daemon_activation::SnapshotBlob>,
    }
    #[async_trait::async_trait]
    impl Incarnation for LingerEngine {
        async fn hydrate(
            &mut self,
            snapshot: daemon_activation::SnapshotBlob,
            _unapplied: Vec<JobCompletion>,
            splices: Vec<daemon_store::InboxSplice>,
            _ctx: daemon_activation::TurnCtx,
        ) -> Result<(), EngineError> {
            self.hydrates.fetch_add(1, Ordering::SeqCst);
            self.snapshot = Some(snapshot);
            self.consumed = splices.last().map(|s| s.splice_seq).or(self.consumed);
            Ok(())
        }
        async fn run(&mut self) -> Result<Step, EngineError> {
            Ok(Step::TurnCommitted)
        }
        fn checkpoint(&self) -> Result<daemon_activation::SnapshotBlob, EngineError> {
            Ok(self.snapshot.clone().expect("hydrated"))
        }
        fn epoch(&self) -> Epoch {
            Epoch::ZERO
        }
        fn consumed_splices(&self) -> Option<u64> {
            self.consumed
        }
    }
    struct LingerFactory {
        created: Arc<AtomicUsize>,
        hydrates: Arc<AtomicUsize>,
    }
    impl EngineFactory for LingerFactory {
        fn create(&self) -> Box<dyn Incarnation> {
            self.created.fetch_add(1, Ordering::SeqCst);
            Box::new(LingerEngine {
                hydrates: self.hydrates.clone(),
                consumed: None,
                snapshot: None,
            })
        }
    }

    fn start_turn(session: &SessionId, op: &str) -> NewSplice {
        NewSplice {
            session_id: session.clone(),
            kind: SpliceKind::StartTurn,
            payload: b"input".to_vec(),
            origin_op: op.into(),
            origin: "conformance".into(),
        }
    }
    /// Poll until the session settles `Idle` (its turn committed) within the bound.
    async fn settle_idle(store: &InMemoryStore, id: &SessionId) {
        for _ in 0..200 {
            if store.status(id).await == Some(SessionStatus::Idle) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("session {id} never settled Idle");
    }

    let store = Arc::new(InMemoryStore::new());
    let created = Arc::new(AtomicUsize::new(0));
    let hydrates = Arc::new(AtomicUsize::new(0));
    let mgr = ActivationManager::with_linger(
        store.clone(),
        Arc::new(LingerFactory {
            created: created.clone(),
            hydrates: hydrates.clone(),
        }),
        PARTITION,
        Some(Duration::from_secs(30)),
    );
    let id = SessionId::new("linger-two-turns");
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
    store
        .create_idle(id.clone(), PARTITION, blob)
        .await
        .expect("create idle");

    // Turn one: the wake returns at the first commit (not after the linger window)...
    store
        .append_splice(start_turn(&id, "turn-1"))
        .await
        .expect("append turn one");
    mgr.wake(id.clone()).await.expect("wake turn one");
    settle_idle(&store, &id).await;
    // ...and the committed incarnation lingers, holding its slot.
    assert_eq!(mgr.active_count(), 1, "the incarnation lingers post-commit");
    assert_eq!(created.load(Ordering::SeqCst), 1);
    assert_eq!(hydrates.load(Ordering::SeqCst), 1);

    // Turn two rides the SAME residency: the wake hands the hint to the lingering incarnation.
    store
        .append_splice(start_turn(&id, "turn-2"))
        .await
        .expect("append turn two");
    mgr.wake(id.clone()).await.expect("wake turn two");
    settle_idle(&store, &id).await;
    assert_eq!(
        created.load(Ordering::SeqCst),
        1,
        "turn two must reuse the resident incarnation, not re-create"
    );
    assert_eq!(
        hydrates.load(Ordering::SeqCst),
        2,
        "turn two hydrates the same instance"
    );
    assert_eq!(mgr.active_count(), 1, "still resident after turn two");

    // Shutdown cancels the linger promptly; the slot is released (invariant #8).
    mgr.shutdown().await;
    assert_eq!(mgr.active_count(), 0, "shutdown passivates the lingerer");
}

/// Session-unification §8: the idle timeout passivates the ALREADY-COMMITTED lingering
/// incarnation — the slot returns to baseline without any further input, and the next turn
/// re-activates through the normal wake path (a fresh incarnation).
#[tokio::test]
async fn linger_timeout_passivates_committed_incarnation() {
    use daemon_activation::{EngineError, EngineFactory, Incarnation, Step};
    use daemon_common::Epoch;
    use daemon_store::{NewSplice, SpliceKind};
    use std::time::Duration;

    struct OneTurnEngine {
        consumed: Option<u64>,
        snapshot: Option<daemon_activation::SnapshotBlob>,
    }
    #[async_trait::async_trait]
    impl Incarnation for OneTurnEngine {
        async fn hydrate(
            &mut self,
            snapshot: daemon_activation::SnapshotBlob,
            _unapplied: Vec<JobCompletion>,
            splices: Vec<daemon_store::InboxSplice>,
            _ctx: daemon_activation::TurnCtx,
        ) -> Result<(), EngineError> {
            self.snapshot = Some(snapshot);
            self.consumed = splices.last().map(|s| s.splice_seq);
            Ok(())
        }
        async fn run(&mut self) -> Result<Step, EngineError> {
            Ok(Step::TurnCommitted)
        }
        fn checkpoint(&self) -> Result<daemon_activation::SnapshotBlob, EngineError> {
            Ok(self.snapshot.clone().expect("hydrated"))
        }
        fn epoch(&self) -> Epoch {
            Epoch::ZERO
        }
        fn consumed_splices(&self) -> Option<u64> {
            self.consumed
        }
    }
    struct OneTurnFactory;
    impl EngineFactory for OneTurnFactory {
        fn create(&self) -> Box<dyn Incarnation> {
            Box::new(OneTurnEngine {
                consumed: None,
                snapshot: None,
            })
        }
    }

    let store = Arc::new(InMemoryStore::new());
    let mgr = ActivationManager::with_linger(
        store.clone(),
        Arc::new(OneTurnFactory),
        PARTITION,
        Some(Duration::from_millis(50)),
    );
    let id = SessionId::new("linger-timeout");
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
    store
        .create_idle(id.clone(), PARTITION, blob)
        .await
        .expect("create idle");
    store
        .append_splice(NewSplice {
            session_id: id.clone(),
            kind: SpliceKind::StartTurn,
            payload: b"input".to_vec(),
            origin_op: "turn-1".into(),
            origin: "conformance".into(),
        })
        .await
        .expect("append");
    mgr.wake(id.clone()).await.expect("wake");
    assert_eq!(store.status(&id).await, Some(SessionStatus::Idle));

    // The already-committed incarnation passivates on the idle timeout — no input required.
    for _ in 0..200 {
        if mgr.active_count() == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the lingering incarnation never passivated on the idle timeout");
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
