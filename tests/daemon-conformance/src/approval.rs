// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use daemon_activation::ActivationManager;
use daemon_common::{Epoch, JobId, PartitionId, SessionId, UsageDelta};
use daemon_core::{
    Capabilities, Failure, ModelOutput, Provider, Request, Snapshot, SystemPrompt, ToolCall,
    ToolCallFormat, ToolRegistry,
};
use daemon_host::CoreEngineFactory;
use daemon_store::{
    Checkpoint, InMemoryStore, ParkedApproval, SessionStatus, SessionStore, SqliteStore,
};
use std::sync::Arc;

const PARTITION: PartitionId = PartitionId::DEFAULT;

/// A conversation-aware deterministic provider: it emits a single fs `write` tool call until the
/// conversation carries a tool result, then completes with final text. Unlike a step-counter
/// provider this is correct across incarnations (the resumed engine re-builds a fresh provider),
/// because it keys off the durable conversation state, like a real model would.
struct WriteThenDone;

#[async_trait::async_trait]
impl Provider for WriteThenDone {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: Some(8192),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let usage = UsageDelta {
            input_tokens: 8,
            output_tokens: 4,
            api_calls: 1,
            ..Default::default()
        };
        if req.has_tool_result() {
            Ok(ModelOutput {
                text: "done".into(),
                reasoning: None,
                tool_calls: Vec::new(),
                usage,
            })
        } else {
            Ok(ModelOutput {
                text: String::new(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    call_id: "call-0".into(),
                    name: "fs".into(),
                    args: r#"{"op":"write","path":"approved.txt","content":"hi"}"#.into(),
                }],
                usage,
            })
        }
    }
}

/// An activation manager whose engines run one fs `write` (gated under the default `Ask` policy)
/// and then complete with final text — the durable approval cycle's driver.
fn writing_manager(store: Arc<dyn SessionStore>) -> ActivationManager {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(daemon_tool_fs::FsTool::new()));
    let provider: daemon_core::ProviderBuilder =
        Arc::new(|| Arc::new(WriteThenDone) as Arc<dyn Provider>);
    let factory = CoreEngineFactory::with_provider(
        provider,
        Arc::new(registry),
        SystemPrompt::new("approval conformance engine"),
    );
    ActivationManager::new(store, Arc::new(factory), PARTITION)
}

async fn seed(store: &dyn SessionStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone())
        .encode()
        .expect("encode fresh snapshot");
    store
        .create_session(id.clone(), PARTITION, blob)
        .await
        .expect("create session");
}

/// A gated fs write on the durable path parks (Suspended, no runnable job on the outbox); the
/// operator allows it; the woken session runs the write and completes.
#[tokio::test]
async fn durable_park_allow_resume_completes() {
    let store = Arc::new(InMemoryStore::new());
    let mgr = writing_manager(store.clone());
    let id = SessionId::new("approve-allow");
    seed(store.as_ref(), &id).await;

    mgr.wake(id.clone()).await.expect("first activation parks");
    // Parked: suspended, with a pending approval and *no* runnable job enqueued.
    assert!(matches!(
        store.status(&id).await,
        Some(SessionStatus::Suspended { .. })
    ));
    assert!(
        store.dequeue_job().await.is_none(),
        "no runnable job parked"
    );
    let pending = store.pending_approvals_of(Some(&id)).await;
    assert_eq!(pending.len(), 1, "exactly one parked approval");
    let request_id = pending[0].job_id.clone();

    // Operator allows: records the decision + wakes; the session resumes and completes.
    assert!(store
        .answer_approval(&id, &request_id, true)
        .await
        .expect("answer"));
    assert!(store.pending_approvals_of(Some(&id)).await.is_empty());
    mgr.wake(id.clone()).await.expect("resume");
    assert_eq!(store.status(&id).await, Some(SessionStatus::Completed));
}

/// A denied approval still resumes the session (the gated tool returns an error, the turn
/// completes) — the deny path never strands the session.
#[tokio::test]
async fn durable_park_deny_resume_completes() {
    let store = Arc::new(InMemoryStore::new());
    let mgr = writing_manager(store.clone());
    let id = SessionId::new("approve-deny");
    seed(store.as_ref(), &id).await;

    mgr.wake(id.clone()).await.expect("first activation parks");
    let request_id = store.pending_approvals_of(Some(&id)).await[0]
        .job_id
        .clone();
    assert!(store
        .answer_approval(&id, &request_id, false)
        .await
        .expect("answer"));
    mgr.wake(id.clone()).await.expect("resume");
    assert_eq!(store.status(&id).await, Some(SessionStatus::Completed));
}

/// A parked approval is durable: a fresh manager over the *same* store resolves it after a
/// simulated restart (the parked row + suspended snapshot survived).
#[tokio::test]
async fn parked_approval_survives_restart() {
    let store = Arc::new(InMemoryStore::new());
    let id = SessionId::new("approve-restart");
    seed(store.as_ref(), &id).await;
    {
        let mgr = writing_manager(store.clone());
        mgr.wake(id.clone()).await.expect("park");
    }
    // The original manager is gone; the parked approval is still listable + answerable.
    let request_id = store.pending_approvals_of(Some(&id)).await[0]
        .job_id
        .clone();
    assert!(store
        .answer_approval(&id, &request_id, true)
        .await
        .expect("answer"));
    let mgr2 = writing_manager(store.clone());
    mgr2.wake(id.clone()).await.expect("resume on new manager");
    assert_eq!(store.status(&id).await, Some(SessionStatus::Completed));
}

/// The store park/list/answer contract, run against both backends so they stay in lockstep:
/// a parked row lists until answered, answering records a wake + a completion, and a redelivered
/// answer is an idempotent no-op.
async fn store_contract(store: &dyn SessionStore) {
    let id = SessionId::new("park-contract");
    seed(store, &id).await;
    let fence = store.acquire_activation_lease(&id).await.expect("lease");
    let blob = Snapshot::fresh(id.clone()).encode().expect("encode");
    let job_id = JobId::new("park-contract:1:approval:0");
    let approval = ParkedApproval {
        session_id: id.clone(),
        job_id: job_id.clone(),
        epoch: Epoch(1),
        prompt: "approve write to a.txt".into(),
        path: Some("a.txt".into()),
        decision: None,
    };
    store
        .park_approval(
            Checkpoint::new(id.clone(), Epoch(1), blob),
            vec![approval],
            fence,
        )
        .await
        .expect("park");
    // Parked: suspended, listed, no runnable job.
    assert!(matches!(
        store.status(&id).await,
        Some(SessionStatus::Suspended { .. })
    ));
    assert!(store.dequeue_job().await.is_none());
    let pending = store.pending_approvals_of(Some(&id)).await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].prompt, "approve write to a.txt");
    // A node-wide listing finds it too.
    assert_eq!(store.pending_approvals_of(None).await.len(), 1);

    // Answer (allow): records a wake + a completion, drops it from the pending list.
    assert!(store
        .answer_approval(&id, &job_id, true)
        .await
        .expect("answer"));
    assert!(store.pending_approvals_of(Some(&id)).await.is_empty());
    assert_eq!(store.dequeue_wake().await, Some(id.clone()));
    let act = store
        .load_for_activation(&id, fence)
        .await
        .expect("activation");
    assert_eq!(act.unapplied.len(), 1, "one completion recorded");
    assert_eq!(act.unapplied[0].job_id, job_id);
    assert_eq!(act.unapplied[0].payload, b"allow");

    // Idempotent: a redelivered answer is a no-op (still answered, no extra wake/completion).
    assert!(store
        .answer_approval(&id, &job_id, true)
        .await
        .expect("re-answer"));
    assert!(store.dequeue_wake().await.is_none(), "no duplicate wake");

    // An unknown request answers false.
    assert!(!store
        .answer_approval(&id, &JobId::new("no-such"), true)
        .await
        .expect("unknown"));
}

#[tokio::test]
async fn store_contract_in_memory() {
    store_contract(&InMemoryStore::new()).await;
}

#[tokio::test]
async fn store_contract_sqlite() {
    let store = SqliteStore::open_in_memory().expect("sqlite");
    store_contract(&store).await;
}
