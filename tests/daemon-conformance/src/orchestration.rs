// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
            Arc::new(
                MockProvider::delegating("orchestrate", "fleet done")
                    .with_delegate_args(r#"{"verb":"spawn","task":"background work"}"#),
            ) as Arc<dyn Provider>
        }),
        Arc::new(registry),
        SystemPrompt::new("parent orchestrator"),
    );
    ActivationManager::new(store, Arc::new(factory), PARTITION)
}

async fn seed(store: &InMemoryStore, id: &SessionId) {
    let blob = Snapshot::fresh(id.clone())
        .encode()
        .expect("encode snapshot");
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
