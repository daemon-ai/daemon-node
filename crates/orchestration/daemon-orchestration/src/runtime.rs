//! The fleet runtime: the delegation-job worker, child fan-in, and the answer/escalation handler.
//!
//! [`FleetRuntime`] is a cloneable handle over shared fleet state. Its phase-4 job is
//! [`FleetRuntime::process_jobs_once`] — the real replacement for the substrate's placeholder
//! worker ([`daemon_activation::ActivationManager::run_workers`] echoes a completion; this spawns a
//! child and folds its outcome instead). The flow per delegation job:
//!
//! 1. spawn the child via the injected [`ChildSpawner`] and register it;
//! 2. install the runtime as the child's answer-authority and subscribe to its events *before*
//!    assigning work (lossless fan-in);
//! 3. drive [`ManageCommand::Assign`], folding `Usage`/status into fleet state until the child
//!    reaches a terminal `Outcome`;
//! 4. record the child as a real (Completed) session in the store, and record the child's outcome as
//!    the parent's [`JobCompletion`] — which wakes the parent as a `BackgroundCompletion`.
//!
//! Child requests ride [`FleetRequestHandler`]: `Delegate` grows the tree (the parent attaches
//! children), everything else is answered by the [`AnswerPolicy`] or re-escalated to the runtime's
//! own supervisor.

use crate::policy::{AnswerPolicy, Decision};
use crate::registry::{ChildRecord, ChildStatus};
use crate::spawner::ChildSpawner;
use async_trait::async_trait;
use daemon_common::{
    Budget, Epoch, PartitionId, ReqId, SessionId, SnapshotBlob, UnitId, UsageDelta,
};
use daemon_store::{Checkpoint, JobCompletion, SessionStore, StoreError};
use daemon_supervision::{
    DelegationSpec, EndReason, FailureClass, ManageCommand, ManageEvent, ManageRequest,
    ManageRequestHandler, ManageRequestKind, ManageResponse, ManageResponseBody, Outcome,
    StreamLagged, WorkRef,
};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The default ceiling on concurrently-attached children before `Delegate` escalates instead.
const DEFAULT_MAX_CHILDREN: usize = 16;

/// Errors the fleet runtime surfaces.
#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    /// A durable store operation failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// The shared fleet state behind a [`FleetRuntime`] handle (and a `Weak` ref in the request handler).
struct FleetInner {
    store: Arc<dyn SessionStore>,
    partition: PartitionId,
    spawner: Arc<dyn ChildSpawner>,
    policy: Arc<dyn AnswerPolicy>,
    parent: Option<Arc<dyn ManageRequestHandler>>,
    children: DashMap<UnitId, ChildRecord>,
    usage: Mutex<UsageDelta>,
    request_log: Mutex<Vec<ManageRequest>>,
    next_child: AtomicU64,
    max_children: usize,
}

impl FleetInner {
    /// Spawn, register, drive, and fold one child to a terminal outcome.
    async fn spawn_and_run(self: &Arc<Self>, spec: &DelegationSpec) -> (UnitId, Outcome) {
        let n = self.next_child.fetch_add(1, Ordering::SeqCst);
        let child_id = UnitId::new(format!("child-{n}"));

        let unit = self.spawner.spawn(child_id.clone(), spec).await;
        self.children
            .insert(child_id.clone(), ChildRecord::new(unit.clone(), spec.work.clone()));

        // Answer-authority for this child + lossless fan-in: install + subscribe before Assign.
        let handler: Arc<dyn ManageRequestHandler> = Arc::new(FleetRequestHandler {
            inner: Arc::downgrade(self),
        });
        unit.install_request_handler(handler);
        let mut events = unit.events();

        unit.command(ManageCommand::Assign {
            request_id: ReqId(0),
            work: spec.work.clone(),
            budget: spec.budget,
        })
        .await;

        let outcome = loop {
            match events.recv().await {
                Ok(ManageEvent::Started { .. }) => self.set_status(&child_id, ChildStatus::Running),
                Ok(ManageEvent::Usage { delta, .. }) => self.usage.lock().unwrap().add(&delta),
                Ok(ManageEvent::Finished { outcome, .. }) => {
                    self.record_terminal(&child_id, ChildStatus::Finished, &outcome);
                    break outcome;
                }
                Ok(ManageEvent::Error { failure, .. }) => {
                    let outcome = Outcome::ended(EndReason::Failed(failure.class));
                    self.record_terminal(&child_id, ChildStatus::Failed, &outcome);
                    break outcome;
                }
                Ok(_) => {}
                Err(StreamLagged::Lagged { .. }) => {}
                Err(StreamLagged::Closed) => {
                    let outcome = Outcome::ended(EndReason::Failed(FailureClass::Internal));
                    self.record_terminal(&child_id, ChildStatus::Failed, &outcome);
                    break outcome;
                }
            }
        };
        (child_id, outcome)
    }

    fn set_status(&self, id: &UnitId, status: ChildStatus) {
        if let Some(mut r) = self.children.get_mut(id) {
            r.status = status;
        }
    }

    fn record_terminal(&self, id: &UnitId, status: ChildStatus, outcome: &Outcome) {
        if let Some(mut r) = self.children.get_mut(id) {
            r.status = status;
            r.outcome = Some(outcome.clone());
        }
    }

    /// Record the child as a real durable session, ending `Completed` (synthesis §4.1: the run tree
    /// lives in the host store, not the runtime's memory).
    async fn record_child_footprint(&self, child: &UnitId) -> Result<(), StoreError> {
        let session = SessionId::new(child.as_str());
        self.store
            .create_session(session.clone(), self.partition, SnapshotBlob::default())
            .await?;
        let fence = self.store.acquire_activation_lease(&session).await?;
        self.store
            .mark_completed(
                Checkpoint {
                    session_id: session,
                    epoch: Epoch::ZERO,
                    snapshot: SnapshotBlob::default(),
                },
                fence,
            )
            .await?;
        Ok(())
    }
}

/// A cloneable handle to a node's fleet runtime (layout §4: the machinery between brain and wire).
#[derive(Clone)]
pub struct FleetRuntime {
    inner: Arc<FleetInner>,
}

impl FleetRuntime {
    /// Construct a runtime over a durable store and the injected placement/answer seams. `parent` is
    /// the runtime's own supervisor handler (the re-escalation target), `None` at the root.
    pub fn new(
        store: Arc<dyn SessionStore>,
        partition: PartitionId,
        spawner: Arc<dyn ChildSpawner>,
        policy: Arc<dyn AnswerPolicy>,
        parent: Option<Arc<dyn ManageRequestHandler>>,
    ) -> Self {
        Self {
            inner: Arc::new(FleetInner {
                store,
                partition,
                spawner,
                policy,
                parent,
                children: DashMap::new(),
                usage: Mutex::new(UsageDelta::default()),
                request_log: Mutex::new(Vec::new()),
                next_child: AtomicU64::new(0),
                max_children: DEFAULT_MAX_CHILDREN,
            }),
        }
    }

    /// Cap the number of concurrently-attached children before `Delegate` escalates.
    pub fn with_max_children(mut self, max: usize) -> Self {
        // Safe: the inner is freshly constructed and not yet shared.
        Arc::get_mut(&mut self.inner)
            .expect("with_max_children before sharing")
            .max_children = max;
        self
    }

    /// The answer-authority handler to install on a child (or hand to an engine as its host).
    pub fn request_handler(&self) -> Arc<dyn ManageRequestHandler> {
        Arc::new(FleetRequestHandler {
            inner: Arc::downgrade(&self.inner),
        })
    }

    /// Drain the durable job outbox, spawning + driving a child per delegation job and recording its
    /// outcome as the parent's completion. The phase-4 replacement for the placeholder worker.
    pub async fn process_jobs_once(&self) -> Result<usize, OrchestrationError> {
        let mut processed = 0usize;
        while let Some(job) = self.inner.store.dequeue_job().await {
            let work_text = String::from_utf8_lossy(&job.payload).to_string();
            let spec = DelegationSpec {
                work: WorkRef::inline(job.job_id.as_str(), work_text),
                budget: Budget::unlimited(),
                toolset: Vec::new(),
            };

            let (child_id, outcome) = self.inner.spawn_and_run(&spec).await;
            self.inner.record_child_footprint(&child_id).await?;

            let payload = outcome
                .summary
                .clone()
                .unwrap_or_else(|| format!("child:{child_id}"))
                .into_bytes();
            let completion = JobCompletion {
                session_id: job.session_id,
                epoch: job.epoch,
                job_id: job.job_id,
                payload,
            };
            self.inner.store.record_completion_and_wake(&completion).await?;
            tracing::debug!(%child_id, "fleet processed a delegation job");
            processed += 1;
        }
        Ok(processed)
    }

    /// Cancel a registered child by id (the orchestrate tool's `cancel` verb).
    pub async fn cancel_child(&self, id: &UnitId) -> bool {
        let unit = self.inner.children.get(id).map(|r| r.unit.clone());
        match unit {
            Some(unit) => {
                unit.command(ManageCommand::Cancel {
                    reason: Some("fleet cancel".into()),
                })
                .await;
                true
            }
            None => false,
        }
    }

    /// The folded fleet usage total (the §7 Usage fan-in; supervision invariant #4).
    pub fn fleet_usage(&self) -> UsageDelta {
        *self.inner.usage.lock().unwrap()
    }

    /// A child's current lifecycle status, if registered.
    pub fn child_status(&self, id: &UnitId) -> Option<ChildStatus> {
        self.inner.children.get(id).map(|r| r.status.clone())
    }

    /// A child's terminal outcome, if it has finished.
    pub fn child_outcome(&self, id: &UnitId) -> Option<Outcome> {
        self.inner.children.get(id).and_then(|r| r.outcome.clone())
    }

    /// The ids of all registered children.
    pub fn children(&self) -> Vec<UnitId> {
        self.inner.children.iter().map(|e| e.key().clone()).collect()
    }

    /// The requests children have raised so far (observability / the gate's answer-authority proof).
    pub fn request_log(&self) -> Vec<ManageRequest> {
        self.inner.request_log.lock().unwrap().clone()
    }
}

/// The child-facing [`ManageRequestHandler`] the runtime installs on each child.
///
/// Holds a `Weak` ref to the fleet to avoid a parent <-> child `Arc` cycle (the runtime owns the
/// child, the child owns this handler).
struct FleetRequestHandler {
    inner: Weak<FleetInner>,
}

#[async_trait]
impl ManageRequestHandler for FleetRequestHandler {
    async fn request(&self, req: ManageRequest) -> ManageResponse {
        let request_id = req.request_id;
        let Some(inner) = self.inner.upgrade() else {
            // The fleet was dropped; the child is being torn down.
            return ManageResponse {
                request_id,
                body: ManageResponseBody::Cancelled,
            };
        };
        inner.request_log.lock().unwrap().push(req.clone());

        // Delegate grows the tree: the parent is the answer-authority that attaches children.
        if let ManageRequestKind::Delegate(specs) = &req.kind {
            if inner.children.len() < inner.max_children {
                let mut ids = Vec::with_capacity(specs.len());
                for spec in specs {
                    let (id, _) = inner.spawn_and_run(spec).await;
                    ids.push(id);
                }
                return ManageResponse {
                    request_id,
                    body: ManageResponseBody::Delegated(ids),
                };
            }
            // Over the fleet budget: fall through to escalate.
        }

        match inner.policy.decide(&req) {
            Decision::Answer(body) => ManageResponse { request_id, body },
            Decision::Escalate => match &inner.parent {
                Some(parent) => parent.request(req).await,
                None => ManageResponse {
                    request_id,
                    body: ManageResponseBody::Escalated(false),
                },
            },
        }
    }
}
