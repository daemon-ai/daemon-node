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
use daemon_api::{
    ManageEventView, Outbound, SessionRole, TreeReport, UnitKind as ApiUnitKind, UnitNode,
    UnitState,
};
use daemon_common::{
    Budget, Epoch, PartitionId, ReqId, SessionId, SnapshotBlob, UnitId, UsageDelta,
};
use daemon_store::{Checkpoint, JobCompletion, SessionStore, StoreError};
use daemon_supervision::{
    Ack, Concurrency, DelegationSpec, EndReason, FailureClass, ManageCommand, ManageEvent,
    ManageRequest, ManageRequestHandler, ManageRequestKind, ManageResponse, ManageResponseBody,
    ManagedUnit, Outcome, ProgressDelta, StreamLagged, UnitKind, WorkRef,
};
use dashmap::DashMap;
use std::collections::HashSet;
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
    /// A fleet-unique prefix for minted child ids, so a sub-fleet's children never collide with a
    /// sibling fleet's (every node in the tree is addressable by a *unique* `UnitId`). Empty at the
    /// top fleet (`child-0`, ...); a sub-fleet uses its owning orchestrator's id (`{orch}/child-0`).
    id_prefix: String,
    /// The id of the unit that *owns* this fleet, when the fleet should project a rooted tree (the
    /// top node fleet). `None` for a sub-fleet — its owning orchestrator's node is built one level
    /// up from that orchestrator's record, so the sub-fleet projects only its members.
    root_id: Option<UnitId>,
}

impl FleetInner {
    /// Spawn, register, drive, and fold one child to a terminal outcome.
    async fn spawn_and_run(self: &Arc<Self>, spec: &DelegationSpec) -> (UnitId, Outcome) {
        let n = self.next_child.fetch_add(1, Ordering::SeqCst);
        let child_id = if self.id_prefix.is_empty() {
            UnitId::new(format!("child-{n}"))
        } else {
            UnitId::new(format!("{}/child-{n}", self.id_prefix))
        };

        let unit = self.spawner.spawn(child_id.clone(), spec).await;
        self.children.insert(
            child_id.clone(),
            ChildRecord::new(unit.clone(), spec.work.clone()),
        );

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
            let event = match events.recv().await {
                Ok(event) => event,
                Err(StreamLagged::Lagged { .. }) => continue,
                Err(StreamLagged::Closed) => {
                    let outcome = Outcome::ended(EndReason::Failed(FailureClass::Internal));
                    self.record_terminal(&child_id, ChildStatus::Failed, &outcome);
                    break outcome;
                }
            };
            // Buffer the event for GUI drill-down + fold per-child usage before handling it.
            self.record_event(&child_id, &event);
            match event {
                ManageEvent::Started { .. } => self.set_status(&child_id, ChildStatus::Running),
                ManageEvent::Usage { delta, .. } => self.usage.lock().unwrap().add(&delta),
                ManageEvent::Finished { outcome, .. } => {
                    self.record_terminal(&child_id, ChildStatus::Finished, &outcome);
                    break outcome;
                }
                ManageEvent::Error { failure, .. } => {
                    let outcome = Outcome::ended(EndReason::Failed(failure.class));
                    self.record_terminal(&child_id, ChildStatus::Failed, &outcome);
                    break outcome;
                }
                _ => {}
            }
        };
        (child_id, outcome)
    }

    fn set_status(&self, id: &UnitId, status: ChildStatus) {
        if let Some(mut r) = self.children.get_mut(id) {
            r.status = status;
        }
    }

    /// Fold a child's event into its record: per-child usage + a bounded view buffer (`unit_events`).
    fn record_event(&self, id: &UnitId, ev: &ManageEvent) {
        if let Some(mut r) = self.children.get_mut(id) {
            if let ManageEvent::Usage { delta, .. } = ev {
                r.usage.add(delta);
            }
            if let Some(view) = project_event(ev) {
                r.push_event(view);
            }
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
                id_prefix: String::new(),
                root_id: None,
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

    /// Set the fleet-unique prefix minted child ids carry (a sub-fleet uses its owning
    /// orchestrator's id), so every node in the whole tree is addressable by a unique `UnitId`.
    pub fn with_id_prefix(mut self, prefix: impl Into<String>) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_id_prefix before sharing")
            .id_prefix = prefix.into();
        self
    }

    /// Mark this fleet as the rooted node tree, projecting a synthetic root node (id `root`) whose
    /// children are the fleet's direct members — so the GUI gets a populated [`TreeReport::root`].
    pub fn with_root_id(mut self, root: UnitId) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_root_id before sharing")
            .root_id = Some(root);
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
            self.inner
                .store
                .record_completion_and_wake(&completion)
                .await?;
            tracing::debug!(%child_id, "fleet processed a delegation job");
            processed += 1;
        }
        Ok(processed)
    }

    /// Spawn + run a child per `spec` synchronously (the same management-level delegation the
    /// orchestrate tool's `Delegate` request takes through [`FleetRequestHandler`]), returning the
    /// spawned child ids. Legacy synchronous drive path retained for the foreign/ephemeral coarse
    /// lifecycle; the durable Core delegation path materializes children as durable sessions instead.
    pub async fn delegate(&self, specs: &[DelegationSpec]) -> Vec<UnitId> {
        let mut ids = Vec::with_capacity(specs.len());
        for spec in specs {
            let (id, _) = self.inner.spawn_and_run(spec).await;
            ids.push(id);
        }
        ids
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
        self.inner
            .children
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// The requests children have raised so far (observability / the gate's answer-authority proof).
    pub fn request_log(&self) -> Vec<ManageRequest> {
        self.inner.request_log.lock().unwrap().clone()
    }

    /// Project the fleet as the GUI/TUI tree: a flat node list with each node's `children` ids
    /// filled, recursing through orchestrator children's opaque sub-fleets (genuine
    /// fleets-of-fleets). When the fleet is rooted ([`Self::with_root_id`]) a synthetic root node is
    /// prepended and [`TreeReport::root`] is populated; a sub-fleet projects only its members (its
    /// owning orchestrator's node is built one level up from that orchestrator's record).
    pub fn tree(&self) -> TreeReport {
        let mut direct = Vec::new();
        let mut nodes = Vec::new();
        for e in self.inner.children.iter() {
            let id = e.key();
            let (node, subtree) = project_child(e.value(), id);
            direct.push(id.clone());
            nodes.push(node);
            nodes.extend(subtree);
        }
        match &self.inner.root_id {
            Some(root) => {
                let mut all = Vec::with_capacity(nodes.len() + 1);
                all.push(self.root_node(root, direct));
                all.extend(nodes);
                TreeReport {
                    root: Some(root.clone()),
                    nodes: all,
                }
            }
            None => TreeReport { root: None, nodes },
        }
    }

    /// The synthetic node-root node (the fleet itself as the GUI's tree root).
    fn root_node(&self, root: &UnitId, children: Vec<UnitId>) -> UnitNode {
        UnitNode {
            id: root.clone(),
            kind: ApiUnitKind::Orchestrator,
            state: UnitState::Running,
            work: None,
            usage: self.fleet_usage(),
            children,
            // The synthetic node-root is the fleet itself, not a session: no profile/session/title,
            // and no `SessionRole` (it is above the `Primary`/child taxonomy).
            profile: None,
            session: None,
            title: None,
            role: None,
        }
    }

    /// One unit's node view by id at any depth: the synthetic root, a direct child (with its
    /// sub-fleet roots filled), or — recursing through an orchestrator child — a grandchild.
    pub fn unit(&self, id: &UnitId) -> Option<UnitNode> {
        if self.inner.root_id.as_ref() == Some(id) {
            return Some(self.root_node(id, self.children()));
        }
        if let Some(r) = self.inner.children.get(id) {
            return Some(project_child(r.value(), id).0);
        }
        self.orchestrator_units()
            .iter()
            .find_map(|u| u.locate_node(id))
    }

    /// A bounded snapshot of a unit's most recent management-event views (GUI drill-down). Returns up
    /// to `max` (most recent; `0` = all buffered). Non-destructive so repeated reads — including the
    /// same call over different transports — observe the same window.
    pub fn unit_events(&self, id: &UnitId, max: u32) -> Vec<ManageEventView> {
        if let Some(r) = self.inner.children.get(id) {
            let len = r.events.len();
            let take = if max == 0 {
                len
            } else {
                (max as usize).min(len)
            };
            return r.events.iter().skip(len - take).cloned().collect();
        }
        // Recurse: an orchestrator child holds its descendants' event buffers in its sub-fleet.
        self.orchestrator_units()
            .iter()
            .map(|u| u.locate_events(id, max))
            .find(|v| !v.is_empty())
            .unwrap_or_default()
    }

    /// Drain up to `max` recent §17 [`Outbound`] items (events + raised host requests) for one unit
    /// — the rich, transcript-fidelity drill-down (the host side of `ControlApi::unit_outbound`).
    /// Routed by id to the registered unit's own retained drain; empty if the unit is unknown or
    /// (e.g. an orchestrator) retains no §17 stream. A destructive drain: each call consumes what it
    /// returns (`max == 0` drains all buffered items), mirroring the per-session poll model.
    pub fn unit_outbound(&self, id: &UnitId, max: u32) -> Vec<Outbound> {
        if let Some(r) = self.inner.children.get(id) {
            return r.unit.drain_outbound(max);
        }
        // Recurse into orchestrator children; only the subtree holding `id` drains (ids are unique).
        self.orchestrator_units()
            .iter()
            .map(|u| u.locate_outbound(id, max))
            .find(|v| !v.is_empty())
            .unwrap_or_default()
    }

    /// The handles of this fleet's orchestrator children — the only members whose opacity hides a
    /// sub-fleet, so the recursion descends through exactly these. Collected up front so no `DashMap`
    /// guard is ever held across an `.await` (the routing recursion is async).
    fn orchestrator_units(&self) -> Vec<Arc<dyn ManagedUnit>> {
        self.inner
            .children
            .iter()
            .filter(|e| matches!(e.value().unit.kind(), UnitKind::Orchestrator))
            .map(|e| e.value().unit.clone())
            .collect()
    }

    /// Route a lifecycle [`ManageCommand`] to the unit `id` at any depth, returning its [`Ack`] (or
    /// `None` if `id` is in no subtree here). A direct child is driven in-process; a deeper unit is
    /// reached through its orchestrator ancestor's `locate_command`.
    pub async fn command_unit(&self, id: &UnitId, cmd: ManageCommand) -> Option<Ack> {
        if let Some(unit) = self.inner.children.get(id).map(|r| r.unit.clone()) {
            return Some(unit.command(cmd).await);
        }
        for unit in self.orchestrator_units() {
            if let Some(ack) = unit.locate_command(id, cmd.clone()).await {
                return Some(ack);
            }
        }
        None
    }

    /// Route a lifecycle command to a unit, returning whether it was accepted. Pause/Resume/Scale are
    /// `Unsupported` at an engine leaf (single conversation), so these return `false` there — the
    /// surface is meaningful for orchestrator sub-fleets.
    async fn route_lifecycle(&self, id: &UnitId, cmd: ManageCommand) -> bool {
        matches!(
            self.command_unit(id, cmd).await,
            Some(Ack::Accepted | Ack::Queued)
        )
    }

    /// Pause a unit's scheduling (orchestrator sub-fleets); `false` if unknown or unsupported.
    pub async fn pause(&self, id: &UnitId) -> bool {
        self.route_lifecycle(id, ManageCommand::Pause).await
    }

    /// Resume a unit's scheduling; `false` if unknown or unsupported.
    pub async fn resume(&self, id: &UnitId) -> bool {
        self.route_lifecycle(id, ManageCommand::Resume).await
    }

    /// Scale a unit (sub-fleet) to `n` members; `false` if unknown or unsupported.
    pub async fn scale(&self, id: &UnitId, n: u32) -> bool {
        self.route_lifecycle(
            id,
            ManageCommand::Scale {
                target: Concurrency(n),
            },
        )
        .await
    }
}

/// Project a direct child into its node — filling `children` with the roots of any sub-fleet it
/// owns — and return its descendant nodes (empty for a leaf) for the flat tree. An orchestrator's
/// descendants come from its opaque [`ManagedUnit::project_subtree`]; its *direct* children are the
/// roots of that subtree forest (ids no other subtree node lists as a child).
fn project_child(record: &ChildRecord, id: &UnitId) -> (UnitNode, Vec<UnitNode>) {
    let mut node = project_unit(id, record);
    let subtree = record.unit.project_subtree();
    if !subtree.is_empty() {
        let referenced: HashSet<UnitId> = subtree
            .iter()
            .flat_map(|n| n.children.iter().cloned())
            .collect();
        node.children = subtree
            .iter()
            .map(|n| n.id.clone())
            .filter(|cid| !referenced.contains(cid))
            .collect();
    }
    (node, subtree)
}

/// Project a child record into the GUI tree-node DTO.
fn project_unit(id: &UnitId, record: &ChildRecord) -> UnitNode {
    let state = match record.status {
        ChildStatus::Finished | ChildStatus::Failed => UnitState::Finished {
            end_reason: record
                .outcome
                .as_ref()
                .map(|o| render_end_reason(&o.end_reason))
                .unwrap_or_else(|| "Unknown".to_string()),
        },
        _ => UnitState::Running,
    };
    let kind = map_kind(record.unit.kind());
    // A fleet child unit's id *is* its session id; engine children are long-lived `ManagedChild`
    // sessions (the fleet runtime creates no ephemeral-subagent marker yet — that distinction is the
    // deferred delegation-seam `ChildLifetime` work). Orchestrator children carry no session role.
    let (session, role) = match kind {
        ApiUnitKind::Engine => (
            Some(SessionId::new(id.as_str())),
            Some(SessionRole::ManagedChild),
        ),
        // Orchestrator / Host units are not leaf sessions: no session id or `SessionRole`.
        _ => (None, None),
    };
    UnitNode {
        id: id.clone(),
        kind,
        state,
        work: Some(render_work(&record.work)),
        usage: record.usage,
        children: Vec::new(),
        // Profile/title are not tracked on the fleet child record yet; the host's `node_for` seam
        // enriches them from session meta when projecting the tree.
        profile: None,
        session,
        title: None,
        role,
    }
}

fn map_kind(kind: UnitKind) -> ApiUnitKind {
    match kind {
        UnitKind::Engine => ApiUnitKind::Engine,
        UnitKind::Orchestrator => ApiUnitKind::Orchestrator,
    }
}

fn render_end_reason(end_reason: &EndReason) -> String {
    format!("{end_reason:?}")
}

fn render_work(work: &WorkRef) -> String {
    if let Some(payload) = &work.payload {
        payload.text.clone()
    } else {
        work.id.as_str().to_string()
    }
}

/// Project a management event into its transport-stable GUI view (`None` for events with no view).
fn project_event(ev: &ManageEvent) -> Option<ManageEventView> {
    Some(match ev {
        ManageEvent::Started { seq, .. } => ManageEventView::Started { seq: *seq },
        ManageEvent::Progress { seq, delta } => ManageEventView::Progress {
            seq: *seq,
            text: render_progress(delta),
        },
        ManageEvent::Usage { seq, delta } => ManageEventView::Usage {
            seq: *seq,
            delta: *delta,
        },
        ManageEvent::Finished { seq, outcome } => ManageEventView::Finished {
            seq: *seq,
            end_reason: render_end_reason(&outcome.end_reason),
            summary: outcome.summary.clone(),
        },
        ManageEvent::Error { seq, failure } => ManageEventView::Error {
            seq: *seq,
            message: failure.message.clone(),
        },
        _ => return None,
    })
}

fn render_progress(delta: &ProgressDelta) -> Option<String> {
    match delta {
        ProgressDelta::Text(t) | ProgressDelta::Reasoning(t) => Some(t.clone()),
        ProgressDelta::ToolStarted(tool) => Some(format!("tool {} started", tool.name)),
        ProgressDelta::ToolFinished(tool) => Some(format!(
            "tool {} {}",
            tool.call_id,
            if tool.ok { "ok" } else { "err" }
        )),
        _ => None,
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
