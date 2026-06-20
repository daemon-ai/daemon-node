//! The nested orchestrator: a [`ManagedUnit`] that *owns a sub-fleet* — genuine fleets-of-fleets.
//!
//! An [`OrchestratorUnit`] presents up the tree as a `UnitKind::Orchestrator` leaf to its parent
//! fleet (which holds its record), but its opacity hides its own [`FleetRuntime`]. When assigned
//! work it delegates into that sub-fleet (spawning grandchildren), and its `project_subtree` /
//! `locate_*` overrides forward the recursive projection/routing seam down one more level — so the
//! GUI addresses any node by `UnitId` at any depth, uniformly.
//!
//! It deliberately drives its sub-fleet through the *synchronous* management-level delegation
//! ([`FleetRuntime::delegate`]) rather than re-running the engine's durable job-outbox path one
//! level down: a nested level needs no second `JobOutboxDispatcher`. The "what to delegate" brain
//! stays `daemon-core`; this is the runtime that materializes the nesting.

use crate::policy::AnswerPolicy;
use crate::runtime::FleetRuntime;
use crate::spawner::ChildSpawner;
use async_trait::async_trait;
use daemon_common::{Budget, PartitionId, UnitId, UsageDelta};
use daemon_api::{ManageEventView, Outbound, UnitNode};
use daemon_store::SessionStore;
use daemon_supervision::{
    Ack, DelegationSpec, EndReason, EventStream, ManageCommand, ManageEvent, ManageRequestHandler,
    ManagedUnit, Outcome, StartTrigger, UnitKind, WorkRef,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// The shared state behind an [`OrchestratorUnit`] (cloned into the backgrounded delegation task).
struct OrchestratorInner {
    id: UnitId,
    sub: FleetRuntime,
    events: broadcast::Sender<ManageEvent>,
    seq: AtomicU64,
    /// `Pause` stops the orchestrator scheduling new work; `Resume` clears it (the lifecycle a leaf
    /// engine rejects as `Unsupported` is meaningful here).
    paused: AtomicBool,
    /// The sub-fleet usage already surfaced upward, so each delegation emits only the *new* delta
    /// (usage aggregates up exactly once; supervision invariant #4).
    reported_usage: Mutex<UsageDelta>,
    /// The parent's answer-authority over this orchestrator (installed by the holding fleet).
    handler: Mutex<Option<Arc<dyn ManageRequestHandler>>>,
}

impl OrchestratorInner {
    /// Emit one management event with the next monotonic `seq`.
    fn emit(&self, make: impl FnOnce(u64) -> ManageEvent) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let _ = self.events.send(make(seq));
    }

    /// Delegate `specs` into the sub-fleet (spawn + run grandchildren) and surface the newly-folded
    /// sub-fleet usage as a single `Usage` event so it aggregates up the tree.
    async fn run_delegation(&self, specs: Vec<DelegationSpec>) -> Vec<UnitId> {
        let ids = self.sub.delegate(&specs).await;
        let total = self.sub.fleet_usage();
        let delta = {
            let mut reported = self.reported_usage.lock().unwrap();
            let delta = UsageDelta {
                input_tokens: total.input_tokens.saturating_sub(reported.input_tokens),
                output_tokens: total.output_tokens.saturating_sub(reported.output_tokens),
                api_calls: total.api_calls.saturating_sub(reported.api_calls),
            };
            *reported = total;
            delta
        };
        if delta != UsageDelta::default() {
            self.emit(|seq| ManageEvent::Usage { seq, delta });
        }
        ids
    }
}

/// A unit that owns a sub-[`FleetRuntime`] — the materialized fleets-of-fleets nesting.
pub struct OrchestratorUnit {
    inner: Arc<OrchestratorInner>,
}

impl OrchestratorUnit {
    /// Wrap a sub-fleet as an orchestrator unit identified by `id`.
    pub fn new(id: UnitId, sub: FleetRuntime) -> Self {
        let (events, _) = broadcast::channel::<ManageEvent>(256);
        Self {
            inner: Arc::new(OrchestratorInner {
                id,
                sub,
                events,
                seq: AtomicU64::new(0),
                paused: AtomicBool::new(false),
                reported_usage: Mutex::new(UsageDelta::default()),
                handler: Mutex::new(None),
            }),
        }
    }

    /// The orchestrator's owned sub-fleet (e.g. for inspection in tests).
    pub fn sub_fleet(&self) -> FleetRuntime {
        self.inner.sub.clone()
    }
}

#[async_trait]
impl ManagedUnit for OrchestratorUnit {
    fn id(&self) -> UnitId {
        self.inner.id.clone()
    }

    fn kind(&self) -> UnitKind {
        UnitKind::Orchestrator
    }

    async fn command(&self, cmd: ManageCommand) -> Ack {
        match cmd {
            ManageCommand::Assign { work, budget, .. } => {
                if self.inner.paused.load(Ordering::SeqCst) {
                    return Ack::Busy;
                }
                // Background the delegation so progress streams up as `ManageEvent`s (a leaf engine's
                // async turn), and the parent's lossless fan-in folds Started/Usage/Finished.
                let inner = self.inner.clone();
                tokio::spawn(async move {
                    inner.emit(|seq| ManageEvent::Started {
                        seq,
                        trigger: StartTrigger::Assigned(work.id.clone()),
                    });
                    let spec = DelegationSpec {
                        work: work.clone(),
                        budget,
                        toolset: Vec::new(),
                    };
                    let ids = inner.run_delegation(vec![spec]).await;
                    inner.emit(move |seq| ManageEvent::Finished {
                        seq,
                        outcome: Outcome {
                            end_reason: EndReason::Completed,
                            summary: Some(format!("sub-fleet ran {} unit(s)", ids.len())),
                            artifacts: Vec::new(),
                        },
                    });
                });
                Ack::Accepted
            }
            // Scale the sub-fleet to `target` members synchronously, so the caller observes the new
            // members on return (the routing recursion proves Scale reaches a sub-fleet).
            ManageCommand::Scale { target } => {
                if self.inner.paused.load(Ordering::SeqCst) {
                    return Ack::Busy;
                }
                let specs: Vec<DelegationSpec> = (0..target.0)
                    .map(|i| DelegationSpec {
                        work: WorkRef::inline(format!("scale-{i}"), "scale"),
                        budget: Budget::unlimited(),
                        toolset: Vec::new(),
                    })
                    .collect();
                self.inner.run_delegation(specs).await;
                Ack::Accepted
            }
            ManageCommand::Pause => {
                self.inner.paused.store(true, Ordering::SeqCst);
                Ack::Accepted
            }
            ManageCommand::Resume => {
                self.inner.paused.store(false, Ordering::SeqCst);
                Ack::Accepted
            }
            ManageCommand::Cancel { .. } | ManageCommand::Shutdown { .. } => {
                for id in self.inner.sub.children() {
                    self.inner.sub.cancel_child(&id).await;
                }
                Ack::Accepted
            }
            ManageCommand::Snapshot { .. } => Ack::Accepted,
            _ => Ack::Unsupported,
        }
    }

    fn events(&self) -> EventStream<ManageEvent> {
        EventStream::new(self.inner.events.subscribe())
    }

    fn install_request_handler(&self, handler: Arc<dyn ManageRequestHandler>) {
        *self.inner.handler.lock().unwrap() = Some(handler);
    }

    // The recursive projection/routing seam: forward one level down into the owned sub-fleet, whose
    // own methods recurse further (an orchestrator grandchild nests transparently).

    fn project_subtree(&self) -> Vec<UnitNode> {
        self.inner.sub.tree().nodes
    }

    fn locate_node(&self, id: &UnitId) -> Option<UnitNode> {
        self.inner.sub.unit(id)
    }

    fn locate_events(&self, id: &UnitId, max: u32) -> Vec<ManageEventView> {
        self.inner.sub.unit_events(id, max)
    }

    fn locate_outbound(&self, id: &UnitId, max: u32) -> Vec<Outbound> {
        self.inner.sub.unit_outbound(id, max)
    }

    async fn locate_command(&self, id: &UnitId, cmd: ManageCommand) -> Option<Ack> {
        self.inner.sub.command_unit(id, cmd).await
    }
}

/// A [`ChildSpawner`] that materializes orchestrator children, each owning a sub-fleet — so a
/// parent fleet built on it grows fleets-of-fleets. `depth` bounds the nesting: at depth 1 a
/// sub-fleet spawns the injected `leaf` units; deeper, a sub-fleet spawns more orchestrators.
pub struct OrchestratorSpawner {
    store: Arc<dyn SessionStore>,
    partition: PartitionId,
    policy: Arc<dyn AnswerPolicy>,
    leaf: Arc<dyn ChildSpawner>,
    depth: usize,
}

impl OrchestratorSpawner {
    /// Build a spawner producing orchestrator children nested `depth` level(s) deep over `leaf`.
    pub fn new(
        store: Arc<dyn SessionStore>,
        partition: PartitionId,
        policy: Arc<dyn AnswerPolicy>,
        leaf: Arc<dyn ChildSpawner>,
        depth: usize,
    ) -> Self {
        Self {
            store,
            partition,
            policy,
            leaf,
            depth: depth.max(1),
        }
    }

    /// The spawner a freshly-materialized orchestrator's sub-fleet uses: the leaf at the deepest
    /// level, otherwise a one-shallower orchestrator spawner.
    fn sub_spawner(&self) -> Arc<dyn ChildSpawner> {
        if self.depth <= 1 {
            self.leaf.clone()
        } else {
            Arc::new(Self {
                store: self.store.clone(),
                partition: self.partition,
                policy: self.policy.clone(),
                leaf: self.leaf.clone(),
                depth: self.depth - 1,
            })
        }
    }
}

#[async_trait]
impl ChildSpawner for OrchestratorSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        // The sub-fleet ids are namespaced by this orchestrator's id, so no grandchild collides with
        // a sibling sub-fleet's child (every node in the tree is addressable by a unique `UnitId`).
        let sub = FleetRuntime::new(
            self.store.clone(),
            self.partition,
            self.sub_spawner(),
            self.policy.clone(),
            None,
        )
        .with_id_prefix(id.as_str().to_string());
        Arc::new(OrchestratorUnit::new(id, sub))
    }
}
