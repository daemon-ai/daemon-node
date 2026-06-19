//! The placement cut: the protocol-aware side of an out-of-process child (host-spec §7, §9).
//!
//! `daemon-provision` opens an OS-level [`CutChannel`](daemon_provision::CutChannel) — a raw,
//! length-framed byte duplex to a child process. This module gives that channel meaning. The cut is
//! the host-authority boundary: the child is a *pure executor* holding no durable state of its own,
//! and the parent's [`SessionStore`] is brokered across the cut so the parent remains the sole
//! fence authority. "Fencing holds across the cut" is then the dual-ownership property (acceptance
//! tests #4/#6) enforced across a real process boundary — a stale incarnation's commit is rejected
//! by the parent store exactly as it would be in-process.
//!
//! Three pieces realize it:
//! - [`PlacedUnit`] — the parent-side [`ManagedUnit`] proxy. It frames `ManageCommand`s down,
//!   relays the child's `ManageEvent`s up, and *serves* the child's brokered store/request traffic
//!   against the parent's authority.
//! - [`RemoteStoreClient`] — the child-side [`SessionStore`] whose every method is a request/reply
//!   round-trip over the cut.
//! - [`run_placed_child`] — the child loop: build a [`RemoteStoreClient`], drive the engine through
//!   the ordinary [`ActivationManager`] under the parent-granted fence, and stream events back.

use async_trait::async_trait;
use daemon_activation::{ActivationManager, ActivationSubstrate, EngineFactory, SubErr};
use daemon_common::{DaemonError, FenceToken, PartitionId, SessionId, SnapshotBlob, UnitId};
use daemon_provision::{ChildGuard, CutChannel, CutWriter, Placement};
use daemon_store::{
    Activation, Checkpoint, JobCommand, JobCompletion, SessionStatus, SessionStore, StoreError,
    StoreErrorWire, StoreStats,
};
use daemon_supervision::{
    Ack, EndReason, EventStream, FailureClass, FailureView, ManageCommand, ManageEvent,
    ManageRequest, ManageRequestHandler, ManageResponseBody, ManagedUnit, Outcome, StartTrigger,
    UnitKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, oneshot, Mutex as AsyncMutex};

// ---------------------------------------------------------------------------
// The cut wire
// ---------------------------------------------------------------------------

/// A multiplexed frame on a placement cut. One enum carries both directions; the child consumes the
/// parent-bound variants and vice versa.
#[derive(Debug, Serialize, Deserialize)]
pub enum CutFrame {
    /// Parent -> child: activate `session` under a parent-granted `fence` and run a turn.
    RunTurn {
        /// The durable session to activate (the child's `UnitId` as a `SessionId`).
        session: SessionId,
        /// The fence the parent acquired and grants the child to commit under.
        fence: FenceToken,
    },
    /// Parent -> child: cancel in-flight work and exit.
    Cancel {
        /// Optional human-readable reason.
        reason: Option<String>,
    },
    /// Parent -> child: stop and exit.
    Shutdown,
    /// Parent -> child: the reply to a brokered [`StoreCall`].
    StoreReply {
        /// Correlates with the originating [`StoreCall`].
        id: u64,
        /// The reply body.
        body: StoreReplyBody,
    },
    /// Parent -> child: the reply to an escalated [`ManageRequest`].
    RequestReply {
        /// Correlates with the originating request.
        id: u64,
        /// The reply body.
        body: ManageResponseBody,
    },
    /// Child -> parent: a management event from the placed unit.
    Event(ManageEvent),
    /// Child -> parent: a brokered durable-store call (served against the parent's authority).
    StoreCall {
        /// Correlation id for the reply.
        id: u64,
        /// The store operation.
        call: StoreCall,
    },
    /// Child -> parent: an escalated blocking request.
    Request {
        /// Correlation id for the reply.
        id: u64,
        /// The request payload.
        req: ManageRequest,
    },
}

/// A brokered [`SessionStore`] operation, mirroring the trait one-to-one (phase-5 cut).
#[derive(Debug, Serialize, Deserialize)]
pub enum StoreCall {
    /// [`SessionStore::create_session`].
    CreateSession {
        /// The session identity.
        id: SessionId,
        /// The owning partition.
        partition: PartitionId,
        /// The initial snapshot.
        snapshot: SnapshotBlob,
    },
    /// [`SessionStore::acquire_activation_lease`].
    AcquireActivationLease {
        /// The session.
        id: SessionId,
    },
    /// [`SessionStore::load_for_activation`].
    LoadForActivation {
        /// The session.
        id: SessionId,
        /// The fence to load under.
        fence: FenceToken,
    },
    /// [`SessionStore::checkpoint_and_enqueue`].
    CheckpointAndEnqueue {
        /// The checkpoint to persist.
        checkpoint: Checkpoint,
        /// The job to enqueue.
        job: JobCommand,
        /// The committing fence.
        fence: FenceToken,
    },
    /// [`SessionStore::mark_completed`].
    MarkCompleted {
        /// The terminal checkpoint.
        checkpoint: Checkpoint,
        /// The committing fence.
        fence: FenceToken,
    },
    /// [`SessionStore::record_completion_and_wake`].
    RecordCompletionAndWake {
        /// The completion to record.
        completion: JobCompletion,
    },
    /// [`SessionStore::scan_resumable`].
    ScanResumable {
        /// The partition to scan.
        partition: PartitionId,
    },
    /// [`SessionStore::dequeue_job`].
    DequeueJob,
    /// [`SessionStore::dequeue_wake`].
    DequeueWake,
    /// [`SessionStore::status`].
    Status {
        /// The session.
        id: SessionId,
    },
    /// [`SessionStore::stats`].
    Stats,
}

/// The reply to a [`StoreCall`], typed per the trait's return shape. Fallible calls carry a
/// [`StoreErrorWire`] so the parent store's verdict (notably `Fenced`) round-trips faithfully.
#[derive(Debug, Serialize, Deserialize)]
pub enum StoreReplyBody {
    /// A `Result<(), _>` reply.
    Unit(Result<(), StoreErrorWire>),
    /// An `acquire_activation_lease` reply.
    Fence(Result<FenceToken, StoreErrorWire>),
    /// A `load_for_activation` reply.
    Activation(Result<Activation, StoreErrorWire>),
    /// A `scan_resumable` reply.
    Scan(Result<Vec<SessionId>, StoreErrorWire>),
    /// A `dequeue_job` reply.
    MaybeJob(Option<JobCommand>),
    /// A `dequeue_wake` reply.
    MaybeWake(Option<SessionId>),
    /// A `status` reply.
    MaybeStatus(Option<SessionStatus>),
    /// A `stats` reply.
    Stats(StoreStats),
}

fn encode(frame: &CutFrame) -> Vec<u8> {
    let mut buf = Vec::new();
    // Our frame types are always serializable; a failure here is a programming error.
    ciborium::into_writer(frame, &mut buf).expect("encode CutFrame");
    buf
}

fn decode(bytes: &[u8]) -> Option<CutFrame> {
    ciborium::from_reader(bytes).ok()
}

/// Serve one brokered store call against the parent's authoritative store (the parent side of the
/// cut). This is what makes the parent the single fence authority for the out-of-process child.
async fn serve_store_call(store: &dyn SessionStore, call: StoreCall) -> StoreReplyBody {
    match call {
        StoreCall::CreateSession {
            id,
            partition,
            snapshot,
        } => StoreReplyBody::Unit(
            store
                .create_session(id, partition, snapshot)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::AcquireActivationLease { id } => StoreReplyBody::Fence(
            store
                .acquire_activation_lease(&id)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::LoadForActivation { id, fence } => StoreReplyBody::Activation(
            store
                .load_for_activation(&id, fence)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::CheckpointAndEnqueue {
            checkpoint,
            job,
            fence,
        } => StoreReplyBody::Unit(
            store
                .checkpoint_and_enqueue(checkpoint, job, fence)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::MarkCompleted { checkpoint, fence } => StoreReplyBody::Unit(
            store
                .mark_completed(checkpoint, fence)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::RecordCompletionAndWake { completion } => StoreReplyBody::Unit(
            store
                .record_completion_and_wake(&completion)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::ScanResumable { partition } => StoreReplyBody::Scan(
            store
                .scan_resumable(partition)
                .await
                .map_err(|e| StoreErrorWire::from(&e)),
        ),
        StoreCall::DequeueJob => StoreReplyBody::MaybeJob(store.dequeue_job().await),
        StoreCall::DequeueWake => StoreReplyBody::MaybeWake(store.dequeue_wake().await),
        StoreCall::Status { id } => StoreReplyBody::MaybeStatus(store.status(&id).await),
        StoreCall::Stats => StoreReplyBody::Stats(store.stats().await),
    }
}

// ---------------------------------------------------------------------------
// Child-side store client
// ---------------------------------------------------------------------------

/// The child-side [`SessionStore`]: every method is a request/reply round-trip over the cut, served
/// by the parent's authoritative store. The child therefore holds no durable state and cannot
/// commit except through the parent's fence check.
pub struct RemoteStoreClient {
    writer: CutWriter,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<StoreReplyBody>>>>,
    next_id: AtomicU64,
}

impl RemoteStoreClient {
    /// Construct a client that frames calls onto `writer`.
    pub fn new(writer: CutWriter) -> Self {
        Self {
            writer,
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(0),
        }
    }

    /// Complete a pending call with the reply that arrived on the child's reader loop.
    pub fn complete(&self, id: u64, body: StoreReplyBody) {
        if let Some(tx) = self.pending.lock().unwrap().remove(&id) {
            let _ = tx.send(body);
        }
    }

    async fn call(&self, call: StoreCall) -> StoreReplyBody {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if self
            .writer
            .send(&encode(&CutFrame::StoreCall { id, call }))
            .await
            .is_err()
        {
            self.pending.lock().unwrap().remove(&id);
            return StoreReplyBody::Unit(Err(StoreErrorWire::Other("cut channel closed".into())));
        }
        rx.await.unwrap_or(StoreReplyBody::Unit(Err(StoreErrorWire::Other(
            "cut reply dropped".into(),
        ))))
    }
}

fn unexpected_reply<T>() -> Result<T, StoreError> {
    Err(StoreError::Common(DaemonError::Other(
        "unexpected store reply on cut".into(),
    )))
}

#[async_trait]
impl SessionStore for RemoteStoreClient {
    async fn create_session(
        &self,
        id: SessionId,
        partition: PartitionId,
        snapshot: SnapshotBlob,
    ) -> Result<(), StoreError> {
        match self
            .call(StoreCall::CreateSession {
                id,
                partition,
                snapshot,
            })
            .await
        {
            StoreReplyBody::Unit(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn acquire_activation_lease(&self, id: &SessionId) -> Result<FenceToken, StoreError> {
        match self
            .call(StoreCall::AcquireActivationLease { id: id.clone() })
            .await
        {
            StoreReplyBody::Fence(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn load_for_activation(
        &self,
        id: &SessionId,
        fence: FenceToken,
    ) -> Result<Activation, StoreError> {
        match self
            .call(StoreCall::LoadForActivation {
                id: id.clone(),
                fence,
            })
            .await
        {
            StoreReplyBody::Activation(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn checkpoint_and_enqueue(
        &self,
        checkpoint: Checkpoint,
        job: JobCommand,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        match self
            .call(StoreCall::CheckpointAndEnqueue {
                checkpoint,
                job,
                fence,
            })
            .await
        {
            StoreReplyBody::Unit(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn mark_completed(
        &self,
        checkpoint: Checkpoint,
        fence: FenceToken,
    ) -> Result<(), StoreError> {
        match self
            .call(StoreCall::MarkCompleted { checkpoint, fence })
            .await
        {
            StoreReplyBody::Unit(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn record_completion_and_wake(&self, c: &JobCompletion) -> Result<(), StoreError> {
        match self
            .call(StoreCall::RecordCompletionAndWake {
                completion: c.clone(),
            })
            .await
        {
            StoreReplyBody::Unit(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn scan_resumable(&self, partition: PartitionId) -> Result<Vec<SessionId>, StoreError> {
        match self.call(StoreCall::ScanResumable { partition }).await {
            StoreReplyBody::Scan(r) => r.map_err(StoreErrorWire::into_store_error),
            _ => unexpected_reply(),
        }
    }

    async fn dequeue_job(&self) -> Option<JobCommand> {
        match self.call(StoreCall::DequeueJob).await {
            StoreReplyBody::MaybeJob(j) => j,
            _ => None,
        }
    }

    async fn dequeue_wake(&self) -> Option<SessionId> {
        match self.call(StoreCall::DequeueWake).await {
            StoreReplyBody::MaybeWake(w) => w,
            _ => None,
        }
    }

    async fn status(&self, id: &SessionId) -> Option<SessionStatus> {
        match self.call(StoreCall::Status { id: id.clone() }).await {
            StoreReplyBody::MaybeStatus(s) => s,
            _ => None,
        }
    }

    async fn stats(&self) -> StoreStats {
        match self.call(StoreCall::Stats).await {
            StoreReplyBody::Stats(s) => s,
            _ => StoreStats::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Parent-side proxy: the placed unit
// ---------------------------------------------------------------------------

type HandlerSlot = Arc<Mutex<Option<Arc<dyn ManageRequestHandler>>>>;

/// The parent-side [`ManagedUnit`] proxy for a child placed across a cut (host-spec §9).
///
/// To a supervisor it is an ordinary unit; underneath, commands are framed to the child process and
/// the child's events stream back. It also holds the parent's authoritative [`SessionStore`] and
/// serves the child's brokered store calls against it — so the parent stays the single fence
/// authority across the cut.
pub struct PlacedUnit {
    id: UnitId,
    writer: CutWriter,
    store: Arc<dyn SessionStore>,
    events: broadcast::Sender<ManageEvent>,
    handler: HandlerSlot,
    child: Arc<AsyncMutex<ChildGuard>>,
}

impl PlacedUnit {
    /// Wrap a live [`Placement`] as a managed unit identified by `id`, serving the child's brokered
    /// store traffic against `store`.
    pub fn new(id: UnitId, placement: Placement, store: Arc<dyn SessionStore>) -> Self {
        let Placement { channel, child } = placement;
        let (writer, mut reader) = channel.split();
        let (events, _) = broadcast::channel::<ManageEvent>(256);
        let handler: HandlerSlot = Arc::new(Mutex::new(None));

        // The parent-side cut reader: relay the child's events up, and serve its brokered store and
        // escalated-request traffic against the parent's authority.
        let out = events.clone();
        let store_for_reader = store.clone();
        let writer_for_reader = writer.clone();
        let handler_for_reader = handler.clone();
        tokio::spawn(async move {
            while let Some(bytes) = reader.recv().await {
                let Some(frame) = decode(&bytes) else {
                    continue;
                };
                match frame {
                    CutFrame::Event(ev) => {
                        let _ = out.send(ev);
                    }
                    CutFrame::StoreCall { id, call } => {
                        let body = serve_store_call(store_for_reader.as_ref(), call).await;
                        let _ = writer_for_reader
                            .send(&encode(&CutFrame::StoreReply { id, body }))
                            .await;
                    }
                    CutFrame::Request { id, req } => {
                        let installed = handler_for_reader.lock().unwrap().clone();
                        let body = match installed {
                            Some(h) => h.request(req).await.body,
                            None => ManageResponseBody::Unsupported,
                        };
                        let _ = writer_for_reader
                            .send(&encode(&CutFrame::RequestReply { id, body }))
                            .await;
                    }
                    _ => {}
                }
            }
        });

        Self {
            id,
            writer,
            store,
            events,
            handler,
            child: Arc::new(AsyncMutex::new(child)),
        }
    }

    /// Drive the placed child to run `session` under an explicit `fence`. The parent is the fence
    /// authority: a stale fence is rejected by the brokered store when the child commits. Used by
    /// `command(Assign)` with a freshly-acquired fence, and directly to exercise the stale case.
    pub async fn activate_under(
        &self,
        session: SessionId,
        fence: FenceToken,
    ) -> std::io::Result<()> {
        self.writer
            .send(&encode(&CutFrame::RunTurn { session, fence }))
            .await
    }
}

#[async_trait]
impl ManagedUnit for PlacedUnit {
    fn id(&self) -> UnitId {
        self.id.clone()
    }

    fn kind(&self) -> UnitKind {
        UnitKind::Engine
    }

    async fn command(&self, cmd: ManageCommand) -> Ack {
        match cmd {
            ManageCommand::Assign { .. } => {
                let session = SessionId::new(self.id.as_str());
                // The parent (placement authority) acquires the lease; the child commits under it.
                let fence = match self.store.acquire_activation_lease(&session).await {
                    Ok(fence) => fence,
                    Err(e) => return Ack::Rejected { reason: e.to_string() },
                };
                match self.activate_under(session, fence).await {
                    Ok(()) => Ack::Accepted,
                    Err(e) => Ack::Rejected {
                        reason: e.to_string(),
                    },
                }
            }
            ManageCommand::Cancel { reason } => {
                let _ = self
                    .writer
                    .send(&encode(&CutFrame::Cancel { reason }))
                    .await;
                Ack::Accepted
            }
            ManageCommand::Shutdown { .. } => {
                let _ = self.writer.send(&encode(&CutFrame::Shutdown)).await;
                self.child.lock().await.shutdown().await;
                Ack::Accepted
            }
            ManageCommand::Snapshot { .. } => Ack::Accepted,
            // No-ops at a single conversation (supervision §4 mapping table).
            ManageCommand::Pause | ManageCommand::Resume | ManageCommand::Scale { .. } => {
                Ack::Unsupported
            }
            _ => Ack::Unsupported,
        }
    }

    fn events(&self) -> EventStream<ManageEvent> {
        EventStream::new(self.events.subscribe())
    }

    fn install_request_handler(&self, handler: Arc<dyn ManageRequestHandler>) {
        *self.handler.lock().unwrap() = Some(handler);
    }
}

// ---------------------------------------------------------------------------
// Child-side loop
// ---------------------------------------------------------------------------

async fn emit(writer: &CutWriter, event: ManageEvent) {
    let _ = writer.send(&encode(&CutFrame::Event(event))).await;
}

/// The placed-child loop (the far side of the cut). Builds a [`RemoteStoreClient`] over `channel`
/// and drives the engine through an ordinary [`ActivationManager`] under the parent-granted fence,
/// so the full `load -> run -> checkpoint` cycle — and its fence check — runs verbatim, just with
/// every durable operation brokered back to the parent. Runs until the parent closes the cut.
pub async fn run_placed_child(
    channel: CutChannel,
    factory: Arc<dyn EngineFactory>,
    partition: PartitionId,
) {
    let (writer, mut reader) = channel.split();
    let client = Arc::new(RemoteStoreClient::new(writer.clone()));
    let manager = ActivationManager::new(client.clone() as Arc<dyn SessionStore>, factory, partition);

    while let Some(bytes) = reader.recv().await {
        let Some(frame) = decode(&bytes) else {
            continue;
        };
        match frame {
            CutFrame::StoreReply { id, body } => client.complete(id, body),
            CutFrame::RunTurn { session, fence } => {
                let manager = manager.clone();
                let writer = writer.clone();
                // The activation drives store calls back over the cut whose replies arrive on THIS
                // reader loop, so it must run concurrently — awaiting it inline would deadlock.
                tokio::spawn(async move {
                    emit(
                        &writer,
                        ManageEvent::Started {
                            seq: 0,
                            trigger: StartTrigger::Resumed,
                        },
                    )
                    .await;
                    let event = match manager.activate(session, fence).await {
                        Ok(()) => ManageEvent::Finished {
                            seq: 1,
                            outcome: Outcome::ended(EndReason::Completed),
                        },
                        Err(SubErr::Store(StoreError::Fenced { .. })) => ManageEvent::Error {
                            seq: 1,
                            failure: FailureView::new(
                                FailureClass::Cancelled,
                                "fenced across the cut",
                            ),
                        },
                        Err(e) => ManageEvent::Error {
                            seq: 1,
                            failure: FailureView::new(FailureClass::Internal, e.to_string()),
                        },
                    };
                    emit(&writer, event).await;
                });
            }
            CutFrame::Cancel { .. } | CutFrame::Shutdown => break,
            _ => {}
        }
    }
}
