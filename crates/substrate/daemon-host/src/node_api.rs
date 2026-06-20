//! [`NodeApiImpl`] — the node's [`daemon_api`] surface implemented over the running host.
//!
//! This is the one place the abstract interface ([`daemon_api::NodeApi`]) is bound to concrete
//! substrate machinery. Every transport (in-process, the Unix socket, the C FFI pump) ultimately
//! reaches *this* object; they differ only in how bytes arrive.
//!
//! - The **control sub-surface** ([`daemon_api::ControlApi`]) projects the durable node: the
//!   resident-service health ([`SupervisorObserver`]), durable queue/session stats and the session
//!   roster ([`SessionStore`]), session assignment (`ActivationManager::wake`, create-if-absent),
//!   and the orchestration fleet (via the injected [`crate::FleetView`]).
//! - The **session sub-surface** ([`daemon_api::SessionApi`]) drives live interactive engine
//!   sessions through the §17 actor ([`spawn_agent_session`]). Each session owns a drain buffer fed
//!   by the actor's event broadcast and a parked-request table so a poll-based embedder (the FFI)
//!   sees events *and* blocking host requests on one queue and answers them with `respond`.

use crate::supervisor::{HealthStatus, SupervisorObserver};
use crate::FleetView;
use async_trait::async_trait;
use daemon_activation::ActivationManager;
use daemon_api::{
    ApiError, ControlApi, FleetReport, HealthReport, Outbound, ServiceHealth, SessionApi,
    SessionInfo, SessionState, StatsReport,
};
use daemon_common::{PartitionId, ReqId, SessionId, UnitId};
use daemon_core::{spawn_agent_session, AgentHandle, Engine, Snapshot};
use daemon_protocol::{
    AgentCommand, HostRequest, HostRequestHandler, HostResponse, HostResponseBody,
};
use daemon_store::{SessionStatus, SessionStore};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;

/// Builds a fresh live [`Engine`] for an interactive session id (the session sub-surface's engine
/// seam — the binary supplies the provider/tools/system).
pub type SessionEngineBuilder = Arc<dyn Fn(SessionId) -> Engine + Send + Sync>;

/// The node interface implemented over a running [`crate::Host`].
#[derive(Clone)]
pub struct NodeApiImpl {
    supervisor: SupervisorObserver,
    store: Arc<dyn SessionStore>,
    manager: ActivationManager,
    fleet: Option<Arc<dyn FleetView>>,
    partition: PartitionId,
    live: Arc<LiveSessions>,
}

impl NodeApiImpl {
    /// Assemble the node surface over the running substrate.
    ///
    /// - `supervisor` is the [`SupervisorObserver`] from `host.start().observer()`.
    /// - `engine_builder` builds a fresh engine for each interactive (session sub-surface) session.
    /// - `fleet` is the optional control-surface fleet projection (`None` => empty fleet report).
    pub fn new(
        supervisor: SupervisorObserver,
        store: Arc<dyn SessionStore>,
        manager: ActivationManager,
        partition: PartitionId,
        engine_builder: SessionEngineBuilder,
        fleet: Option<Arc<dyn FleetView>>,
    ) -> Self {
        Self {
            supervisor,
            store,
            manager,
            fleet,
            partition,
            live: Arc::new(LiveSessions::new(engine_builder)),
        }
    }
}

fn map_state(status: SessionStatus) -> SessionState {
    match status {
        SessionStatus::Active => SessionState::Active,
        SessionStatus::Suspended { job_id } => SessionState::Suspended {
            job_id: job_id.to_string(),
        },
        SessionStatus::Ready => SessionState::Ready,
        SessionStatus::Completed => SessionState::Completed,
    }
}

#[async_trait]
impl ControlApi for NodeApiImpl {
    async fn health(&self) -> HealthReport {
        let services = self
            .supervisor
            .service_names()
            .into_iter()
            .map(|name| {
                let restarts = self.supervisor.restarts(&name).unwrap_or(0);
                let (ok, detail) = match self.supervisor.health(&name) {
                    Some(HealthStatus::Ok) => (true, None),
                    Some(HealthStatus::Degraded { reason })
                    | Some(HealthStatus::Unhealthy { reason }) => (false, Some(reason)),
                    None => (false, Some("unknown service".to_string())),
                };
                ServiceHealth {
                    name,
                    ok,
                    restarts,
                    detail,
                }
            })
            .collect();
        HealthReport {
            all_ok: self.supervisor.all_ok(),
            services,
        }
    }

    async fn stats(&self) -> StatsReport {
        let s = self.store.stats().await;
        StatsReport {
            pending_jobs: s.pending_jobs as u64,
            pending_wakes: s.pending_wakes as u64,
            sessions: s.sessions as u64,
            active: self.manager.active_count() as u64,
        }
    }

    async fn sessions(&self) -> Vec<SessionInfo> {
        self.store
            .list_sessions()
            .await
            .into_iter()
            .map(|(session, status)| SessionInfo {
                session,
                state: map_state(status),
            })
            .collect()
    }

    async fn assign(&self, session: SessionId) -> Result<(), ApiError> {
        // Create-if-absent: a fresh durable session row with the engine's initial snapshot.
        if self.store.status(&session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            self.store
                .create_session(session.clone(), self.partition, blob)
                .await
                .map_err(|e| ApiError::Other(format!("create session: {e}")))?;
        }
        // Wake it: the activation manager runs (or resumes) the engine; the resident services then
        // carry the durable delegate -> suspend -> resume -> complete cycle forward.
        self.manager
            .wake(session)
            .await
            .map_err(|e| ApiError::Other(format!("wake: {e}")))
    }

    async fn cancel(&self, session: SessionId) -> Result<(), ApiError> {
        // Best-effort: cancel a matching fleet child and interrupt a matching live session.
        if let Some(fleet) = &self.fleet {
            fleet.cancel(&UnitId::new(session.as_str())).await;
        }
        self.live.interrupt(&session).await;
        Ok(())
    }

    async fn fleet(&self) -> FleetReport {
        match &self.fleet {
            Some(fleet) => fleet.report().await,
            None => FleetReport::default(),
        }
    }
}

#[async_trait]
impl SessionApi for NodeApiImpl {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        self.live.submit(session, command).await
    }

    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        self.live.poll(&session, max)
    }

    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError> {
        self.live.respond(&session, response)
    }
}

// ---------------------------------------------------------------------------
// Live interactive sessions (the §17 actor, exposed via the poll/drain model)
// ---------------------------------------------------------------------------

type Drain = Arc<Mutex<VecDeque<Outbound>>>;
type Pending = Arc<Mutex<HashMap<ReqId, oneshot::Sender<HostResponse>>>>;

struct LiveSession {
    handle: AgentHandle,
    drain: Drain,
    pending: Pending,
    /// The event pump task; aborted when the session is dropped.
    pump: JoinHandle<()>,
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.pump.abort();
    }
}

struct LiveSessions {
    sessions: DashMap<SessionId, LiveSession>,
    builder: SessionEngineBuilder,
}

impl LiveSessions {
    fn new(builder: SessionEngineBuilder) -> Self {
        Self {
            sessions: DashMap::new(),
            builder,
        }
    }

    /// Spawn (or reuse) the actor for `session`, returning its handle.
    fn ensure(&self, session: &SessionId) -> AgentHandle {
        if let Some(s) = self.sessions.get(session) {
            return s.handle.clone();
        }
        let engine = (self.builder)(session.clone());
        let drain: Drain = Arc::new(Mutex::new(VecDeque::new()));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let host = Arc::new(ParkingHandler {
            drain: drain.clone(),
            pending: pending.clone(),
        });
        let handle = spawn_agent_session(engine, host);

        // Pump §17 events from the actor broadcast into the drain queue (lossless until polled).
        let mut rx = handle.subscribe();
        let pump_drain = drain.clone();
        let pump = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => pump_drain.lock().unwrap().push_back(Outbound::Event(ev)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        self.sessions.insert(
            session.clone(),
            LiveSession {
                handle: handle.clone(),
                drain,
                pending,
                pump,
            },
        );
        handle
    }

    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        match command {
            AgentCommand::StartTurn { input, .. } => {
                // Opening command: spawn-if-absent, then run the turn in the background so events
                // (including the terminal `TurnFinished`) flow to the drain queue for `poll`.
                let handle = self.ensure(&session);
                tokio::spawn(async move {
                    let _ = handle.start_turn(input).await;
                });
                Ok(())
            }
            AgentCommand::Interrupt { reason } => {
                let handle = self.existing(&session)?;
                handle.interrupt(reason).await;
                Ok(())
            }
            AgentCommand::Shutdown => {
                if let Some((_, s)) = self.sessions.remove(&session) {
                    s.handle.shutdown().await;
                }
                Ok(())
            }
            AgentCommand::Steer { .. } => Err(ApiError::Unsupported(
                "Steer is not yet wired through the session actor".into(),
            )),
            AgentCommand::Snapshot { .. } => Err(ApiError::Unsupported(
                "Snapshot is not yet wired through the session actor".into(),
            )),
            _ => Err(ApiError::Unsupported("unknown agent command".into())),
        }
    }

    fn existing(&self, session: &SessionId) -> Result<AgentHandle, ApiError> {
        self.sessions
            .get(session)
            .map(|s| s.handle.clone())
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))
    }

    fn poll(&self, session: &SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut q = s.drain.lock().unwrap();
        let take = if max == 0 { q.len() } else { (max as usize).min(q.len()) };
        Ok(q.drain(..take).collect())
    }

    fn respond(&self, session: &SessionId, response: HostResponse) -> Result<(), ApiError> {
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let tx = s.pending.lock().unwrap().remove(&response.request_id);
        match tx {
            Some(tx) => {
                let _ = tx.send(response);
                Ok(())
            }
            None => Err(ApiError::Other(format!(
                "no parked request {:?} on session {}",
                response.request_id, session
            ))),
        }
    }

    async fn interrupt(&self, session: &SessionId) -> bool {
        match self.sessions.get(session) {
            Some(s) => {
                s.handle.interrupt(Some("control cancel".into())).await;
                true
            }
            None => false,
        }
    }
}

/// The session sub-surface's host handler: park each blocking §17 request into the drain queue and
/// a pending table, await its `respond`. Events and parked requests thus ride one ordered queue
/// (daemon-ffi-spec §3.3).
struct ParkingHandler {
    drain: Drain,
    pending: Pending,
}

#[async_trait]
impl HostRequestHandler for ParkingHandler {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let (tx, rx) = oneshot::channel();
        let request_id = req.request_id;
        self.pending.lock().unwrap().insert(request_id, tx);
        self.drain
            .lock()
            .unwrap()
            .push_back(Outbound::Request(req));
        match rx.await {
            Ok(resp) => resp,
            // The session was dropped before an answer arrived: decline safely.
            Err(_) => HostResponse {
                request_id,
                body: HostResponseBody::Approved(false),
            },
        }
    }
}
