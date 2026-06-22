//! Unit tests for the inbound gate's decision table over a mock [`NodeApi`] that records every
//! submitted command. No engine, no host — the gate's busy state is driven directly via
//! `note_turn_started` / `note_turn_finished`, exactly as an adapter drives it from the outbound
//! turn lifecycle.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    ApiError, ControlApi, FleetReport, HealthReport, ModelApi, NodeApi, Outbound, SessionApi,
    SessionInfo, StatsReport,
};
use daemon_common::{SessionId, UnitId};
use daemon_ingest::{AmbientPolicy, BusyPolicy, IngestPolicy, Ingestor, Reception};
use daemon_protocol::{
    session_id_for, AgentCommand, HostResponse, IsolationPolicy, Origin, OriginScope, UserMsg,
};

/// Records every command the ingestor submits, via either `submit_routed` (the receive path) or
/// `submit` (the turn-finished flush path).
#[derive(Default)]
struct Recorder {
    commands: Mutex<Vec<AgentCommand>>,
}

#[async_trait]
impl SessionApi for Recorder {
    async fn submit(&self, _: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        self.commands.lock().unwrap().push(command);
        Ok(())
    }
    async fn submit_routed(
        &self,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<SessionId, ApiError> {
        self.commands.lock().unwrap().push(command);
        Ok(session_id_for(&origin, IsolationPolicy::PerThread))
    }
    async fn poll(&self, _: SessionId, _: u32) -> Result<Vec<Outbound>, ApiError> {
        Ok(Vec::new())
    }
    async fn respond(&self, _: SessionId, _: HostResponse) -> Result<(), ApiError> {
        Ok(())
    }
}

#[async_trait]
impl ControlApi for Recorder {
    async fn health(&self) -> HealthReport {
        HealthReport {
            all_ok: true,
            services: Vec::new(),
        }
    }
    async fn stats(&self) -> StatsReport {
        StatsReport::default()
    }
    async fn sessions(&self) -> Vec<SessionInfo> {
        Vec::new()
    }
    async fn assign(&self, _: SessionId) -> Result<(), ApiError> {
        Ok(())
    }
    async fn cancel(&self, _: SessionId) -> Result<(), ApiError> {
        Ok(())
    }
    async fn fleet(&self) -> FleetReport {
        FleetReport::default()
    }
    async fn unit(&self, _: UnitId) -> Option<daemon_api::UnitNode> {
        None
    }
}

impl ModelApi for Recorder {}
impl daemon_api::ProfileApi for Recorder {}
impl daemon_api::CredentialApi for Recorder {}

fn origin(chat: &str) -> Origin {
    Origin::new(
        "matrix/@bot:hs",
        OriginScope::Group {
            chat: chat.into(),
            thread: None,
        },
    )
}

fn addressed(chat: &str, text: &str) -> Reception {
    Reception {
        origin: origin(chat),
        input: UserMsg::new(text),
        addressed: true,
    }
}

fn ambient(chat: &str, text: &str) -> Reception {
    Reception {
        origin: origin(chat),
        input: UserMsg::new(text),
        addressed: false,
    }
}

fn ingestor(api: &Arc<Recorder>, policy: IngestPolicy) -> Ingestor {
    let api: Arc<dyn NodeApi> = api.clone();
    Ingestor::with_policy(api, policy)
}

#[tokio::test]
async fn idle_addressed_starts_turn() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default());

    ing.receive(addressed("#a", "hello")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 1);
    assert!(matches!(
        &cmds[0],
        AgentCommand::StartTurn { input, .. } if input.text == "hello"
    ));
}

#[tokio::test]
async fn idle_ambient_observes_by_default() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default());

    ing.receive(ambient("#a", "chatter")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 1);
    assert!(matches!(
        &cmds[0],
        AgentCommand::Observe { input, .. } if input.text == "chatter"
    ));
}

#[tokio::test]
async fn busy_ambient_still_observes() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default());
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    ing.note_turn_started(&session);
    ing.receive(ambient("#a", "mid-turn chatter")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 1, "ambient surfaces as Observe even while busy");
    assert!(matches!(&cmds[0], AgentCommand::Observe { .. }));
}

#[tokio::test]
async fn ambient_fold_prepends_into_next_start_turn() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(
        &api,
        IngestPolicy {
            ambient: AmbientPolicy::Fold,
            ..IngestPolicy::default()
        },
    );

    // Two ambient messages buffer silently in Fold mode (no submit).
    ing.receive(ambient("#a", "ctx-1")).await.unwrap();
    ing.receive(ambient("#a", "ctx-2")).await.unwrap();
    assert!(api.commands.lock().unwrap().is_empty(), "fold buffers, no submit");

    // The next addressed message opens a turn carrying the folded context, oldest first.
    ing.receive(addressed("#a", "do it")).await.unwrap();
    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 1);
    match &cmds[0] {
        AgentCommand::StartTurn { input, .. } => {
            assert_eq!(input.text, "ctx-1\nctx-2\ndo it");
        }
        other => panic!("expected StartTurn, got {other:?}"),
    }
}

#[tokio::test]
async fn busy_queue_holds_then_flushes_on_finish() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default()); // BusyPolicy::Queue
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    // First addressed message opens a turn.
    ing.receive(addressed("#a", "first")).await.unwrap();
    ing.note_turn_started(&session);

    // Two more addressed messages arrive mid-turn: queued, nothing submitted yet.
    ing.receive(addressed("#a", "second")).await.unwrap();
    ing.receive(addressed("#a", "third")).await.unwrap();
    assert_eq!(api.commands.lock().unwrap().len(), 1, "only the first StartTurn so far");

    // Turn finishes: the queued messages flush as a single follow-up StartTurn.
    ing.note_turn_finished(&session).await.unwrap();
    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 2);
    match &cmds[1] {
        AgentCommand::StartTurn { input, .. } => assert_eq!(input.text, "second\nthird"),
        other => panic!("expected flushed StartTurn, got {other:?}"),
    }
}

#[tokio::test]
async fn busy_queue_empty_finish_submits_nothing() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default());
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    ing.receive(addressed("#a", "first")).await.unwrap();
    ing.note_turn_started(&session);
    ing.note_turn_finished(&session).await.unwrap();

    assert_eq!(
        api.commands.lock().unwrap().len(),
        1,
        "no queued input -> no flush turn"
    );
}

#[tokio::test]
async fn busy_interrupt_interrupts_then_starts() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(
        &api,
        IngestPolicy {
            busy: BusyPolicy::Interrupt,
            ..IngestPolicy::default()
        },
    );
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    ing.note_turn_started(&session);
    ing.receive(addressed("#a", "urgent")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 2);
    assert!(matches!(&cmds[0], AgentCommand::Interrupt { .. }));
    assert!(matches!(
        &cmds[1],
        AgentCommand::StartTurn { input, .. } if input.text == "urgent"
    ));
}

#[tokio::test]
async fn busy_steer_injects_steer() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(
        &api,
        IngestPolicy {
            busy: BusyPolicy::Steer,
            ..IngestPolicy::default()
        },
    );
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    ing.note_turn_started(&session);
    ing.receive(addressed("#a", "also consider X")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    assert_eq!(cmds.len(), 1);
    assert!(matches!(
        &cmds[0],
        AgentCommand::Steer { text, .. } if text == "also consider X"
    ));
}

#[tokio::test]
async fn fold_buffer_respects_ring_cap() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(
        &api,
        IngestPolicy {
            ambient: AmbientPolicy::Fold,
            queue_cap: 2,
            ..IngestPolicy::default()
        },
    );

    // Three ambient messages with a cap of 2: the oldest ("ctx-1") is evicted.
    ing.receive(ambient("#a", "ctx-1")).await.unwrap();
    ing.receive(ambient("#a", "ctx-2")).await.unwrap();
    ing.receive(ambient("#a", "ctx-3")).await.unwrap();
    ing.receive(addressed("#a", "go")).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    match &cmds[0] {
        AgentCommand::StartTurn { input, .. } => assert_eq!(input.text, "ctx-2\nctx-3\ngo"),
        other => panic!("expected StartTurn, got {other:?}"),
    }
}

#[tokio::test]
async fn queue_buffer_respects_ring_cap() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(
        &api,
        IngestPolicy {
            queue_cap: 2,
            ..IngestPolicy::default()
        },
    );
    let session = session_id_for(&origin("#a"), IsolationPolicy::PerThread);

    ing.receive(addressed("#a", "first")).await.unwrap();
    ing.note_turn_started(&session);
    // Three queued while busy, cap 2: "q1" evicted.
    ing.receive(addressed("#a", "q1")).await.unwrap();
    ing.receive(addressed("#a", "q2")).await.unwrap();
    ing.receive(addressed("#a", "q3")).await.unwrap();
    ing.note_turn_finished(&session).await.unwrap();

    let cmds = api.commands.lock().unwrap();
    match &cmds[1] {
        AgentCommand::StartTurn { input, .. } => assert_eq!(input.text, "q2\nq3"),
        other => panic!("expected flushed StartTurn, got {other:?}"),
    }
}

#[tokio::test]
async fn distinct_origins_key_distinct_sessions() {
    let api = Arc::new(Recorder::default());
    let ing = ingestor(&api, IngestPolicy::default());

    let sa = ing.receive(addressed("#a", "hi")).await.unwrap();
    let sb = ing.receive(addressed("#b", "hi")).await.unwrap();
    assert_ne!(sa, sb, "different chats derive different sessions");
}
