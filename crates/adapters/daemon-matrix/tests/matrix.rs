//! Vertical tests for the Matrix adapter over a wiremock-backed homeserver (`MatrixMockServer`) and
//! a recording mock [`NodeApi`] (no real engine/host). They cover the three seams:
//!   - **login/restore round-trip**: a `MatrixSession` survives the credential blob serialization.
//!   - **inbound**: a synced `m.room.message` -> `Origin` -> `StartTurn` through the real handler.
//!   - **outbound**: a `TurnFinished` with `final_text` -> a real `m.room.message` send.
//!   - **delivery dedup**: the incremental `DeliveryManager` subscribes each session at most once.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;

use daemon_api::{
    ApiError, ControlApi, FleetReport, HealthReport, LogStream, ModelApi, NodeApi, Outbound,
    SessionApi, SessionInfo, StatsReport,
};
use daemon_common::{SessionId, UnitId, UsageDelta};
use daemon_delivery::Projector;
use daemon_ingest::{IngestPolicy, Ingestor};
use daemon_matrix::{DeliveryManager, InboundCtx, MatrixProjector};
use daemon_protocol::{
    session_id_for, AgentCommand, AgentEvent, DeliveryTarget, Direction, Disposition, EndReason,
    HostResponse, IsolationPolicy, Origin, OriginScope, SessionLogEntry, SessionPayload, SinkKind,
    TransportId, TurnSummary,
};

use matrix_sdk::ruma::{device_id, event_id, room_id, user_id};
use matrix_sdk::test_utils::mocks::{LoginResponseTemplate200, MatrixMockServer};
use matrix_sdk_test::event_factory::EventFactory;
use matrix_sdk_test::JoinedRoomBuilder;

/// A recording mock node: captures every submitted command and answers the two delivery primitives
/// with a single `Primary` target on `transport`/`route`. `pending_subscribe` makes `subscribe`
/// hang (so a `DeliveryManager` task stays alive for the dedup assertion).
struct Recorder {
    commands: Mutex<Vec<AgentCommand>>,
    transport: TransportId,
    route: String,
    pending_subscribe: bool,
}

impl Recorder {
    fn new(transport: &str, route: &str) -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
            transport: TransportId::new(transport),
            route: route.to_string(),
            pending_subscribe: false,
        }
    }

    fn pending(transport: &str, route: &str) -> Self {
        Self {
            pending_subscribe: true,
            ..Self::new(transport, route)
        }
    }
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
    async fn subscribe(&self, _: SessionId, _: u64) -> Result<LogStream, ApiError> {
        if self.pending_subscribe {
            Ok(futures::stream::pending().boxed())
        } else {
            Ok(futures::stream::empty().boxed())
        }
    }
    async fn delivery_sessions(&self, _: TransportId) -> Vec<SessionId> {
        Vec::new()
    }
    async fn delivery_targets(&self, _: SessionId) -> Vec<DeliveryTarget> {
        vec![DeliveryTarget::new(
            self.transport.as_str(),
            &self.route,
            SinkKind::Primary,
        )]
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
impl daemon_api::AuthApi for Recorder {}

fn perthread_policy() -> IngestPolicy {
    IngestPolicy {
        isolation: IsolationPolicy::PerThread,
        ..IngestPolicy::default()
    }
}

#[tokio::test]
async fn session_blob_round_trips() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    let session = client
        .matrix_auth()
        .session()
        .expect("mock client is logged in");

    let stored = daemon_matrix::StoredSession {
        homeserver: "https://hs.example".to_string(),
        session: session.clone(),
    };
    let blob = stored.to_blob().unwrap();
    let back = daemon_matrix::StoredSession::from_blob(&blob).unwrap();

    assert_eq!(back.homeserver, "https://hs.example");
    assert_eq!(back.session.meta.user_id, session.meta.user_id);
    assert_eq!(back.session.meta.device_id, session.meta.device_id);
}

#[tokio::test]
async fn inbound_room_message_opens_a_turn() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;
    let me = client.user_id().expect("logged in").to_owned();

    let transport = "matrix/@bot:localhost";
    let api = Arc::new(Recorder::new(transport, "!room:localhost"));
    let napi: Arc<dyn NodeApi> = api.clone();
    let ingestor = Arc::new(Ingestor::with_policy(napi.clone(), perthread_policy()));
    let projector = Arc::new(MatrixProjector::new(
        napi.clone(),
        ingestor.clone(),
        HashMap::new(),
    ));
    let delivery = Arc::new(DeliveryManager::new(napi.clone(), projector));

    let ctx = InboundCtx {
        ingestor,
        delivery,
        routes: Arc::new(Vec::new()),
        bare: "@bot:localhost".to_string(),
        transport: TransportId::new(transport),
        me,
    };
    client.add_event_handler_context(ctx);
    client.add_event_handler(daemon_matrix::on_room_message);

    let room = room_id!("!room:localhost");
    let factory = EventFactory::new();
    server
        .sync_room(
            &client,
            JoinedRoomBuilder::new(room).add_timeline_event(
                factory
                    .text_msg("!ping please")
                    .sender(user_id!("@alice:localhost")),
            ),
        )
        .await;

    // Handlers are dispatched during sync processing; a short tick covers the receive's await chain.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let cmds = api.commands.lock().unwrap();
    assert!(
        cmds.iter().any(|c| matches!(
            c,
            AgentCommand::StartTurn { input, .. } if input.text.contains("!ping please")
        )),
        "expected a StartTurn carrying the message, got {cmds:?}"
    );
}

#[tokio::test]
async fn outbound_turn_finished_posts_reply() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;

    let room = room_id!("!room:localhost");
    // Make the client aware of the joined room (with a seed event so it materializes in the state
    // store) so `get_room` resolves for the reply.
    let factory = EventFactory::new();
    server
        .sync_room(
            &client,
            JoinedRoomBuilder::new(room).add_timeline_event(
                factory.text_msg("seed").sender(user_id!("@alice:localhost")),
            ),
        )
        .await;
    // Expect exactly one send; verified when the mock server drops at end of test.
    server
        .mock_room_send()
        .ok(event_id!("$evt:localhost"))
        .expect(1)
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    let api = Arc::new(Recorder::new(transport.as_str(), room.as_str()));
    let napi: Arc<dyn NodeApi> = api.clone();
    let ingestor = Arc::new(Ingestor::with_policy(napi.clone(), perthread_policy()));
    let mut clients = HashMap::new();
    clients.insert(transport.clone(), client.clone());
    let projector = MatrixProjector::new(napi.clone(), ingestor, clients);

    let entry = SessionLogEntry {
        seq: 1,
        direction: Direction::Outbound,
        origin: Origin::new("engine", OriginScope::Api { key: "k".into() }),
        disposition: Disposition::Context,
        payload: SessionPayload::Event(AgentEvent::TurnFinished {
            seq: 1,
            summary: TurnSummary {
                end_reason: EndReason::Completed,
                final_text: Some("the reply".to_string()),
                usage: UsageDelta::default(),
            },
        }),
    };
    projector.project(SessionId::new("s1"), entry).await;
}

/// The two-step SSO seam (`daemon-interactive-auth-spec`, proven for the matrix family): `sso_begin`
/// mints the homeserver authorization URL against the caller-owned redirect, and `sso_complete`
/// finishes from the captured `loginToken`, producing the persistable session blob + identity. Both
/// run against the wiremock homeserver (versions + login mocked); no live engine/host is involved.
#[tokio::test]
async fn sso_begin_mints_url_then_complete_persists_session() {
    let server = MatrixMockServer::new().await;
    server.mock_versions().ok().mount().await;
    server
        .mock_login()
        .ok_with(LoginResponseTemplate200::new(
            "sso-access-token",
            device_id!("SSODEVICE"),
            user_id!("@bot:localhost"),
        ))
        .mount()
        .await;

    let store_root = std::env::temp_dir().join(format!(
        "daemon-matrix-sso-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let redirect_uri = "http://127.0.0.1:65000/cb";

    // begin: build the on-disk client + mint the SSO authorization URL pointing at our redirect.
    let session = daemon_matrix::sso_begin(
        &store_root,
        &server.uri(),
        "matrix-bot",
        redirect_uri,
        None,
    )
    .await
    .expect("sso_begin mints an authorization url");
    assert!(
        session
            .authorization_url
            .contains("/_matrix/client/v3/login/sso/redirect"),
        "authorization url targets the homeserver SSO redirect: {}",
        session.authorization_url
    );
    assert!(
        session.authorization_url.contains("redirectUrl="),
        "authorization url carries our redirect: {}",
        session.authorization_url
    );

    // complete: finish from the captured `loginToken` (a full callback URL is accepted).
    let callback = format!("{redirect_uri}?loginToken=secret-login-token");
    let login = daemon_matrix::sso_complete(session, &callback)
        .await
        .expect("sso_complete exchanges the loginToken");

    assert_eq!(login.user_id, "@bot:localhost");
    assert_eq!(login.credential_ref, "matrix-bot");
    assert_eq!(login.transport_instance.as_str(), "matrix/@bot:localhost");
    let back = daemon_matrix::StoredSession::from_blob(&login.credential_blob)
        .expect("the persisted blob round-trips");
    assert_eq!(back.session.meta.user_id.as_str(), "@bot:localhost");

    let _ = std::fs::remove_dir_all(&store_root);
}

#[tokio::test]
async fn delivery_manager_dedups_sessions() {
    let transport = TransportId::new("matrix/@bot:localhost");
    let api = Arc::new(Recorder::pending(transport.as_str(), "!room:localhost"));
    let napi: Arc<dyn NodeApi> = api.clone();
    let ingestor = Arc::new(Ingestor::with_policy(napi.clone(), perthread_policy()));
    let projector = Arc::new(MatrixProjector::new(napi.clone(), ingestor, HashMap::new()));
    let delivery = Arc::new(DeliveryManager::new(napi.clone(), projector));

    let s1 = SessionId::new("s1");
    delivery.ensure(s1.clone(), transport.clone());
    delivery.ensure(s1.clone(), transport.clone()); // duplicate -> ignored
    delivery.ensure(SessionId::new("s2"), transport.clone());

    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        delivery.active_count(),
        2,
        "two distinct sessions delivered, duplicate ignored"
    );
}
