// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Phase 4: integration test crate; raw fs/reqwest/Command are expected in tests.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

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

use matrix_sdk::ruma::{device_id, event_id, mxc_uri, room_id, user_id};
use matrix_sdk::test_utils::mocks::{LoginResponseTemplate200, MatrixMockServer};
use matrix_sdk_test::event_factory::EventFactory;
use matrix_sdk_test::JoinedRoomBuilder;

use daemon_api::{
    ChatMessage, ContactInfo, ConvChange, ConvSendArgs, DisconnectReason, FileTransfer,
    LifecycleSink, MembershipChange, Participant, SupportsConversations, TransportAdapter,
};
use daemon_host::{AccountProvisioning, BlobStore, FileBlobStore, ProvisionedAccount};
use daemon_matrix::MatrixAdapter;
use daemon_protocol::UserMsg;

/// A no-op provisioning seam: the file-transfer tests resolve the live client from the seeded
/// registry (via `register_live_client`), never from provisioning.
struct MockProvisioning;

impl AccountProvisioning for MockProvisioning {
    fn bound_accounts(&self, _family: &str) -> Vec<ProvisionedAccount> {
        Vec::new()
    }
    fn account_credential(&self, _credential_ref: &str) -> Option<String> {
        None
    }
    fn store_account_credential(&self, _credential_ref: &str, _blob: &str) -> Result<(), ApiError> {
        Ok(())
    }
}

fn blob_root(tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "daemon-matrix-ft-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    root
}

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
    async fn delivery_sessions(
        &self,
        _: TransportId,
        _: Option<String>,
    ) -> daemon_api::WirePage<SessionId> {
        daemon_api::WirePage::default()
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
impl daemon_api::AccessControlApi for Recorder {}

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
        sink: None,
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
                factory
                    .text_msg("seed")
                    .sender(user_id!("@alice:localhost")),
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
                failure: None,
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
    let session =
        daemon_matrix::sso_begin(&store_root, &server.uri(), "matrix-bot", redirect_uri, None)
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

/// W2-H: `SupportsFileTransfer::send` reads the blob's bytes from the node store and uploads them to
/// the Matrix content repository (`POST /_matrix/media/v3/upload`, mocked).
#[tokio::test]
async fn file_transfer_send_uploads_media() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server
        .mock_upload()
        .ok(mxc_uri!("mxc://localhost/uploaded"))
        .expect(1)
        .mount()
        .await;

    let root = blob_root("send");
    let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&root).unwrap());
    let blob = blobs.put(b"the outbound file").await.unwrap();

    let transport = TransportId::new("matrix/@bot:localhost");
    let adapter =
        MatrixAdapter::with_blobs(Arc::new(MockProvisioning), Default::default(), None, blobs);
    adapter
        .register_live_client(transport.clone(), client)
        .await;

    assert!(TransportAdapter::info(&*adapter).capabilities.file_transfer);
    let ft = adapter
        .messaging()
        .unwrap()
        .file_transfer()
        .expect("blobs wired ⟹ file transfer present");
    assert!(ft.supported().send && ft.supported().receive);

    let transfer = FileTransfer {
        name: "cat.png".into(),
        blob,
        content_type: Some("image/png".into()),
        ..Default::default()
    };
    ft.send(transport, transfer)
        .await
        .expect("media upload succeeds against the mock");

    let _ = std::fs::remove_dir_all(&root);
}

/// W2-H: `SupportsFileTransfer::receive` downloads the `source` `mxc://` content
/// (`GET /_matrix/client/v1/media/download/...`, mocked) and stores it back into the node store.
#[tokio::test]
async fn file_transfer_receive_downloads_media() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server
        .mock_authed_media_download()
        .ok_bytes(b"the inbound file".to_vec())
        .expect(1)
        .mount()
        .await;

    let root = blob_root("recv");
    let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&root).unwrap());

    let transport = TransportId::new("matrix/@bot:localhost");
    let adapter = MatrixAdapter::with_blobs(
        Arc::new(MockProvisioning),
        Default::default(),
        None,
        blobs.clone(),
    );
    adapter
        .register_live_client(transport.clone(), client)
        .await;
    let ft = adapter
        .messaging()
        .unwrap()
        .file_transfer()
        .expect("file transfer present");

    let transfer = FileTransfer {
        name: "in.png".into(),
        source: Some("mxc://localhost/inbound".into()),
        ..Default::default()
    };
    ft.receive(transport, transfer)
        .await
        .expect("media download succeeds against the mock");

    // The downloaded bytes are now resident in the node blob store.
    let expected_root = blob_root("recv-expected");
    let expected: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(&expected_root).unwrap());
    let expected_ref = expected.put(b"the inbound file").await.unwrap();
    assert!(
        blobs.has(&expected_ref.hash).await,
        "receive stored the downloaded content in the node store"
    );

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&expected_root);
}

/// N4 (wire vNEXT — conversation hierarchy): a synced `m.space` room projects to
/// `ConversationType::Space` through the public conversation projection (`SupportsConversations::get`
/// → `room_to_info`). The room type is sourced from the SDK's `Room::is_space()` (the `m.space`
/// `m.room.create` `type`), fabricated here with the harness's `EventFactory::create(..).with_space_type()`.
#[tokio::test]
async fn space_room_projects_as_space_conversation() {
    use daemon_api::ConversationType;
    use matrix_sdk::ruma::RoomVersionId;

    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;
    let creator = client.user_id().expect("logged in").to_owned();

    let space = room_id!("!space:localhost");
    let factory = EventFactory::new().sender(&creator);
    server
        .sync_room(
            &client,
            JoinedRoomBuilder::new(space).add_state_event(
                factory
                    .create(&creator, RoomVersionId::V1)
                    .with_space_type(),
            ),
        )
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    let adapter = MatrixAdapter::new(Arc::new(MockProvisioning), Default::default(), None);
    adapter
        .register_live_client(transport.clone(), client)
        .await;
    let convs = adapter
        .clone()
        .messaging()
        .unwrap()
        .conversations()
        .unwrap();

    let info = convs
        .get(transport, space.as_str().to_string())
        .await
        .expect("the space room resolves");
    assert_eq!(info.kind, ConversationType::Space);
    assert_eq!(info.parent, None, "a top-level space is a hierarchy root");
}

/// N4 (wire vNEXT): a child room carries its containing space as `parent`, derived from the SDK's
/// `Room::parent_spaces()` (the `m.space.parent` state relation). Only the child is synced, so the
/// SDK reports the parent as `ParentSpace::Unverifiable(space_id)` — the projection still emits the
/// id, and (dangling/unknown parents being a client concern) the node reports what the protocol says.
#[tokio::test]
async fn child_room_carries_parent_space() {
    use daemon_api::ConversationType;
    use matrix_sdk::ruma::events::space::parent::SpaceParentEventContent;
    use matrix_sdk::ruma::server_name;

    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;

    let space = room_id!("!space:localhost");
    let child = room_id!("!child:localhost");
    let factory = EventFactory::new().sender(user_id!("@bot:localhost"));
    let parent = SpaceParentEventContent::new(vec![server_name!("localhost").to_owned()]);
    server
        .sync_room(
            &client,
            JoinedRoomBuilder::new(child)
                .add_timeline_event(factory.text_msg("seed"))
                .add_state_event(factory.event(parent).state_key(space.to_string())),
        )
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    let adapter = MatrixAdapter::new(Arc::new(MockProvisioning), Default::default(), None);
    adapter
        .register_live_client(transport.clone(), client)
        .await;
    let convs = adapter
        .clone()
        .messaging()
        .unwrap()
        .conversations()
        .unwrap();

    let info = convs
        .get(transport, child.as_str().to_string())
        .await
        .expect("the child room resolves");
    assert_eq!(
        info.parent.as_deref(),
        Some(space.as_str()),
        "child advertises its containing space as parent"
    );
    assert_ne!(
        info.kind,
        ConversationType::Space,
        "a child room is not itself a space"
    );
}

/// N4 (wire vNEXT): a plain (non-space, no `m.space.parent`) room projects with `parent == None` and
/// a non-`Space` kind — proving the new field is only populated when the protocol actually reports a
/// hierarchy relation.
#[tokio::test]
async fn plain_room_has_no_parent() {
    use daemon_api::ConversationType;

    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;

    let room = room_id!("!plain:localhost");
    let factory = EventFactory::new();
    server
        .sync_room(
            &client,
            JoinedRoomBuilder::new(room).add_timeline_event(
                factory
                    .text_msg("hello")
                    .sender(user_id!("@alice:localhost")),
            ),
        )
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    let adapter = MatrixAdapter::new(Arc::new(MockProvisioning), Default::default(), None);
    adapter
        .register_live_client(transport.clone(), client)
        .await;
    let convs = adapter
        .clone()
        .messaging()
        .unwrap()
        .conversations()
        .unwrap();

    let info = convs
        .get(transport, room.as_str().to_string())
        .await
        .expect("the plain room resolves");
    assert_eq!(info.parent, None);
    assert_ne!(info.kind, ConversationType::Space);
}

/// A recording node-lifecycle sink: captures every `chat_message` report (the wire-vNEXT journal
/// obligation seam) so the vertical tests can assert the adapter reports each outbound send and
/// inbound delivery exactly once, with a properly populated [`ChatMessage`].
#[derive(Default)]
struct RecordingSink {
    chats: Mutex<Vec<(TransportId, String, ChatMessage)>>,
}

#[async_trait]
impl LifecycleSink for RecordingSink {
    async fn transport_disconnected(
        &self,
        _transport: TransportId,
        _reason: DisconnectReason,
        _message: Option<String>,
    ) {
    }
    async fn conversations_changed(
        &self,
        _transport: TransportId,
        _conv: String,
        _change: ConvChange,
    ) {
    }
    #[allow(clippy::too_many_arguments)]
    async fn membership_changed(
        &self,
        _transport: TransportId,
        _conv: String,
        _member: String,
        _change: MembershipChange,
        _actor: Option<String>,
        _reason: Option<String>,
        _is_self: bool,
    ) {
    }
    async fn chat_message(&self, transport: TransportId, conv: String, message: ChatMessage) {
        self.chats.lock().unwrap().push((transport, conv, message));
    }
}

/// The journal obligation on the matrix send path (wire vNEXT): a successful `ConvSend` reports
/// exactly one `chat_message` through the node sink — author = the `from` participant, RAW text,
/// the server-acked event id as `ChatMessage::id`, delivered stamped — for the node to journal on
/// `conv:<transport>:<room>` and announce via `MessagesChanged`.
#[tokio::test]
async fn send_reports_chat_message_through_the_sink() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    server.mock_room_state_encryption().plain().mount().await;
    let room = room_id!("!room:localhost");
    server.sync_joined_room(&client, room).await;
    server
        .mock_room_send()
        .ok(event_id!("$evt:localhost"))
        .expect(1)
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    let sink = Arc::new(RecordingSink::default());
    let lifecycle: Arc<dyn LifecycleSink> = sink.clone();
    let adapter = MatrixAdapter::new(
        Arc::new(MockProvisioning),
        Default::default(),
        Some(lifecycle),
    );
    adapter
        .register_live_client(transport.clone(), client)
        .await;

    let author = Participant::Contact(ContactInfo {
        id: "@op:localhost".into(),
        ..ContactInfo::default()
    });
    SupportsConversations::send(
        &*adapter,
        ConvSendArgs {
            transport: transport.clone(),
            conv: room.as_str().to_string(),
            from: Some(author.clone()),
            message: UserMsg::new("hello"),
        },
    )
    .await
    .expect("send succeeds against the mock");

    let chats = sink.chats.lock().unwrap();
    assert_eq!(
        chats.len(),
        1,
        "one successful send = one chat_message report, got {chats:?}"
    );
    let (t, conv, msg) = &chats[0];
    assert_eq!(t, &transport);
    assert_eq!(conv, room.as_str());
    assert_eq!(msg.text, "hello");
    assert_eq!(
        msg.author,
        Some(author),
        "the ConvSend `from` attribution rides ChatMessage::author"
    );
    assert_eq!(
        msg.id.as_deref(),
        Some("$evt:localhost"),
        "the server-acked event id rides ChatMessage::id"
    );
    assert!(msg.timestamp.is_some(), "the send stamps a timestamp");
    assert!(msg.delivered(), "a server-acked send is stamped delivered");
}

/// The journal obligation on the matrix inbound path (wire vNEXT): a synced `m.room.message`
/// reports one `chat_message` through the node sink — structured author (the sender MXID), RAW
/// body, the matrix event id — in ADDITION to the existing agent-session `Ingestor` routing
/// (the `StartTurn` still fires; journaling never replaces it).
#[tokio::test]
async fn inbound_room_message_reports_chat_message_through_the_sink() {
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
    let sink = Arc::new(RecordingSink::default());
    let lifecycle: Arc<dyn LifecycleSink> = sink.clone();

    let ctx = InboundCtx {
        ingestor,
        delivery,
        routes: Arc::new(Vec::new()),
        bare: "@bot:localhost".to_string(),
        transport: TransportId::new(transport),
        me,
        sink: Some(lifecycle),
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

    // Handlers dispatch during sync processing; poll briefly for the async report to land.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while sink.chats.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let chats = sink.chats.lock().unwrap();
    assert_eq!(
        chats.len(),
        1,
        "one inbound message = one chat_message report, got {chats:?}"
    );
    let (t, conv, msg) = &chats[0];
    assert_eq!(t.as_str(), transport);
    assert_eq!(conv, room.as_str());
    assert_eq!(
        msg.text, "!ping please",
        "the journal carries the RAW body — attribution is structured, never text-prefixed"
    );
    assert!(
        matches!(&msg.author, Some(Participant::Contact(c)) if c.id == "@alice:localhost"),
        "the sender MXID rides ChatMessage::author, got {:?}",
        msg.author
    );
    assert!(
        msg.id.is_some(),
        "the matrix event id rides ChatMessage::id"
    );
    assert!(
        msg.timestamp.is_some(),
        "origin_server_ts rides ChatMessage::timestamp"
    );

    // The existing Ingestor routing is untouched: the addressed message still opened a turn.
    let cmds = api.commands.lock().unwrap();
    assert!(
        cmds.iter().any(|c| matches!(
            c,
            AgentCommand::StartTurn { input, .. } if input.text.contains("!ping please")
        )),
        "journaling is in ADDITION to ingest routing, got {cmds:?}"
    );
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
