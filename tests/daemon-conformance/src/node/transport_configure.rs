// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Transport account settings (N2; wire v38) end to end over the Unix socket: the
//! `TransportSettings` read + `TransportConfigure` merge-edit of a transport instance's persisted
//! NON-SECRET settings values. Proves the node-authoritative behaviors this package adds:
//!
//! 1. Configure merge-persists into the same per-transport prefs store the label/enabled ops use,
//!    and `TransportSettings` reads the merged map back (upsert semantics: unspecified keys keep
//!    their value).
//! 2. Keys outside the owning adapter's `account_schema` are rejected with a clear error and
//!    nothing is persisted.
//! 3. An adapter `validate_account` failure (the testkit Fake's marker value) is surfaced and
//!    nothing is persisted.
//! 4. Apply-by-reconnect: configuring a currently-connected instance cycles disconnect → connect,
//!    observable as the existing `TransportChanged` Offline + serve-start Connected pushes on the
//!    L3 events feed (no new event type).
//! 5. Settings survive an instance/node restart: a fresh node over the SAME durable store reads
//!    the persisted values back.
//!
//! SECURITY INVARIANT (documented on the ops): secrets never live in this settings store — they
//! go to the credential store via the interactive-auth flows; these ops carry only non-secret
//! configuration.

use super::harness::*;
use daemon_api::{
    AccountSettingsSchema, AccountSettingsValues, AdapterCapabilities, AdapterInfo, AuthParamField,
    ConnectionState, MessagingProtocol, NodeApi, NodeEvent, PresenceState, TransportAdapter,
    TransportInstanceInfo,
};
use daemon_api_testkit::FakeProtocol;
use daemon_protocol::TransportId;
use std::collections::BTreeMap;

/// The N2 configure-target mock — the `transport_presence.rs` MockTransport evolved into a
/// [`MessagingProtocol`]: one Connected instance, a parked serve loop (abort = disconnect,
/// re-spawn = reconnect), a two-key account schema, and `validate_account` delegated to
/// daemon-api-testkit's [`FakeProtocol`] (which rejects its marker value).
struct ConfigurableMock {
    fake: Arc<FakeProtocol>,
}

impl ConfigurableMock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fake: FakeProtocol::new(),
        })
    }
}

#[async_trait::async_trait]
impl TransportAdapter for ConfigurableMock {
    fn family(&self) -> &str {
        "mock"
    }

    fn info(&self) -> AdapterInfo {
        AdapterInfo {
            family: "mock".to_string(),
            display_name: "Mock transport".to_string(),
            capabilities: AdapterCapabilities::default(),
            account_schema: AccountSettingsSchema {
                fields: vec![
                    AuthParamField {
                        key: "server".into(),
                        label: "Server".into(),
                        required: true,
                        ..Default::default()
                    },
                    AuthParamField {
                        key: "nick".into(),
                        label: "Nickname".into(),
                        required: false,
                        ..Default::default()
                    },
                ],
            },
            ..Default::default()
        }
    }

    async fn serve(self: Arc<Self>, _api: Arc<dyn NodeApi>) {
        // Park forever: the supervisor owns the task, so a disconnect aborts it and a reconnect
        // re-enters serve (emitting the serve-start `TransportChanged`).
        futures::future::pending::<()>().await;
    }

    async fn instances(&self) -> Vec<TransportInstanceInfo> {
        vec![TransportInstanceInfo {
            transport: TransportId::new("mock/acct"),
            family: "mock".into(),
            display_name: "mock account".into(),
            connection: ConnectionState::Connected,
            presence: PresenceState::Unknown,
            bound_profile: None,
            reason: None,
            message: None,
            fatal: false,
            enabled: true,
            label: None,
        }]
    }

    fn messaging(self: Arc<Self>) -> Option<Arc<dyn MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl MessagingProtocol for ConfigurableMock {
    async fn validate_account(
        &self,
        settings: &AccountSettingsValues,
    ) -> Result<(), daemon_api::ApiError> {
        // Delegate to the testkit reference fake: marker-value settings fail validation.
        self.fake.validate_account(settings).await
    }
}

/// Poll `events_page` from `after` until a `TransportChanged` for `transport` with `want`
/// connection appears (or a 10s deadline elapses).
async fn await_transport_state(
    node: &Arc<NodeApiImpl>,
    after: u64,
    transport: &str,
    want: ConnectionState,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = node.events_page(after, 0).await;
        let hit = page.events.iter().any(|e| {
            matches!(
                e,
                NodeEvent::TransportChanged { transport: t, connection, .. }
                    if t.as_str() == transport && *connection == want
            )
        });
        if hit {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {transport} -> {want:?} past cursor {after}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// One assembled node with the mock adapter spawned and a Unix-socket client on it.
async fn mock_node_over_socket() -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    tokio::task::JoinHandle<()>,
    ApiClient,
) {
    let (node, handle) = assemble();
    node.set_adapters(daemon_host::AdapterRegistry::new().with_adapter(ConfigurableMock::new()));
    let _tasks = node.spawn_adapters().await;
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path);
    (node, handle, server, client)
}

/// The settings values a `TransportSettings` reply carries, or a panic on any other shape.
fn settings_of(resp: ApiResponse) -> BTreeMap<String, String> {
    match resp {
        ApiResponse::TransportSettings(v) => v.values,
        other => panic!("expected TransportSettings, got {other:?}"),
    }
}

/// One key→value map (the configure payloads are tiny).
fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// (1) Read-back + merge semantics: a fresh instance reads empty; each configure upserts its keys
/// over the persisted map; unspecified keys keep their value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_settings_configure_round_trip_over_socket() {
    let (_node, handle, server, client) = mock_node_over_socket().await;
    let mock = TransportId::new("mock/acct");

    // A never-configured instance reads back the empty map (not an error).
    let got = settings_of(
        client
            .call(ApiRequest::TransportSettings {
                transport: mock.clone(),
            })
            .await
            .unwrap(),
    );
    assert!(got.is_empty(), "fresh instance has no settings: {got:?}");

    // First configure persists its keys.
    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock.clone(),
            settings: AccountSettingsValues {
                values: map(&[("server", "hs.example.org")]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let got = settings_of(
        client
            .call(ApiRequest::TransportSettings {
                transport: mock.clone(),
            })
            .await
            .unwrap(),
    );
    assert_eq!(got, map(&[("server", "hs.example.org")]));

    // A second configure with a DIFFERENT key merges (upsert) — `server` survives.
    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock.clone(),
            settings: AccountSettingsValues {
                values: map(&[("nick", "daemon-bot")]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let got = settings_of(
        client
            .call(ApiRequest::TransportSettings { transport: mock })
            .await
            .unwrap(),
    );
    assert_eq!(
        got,
        map(&[("server", "hs.example.org"), ("nick", "daemon-bot")]),
        "configure merges over the persisted map"
    );

    server.abort();
    drop(handle);
}

/// (2) A key outside the adapter's `account_schema` is rejected with an error naming the key, and
/// nothing is persisted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_configure_rejects_unknown_key() {
    let (_node, handle, server, client) = mock_node_over_socket().await;
    let mock = TransportId::new("mock/acct");

    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock.clone(),
            settings: AccountSettingsValues {
                values: map(&[("bogus", "x")]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Error(e) => {
            let text = format!("{e:?}");
            assert!(
                text.contains("bogus"),
                "the error names the unknown key: {text}"
            );
        }
        other => panic!("expected Error for an unknown key, got {other:?}"),
    }
    // Nothing was persisted.
    let got = settings_of(
        client
            .call(ApiRequest::TransportSettings { transport: mock })
            .await
            .unwrap(),
    );
    assert!(got.is_empty(), "a rejected configure persists nothing");

    server.abort();
    drop(handle);
}

/// (3) The adapter's `validate_account` runs over the merged values; its failure (the testkit
/// Fake's marker value) is surfaced and nothing is persisted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_configure_surfaces_validate_account_error() {
    let (_node, handle, server, client) = mock_node_over_socket().await;
    let mock = TransportId::new("mock/acct");

    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock.clone(),
            settings: AccountSettingsValues {
                values: map(&[("nick", FakeProtocol::VALIDATE_REJECT_VALUE)]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Error(e) => {
            let text = format!("{e:?}");
            assert!(
                text.contains("validate_account"),
                "the adapter's validation error is surfaced: {text}"
            );
        }
        other => panic!("expected Error from validate_account, got {other:?}"),
    }
    // Nothing was persisted.
    let got = settings_of(
        client
            .call(ApiRequest::TransportSettings { transport: mock })
            .await
            .unwrap(),
    );
    assert!(got.is_empty(), "a rejected configure persists nothing");

    server.abort();
    drop(handle);
}

/// (4) Apply-by-reconnect: configuring a currently-connected instance cycles disconnect →
/// connect, pushing the existing `TransportChanged` Offline + serve-start Connected events past
/// the pre-configure cursor (mirroring the `TransportSetEnabled(false → true)` cycle).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_configure_reconnects_connected_instance() {
    let (node, handle, server, client) = mock_node_over_socket().await;
    let mock = TransportId::new("mock/acct");

    // Serve start: the instance reports Connected on the feed.
    await_transport_state(&node, 0, "mock/acct", ConnectionState::Connected).await;
    let before = node.events_page(0, 0).await.head_cursor;

    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock,
            settings: AccountSettingsValues {
                values: map(&[("server", "hs2.example.org")]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }

    // The cycle emits Offline (disconnect) then a fresh serve-start Connected — both past the
    // pre-configure cursor, so the Connected here is the RE-connect, not the boot start.
    await_transport_state(&node, before, "mock/acct", ConnectionState::Offline).await;
    await_transport_state(&node, before, "mock/acct", ConnectionState::Connected).await;

    server.abort();
    drop(handle);
}

/// (5) Settings survive an instance/node restart: a fresh node over the SAME durable (sqlite)
/// store reads the persisted values back through `transport_settings`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transport_settings_survive_restart() {
    let dir = std::env::temp_dir().join(format!("daemon-transport-cfg-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store: Arc<dyn SessionStore> =
        Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));

    // Node 1: configure over the socket, then tear the node down.
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x51; 32], fast_host_config());
    node.set_adapters(daemon_host::AdapterRegistry::new().with_adapter(ConfigurableMock::new()));
    let _tasks = node.spawn_adapters().await;
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path);
    let mock = TransportId::new("mock/acct");
    match client
        .call(ApiRequest::TransportConfigure {
            transport: mock.clone(),
            settings: AccountSettingsValues {
                values: map(&[("server", "hs.example.org"), ("nick", "daemon-bot")]),
            },
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    server.abort();
    drop(handle);

    // Node 2 over the SAME store (a fresh adapter registry — the "restarted" instance): the
    // persisted settings read back.
    let AssembledNode {
        node: node2,
        handle: handle2,
        ..
    } = assemble_over(store.clone(), 0, [0x52; 32], fast_host_config());
    node2.set_adapters(daemon_host::AdapterRegistry::new().with_adapter(ConfigurableMock::new()));
    let got = node2
        .transport_settings(mock)
        .await
        .expect("read persisted settings on the fresh node");
    assert_eq!(
        got.values,
        map(&[("server", "hs.example.org"), ("nick", "daemon-bot")]),
        "settings persisted in the prefs store survive the restart"
    );

    drop(handle2);
    let _ = std::fs::remove_dir_all(&dir);
}
