// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Account management (wire v35) end to end over the Unix socket, with the Rooms adapter as the
//! grounding consumer. Proves the three node-authoritative behaviors this phase adds:
//!
//! 1. `transport_disconnect` is reversible — `transport_connect` re-spawns the owning family's
//!    supervised serve loop (observed via the L3 `events_page` serve-start `TransportChanged`).
//! 2. The desired enabled state persists and is honored at spawn — a fresh node over the SAME
//!    store skips a disabled family (its `spawn_adapters` returns no handle), and the per-instance
//!    `enabled` flag is surfaced regardless.
//! 3. Per-instance + per-credential human labels round-trip and overlay in `transport_instances` /
//!    `credential_list`.

use super::harness::*;
use daemon_api::{ConnectionState, NodeEvent};
use daemon_protocol::TransportId;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn account_management_transport_over_socket() {
    // Rooms persist to the durable store (InMemoryStore's `room_*` are no-ops), so use sqlite — and
    // a durable store also lets a second node reopen the SAME state to prove the enabled-skip.
    let dir = std::env::temp_dir().join(format!("daemon-acctmgmt-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store: Arc<dyn SessionStore> =
        Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x3e; 32], fast_host_config());

    let rooms_cfg = daemon_rooms::RoomsConfig {
        enabled: true,
        max_turns: 8,
    };
    let registry = daemon_host::AdapterRegistry::new().with_adapter(
        daemon_rooms::RoomsAdapter::new(store.clone(), signer, rooms_cfg),
    );
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters().await;
    assert_eq!(
        adapter_tasks.len(),
        1,
        "the enabled rooms family is spawned at boot"
    );

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());
    let room = TransportId::new("room");

    // Serve start: the rooms instance reports Connected on the feed.
    await_transport_state(&node, 0, "room", ConnectionState::Connected).await;

    // (3) Label round-trips + overlays onto TransportInstanceInfo (with enabled still true).
    match client
        .call(ApiRequest::TransportSetLabel {
            transport: room.clone(),
            label: Some("My Rooms".into()),
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let instances = match client.call(ApiRequest::TransportInstances).await.unwrap() {
        ApiResponse::TransportInstances(v) => v,
        other => panic!("expected TransportInstances, got {other:?}"),
    };
    let room_info = instances
        .iter()
        .find(|i| i.transport == room)
        .expect("room instance enumerated");
    assert_eq!(
        room_info.label.as_deref(),
        Some("My Rooms"),
        "label overlaid"
    );
    assert!(room_info.enabled, "enabled defaults true with no disable");

    // (1) Reversible disconnect -> connect. Disconnect aborts the family serve loop (Offline push);
    // connect re-spawns it (a fresh serve-start Connected push past the disconnect cursor).
    let before_disconnect = node.events_page(0, 0).await.head_cursor;
    match client
        .call(ApiRequest::TransportDisconnect {
            transport: room.clone(),
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    await_transport_state(&node, before_disconnect, "room", ConnectionState::Offline).await;
    let before_connect = node.events_page(0, 0).await.head_cursor;
    match client
        .call(ApiRequest::TransportConnect {
            transport: room.clone(),
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    // The serve loop resumed: a new Connected push arrives past the connect cursor.
    await_transport_state(&node, before_connect, "room", ConnectionState::Connected).await;
    // Idempotent: a second connect while already running is a clean no-op Ok.
    assert!(matches!(
        client
            .call(ApiRequest::TransportConnect {
                transport: room.clone(),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // (2) Disable persists; a fresh node over the SAME store skips the (now fully-disabled) family.
    match client
        .call(ApiRequest::TransportSetEnabled {
            transport: room.clone(),
            enabled: false,
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let instances = match client.call(ApiRequest::TransportInstances).await.unwrap() {
        ApiResponse::TransportInstances(v) => v,
        other => panic!("expected TransportInstances, got {other:?}"),
    };
    assert!(
        !instances
            .iter()
            .find(|i| i.transport == room)
            .expect("room instance")
            .enabled,
        "disable persisted + overlaid onto the instance"
    );

    // A fresh NodeApiImpl over the same store re-reads the disabled desire and skips the family.
    let AssembledNode {
        node: node2,
        handle: handle2,
        signer: signer2,
        ..
    } = assemble_over(store.clone(), 0, [0x3f; 32], fast_host_config());
    let registry2 =
        daemon_host::AdapterRegistry::new().with_adapter(daemon_rooms::RoomsAdapter::new(
            store.clone(),
            signer2,
            daemon_rooms::RoomsConfig {
                enabled: true,
                max_turns: 8,
            },
        ));
    node2.set_adapters(registry2);
    let tasks2 = node2.spawn_adapters().await;
    assert!(
        tasks2.is_empty(),
        "a fully-disabled family is skipped at spawn on the fresh node"
    );
    let instances2 = node2.transport_instances().await;
    assert!(
        !instances2
            .iter()
            .find(|i| i.transport == room)
            .expect("room instance on fresh node")
            .enabled,
        "the fresh node surfaces the persisted disabled state"
    );

    // Re-enabling on the fresh node persists + (re)connects: the family now serves (a serve-start
    // Connected push appears on node2's feed, which had none while the family was skipped).
    let before_enable = node2.events_page(0, 0).await.head_cursor;
    node2
        .transport_set_enabled(room.clone(), true)
        .await
        .unwrap();
    let instances2 = node2.transport_instances().await;
    assert!(
        instances2
            .iter()
            .find(|i| i.transport == room)
            .expect("room instance")
            .enabled,
        "re-enable persisted"
    );
    await_transport_state(&node2, before_enable, "room", ConnectionState::Connected).await;

    server.abort();
    drop(handle);
    drop(handle2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn account_management_credential_label_over_socket() {
    // A node wired with a credential store (so `credential_list` returns a row to overlay).
    let (node, handle) = assemble_demo();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Store a credential so it appears (redacted) in the list.
    client
        .call(ApiRequest::CredentialSet {
            profile: "acct-1".into(),
            secret: "s3cr3t-value".into(),
        })
        .await
        .unwrap();
    // Set a human label, then confirm credential_list overlays it.
    match client
        .call(ApiRequest::CredentialSetLabel {
            profile: "acct-1".into(),
            label: Some("Home".into()),
        })
        .await
        .unwrap()
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
    let creds = match client.call(ApiRequest::CredentialList).await.unwrap() {
        ApiResponse::Credentials(v) => v,
        other => panic!("expected Credentials, got {other:?}"),
    };
    let row = creds
        .iter()
        .find(|c| c.profile == "acct-1")
        .expect("credential listed");
    assert!(row.present, "the secret is present");
    assert_eq!(row.label.as_deref(), Some("Home"), "label overlaid");

    // Clearing the label removes the overlay.
    client
        .call(ApiRequest::CredentialSetLabel {
            profile: "acct-1".into(),
            label: None,
        })
        .await
        .unwrap();
    let creds = match client.call(ApiRequest::CredentialList).await.unwrap() {
        ApiResponse::Credentials(v) => v,
        other => panic!("expected Credentials, got {other:?}"),
    };
    assert_eq!(
        creds
            .iter()
            .find(|c| c.profile == "acct-1")
            .expect("credential listed")
            .label,
        None,
        "cleared label is gone"
    );

    server.abort();
    drop(handle);
}
