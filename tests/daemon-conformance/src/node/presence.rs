// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! W2-F saved-presence wire surface, end-to-end through the real `daemon_node::assemble` composition
//! root: `PresenceList` / `PresenceSave` / `PresenceSetActive` / `PresenceDelete` over both the
//! in-process trait and the Unix-socket round-trip (transport parity), proving the host
//! `PresenceManager` is wired and the CBOR shapes ride the wire.

use super::harness::*;
use daemon_api::{PresencePrimitive, SavedPresence};

fn presence_list(resp: ApiResponse) -> Vec<SavedPresence> {
    match resp {
        ApiResponse::SavedPresences(list) => list,
        other => panic!("expected SavedPresences, got {other:?}"),
    }
}

/// The freshly-assembled node surfaces the two default presences (Offline + Available), a save adds
/// a third, set-active + delete round-trip, and the in-process surface agrees with the socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn presence_crud_over_socket_and_in_process() {
    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // The two defaults are seeded on the first list (seed-on-read).
    let defaults = presence_list(client.call(ApiRequest::PresenceList).await.unwrap());
    assert_eq!(defaults.len(), 2, "Offline + Available defaults are seeded");
    assert!(defaults
        .iter()
        .any(|p| p.primitive == PresencePrimitive::Offline));
    assert!(defaults
        .iter()
        .any(|p| p.primitive == PresencePrimitive::Available));

    // Save a new named presence (empty id → the node mints one).
    let saved = SavedPresence {
        id: String::new(),
        name: Some("Lunch".into()),
        primitive: PresencePrimitive::Away,
        message: Some("back soon".into()),
        emoji: Some("🥪".into()),
        last_used: None,
        use_count: 0,
    };
    assert!(matches!(
        client
            .call(ApiRequest::PresenceSave { presence: saved })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    let listed = presence_list(client.call(ApiRequest::PresenceList).await.unwrap());
    assert_eq!(listed.len(), 3, "the saved presence is listed");
    let lunch = listed
        .iter()
        .find(|p| p.name.as_deref() == Some("Lunch"))
        .expect("saved presence present");
    assert!(!lunch.id.is_empty(), "the node minted an id");
    let lunch_id = lunch.id.clone();

    // Activating bumps its use-count + last-used.
    assert!(matches!(
        client
            .call(ApiRequest::PresenceSetActive {
                id: lunch_id.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let after = presence_list(client.call(ApiRequest::PresenceList).await.unwrap());
    let lunch = after
        .iter()
        .find(|p| p.id == lunch_id)
        .expect("still present");
    assert_eq!(lunch.use_count, 1, "activation bumped the use-count");
    assert!(lunch.last_used.is_some(), "activation stamped last-used");

    // In-process parity: the trait surface agrees with the socket.
    let inproc = node.presence_list().await;
    assert_eq!(
        inproc.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "presence_list must agree across transports"
    );

    // Delete removes it (idempotent).
    assert!(matches!(
        client
            .call(ApiRequest::PresenceDelete {
                id: lunch_id.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let final_list = presence_list(client.call(ApiRequest::PresenceList).await.unwrap());
    assert_eq!(final_list.len(), 2, "back to the two defaults");
    assert!(!final_list.iter().any(|p| p.id == lunch_id));

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}
