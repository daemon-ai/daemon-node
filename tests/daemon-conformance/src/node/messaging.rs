// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// The messaging-adapter management surface (daemon-messaging-adapter-spec.md §12.2) end to end
/// over the Unix socket, with the Rooms adapter as the grounding consumer and a Matrix adapter
/// registered alongside it to prove the interface generalizes (two adapters, different capability
/// subsets, no host changes). Exercises the full vertical slice: registry-driven lifecycle,
/// `Conv*`/`Member*` CBOR ops, store persistence, the floor-gated `ConvSend` fan-out opening a
/// turn on the invited member's session, and the sealed dCBOR management audit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_adapter_rooms_manage_over_socket() {
    use daemon_common::{JournalStreamId, UnitId};
    use daemon_protocol::{TransportId, UserMsg};

    // Rooms persist to the durable store (InMemoryStore's `room_*` are no-ops), so use sqlite.
    let dir = std::env::temp_dir().join(format!("daemon-rooms-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store: Arc<dyn SessionStore> =
        Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x5d; 32], fast_host_config());

    // Register the Rooms adapter (enabled) + a Matrix adapter (off; enumeration only), then drive
    // lifecycle from the node exactly as `bins/daemon` does.
    let rooms_cfg = daemon_rooms::RoomsConfig {
        enabled: true,
        max_turns: 8,
    };
    let provisioning: Arc<dyn daemon_host::AccountProvisioning> = node.clone();
    let registry = daemon_host::AdapterRegistry::new()
        .with_adapter(daemon_rooms::RoomsAdapter::new(
            store.clone(),
            signer,
            rooms_cfg,
        ))
        .with_adapter(daemon_matrix::MatrixAdapter::new(
            provisioning,
            daemon_matrix::MatrixConfig::default(),
        ));
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters();

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());
    let room = TransportId::new("room");

    // Two adapters enumerate, with different capability subsets (Matrix has interactive_auth +
    // file_transfer; Rooms does not) — the same interface, no host changes.
    let adapters = match client.call(ApiRequest::TransportAdapters).await.unwrap() {
        ApiResponse::Adapters(a) => a,
        other => panic!("expected Adapters, got {other:?}"),
    };
    let matrix = adapters
        .iter()
        .find(|a| a.family == "matrix")
        .expect("matrix adapter enumerated");
    let rooms = adapters
        .iter()
        .find(|a| a.family == "room")
        .expect("rooms adapter enumerated");
    assert!(
        matrix.capabilities.interactive_auth && !rooms.capabilities.interactive_auth,
        "matrix vs rooms capability subset must differ"
    );

    // ConvCreate("room", …) then ConvList("room") returns it.
    let mut details = daemon_api::CreateConversationDetails::default();
    details.extras.values.insert("id".into(), "r1".into());
    details
        .extras
        .values
        .insert("name".into(), "Room One".into());
    details
        .extras
        .values
        .insert("policy".into(), "addressed_only".into());
    let created = match client
        .call(ApiRequest::ConvCreate {
            transport: room.clone(),
            details,
        })
        .await
        .unwrap()
    {
        ApiResponse::Conversation(Some(info)) => info,
        other => panic!("expected Conversation, got {other:?}"),
    };
    assert_eq!(created.id, "r1");
    let convs = match client
        .call(ApiRequest::ConvList {
            transport: room.clone(),
            after: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::Conversations(page) => page.items,
        other => panic!("expected Conversations, got {other:?}"),
    };
    assert!(convs.iter().any(|c| c.id == "r1"), "created room is listed");

    // ConvSetTopic reflects in ConvGet.
    client
        .call(ApiRequest::ConvSetTopic {
            transport: room.clone(),
            conv: "r1".into(),
            topic: Some("standup".into()),
        })
        .await
        .unwrap();
    let got = conv_get(&client, &room, "r1").await;
    assert_eq!(got.topic.as_deref(), Some("standup"));

    // MemberInvite reflects in ConvGet.members with a bound session.
    let who = daemon_api::Participant::Agent {
        profile: ProfileRef::new("openai"),
        member: "@bot".into(),
    };
    assert!(matches!(
        client
            .call(ApiRequest::MemberInvite(daemon_api::MemberInviteArgs {
                transport: room.clone(),
                conv: "r1".into(),
                who: who.clone(),
                message: None,
            }))
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let got = conv_get(&client, &room, "r1").await;
    let member = got
        .members
        .iter()
        .find(|m| m.contact.id == "@bot")
        .expect("invited member present");
    let member_session = member.session.clone().expect("member bound to a session");

    // ConvSend addressed to that member opens a turn on its session (the floor-gated fan-out).
    assert!(matches!(
        client
            .call(ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                transport: room.clone(),
                conv: "r1".into(),
                from: None,
                message: UserMsg::new("hey @bot please help"),
            }))
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let opened = {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut opened = false;
        while Instant::now() < deadline {
            if let ApiResponse::Drained(items) = client
                .call(ApiRequest::Poll {
                    session: member_session.clone(),
                    max: 0,
                })
                .await
                .unwrap()
            {
                if !items.is_empty() {
                    opened = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        opened
    };
    assert!(
        opened,
        "ConvSend to an addressed member must open a turn on that member's session"
    );

    // MemberRemove drops them from ConvGet.members.
    assert!(matches!(
        client
            .call(ApiRequest::MemberRemove(daemon_api::MemberRemoveArgs {
                transport: room.clone(),
                conv: "r1".into(),
                who,
                reason: None,
            }))
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let got = conv_get(&client, &room, "r1").await;
    assert!(
        !got.members.iter().any(|m| m.contact.id == "@bot"),
        "removed member is gone"
    );

    // TransportInstances enumerates the room instance.
    let instances = match client.call(ApiRequest::TransportInstances).await.unwrap() {
        ApiResponse::TransportInstances(i) => i,
        other => panic!("expected TransportInstances, got {other:?}"),
    };
    assert!(
        instances.iter().any(|i| i.transport.as_str() == "room"),
        "room instance enumerated"
    );

    // A mutating op produced a sealed dCBOR entry on the `node-management` stream.
    let seg = store
        .load_trace_segment(&JournalStreamId::unit(&UnitId::new("node-management")), 0)
        .await;
    assert!(
        seg.map(|s| !s.entries.is_empty()).unwrap_or(false),
        "a management mutation must seal a dCBOR entry on the node-management stream"
    );

    // --- Cascading multi-agent conversation (RoundRobin) + merged transcript + delete ---
    let mut rr = daemon_api::CreateConversationDetails::default();
    rr.extras.values.insert("id".into(), "r2".into());
    rr.extras.values.insert("name".into(), "Round Robin".into());
    rr.extras
        .values
        .insert("policy".into(), "round_robin".into());
    assert!(matches!(
        client
            .call(ApiRequest::ConvCreate {
                transport: room.clone(),
                details: rr
            })
            .await
            .unwrap(),
        ApiResponse::Conversation(Some(_))
    ));
    for member in ["@alice", "@bob"] {
        let who = daemon_api::Participant::Agent {
            profile: ProfileRef::new("openai"),
            member: member.into(),
        };
        assert!(matches!(
            client
                .call(ApiRequest::MemberInvite(daemon_api::MemberInviteArgs {
                    transport: room.clone(),
                    conv: "r2".into(),
                    who,
                    message: None,
                }))
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
    }
    let r2 = conv_get(&client, &room, "r2").await;
    let sessions: Vec<_> = r2
        .members
        .iter()
        .filter_map(|m| m.session.clone())
        .collect();
    assert_eq!(sessions.len(), 2, "two agent members bound");

    // An operator post kicks off the round-robin cascade: member A opens a turn; its reply
    // re-injects to member B; and so on, bounded by `max_turns`. Both member sessions must turn.
    assert!(matches!(
        client
            .call(ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                transport: room.clone(),
                conv: "r2".into(),
                from: None,
                message: UserMsg::new("kick off the discussion"),
            }))
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let opened = {
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut opened = std::collections::HashSet::new();
        while Instant::now() < deadline && opened.len() < sessions.len() {
            for s in &sessions {
                if let ApiResponse::Drained(items) = client
                    .call(ApiRequest::Poll {
                        session: s.clone(),
                        max: 0,
                    })
                    .await
                    .unwrap()
                {
                    if !items.is_empty() {
                        opened.insert(s.clone());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        opened.len()
    };
    assert_eq!(
        opened, 2,
        "the round-robin cascade must re-inject a reply and open a turn on both member sessions"
    );

    // The merged room transcript records every post (operator + agent replies), verified.
    let history = match client
        .call(ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
            transport: room.clone(),
            conv: "r2".into(),
            after_cursor: 0,
            max: 0,
        }))
        .await
        .unwrap()
    {
        ApiResponse::Journal(page) => page,
        other => panic!("expected Journal, got {other:?}"),
    };
    let blocks = history
        .entries
        .iter()
        .filter(|e| matches!(e.payload, daemon_api::JournalRecordPayload::Block { .. }))
        .count();
    assert!(
        blocks >= 2,
        "room transcript must contain the operator post + >=1 agent reply, got {blocks}"
    );
    assert!(
        history.entries.iter().all(|e| e.verified),
        "every transcript block must verify against the node signer"
    );

    // Delete the room: it disappears from `get`.
    assert!(matches!(
        client
            .call(ApiRequest::ConvDelete {
                transport: room.clone(),
                conv: "r2".into()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    assert!(
        matches!(
            client
                .call(ApiRequest::ConvGet {
                    transport: room.clone(),
                    conv: "r2".into()
                })
                .await
                .unwrap(),
            ApiResponse::Conversation(None)
        ),
        "deleted room is gone from get"
    );

    server.abort();
    for task in &adapter_tasks {
        task.abort();
    }
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Wire page bound (v25): `ConvList` over a transport holding more than `WIRE_PAGE_MAX`
/// conversations is served in cursor pages through real dispatch/CBOR — 70 rooms page as 64 + 6,
/// the `next` cursor chains the pages, and the union is exactly the full set with no dup or gap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_list_pages_beyond_the_wire_bound() {
    use daemon_api::WIRE_PAGE_MAX;
    use daemon_protocol::TransportId;

    let dir = std::env::temp_dir().join(format!("daemon-rooms-page-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store: Arc<dyn SessionStore> =
        Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble_over(store.clone(), 0, [0x5e; 32], fast_host_config());

    let rooms_cfg = daemon_rooms::RoomsConfig {
        enabled: true,
        max_turns: 8,
    };
    let registry = daemon_host::AdapterRegistry::new().with_adapter(
        daemon_rooms::RoomsAdapter::new(store.clone(), signer, rooms_cfg),
    );
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters();

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());
    let room = TransportId::new("room");

    // 70 conversations, ids chosen so the id (cursor) order is deterministic.
    let total = WIRE_PAGE_MAX + 6;
    for i in 0..total {
        let mut details = daemon_api::CreateConversationDetails::default();
        details
            .extras
            .values
            .insert("id".into(), format!("pg-{i:03}"));
        details
            .extras
            .values
            .insert("name".into(), format!("Page Room {i}"));
        assert!(matches!(
            client
                .call(ApiRequest::ConvCreate {
                    transport: room.clone(),
                    details,
                })
                .await
                .unwrap(),
            ApiResponse::Conversation(Some(_))
        ));
    }

    // Walk the pages through real dispatch/CBOR: sizes 64 then 6, cursor-chained.
    let mut sizes = Vec::new();
    let mut all = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let page = match client
            .call(ApiRequest::ConvList {
                transport: room.clone(),
                after: after.take(),
            })
            .await
            .unwrap()
        {
            ApiResponse::Conversations(page) => page,
            other => panic!("expected Conversations, got {other:?}"),
        };
        assert!(
            page.items.len() <= WIRE_PAGE_MAX,
            "a wire page must never exceed WIRE_PAGE_MAX, got {}",
            page.items.len()
        );
        sizes.push(page.items.len());
        all.extend(page.items.into_iter().map(|c| c.id));
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    assert_eq!(sizes, vec![WIRE_PAGE_MAX, 6], "70 rooms page as 64 + 6");
    let expected: Vec<String> = (0..total).map(|i| format!("pg-{i:03}")).collect();
    assert_eq!(all, expected, "pages chain without dup or gap, in id order");

    server.abort();
    for task in &adapter_tasks {
        task.abort();
    }
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}
