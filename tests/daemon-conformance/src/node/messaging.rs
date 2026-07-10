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
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x5d; 32], fast_host_config());

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
            rooms_cfg,
            Some(node.lifecycle_sink()),
        ))
        .with_adapter(daemon_matrix::MatrixAdapter::new(
            provisioning,
            daemon_matrix::MatrixConfig::default(),
            None,
        ));
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters().await;

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

    // The merged room transcript records every post (operator + agent replies) as rich
    // `JournalRecordPayload::Chat` records (wire v38; the coarse `Block` shape is retired from
    // the conv journal), verified, in append order.
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
    let chats: Vec<&daemon_api::JournalRecord> = history
        .entries
        .iter()
        .filter(|e| matches!(e.payload, daemon_api::JournalRecordPayload::Chat { .. }))
        .collect();
    assert!(
        chats.len() >= 2,
        "room transcript must contain the operator post + >=1 agent reply as Chat records, got {}",
        chats.len()
    );
    assert!(
        history.entries.iter().all(|e| e.verified),
        "every transcript record must verify against the node signer"
    );
    assert!(
        history
            .entries
            .windows(2)
            .all(|w| w[0].cursor < w[1].cursor),
        "history reads back in append order with strictly-increasing cursors"
    );
    // The operator post is first (account-originated: no author), carrying the RAW text —
    // attribution rides the structured `author`, never the body.
    match &chats[0].payload {
        daemon_api::JournalRecordPayload::Chat { message } => {
            assert_eq!(message.text, "kick off the discussion");
            assert_eq!(message.author, None, "operator post has no author");
        }
        other => panic!("expected Chat, got {other:?}"),
    }
    // The re-injected member replies carry the member's structured identity.
    assert!(
        chats[1..].iter().any(|e| matches!(
            &e.payload,
            daemon_api::JournalRecordPayload::Chat { message } if message.author.is_some()
        )),
        "a loopback-delivered member reply is journaled with its author"
    );
    // Every append raised the granular MessagesChanged pointer for (room, r2) on the L3 feed.
    let messages_changed = match client
        .call(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::EventsPage(page) => page
            .events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    daemon_api::NodeEvent::MessagesChanged { transport: t, conv }
                        if t.as_str() == "room" && conv == "r2"
                )
            })
            .count(),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert!(
        messages_changed >= chats.len(),
        "MessagesChanged must be emitted per Chat append (>= {} for r2, got {messages_changed})",
        chats.len()
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

/// A minimal in-test messaging adapter whose single feature is [`SupportsRoster`], backed by an
/// in-memory contact list. It grounds the wire-v34 roster surface (`RosterList`/`RosterAdd`/
/// `RosterUpdate`/`RosterRemove` + the `ContactsChanged` event) end to end through real
/// dispatch/CBOR without a real protocol server — the reference `daemon-rooms`/`daemon-telegram`
/// `SupportsRoster` impls land in Wave 3 (see the plan). Mirrors how the messaging test above uses
/// the Rooms adapter to ground the conv/member surface.
struct RosterMockAdapter {
    contacts: std::sync::Mutex<Vec<daemon_api::ContactInfo>>,
}

impl RosterMockAdapter {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            contacts: std::sync::Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl daemon_api::TransportAdapter for RosterMockAdapter {
    fn family(&self) -> &str {
        "rostmock"
    }
    fn info(&self) -> daemon_api::AdapterInfo {
        daemon_api::AdapterInfo {
            family: "rostmock".into(),
            display_name: "Roster Mock".into(),
            ..Default::default()
        }
    }
    async fn serve(self: Arc<Self>, _api: Arc<dyn daemon_api::NodeApi>) {}
    fn messaging(self: Arc<Self>) -> Option<Arc<dyn daemon_api::MessagingProtocol>> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl daemon_api::MessagingProtocol for RosterMockAdapter {
    fn roster(self: Arc<Self>) -> Option<Arc<dyn daemon_api::SupportsRoster>> {
        Some(self)
    }
}

#[async_trait::async_trait]
impl daemon_api::SupportsRoster for RosterMockAdapter {
    fn supported(&self) -> daemon_api::RosterOps {
        daemon_api::RosterOps {
            list: true,
            add: true,
            update: true,
            remove: true,
        }
    }
    async fn list(&self, _transport: daemon_protocol::TransportId) -> Vec<daemon_api::ContactInfo> {
        self.contacts.lock().unwrap().clone()
    }
    async fn add(
        &self,
        _transport: daemon_protocol::TransportId,
        contact: daemon_api::ContactInfo,
    ) -> Result<(), daemon_api::ApiError> {
        self.contacts.lock().unwrap().push(contact);
        Ok(())
    }
    async fn update(
        &self,
        _transport: daemon_protocol::TransportId,
        contact: daemon_api::ContactInfo,
    ) -> Result<(), daemon_api::ApiError> {
        let mut c = self.contacts.lock().unwrap();
        match c.iter_mut().find(|x| x.id == contact.id) {
            Some(slot) => {
                *slot = contact;
                Ok(())
            }
            None => Err(daemon_api::ApiError::Other("no such contact".into())),
        }
    }
    async fn remove(
        &self,
        _transport: daemon_protocol::TransportId,
        contact: daemon_api::ContactInfo,
    ) -> Result<(), daemon_api::ApiError> {
        self.contacts.lock().unwrap().retain(|x| x.id != contact.id);
        Ok(())
    }
}

/// List a transport's roster over the socket, returning the contacts (panics on the wrong shape).
async fn roster_items(
    client: &ApiClient,
    transport: &daemon_protocol::TransportId,
) -> Vec<daemon_api::ContactInfo> {
    match client
        .call(ApiRequest::RosterList {
            transport: transport.clone(),
            after: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::ContactPage(page) => page.items,
        other => panic!("expected ContactPage, got {other:?}"),
    }
}

/// The wire-v34 server-side roster surface, end to end over the Unix socket against an in-test
/// `SupportsRoster` adapter: the node reports `roster_ops` (all four verbs) in `TransportAdapters`;
/// `RosterAdd`/`RosterUpdate`/`RosterRemove` mutate the roster and `RosterList` reflects each change
/// (sorted + paged, contact-id order); and every successful mutation raises a `ContactsChanged`
/// pointer on the node-wide event feed (the deterministic one-shot `EventsSince` re-read).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_adapter_roster_manage_over_socket() {
    use daemon_api::{ContactInfo, ContactPermission, NodeEvent, Presence};
    use daemon_protocol::TransportId;

    let (node, handle) = assemble();
    let mock = RosterMockAdapter::new();
    let registry = daemon_host::AdapterRegistry::new().with_adapter(mock);
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters().await;

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());
    let transport = TransportId::new("rostmock");

    // The node reports the per-verb roster capabilities from the adapter's `supported()` probe.
    let adapters = match client.call(ApiRequest::TransportAdapters).await.unwrap() {
        ApiResponse::Adapters(a) => a,
        other => panic!("expected Adapters, got {other:?}"),
    };
    let ops = adapters
        .iter()
        .find(|a| a.family == "rostmock")
        .and_then(|a| a.roster_ops)
        .expect("roster mock reports roster_ops");
    assert!(
        ops.list && ops.add && ops.update && ops.remove,
        "the mock reports every roster verb"
    );

    let contact = |id: &str, name: Option<&str>| ContactInfo {
        id: id.into(),
        display_name: name.map(|s| s.into()),
        presence: Presence::default(),
        permission: ContactPermission::Allow,
    };

    // Empty to start.
    assert!(roster_items(&client, &transport).await.is_empty());

    // Add two (inserted out of id order) — the list comes back sorted by contact id.
    for c in [
        contact("@carol:hs", Some("Carol")),
        contact("@bob:hs", Some("Bob")),
    ] {
        assert!(matches!(
            client
                .call(ApiRequest::RosterAdd {
                    transport: transport.clone(),
                    contact: c,
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
    }
    let ids: Vec<String> = roster_items(&client, &transport)
        .await
        .into_iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(ids, vec!["@bob:hs".to_string(), "@carol:hs".to_string()]);

    // Update reflects in the list.
    assert!(matches!(
        client
            .call(ApiRequest::RosterUpdate {
                transport: transport.clone(),
                contact: contact("@bob:hs", Some("Bobby")),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let bob = roster_items(&client, &transport)
        .await
        .into_iter()
        .find(|c| c.id == "@bob:hs")
        .expect("bob present");
    assert_eq!(bob.display_name.as_deref(), Some("Bobby"));

    // Remove drops it.
    assert!(matches!(
        client
            .call(ApiRequest::RosterRemove {
                transport: transport.clone(),
                contact: contact("@bob:hs", None),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let ids: Vec<String> = roster_items(&client, &transport)
        .await
        .into_iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(ids, vec!["@carol:hs".to_string()]);

    // Every successful mutation raised a ContactsChanged pointer for this transport (the
    // deterministic one-shot re-read of the retained node-event feed).
    let saw_contacts_changed = match client
        .call(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::EventsPage(page) => page.events.iter().any(|e| {
            matches!(e, NodeEvent::ContactsChanged { transport: t } if t.as_str() == "rostmock")
        }),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert!(
        saw_contacts_changed,
        "a successful roster mutation must raise ContactsChanged on the node-wide feed"
    );

    server.abort();
    for task in &adapter_tasks {
        task.abort();
    }
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// The wire-v34 server-side roster surface end to end over the Unix socket against the *real*
/// [`daemon_rooms::RoomsAdapter`] `SupportsRoster` impl (not the in-test mock): the node reports the
/// rooms family's `roster_ops` (all four verbs) in `TransportAdapters`; `RosterAdd`/`RosterUpdate`/
/// `RosterRemove` mutate the adapter's in-memory roster and `RosterList` reflects each change (sorted
/// + paged, contact-id order); and every successful mutation raises a `ContactsChanged` pointer for
/// the `room` transport. Grounds the roster surface on the reference adapter, mirroring how
/// `messaging_adapter_rooms_manage_over_socket` grounds the conv/member surface.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn messaging_adapter_rooms_roster_manage_over_socket() {
    use daemon_api::{ContactInfo, ContactPermission, NodeEvent, Presence};
    use daemon_protocol::TransportId;

    // The roster is in-memory, but the adapter still needs a store for its rooms/membership wiring.
    let dir = std::env::temp_dir().join(format!("daemon-rooms-roster-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store: Arc<dyn SessionStore> =
        Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x5f; 32], fast_host_config());

    let rooms_cfg = daemon_rooms::RoomsConfig {
        enabled: true,
        max_turns: 8,
    };
    let registry = daemon_host::AdapterRegistry::new().with_adapter(
        daemon_rooms::RoomsAdapter::new(store.clone(), rooms_cfg, Some(node.lifecycle_sink())),
    );
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters().await;

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());
    let transport = TransportId::new("room");

    // The node reports the rooms family's per-verb roster capabilities from `supported()`.
    let adapters = match client.call(ApiRequest::TransportAdapters).await.unwrap() {
        ApiResponse::Adapters(a) => a,
        other => panic!("expected Adapters, got {other:?}"),
    };
    let ops = adapters
        .iter()
        .find(|a| a.family == "room")
        .and_then(|a| a.roster_ops)
        .expect("rooms adapter reports roster_ops");
    assert!(
        ops.list && ops.add && ops.update && ops.remove,
        "the rooms adapter reports every roster verb"
    );

    let contact = |id: &str, name: Option<&str>| ContactInfo {
        id: id.into(),
        display_name: name.map(|s| s.into()),
        presence: Presence::default(),
        permission: ContactPermission::Allow,
    };

    // Empty to start.
    assert!(roster_items(&client, &transport).await.is_empty());

    // Add two (inserted out of id order) — the list comes back sorted by contact id (host-central).
    for c in [
        contact("agent-charlie", Some("Charlie")),
        contact("agent-alice", Some("Alice")),
    ] {
        assert!(matches!(
            client
                .call(ApiRequest::RosterAdd {
                    transport: transport.clone(),
                    contact: c,
                })
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
    }
    let ids: Vec<String> = roster_items(&client, &transport)
        .await
        .into_iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(
        ids,
        vec!["agent-alice".to_string(), "agent-charlie".to_string()]
    );

    // Adding a duplicate id is refused (the adapter errors; the id already on the roster).
    assert!(matches!(
        client
            .call(ApiRequest::RosterAdd {
                transport: transport.clone(),
                contact: contact("agent-alice", Some("Alice II")),
            })
            .await
            .unwrap(),
        ApiResponse::Error(_)
    ));

    // Update reflects in the list.
    assert!(matches!(
        client
            .call(ApiRequest::RosterUpdate {
                transport: transport.clone(),
                contact: contact("agent-alice", Some("Alice Cooper")),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let alice = roster_items(&client, &transport)
        .await
        .into_iter()
        .find(|c| c.id == "agent-alice")
        .expect("alice present");
    assert_eq!(alice.display_name.as_deref(), Some("Alice Cooper"));

    // Updating a missing id is refused.
    assert!(matches!(
        client
            .call(ApiRequest::RosterUpdate {
                transport: transport.clone(),
                contact: contact("agent-nobody", None),
            })
            .await
            .unwrap(),
        ApiResponse::Error(_)
    ));

    // Remove drops it; removing a missing id is refused.
    assert!(matches!(
        client
            .call(ApiRequest::RosterRemove {
                transport: transport.clone(),
                contact: contact("agent-alice", None),
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::RosterRemove {
                transport: transport.clone(),
                contact: contact("agent-alice", None),
            })
            .await
            .unwrap(),
        ApiResponse::Error(_)
    ));
    let ids: Vec<String> = roster_items(&client, &transport)
        .await
        .into_iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(ids, vec!["agent-charlie".to_string()]);

    // Every successful mutation raised a ContactsChanged pointer for the `room` transport.
    let saw_contacts_changed = match client
        .call(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .unwrap()
    {
        ApiResponse::EventsPage(page) => page.events.iter().any(
            |e| matches!(e, NodeEvent::ContactsChanged { transport: t } if t.as_str() == "room"),
        ),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert!(
        saw_contacts_changed,
        "a successful roster mutation must raise ContactsChanged on the node-wide feed"
    );

    server.abort();
    for task in &adapter_tasks {
        task.abort();
    }
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A node + live Rooms adapter served over a Unix socket — the shared bring-up for the chat-journal
/// tests below (mirrors `messaging_adapter_rooms_manage_over_socket`, which keeps its own inline
/// copy). Rooms persist to the durable store (InMemoryStore's `room_*` are no-ops), so sqlite.
struct RoomsSocket {
    client: ApiClient,
    handle: daemon_host::SupervisorHandle,
    server: tokio::task::JoinHandle<()>,
    adapter_tasks: Vec<tokio::task::JoinHandle<()>>,
    path: std::path::PathBuf,
    dir: std::path::PathBuf,
}

impl RoomsSocket {
    async fn bring_up(tag: &str, seed: [u8; 32]) -> Self {
        let dir = std::env::temp_dir().join(format!("daemon-rooms-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store: Arc<dyn SessionStore> =
            Arc::new(SqliteStore::open(dir.join("store.sqlite")).expect("open sqlite store"));
        let AssembledNode { node, handle, .. } =
            assemble_over(store.clone(), 0, seed, fast_host_config());

        let rooms_cfg = daemon_rooms::RoomsConfig {
            enabled: true,
            max_turns: 8,
        };
        let registry = daemon_host::AdapterRegistry::new().with_adapter(
            daemon_rooms::RoomsAdapter::new(store.clone(), rooms_cfg, Some(node.lifecycle_sink())),
        );
        node.set_adapters(registry);
        let adapter_tasks = node.spawn_adapters().await;

        let path = temp_socket();
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).expect("bind api socket");
        let server = tokio::spawn(serve_api_unix(listener, node.clone()));
        let client = ApiClient::new(path.clone());
        Self {
            client,
            handle,
            server,
            adapter_tasks,
            path,
            dir,
        }
    }

    /// Create a members-less room `id` (journal-only posts: the floor/fan-out half stays inert).
    async fn create_room(&self, id: &str) {
        let mut details = daemon_api::CreateConversationDetails::default();
        details.extras.values.insert("id".into(), id.into());
        details.extras.values.insert("name".into(), id.into());
        assert!(matches!(
            self.client
                .call(ApiRequest::ConvCreate {
                    transport: daemon_protocol::TransportId::new("room"),
                    details,
                })
                .await
                .unwrap(),
            ApiResponse::Conversation(Some(_))
        ));
    }

    /// `ConvSend` to `conv` (Ok expected; the journal append happens on the async serve loop).
    async fn send(&self, conv: &str, from: Option<daemon_api::Participant>, text: &str) {
        use daemon_protocol::UserMsg;
        assert!(matches!(
            self.client
                .call(ApiRequest::ConvSend(daemon_api::ConvSendArgs {
                    transport: daemon_protocol::TransportId::new("room"),
                    conv: conv.into(),
                    from,
                    message: UserMsg::new(text),
                }))
                .await
                .unwrap(),
            ApiResponse::Ok
        ));
    }

    /// One `ConvHistory` page for `conv`.
    async fn history(
        &self,
        conv: &str,
        after_cursor: u64,
        max: u32,
    ) -> daemon_api::JournalPageView {
        match self
            .client
            .call(ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
                transport: daemon_protocol::TransportId::new("room"),
                conv: conv.into(),
                after_cursor,
                max,
            }))
            .await
            .unwrap()
        {
            ApiResponse::Journal(page) => page,
            other => panic!("expected Journal, got {other:?}"),
        }
    }

    /// Poll `ConvHistory` until it holds at least `n` entries, all sealed+verified (the send path
    /// journals on the adapter's async serve loop; the seal follows the append), or the deadline
    /// passes — returning the page either way so the caller's assertions report the actual state.
    async fn history_at_least(&self, conv: &str, n: usize) -> daemon_api::JournalPageView {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let page = self.history(conv, 0, 0).await;
            let settled = page.entries.len() >= n && page.entries.iter().all(|e| e.verified);
            if settled || Instant::now() >= deadline {
                return page;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// The `MessagesChanged` pointers currently on the node-wide feed for `(room, conv)`.
    async fn messages_changed(&self, conv: &str) -> usize {
        match self
            .client
            .call(ApiRequest::EventsSince {
                cursor: 0,
                wait_ms: None,
            })
            .await
            .unwrap()
        {
            ApiResponse::EventsPage(page) => page
                .events
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        daemon_api::NodeEvent::MessagesChanged { transport, conv: c }
                            if transport.as_str() == "room" && c == conv
                    )
                })
                .count(),
            other => panic!("expected EventsPage, got {other:?}"),
        }
    }

    async fn tear_down(self) {
        self.server.abort();
        for task in &self.adapter_tasks {
            task.abort();
        }
        self.handle.shutdown().await;
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The journal obligation on the rooms send path (wire v38): every `ConvSend` — operator
/// (`from: None`) and contact-attributed alike — appends one `JournalRecordPayload::Chat` with a
/// properly populated `ChatMessage` (structured author, RAW text, timestamp) to
/// `conv:room:<conv>`, readable via `ConvHistory` in append order, and each append raises exactly
/// one granular `NodeEvent::MessagesChanged { transport: "room", conv }` on the L3 feed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_send_journals_chat_and_emits_messages_changed() {
    let h = RoomsSocket::bring_up("chatjournal", [0x60; 32]).await;
    h.create_room("j1").await;

    // An operator post (`from: None`) and a contact-attributed post.
    h.send("j1", None, "hello there").await;
    let alice = daemon_api::Participant::Contact(daemon_api::ContactInfo {
        id: "@alice:ext".into(),
        ..daemon_api::ContactInfo::default()
    });
    h.send("j1", Some(alice.clone()), "hi from alice").await;

    let page = h.history_at_least("j1", 2).await;
    assert_eq!(
        page.entries.len(),
        2,
        "two sends = two journal records, got {page:?}"
    );
    assert!(
        page.entries.windows(2).all(|w| w[0].cursor < w[1].cursor),
        "append order with strictly-increasing cursors"
    );
    for entry in &page.entries {
        assert_eq!(entry.kind, "chat.message");
        assert!(entry.verified, "per-message segments seal + verify");
    }
    match &page.entries[0].payload {
        daemon_api::JournalRecordPayload::Chat { message } => {
            assert_eq!(message.text, "hello there");
            assert_eq!(message.author, None, "operator post: account-originated");
            assert!(message.timestamp.is_some(), "the append stamps a timestamp");
        }
        other => panic!("expected Chat, got {other:?}"),
    }
    match &page.entries[1].payload {
        daemon_api::JournalRecordPayload::Chat { message } => {
            assert_eq!(message.text, "hi from alice");
            assert_eq!(
                message.author,
                Some(alice),
                "the ConvSend `from` participant rides ChatMessage::author"
            );
        }
        other => panic!("expected Chat, got {other:?}"),
    }

    // One MessagesChanged per append, carrying the right (transport, conv).
    assert_eq!(
        h.messages_changed("j1").await,
        2,
        "each Chat append emits exactly one MessagesChanged"
    );
    assert_eq!(
        h.messages_changed("nonexistent").await,
        0,
        "pointers are granular per conversation"
    );

    h.tear_down().await;
}

/// ConvHistory paging over Chat records (wire v38): N messages page through `after_cursor + max`
/// with stable, strictly-increasing cursors, no dup or gap, in append order — and a re-read from
/// the same cursor returns the same page (non-destructive).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conv_history_pages_chat_records_with_stable_cursors() {
    let h = RoomsSocket::bring_up("chatpaging", [0x61; 32]).await;
    h.create_room("pg").await;

    let total = 5usize;
    for i in 0..total {
        h.send("pg", None, &format!("m{i}")).await;
    }
    let all = h.history_at_least("pg", total).await;
    assert_eq!(all.entries.len(), total, "all sends journaled, got {all:?}");

    // Page with max=2: sizes 2+2+1, cursor-chained, union == append order, cursors monotonic.
    let mut after = 0u64;
    let mut sizes = Vec::new();
    let mut texts = Vec::new();
    let mut last_cursor = 0u64;
    loop {
        let page = h.history("pg", after, 2).await;
        if page.entries.is_empty() {
            break;
        }
        assert!(page.entries.len() <= 2, "max bounds the page");
        // Stability: a re-read from the same cursor returns the identical page.
        let again = h.history("pg", after, 2).await;
        assert_eq!(page.entries, again.entries, "reads are non-destructive");
        for entry in &page.entries {
            assert!(entry.cursor > last_cursor, "cursors strictly increase");
            last_cursor = entry.cursor;
            match &entry.payload {
                daemon_api::JournalRecordPayload::Chat { message } => {
                    texts.push(message.text.clone())
                }
                other => panic!("expected Chat, got {other:?}"),
            }
        }
        sizes.push(page.entries.len());
        assert_eq!(
            page.next_cursor, last_cursor,
            "next_cursor is the last entry's cursor"
        );
        after = page.next_cursor;
    }
    assert_eq!(sizes, vec![2, 2, 1], "5 messages page as 2 + 2 + 1");
    let expected: Vec<String> = (0..total).map(|i| format!("m{i}")).collect();
    assert_eq!(texts, expected, "sent messages read back in append order");

    h.tear_down().await;
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
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x5e; 32], fast_host_config());

    let rooms_cfg = daemon_rooms::RoomsConfig {
        enabled: true,
        max_turns: 8,
    };
    let registry = daemon_host::AdapterRegistry::new().with_adapter(
        daemon_rooms::RoomsAdapter::new(store.clone(), rooms_cfg, Some(node.lifecycle_sink())),
    );
    node.set_adapters(registry);
    let adapter_tasks = node.spawn_adapters().await;

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
