// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// Fold one page of node-events into the `saw_*` flags: any `RosterChanged`, and any
/// `SessionMetaChanged`/`SessionAdvanced` for `want`. Shared by the live push read and the
/// deterministic retained-ring re-read in [`events_since_feed_streams_node_events_and_resyncs`].
fn scan_node_events(
    events: Vec<daemon_api::NodeEvent>,
    want: &SessionId,
    saw_roster: &mut bool,
    saw_session: &mut bool,
) {
    use daemon_api::NodeEvent;
    for ev in events {
        match ev {
            NodeEvent::RosterChanged { .. } => *saw_roster = true,
            NodeEvent::SessionMetaChanged { session: s, .. }
            | NodeEvent::SessionAdvanced { session: s, .. }
                if s == *want =>
            {
                *saw_session = true
            }
            _ => {}
        }
    }
}

/// L2 resync: the live merged log carries a session-activation `epoch` that strictly increases
/// on each (re)activation. Simulated as a daemon restart by assembling two nodes over one shared
/// durable store: the second activation of the same session must report a greater epoch than the
/// first, which is exactly the signal a client uses to detect a generation change and re-baseline
/// from the durable journal instead of misapplying a fresh log onto a stale cursor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_log_epoch_bumps_on_reactivation() {
    use daemon_api::SessionApi;
    use daemon_common::ReqId;
    use daemon_protocol::{AgentCommand, UserMsg};

    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let session = SessionId::new("epoch-reactivate");
    let cmd = || AgentCommand::StartTurn {
        input: UserMsg::new("hi"),
        request_id: ReqId(1),
    };

    // First activation -> epoch 0; the host persists the bumped generation to the shared store.
    let AssembledNode {
        node: n1,
        handle: h1,
        ..
    } = assemble_over(store.clone(), 0, [0x11; 32], fast_host_config());
    n1.submit(session.clone(), cmd()).await.expect("submit 1");
    let e0 = n1
        .log_after(session.clone(), 0, 0)
        .await
        .expect("log_after 1")
        .epoch;
    h1.shutdown().await;

    // Reactivation over the same durable store (the daemon-restart scenario): strictly greater.
    let AssembledNode {
        node: n2,
        handle: h2,
        ..
    } = assemble_over(store.clone(), 0, [0x11; 32], fast_host_config());
    n2.submit(session.clone(), cmd()).await.expect("submit 2");
    let e1 = n2
        .log_after(session.clone(), 0, 0)
        .await
        .expect("log_after 2")
        .epoch;
    h2.shutdown().await;

    assert_eq!(e0, 0, "the first activation is epoch 0");
    assert!(
        e1 > e0,
        "reactivation must yield a strictly greater epoch (got {e0} then {e1})"
    );
}

/// The multiplexed/server-streaming socket envelope (wire L0; daemon-sync-protocol-spec.md §2):
/// the Hello handshake, one-shot Call/Reply correlation, a push Open/Item/End `Subscribe`
/// stream with Cancel, and that a legacy (no-Hello) client still round-trips on the same server.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mux_envelope_one_shot_stream_and_legacy_fallback() {
    use daemon_api::WireS2C;
    use daemon_common::ReqId;
    use daemon_host::MuxApiClient;
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    // 1. Multiplexed one-shot: connect performs the Hello handshake; Call/Reply correlates.
    let mut mux = MuxApiClient::connect(path.clone())
        .await
        .expect("mux connect + hello");
    match mux.call(ApiRequest::Health).await.expect("mux health") {
        ApiResponse::Health(h) => assert!(h.services.len() >= 4),
        other => panic!("expected Health, got {other:?}"),
    }

    // 2. A live session with a merged log to stream.
    let session = SessionId::new("mux-stream");
    match mux
        .call(ApiRequest::Submit {
            session: session.clone(),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        })
        .await
        .expect("mux submit")
    {
        ApiResponse::Ok | ApiResponse::Routed { .. } => {}
        other => panic!("expected Ok/Routed, got {other:?}"),
    }

    // 3. Open a push subscription: the server streams Item(LogPage) frames under the stream id.
    let id = mux
        .open(ApiRequest::Subscribe {
            session: session.clone(),
            after_seq: 0,
            max: 64,
        })
        .await
        .expect("open subscribe");
    let mut got_item = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match mux.next().await.expect("stream frame") {
            WireS2C::Item { id: rid, res } => {
                assert_eq!(rid, id, "Item must carry the stream id");
                match res {
                    // First activation streams epoch 0 (L2).
                    ApiResponse::LogPage(page) => assert_eq!(page.epoch, 0),
                    other => panic!("Item must wrap a LogPage, got {other:?}"),
                }
                got_item = true;
                break;
            }
            WireS2C::End { id: rid, error } => {
                panic!("stream ended early: id={rid} error={error:?}")
            }
            _ => continue,
        }
    }
    assert!(got_item, "the push subscription delivered no Item");

    // 4. Cancel tears the stream down with End.
    mux.cancel(id).await.expect("cancel");
    let mut ended = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let WireS2C::End { id: rid, .. } = mux.next().await.expect("frame after cancel") {
            assert_eq!(rid, id);
            ended = true;
            break;
        }
    }
    assert!(ended, "Cancel did not close the stream with End");

    // 5. Legacy fallback: a bare (no-Hello) client still round-trips on the same server.
    let legacy = ApiClient::new(path.clone());
    assert!(matches!(
        legacy.call(ApiRequest::Health).await.unwrap(),
        ApiResponse::Health(_)
    ));

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
}

/// The node-wide event feed (L3 `EventsSince`; daemon-sync-protocol-spec.md §5): an `Open`
/// `EventsSince` push stream delivers the payload-free `NodeEvent` pointers (a `Submit` raises
/// `RosterChanged`/`SessionMetaChanged`/`SessionAdvanced`), a `Cancel` closes it with `End`, and
/// the one-shot `Call` form re-reads the same retained feed from a cursor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_since_feed_streams_node_events_and_resyncs() {
    use daemon_api::{NodeEvent, WireS2C};
    use daemon_common::ReqId;
    use daemon_host::MuxApiClient;
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let mut mux = MuxApiClient::connect(path.clone())
        .await
        .expect("mux connect + hello");

    // Open the node-wide feed from the start of the retained ring.
    let feed_id = mux
        .open(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .expect("open events-since");

    // A submit activates a session (RosterChanged), notes activity (SessionMetaChanged) and grows
    // the merged log (SessionAdvanced) — all funnel onto the feed.
    let session = SessionId::new("feed-session");
    match mux
        .call(ApiRequest::Submit {
            session: session.clone(),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hello feed"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        })
        .await
        .expect("mux submit")
    {
        ApiResponse::Ok | ApiResponse::Routed { .. } => {}
        other => panic!("expected Ok/Routed, got {other:?}"),
    }

    // Collect node-events off the push stream. The live broadcast is intentionally lossy: under a
    // backlog burst it coalesces/drops and surfaces a `ResyncNeeded` pointer instead of replaying
    // every event (the feed's §5 contract), so the streaming read alone is NOT a deterministic
    // oracle for any single event — which is exactly why the historical `RosterChanged` assertion
    // here flaked under the heavily-parallel run. We scan the push stream best-effort (breaking out
    // on a `ResyncNeeded`), then settle the assertion against the retained ring via the one-shot
    // re-read below, which deterministically holds the coalesce-exempt `RosterChanged` and the
    // session's activity. This proves the same guarantee without the timing race.
    let mut saw_roster = false;
    let mut saw_session = false;
    let mut resynced = false;
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline && !(saw_roster && saw_session) && !resynced {
        match mux.next().await.expect("feed frame") {
            WireS2C::Item { id: rid, res } => {
                assert_eq!(rid, feed_id, "Item must carry the feed stream id");
                let ApiResponse::EventsPage(page) = res else {
                    panic!("EventsSince Item must wrap an EventsPage, got {res:?}");
                };
                resynced = page
                    .events
                    .iter()
                    .any(|e| matches!(e, NodeEvent::ResyncNeeded { .. }));
                scan_node_events(page.events, &session, &mut saw_roster, &mut saw_session);
            }
            WireS2C::End { id: rid, error } => {
                panic!("feed ended early: id={rid} error={error:?}")
            }
            _ => continue,
        }
    }

    // The one-shot Call form re-reads the same retained feed (non-destructive) from cursor 0. This
    // is the deterministic backstop: the ring retains `RosterChanged` (never coalesced) and the
    // session's activity regardless of any live-broadcast lag, so the assertion is race-free.
    match mux
        .call(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .expect("events-since call")
    {
        ApiResponse::EventsPage(page) => {
            assert!(
                !page.events.is_empty(),
                "the one-shot EventsSince re-read should see the retained events"
            );
            assert!(page.head_cursor >= page.next_cursor);
            scan_node_events(page.events, &session, &mut saw_roster, &mut saw_session);
        }
        other => panic!("expected EventsPage, got {other:?}"),
    }

    assert!(
        saw_roster,
        "neither the push stream nor the retained feed delivered RosterChanged"
    );
    assert!(
        saw_session,
        "neither the push stream nor the retained feed delivered SessionAdvanced/SessionMetaChanged"
    );

    // Cancel tears the feed stream down with End.
    mux.cancel(feed_id).await.expect("cancel feed");
    let mut ended = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let WireS2C::End { id: rid, .. } = mux.next().await.expect("frame after cancel") {
            if rid == feed_id {
                ended = true;
                break;
            }
        }
    }
    assert!(ended, "Cancel did not close the feed stream with End");

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
}

/// Wire page bound (v24): the one-shot `EventsSince` — through the shared `dispatch` every socket
/// transport runs — returns at most WIRE_PAGE_MAX events per page (previously it served up to the
/// whole retained ring in one response), and the `next_cursor` loop reads the rest to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_since_one_shot_is_bounded_at_the_wire_page_max() {
    use daemon_api::{dispatch, SessionApi, WIRE_PAGE_MAX};

    let (node, handle) = assemble();
    // 70 node-authoritative session creations: each emits an un-coalesced `RosterChanged` onto the
    // retained feed, so the ring holds more than one wire page.
    for n in 0..70 {
        node.session_create(Some(SessionId::new(format!("evt-{n:03}"))), None)
            .await
            .expect("session_create");
    }

    let page = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page,
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert_eq!(
        page.events.len(),
        WIRE_PAGE_MAX,
        "the first page of an over-full ring must be exactly the wire bound"
    );
    assert!(
        page.next_cursor < page.head_cursor,
        "more events remain past the first page"
    );

    // The cursor loop drains the rest, each page within the bound.
    let mut cursor = page.next_cursor;
    let mut total = page.events.len();
    let mut pages = 1usize;
    loop {
        let page = match dispatch(
            node.as_ref(),
            ApiRequest::EventsSince {
                cursor,
                wait_ms: None,
            },
        )
        .await
        {
            ApiResponse::EventsPage(page) => page,
            other => panic!("expected EventsPage, got {other:?}"),
        };
        assert!(
            page.events.len() <= WIRE_PAGE_MAX,
            "every page must respect the wire bound, got {}",
            page.events.len()
        );
        total += page.events.len();
        cursor = page.next_cursor;
        pages += 1;
        assert!(pages <= 4, "runaway pagination");
        if cursor >= page.head_cursor {
            break;
        }
    }
    assert!(
        total >= 70,
        "the cursor loop must deliver every retained event, got {total}"
    );

    handle.shutdown().await;
}

/// Live fleet push: a durable delegation (the default node delegates once on `Assign`) changes the
/// subagent tree, and the `assemble()` bridge forwards the fleet bus onto the node-wide feed as a
/// `FleetChanged` so an `EventsSince` client re-fetches `Tree` live (not just on focus/reconnect).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_since_feed_delivers_fleet_changed_on_delegation() {
    use daemon_api::{NodeEvent, WireS2C};
    use daemon_host::MuxApiClient;

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let mut mux = MuxApiClient::connect(path.clone())
        .await
        .expect("mux connect + hello");
    let feed_id = mux
        .open(ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        })
        .await
        .expect("open events-since");

    match mux
        .call(ApiRequest::Assign {
            session: SessionId::new("fleet-feed-op"),
        })
        .await
        .expect("assign drives a delegation")
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }

    let mut saw_fleet = false;
    while !saw_fleet {
        let frame = match tokio::time::timeout(Duration::from_secs(30), mux.next()).await {
            Ok(f) => f.expect("feed frame"),
            Err(_) => break, // deadline: no FleetChanged arrived
        };
        match frame {
            WireS2C::Item { id: rid, res } => {
                assert_eq!(rid, feed_id, "Item must carry the feed stream id");
                let ApiResponse::EventsPage(page) = res else {
                    panic!("EventsSince Item must wrap an EventsPage, got {res:?}");
                };
                if page
                    .events
                    .iter()
                    .any(|e| matches!(e, NodeEvent::FleetChanged { .. }))
                {
                    saw_fleet = true;
                }
            }
            WireS2C::End { id: rid, error } => {
                panic!("feed ended early: id={rid} error={error:?}")
            }
            _ => continue,
        }
    }
    assert!(
        saw_fleet,
        "the feed delivered no FleetChanged after a delegation"
    );

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
}
