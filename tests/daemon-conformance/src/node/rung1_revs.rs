// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Rung 1 (per-collection revisions + feed epoch, api/39) end-to-end over the assembled node.
//!
//! These prove the node-authoritative behavior the wire deltas add: a domain mutation bumps its
//! collection's coalescing revision exactly once and the matching list response echoes it (so a
//! thin client skips an unchanged refetch), the `Tree` report echoes `FleetChanged.rev` (closing
//! the compare loop), and every `EventsPage` is stamped with the feed generation (`epoch`).

use super::harness::*;

/// A `Tree` report echoes the current `FleetChanged.rev` (rung 1): after a delegation raises the
/// coalescing fleet pointer, the report's `rev` equals the pointer's, so a client compares the two
/// and skips a `Tree` refetch when unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_report_rev_echoes_fleet_changed_rev() {
    as_system(tree_report_rev_echoes_fleet_changed_rev_impl()).await;
}
async fn tree_report_rev_echoes_fleet_changed_rev_impl() {
    use daemon_api::{dispatch, NodeEvent};

    let (node, handle) = assemble();

    // Drive a delegation: the default orchestrator delegates once, changing the fleet tree, which
    // the assembly bridge forwards onto the node-wide feed as a coalescing `FleetChanged`.
    match dispatch(
        node.as_ref(),
        ApiRequest::Assign {
            session: SessionId::new("rung1-tree-rev"),
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from Assign, got {other:?}"),
    }

    // The bridge emits asynchronously; poll the retained feed until a FleetChanged lands, then take
    // its (latest) rev — the value the Tree report must echo.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut fleet_rev = None;
    while Instant::now() < deadline && fleet_rev.is_none() {
        if let ApiResponse::EventsPage(page) = dispatch(
            node.as_ref(),
            ApiRequest::EventsSince {
                cursor: 0,
                wait_ms: None,
            },
        )
        .await
        {
            fleet_rev = page
                .events
                .iter()
                .filter_map(|e| match e {
                    NodeEvent::FleetChanged { rev } => Some(*rev),
                    _ => None,
                })
                .max();
        }
        if fleet_rev.is_none() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
    let fleet_rev = fleet_rev.expect("a delegation must raise FleetChanged on the feed");
    assert!(fleet_rev >= 1, "the fleet rev must have advanced past 0");

    let tree_rev = match dispatch(node.as_ref(), ApiRequest::Tree { after: None }).await {
        ApiResponse::Tree(report) => report.rev,
        other => panic!("expected Tree, got {other:?}"),
    };
    assert_eq!(
        tree_rev, fleet_rev,
        "tree-report.rev must echo the current FleetChanged.rev"
    );

    handle.shutdown().await;
}

/// A person-registry mutation bumps the persons rev exactly once per change and `PersonList` echoes
/// the current value, so the pointer (`PersonsChanged.rev`) and the read agree on the generation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn person_add_bumps_persons_rev_once_and_list_echoes_it() {
    as_system(person_add_bumps_persons_rev_once_and_list_echoes_it_impl()).await;
}
async fn person_add_bumps_persons_rev_once_and_list_echoes_it_impl() {
    use daemon_api::{dispatch, NodeEvent};

    let (node, handle) = assemble();

    let person = |id: &str| daemon_api::Person {
        id: id.into(),
        alias: None,
        avatar: None,
        endpoints: Vec::new(),
    };

    node.person_add(person("p1"));
    let rev1 = person_list_rev(&node).await;
    node.person_add(person("p2"));
    let rev2 = person_list_rev(&node).await;

    assert_eq!(rev1, 1, "the first person add bumps the persons rev to 1");
    assert_eq!(
        rev2, 2,
        "the second add bumps it exactly once more (not twice per emit)"
    );

    // The feed carries a PersonsChanged per add, and its latest rev matches the list's echo.
    let latest_pointer_rev = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page
            .events
            .iter()
            .filter_map(|e| match e {
                NodeEvent::PersonsChanged { rev } => Some(*rev),
                _ => None,
            })
            .max(),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert_eq!(
        latest_pointer_rev,
        Some(rev2),
        "the PersonsChanged pointer rev must agree with the PersonList echo"
    );

    handle.shutdown().await;
}

async fn person_list_rev(node: &Arc<NodeApiImpl>) -> u64 {
    use daemon_api::dispatch;
    match dispatch(node.as_ref(), ApiRequest::PersonList { since_rev: None }).await {
        ApiResponse::Persons(list) => list.rev,
        other => panic!("expected Persons, got {other:?}"),
    }
}

/// A notification-set mutation bumps the notifications rev exactly once and `NotificationList`
/// echoes it (mirrors persons; a second emit site proving the "once per change" contract).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn notification_add_bumps_notifications_rev_once_and_list_echoes_it() {
    as_system(notification_add_bumps_notifications_rev_once_and_list_echoes_it_impl()).await;
}
async fn notification_add_bumps_notifications_rev_once_and_list_echoes_it_impl() {
    use daemon_protocol::TransportId;

    let (node, handle) = assemble();

    node.notify_add(daemon_api::NotificationInfo::new_connection_error(
        Some("n1".into()),
        TransportId::new("matrix/@me:hs.org"),
    ));
    let rev1 = notification_list_rev(&node).await;
    node.notify_add(daemon_api::NotificationInfo::new_connection_error(
        Some("n2".into()),
        TransportId::new("discord/bot"),
    ));
    let rev2 = notification_list_rev(&node).await;

    assert_eq!(rev1, 1, "the first notification add bumps the rev to 1");
    assert_eq!(rev2, 2, "the second add bumps it exactly once more");

    handle.shutdown().await;
}

async fn notification_list_rev(node: &Arc<NodeApiImpl>) -> u64 {
    use daemon_api::dispatch;
    match dispatch(node.as_ref(), ApiRequest::NotificationList).await {
        ApiResponse::Notifications(list) => list.rev,
        other => panic!("expected Notifications, got {other:?}"),
    }
}

/// Projection-sync stage 1 (daemon-projection-sync-spec.md §10): every per-session override write
/// (`SetSessionModel` / `SetSessionMode` / `SetSessionOverlay`) emits `SessionMetaChanged` from the
/// single durable persistence path (`update_overlay`), so a second client's cached session detail
/// is invalidated — previously these persisted silently and clients diverged until reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlay_writes_emit_session_meta_changed() {
    as_system(overlay_writes_emit_session_meta_changed_impl()).await;
}
async fn overlay_writes_emit_session_meta_changed_impl() {
    use daemon_api::{dispatch, NodeEvent, SessionApi, SessionOverlay};
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();
    let session = SessionId::new("ovl-meta");

    // Open the session (creates its durable meta; also emits its own roster/meta activity).
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hi"),
            request_id: daemon_common::ReqId(1),
        },
    )
    .await
    .expect("submit opens the session");

    // Drain the feed so only the override writes below land past `after`.
    let after = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page.next_cursor,
        other => panic!("expected EventsPage, got {other:?}"),
    };

    // All three override writes: model, mode (narrowing — no operator gate), unified overlay.
    match dispatch(
        node.as_ref(),
        ApiRequest::SetSessionModel {
            session: session.clone(),
            model: "claude-opus-4-8".into(),
            provider: None,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from SetSessionModel, got {other:?}"),
    }
    match dispatch(
        node.as_ref(),
        ApiRequest::SetSessionMode {
            session: session.clone(),
            mode: daemon_api::ApprovalMode::Ask,
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from SetSessionMode, got {other:?}"),
    }
    match dispatch(
        node.as_ref(),
        ApiRequest::SetSessionOverlay {
            session: session.clone(),
            overlay: SessionOverlay {
                model: Some("claude-3-5-sonnet-latest".into()),
                ..Default::default()
            },
        },
    )
    .await
    {
        ApiResponse::Ok => {}
        other => panic!("expected Ok from SetSessionOverlay, got {other:?}"),
    }

    // The emission is synchronous with the write (no async bridge), so one page suffices: three
    // SessionMetaChanged pointers for this session, revs strictly increasing (one bump per write).
    let revs: Vec<u64> = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: after,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page
            .events
            .iter()
            .filter_map(|e| match e {
                NodeEvent::SessionMetaChanged {
                    session: s, rev, ..
                } if *s == session => Some(*rev),
                _ => None,
            })
            .collect(),
        other => panic!("expected EventsPage, got {other:?}"),
    };
    assert_eq!(
        revs.len(),
        3,
        "each override write must emit exactly one SessionMetaChanged, got revs {revs:?}"
    );
    assert!(
        revs.windows(2).all(|w| w[0] < w[1]),
        "override writes bump the roster rev monotonically, got {revs:?}"
    );

    handle.shutdown().await;
}

/// Every `EventsPage` is stamped with the feed generation (rung 1): the one-shot `EventsSince` read
/// carries `Some(epoch)`, the signal a client uses to distinguish a new feed generation from a ring
/// overflow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_page_is_stamped_with_the_feed_epoch() {
    as_system(events_page_is_stamped_with_the_feed_epoch_impl()).await;
}
async fn events_page_is_stamped_with_the_feed_epoch_impl() {
    use daemon_api::dispatch;

    let (node, handle) = assemble();
    match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => assert!(
            page.epoch.is_some(),
            "every EventsPage must be stamped with the feed epoch (rung 1)"
        ),
        other => panic!("expected EventsPage, got {other:?}"),
    }
    handle.shutdown().await;
}
