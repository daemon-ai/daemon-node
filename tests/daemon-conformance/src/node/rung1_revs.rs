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
