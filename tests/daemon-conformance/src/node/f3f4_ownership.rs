// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! F3/F4 cross-owner authorization: the fleet/unit surface (F3) and the node-wide `EventsSince`
//! feed + transport-keyed `DeliverySessions` (F4) must be owner-scoped, exactly like the roster /
//! tree / checkpoints already are (Auth 4). These are the bug-repro pair: each asserts a non-owner
//! peer (`bob`) sees NONE of `alice`'s fleet units / node events / delivery sessions, while the
//! owner (`alice`) and an operator (`SessionSeeAll`) still do. They FAIL before the owner-scoping
//! gates land (the leak) and pass after — the F3/F4 residuals the `ownership_matrix` deny-table
//! moved from `KnownGap` to `OwnerGated`.

use super::harness::*;
use daemon_api::{ControlApi, NodeEvent, SessionApi};
use daemon_auth::{Principal, Role};
use daemon_common::{ReqId, UnitId};
use daemon_host::{with_request_context, RequestContext};
use daemon_protocol::{AgentCommand, DeliveryTarget, SinkKind, UserMsg};

/// A request context bound to `name` (its own `user_id`) holding exactly `role`.
fn ctx(name: &str, role: Role) -> RequestContext {
    RequestContext::authenticated(Principal::from_roles(name, name, vec![role]), None)
}

fn start_turn() -> AgentCommand {
    AgentCommand::StartTurn {
        input: UserMsg::new("hi"),
        request_id: ReqId(1),
    }
}

/// Whether any node-event names `s` (the three session-bearing variants; the payload-free node-wide
/// pointers are not owner-identifying and intentionally pass to any authenticated principal).
fn events_name_session(events: &[NodeEvent], s: &SessionId) -> bool {
    events.iter().any(|e| {
        matches!(e,
            NodeEvent::SessionAdvanced { session, .. }
            | NodeEvent::SessionMetaChanged { session, .. }
            | NodeEvent::ApprovalPending { session, .. }
                if session == s)
    })
}

// ---------------------------------------------------------------------------
// F3 — the fleet/unit surface is owner-scoped (RED before the gates, GREEN after)
// ---------------------------------------------------------------------------

/// `fleet`/`unit`/`unit_events` must be owner-scoped: a non-owner peer sees none of another owner's
/// fleet topology or unit drill-downs, while the owner and an operator do. BEFORE the fix these
/// `FleetRead` surfaces had NO ownership check, so `bob` (a plain `User`, which holds `FleetRead`)
/// read `alice`'s unit tree and per-unit management events cross-owner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f3_fleet_unit_is_owner_scoped() {
    let (node, handle) = assemble();

    // alice owns a session and drives one durable delegation (the default node delegates once on
    // assign), so the fleet holds an alice-owned parent + a managed-child Engine unit (the child
    // inherits the parent's owner at the delegation seam).
    let parent = SessionId::new("f3-op");
    with_request_context(ctx("alice", Role::User), async {
        node.assign(parent.clone()).await
    })
    .await
    .expect("alice assigns her own session");

    // Wait until the parent completes (its delegation child then exists in the tree). Poll as an
    // operator (SeeAll) so the wait loop is not itself affected by the scoping under test.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let done = with_request_context(ctx("op", Role::Operator), async { node.sessions().await })
            .await
            .iter()
            .any(|i| i.session == parent && i.state == SessionState::Completed);
        if done {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "alice's assigned session never completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The managed-child Engine unit, from the operator's full tree view.
    let child = with_request_context(ctx("op", Role::Operator), async { node.tree(None).await })
        .await
        .nodes
        .into_iter()
        .find(|n| n.kind == daemon_api::UnitKind::Engine)
        .expect("a managed-child Engine unit is present in the tree");
    let child_unit = child.id.clone();
    let parent_unit = UnitId::new(parent.as_str());

    // A non-owner peer must see NONE of it.
    let bob_unit = with_request_context(ctx("bob", Role::User), async {
        node.unit(child_unit.clone()).await
    })
    .await;
    assert!(
        bob_unit.is_none(),
        "SECURITY: a non-owner resolved alice's unit {child_unit}"
    );
    let bob_events = with_request_context(ctx("bob", Role::User), async {
        node.unit_events(child_unit.clone(), 0).await
    })
    .await;
    assert!(
        bob_events.is_empty(),
        "SECURITY: a non-owner read {} of alice's unit management events",
        bob_events.len()
    );
    let bob_fleet =
        with_request_context(ctx("bob", Role::User), async { node.fleet().await }).await;
    assert!(
        !bob_fleet.children.contains(&child_unit) && !bob_fleet.children.contains(&parent_unit),
        "SECURITY: a non-owner's fleet report listed alice's units {:?}",
        bob_fleet.children
    );

    // The owner and an operator (SeeAll) DO see it.
    for (name, role) in [("alice", Role::User), ("op", Role::Operator)] {
        let unit = with_request_context(ctx(name, role), async {
            node.unit(child_unit.clone()).await
        })
        .await;
        assert!(
            unit.is_some(),
            "{name} must resolve their own/any unit {child_unit}"
        );
        let events = with_request_context(ctx(name, role), async {
            node.unit_events(child_unit.clone(), 0).await
        })
        .await;
        assert!(
            !events.is_empty(),
            "{name} must read the unit's management events"
        );
        let fleet = with_request_context(ctx(name, role), async { node.fleet().await }).await;
        assert!(
            fleet.children.contains(&child_unit),
            "{name}'s fleet report must include the unit {child_unit}, got {:?}",
            fleet.children
        );
    }

    handle.shutdown().await;
}

// ---------------------------------------------------------------------------
// F4 — the node-wide feeds are owner-scoped
// ---------------------------------------------------------------------------

/// `events_page` (the `EventsSince` feed) must be owner-scoped: a non-owner peer sees no
/// session-bearing node-event for another owner's session, while the owner and an operator do.
/// BEFORE the fix the feed was node-wide, so `bob` (holding `ControlRead`) learned that `alice`'s
/// session advanced (its id + activity) cross-owner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f4_events_since_is_owner_scoped() {
    let (node, handle) = assemble();

    // alice submits a turn to her session, which advances the live log and emits a session-bearing
    // node-event (`SessionAdvanced`/`SessionMetaChanged`) into the feed.
    let s = SessionId::new("f4-events");
    with_request_context(ctx("alice", Role::User), async {
        node.submit(s.clone(), start_turn()).await
    })
    .await
    .expect("alice submits to her own session");

    // Wait until an alice-session-bearing event is retained in the feed (poll as operator/SeeAll).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let seen = with_request_context(ctx("op", Role::Operator), async {
            node.events_page(0, 256).await
        })
        .await;
        if events_name_session(&seen.events, &s) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "no alice-session-bearing node-event ever appeared in the feed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // A non-owner peer must see NONE of alice's session in the feed.
    let bob = with_request_context(ctx("bob", Role::User), async {
        node.events_page(0, 256).await
    })
    .await;
    assert!(
        !events_name_session(&bob.events, &s),
        "SECURITY: a non-owner's node-event feed named alice's session"
    );

    // The owner and an operator DO see it.
    for (name, role) in [("alice", Role::User), ("op", Role::Operator)] {
        let page =
            with_request_context(ctx(name, role), async { node.events_page(0, 256).await }).await;
        assert!(
            events_name_session(&page.events, &s),
            "{name} must see their own/any session's node-events"
        );
    }

    handle.shutdown().await;
}

/// `delivery_sessions` must be owner-scoped: a non-owner peer sees none of another owner's sessions
/// bound to a transport, while the owner and an operator do. BEFORE the fix the enumeration was
/// transport-keyed only, so `bob` (holding `SessionRead`) enumerated `alice`'s sessions on a shared
/// transport cross-owner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f4_delivery_sessions_is_owner_scoped() {
    use daemon_protocol::TransportId;

    let (node, handle) = assemble();
    let transport = TransportId::new("matrix/acct");

    // alice opens a live session and hands it over to a Primary delivery target on `transport`, so
    // the session is enumerable by `delivery_sessions(transport)` and owned by alice.
    let s = SessionId::new("f4-delivery");
    with_request_context(ctx("alice", Role::User), async {
        node.submit(s.clone(), start_turn()).await
    })
    .await
    .expect("alice opens a live session");
    with_request_context(ctx("alice", Role::User), async {
        node.handover(
            s.clone(),
            DeliveryTarget::new("matrix/acct", "!room:hs", SinkKind::Primary),
        )
        .await
    })
    .await
    .expect("alice binds her session's Primary delivery target");

    // A non-owner peer must NOT enumerate alice's session on the transport.
    let bob = with_request_context(ctx("bob", Role::User), async {
        node.delivery_sessions(transport.clone(), None).await
    })
    .await;
    assert!(
        !bob.items.contains(&s),
        "SECURITY: a non-owner enumerated alice's delivery session {s}"
    );

    // The owner and an operator DO enumerate it.
    for (name, role) in [("alice", Role::User), ("op", Role::Operator)] {
        let page = with_request_context(ctx(name, role), async {
            node.delivery_sessions(transport.clone(), None).await
        })
        .await;
        assert!(
            page.items.contains(&s),
            "{name} must enumerate their own/any delivery session on the transport, got {:?}",
            page.items
        );
    }

    handle.shutdown().await;
}
