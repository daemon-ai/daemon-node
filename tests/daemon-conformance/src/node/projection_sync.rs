// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Projection-sync stage 3 (daemon-projection-sync-spec.md §3–§8), API-level: the node
//! dual-emits every legacy invalidation arm's `ProjectionChanged` twin with the identical rev,
//! `Bootstrap` carries the incarnation id + the uniform domain-rev table, the §6 visibility
//! classes gate delivery per principal (owner scope for Sessions keys, read capability for the
//! admin domains), and the mux server `Hello` stamps the incarnation so a client detects a node
//! restart at the handshake. The feed-internal mechanics (coalescing matrix, effect recording,
//! `note_domain_change`) are pinned by the `daemon-host` unit suite; these tests pin the
//! **wire-visible** contract a `daemon-app` migrates against in stage 5.

use super::harness::*;
use super::wire_client::MuxConn;
use daemon_api::{dispatch, ChangeScope, ControlApi, NodeEvent, ProjectionId, SessionApi};
use daemon_auth::{Principal, Role};
use daemon_host::{serve_api_unix, with_request_context, RequestContext};
use daemon_protocol::{AgentCommand, UserMsg};
use tokio::net::{UnixListener, UnixStream};

/// A request context bound to `name` (its own `user_id`) holding exactly `role`.
fn ctx(name: &str, role: Role) -> RequestContext {
    RequestContext::authenticated(Principal::from_roles(name, name, vec![role]), None)
}

/// Every override write dual-emits: the legacy `SessionMetaChanged` AND its
/// `ProjectionChanged { Sessions, Key(session) }` twin with the IDENTICAL rev — the
/// migration-window invariant that lets a client switch arms per vertical without rev skew.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn override_write_dual_emits_the_sessions_twin() {
    as_system(override_write_dual_emits_the_sessions_twin_impl()).await;
}
async fn override_write_dual_emits_the_sessions_twin_impl() {
    let (node, handle) = assemble();
    let session = SessionId::new("psync-twin");

    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("hi"),
            request_id: daemon_common::ReqId(1),
        },
    )
    .await
    .expect("submit opens the session");

    // Drain, so only the override write below lands past `after`.
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
        other => panic!("expected Ok, got {other:?}"),
    }

    let events = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: after,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page.events,
        other => panic!("expected EventsPage, got {other:?}"),
    };
    let legacy_rev = events
        .iter()
        .find_map(|e| match e {
            NodeEvent::SessionMetaChanged {
                session: s, rev, ..
            } if *s == session => Some(*rev),
            _ => None,
        })
        .expect("the legacy SessionMetaChanged still rides in the migration window");
    assert!(
        events.iter().any(|e| matches!(e,
            NodeEvent::ProjectionChanged {
                projection: ProjectionId::Sessions,
                partition: None,
                scope: ChangeScope::Key { key },
                rev,
                ..
            } if key == session.as_str() && *rev == legacy_rev)),
        "the twin must carry the SAME rev + session key, got {events:?}"
    );

    handle.shutdown().await;
}

/// §6 CapabilityScoped: a `ProjectionChanged` for an admin domain (Fingerprints — read capability
/// `AccessAdmin`) is invisible to an operator (who holds `SessionSeeAll` but NOT `AccessAdmin`)
/// and visible to an admin. Pins that the old SessionSeeAll whole-feed short-circuit is gone —
/// see-all is an OWNERSHIP override, not a capability bypass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capability_scoped_projection_events_require_the_read_capability() {
    let (node, handle) = assemble();

    node.emit_node_event(NodeEvent::ProjectionChanged {
        projection: ProjectionId::Fingerprints,
        partition: None,
        scope: ChangeScope::All,
        rev: 1,
        origin_op: None,
    });

    let sees = |name: &'static str, role: Role| {
        let node = node.clone();
        async move {
            with_request_context(ctx(name, role), async { node.events_page(0, 256).await })
                .await
                .events
                .iter()
                .any(|e| {
                    matches!(
                        e,
                        NodeEvent::ProjectionChanged {
                            projection: ProjectionId::Fingerprints,
                            ..
                        }
                    )
                })
        }
    };
    assert!(
        !sees("op", Role::Operator).await,
        "SECURITY: an operator without AccessAdmin saw a Fingerprints invalidation"
    );
    assert!(
        !sees("bob", Role::User).await,
        "SECURITY: a plain user saw a Fingerprints invalidation"
    );
    assert!(
        sees("root", Role::Admin).await,
        "an admin (AccessAdmin) must see the Fingerprints invalidation"
    );

    handle.shutdown().await;
}

/// §6 OwnerScoped: the Sessions twin is filtered exactly like its legacy arm — a Key-scoped
/// envelope naming another owner's session never reaches a non-owner, while All-scope roster
/// pointers pass (the refetch they nudge is itself owner-filtered).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sessions_key_twin_is_owner_scoped() {
    let (node, handle) = assemble();
    let s = SessionId::new("psync-owner");

    with_request_context(ctx("alice", Role::User), async {
        node.submit(
            s.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
        )
        .await
    })
    .await
    .expect("alice submits to her own session");

    let key_twin_for = |events: &[NodeEvent]| {
        events.iter().any(|e| {
            matches!(e,
            NodeEvent::ProjectionChanged {
                projection: ProjectionId::Sessions,
                scope: ChangeScope::Key { key },
                ..
            } if key == s.as_str())
        })
    };

    // Wait until the twin is retained (poll as operator — see-all grants session ownership).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = with_request_context(ctx("op", Role::Operator), async {
            node.events_page(0, 256).await
        })
        .await;
        if key_twin_for(&page.events) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "alice's Sessions Key twin never appeared in the feed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let bob = with_request_context(ctx("bob", Role::User), async {
        node.events_page(0, 256).await
    })
    .await;
    assert!(
        !key_twin_for(&bob.events),
        "SECURITY: a non-owner's feed carried a Sessions Key twin naming alice's session"
    );
    let alice = with_request_context(ctx("alice", Role::User), async {
        node.events_page(0, 256).await
    })
    .await;
    assert!(
        key_twin_for(&alice.events),
        "the owner must see her own session's twin"
    );

    handle.shutdown().await;
}

/// §8: the `Bootstrap` reply carries the process-incarnation id and the uniform domain-rev table,
/// with the `ProjectionId`-keyed entries numerically equal to the legacy string-keyed `revs` (the
/// migration-window agreement a client verifies once and then trusts).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bootstrap_carries_incarnation_and_the_domain_rev_table() {
    as_system(bootstrap_carries_incarnation_and_the_domain_rev_table_impl()).await;
}
async fn bootstrap_carries_incarnation_and_the_domain_rev_table_impl() {
    let (node, handle) = assemble();

    // Touch the roster so the Sessions rev is non-zero (a fresh feed serves all-zero revs).
    node.submit(
        SessionId::new("psync-boot"),
        AgentCommand::StartTurn {
            input: UserMsg::new("hi"),
            request_id: daemon_common::ReqId(1),
        },
    )
    .await
    .expect("submit touches the roster");

    let report = match dispatch(node.as_ref(), ApiRequest::Bootstrap).await {
        ApiResponse::Bootstrap(report) => report,
        other => panic!("expected Bootstrap, got {other:?}"),
    };
    assert!(
        report.incarnation.is_some(),
        "the Bootstrap reply must carry the incarnation id"
    );
    let sessions = report
        .domain_revs
        .iter()
        .find(|d| d.projection == ProjectionId::Sessions && d.partition.is_none())
        .expect("the Sessions domain rides the table");
    assert_eq!(
        sessions.rev, report.revs["roster"],
        "the domain table and the legacy revs map must agree in the migration window"
    );
    assert!(
        report
            .domain_revs
            .iter()
            .any(|d| d.projection == ProjectionId::Agents),
        "Agents rides the domain table (it was missing from the legacy revs map)"
    );

    handle.shutdown().await;
}

/// Stage 4 (spec §5): the census gap closures — every formerly-silent mutation now lands its
/// `ProjectionChanged` pointer on the feed, from the handler's own success path. One dispatch per
/// domain, then one feed read asserting each expected `(projection, scope)` arrived. A domain
/// regressing to silence fails here by name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn formerly_silent_mutations_emit_their_domain_pointer() {
    as_system(formerly_silent_mutations_emit_their_domain_pointer_impl()).await;
}
async fn formerly_silent_mutations_emit_their_domain_pointer_impl() {
    let (node, handle) = assemble();

    let ok = |resp: ApiResponse, what: &str| match resp {
        ApiResponse::Error(e) => panic!("{what}: {e:?}"),
        other => other,
    };

    // Consent toggles.
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::TelemetryConsentSet { enabled: true },
        )
        .await,
        "telemetry consent",
    );
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::CrashConsentSet { enabled: false },
        )
        .await,
        "crash consent",
    );
    // Routing pin + unbind.
    let origin = daemon_protocol::Origin::new(
        daemon_protocol::TransportId::new("room"),
        daemon_protocol::OriginScope::Group {
            chat: "psync".into(),
            thread: None,
        },
    );
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::RoutingBindChat {
                origin: origin.clone(),
                session: SessionId::new("psync-routed"),
                profile: None,
            },
        )
        .await,
        "routing bind",
    );
    ok(
        dispatch(node.as_ref(), ApiRequest::RoutingUnbindChat { origin }).await,
        "routing unbind",
    );
    // Tool enablement override.
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::ToolSetEnabled {
                tool: "psync-tool".into(),
                enabled: false,
            },
        )
        .await,
        "tool override",
    );
    // Credential label (durable-store backed, no credential store required).
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::CredentialSetLabel {
                profile: "psync-cred".into(),
                label: Some("label".into()),
            },
        )
        .await,
        "credential label",
    );
    // Custom provider set + remove.
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::CustomProviderSet {
                provider: daemon_api::CustomProvider {
                    id: "psync-provider".into(),
                    display_name: "Projection Sync".into(),
                    base_url: "https://example.invalid/v1".into(),
                    wire_selector: daemon_api::ProviderSelector::DaemonApi,
                    requires_key: false,
                    credential_ref: None,
                    source: daemon_api::CustomProviderSource::User,
                },
            },
        )
        .await,
        "custom provider set",
    );
    // Cron job create.
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::CronCreate {
                spec: daemon_api::CronSpec {
                    name: "psync-cron".into(),
                    schedule: "0 0 * * *".into(),
                    payload: "hi".into(),
                    enabled: false,
                    ..Default::default()
                },
            },
        )
        .await,
        "cron create",
    );
    // Assign: creates + re-arms a durable row -> RosterChanged -> the Sessions/All twin.
    ok(
        dispatch(
            node.as_ref(),
            ApiRequest::Assign {
                session: SessionId::new("psync-assigned"),
            },
        )
        .await,
        "assign",
    );

    let events = match dispatch(
        node.as_ref(),
        ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: None,
        },
    )
    .await
    {
        ApiResponse::EventsPage(page) => page.events,
        other => panic!("expected EventsPage, got {other:?}"),
    };
    let has = |p: ProjectionId, all: bool| {
        events.iter().any(|e| {
            matches!(e,
            NodeEvent::ProjectionChanged { projection, scope, .. }
                if *projection == p && (matches!(scope, ChangeScope::All) == all))
        })
    };
    for p in [
        ProjectionId::TelemetryConsent,
        ProjectionId::CrashConsent,
        ProjectionId::Routing,
        ProjectionId::Tools,
        ProjectionId::Credentials,
        ProjectionId::CustomProviders,
        ProjectionId::Cron,
        ProjectionId::Sessions,
    ] {
        assert!(
            has(p, true),
            "expected an All-scope ProjectionChanged for {p:?}, got {events:?}"
        );
    }

    handle.shutdown().await;
}

/// §4.3: the mux server `Hello` stamps the node's incarnation id — equal to `Bootstrap`'s and
/// stable across connections to the same process, so a reconnecting client detects a restart at
/// the handshake without an extra round-trip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mux_hello_stamps_the_node_incarnation() {
    let (node, handle) = assemble();
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let conn_a = MuxConn::handshake(UnixStream::connect(&path).await.expect("connect"))
        .await
        .expect("hello handshake");
    let hello_inc = conn_a
        .incarnation
        .clone()
        .expect("the server Hello carries the incarnation id");
    let conn_b = MuxConn::handshake(UnixStream::connect(&path).await.expect("connect"))
        .await
        .expect("hello handshake");
    assert_eq!(
        conn_b.incarnation.as_deref(),
        Some(hello_inc.as_str()),
        "stable across connections to the same process"
    );
    let bootstrap_inc = as_system(async {
        match dispatch(node.as_ref(), ApiRequest::Bootstrap).await {
            ApiResponse::Bootstrap(report) => report.incarnation,
            other => panic!("expected Bootstrap, got {other:?}"),
        }
    })
    .await;
    assert_eq!(
        bootstrap_inc.as_deref(),
        Some(hello_inc.as_str()),
        "Hello and Bootstrap advertise the same incarnation"
    );

    server.abort();
    let _ = std::fs::remove_file(&path);
    handle.shutdown().await;
}
