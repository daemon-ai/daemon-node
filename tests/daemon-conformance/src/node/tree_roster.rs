// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// The tree-aware control surface (the GUI's real surface) is transport-agnostic: `tree`/`unit`/
/// `unit_events` and the lifecycle ops agree in-process and over the socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_surface_is_transport_agnostic() {
    use daemon_api::ApiError;

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Drive a delegation child to completion so the tree has a unit to project.
    let session = SessionId::new("tree-op");
    assert!(matches!(
        client
            .call(ApiRequest::Assign {
                session: session.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let ApiResponse::Sessions(list) = client.call(ApiRequest::Sessions).await.unwrap() {
            if list
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed)
            {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the assigned session never completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // tree() parity + the fleet child presents as an Engine leaf.
    let inproc_tree = node.tree().await;
    let socket_tree = match client.call(ApiRequest::Tree).await.unwrap() {
        ApiResponse::Tree(t) => t,
        other => panic!("expected Tree, got {other:?}"),
    };
    assert_eq!(
        inproc_tree, socket_tree,
        "tree must agree across transports"
    );
    assert!(
        !socket_tree.nodes.is_empty(),
        "expected at least one unit in the tree"
    );
    // The tree is rooted at the node's synthetic root, whose children are the fleet members.
    let root = socket_tree.root.clone().expect("the node tree is rooted");
    let root_node = socket_tree
        .nodes
        .iter()
        .find(|n| n.id == root)
        .expect("the root node is present");
    assert_eq!(
        root_node.kind,
        daemon_api::UnitKind::Orchestrator,
        "the node root projects as an orchestrator"
    );
    // The fleet child presents as an Engine leaf (a flat node, depth 0).
    let child = socket_tree
        .nodes
        .iter()
        .find(|n| n.kind == daemon_api::UnitKind::Engine)
        .expect("a fleet child Engine leaf is present")
        .clone();
    assert!(
        root_node.children.contains(&child.id),
        "the engine leaf is a direct child of the node root"
    );

    // unit() parity.
    let inproc_unit = node.unit(child.id.clone()).await;
    let socket_unit = match client
        .call(ApiRequest::Unit {
            unit: child.id.clone(),
        })
        .await
        .unwrap()
    {
        ApiResponse::Unit(u) => u,
        other => panic!("expected Unit, got {other:?}"),
    };
    assert_eq!(
        inproc_unit, socket_unit,
        "unit view must agree across transports"
    );
    assert!(socket_unit.is_some(), "the child unit should resolve");

    // unit_events() parity: the child emitted at least Started + Finished views.
    let inproc_events = node.unit_events(child.id.clone(), 0).await;
    let socket_events = match client
        .call(ApiRequest::UnitEvents {
            unit: child.id.clone(),
            max: 0,
        })
        .await
        .unwrap()
    {
        ApiResponse::UnitEvents(e) => e,
        other => panic!("expected UnitEvents, got {other:?}"),
    };
    assert_eq!(
        inproc_events, socket_events,
        "unit events must agree across transports"
    );
    assert!(
        !socket_events.is_empty(),
        "expected buffered drill-down events for the child"
    );

    // Lifecycle parity: an engine leaf does not support pause/resume/scale — identically on both
    // transports (the surface is meaningful for orchestrator sub-fleets).
    for (req, label) in [
        (
            ApiRequest::Pause {
                unit: child.id.clone(),
            },
            "pause",
        ),
        (
            ApiRequest::Resume {
                unit: child.id.clone(),
            },
            "resume",
        ),
        (
            ApiRequest::Scale {
                unit: child.id.clone(),
                n: 2,
            },
            "scale",
        ),
    ] {
        let socket = client.call(req).await.unwrap();
        assert!(
            matches!(socket, ApiResponse::Error(ApiError::Unsupported(_))),
            "{label} should be Unsupported over the socket, got {socket:?}"
        );
    }
    assert!(node.pause(child.id.clone()).await.is_err());
    assert!(node.resume(child.id.clone()).await.is_err());
    assert!(node.scale(child.id.clone(), 2).await.is_err());

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// Roster correctness (Phase-2 A1 regression): a durable delegation child is stamped
/// `role = ManagedChild`/`parent` at the delegation seam, so the `TopLevel` inbox scope excludes
/// it (it is reached only by walking `tree()`), while the `All` scope still surfaces it. The
/// scoped roster is byte-identical in-process and over the socket (live+durable parity).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_top_level_excludes_managed_children_across_transports() {
    use daemon_api::{SessionQuery, SessionRole, SessionScope};

    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Drive one delegation so the durable graph has a parent + a managed child.
    let session = SessionId::new("roster-op");
    assert!(matches!(
        client
            .call(ApiRequest::Assign {
                session: session.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let ApiResponse::Sessions(list) = client.call(ApiRequest::Sessions).await.unwrap() {
            if list
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed)
            {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the assigned session never completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The managed child id, sourced from the tree projection (the GUI's drill-down).
    let tree = node.tree().await;
    let child = tree
        .nodes
        .iter()
        .find(|n| n.kind == daemon_api::UnitKind::Engine)
        .and_then(|n| n.session.clone())
        .expect("a managed child session is present in the tree");
    assert_ne!(
        child, session,
        "the child is a distinct session from the parent"
    );

    // The child carries role ManagedChild + parent in the roster (the A1 stamp).
    let all = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            ..Default::default()
        })
        .await;
    let child_line = all
        .sessions
        .iter()
        .find(|i| i.session == child)
        .expect("the child appears in the All scope");
    assert_eq!(
        child_line.role,
        SessionRole::ManagedChild,
        "the durable delegation child must be stamped ManagedChild (A1)"
    );
    assert_eq!(
        child_line.parent.as_ref(),
        Some(&session),
        "the child must record its delegating parent"
    );

    // TopLevel (the inbox) excludes the managed child; the parent stays.
    let top = node.sessions_query(SessionQuery::default()).await.sessions;
    assert!(
        top.iter().all(|i| i.role == SessionRole::Primary),
        "TopLevel must contain only Primary conversations"
    );
    assert!(
        !top.iter().any(|i| i.session == child),
        "the managed child must NOT leak into the TopLevel inbox (A1 regression)"
    );

    // Transport parity: the scoped roster agrees in-process and over the socket.
    let socket_top = match client
        .call(ApiRequest::SessionsQuery {
            query: SessionQuery::default(),
        })
        .await
        .unwrap()
    {
        ApiResponse::SessionPage(page) => page.sessions,
        other => panic!("expected SessionPage, got {other:?}"),
    };
    assert_eq!(
        top, socket_top,
        "TopLevel roster must agree across transports"
    );

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// The scoped roster's cursor pagination is *total*: walking `All` one bounded page at a time
/// visits every session exactly once and terminates (no gaps, no repeats, `next_cursor == None`
/// on the last page). The order is stable (most-recent-first, id tie-break) so the cursor is
/// well-defined even when activity timestamps collide.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_pagination_cursor_is_total() {
    use daemon_api::{SessionApi, SessionQuery, SessionScope};
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();

    // Several live top-level sessions.
    let ids: Vec<SessionId> = (0..5)
        .map(|n| SessionId::new(format!("page-{n}")))
        .collect();
    for id in &ids {
        node.submit(
            id.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
        )
        .await
        .expect("submit opens a live session");
    }

    // The full unpaged view (the ground truth).
    let full: Vec<SessionId> = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            ..Default::default()
        })
        .await
        .sessions
        .into_iter()
        .map(|i| i.session)
        .collect();
    assert!(
        ids.iter().all(|id| full.contains(id)),
        "every live session is in the All roster, got {full:?}"
    );

    // Walk it one page of two at a time, accumulating ids.
    let mut seen: Vec<SessionId> = Vec::new();
    let mut after: Option<SessionId> = None;
    let mut pages = 0;
    loop {
        let page = node
            .sessions_query(SessionQuery {
                scope: SessionScope::All,
                after: after.clone(),
                limit: 2,
                since_rev: None,
            })
            .await;
        for info in &page.sessions {
            assert!(
                !seen.contains(&info.session),
                "a session must not appear on two pages: {}",
                info.session
            );
            seen.push(info.session.clone());
        }
        pages += 1;
        assert!(pages <= 16, "pagination must terminate");
        match page.next_cursor {
            Some(cursor) => after = Some(cursor),
            None => break,
        }
    }
    assert_eq!(
        seen, full,
        "paginated traversal must visit exactly the unpaged set, in the same order"
    );

    handle.shutdown().await;
}

/// `sessions_by_profile` groups the `Primary` roster by bound profile (the per-agent view). A
/// session opened "as agent X" (sticky profile bind on first open) lands under that profile's
/// group; a managed child never appears (it is not `Primary`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_sessions_by_profile_groups_primary_sessions() {
    use daemon_api::SessionApi;
    use daemon_common::ProfileRef;
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();
    let profile = ProfileRef::new("openai");

    let ids: Vec<SessionId> = (0..2)
        .map(|n| SessionId::new(format!("byprof-{n}")))
        .collect();
    for id in &ids {
        node.submit_as(daemon_api::SubmitAsArgs {
            session: id.clone(),
            origin: None,
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
            profile: Some(profile.clone()),
        })
        .await
        .expect("submit_as binds the profile and opens the session");
    }

    let grouped = node.sessions_by_profile().await;
    let group = grouped
        .iter()
        .find(|(p, _)| p == &profile)
        .map(|(_, s)| s)
        .expect("a group for the bound profile");
    for id in &ids {
        assert!(
            group.iter().any(|i| &i.session == id),
            "session {id} must appear under its bound profile"
        );
    }

    handle.shutdown().await;
}

/// L4 delta roster (daemon-sync-protocol-spec.md §6): `SessionsQuery` stamps a monotonic `rev`;
/// `since_rev` returns only the sessions changed after that revision; a `since_rev` ahead of the
/// node's rev (the daemon-restart case, in-memory index reset) falls back to a full page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roster_delta_since_rev_returns_changed_and_falls_back_to_full() {
    use daemon_api::{ControlApi, SessionApi, SessionMetaPatch, SessionQuery, SessionScope};
    use daemon_protocol::{AgentCommand, UserMsg};

    let (node, handle) = assemble();

    // Two live sessions; each submit activates (RosterChanged) + notes activity
    // (SessionMetaChanged), so the roster rev advances.
    for id in ["d-a", "d-b"] {
        node.submit(
            SessionId::new(id),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: daemon_common::ReqId(1),
            },
        )
        .await
        .expect("submit opens a live session");
    }

    // A full page (no since_rev) is the baseline; capture its rev.
    let full = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            ..Default::default()
        })
        .await;
    let r1 = full.rev;
    assert!(r1 > 0, "the roster rev advances as sessions activate");
    assert!(
        full.removed.is_empty(),
        "a full page carries no removed list"
    );

    // Nothing changed since r1 -> an empty delta at the same rev.
    let empty = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            since_rev: Some(r1),
            ..Default::default()
        })
        .await;
    assert!(
        empty.sessions.is_empty(),
        "no changes since r1 -> empty delta, got {:?}",
        empty.sessions
    );
    assert_eq!(empty.rev, r1);

    // Rename d-a; only it should come back in a delta past r1.
    node.session_update_meta(
        SessionId::new("d-a"),
        SessionMetaPatch {
            title: Some(Some("renamed".into())),
            ..Default::default()
        },
    )
    .await
    .expect("rename d-a");
    let delta = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            since_rev: Some(r1),
            ..Default::default()
        })
        .await;
    let ids: Vec<String> = delta
        .sessions
        .iter()
        .map(|i| i.session.as_str().to_string())
        .collect();
    assert!(
        ids.iter().any(|s| s == "d-a"),
        "the renamed session is in the delta, got {ids:?}"
    );
    assert!(
        !ids.iter().any(|s| s == "d-b"),
        "an unchanged session is NOT in the delta, got {ids:?}"
    );
    assert!(delta.rev > r1, "the rev advanced past the rename");

    // A since_rev ahead of the node's rev (daemon restarted, index reset) is unservable -> the
    // server returns a full page so the client replaces its roster.
    let fallback = node
        .sessions_query(SessionQuery {
            scope: SessionScope::All,
            since_rev: Some(delta.rev + 1000),
            ..Default::default()
        })
        .await;
    assert!(
        fallback
            .sessions
            .iter()
            .any(|i| i.session.as_str() == "d-b"),
        "an unservable since_rev falls back to a full page (all sessions present)"
    );

    // Scope-relative removal: archiving d-b makes it leave the TopLevel scope. A TopLevel delta
    // past the pre-archive rev must report it under `removed` (so the client prunes it), not just
    // silently omit it.
    let top_rev = node.sessions_query(SessionQuery::default()).await.rev;
    node.session_update_meta(
        SessionId::new("d-b"),
        SessionMetaPatch {
            archived: Some(true),
            ..Default::default()
        },
    )
    .await
    .expect("archive d-b");
    let top_delta = node
        .sessions_query(SessionQuery {
            since_rev: Some(top_rev),
            ..Default::default()
        })
        .await;
    assert!(
        top_delta.removed.iter().any(|s| s.as_str() == "d-b"),
        "an archived session must appear in the TopLevel delta's removed list, got {:?}",
        top_delta.removed
    );
    assert!(
        !top_delta
            .sessions
            .iter()
            .any(|i| i.session.as_str() == "d-b"),
        "the archived session must not be in the TopLevel delta body"
    );

    handle.shutdown().await;
}

/// Live fleet push (Phase-3 B, I4/I8): `tree_subscribe` is a real event-driven merge, not a
/// poll. Proves: (a) the stream opens with an immediate `Snapshot`; (b) a delegation spawn pushes
/// a live delta **promptly** — well inside what any old fixed poll interval would have been; and
/// (c) `include_ephemeral=false` still delivers the (non-ephemeral) managed-child delta.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_subscribe_pushes_delegation_spawn_promptly() {
    use daemon_api::{ControlApi, TreeEvent, TreeSubFilter};
    use futures::StreamExt;

    let (node, handle) = assemble();

    // Subscribe first (stable topology only) so no spawn delta is missed.
    let mut stream = node
        .tree_subscribe(TreeSubFilter {
            include_ephemeral: false,
            coalesce_ms: None,
        })
        .await
        .expect("tree_subscribe opens");

    // (a) The first event is the initial snapshot.
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("an initial event arrives")
        .expect("the stream yields");
    assert!(
        matches!(first, TreeEvent::Snapshot(_)),
        "the stream must open with a Snapshot, got {first:?}"
    );

    // Drive one durable delegation (the default node delegates once on Assign).
    node.assign(SessionId::new("push-op"))
        .await
        .expect("assign drives a delegation");

    // (b)+(c) A live delta arrives promptly (a managed-child spawn passes the ephemeral filter).
    // No poll interval is involved: the bus pushes the delta as soon as the child is created.
    let pushed = tokio::time::timeout(Duration::from_secs(10), async {
        match stream.next().await {
            Some(ev) => ev,
            None => panic!("the stream closed before a live delta"),
        }
    })
    .await
    .expect("a live delta is pushed promptly after the spawn");
    match pushed {
        // The forward-every-delta path delivers the spawn marker directly.
        TreeEvent::Subagent(view) => assert!(
            matches!(
                view,
                daemon_protocol::ManageEventView::Subagent { .. }
                    | daemon_protocol::ManageEventView::Started { .. }
                    | daemon_protocol::ManageEventView::Finished { .. }
                    | daemon_protocol::ManageEventView::Progress { .. }
                    | daemon_protocol::ManageEventView::Usage { .. }
                    | daemon_protocol::ManageEventView::Error { .. }
            ),
            "a subagent delta is pushed"
        ),
        // A re-projected snapshot is also an acceptable prompt push.
        TreeEvent::Snapshot(_) => {}
    }

    handle.shutdown().await;
}

/// The recursive durable delegation tree (the GUI's real surface), re-sourced from the durable
/// session graph: one delegation chain two levels deep — top -> orchestrator child -> leaf
/// grandchild — projects a genuine multi-level tree where every node, including the *grandchild*,
/// is addressable by `UnitId` at its true depth. A node is an orchestrator iff it actually
/// delegated (has durable children). The whole projection (and the grandchild's verifiable
/// history) is byte-identical in-process and over the socket.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_tree_projection_is_recursive_and_transport_agnostic() {
    let (node, handle) = assemble_nested(1);
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // One durable delegation: the top fleet spawns an orchestrator child, which delegates to a
    // leaf grandchild in its own sub-fleet — two levels below the node root.
    let session = SessionId::new("nest-op");
    assert!(matches!(
        client
            .call(ApiRequest::Assign {
                session: session.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let ApiResponse::Sessions(list) = client.call(ApiRequest::Sessions).await.unwrap() {
            if list
                .iter()
                .any(|i| i.session == session && i.state == SessionState::Completed)
            {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the assigned session never completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // (a) tree() projects root + 2+ levels with correct per-node children ids, identically on
    // both transports.
    let inproc_tree = node.tree().await;
    let socket_tree = match client.call(ApiRequest::Tree).await.unwrap() {
        ApiResponse::Tree(t) => t,
        other => panic!("expected Tree, got {other:?}"),
    };
    assert_eq!(
        inproc_tree, socket_tree,
        "tree must agree across transports"
    );

    let root = socket_tree.root.clone().expect("the node tree is rooted");
    let orchestrator = socket_tree
        .nodes
        .iter()
        .find(|n| n.kind == daemon_api::UnitKind::Orchestrator && n.id != root)
        .expect("an orchestrator child is present")
        .clone();
    let grandchild = socket_tree
        .nodes
        .iter()
        .find(|n| n.kind == daemon_api::UnitKind::Engine)
        .expect("a leaf grandchild is present")
        .clone();
    // The root owns the orchestrator; the orchestrator owns the grandchild (real nesting).
    assert!(
        socket_tree
            .nodes
            .iter()
            .find(|n| n.id == root)
            .unwrap()
            .children
            .contains(&orchestrator.id),
        "the node root's children include the orchestrator"
    );
    assert!(
        orchestrator.children.contains(&grandchild.id),
        "the orchestrator's children include the grandchild ({:?} not in {:?})",
        grandchild.id,
        orchestrator.children
    );
    assert!(
        grandchild.id.as_str().contains('/'),
        "the grandchild id is namespaced under its sub-fleet, got {:?}",
        grandchild.id
    );

    // (b) unit / unit_events / unit_outbound / unit_history resolve the *grandchild* by id at
    // depth, identically on both transports.
    let inproc_unit = node.unit(grandchild.id.clone()).await;
    let socket_unit = match client
        .call(ApiRequest::Unit {
            unit: grandchild.id.clone(),
        })
        .await
        .unwrap()
    {
        ApiResponse::Unit(u) => u,
        other => panic!("expected Unit, got {other:?}"),
    };
    assert_eq!(inproc_unit, socket_unit, "grandchild unit view must agree");
    assert_eq!(
        socket_unit.expect("grandchild resolves by id").id,
        grandchild.id,
        "the resolved node is the grandchild"
    );

    let socket_events = match client
        .call(ApiRequest::UnitEvents {
            unit: grandchild.id.clone(),
            max: 0,
        })
        .await
        .unwrap()
    {
        ApiResponse::UnitEvents(e) => e,
        other => panic!("expected UnitEvents, got {other:?}"),
    };
    assert_eq!(
        node.unit_events(grandchild.id.clone(), 0).await,
        socket_events,
        "grandchild events must agree across transports"
    );
    assert!(
        !socket_events.is_empty(),
        "expected buffered drill-down events for the grandchild"
    );

    // A durable session retains no *live* §17 outbound stream (it is driven one turn at a time
    // through activation, not a persistent actor): the rich, byte-faithful transcript is the
    // durable verifiable journal, read by id below via `unit_history`. So the live drain is empty
    // — identically on both transports.
    let socket_outbound = match client
        .call(ApiRequest::UnitOutbound {
            unit: grandchild.id.clone(),
            max: 0,
        })
        .await
        .unwrap()
    {
        ApiResponse::Drained(o) => o,
        other => panic!("expected Drained, got {other:?}"),
    };
    assert!(
        socket_outbound.is_empty(),
        "a durable grandchild has no live §17 drain; its transcript is the journal"
    );

    // The grandchild's durable, verifiable history routes by its id (it journaled its turn).
    let history_deadline = Instant::now() + Duration::from_secs(10);
    let socket_history = loop {
        let page = match client
            .call(ApiRequest::UnitHistory {
                unit: grandchild.id.clone(),
                after_cursor: 0,
                max: 0,
            })
            .await
            .unwrap()
        {
            ApiResponse::Journal(p) => p,
            other => panic!("expected Journal, got {other:?}"),
        };
        if !page.entries.is_empty() || Instant::now() >= history_deadline {
            break page;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        !socket_history.entries.is_empty(),
        "expected durable history entries for the grandchild"
    );
    assert_eq!(
        node.unit_history(grandchild.id.clone(), 0, 0).await,
        socket_history,
        "grandchild history must agree across transports"
    );

    // (c) pause/resume/scale are vestigial on the durable path: a durable session has no live
    // scheduling to pause/resume/scale (it is suspended/resumed by the activation lifecycle), so
    // these report Unsupported — identically on both transports — for an orchestrator session too.
    use daemon_api::ApiError;
    for (req, label) in [
        (
            ApiRequest::Pause {
                unit: orchestrator.id.clone(),
            },
            "pause",
        ),
        (
            ApiRequest::Resume {
                unit: orchestrator.id.clone(),
            },
            "resume",
        ),
        (
            ApiRequest::Scale {
                unit: orchestrator.id.clone(),
                n: 2,
            },
            "scale",
        ),
    ] {
        let socket = client.call(req).await.unwrap();
        assert!(
            matches!(socket, ApiResponse::Error(ApiError::Unsupported(_))),
            "{label} is vestigial on the durable path, got {socket:?}"
        );
    }
    assert!(node.pause(orchestrator.id.clone()).await.is_err());
    assert!(node.resume(orchestrator.id.clone()).await.is_err());
    assert!(node.scale(orchestrator.id.clone(), 2).await.is_err());

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// THE UNIFIED-DELEGATION RECOVERY GATE: a node crashes *mid* a nested durable delegation, and a
/// fresh node rebuilt from the same durable store alone — no in-memory state carried over —
/// recovers and unwinds the whole chain to completion. This is the new value the unified durable
/// orchestrator model unlocks: a nested delegation is as crash-recoverable as a top-level one,
/// because every level is a parent-bound durable session driven by the one shared outbox +
/// recovery scanner. Asserted against both store backends (`InMemoryStore`, `SqliteStore`).
async fn nested_delegation_recovers_after_restart(store: Arc<dyn SessionStore>) {
    let session = SessionId::new("rec-op");

    // Node A: a stalled cadence so its resident services never advance the delegation after the
    // synchronous `assign`. `assign` runs the top's first turn to a suspension with a delegation
    // job pending on the durable outbox and *no child created yet* — genuinely mid-delegation.
    let stalled = HostConfig {
        partition: PARTITION,
        dispatch_interval: Duration::from_secs(3600),
        scan_interval: Duration::from_secs(3600),
        ..HostConfig::default()
    };
    let AssembledNode {
        node: node_a,
        handle: handle_a,
        ..
    } = assemble_over(store.clone(), 1, [0x33; 32], stalled);
    node_a.assign(session.clone()).await.expect("assign");
    // The top is now mid-delegation in the durable store (suspended on / running toward a
    // delegation job), and node A's stalled cadence will not advance it any further.
    let after_assign = store.status(&session).await;
    assert!(
        !matches!(
            after_assign,
            Some(daemon_store::SessionStatus::Completed) | None
        ),
        "the top should be mid-flight (not completed) after assign, got {after_assign:?}"
    );
    // Crash: stop node A. The durable store retains the mid-flight top (+ any pending job).
    handle_a.shutdown().await;
    drop(node_a);

    // Node B: a fresh process over the *same* durable store. Its recovery scanner + dispatchers
    // drain the pending job, create+drive the child (which itself delegates to a leaf
    // grandchild), and resume the chain bottom-up to completion — all from durable state alone.
    let AssembledNode {
        node: node_b,
        handle: handle_b,
        ..
    } = assemble_over(store.clone(), 1, [0x33; 32], fast_host_config());
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if node_b
            .sessions()
            .await
            .iter()
            .any(|i| i.session == session && i.state == SessionState::Completed)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the nested delegation never recovered to completion"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The recovered tree shows the full depth: a depth-2 grandchild (two `/` path segments) is
    // present and addressable, proving the *nested* delegation — not just the top — recovered.
    let tree = node_b.tree().await;
    assert!(
        tree.nodes
            .iter()
            .any(|n| n.id.as_str().matches('/').count() == 2),
        "a depth-2 grandchild is present after recovery, got {:?}",
        tree.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>()
    );
    handle_b.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_delegation_recovers_after_restart_in_memory() {
    nested_delegation_recovers_after_restart(Arc::new(InMemoryStore::new())).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_delegation_recovers_after_restart_sqlite() {
    nested_delegation_recovers_after_restart(Arc::new(
        SqliteStore::open_in_memory().expect("open sqlite store"),
    ))
    .await;
}
