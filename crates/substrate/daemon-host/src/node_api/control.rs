// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl ControlApi for NodeApiImpl {
    async fn events_page(&self, cursor: u64, max: u32) -> EventsPage {
        match &self.node_events {
            Some(feed) => feed.page(cursor, max),
            None => EventsPage::default(),
        }
    }

    async fn events_subscribe(&self, cursor: u64) -> Result<NodeEventStream, ApiError> {
        Ok(match &self.node_events {
            Some(feed) => feed.subscribe(cursor),
            None => stream::empty().boxed(),
        })
    }

    async fn health(&self) -> HealthReport {
        let services = self
            .supervisor
            .service_names()
            .into_iter()
            .map(|name| {
                let restarts = self.supervisor.restarts(&name).unwrap_or(0);
                let (ok, detail) = match self.supervisor.health(&name) {
                    Some(HealthStatus::Ok) => (true, None),
                    Some(HealthStatus::Degraded { reason })
                    | Some(HealthStatus::Unhealthy { reason }) => (false, Some(reason)),
                    None => (false, Some("unknown service".to_string())),
                };
                ServiceHealth {
                    name,
                    ok,
                    restarts,
                    detail,
                }
            })
            .collect();
        HealthReport {
            all_ok: self.supervisor.all_ok(),
            services,
        }
    }

    async fn stats(&self) -> StatsReport {
        let s = self.store.stats().await;
        StatsReport {
            pending_jobs: s.pending_jobs as u64,
            pending_wakes: s.pending_wakes as u64,
            sessions: s.sessions as u64,
            active: self.manager.active_count() as u64,
            usage: self.folded_usage().await,
        }
    }

    async fn telemetry(&self) -> TelemetryDump {
        let s = self.store.stats().await;
        // Prefer the resident aggregator's folded usage + event count when present; otherwise fall
        // back to the durable per-session fold (with no event counter).
        let (usage, events) = match &self.metrics {
            Some(m) => (m.usage(), m.events()),
            None => (self.folded_usage().await, 0),
        };
        TelemetryDump {
            usage,
            events,
            healthy: self.supervisor.all_ok(),
            pending_jobs: s.pending_jobs as u64,
            pending_wakes: s.pending_wakes as u64,
            sessions: s.sessions as u64,
            active: self.manager.active_count() as u64,
        }
    }

    async fn sessions(&self) -> Vec<SessionInfo> {
        self.sessions_query(SessionQuery::default()).await.sessions
    }

    async fn sessions_query(&self, query: SessionQuery) -> SessionPage {
        // L4: the current roster revision this page reflects (0 when no feed is wired — the client
        // then always takes full pages). Read once up front so the page is consistent with the rev.
        let rev = self.node_feed().map(|f| f.roster_rev()).unwrap_or(0);
        // L4 delta read: when `since_rev` is set and the feed can serve it, restrict the page to the
        // sessions changed after that revision (+ the removed list) instead of the full roster. An
        // unservable `since_rev` (daemon restarted -> in-memory index reset) falls through to a full
        // page, which the client applies as a replace.
        let delta = match (query.since_rev, self.node_feed()) {
            (Some(since), Some(feed)) => feed.roster_delta(since),
            _ => None,
        };
        // Auth 4: the roster is scoped to the request principal (a peer sees only its own sessions;
        // an operator with `SessionSeeAll` sees all, incl. legacy owner-NULL rows) *before* the
        // `SessionScope`/sort/pagination below.
        let mut roster = self.roster_scoped().await;
        // The scope predicate, evaluated per session (so a delta can tell "still in scope" from "left
        // the scope"). `ByTransport` needs the live owned-session set, resolved once.
        let owned: std::collections::HashSet<SessionId> = match &query.scope {
            SessionScope::ByTransport(t) => self.live.delivery_sessions(t).into_iter().collect(),
            _ => std::collections::HashSet::new(),
        };
        let in_scope = |i: &SessionInfo| session_in_scope(i, &query.scope, &owned);
        // L4 removals are scope-relative: a session that changed but no longer matches the queried
        // scope (e.g. an archive leaving `TopLevel`, or a hard-removed id) must be pruned client-side,
        // so it rides the `removed` list rather than silently vanishing from the delta.
        let mut removed: Vec<SessionId> = Vec::new();
        let mut delta_served = false;
        if let Some((changed, removed_hard, _)) = &delta {
            let changed_set: std::collections::HashSet<&SessionId> = changed.iter().collect();
            removed.extend(removed_hard.iter().cloned());
            // Changed-but-now-out-of-scope sessions left the client's view.
            for i in roster
                .iter()
                .filter(|i| changed_set.contains(&i.session) && !in_scope(i))
            {
                removed.push(i.session.clone());
            }
            // Changed ids absent from the roster entirely (a true hard removal) also prune.
            let present: std::collections::HashSet<&SessionId> =
                roster.iter().map(|i| &i.session).collect();
            for id in changed.iter().filter(|id| !present.contains(id)) {
                removed.push((*id).clone());
            }
            if removed.len() <= daemon_api::WIRE_PAGE_MAX {
                // The page body is the changed + still-in-scope sessions.
                roster.retain(|i| changed_set.contains(&i.session) && in_scope(i));
                delta_served = true;
            } else {
                // Delta guard: `removed` rides unpaginated next to the page body, so a tombstone
                // list past the wire page bound would be un-decodable by the fixed-buffer client
                // codec. Serve a full page instead (the same fallback an unservable `since_rev`
                // takes below); the client applies it as a replace + prune, needing no removals.
                removed.clear();
            }
        }
        if !delta_served {
            roster.retain(|i| in_scope(i));
        }
        // Stable order: pinned conversations first, then most-recently-active, then id as the final
        // tie-break (so the cursor stays total across pages).
        roster.sort_by(|a, b| {
            b.pinned
                .cmp(&a.pinned)
                .then_with(|| b.last_activity_ms.cmp(&a.last_activity_ms))
                .then_with(|| a.session.as_str().cmp(b.session.as_str()))
        });
        // Cursor pagination: `after` is the last id of the previous page; skip through it.
        let next_cursor = paginate_roster(&mut roster, query.after.as_ref(), query.limit);
        SessionPage {
            sessions: roster,
            next_cursor,
            rev,
            // Populated only on a delta read (scope-relative + hard removals); a full page replaces
            // the client's roster wholesale, so it carries no removal list.
            removed,
        }
    }

    async fn session_get(&self, session: SessionId) -> Option<SessionDetail> {
        let status = self.store.status(&session).await;
        let is_live = self.live.live_ids().iter().any(|s| s == &session);
        if status.is_none() && !is_live {
            return None;
        }
        let meta = self.store.session_meta(&session).await.unwrap_or_default();
        // Auth 4 (read-of-one): mint the ownership proof (own-or-`SessionSeeAll`); behave as
        // not-found for a session the caller may not see (no existence oracle). The proof is also
        // what the guarded `delivery_targets` read below requires — so the compile-time gate and the
        // visibility gate are one and the same here.
        let Ok(auth) = self.require_session_access(&session, false).await else {
            return None;
        };
        let lifecycle = if status.is_some() {
            ApiLifecycle::Durable
        } else {
            ApiLifecycle::Live
        };
        let info = session_info_from(
            &session,
            status,
            &meta,
            lifecycle,
            self.session_rewindable(&session),
        );
        let overlay = (!meta.overlay.is_empty()).then(|| decode_overlay(&meta.overlay));
        let model = self.session_models.get(&session).map(|m| m.clone());
        let delivery_targets = self.live.delivery_targets(&auth);
        let children = self.store.children_of(&session).await;
        let checkpoints = match &self.checkpoints {
            Some(store) => store.list(Some(session.as_str())).await.len() as u32,
            None => 0,
        };
        Some(SessionDetail {
            info,
            overlay,
            model,
            delivery_targets,
            children,
            checkpoints,
        })
    }

    async fn session_search(&self, query: String, limit: u32) -> Vec<SessionSearchHit> {
        // Auth 4 (read-of-one): drop hits the caller may not see (own sessions only, unless the
        // caller holds `SessionSeeAll`). Owner is read per hit; the result set is small/capped.
        let principal = current_principal();
        let mut out = Vec::new();
        for hit in self.store.search_sessions(&query, limit).await {
            let owner = self
                .store
                .session_meta(&hit.session_id)
                .await
                .and_then(|m| m.owner);
            if !owner_visible(&principal, &owner) {
                continue;
            }
            out.push(SessionSearchHit {
                session: hit.session_id,
                title: hit.title,
                snippet: hit.snippet,
            });
        }
        out
    }

    async fn session_recap(&self, session: SessionId) -> Option<daemon_api::SessionRecap> {
        // Auth 4 (read-of-one): behave as not-found for a session the caller may not see (own
        // sessions only unless `SessionSeeAll`), exactly like `session_get` — no existence oracle.
        let meta = self.store.session_meta(&session).await.unwrap_or_default();
        if !owner_visible(&current_principal(), &meta.owner) {
            return None;
        }
        // Source order: the durable snapshot first (full fidelity — tool args feed `files_touched`;
        // for a resident mid-turn session this is its LAST CHECKPOINT, by design), else a resident
        // live session's conversation view (tool names only — `files_touched` stays empty there).
        let turns = match self.store.peek_snapshot(&session).await {
            Some(blob) => crate::session_index::turns_from_conversation(
                &Snapshot::decode(&blob).ok()?.conversation,
            ),
            None => crate::session_index::turns_from_view(&self.live.conv_view(&session).await?),
        };
        Some(crate::session_index::build_recap(&turns, meta.title))
    }

    async fn session_update_meta(
        &self,
        session: SessionId,
        patch: SessionMetaPatch,
    ) -> Result<(), ApiError> {
        // Auth 4: rename/pin/archive is a control op — owner or `SessionControlAny` only.
        self.require_session_access(&session, true).await?;
        // Read-modify-write of the durable `SessionMeta`, preserving the fields the patch does not
        // touch (overlay/role/parent/bound profile/last activity/owner). Each `None` patch field is a
        // leave-unchanged; `title: Some(None)` clears the title (rename-to-empty).
        let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
        if let Some(title) = patch.title {
            meta.title = title;
        }
        if let Some(pinned) = patch.pinned {
            meta.pinned = pinned;
        }
        if let Some(archived) = patch.archived {
            meta.archived = archived;
        }
        self.store
            .set_session_meta(&session, meta)
            .await
            .map_err(|e| ApiError::Other(e.to_string()))?;
        // Nudge live roster/tree subscribers so the rename/pin/archive shows up without a poll.
        self.emit_tree_changed();
        // L3: a rename/pin/archive changed this session's roster metadata.
        if let Some(feed) = self.node_feed() {
            let rev = feed.note_roster_change(&session);
            feed.emit(NodeEvent::SessionMetaChanged { session, rev });
        }
        Ok(())
    }

    async fn approvals_pending(
        &self,
        session: Option<SessionId>,
        after: Option<String>,
    ) -> daemon_api::WirePage<ApprovalInfo> {
        let mut approvals: Vec<ApprovalInfo> = self
            .store
            .pending_approvals_of(session.as_ref())
            .await
            .into_iter()
            .map(|p| ApprovalInfo {
                session: p.session_id,
                request_id: p.job_id.as_str().to_string(),
                prompt: p.prompt,
                path: p.path,
            })
            .collect();
        // The store lists in rowseq (arrival) order; the cursor is the request_id, so sort by it
        // before slicing — the pending set is small and a deterministic id order keeps the cursor
        // stable across pages even as decisions land in between.
        approvals.sort_by(|a, b| a.request_id.cmp(&b.request_id));
        daemon_api::paginate(
            approvals,
            after.as_deref(),
            daemon_api::WIRE_PAGE_MAX,
            |a| a.request_id.clone(),
        )
    }

    async fn approval_decide(
        &self,
        session: SessionId,
        request_id: String,
        allow: bool,
    ) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may decide a session's approval.
        self.require_session_access(&session, true).await?;
        // Record the decision + enqueue the wake durably (one transaction in the store), then nudge
        // the activation manager so the dormant session rehydrates promptly and resolves the gated
        // tool call (allow -> runs it; deny -> injects a tool error). Idempotent in the store.
        let answered = self
            .store
            .answer_approval(&session, &JobId::new(request_id.clone()), allow)
            .await
            .map_err(|e| ApiError::Other(format!("answer approval: {e}")))?;
        if !answered {
            return Err(ApiError::Other(format!(
                "no pending approval {request_id} on session {session}"
            )));
        }
        self.manager
            .wake(session)
            .await
            .map_err(|e| ApiError::Other(format!("wake: {e}")))
    }

    async fn assign(&self, session: SessionId) -> Result<(), ApiError> {
        // Auth 4: a peer may only (re)assign a session it owns; an `Absent` session is allowed
        // through so the create-below path can stamp the caller as owner. `SessionControlAny`
        // (operator) crosses ownership.
        self.require_session_access(&session, true).await?;
        // Guard-rail: a session driven through the durable control surface must not also be a live
        // interactive session (two divergent engine instances for one id).
        self.claim(&session, Lifecycle::Durable)?;
        // Create-if-absent: a fresh durable session row with the engine's initial snapshot.
        if self.store.status(&session).await.is_none() {
            let blob = Snapshot::fresh(session.clone())
                .encode()
                .map_err(|e| ApiError::Other(format!("encode initial snapshot: {e}")))?;
            self.store
                .create_session(session.clone(), self.partition, blob)
                .await
                .map_err(|e| ApiError::Other(format!("create session: {e}")))?;
            // Stamp ownership from the request principal (Auth 4); inherited paths (delegation /
            // background / cron) stamp at their own creation sites. Best-effort meta upsert.
            let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
            if meta.owner.is_none() {
                meta.owner = current_principal().map(|p| p.user_id);
                let _ = self.store.set_session_meta(&session, meta).await;
            }
        }
        // Wake it: the activation manager runs (or resumes) the engine; the resident services then
        // carry the durable delegate -> suspend -> resume -> complete cycle forward.
        self.manager
            .wake(session)
            .await
            .map_err(|e| ApiError::Other(format!("wake: {e}")))
    }

    async fn cancel(&self, session: SessionId) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may cancel a session.
        let auth = self.require_session_access(&session, true).await?;
        // Best-effort: cancel a matching fleet child and interrupt a matching live session.
        if let Some(fleet) = &self.fleet {
            fleet.cancel(&UnitId::new(session.as_str())).await;
        }
        self.live.interrupt(&auth).await;
        // Release the lifecycle claim so the id can be reused by either surface.
        self.owners.remove(&session);
        Ok(())
    }

    async fn fleet(&self) -> FleetReport {
        match &self.fleet {
            Some(fleet) => fleet.report().await,
            None => FleetReport::default(),
        }
    }

    async fn tree(&self, after: Option<String>) -> TreeReport {
        // Auth 4: the orchestration tree is scoped to the request principal (children inherit the
        // parent owner, so an owned subtree is kept whole and a foreign one dropped whole).
        let mut report = self.tree_owned(&current_principal()).await;
        // Page `nodes` in unit-id order (the fleet projection has no stable order of its own);
        // `root` rides every page and the id-linked structure reassembles client-side.
        report
            .nodes
            .sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        let page = daemon_api::paginate(
            report.nodes,
            after.as_deref(),
            daemon_api::WIRE_PAGE_MAX,
            |n| n.id.as_str().to_string(),
        );
        report.nodes = page.items;
        report.next = page.next;
        report
    }

    async fn unit(&self, id: UnitId) -> Option<UnitNode> {
        match &self.fleet {
            Some(fleet) => fleet.unit(&id).await,
            None => None,
        }
    }

    async fn tree_subscribe(
        &self,
        filter: daemon_api::TreeSubFilter,
    ) -> Result<daemon_api::TreeStream, ApiError> {
        // Real event-driven merge (I4/I8): subscribe to the host fleet bus *first* (so no delta is
        // lost between the initial snapshot and the live tail), emit the current snapshot, then
        // forward live topology deltas. The `TreeSubFilter` is applied on the way out:
        //   - `include_ephemeral=false` drops `EphemeralSubagent` nodes from snapshots and drops
        //     `Subagent` deltas whose role is ephemeral (stable-topology-only subscribers).
        //   - `coalesce_ms` debounces a burst of deltas into one fresh `tree()` snapshot; `None`
        //     forwards every delta as it arrives.
        let this = self.clone();
        let rx = self.fleet_events.as_ref().map(|tx| tx.subscribe());

        // Auth 4: capture the subscriber's principal AT SUBSCRIBE TIME — the returned long-lived
        // stream is polled outside this request's task-local scope, so `current_principal()` would be
        // `None` there. Every snapshot/delta below is owner-scoped to this captured principal.
        let principal = current_principal();
        // An operator (`SessionSeeAll`) gets the efficient per-delta stream; any other subscriber
        // gets owner-scoped *snapshots* re-projected on each bus wake, so a foreign node can never
        // ride a raw delta to them.
        let sees_all = principal
            .as_ref()
            .is_some_and(|p| p.has(daemon_auth::Capability::SessionSeeAll));

        // The initial snapshot is always emitted, bus or not — owner-scoped + ephemeral-filtered.
        let initial =
            daemon_api::TreeEvent::Snapshot(filtered_tree(&this, &filter, &principal).await);

        // No bus wired: fall back to the snapshot-only foundation (a single initial snapshot).
        let Some(rx) = rx else {
            return Ok(stream::once(async move { initial }).boxed());
        };

        let live = stream::unfold(
            (this, rx, filter, principal, sees_all),
            move |(this, mut rx, filter, principal, sees_all)| async move {
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            if filter.coalesce_ms.is_some() || !sees_all {
                                // Coalescing subscribers, and every non-operator subscriber, get a
                                // fresh owner-scoped re-projection rather than a raw delta (the delta
                                // owner-filter would otherwise have to map every event variant to a
                                // session). Drain any burst first when coalescing.
                                if let Some(window) = filter.coalesce_ms {
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        window.max(1),
                                    ))
                                    .await;
                                    while rx.try_recv().is_ok() {}
                                }
                                let report = filtered_tree(&this, &filter, &principal).await;
                                return Some((
                                    daemon_api::TreeEvent::Snapshot(report),
                                    (this, rx, filter, principal, sees_all),
                                ));
                            }
                            // Operator, no coalescing: forward the delta, applying the ephemeral
                            // filter (an operator sees every owner's nodes by `SessionSeeAll`).
                            match forward_event(event, &filter) {
                                Some(out) => {
                                    return Some((out, (this, rx, filter, principal, sees_all)))
                                }
                                None => continue,
                            }
                        }
                        // We fell behind the bus: re-sync with a fresh authoritative snapshot.
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let report = filtered_tree(&this, &filter, &principal).await;
                            return Some((
                                daemon_api::TreeEvent::Snapshot(report),
                                (this, rx, filter, principal, sees_all),
                            ));
                        }
                        Err(broadcast::error::RecvError::Closed) => return None,
                    }
                }
            },
        );

        let stream = stream::once(async move { initial }).chain(live);
        Ok(stream.boxed())
    }

    async fn routing_list_chats(&self, after: Option<String>) -> daemon_api::WirePage<ChatRoute> {
        // The store lists `ORDER BY key` (key = origin_pin_key), so the listing is already in
        // cursor order; the cursor is recomputed from each route's origin.
        let routes: Vec<ChatRoute> = self
            .store
            .routing_list()
            .await
            .iter()
            .filter_map(wire_route_from_store)
            .collect();
        daemon_api::paginate(routes, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |r| {
            crate::routing::origin_pin_key(&r.origin)
        })
    }

    async fn routing_get(&self, origin: Origin) -> Option<ChatRoute> {
        let key = crate::routing::origin_pin_key(&origin);
        self.store
            .routing_get(&key)
            .await
            .as_ref()
            .and_then(wire_route_from_store)
    }

    async fn routing_set(&self, route: ChatRoute) -> Result<(), ApiError> {
        self.store
            .routing_set(store_route_from_wire(&route))
            .await
            .map_err(|e| ApiError::Other(format!("routing set: {e}")))?;
        // Ride the §5.9 hot-reload seam: reload pins into the live registry so the new pin resolves
        // immediately (resolve-first), without a restart.
        self.load_routing_pins().await;
        Ok(())
    }

    async fn routing_bind_chat(
        &self,
        origin: Origin,
        session: SessionId,
        profile: Option<ProfileRef>,
    ) -> Result<(), ApiError> {
        // The convenience form: a pin with the registry's default (`PerThread`) naming — the pinned
        // session id is authoritative, so the recorded isolation is informational.
        self.routing_set(ChatRoute {
            origin,
            session,
            profile,
            isolation: IsolationPolicy::PerThread,
        })
        .await
    }

    async fn routing_unbind_chat(&self, origin: Origin) -> Result<(), ApiError> {
        let key = crate::routing::origin_pin_key(&origin);
        self.store
            .routing_remove(&key)
            .await
            .map_err(|e| ApiError::Other(format!("routing unbind: {e}")))?;
        self.load_routing_pins().await;
        Ok(())
    }

    async fn transport_rooms(
        &self,
        transport: TransportId,
        after: Option<String>,
    ) -> daemon_api::WirePage<RoomInfo> {
        // Read-only enumeration backed by the durable routing pins: the rooms this transport instance
        // (or family) has a pin for, each carrying its pinned session. A live adapter-backed room
        // listing (e.g. Matrix joined rooms) can layer on later behind the same shape.
        let mut rooms: Vec<RoomInfo> = self
            .store
            .routing_list()
            .await
            .iter()
            .filter_map(wire_route_from_store)
            .filter(|r| transport_family_matches(&r.origin.transport, &transport))
            .map(|r| RoomInfo {
                transport: r.origin.transport.clone(),
                room: room_label(&r.origin.scope),
                name: None,
                session: Some(r.session.clone()),
            })
            .collect();
        // The cursor is the room label, so re-sort by it (the pins arrive in pin-key order).
        rooms.sort_by(|a, b| a.room.cmp(&b.room));
        daemon_api::paginate(rooms, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |r| {
            r.room.clone()
        })
    }

    async fn transport_adapters(&self) -> Vec<AdapterInfo> {
        // Read-only enumeration from the host adapter registry (daemon-transport-adapter-spec.md
        // §3.4). Empty until the assembling binary installs adapters via `with_adapters`; lifecycle
        // (`serve`) still runs from `bins/daemon` in the skeleton. `transport_instances` (live
        // per-account connection/presence) is deferred and inherits the empty `ControlApi` default.
        self.adapters.load().infos()
    }

    async fn transport_instances(&self) -> Vec<TransportInstanceInfo> {
        // Live per-account connection/presence, aggregated across every registered adapter.
        self.adapters.load_full().instances().await
    }

    // ----- messaging-adapter management (daemon-messaging-adapter-spec.md §6): forwarded generically
    // through the registry to the addressed transport's `MessagingProtocol` feature traits; mutating
    // ops are sealed onto the verifiable `node-management` stream. -----

    async fn conv_list(
        &self,
        transport: TransportId,
        after: Option<String>,
    ) -> daemon_api::WirePage<ConversationInfo> {
        // The adapters return unbounded, adapter-ordered listings; sort + page here (once) rather
        // than teaching every `SupportsConversations` impl the cursor. The cursor is the
        // conversation id.
        let mut convs = match self.conversations_for(&transport) {
            Ok(c) => c.list(transport).await,
            Err(_) => Vec::new(),
        };
        convs.sort_by(|a, b| a.id.cmp(&b.id));
        daemon_api::paginate(convs, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |c| {
            c.id.clone()
        })
    }

    async fn conv_get(&self, transport: TransportId, conv: String) -> Option<ConversationInfo> {
        self.conversations_for(&transport)
            .ok()?
            .get(transport, conv)
            .await
    }

    async fn conv_create_details(&self, transport: TransportId) -> CreateConversationDetails {
        match self.conversations_for(&transport) {
            Ok(c) => c.create_details(transport).await,
            Err(_) => CreateConversationDetails::default(),
        }
    }

    async fn conv_create(
        &self,
        transport: TransportId,
        details: CreateConversationDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let info = self
            .conversations_for(&transport)?
            .create(transport, details)
            .await?;
        self.audit_management(
            "mgmt.conv.create",
            format!(
                "transport={} conv={} kind={:?}",
                info.transport.as_str(),
                info.id,
                info.kind
            ),
        )
        .await;
        Ok(info)
    }

    async fn conv_join_details(&self, transport: TransportId) -> ChannelJoinDetails {
        match self.conversations_for(&transport) {
            Ok(c) => c.channel_join_details(transport).await,
            Err(_) => ChannelJoinDetails::default(),
        }
    }

    async fn conv_join(
        &self,
        transport: TransportId,
        details: ChannelJoinDetails,
    ) -> Result<ConversationInfo, ApiError> {
        let info = self
            .conversations_for(&transport)?
            .join_channel(transport, details)
            .await?;
        self.audit_management(
            "mgmt.conv.join",
            format!("transport={} conv={}", info.transport.as_str(), info.id),
        )
        .await;
        Ok(info)
    }

    async fn conv_leave(&self, transport: TransportId, conv: String) -> Result<(), ApiError> {
        let detail = format!("transport={} conv={}", transport.as_str(), conv);
        self.audited(
            "mgmt.conv.leave",
            detail,
            self.conversations_for(&transport)?
                .leave(transport.clone(), conv.clone()),
        )
        .await
    }

    async fn conv_send(&self, args: ConvSendArgs) -> Result<(), ApiError> {
        let detail = format!("transport={} conv={}", args.transport.as_str(), args.conv);
        self.audited(
            "mgmt.conv.send",
            detail,
            self.conversations_for(&args.transport)?.send(args),
        )
        .await
    }

    async fn conv_set_topic(
        &self,
        transport: TransportId,
        conv: String,
        topic: Option<String>,
    ) -> Result<(), ApiError> {
        let detail = format!(
            "transport={} conv={} topic={:?}",
            transport.as_str(),
            conv,
            topic
        );
        self.audited(
            "mgmt.conv.set_topic",
            detail,
            self.conversations_for(&transport)?.set_topic(
                transport.clone(),
                conv.clone(),
                topic.clone(),
            ),
        )
        .await
    }

    async fn conv_set_title(
        &self,
        transport: TransportId,
        conv: String,
        title: Option<String>,
    ) -> Result<(), ApiError> {
        let detail = format!(
            "transport={} conv={} title={:?}",
            transport.as_str(),
            conv,
            title
        );
        self.audited(
            "mgmt.conv.set_title",
            detail,
            self.conversations_for(&transport)?.set_title(
                transport.clone(),
                conv.clone(),
                title.clone(),
            ),
        )
        .await
    }

    async fn conv_set_description(
        &self,
        transport: TransportId,
        conv: String,
        description: Option<String>,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} conv={}", transport.as_str(), conv);
        self.audited(
            "mgmt.conv.set_description",
            detail,
            self.conversations_for(&transport)?.set_description(
                transport.clone(),
                conv.clone(),
                description.clone(),
            ),
        )
        .await
    }

    async fn conv_delete(&self, transport: TransportId, conv: String) -> Result<(), ApiError> {
        let detail = format!("transport={} conv={}", transport.as_str(), conv);
        self.audited(
            "mgmt.conv.delete",
            detail,
            self.conversations_for(&transport)?
                .delete(transport.clone(), conv.clone()),
        )
        .await
    }

    async fn conv_history(&self, args: ConvHistoryArgs) -> JournalPageView {
        let ConvHistoryArgs {
            transport,
            conv,
            after_cursor,
            max,
        } = args;
        // The merged conversation transcript is a verifiable journal stream keyed generically as
        // `conv:<transport>:<conv>` (the same id the messaging adapter writes its posts to); reuse the
        // shared history reader so the blocks are decoded + segment-verified like any other stream.
        let stream = JournalStreamId::unit(&UnitId::new(format!(
            "conv:{}:{}",
            transport.as_str(),
            conv
        )));
        self.read_history(stream, after_cursor, max).await
    }

    async fn member_invite(&self, args: MemberInviteArgs) -> Result<(), ApiError> {
        let label = participant_label(&args.who);
        let detail = format!(
            "transport={} conv={} who={label}",
            args.transport.as_str(),
            args.conv
        );
        self.audited(
            "mgmt.member.invite",
            detail,
            self.membership_for(&args.transport)?.invite(args),
        )
        .await
    }

    async fn member_remove(&self, args: MemberRemoveArgs) -> Result<(), ApiError> {
        let label = participant_label(&args.who);
        let detail = format!(
            "transport={} conv={} who={label}",
            args.transport.as_str(),
            args.conv
        );
        self.audited(
            "mgmt.member.remove",
            detail,
            self.membership_for(&args.transport)?.remove(args),
        )
        .await
    }

    async fn member_ban(&self, args: MemberBanArgs) -> Result<(), ApiError> {
        let label = participant_label(&args.who);
        let detail = format!(
            "transport={} conv={} who={label}",
            args.transport.as_str(),
            args.conv
        );
        self.audited(
            "mgmt.member.ban",
            detail,
            self.membership_for(&args.transport)?.ban(args),
        )
        .await
    }

    async fn member_set_role(&self, args: MemberSetRoleArgs) -> Result<(), ApiError> {
        let label = participant_label(&args.who);
        let detail = format!(
            "transport={} conv={} who={label} role={:?}",
            args.transport.as_str(),
            args.conv,
            args.role
        );
        self.audited(
            "mgmt.member.set_role",
            detail,
            self.membership_for(&args.transport)?.set_role(args),
        )
        .await
    }

    async fn contact_get_profile(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Result<String, ApiError> {
        self.contacts_for(&transport)?
            .get_profile(transport, contact)
            .await
    }

    async fn contact_action_menu(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Option<ActionMenu> {
        self.contacts_for(&transport)
            .ok()?
            .action_menu(transport, contact)
    }

    async fn contact_set_alias(
        &self,
        transport: TransportId,
        contact: ContactInfo,
        alias: Option<String>,
    ) -> Result<(), ApiError> {
        let id = contact.id.clone();
        let detail = format!(
            "transport={} contact={id} alias={alias:?}",
            transport.as_str()
        );
        self.audited(
            "mgmt.contact.set_alias",
            detail,
            self.contacts_for(&transport)?
                .set_alias(transport.clone(), contact, alias.clone()),
        )
        .await
    }

    async fn directory_search(
        &self,
        transport: TransportId,
        query: Option<String>,
    ) -> Result<Vec<ContactInfo>, ApiError> {
        self.directory_for(&transport)?
            .search_contacts(transport, query)
            .await
    }

    async fn acp_discover(&self) -> Vec<AcpAgentEntry> {
        // Probe the curated direct-binary recipe table via the injected ACP hook (the binary owns the
        // ACP runtime). Cache the results so `acp_catalog` surfaces them without re-probing, then
        // return the merged catalog (discovery results + durable manual registrations).
        if let Some(acp) = &self.acp {
            let discovered = acp.discover().await;
            *self.last_acp.write().unwrap() = discovered;
        }
        self.acp_catalog().await
    }

    async fn acp_catalog(&self) -> Vec<AcpAgentEntry> {
        // The durable manual registrations (source = Manual) take precedence over a builtin of the
        // same name; the in-memory last-discovery results fill in the auto-detected builtins.
        let mut by_name: std::collections::BTreeMap<String, AcpAgentEntry> =
            std::collections::BTreeMap::new();
        for entry in self.last_acp.read().unwrap().iter() {
            by_name.insert(entry.name.clone(), entry.clone());
        }
        for stored in self.store.acp_list().await {
            if let Ok(entry) = from_cbor::<AcpAgentEntry>(&stored.entry) {
                by_name.insert(entry.name.clone(), entry);
            }
        }
        by_name.into_values().collect()
    }

    async fn acp_register(&self, mut entry: AcpAgentEntry) -> Result<(), ApiError> {
        // A manual registration: force `source = Manual`, then verify/enrich it via the ACP
        // `initialize` handshake when a discovery hook is wired (fills installed/version/caps).
        entry.source = AcpSource::Manual;
        if let Some(acp) = &self.acp {
            entry = acp.probe(entry).await;
            entry.source = AcpSource::Manual;
        }
        self.store
            .acp_set(daemon_store::AcpEntry {
                name: entry.name.clone(),
                entry: to_cbor(&entry),
            })
            .await
            .map_err(|e| ApiError::Other(format!("acp register: {e}")))
    }

    async fn acp_remove(&self, name: String) -> Result<(), ApiError> {
        self.last_acp.write().unwrap().retain(|e| e.name != name);
        self.store
            .acp_remove(&name)
            .await
            .map_err(|e| ApiError::Other(format!("acp remove: {e}")))
    }

    // -- Cron (I15): every op delegates to the shared `CronOps`; absent it, the trait defaults
    //    (empty list / `Unsupported`) stand. --

    async fn cron_list(&self) -> Vec<daemon_api::CronJob> {
        match &self.cron {
            Some(cron) => cron.list().await,
            None => Vec::new(),
        }
    }

    async fn cron_create(&self, spec: daemon_api::CronSpec) -> Result<String, ApiError> {
        match &self.cron {
            Some(cron) => cron.create(spec).await,
            None => Err(ApiError::Unsupported("cron_create".into())),
        }
    }

    async fn cron_update(&self, id: String, spec: daemon_api::CronSpec) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.update(id, spec).await,
            None => Err(ApiError::Unsupported("cron_update".into())),
        }
    }

    async fn cron_delete(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.delete(id).await,
            None => Err(ApiError::Unsupported("cron_delete".into())),
        }
    }

    async fn cron_trigger(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.trigger(id).await,
            None => Err(ApiError::Unsupported("cron_trigger".into())),
        }
    }

    async fn cron_runs(&self, id: String) -> Vec<daemon_api::CronRun> {
        match &self.cron {
            Some(cron) => cron.runs(id).await,
            None => Vec::new(),
        }
    }

    async fn cron_pause(&self, id: String, paused: bool) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.pause(id, paused).await,
            None => Err(ApiError::Unsupported("cron_pause".into())),
        }
    }

    async fn cron_suggestions(&self) -> Vec<daemon_api::CronSuggestion> {
        match &self.cron {
            Some(cron) => cron.suggestions().await,
            None => Vec::new(),
        }
    }

    async fn cron_accept_suggestion(&self, id: String) -> Result<String, ApiError> {
        match &self.cron {
            Some(cron) => cron.accept_suggestion(id).await,
            None => Err(ApiError::Unsupported("cron_accept_suggestion".into())),
        }
    }

    async fn cron_dismiss_suggestion(&self, id: String) -> Result<(), ApiError> {
        match &self.cron {
            Some(cron) => cron.dismiss_suggestion(id).await,
            None => Err(ApiError::Unsupported("cron_dismiss_suggestion".into())),
        }
    }

    async fn unit_events(&self, id: UnitId, max: u32) -> Vec<ManageEventView> {
        match &self.fleet {
            Some(fleet) => fleet.unit_events(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_outbound(&self, id: UnitId, max: u32) -> Vec<Outbound> {
        match &self.fleet {
            Some(fleet) => fleet.unit_outbound(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_history(&self, id: UnitId, after_cursor: u64, max: u32) -> JournalPageView {
        self.read_history(JournalStreamId::unit(&id), after_cursor, max)
            .await
    }

    async fn pause(&self, id: UnitId) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.pause(&id).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("pause {id}"))),
        }
    }

    async fn resume(&self, id: UnitId) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.resume(&id).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("resume {id}"))),
        }
    }

    async fn scale(&self, id: UnitId, n: u32) -> Result<(), ApiError> {
        match &self.fleet {
            Some(fleet) if fleet.scale(&id, n).await => Ok(()),
            _ => Err(ApiError::Unsupported(format!("scale {id}"))),
        }
    }

    async fn verifying_key(&self) -> Option<String> {
        self.verifier.as_ref().map(|s| s.verifying_key().to_hex())
    }

    async fn checkpoints(
        &self,
        session: Option<SessionId>,
        after: Option<String>,
    ) -> daemon_api::WirePage<daemon_api::CheckpointInfo> {
        let Some(store) = &self.checkpoints else {
            return daemon_api::WirePage::default();
        };
        let filter = session.as_ref().map(|s| s.to_string());
        let mut checkpoints: Vec<daemon_api::CheckpointInfo> = store
            .list(filter.as_deref())
            .await
            .into_iter()
            .map(|r| daemon_api::CheckpointInfo {
                id: r.id,
                session: SessionId::new(r.session),
                tool: r.tool,
                created_unix: r.created_unix,
                // The turn/cursor correlation is not yet recorded on the checkpoint ledger; the wire
                // fields exist (rewind-unify foundation) and fill in when the ledger carries them.
                turn_ordinal: None,
                cursor: None,
            })
            .collect();
        // The uniform ascending-by-key page order (the cursor is the checkpoint id). The store's
        // newest-first order was an internal convenience with no wire consumers; a client wanting
        // newest-first re-sorts by `created_unix` after collecting its pages.
        checkpoints.sort_by(|a, b| a.id.cmp(&b.id));
        daemon_api::paginate(
            checkpoints,
            after.as_deref(),
            daemon_api::WIRE_PAGE_MAX,
            |c| c.id.clone(),
        )
    }

    async fn checkpoint_rewind(
        &self,
        session: SessionId,
        checkpoint_id: String,
    ) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may rewind a session.
        self.require_session_access(&session, true).await?;
        tracing::info!(
            trace_id = %current_trace(),
            session = %session,
            checkpoint_id = %checkpoint_id,
            "checkpoint.rewind.api"
        );
        let store = self
            .checkpoints
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("checkpoint_rewind".into()))?;
        let record = store
            .get(&checkpoint_id)
            .await
            .ok_or_else(|| ApiError::Other(format!("unknown checkpoint: {checkpoint_id}")))?;
        store
            .restore(&record)
            .await
            .map_err(|e| ApiError::Other(format!("rewind failed: {e}")))
    }

    // ----- filesystem / workspace surface (daemon-fs-surface-spec.md) -----

    async fn fs_roots(&self) -> Vec<FsRoot> {
        let Some(ws) = &self.workspace else {
            return Vec::new();
        };
        let mut roots = Vec::new();
        // Host browse roots (home + operator allowlist) — discovery before binding.
        for (id, _dir) in ws.roots().browse_roots() {
            roots.push(FsRoot {
                id: FsRootId::Host(id.clone()),
                label: id.clone(),
                kind: FsRootKind::Host,
                session: None,
            });
        }
        // The node workspace root.
        roots.push(FsRoot {
            id: FsRootId::Workspace,
            label: "workspace".to_string(),
            kind: FsRootKind::Workspace,
            session: None,
        });
        // Opened (live) session sandboxes.
        for sid in self.live.live_ids() {
            roots.push(FsRoot {
                id: FsRootId::Session(sid.clone()),
                label: sid.as_str().to_string(),
                kind: FsRootKind::Session,
                session: Some(sid),
            });
        }
        roots
    }

    async fn fs_list(
        &self,
        root: FsRootId,
        dir: String,
        show_ignored: bool,
        after: Option<String>,
    ) -> Result<FsListPage, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_list".into()))?;
        ws.list(&root, &dir, show_ignored, after.as_deref()).await
    }

    async fn fs_stat(&self, root: FsRootId, path: String) -> Result<FsEntry, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_stat".into()))?;
        ws.stat(&root, &path).await
    }

    async fn fs_read(
        &self,
        root: FsRootId,
        path: String,
        max_bytes: u64,
    ) -> Result<FsContent, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_read".into()))?;
        let mut content = ws.read(&root, &path, max_bytes).await?;
        // When a content store is bound and the whole file was returned, attach a content-addressed
        // ref so a client can hand the same bytes to an agent without re-uploading.
        if !content.truncated {
            if let Some(blobs) = &self.blobs {
                if let Ok(blob_ref) = blobs.put(&content.bytes).await {
                    content.blob_ref = Some(blob_ref);
                }
            }
        }
        Ok(content)
    }

    async fn fs_write(&self, args: FsWriteArgs) -> Result<FsRevision, ApiError> {
        self.write_gated(args).await
    }

    async fn fs_write_from_blob(&self, args: FsWriteFromBlobArgs) -> Result<FsRevision, ApiError> {
        let FsWriteFromBlobArgs {
            root,
            path,
            hash,
            base_revision,
            force,
        } = args;
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_write_from_blob".into()))?;
        let bytes = blobs
            .get(&hash, None)
            .await
            .map_err(|e| ApiError::Other(format!("blob fetch: {e}")))?;
        self.write_gated(FsWriteArgs {
            root,
            path,
            bytes,
            base_revision,
            force,
        })
        .await
    }

    async fn blob_put(&self, bytes: Vec<u8>) -> Result<BlobRef, ApiError> {
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("blob_put".into()))?;
        blobs
            .put(&bytes)
            .await
            .map_err(|e| ApiError::Other(format!("blob put: {e}")))
    }

    async fn blob_get(
        &self,
        hash: ContentHash,
        range: Option<ByteRange>,
    ) -> Result<Vec<u8>, ApiError> {
        let blobs = self
            .blobs
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("blob_get".into()))?;
        blobs
            .get(&hash, range)
            .await
            .map_err(|e| ApiError::Other(format!("blob get: {e}")))
    }

    async fn blob_stat(&self, hash: ContentHash) -> BlobStat {
        match &self.blobs {
            Some(blobs) => match blobs.stat(&hash).await {
                Some(size) => BlobStat {
                    size,
                    present: true,
                },
                None => BlobStat {
                    size: 0,
                    present: false,
                },
            },
            None => BlobStat {
                size: 0,
                present: false,
            },
        }
    }

    async fn fs_search(
        &self,
        root: FsRootId,
        query: FsSearchQuery,
    ) -> Result<FsSearchPage, ApiError> {
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_search".into()))?;
        ws.search(&root, &query).await
    }

    async fn fs_watch_after(&self, args: FsWatchAfterArgs) -> Result<FsWatchPageView, ApiError> {
        let FsWatchAfterArgs {
            root,
            dir,
            after_seq,
            max,
        } = args;
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_watch_after".into()))?;
        ws.watch_after(&root, &dir, after_seq, max).await
    }

    async fn rewind(
        &self,
        session: SessionId,
        point: daemon_api::RewindPoint,
    ) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may rewind a session.
        let auth = self.require_session_access(&session, true).await?;
        // The unified rewind (conversation-rewind spec): truncate the transcript at `point.anchor`
        // and, when `point.restore_workspace`, roll the workspace back to the matching checkpoint —
        // sealing the journal on the way out. A resident session rewinds its in-process engine
        // directly through the shared seal+rollback seam that the live `RewindTo` command and the
        // managed/fleet engine path also call (so all three stay consistent). A resident foreign
        // (ACP) session is refused inside `rewind_resident` (not rewindable).
        if self.live.is_resident(&session) {
            return self
                .live
                .rewind_resident(&auth, point.anchor, point.restore_workspace)
                .await;
        }
        // A durable (non-resident) session has no live engine to truncate: its transcript is the
        // sealed journal, and rewinding it means re-incarnating the engine to truncate-and-reseal.
        // That activation-driven path is deferred (the checkpoint-ledger extension it needs is out of
        // scope this phase); surface it explicitly rather than silently no-op.
        Err(ApiError::Unsupported(
            "rewind of a non-resident durable session (re-incarnation path deferred)".into(),
        ))
    }

    async fn command_list(&self) -> Vec<CommandSpec> {
        match self.commands.load().as_ref() {
            Some(reg) => reg.specs(),
            None => Vec::new(),
        }
    }

    async fn command_invoke(
        &self,
        invocation: CommandInvocation,
    ) -> Result<CommandOutput, ApiError> {
        let reg = self.commands.load_full();
        let reg = reg
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("command_invoke".into()))?;
        let entry = reg
            .resolve(&invocation.name)
            .ok_or_else(|| ApiError::Other(format!("unknown command: {}", invocation.name)))?;
        // Access gate (the `slash_access.py` analog): the caller's tier vs the command's `min_access`,
        // with the read-only `User` floor always allowed.
        let caller = crate::commands::caller_access(invocation.origin.as_ref());
        if !crate::commands::access_allows(entry.spec.min_access, caller) {
            return Err(ApiError::Other(format!(
                "command /{} requires operator (admin) access",
                entry.spec.name
            )));
        }
        // Session-scoped commands need a session to act on.
        if entry.spec.scope == CommandScope::Session && invocation.session.is_none() {
            return Err(ApiError::Other(format!(
                "command /{} requires an active session",
                entry.spec.name
            )));
        }
        match &entry.owner {
            crate::commands::Owner::Builtin(builtin) => {
                self.run_builtin(*builtin, &invocation).await
            }
            crate::commands::Owner::Provider(provider) => {
                let core_inv = daemon_core::CommandInvocation {
                    name: invocation.name.clone(),
                    args: invocation.args.clone(),
                    session: invocation.session.clone(),
                };
                let cx = match &invocation.session {
                    Some(s) => daemon_core::CommandCx::session(s.clone()),
                    None => daemon_core::CommandCx::node(),
                };
                let out = provider
                    .run_command(&core_inv, &cx)
                    .await
                    .map_err(command_err_to_api)?;
                Ok(CommandOutput {
                    text: out.text,
                    ephemeral: out.ephemeral,
                })
            }
        }
    }
}

impl NodeApiImpl {
    /// The gated workspace write shared by `fs_write` and `fs_write_from_blob`: `Workspace`/`Session`
    /// roots only, sensitive-path + per-session `Deny` gate (overridable by `force`), a pre-mutation
    /// checkpoint for session roots, and the `Conflict`-on-stale-`base_revision` guard inside
    /// `WorkspaceFs::write`.
    async fn write_gated(&self, args: FsWriteArgs) -> Result<FsRevision, ApiError> {
        let FsWriteArgs {
            root,
            path,
            bytes,
            base_revision,
            force,
        } = args;
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_write".into()))?;
        // Host browse roots are read-only.
        if !ws.writable(&root)? {
            return Err(ApiError::Unsupported(
                "host browse roots are read-only".into(),
            ));
        }
        // Sensitive-path gate (the same `.git`/`.ssh`/dotenv/keys rule the agent fs tool uses);
        // `force` overrides. The operator *is* the human, so this never routes through a host ask.
        if !force && is_sensitive_path(&path) {
            return Err(ApiError::Other(format!(
                "sensitive path {path:?} blocked; set force to override"
            )));
        }
        if let FsRootId::Session(sid) = &root {
            self.deny_gate_session(sid, force)?;
            self.capture_pre_write(sid, &path, ws).await;
        }
        ws.write(&root, &path, &bytes, base_revision).await
    }

    /// A `Deny`-mode session blocks operator writes too, unless `force`d.
    fn deny_gate_session(&self, sid: &SessionId, force: bool) -> Result<(), ApiError> {
        if force {
            return Ok(());
        }
        let Some(policy) = self.session_modes.get(sid) else {
            return Ok(());
        };
        if *policy == ApprovalPolicy::Deny {
            return Err(ApiError::Other(format!(
                "session {} is in deny mode; set force to override",
                sid.as_str()
            )));
        }
        Ok(())
    }

    /// Capture a checkpoint before mutating a session root, so an operator edit is rewindable like an
    /// agent edit (best-effort; a capture failure never blocks the write). No-op when no checkpoint
    /// store is wired.
    async fn capture_pre_write(
        &self,
        sid: &SessionId,
        path: &str,
        ws: &Arc<crate::workspace_fs::WorkspaceFs>,
    ) {
        let Some(store) = &self.checkpoints else {
            return;
        };
        let env = LocalEnvironment::new(ws.roots().session_root(sid.as_str()));
        let call_id = format!("operator-fs-write:{path}");
        let _ = store
            .capture(sid.as_str(), &call_id, "operator_fs_write", &env)
            .await;
    }
}
