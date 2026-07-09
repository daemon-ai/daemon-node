// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;

#[async_trait]
impl ControlApi for NodeApiImpl {
    async fn events_page(&self, cursor: u64, max: u32) -> EventsPage {
        let Some(feed) = &self.node_events else {
            return EventsPage::default();
        };
        let page = feed.page(cursor, max);
        // Auth 4 (F4): scope the node-wide feed to the request principal — a non-owner must not learn
        // another owner's session advanced / changed / is awaiting approval. An operator
        // (SessionSeeAll) reads the whole feed; fail-closed on a missing principal.
        let principal = current_principal();
        if principal
            .as_ref()
            .is_some_and(|p| p.has(daemon_auth::Capability::SessionSeeAll))
        {
            return page;
        }
        self.scope_events_page(page, &principal).await
    }

    async fn events_subscribe(&self, cursor: u64) -> Result<NodeEventStream, ApiError> {
        let Some(feed) = &self.node_events else {
            return Ok(stream::empty().boxed());
        };
        let raw = feed.subscribe(cursor);
        // Auth 4 (F4): capture the subscriber's principal AT SUBSCRIBE TIME — the returned long-lived
        // stream is polled outside this request's task-local scope (the same rule `tree_subscribe`
        // notes), so `current_principal()` would be `None` there. An operator (SessionSeeAll) gets
        // the raw feed; any other subscriber has each page owner-scoped so no foreign-session event
        // ever rides through.
        let principal = current_principal();
        if principal
            .as_ref()
            .is_some_and(|p| p.has(daemon_auth::Capability::SessionSeeAll))
        {
            return Ok(raw);
        }
        let this = self.clone();
        Ok(raw
            .then(move |page| {
                let this = this.clone();
                let principal = principal.clone();
                async move { this.scope_events_page(page, &principal).await }
            })
            .boxed())
    }

    async fn health(&self) -> HealthReport {
        let mut services: Vec<ServiceHealth> = self
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
        // Node-managed backend resources (the gateway + local inference) report alongside the
        // resident services. Clone the handles out from under the lock first — a `ManagedResource`
        // health probe is async and must not be polled while holding a std mutex.
        let managed: Vec<Arc<dyn crate::managed::ManagedResource>> =
            self.managed.lock().unwrap().clone();
        for resource in managed {
            services.push(resource.health().await);
        }
        // `all_ok` folds the resident supervisor's health with every managed resource's `ok` bit.
        let all_ok = self.supervisor.all_ok() && services.iter().all(|s| s.ok);
        HealthReport { all_ok, services }
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
        // Denormalize the backend from the bound profile so a detail pane renders it without a
        // second `profile_get`: the engine selector and (for a foreign engine) its model backend.
        // A session with no bound profile / a node without profile management is the native default.
        let (engine, foreign_backend) = match (&self.profiles, meta.bound_profile.as_ref()) {
            (Some(store), Some(bound)) => store
                .get(bound.as_str())
                .ok()
                .flatten()
                .map(|spec| (spec.engine, spec.foreign_backend))
                .unwrap_or_else(|| {
                    (
                        daemon_api::EngineSelector::Core,
                        daemon_api::ForeignBackend::default(),
                    )
                }),
            _ => (
                daemon_api::EngineSelector::Core,
                daemon_api::ForeignBackend::default(),
            ),
        };
        // Phase 3: a resident foreign session's live `Model` selector (choices + current), so a
        // detail pane can render a foreign model picker without a side channel.
        let model_selector = self.live.model_selector(&session);
        Some(SessionDetail {
            info,
            overlay,
            model,
            delivery_targets,
            children,
            checkpoints,
            engine,
            foreign_backend,
            model_selector,
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
        let raw = self.store.pending_approvals_of(session.as_ref()).await;
        // Auth 4 (read-of-one/many): scope to the request principal — a peer must not read another
        // owner's parked-approval prompts/paths (own sessions only unless `SessionSeeAll`). Owner is
        // read per row; the pending set is small/capped. Mirrors `session_search`'s per-hit filter.
        let principal = current_principal();
        let mut approvals: Vec<ApprovalInfo> = Vec::with_capacity(raw.len());
        for p in raw {
            let owner = self
                .store
                .session_meta(&p.session_id)
                .await
                .and_then(|m| m.owner);
            if !owner_visible(&principal, &owner) {
                continue;
            }
            // wire v30 (item 7): attach a node-computed structured detail for fs/edit approvals — a
            // `tool-detail` with kind "fs.diff" and JSON body `{path, diff}`. The diff is sourced
            // from the engine's node-computed approval prompt, which for a path-bearing (fs edit)
            // gate IS the diff summary the operator is asked to approve (the proposed edit content
            // is carried to daemon-host on the durable parked row, so the compute happens here at
            // park time — decision F). Command approvals (no path) carry no diff detail.
            let detail = p.path.as_ref().map(|path| {
                let body = serde_json::to_vec(&serde_json::json!({
                    "path": path,
                    "diff": p.prompt,
                }))
                .unwrap_or_default();
                daemon_protocol::ToolDetail::new("fs.diff", body)
            });
            approvals.push(ApprovalInfo {
                session: p.session_id,
                request_id: p.job_id.as_str().to_string(),
                prompt: p.prompt,
                path: p.path,
                // wire v28: surface the stamped command fingerprint structurally (was only inside
                // `prompt`). Enforcement stays snapshot-side; this is display/correlation only.
                fingerprint: p.fingerprint,
                detail,
            });
        }
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
        allow_permanent: bool,
        reason: Option<String>,
    ) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may decide a session's approval.
        self.require_session_access(&session, true).await?;
        // Record the decision + enqueue the wake durably (one transaction in the store), then nudge
        // the activation manager so the dormant session rehydrates promptly and resolves the gated
        // tool call (allow -> runs it; deny -> injects a tool error). `allow_permanent` rides the
        // completion payload so the engine's `resolve_approvals` remembers the verified fingerprint;
        // a deny `reason` rides it too, becoming the injected tool error's content (wire v29).
        // Idempotent in the store.
        let answered = self
            .store
            .answer_approval(
                &session,
                &JobId::new(request_id.clone()),
                allow,
                allow_permanent,
                reason,
            )
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

    async fn caps(&self) -> daemon_api::CapsReport {
        self.caps
    }

    async fn gateway_get(&self) -> Result<GatewayStatus, ApiError> {
        // Forward to the injected gateway control seam (the resident gateway resource owns the
        // store-backed override + live listener). Clone the handle out from under the lock so the
        // async `get` is not polled while holding a std mutex.
        let gateway = self.gateway.lock().unwrap().clone();
        match gateway {
            Some(gw) => Ok(gw.get().await),
            None => Err(ApiError::Unsupported("gateway_get".into())),
        }
    }

    async fn gateway_set(
        &self,
        enabled: bool,
        addr: Option<String>,
    ) -> Result<GatewayStatus, ApiError> {
        let gateway = self.gateway.lock().unwrap().clone();
        match gateway {
            Some(gw) => gw.set(enabled, addr).await,
            None => Err(ApiError::Unsupported("gateway_set".into())),
        }
    }

    async fn tool_list(&self) -> Vec<daemon_api::ToolInfo> {
        // The node-wide inventory the binary late-bound (it owns the tool build gates), overlaid
        // with the durable `ToolSetEnabled` overrides (wire v30, item 6). Bounded at the wire page
        // cap like every other unpaged list.
        let mut tools: Vec<daemon_api::ToolInfo> = match self.tools_inventory.load_full() {
            Some(tools) => tools
                .iter()
                .take(daemon_api::WIRE_PAGE_MAX)
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        for (name, enabled) in self.store.tool_overrides().await {
            if let Some(row) = tools.iter_mut().find(|t| t.name == name) {
                if !enabled {
                    // Force-disable is always honored.
                    row.enabled = false;
                } else if row.requires.is_none() {
                    // Force-enable re-enables a policy-disabled tool but can never conjure one
                    // missing its build feature (a `requires` row stays disabled — decision E).
                    row.enabled = true;
                }
            }
        }
        tools
    }

    async fn tool_set_enabled(&self, tool: String, enabled: bool) -> Result<(), ApiError> {
        // Persist the node-wide override (wire v30, item 6). `tool_list` overlays it and per-session
        // tool wiring consults it, so a disabled tool disappears from new turns.
        self.store
            .set_tool_override(&tool, enabled)
            .await
            .map_err(|e| ApiError::Other(format!("set tool override: {e}")))
    }

    async fn feedback_submit(&self, args: FeedbackSubmitArgs) -> Result<FeedbackAck, ApiError> {
        let FeedbackSubmitArgs {
            kind,
            target,
            rating,
            comment,
            include_content,
            diagnostics,
            surface,
        } = args;

        // -- server-side validation (the node is the enforcement point; client checks are UX sugar) --
        if let Some(comment) = &comment {
            if comment.len() > daemon_api::FEEDBACK_COMMENT_MAX {
                return Err(ApiError::Other(format!(
                    "feedback comment exceeds {} bytes",
                    daemon_api::FEEDBACK_COMMENT_MAX
                )));
            }
        }
        match kind {
            // Response feedback rates a specific turn: it needs a target AND a rating.
            FeedbackKind::Response => {
                let Some(target) = &target else {
                    return Err(ApiError::Other(
                        "response feedback requires a target".into(),
                    ));
                };
                if rating.is_none() {
                    return Err(ApiError::Other(
                        "response feedback requires a rating".into(),
                    ));
                }
                // The target session must exist (durable or live).
                if self.store.status(&target.session).await.is_none() {
                    return Err(ApiError::UnknownSession(target.session.to_string()));
                }
            }
            // App feedback is free-form: it needs at least a comment or a rating.
            FeedbackKind::App => {
                if comment.is_none() && rating.is_none() {
                    return Err(ApiError::Other(
                        "app feedback requires a comment or a rating".into(),
                    ));
                }
            }
        }

        // Consent provenance: explicit feedback is per-event consent (accepted+queued regardless),
        // but we record WHETHER the global telemetry toggle was on at submit time so the exporter
        // can distinguish opted-in telemetry from a one-shot explicit grant.
        let consent = if self.store.telemetry_consent_get().await {
            "opted-in"
        } else {
            "explicit-one-shot"
        };
        let created_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Best-effort embodiment (the "disembodied thumbs" fix): attach the rated turn's model and,
        // when the submitter consented via `include_content`, the rated response text, so the
        // exported event is self-describing rather than a bare `(session, cursor)` anchor. Both are
        // read from the durable journal at submit time (frozen as-rated; the drain stays a pure
        // record->event map). Per-turn end_reason/usage are currently only journaled as a
        // `mgmt.turn_finished` debug string, so they are left None here (structured per-turn summary
        // is a follow-up); the record + `FeedbackEvent` keep the slots.
        let (turn_model, response_content) = match (kind, target.as_ref()) {
            (FeedbackKind::Response, Some(t)) => {
                let model = self.session_models.get(&t.session).map(|m| m.clone());
                let content = if include_content {
                    self.rated_response_text(&t.session, t.cursor).await
                } else {
                    None
                };
                (model, content)
            }
            _ => (None, None),
        };

        let record = FeedbackRecord {
            id: mint_feedback_id(),
            created_at_ms,
            kind: match kind {
                FeedbackKind::Response => "response".into(),
                FeedbackKind::App => "app".into(),
            },
            rating: rating.map(|r| match r {
                FeedbackRating::Up => "up".into(),
                FeedbackRating::Down => "down".into(),
            }),
            comment,
            include_content,
            session: target.as_ref().map(|t| t.session.to_string()),
            cursor: target.as_ref().map(|t| t.cursor),
            trace: target.as_ref().and_then(|t| t.trace).map(|t| t.0),
            surface,
            app_version: diagnostics.as_ref().and_then(|d| d.app_version.clone()),
            os: diagnostics.and_then(|d| d.os),
            consent: consent.into(),
            node_version: daemon_common::VERSION.to_string(),
            model: turn_model,
            provider: None,
            end_reason: None,
            input_tokens: None,
            output_tokens: None,
            response_content,
            delivered: false,
        };

        self.store
            .feedback_enqueue(record)
            .await
            .map_err(|e| ApiError::Other(format!("enqueue feedback: {e}")))?;
        // Trigger a best-effort, detached drain of the outbox to the OTLP exporter (N1 → N2). This
        // does NOT block the ack, and is a no-op when export is inert (no `telemetry.feedback_endpoint`
        // or the `otel` feature is off) — the record then just stays queued.
        self.spawn_feedback_drain();
        // The ack means accepted+queued to the durable outbox. Delivery to the collector is the
        // separate best-effort drain above; a queued record survives a failed/absent export.
        Ok(FeedbackAck {
            accepted: true,
            queued: true,
        })
    }

    async fn telemetry_consent_get(&self) -> Result<bool, ApiError> {
        Ok(self.store.telemetry_consent_get().await)
    }

    async fn telemetry_consent_set(&self, enabled: bool) -> Result<bool, ApiError> {
        self.store
            .telemetry_consent_set(enabled)
            .await
            .map_err(|e| ApiError::Other(format!("set telemetry consent: {e}")))?;
        Ok(enabled)
    }

    async fn fingerprint_list(
        &self,
        session: SessionId,
    ) -> Result<Vec<daemon_api::RememberedFingerprint>, ApiError> {
        // Auth 4: owner-or-`SessionSeeAll` may read a session's remembered approvals.
        self.require_session_access(&session, false).await?;
        // The allow-list of a LIVE-resident session lives in the resident engine's memory (it is
        // ephemeral — it dies with the residency and is never persisted); the durable snapshot is
        // the only managed storage. Refuse rather than mislead with an empty durable read.
        if self.live.resident_is_foreign(&session).is_some() {
            return Err(ApiError::Unsupported(
                "fingerprint management targets durable sessions; this session is live-resident \
                 (its allow-list is in-memory and resets at session close)"
                    .into(),
            ));
        }
        let Some(blob) = self.store.peek_snapshot(&session).await else {
            return Ok(Vec::new());
        };
        let snap = Snapshot::decode(&blob)
            .map_err(|e| ApiError::Other(format!("decode session snapshot: {e}")))?;
        Ok(snap
            .session_allow_fingerprints
            .iter()
            .take(daemon_api::WIRE_PAGE_MAX)
            .map(|r| daemon_api::RememberedFingerprint {
                fingerprint: r.fingerprint.as_str().to_string(),
                // Provenance (wire v30): the label + capture timestamp recorded at the decide path.
                label: r.label.clone(),
                remembered_at_ms: r.remembered_at_ms,
            })
            .collect())
    }

    async fn fingerprint_revoke(
        &self,
        session: SessionId,
        fingerprint: String,
    ) -> Result<(), ApiError> {
        // Auth 4: only the owner (or a `SessionControlAny` operator) may edit the allow-list.
        self.require_session_access(&session, true).await?;
        if self.live.resident_is_foreign(&session).is_some() {
            return Err(ApiError::Unsupported(
                "fingerprint management targets durable sessions; this session is live-resident \
                 (its allow-list is in-memory and resets at session close)"
                    .into(),
            ));
        }
        // Read-modify-write of the DORMANT durable snapshot under the store's compare-and-swap:
        // an Active session (or a concurrent snapshot writer) refuses instead of losing the edit
        // to the incarnation's next checkpoint. The revoke takes effect at the next activation,
        // which reseeds the engine's allow-list view from this snapshot.
        let Some(blob) = self.store.peek_snapshot(&session).await else {
            return Err(ApiError::Other(format!(
                "no durable session {session} to revoke a fingerprint on"
            )));
        };
        let mut snap = Snapshot::decode(&blob)
            .map_err(|e| ApiError::Other(format!("decode session snapshot: {e}")))?;
        let before = snap.session_allow_fingerprints.len();
        snap.session_allow_fingerprints
            .retain(|r| r.fingerprint.as_str() != fingerprint);
        if snap.session_allow_fingerprints.len() == before {
            return Err(ApiError::Other(format!(
                "no remembered fingerprint {fingerprint} on session {session}"
            )));
        }
        let new = snap
            .encode()
            .map_err(|e| ApiError::Other(format!("encode session snapshot: {e}")))?;
        let swapped = self
            .store
            .swap_snapshot_if_dormant(&session, &blob, new)
            .await
            .map_err(|e| ApiError::Other(format!("swap snapshot: {e}")))?;
        if !swapped {
            return Err(ApiError::Other(format!(
                "session {session} is running (or its state changed concurrently) — retry the \
                 revoke when it is dormant"
            )));
        }
        Ok(())
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
        let Some(fleet) = &self.fleet else {
            return FleetReport::default();
        };
        let full = fleet.report().await;
        // Auth 4 (F3): an operator (SessionSeeAll) sees the whole fleet unchanged; any other
        // principal sees only the units it owns (UnitId -> session -> owner), with usage folded over
        // just those so the total never sums another owner's work. Fail-closed on a missing
        // principal (owner_visible denies None).
        let principal = current_principal();
        if principal
            .as_ref()
            .is_some_and(|p| p.has(daemon_auth::Capability::SessionSeeAll))
        {
            return full;
        }
        let mut children = Vec::new();
        let mut usage = daemon_common::UsageDelta::default();
        for id in full.children {
            let node = fleet.unit(&id).await;
            let owner = match &node {
                Some(n) => match &n.session {
                    Some(s) => self.store.session_meta(s).await.and_then(|m| m.owner),
                    None => None,
                },
                None => None,
            };
            if owner_visible(&principal, &owner) {
                if let Some(n) = &node {
                    usage.add(&n.usage);
                }
                children.push(id);
            }
        }
        FleetReport { children, usage }
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
        let node = match &self.fleet {
            Some(fleet) => fleet.unit(&id).await?,
            None => return None,
        };
        // Auth 4 (F3): a non-owner must not resolve another owner's unit (own units only unless
        // SessionSeeAll). Owner resolved from the fetched node's backing session; deny -> None.
        let owner = match &node.session {
            Some(s) => self.store.session_meta(s).await.and_then(|m| m.owner),
            None => None,
        };
        if owner_visible(&current_principal(), &owner) {
            Some(node)
        } else {
            None
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
        //
        // Wire v33: enrich each row centrally with the per-verb ops descriptors by probing the
        // adapter's `MessagingProtocol` feature-trait accessors and calling each trait's
        // `supported()` — the same `messaging()->feature()` path the per-op helpers
        // (`conversations_for`/`membership_for`/… in node_api/messaging.rs) walk, but done once here
        // so every adapter gets it for free (zero per-adapter duplication). `None` on a field means
        // the adapter does not implement that feature trait at all.
        self.adapters
            .load_full()
            .adapters()
            .iter()
            .map(|adapter| {
                let mut info = adapter.info();
                if let Some(messaging) = adapter.clone().messaging() {
                    info.conversation_ops =
                        messaging.clone().conversations().map(|c| c.supported());
                    info.membership_ops = messaging.clone().membership().map(|m| m.supported());
                    info.contacts_ops = messaging.clone().contacts().map(|c| c.supported());
                    info.roster_ops = messaging.clone().roster().map(|r| r.supported());
                    // Directory has no per-verb ops struct — presence of the trait AND its own
                    // `supported()` probe collapse to a single bool (absent trait => false).
                    info.directory = messaging
                        .clone()
                        .directory()
                        .map(|d| d.supported())
                        .unwrap_or(false);
                }
                info
            })
            .collect()
    }

    async fn transport_instances(&self) -> Vec<TransportInstanceInfo> {
        // Live per-account connection/presence, aggregated across every registered adapter, then
        // overlaid with the node-owned desired state (wire v35): the persisted `enabled` flag and
        // human `label`. The node is authoritative — adapters report the inert defaults
        // (`enabled=true`, `label=None`), and the store's row (when present) wins.
        let mut instances = self.adapters.load_full().instances().await;
        let prefs: std::collections::HashMap<String, (bool, Option<String>)> = self
            .store
            .transport_prefs()
            .await
            .into_iter()
            .map(|p| (p.transport, (p.enabled, p.label)))
            .collect();
        for info in &mut instances {
            if let Some((enabled, label)) = prefs.get(info.transport.as_str()) {
                info.enabled = *enabled;
                info.label = label.clone();
            }
        }
        instances
    }

    async fn transport_disconnect(&self, transport: TransportId) -> Result<(), ApiError> {
        // Reversible (wire v30, item 1): abort the owning adapter's supervised serve loop and mark
        // the instance Offline, KEEPING its credential/config/bound_profile. The serve loop is
        // per-adapter (the coarsest per-instance granularity this architecture supports), so we key
        // the handle by family. A later re-`spawn_adapters` (or restart) resumes it.
        let family = self
            .adapters
            .load_full()
            .adapter_for_transport(&transport)
            .map(|a| a.family().to_string())
            .ok_or_else(|| {
                ApiError::Other(format!("no adapter owns transport {}", transport.as_str()))
            })?;
        if let Some(handle) = self.adapter_handles.lock().unwrap().remove(&family) {
            handle.abort();
        }
        // A user-requested disconnect is transient (not fatal): a reconnect can resume it.
        self.disconnect_fatal.insert(transport.clone(), false);
        if let Some(feed) = self.node_feed() {
            feed.emit(daemon_api::NodeEvent::TransportChanged {
                transport,
                connection: daemon_api::ConnectionState::Offline,
                presence: daemon_api::PresenceState::Offline,
                reason: Some(daemon_api::DisconnectReason::UserRequested),
                message: None,
                fatal: false,
            });
        }
        Ok(())
    }

    async fn transport_connect(&self, transport: TransportId) -> Result<(), ApiError> {
        // Reversible reconnect (wire v35, item 1): re-spawn the owning adapter FAMILY's supervised
        // serve loop. The serve loop is per-family (the coarsest granularity), matching
        // `transport_disconnect`'s abort granularity. Error if no adapter owns the transport.
        let family = self
            .adapters
            .load_full()
            .adapter_for_transport(&transport)
            .map(|a| a.family().to_string())
            .ok_or_else(|| {
                ApiError::Other(format!("no adapter owns transport {}", transport.as_str()))
            })?;
        // Clear any fatal marker so the supervisor does not immediately short-circuit (defensive;
        // `supervise_adapter` also clears it for each instance at serve start).
        self.disconnect_fatal.remove(&transport);
        // We need an owned `Arc<Self>` to spawn the serve loop; recover it from the weak handle
        // captured by `spawn_adapters` at boot. Absent => adapters were never spawned on this node,
        // so there is nothing to resume.
        let node = self
            .self_weak
            .get()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| ApiError::Unsupported("transport_connect".into()))?;
        // Idempotent: `spawn_adapter_family` is a no-op when the family's serve loop is already
        // running, or when every instance of the family is disabled in the store (honoring the
        // persisted desired state). On a real (re)spawn the loop emits its own serve-start
        // `TransportChanged`, so we do not emit here (avoid double-emitting).
        node.spawn_adapter_family(&family).await;
        Ok(())
    }

    async fn transport_set_enabled(
        &self,
        transport: TransportId,
        enabled: bool,
    ) -> Result<(), ApiError> {
        // Persist the operator's desired state (wire v35, item 2), then reconcile the live serve
        // loop. Family-vs-instance mismatch (documented): the serve loop is per-family, so
        // `enabled=false` disconnects the WHOLE owning family; a sibling instance that remains
        // enabled is restored on the next `transport_connect`/boot (the family re-serves because
        // not all its instances are disabled). In the common one-instance-per-family case this is
        // exactly "disconnect and stay disconnected". `enabled=true` persists then reconnects.
        self.store
            .set_transport_enabled(transport.as_str(), enabled)
            .await
            .map_err(|e| ApiError::Other(format!("set_transport_enabled: {e}")))?;
        if enabled {
            // Best-effort: attempt to (re)connect. A transport with no owning adapter simply has
            // nothing to spawn — the persisted desire still stands for a future boot.
            let _ = self.transport_connect(transport).await;
        } else {
            // Disconnect now (reuse the reversible v30 op; it emits the Offline `TransportChanged`).
            self.transport_disconnect(transport).await?;
        }
        Ok(())
    }

    async fn transport_set_label(
        &self,
        transport: TransportId,
        label: Option<String>,
    ) -> Result<(), ApiError> {
        // Persist the human label (wire v35, item 3); it is overlaid onto `TransportInstanceInfo`
        // in `transport_instances()`.
        self.store
            .set_transport_label(transport.as_str(), label)
            .await
            .map_err(|e| ApiError::Other(format!("set_transport_label: {e}")))?;
        // Nudge clients to refetch the instance list via the existing per-transport
        // `TransportChanged` pointer, carrying the instance's CURRENT live connection/presence (so
        // no spurious status flip — the label rides `TransportInstanceInfo`, refetched on the same
        // event). Reuses the event the app already handles per transport instead of inventing a new
        // one. Best-effort: skipped when the instance is not currently reported.
        if let Some(feed) = self.node_feed() {
            if let Some(cur) = self
                .transport_instances()
                .await
                .into_iter()
                .find(|i| i.transport == transport)
            {
                feed.emit(daemon_api::NodeEvent::TransportChanged {
                    transport,
                    connection: cur.connection,
                    presence: cur.presence,
                    reason: cur.reason,
                    message: cur.message,
                    fatal: cur.fatal,
                });
            }
        }
        Ok(())
    }

    async fn transport_remove(&self, transport: TransportId) -> Result<(), ApiError> {
        // Remove implies disconnect, then ONE node-side teardown (wire v30, item 1): the client
        // issues a single intent; the node sequences the steps. (1) disconnect, (2) leave every
        // conversation the instance owns, (3) unbind its routing pins, (4) drop its credential.
        self.transport_disconnect(transport.clone()).await?;
        // (2) close conversations (best-effort; adapters that do not support leave are skipped).
        let mut after: Option<String> = None;
        loop {
            let page = self.conv_list(transport.clone(), after.take()).await;
            for conv in &page.items {
                let _ = self.conv_leave(transport.clone(), conv.id.clone()).await;
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        // (3) unbind routing pins whose origin belongs to this transport instance.
        for route in self.store.routing_list().await {
            if let Some(wire) = super::routing::wire_route_from_store(&route) {
                if super::routing::transport_family_matches(&wire.origin.transport, &transport) {
                    let _ = self.store.routing_remove(&route.key).await;
                }
            }
        }
        self.load_routing_pins().await;
        // (4) drop the instance's credential (best-effort; keyed by the instance-qualified id).
        let _ = self.credential_remove(transport.as_str().to_string()).await;
        Ok(())
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

    async fn roster_list(
        &self,
        transport: TransportId,
        after: Option<String>,
    ) -> daemon_api::WirePage<ContactInfo> {
        // The adapter returns the unbounded, adapter-ordered roster; sort + page here (once) rather
        // than teaching every `SupportsRoster` impl the cursor. The cursor is the contact id
        // (mirrors `conv_list`).
        let mut contacts = match self.roster_for(&transport) {
            Ok(r) => r.list(transport).await,
            Err(_) => Vec::new(),
        };
        contacts.sort_by(|a, b| a.id.cmp(&b.id));
        daemon_api::paginate(contacts, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |c| {
            c.id.clone()
        })
    }

    async fn notification_list(&self) -> Vec<daemon_api::NotificationInfo> {
        // The node-authoritative notification list (wire vNEXT), newest first — a snapshot of the
        // node's `NotificationManager` (ported from libpurple's `PurpleNotificationManager`).
        // Clients re-list on a `NotificationsChanged` pointer.
        self.notifications_snapshot()
    }

    async fn person_list(&self) -> Vec<daemon_api::Person> {
        // The node-authoritative person/metacontact registry (wire vNEXT), insertion order — a
        // snapshot of the node's `PersonManager` (ported from the person half of libpurple's
        // `PurpleContactManager`). Clients re-list on a `PersonsChanged` pointer.
        self.persons_snapshot()
    }

    async fn roster_add(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} contact={}", transport.as_str(), contact.id);
        let res = self
            .audited(
                "mgmt.roster.add",
                detail,
                self.roster_for(&transport)?.add(transport.clone(), contact),
            )
            .await;
        if res.is_ok() {
            self.emit_contacts_changed(transport);
        }
        res
    }

    async fn roster_update(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} contact={}", transport.as_str(), contact.id);
        let res = self
            .audited(
                "mgmt.roster.update",
                detail,
                self.roster_for(&transport)?
                    .update(transport.clone(), contact),
            )
            .await;
        if res.is_ok() {
            self.emit_contacts_changed(transport);
        }
        res
    }

    async fn roster_remove(
        &self,
        transport: TransportId,
        contact: ContactInfo,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} contact={}", transport.as_str(), contact.id);
        let res = self
            .audited(
                "mgmt.roster.remove",
                detail,
                self.roster_for(&transport)?
                    .remove(transport.clone(), contact),
            )
            .await;
        if res.is_ok() {
            self.emit_contacts_changed(transport);
        }
        res
    }

    async fn ft_send(
        &self,
        transport: TransportId,
        transfer: daemon_api::FileTransfer,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} name={}", transport.as_str(), transfer.name);
        self.audited(
            "mgmt.ft.send",
            detail,
            self.file_transfer_for(&transport)?
                .send(transport.clone(), transfer),
        )
        .await
    }

    async fn ft_receive(
        &self,
        transport: TransportId,
        transfer: daemon_api::FileTransfer,
    ) -> Result<(), ApiError> {
        let detail = format!("transport={} name={}", transport.as_str(), transfer.name);
        self.audited(
            "mgmt.ft.receive",
            detail,
            self.file_transfer_for(&transport)?
                .receive(transport.clone(), transfer),
        )
        .await
    }

    async fn agent_discover(&self) -> Vec<AgentEntry> {
        // Probe the curated direct-binary recipe table via the injected discovery hook (the binary
        // owns the ACP runtime). Cache the results so `agent_catalog` surfaces them without
        // re-probing, then return the merged catalog (discovery results + durable manual
        // registrations).
        if let Some(agents) = &self.agents {
            let discovered = agents.discover().await;
            *self.last_agents.write().unwrap() = discovered;
        }
        self.agent_catalog().await
    }

    async fn agent_catalog(&self) -> Vec<AgentEntry> {
        // The durable manual registrations (source = Manual) take precedence over a builtin of the
        // same name; the in-memory last-discovery results fill in the auto-detected builtins.
        // (The store table keeps its historical `acp_*` name — the rows are opaque CBOR of the
        // wire `AgentEntry`, whose added `protocol` field defaults to `Acp` for pre-v29 rows.)
        let mut by_name: std::collections::BTreeMap<String, AgentEntry> =
            std::collections::BTreeMap::new();
        for entry in self.last_agents.read().unwrap().iter() {
            by_name.insert(entry.name.clone(), entry.clone());
        }
        for stored in self.store.acp_list().await {
            if let Ok(mut entry) = from_cbor::<AgentEntry>(&stored.entry) {
                // Re-derive the trust status from the durable installed/protocol/version so a
                // legacy row stored before the wire carried `verification` (decodes as the default)
                // surfaces the node's current verdict — the single node-side derivation.
                entry.refresh_verification();
                by_name.insert(entry.name.clone(), entry);
            }
        }
        by_name.into_values().collect()
    }

    async fn agent_register(&self, mut entry: AgentEntry) -> Result<(), ApiError> {
        // A manual registration: force `source = Manual`, then verify/enrich it via the discovery
        // hook when wired (PATH check for every protocol; the ACP `initialize` handshake fills
        // version/caps for ACP entries only).
        entry.source = AgentSource::Manual;
        if let Some(agents) = &self.agents {
            entry = agents.probe(entry).await;
            entry.source = AgentSource::Manual;
        }
        // Re-derive the trust status from the (probed, if a hook is wired) installed/protocol/version
        // — never trust the caller-supplied value, and cover the no-hook path where `probe` did not
        // run. The single node-side derivation.
        entry.refresh_verification();
        self.store
            .acp_set(daemon_store::AcpEntry {
                name: entry.name.clone(),
                entry: to_cbor(&entry),
            })
            .await
            .map_err(|e| ApiError::Other(format!("agent register: {e}")))
    }

    async fn agent_remove(&self, name: String) -> Result<(), ApiError> {
        self.last_agents.write().unwrap().retain(|e| e.name != name);
        self.store
            .acp_remove(&name)
            .await
            .map_err(|e| ApiError::Other(format!("agent remove: {e}")))
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

    // -- Saved presences (W2-F; wire vNEXT): every op delegates to the shared `PresenceManager`;
    //    absent it, the trait defaults (empty list / `Unsupported`) apply.
    async fn presence_list(&self) -> Vec<daemon_api::SavedPresence> {
        match &self.presences {
            Some(presences) => presences.list().await,
            None => Vec::new(),
        }
    }

    async fn presence_save(&self, presence: daemon_api::SavedPresence) -> Result<(), ApiError> {
        match &self.presences {
            Some(presences) => presences.save(presence).await,
            None => Err(ApiError::Unsupported("presence_save".into())),
        }
    }

    async fn presence_delete(&self, id: String) -> Result<(), ApiError> {
        match &self.presences {
            Some(presences) => presences.remove(&id).await.map(|_| ()),
            None => Err(ApiError::Unsupported("presence_delete".into())),
        }
    }

    async fn presence_set_active(&self, id: String) -> Result<(), ApiError> {
        match &self.presences {
            Some(presences) => presences.set_active(&id).await,
            None => Err(ApiError::Unsupported("presence_set_active".into())),
        }
    }

    async fn unit_events(&self, id: UnitId, max: u32) -> Vec<ManageEventView> {
        // Auth 4 (F3): a non-owner must not read another owner's unit management events (deny ->
        // empty; own units only unless SessionSeeAll).
        if !self.unit_owner_visible(&id).await {
            return Vec::new();
        }
        match &self.fleet {
            Some(fleet) => fleet.unit_events(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_outbound(&self, id: UnitId, max: u32) -> Vec<Outbound> {
        // Auth 4 (F3): gate BEFORE the drain — a non-owner must never consume another owner's
        // outbound buffer (this is a destructive single-consumer drain). Deny -> empty.
        if !self.unit_owner_visible(&id).await {
            return Vec::new();
        }
        match &self.fleet {
            Some(fleet) => fleet.unit_outbound(&id, max).await,
            None => Vec::new(),
        }
    }

    async fn unit_history(&self, id: UnitId, after_cursor: u64, max: u32) -> JournalPageView {
        // Auth 4 (F3): a non-owner must not read another owner's unit history (deny -> empty page).
        if !self.unit_owner_visible(&id).await {
            return JournalPageView::default();
        }
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
        // Auth 4 (read-of-one/many): scope to the request principal — a peer must not read another
        // owner's checkpoint metadata (own sessions only unless `SessionSeeAll`). Owner is read per
        // record; the listed set is small/capped. Mirrors `approvals_pending`'s per-row filter.
        let principal = current_principal();
        let mut checkpoints: Vec<daemon_api::CheckpointInfo> = Vec::new();
        for r in store.list(filter.as_deref()).await {
            let session = SessionId::new(r.session);
            let owner = self
                .store
                .session_meta(&session)
                .await
                .and_then(|m| m.owner);
            if !owner_visible(&principal, &owner) {
                continue;
            }
            checkpoints.push(daemon_api::CheckpointInfo {
                id: r.id,
                session,
                tool: r.tool,
                created_unix: r.created_unix,
                // The turn/cursor correlation is not yet recorded on the checkpoint ledger; the wire
                // fields exist (rewind-unify foundation) and fill in when the ledger carries them.
                turn_ordinal: None,
                cursor: None,
            });
        }
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
        // Opened (live) session sandboxes — Auth 4 (F2): only the caller's own (or, for a
        // `SessionSeeAll` operator, every) live session is advertised, so the enumeration never
        // reveals another owner's session ids/sandboxes.
        let principal = current_principal();
        for sid in self.live.live_ids() {
            let owner = self.store.session_meta(&sid).await.and_then(|m| m.owner);
            if !owner_visible(&principal, &owner) {
                continue;
            }
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
        // Auth 4 (F2): a `Session` root is owner-gated (read).
        self.require_fs_root_access(&root, false).await?;
        let ws = self
            .workspace
            .as_ref()
            .ok_or_else(|| ApiError::Unsupported("fs_list".into()))?;
        ws.list(&root, &dir, show_ignored, after.as_deref()).await
    }

    async fn fs_stat(&self, root: FsRootId, path: String) -> Result<FsEntry, ApiError> {
        // Auth 4 (F2): a `Session` root is owner-gated (read).
        self.require_fs_root_access(&root, false).await?;
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
        // Auth 4 (F2): a `Session` root is owner-gated (read).
        self.require_fs_root_access(&root, false).await?;
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
        // Auth 4 (F2): a `Session` root is that session's sandbox — gate BEFORE the blob fetch so a
        // non-owner is denied `Forbidden` (not a blob/Unsupported error) and `write_gated` re-checks.
        self.require_fs_root_access(&root, true).await?;
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
        // Auth 4 (F2): a `Session` root is owner-gated (read).
        self.require_fs_root_access(&root, false).await?;
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
        // Auth 4 (F2): a `Session` root is owner-gated (read).
        self.require_fs_root_access(&root, false).await?;
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
    /// Auth 4 for the filesystem surface (Cluster A, F2): a [`FsRootId::Session`] root addresses that
    /// session's workspace sandbox, so reading/writing it is a per-session op and must carry the same
    /// ownership proof as the session log/history — own-or-[`SessionSeeAll`](daemon_auth::Capability::SessionSeeAll)
    /// for a read (`control = false`), own-or-[`SessionControlAny`](daemon_auth::Capability::SessionControlAny)
    /// for a write (`control = true`). `Host`/`Workspace` roots are node-level (not per-owner) and
    /// pass through. Checked at each fs handler's entry, BEFORE the workspace unwrap, so a non-owner
    /// is denied `Forbidden` regardless of whether a workspace is wired. The minted token is unused
    /// (the fs op addresses the root directly); only success/denial matters here.
    async fn require_fs_root_access(&self, root: &FsRootId, control: bool) -> Result<(), ApiError> {
        if let FsRootId::Session(sid) = root {
            self.require_session_access(sid, control).await?;
        }
        Ok(())
    }

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
        // Auth 4: a `Session` root is that session's sandbox — a write requires session control.
        self.require_fs_root_access(&root, true).await?;
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

/// Mint a fresh feedback id: `fb-<32 hex>` from 16 random bytes (mirrors `mint_session_id`). A
/// getrandom failure is astronomically unlikely; fall back to a time-seeded id rather than panicking.
fn mint_feedback_id() -> String {
    let mut bytes = [0u8; 16];
    if getrandom::getrandom(&mut bytes).is_err() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        bytes.copy_from_slice(&nanos.to_le_bytes());
    }
    let mut hex = String::with_capacity(3 + bytes.len() * 2);
    hex.push_str("fb-");
    for b in bytes {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}
