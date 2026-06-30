// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Roster / session-meta projection: the unified durable+live session list, per-turn activity
//! stamping, and the tree/roster change notifications pushed onto the fleet bus + L3 event feed.

use super::*;

impl NodeApiImpl {
    /// The node-wide event feed, when wired (cloned out for an emit / `bump_rev` in the §5 hooks that
    /// hang off `NodeApiImpl` directly — roster/meta changes).
    pub(crate) fn node_feed(&self) -> Option<Arc<NodeEventFeed>> {
        self.node_events.clone()
    }

    /// Ping the fleet bus that the roster/tree changed (a rename/pin/archive that no producer models
    /// as a subagent transition). Projects a fresh `tree()` snapshot onto the bus off-thread so live
    /// `tree_subscribe` subscribers refresh promptly; a no-op when no bus is wired or there are no
    /// subscribers (so the projection cost is only paid when someone is watching).
    pub fn emit_tree_changed(&self) {
        if let Some(tx) = &self.fleet_events {
            if tx.receiver_count() == 0 {
                return;
            }
            let this = self.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                let report = this.tree().await;
                let _ = tx.send(daemon_api::TreeEvent::Snapshot(report));
            });
        }
    }

    /// Forward a concrete subagent/delegation lifecycle delta onto the fleet bus. A no-op when no bus
    /// is wired or there are no subscribers.
    pub fn emit_subagent(&self, view: daemon_protocol::ManageEventView) {
        if let Some(tx) = &self.fleet_events {
            let _ = tx.send(daemon_api::TreeEvent::Subagent(view));
        }
    }

    /// Fold the durable per-session usage totals across every known session — the node-wide
    /// accounting line (tokens, cache, reasoning, estimated `cost_micros`) reported on `stats`.
    pub(crate) async fn folded_usage(&self) -> UsageDelta {
        let mut total = UsageDelta::default();
        for (session, _status) in self.store.list_sessions().await {
            total.add(&self.store.usage_of(&session).await);
        }
        total
    }

    /// The unified, unscoped roster: every durable `session_record` row plus every live-interactive
    /// session, each enriched with its host meta (profile/title/last_activity/role/parent). The
    /// durable status wins when an id exists in both. The scope filter, sort, and pagination are
    /// applied by [`ControlApi::sessions_query`] on top of this.
    pub(crate) async fn roster(&self) -> Vec<SessionInfo> {
        let mut seen: std::collections::HashSet<SessionId> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (session, status) in self.store.list_sessions().await {
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            out.push(session_info_from(
                &session,
                Some(status),
                &meta,
                ApiLifecycle::Durable,
            ));
            seen.insert(session);
        }
        for session in self.live.live_ids() {
            if seen.contains(&session) {
                continue;
            }
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            out.push(session_info_from(&session, None, &meta, ApiLifecycle::Live));
        }
        out
    }

    /// Record activity on `session` from an inbound `command`: stamp `last_activity_ms` to now
    /// (roster sort key), seed a title from the first user turn when none is set, and index the
    /// turn's user text into the durable FTS surface (`session_search`). Read-modify-writes the host
    /// meta so the overlay/profile/role stay intact. Best-effort: a store error is swallowed (a
    /// missed stamp/index must never fail a submit).
    pub(crate) async fn note_activity(&self, session: &SessionId, command: &AgentCommand) {
        let turn_text = match command {
            AgentCommand::StartTurn { input, .. } => Some(input.text.clone()),
            AgentCommand::Steer { text, .. } => Some(text.clone()),
            _ => None,
        };
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        meta.last_activity_ms = Some(now_ms());
        // Seed a roster title from the opening user turn (truncated) when the session has none yet;
        // a real generated title can replace it later (the field is the foundation).
        if meta.title.is_none() {
            meta.title = seed_title(turn_text.as_deref());
        }
        let title = meta.title.clone();
        let _ = self.store.set_session_meta(session, meta).await;
        // L3: a turn touched this session (recency + maybe a seeded title changed), so its roster row
        // is stale. Turn-level granularity (not per-delta — `SessionAdvanced` covers token growth).
        self.emit_session_meta_changed(session);
        if let Some(text) = turn_text.filter(|t| !t.trim().is_empty()) {
            self.store.index_session_text(session, title, &text).await;
        }
        // Materialize any inbound message attachments into the session workspace `inbox/` before the
        // turn runs (daemon-content-transfer-spec.md Phase 2b), node-mediated: the client first
        // `blob_put`s the bytes, then submits the refs; the engine then sees the on-disk files.
        if let AgentCommand::StartTurn { input, .. } = command {
            if !input.attachments.is_empty() {
                self.materialize_inbound(session, &input.attachments).await;
            }
        }
    }

    /// Emit the L3 `SessionMetaChanged` notification for a stale roster row (recency / title / pin /
    /// archive change). No-op when no event feed is wired.
    fn emit_session_meta_changed(&self, session: &SessionId) {
        if let Some(feed) = self.node_feed() {
            let rev = feed.note_roster_change(session);
            feed.emit(NodeEvent::SessionMetaChanged {
                session: session.clone(),
                rev,
            });
        }
    }

    /// Materialize inbound message attachment blobs into the session workspace `inbox/`. Best-effort:
    /// a fetch/write failure is skipped, never failing the submit. No-op when no workspace/blob store
    /// is bound.
    async fn materialize_inbound(&self, session: &SessionId, attachments: &[BlobRef]) {
        let (Some(ws), Some(blobs)) = (&self.workspace, &self.blobs) else {
            return;
        };
        let inbox = ws.roots().session_root(session.as_str()).join("inbox");
        if tokio::fs::create_dir_all(&inbox).await.is_err() {
            return;
        }
        for att in attachments {
            let Ok(bytes) = blobs.get(&att.hash, None).await else {
                continue;
            };
            let name = att
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.bin", att.hash.to_hex()));
            // Guard against a malicious name escaping inbox/ (use only the basename).
            let base = std::path::Path::new(&name)
                .file_name()
                .map(|n| n.to_owned())
                .unwrap_or_else(|| std::ffi::OsStr::new("attachment").to_owned());
            let _ = tokio::fs::write(inbox.join(base), bytes).await;
        }
    }
}

fn map_state(status: SessionStatus) -> SessionState {
    match status {
        SessionStatus::Active => SessionState::Active,
        SessionStatus::Suspended { job_id } => SessionState::Suspended {
            job_id: job_id.to_string(),
        },
        SessionStatus::Ready => SessionState::Ready,
        SessionStatus::Completed => SessionState::Completed,
    }
}

/// Map a store-level [`StoreRole`] to the wire [`SessionRole`] (the two are distinct types so the
/// store stays protocol-free); a `None` role on a legacy meta row is a top-level `Primary`.
fn map_role(role: Option<StoreRole>) -> SessionRole {
    match role {
        Some(StoreRole::Primary) | None => SessionRole::Primary,
        Some(StoreRole::ManagedChild) => SessionRole::ManagedChild,
        Some(StoreRole::EphemeralSubagent) => SessionRole::EphemeralSubagent,
    }
}

/// Unix-millis now (roster `last_activity_ms` stamp).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The default roster page size when [`SessionQuery::limit`] is `0`.
const DEFAULT_ROSTER_PAGE: usize = 50;

/// Whether a roster entry matches a queried [`SessionScope`]. `owned` is the resolved owned-session
/// set, used only by `ByTransport` (empty for other scopes).
pub(crate) fn session_in_scope(
    i: &SessionInfo,
    scope: &SessionScope,
    owned: &std::collections::HashSet<SessionId>,
) -> bool {
    match scope {
        SessionScope::TopLevel => i.role == SessionRole::Primary && !i.archived,
        SessionScope::ByProfile(p) => i.bound_profile.as_ref() == Some(p) && !i.archived,
        SessionScope::ByTransport(_) => owned.contains(&i.session) && !i.archived,
        SessionScope::Archived => i.role == SessionRole::Primary && i.archived,
        SessionScope::All => true,
    }
}

/// Apply cursor pagination in place: skip through the `after` id (exclusive), cap to the effective
/// limit, and return the next cursor (the last retained id) when the page was truncated.
pub(crate) fn paginate_roster(
    roster: &mut Vec<SessionInfo>,
    after: Option<&SessionId>,
    limit: u32,
) -> Option<SessionId> {
    if let Some(after) = after {
        if let Some(pos) = roster.iter().position(|i| &i.session == after) {
            roster.drain(..=pos);
        }
    }
    let limit = if limit == 0 {
        DEFAULT_ROSTER_PAGE
    } else {
        limit as usize
    };
    if roster.len() > limit {
        roster.truncate(limit);
        roster.last().map(|i| i.session.clone())
    } else {
        None
    }
}

/// A roster title seeded from the first user turn: the first line, trimmed to ~60 chars on a word
/// boundary with an ellipsis. A placeholder until a real generated title replaces it.
fn title_from_text(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or(text).trim();
    const MAX: usize = 60;
    if first_line.chars().count() <= MAX {
        return first_line.to_string();
    }
    let truncated: String = first_line.chars().take(MAX).collect();
    let cut = truncated
        .rsplit_once(' ')
        .map(|(h, _)| h)
        .unwrap_or(&truncated);
    format!("{}…", cut.trim_end())
}

/// The roster title to seed from an inbound turn's text when the session has none yet: the first
/// non-empty turn text, truncated by [`title_from_text`]. `None` leaves the existing title intact
/// (no turn text, or whitespace-only).
fn seed_title(turn_text: Option<&str>) -> Option<String> {
    let trimmed = turn_text?.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(title_from_text(trimmed))
}

/// Build a wire [`SessionInfo`] from a session id + its (optional) durable status + host meta +
/// lifecycle. The single place the enriched roster line is assembled, so the durable, live, and
/// detail paths stay consistent. A live session with no durable row reports `Active`.
pub(crate) fn session_info_from(
    session: &SessionId,
    status: Option<SessionStatus>,
    meta: &SessionMeta,
    lifecycle: ApiLifecycle,
) -> SessionInfo {
    SessionInfo {
        session: session.clone(),
        state: status.map(map_state).unwrap_or(SessionState::Active),
        // Daemon-core-backed engines own their conversation state and can truncate it, so durable
        // and live sessions are both rewindable; foreign ACP units are surfaced via the fleet API.
        rewindable: true,
        bound_profile: meta.bound_profile.clone(),
        title: meta.title.clone(),
        last_activity_ms: meta.last_activity_ms,
        lifecycle,
        role: map_role(meta.role),
        parent: meta.parent.clone(),
        pinned: meta.pinned,
        archived: meta.archived,
    }
}

/// Project a fresh tree snapshot, applying the subscriber's ephemeral filter — the re-projection a
/// coalescing `tree_subscribe` collapses a burst into, and the re-sync after a broadcast lag.
pub(crate) async fn filtered_tree(
    this: &NodeApiImpl,
    filter: &daemon_api::TreeSubFilter,
) -> TreeReport {
    let mut report = this.tree().await;
    if !filter.include_ephemeral {
        report
            .nodes
            .retain(|n| n.role != Some(SessionRole::EphemeralSubagent));
    }
    report
}

/// Apply the `TreeSubFilter` to one live bus event for the no-coalesce (forward-every-delta) path.
/// Returns `None` for events a stable-topology-only subscriber filters out (ephemeral subagent
/// deltas); a `Snapshot` is re-filtered to drop ephemeral nodes.
pub(crate) fn forward_event(
    event: daemon_api::TreeEvent,
    filter: &daemon_api::TreeSubFilter,
) -> Option<daemon_api::TreeEvent> {
    match event {
        daemon_api::TreeEvent::Snapshot(mut report) => {
            if !filter.include_ephemeral {
                report
                    .nodes
                    .retain(|n| n.role != Some(SessionRole::EphemeralSubagent));
            }
            Some(daemon_api::TreeEvent::Snapshot(report))
        }
        daemon_api::TreeEvent::Subagent(view) => {
            // Stable-topology-only subscribers drop ephemeral-subagent deltas; everything else
            // forwards unchanged.
            let drop_ephemeral = !filter.include_ephemeral
                && matches!(
                    &view,
                    daemon_protocol::ManageEventView::Subagent { role, .. }
                        if *role == SessionRole::EphemeralSubagent
                );
            if drop_ephemeral {
                None
            } else {
                Some(daemon_api::TreeEvent::Subagent(view))
            }
        }
    }
}
