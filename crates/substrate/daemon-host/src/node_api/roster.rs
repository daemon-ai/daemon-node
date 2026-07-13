// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: the fs here writes the daemon-internal engine inbox IPC dir under the node data root
// (not attacker-influenced); raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! Roster / session-meta projection: the unified durable+live session list, per-turn activity
//! stamping, and the tree/roster change notifications pushed onto the fleet bus + L3 event feed.

use super::*;

/// The Auth 4 ownership state of a session id, resolved by
/// [`NodeApiImpl::session_ownership`](NodeApiImpl::session_ownership).
pub(crate) enum SessionOwnership {
    /// No durable row, no live session, no meta — a creation / not-found path.
    Absent,
    /// The session exists and is owned by this `user_id`.
    Owned(String),
    /// The session exists but carries no owner (a pre-Auth-4 / system row).
    LegacyUnowned,
}

impl NodeApiImpl {
    /// The node-wide event feed, when wired (cloned out for an emit / `bump_rev` in the §5 hooks that
    /// hang off `NodeApiImpl` directly — roster/meta changes).
    pub(crate) fn node_feed(&self) -> Option<Arc<NodeEventFeed>> {
        self.node_events.clone()
    }

    /// Emit a node-wide [`NodeEvent`](daemon_api::NodeEvent) onto the `events_subscribe` feed (when
    /// wired). The additive hook the post-`Arc` swarm service (B3) uses to publish its
    /// `SwarmChanged` invalidation pointers onto the existing feed (§10.4 — the app refetches
    /// `swarm_run_detail`). A no-op when no feed is wired (tests / headless).
    pub fn emit_node_event(&self, event: daemon_api::NodeEvent) {
        if let Some(feed) = self.node_feed() {
            feed.emit(event);
        }
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
                // The bus carries the FULL (unscoped) snapshot: this runs in a spawned task with no
                // request principal, and each `tree_subscribe` consumer applies its own Auth 4 owner
                // scope (operators forward it raw; non-operators re-project owner-scoped). Using the
                // owner-scoped `tree()` here would broadcast an empty tree (deny-closed, no principal).
                let report = match &this.fleet {
                    Some(fleet) => fleet.tree().await,
                    None => daemon_api::TreeReport::default(),
                };
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

    /// The unified, unscoped roster rows: every durable `session_record` row plus every
    /// live-interactive session, each enriched with its host meta and paired with its owner
    /// `user_id` (Auth 4). The durable status wins when an id exists in both. The single fan-in for
    /// [`roster_scoped`](Self::roster_scoped) — so the owner read happens exactly once per session.
    pub(crate) async fn roster_rows(&self) -> Vec<(SessionInfo, Option<String>)> {
        let mut seen: std::collections::HashSet<SessionId> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (session, status) in self.store.list_sessions().await {
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            let owner = meta.owner.clone();
            let rewindable = self.session_rewindable(&session);
            out.push((
                session_info_from(
                    &session,
                    Some(status),
                    &meta,
                    ApiLifecycle::Durable,
                    rewindable,
                ),
                owner,
            ));
            seen.insert(session);
        }
        for session in self.live.live_ids() {
            if seen.contains(&session) {
                continue;
            }
            let meta = self.store.session_meta(&session).await.unwrap_or_default();
            let owner = meta.owner.clone();
            let rewindable = self.session_rewindable(&session);
            out.push((
                session_info_from(&session, None, &meta, ApiLifecycle::Live, rewindable),
                owner,
            ));
        }
        out
    }

    /// Whether a session's conversation is rewindable: `daemon-core`-backed engines own their
    /// conversation state and can truncate it (durable sessions and native live sessions alike);
    /// a resident FOREIGN (ACP) session cannot — the agent owns the conversation and ACP has no
    /// truncate-at-anchor primitive. Non-resident sessions default to rewindable (durable = core).
    pub(crate) fn session_rewindable(&self, session: &SessionId) -> bool {
        self.live.resident_is_foreign(session) != Some(true)
    }

    /// The roster scoped to the **current request principal** (Auth 4): a peer sees only sessions it
    /// owns; a `SessionSeeAll` holder (operator) sees every session including legacy `owner IS NULL`
    /// rows; an absent principal sees nothing (deny-closed). The `SessionScope` filter, sort, and
    /// pagination are layered on top by [`ControlApi::sessions_query`].
    pub(crate) async fn roster_scoped(&self) -> Vec<SessionInfo> {
        let principal = crate::request_context::current_principal();
        self.roster_rows()
            .await
            .into_iter()
            .filter(|(_info, owner)| owner_visible(&principal, owner))
            .map(|(info, _owner)| info)
            .collect()
    }

    /// Resolve a session's Auth 4 ownership state: `Absent` (no durable row, no live session, no
    /// meta — a creation/`NotFound` path), `Owned(user_id)`, or `LegacyUnowned` (it exists but
    /// carries no owner — a pre-Auth-4 / system row, reachable only via an override capability).
    pub(crate) async fn session_ownership(&self, session: &SessionId) -> SessionOwnership {
        let meta = self.store.session_meta(session).await;
        let owner = meta.as_ref().and_then(|m| m.owner.clone());
        let exists = meta.is_some()
            || self.store.status(session).await.is_some()
            || self.live.live_ids().iter().any(|s| s == session);
        match (exists, owner) {
            (false, _) => SessionOwnership::Absent,
            (true, Some(owner)) => SessionOwnership::Owned(owner),
            (true, None) => SessionOwnership::LegacyUnowned,
        }
    }

    /// The operator-tier gate for **security-widening** mutations (Cluster E policy partition):
    /// widening a session overlay's autonomy/tool-surface (`approval_mode` -> `AcceptEdits`/
    /// `AutoAllow`, or `ToolsOverride::FullToolset`) and setting cron `workdir`/`enabled_toolsets`.
    /// Requires [`SessionControlAny`](daemon_auth::Capability::SessionControlAny) — held only by
    /// `Role::Operator`/`Admin` (and the synthetic `system`/`internal` in-process principals), never
    /// by `User`/`Viewer`. Fail-closed: a `None` request principal is DENIED, matching the ownership
    /// layer. This is enforced *beneath* the coarse `SessionWrite`/`CronWrite` capability gate — it
    /// partitions the security-relevant subset of an otherwise user-tier write.
    pub(crate) fn require_operator(&self, what: &str) -> Result<(), ApiError> {
        match crate::request_context::current_principal() {
            Some(p) if p.has(daemon_auth::Capability::SessionControlAny) => Ok(()),
            Some(_) => Err(ApiError::Forbidden(format!(
                "{what} requires an operator-tier capability"
            ))),
            None => Err(ApiError::Unauthenticated(
                "no authenticated principal bound to this request".into(),
            )),
        }
    }

    /// The orchestration tree projected for `principal` (Auth 4). A `SessionSeeAll` holder sees the
    /// whole tree; any other principal sees only the subtrees it owns (children inherit the parent's
    /// owner, so a node is retained iff its backing session is owner-visible, and a dropped parent's
    /// child refs / a dropped root are pruned). A sessionless unit has no owner ⇒ operator-only.
    pub(crate) async fn tree_owned(
        &self,
        principal: &Option<daemon_auth::Principal>,
    ) -> TreeReport {
        let mut report = match &self.fleet {
            Some(fleet) => fleet.tree().await,
            None => return TreeReport::default(),
        };
        if principal
            .as_ref()
            .is_some_and(|p| p.has(daemon_auth::Capability::SessionSeeAll))
        {
            return report;
        }
        let mut visible: std::collections::HashSet<UnitId> = std::collections::HashSet::new();
        for node in &report.nodes {
            let owner = match &node.session {
                Some(s) => self.store.session_meta(s).await.and_then(|m| m.owner),
                None => None,
            };
            if owner_visible(principal, &owner) {
                visible.insert(node.id.clone());
            }
        }
        report.nodes.retain(|n| visible.contains(&n.id));
        for node in &mut report.nodes {
            node.children.retain(|c| visible.contains(c));
        }
        if let Some(root) = &report.root {
            if !visible.contains(root) {
                report.root = None;
            }
        }
        report
    }

    /// Auth 4 (F3): whether the current request principal may see fleet unit `id`. Resolves the unit
    /// to its owner via `UnitId -> UnitNode.session -> session_meta.owner` (the same
    /// [`owner_visible`] policy the roster/tree/checkpoints use), so an owned subtree is visible
    /// whole and a foreign one is denied whole. A sessionless or unknown unit has no owner ⇒
    /// operator-only (fail-closed on an unknown owner). A `SessionSeeAll` holder sees every unit.
    /// Mirrors [`tree_owned`](Self::tree_owned)'s per-node ownership resolution (children inherit the
    /// delegating parent's owner at the delegation seam, so the mapping is well-defined at depth).
    pub(crate) async fn unit_owner_visible(&self, id: &UnitId) -> bool {
        let principal = crate::request_context::current_principal();
        let owner = match &self.fleet {
            Some(fleet) => match fleet.unit(id).await {
                Some(node) => match node.session {
                    Some(s) => self.store.session_meta(&s).await.and_then(|m| m.owner),
                    None => None,
                },
                None => None,
            },
            None => None,
        };
        owner_visible(&principal, &owner)
    }

    /// Auth 4 (F4): keep only the node-events a non-`SessionSeeAll` principal may see. The three
    /// session-bearing variants (`SessionAdvanced`/`SessionMetaChanged`/`ApprovalPending`) are
    /// dropped unless the referenced session's owner is visible to `principal`; the payload-free
    /// node-wide pointers (`RosterChanged`/`FleetChanged`/`CatalogChanged`/`DownloadProgress`/
    /// `ResyncNeeded`) carry no foreign session id and pass (the refetch they nudge —
    /// `SessionsQuery`/`Tree`/`ModelCatalog` — is itself owner-scoped or non-session). The page
    /// cursors are left untouched so a client still advances correctly past filtered events.
    /// Fail-closed: a `None` principal sees no session-bearing event (`owner_visible` denies `None`).
    /// A `SessionSeeAll` holder is short-circuited by the caller (the whole feed, unscoped).
    pub(crate) async fn scope_events_page(
        &self,
        mut page: daemon_api::EventsPage,
        principal: &Option<daemon_auth::Principal>,
    ) -> daemon_api::EventsPage {
        use daemon_api::NodeEvent;
        let mut kept = Vec::with_capacity(page.events.len());
        for ev in page.events {
            let session = match &ev {
                NodeEvent::SessionAdvanced { session, .. }
                | NodeEvent::SessionMetaChanged { session, .. }
                | NodeEvent::ApprovalPending { session, .. } => Some(session.clone()),
                _ => None,
            };
            let visible = match session {
                Some(s) => owner_visible(
                    principal,
                    &self.store.session_meta(&s).await.and_then(|m| m.owner),
                ),
                None => true,
            };
            if visible {
                kept.push(ev);
            }
        }
        page.events = kept;
        page
    }

    /// Record activity on `session` from an inbound `command`: stamp `last_activity_ms` to now
    /// (roster sort key) and seed a title from the first user turn when none is set.
    /// Read-modify-writes the host meta so the overlay/profile/role stay intact. Best-effort: a
    /// store error is swallowed (a missed stamp must never fail a submit). FTS indexing happens at
    /// the turn boundary instead (the live event pump / the durable incarnation), where the FULL
    /// coalesced conversation replaces the row — not just this opening turn's text.
    pub(crate) async fn note_activity(&self, session: &SessionId, command: &AgentCommand) {
        let turn_text = match command {
            AgentCommand::StartTurn { input, .. } => Some(input.text.clone()),
            AgentCommand::Steer { text, .. } => Some(text.clone()),
            _ => None,
        };
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        meta.last_activity_ms = Some(now_ms());
        // Auth 4: stamp the owner from the request principal on first touch (creation), never
        // overwriting an existing owner — the first interactive submit fixes ownership for the
        // session's life. A principal-less path (none bound) leaves it `None` (legacy/operator-only).
        if meta.owner.is_none() {
            meta.owner = crate::request_context::current_principal().map(|p| p.user_id);
        }
        // Seed a roster title from the opening user turn (truncated) when the session has none yet;
        // the background title generator replaces the seed after the first exchange completes.
        if meta.title.is_none() {
            meta.title = seed_title(turn_text.as_deref());
        }
        let _ = self.store.set_session_meta(session, meta).await;
        // L3: a turn touched this session (recency + maybe a seeded title changed), so its roster row
        // is stale. Turn-level granularity (not per-delta — `SessionAdvanced` covers token growth).
        self.emit_session_meta_changed(session);
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
            // rung 3 (api/39): stamp the causing op token when this fires inside an op-carrying
            // dispatch (a `SessionUpdateMeta` that minted an op_id); `None` for activity/recency
            // stamps outside any op context. Uniform — the same call reads the dispatch context.
            let origin_op = daemon_api::current_op_id();
            let rev = feed.note_roster_change_op(session, origin_op.clone());
            feed.emit(NodeEvent::SessionMetaChanged {
                session: session.clone(),
                rev,
                origin_op,
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

/// Whether `owner` is visible to `principal` under the Auth 4 enumeration policy:
/// - a `SessionSeeAll` holder (operator, and the synthetic `system`/`internal` principals which are
///   operator-or-above) sees everything, including legacy `owner IS NULL` rows;
/// - any other authenticated principal sees only rows it owns (a peer never sees another's session,
///   and a legacy/unowned row is hidden — deny-closed on an *unknown owner*);
/// - `None` (no bound principal) is DENIED — **fail-closed**. Every legitimate in-process caller now
///   enters an explicit [`system`](crate::RequestContext::system) /
///   [`internal`](crate::RequestContext::internal) scope, so an unscoped read is a bug, not implicit
///   trust; it must reveal nothing.
///
/// Used by every read/enumeration surface (roster, `session_get`, `session_search`, the tree).
pub(crate) fn owner_visible(
    principal: &Option<daemon_auth::Principal>,
    owner: &Option<String>,
) -> bool {
    match principal {
        None => false,
        Some(p) if p.has(daemon_auth::Capability::SessionSeeAll) => true,
        Some(p) => owner.as_deref() == Some(p.user_id.as_str()),
    }
}

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
        // The explicit archived view surfaces archived sessions of ANY role: an archived subagent
        // child (`ManagedChild`/`EphemeralSubagent`) has no other enumeration path — `TopLevel`/
        // `ByProfile`/`ByTransport` exclude archived, and `tree()` only drills from a live parent —
        // so restricting to `Primary` here would strand them. This is the opt-in complete view.
        SessionScope::Archived => i.archived,
        SessionScope::All => true,
    }
}

/// Apply cursor pagination in place: skip through the `after` id (exclusive), cap to the effective
/// limit, and return the next cursor (the last retained id) when the page was truncated.
///
/// Deliberately NOT `daemon_api::paginate` (the shared slicer for the uniform `WirePage` ops):
/// the roster sorts by (pinned, recency, id) while its cursor is the bare session id, and the
/// generic helper's deleted-cursor fallback assumes the listing is sorted BY the cursor key. Here
/// a vanished cursor restarts from the top instead (the roster is a live view; re-serving beats
/// skipping). `SessionPage` also carries extra fields (`rev`/`removed`) outside the envelope.
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
    // An explicit limit is additionally clamped to the wire page bound: the zcbor client codec
    // decodes `sessions` into a fixed WIRE_PAGE_MAX buffer, so a larger page can never be served.
    let limit = if limit == 0 {
        DEFAULT_ROSTER_PAGE
    } else {
        limit as usize
    }
    .min(daemon_api::WIRE_PAGE_MAX);
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
/// (no turn text, or whitespace-only). `pub(crate)`: the live pump's title generator uses it to
/// recognize a still-seeded (replaceable) title.
pub(crate) fn seed_title(turn_text: Option<&str>) -> Option<String> {
    let trimmed = turn_text?.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(title_from_text(trimmed))
}

/// Build a wire [`SessionInfo`] from a session id + its (optional) durable status + host meta +
/// lifecycle. The single place the enriched roster line is assembled, so the durable, live, and
/// detail paths stay consistent. A live session with no durable row reports `Active`.
/// `rewindable` comes from [`NodeApiImpl::session_rewindable`]: `true` for daemon-core-backed
/// engines (durable + native live), `false` for a resident foreign (ACP) session.
pub(crate) fn session_info_from(
    session: &SessionId,
    status: Option<SessionStatus>,
    meta: &SessionMeta,
    lifecycle: ApiLifecycle,
    rewindable: bool,
) -> SessionInfo {
    SessionInfo {
        session: session.clone(),
        state: status.map(map_state).unwrap_or(SessionState::Active),
        rewindable,
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

/// Project a fresh tree snapshot for `principal`, applying the subscriber's ephemeral filter — the
/// re-projection a coalescing `tree_subscribe` collapses a burst into, and the re-sync after a
/// broadcast lag. Owner-scoped (Auth 4): the snapshot is `tree_owned`, so a non-operator subscriber
/// only ever sees its own subtrees.
pub(crate) async fn filtered_tree(
    this: &NodeApiImpl,
    filter: &daemon_api::TreeSubFilter,
    principal: &Option<daemon_auth::Principal>,
) -> TreeReport {
    let mut report = this.tree_owned(principal).await;
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

#[cfg(test)]
mod owner_visible_tests {
    use super::owner_visible;
    use daemon_auth::{Principal, Role};

    fn principal(name: &str, role: Role) -> Option<Principal> {
        Some(Principal::from_roles(name, name, vec![role]))
    }

    #[test]
    fn peer_sees_only_its_own_and_legacy_is_hidden() {
        let alice = principal("alice", Role::User);
        // Owns it -> visible.
        assert!(owner_visible(&alice, &Some("alice".to_string())));
        // Another user's session -> hidden.
        assert!(!owner_visible(&alice, &Some("bob".to_string())));
        // Legacy/unowned (owner NULL) -> hidden from a non-operator (deny-closed on unknown owner).
        assert!(!owner_visible(&alice, &None));
    }

    #[test]
    fn operator_with_see_all_sees_everything_including_legacy() {
        let op = principal("op", Role::Operator); // Operator grants SessionSeeAll
        assert!(owner_visible(&op, &Some("alice".to_string())));
        assert!(owner_visible(&op, &Some("bob".to_string())));
        assert!(owner_visible(&op, &None));
    }

    #[test]
    fn no_principal_is_denied_fail_closed() {
        // Fail-closed: an unscoped read (no bound principal) reveals nothing. Every legitimate
        // in-process caller now enters an explicit `system()` / `internal()` scope instead.
        assert!(!owner_visible(&None, &Some("alice".to_string())));
        assert!(!owner_visible(&None, &None));
    }

    #[test]
    fn internal_marker_sees_all_like_an_operator() {
        // The synthetic in-process `internal` principal (Operator ⇒ SessionSeeAll) crosses ownership
        // for the legitimate embedded callers (delivery pumps, ingest, injection).
        let internal = crate::request_context::RequestContext::internal().principal;
        assert!(owner_visible(
            &Some(internal.clone()),
            &Some("alice".to_string())
        ));
        assert!(owner_visible(&Some(internal), &None));
    }
}
