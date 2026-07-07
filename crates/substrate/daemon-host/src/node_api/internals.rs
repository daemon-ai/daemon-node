// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Live interactive session internals (the §17 actor, poll/drain model) and the L3 node event feed.

use super::*;

/// The session's own attribution for engine-emitted (outbound) merged-log entries.
fn engine_origin() -> Origin {
    Origin {
        transport: TransportId::new("engine"),
        scope: OriginScope::Internal,
        sender: None,
    }
}

/// The attribution stamped on inbound items entering through the node api surface. The api `submit`
/// op carries no per-event origin yet (the surface-aware transports thread real origins in a later
/// phase), so node-api inbound is tagged with this generic local-api origin.
fn api_origin() -> Origin {
    Origin {
        transport: TransportId::new("api"),
        scope: OriginScope::Internal,
        sender: None,
    }
}

/// The floor of the host-internal [`ReqId`] range: snapshot requests the host issues for its own
/// bookkeeping (post-turn FTS indexing / title generation, `live_conv_view`) allocate ids above
/// this, and the event pump swallows their [`AgentEvent::Snapshot`] replies so they never surface
/// on the client drain/log. Client request ids live far below (they are small counters).
const INTERNAL_REQ_BASE: u64 = 1 << 62;

/// Allocate the next host-internal [`ReqId`] (monotonic above [`INTERNAL_REQ_BASE`]).
fn next_internal_req() -> ReqId {
    static NEXT: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(INTERNAL_REQ_BASE);
    ReqId(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

/// Whether an event is the reply to a host-internal snapshot request (see [`INTERNAL_REQ_BASE`]).
fn is_internal_snapshot(ev: &AgentEvent) -> bool {
    matches!(ev, AgentEvent::Snapshot { request_id, .. } if request_id.0 >= INTERNAL_REQ_BASE)
}

// ---------------------------------------------------------------------------
// Live interactive sessions (the §17 actor, exposed via the poll/drain model)
// ---------------------------------------------------------------------------
pub(crate) type Drain = Arc<Mutex<VecDeque<Outbound>>>;
pub(crate) type Pending = Arc<Mutex<HashMap<ReqId, oneshot::Sender<HostResponse>>>>;
pub(crate) type Merged = Arc<Mutex<MergedLog>>;
/// A live session's outbound delivery targets (where its replies post). Seeded from the opening
/// origin; re-pointed by `handover`. The actual posting to a Primary is a chat transport's job (P5);
/// here it is the authoritative session-owned routing state.
pub(crate) type Delivery = Arc<Mutex<Vec<DeliveryTarget>>>;

/// The authoritative, **non-destructive** merged session event log for one live session: one
/// `seq`-stamped timeline across both directions (inbound commands/responses, outbound events +
/// raised host requests). Unlike the destructive `drain` (single-consumer `poll`), this is the
/// multi-surface observability surface — N consumers each page from their own cursor (`log_after`)
/// or hold a live push subscription (`subscribe`), and never steal each other's events.
pub(crate) struct MergedLog {
    /// The session-activation generation (L2 resync): a fresh log after a restart/reactivation
    /// carries a strictly greater epoch, sourced from the durable `SessionMeta.activation_epoch` in
    /// `ensure()`. Stamped onto every `LogPageView` so a client detects a generation change.
    epoch: u64,
    /// The full ordered history as a shared cursored ring (unbounded: a late joiner can backfill
    /// from any cursor). The entry's own `seq` equals its ring id.
    ring: CursoredRing<SessionLogEntry>,
    /// The live fan-out to push subscribers.
    tx: broadcast::Sender<SessionLogEntry>,
    /// This log's session id (stamped onto the `SessionAdvanced` node-event).
    session: SessionId,
    /// The node-wide event feed (L3): every `append` emits a coalesced `SessionAdvanced` so the
    /// client learns an *out-of-focus* session grew without polling it. `None` => no feed wired.
    feed: Option<Arc<NodeEventFeed>>,
}

/// The content of a merged-log entry minus its server-assigned `seq`: the (origin, disposition,
/// payload) triple that travels together through [`MergedLog::append`] / [`LiveSessions::record_inbound`].
pub(crate) struct LogEntryParts {
    pub(crate) origin: Origin,
    pub(crate) disposition: Disposition,
    pub(crate) payload: SessionPayload,
}

impl MergedLog {
    pub(crate) fn new(session: SessionId, epoch: u64, feed: Option<Arc<NodeEventFeed>>) -> Self {
        let (tx, _rx) = broadcast::channel(256);
        // Unbounded ring: seq starts at 1 so the `after_seq` cursor convention (exclusive lower
        // bound; 0 = "from the start") can address the very first entry.
        Self {
            epoch,
            ring: CursoredRing::new(0),
            tx,
            session,
            feed,
        }
    }

    /// Stamp the next `seq`, record the entry, fan it out to live subscribers, and return the
    /// stamped entry (so an in-process pusher delivers exactly what subscribers see).
    pub(crate) fn append(&mut self, direction: Direction, parts: LogEntryParts) -> SessionLogEntry {
        let LogEntryParts {
            origin,
            disposition,
            payload,
        } = parts;
        // The id the ring will assign equals the entry's seq (the ring is monotonic from 1).
        let seq = self.ring.head() + 1;
        let entry = SessionLogEntry {
            seq,
            direction,
            origin,
            disposition,
            payload,
        };
        self.ring.push(entry.clone());
        // A send error only means there are no live subscribers; the history retains the entry.
        let _ = self.tx.send(entry.clone());
        // L3: tell the node-wide feed this session advanced (coalesced per session in the feed's
        // backlog ring). Payload-free — a focused tab streams the entry directly; an out-of-focus
        // observer just learns "this session has new activity at head_seq" and lazily refetches.
        if let Some(feed) = &self.feed {
            feed.emit(NodeEvent::SessionAdvanced {
                session: self.session.clone(),
                epoch: self.epoch,
                head_seq: seq,
            });
        }
        entry
    }

    /// A non-destructive page of entries with `seq > after_seq` (up to `max`, 0 = all).
    pub(crate) fn page(&self, after_seq: u64, max: u32) -> LogPageView {
        let head_seq = self.ring.head();
        let entries: Vec<SessionLogEntry> = self
            .ring
            .page(after_seq, max as usize)
            .into_iter()
            .map(|(_, e)| e)
            .collect();
        let next_seq = entries.last().map(|e| e.seq).unwrap_or(after_seq);
        LogPageView {
            entries,
            next_seq,
            head_seq,
            epoch: self.epoch,
        }
    }

    /// A push stream that backfills `seq > after_seq` from history, then continues live. The caller
    /// holds the log mutex while calling this, so the backlog snapshot and the live subscription are
    /// taken atomically (no entry can slip between them).
    pub(crate) fn subscribe(&self, after_seq: u64) -> LogStream {
        // The ring is unbounded, so the backlog is always the full tail (never a Lagged marker); a
        // lossy lag only arises on the live broadcast below.
        let backlog: Vec<LogStreamItem> = self
            .ring
            .page(after_seq, 0)
            .into_iter()
            .map(|(_, e)| LogStreamItem::Entry(e))
            .collect();
        let rx = self.tx.subscribe();
        // Surface a lossy lag as `LogStreamItem::Lagged` (instead of silently dropping it) so a
        // re-baseline-capable transport can emit a `Reset`; the channel closing ends the stream.
        let live = BroadcastStream::new(rx).map(|r| match r {
            Ok(entry) => LogStreamItem::Entry(entry),
            Err(BroadcastStreamRecvError::Lagged(_)) => LogStreamItem::Lagged,
        });
        stream::iter(backlog).chain(live).boxed()
    }
}

/// The node-wide event feed (L3 `EventsSince`): a retained, cursored ring of payload-free
/// [`NodeEvent`]s plus a live broadcast. Producers `emit`; a client reads via [`Self::page`]
/// (one-shot/long-poll) or [`Self::subscribe`] (push). A reader whose cursor aged out of the ring
/// (or a lagging push subscriber) gets a `ResyncNeeded` event so it re-baselines rather than
/// silently missing notifications. Unlike `fleet_events` (a lossy, cursor-less bus) this is
/// re-readable from a cursor — the property `EventsSince` requires.
pub struct NodeEventFeed {
    inner: Mutex<NodeFeedInner>,
    tx: broadcast::Sender<NodeFeedEntry>,
}

#[derive(Clone)]
pub(crate) struct NodeFeedEntry {
    cursor: u64,
    event: NodeEvent,
}

pub(crate) struct NodeFeedInner {
    /// The bounded retained ring of payload-free events (the shared cursored-ring primitive). Its
    /// cursor is monotonic from 1; an overflow eviction raises the ring's floor (-> `ResyncNeeded`).
    ring: CursoredRing<NodeEvent>,
    /// The monotonic roster revision (L4): stamped onto `RosterChanged`/`SessionMetaChanged` AND
    /// returned by `SessionsQuery`, so the two agree on which generation a refetch reflects. In-memory
    /// — resets to 0 on restart, which makes a stale client `since_rev` unservable (-> full page).
    rev: u64,
    /// L4 delta index: the `rev` at each session's last roster change (rename/pin/archive/activity/
    /// activation). `roster_delta(R)` returns the sessions whose value is `> R`.
    changed: HashMap<SessionId, u64>,
    /// L4 removal tombstones `(rev, session)` for sessions hard-removed from the roster. Effectively
    /// empty today (archive is a *change* with `archived=true`, not a removal; the store has no
    /// hard-delete) — reserved so the wire `removed` field is populated when a delete path lands.
    removed: VecDeque<(u64, SessionId)>,
    /// The monotonic fleet revision: bumped on every fleet/tree change and stamped onto
    /// `FleetChanged` (its coalescing key; the client re-fetches `Tree` regardless of the value).
    fleet_rev: u64,
    /// The monotonic profiles revision (Phase 3): bumped on every profile author/edit/delete and
    /// stamped onto `ProfilesChanged` (its coalescing key; the client re-fetches the profile list).
    profiles_rev: u64,
}

impl NodeEventFeed {
    pub fn new(capacity: usize) -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(256);
        Arc::new(Self {
            inner: Mutex::new(NodeFeedInner {
                ring: CursoredRing::new(capacity),
                rev: 0,
                changed: HashMap::new(),
                removed: VecDeque::new(),
                fleet_rev: 0,
                profiles_rev: 0,
            }),
            tx,
        })
    }

    /// Bump the roster revision, record it as `session`'s last-change rev (L4 delta index), and
    /// return it. The §5 emit hooks call this then stamp the returned `rev` onto the
    /// `RosterChanged`/`SessionMetaChanged` event, so the feed's `rev` and `SessionsQuery.rev` agree.
    pub(crate) fn note_roster_change(&self, session: &SessionId) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.rev += 1;
        let rev = g.rev;
        g.changed.insert(session.clone(), rev);
        rev
    }

    /// The L4 roster delta past `since_rev`: the sessions whose roster metadata changed after
    /// `since_rev`, the sessions removed since then, and the current `rev`. Returns `None` when the
    /// delta is unservable — `since_rev` is ahead of our `rev` (the daemon restarted and reset the
    /// in-memory index, so the client must take a full page) — which the caller maps to a full query.
    pub(crate) fn roster_delta(
        &self,
        since_rev: u64,
    ) -> Option<(Vec<SessionId>, Vec<SessionId>, u64)> {
        let g = self.inner.lock().unwrap();
        if since_rev > g.rev {
            return None;
        }
        let changed: Vec<SessionId> = g
            .changed
            .iter()
            .filter(|(_, rev)| **rev > since_rev)
            .map(|(s, _)| s.clone())
            .collect();
        let removed: Vec<SessionId> = g
            .removed
            .iter()
            .filter(|(rev, _)| *rev > since_rev)
            .map(|(_, s)| s.clone())
            .collect();
        Some((changed, removed, g.rev))
    }

    /// The current roster revision (stamped on every `SessionsQuery` page, delta or full).
    pub(crate) fn roster_rev(&self) -> u64 {
        self.inner.lock().unwrap().rev
    }

    /// Bump the fleet revision and return it. The fleet bridge (in the node crate) calls this then
    /// stamps the returned `rev` onto a `FleetChanged` so a spawn burst collapses to one `Tree` refetch.
    pub fn note_fleet_change(&self) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.fleet_rev += 1;
        g.fleet_rev
    }

    /// Bump the profiles revision and return it (Phase 3). The profile author/delete paths call this
    /// then stamp the returned `rev` onto a `ProfilesChanged`, so a burst of profile writes collapses
    /// to one profile-list refetch.
    pub fn note_profiles_change(&self) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.profiles_rev += 1;
        g.profiles_rev
    }

    /// Assign a cursor, retain in the bounded ring, and broadcast live. Consecutive
    /// `SessionAdvanced` for the same session are coalesced in the *backlog* (latest wins) so a
    /// reconnecting reader isn't flooded; the live broadcast still fires per emit (the client
    /// dedups/throttles per-session activity).
    pub fn emit(&self, event: NodeEvent) {
        let mut g = self.inner.lock().unwrap();
        if let NodeEvent::SessionAdvanced { session, .. } = &event {
            let session = session.clone();
            // Coalesce a superseded per-session advance (floor-exempt: the later one carries the
            // latest head_seq, so no information is lost).
            g.ring.coalesce(
                |e| matches!(e, NodeEvent::SessionAdvanced { session: s, .. } if *s == session),
            );
        }
        // FleetChanged coalesces globally (the client always refetches the whole Tree), so a backlog
        // never holds more than the latest one — a spawn burst is one refetch for a reconnecting reader.
        // Floor-exempt: collapsing superseded fleet pings loses no information.
        if matches!(&event, NodeEvent::FleetChanged { .. }) {
            g.ring
                .coalesce(|e| matches!(e, NodeEvent::FleetChanged { .. }));
        }
        // CatalogChanged coalesces globally too (a refetch reads the whole installed-model
        // catalog), so a burst of installs/deletes is one client refetch.
        if matches!(&event, NodeEvent::CatalogChanged) {
            g.ring.coalesce(|e| matches!(e, NodeEvent::CatalogChanged));
        }
        // ProfilesChanged coalesces globally (a refetch reads the whole profile list), so a burst of
        // profile writes is one client refetch. Floor-exempt: collapsing superseded pings loses no
        // information.
        if matches!(&event, NodeEvent::ProfilesChanged { .. }) {
            g.ring
                .coalesce(|e| matches!(e, NodeEvent::ProfilesChanged { .. }));
        }
        // push assigns the cursor + raises the floor on a capacity eviction.
        let cursor = g.ring.push(event.clone());
        drop(g);
        let _ = self.tx.send(NodeFeedEntry { cursor, event });
    }

    /// The one-shot cursor read: the retained events past `after_cursor` (capped at `max`, `0` = all),
    /// or a single `ResyncNeeded` when `after_cursor` aged out of the ring.
    pub(crate) fn page(&self, after_cursor: u64, max: u32) -> EventsPage {
        let g = self.inner.lock().unwrap();
        let head_cursor = g.ring.head();
        if g.ring.lagged(after_cursor) {
            return EventsPage {
                events: vec![NodeEvent::ResyncNeeded {
                    scope: "all".into(),
                }],
                next_cursor: head_cursor,
                head_cursor,
            };
        }
        let mut events = Vec::new();
        let mut next = after_cursor;
        for (cursor, event) in g.ring.page(after_cursor, max as usize) {
            events.push(event);
            next = cursor;
        }
        EventsPage {
            events,
            next_cursor: next,
            head_cursor,
        }
    }

    /// The push read: backlog (one page per retained event past `after_cursor`, or a `ResyncNeeded`
    /// when aged out) chained to the live broadcast (a lag surfaces as `ResyncNeeded`).
    pub(crate) fn subscribe(&self, after_cursor: u64) -> NodeEventStream {
        let g = self.inner.lock().unwrap();
        let head_cursor = g.ring.head();
        let mut backlog: Vec<EventsPage> = Vec::new();
        if g.ring.lagged(after_cursor) {
            backlog.push(EventsPage {
                events: vec![NodeEvent::ResyncNeeded {
                    scope: "all".into(),
                }],
                next_cursor: head_cursor,
                head_cursor,
            });
        } else {
            for (cursor, event) in g.ring.page(after_cursor, 0) {
                backlog.push(EventsPage {
                    events: vec![event],
                    next_cursor: cursor,
                    head_cursor,
                });
            }
        }
        let rx = self.tx.subscribe();
        drop(g);
        let live = BroadcastStream::new(rx).map(|r| match r {
            Ok(entry) => EventsPage {
                events: vec![entry.event],
                next_cursor: entry.cursor,
                head_cursor: entry.cursor,
            },
            Err(BroadcastStreamRecvError::Lagged(_)) => EventsPage {
                events: vec![NodeEvent::ResyncNeeded {
                    scope: "all".into(),
                }],
                next_cursor: 0,
                head_cursor: 0,
            },
        });
        stream::iter(backlog).chain(live).boxed()
    }
}

/// The node feed IS the profile-change sink (Phase 3): a profile author/edit/delete bumps the
/// profiles revision and emits the coalesced `ProfilesChanged` pointer, so a thin client refetches
/// the profile list without polling. Wired into the shared [`ProfileOps`](crate::ProfileOps) so both
/// the operator ops and the agent `profile_manage` tool emit through one path.
impl crate::ProfileEvents for NodeEventFeed {
    fn profiles_changed(&self) {
        let rev = self.note_profiles_change();
        self.emit(NodeEvent::ProfilesChanged { rev });
    }
}

/// A resident live session's backend handle: the native in-process §17 actor, or a foreign engine
/// session (e.g. an ACP agent) behind the transport-agnostic [`AgentSession`](crate::AgentSession)
/// seam. Both feed the same pump/log/journal; only command dispatch differs (the actor exposes
/// typed calls, a foreign session takes raw [`AgentCommand`]s).
#[derive(Clone)]
pub(crate) enum LiveHandle {
    /// The in-process `daemon-core` engine actor.
    Core(AgentHandle),
    /// A foreign engine session (constructed by the injected [`ForeignSessionFactory`]).
    Foreign(Arc<dyn crate::AgentSession>),
}

impl LiveHandle {
    /// Subscribe to the backend's lossless-primary §17 event stream (identical for both kinds).
    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        match self {
            LiveHandle::Core(handle) => handle.subscribe(),
            LiveHandle::Foreign(session) => session.subscribe(),
        }
    }

    /// Request a read-only snapshot; the reply rides the event stream as [`AgentEvent::Snapshot`]
    /// with the echoed `request_id` (served immediately when idle, or at the next phase boundary).
    async fn request_snapshot(&self, request_id: ReqId) {
        match self {
            LiveHandle::Core(handle) => handle.snapshot(request_id).await,
            LiveHandle::Foreign(session) => {
                session.submit(AgentCommand::Snapshot { request_id }).await;
            }
        }
    }

    /// Whether this backend is a foreign engine (no in-process actor, not rewindable).
    fn is_foreign(&self) -> bool {
        matches!(self, LiveHandle::Foreign(_))
    }
}

pub(crate) struct LiveSession {
    handle: LiveHandle,
    drain: Drain,
    pending: Pending,
    /// The non-destructive merged event log (multi-surface observability).
    log: Merged,
    /// Where this session's outbound replies post (the `Primary`) + passive `Spectator`s.
    delivery: Delivery,
    /// The event pump task; aborted when the session is dropped.
    pump: JoinHandle<()>,
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.pump.abort();
    }
}

pub(crate) struct LiveSessions {
    sessions: DashMap<SessionId, LiveSession>,
    builder: SessionEngineBuilder,
    /// The durable session store: read at `ensure` to restore a session's persisted overlay (so a
    /// live model/tools/mode override survives an actor respawn) and to record its bound profile.
    store: Arc<dyn SessionStore>,
    /// The verifiable-journal store + signer, when journaling is enabled for live sessions.
    journal: Mutex<Option<JournalConfig>>,
    /// The §12 workspace-checkpoint store, when wired: a `RewindTo` rolls the filesystem back to the
    /// sealed-off range's earliest pre-mutation checkpoint (conversation-rewind spec §6).
    checkpoints: Mutex<Option<Arc<dyn daemon_core::CheckpointStore>>>,
    /// The §4.3 background-spawn materializer, when configured: lets a live session's `Effect::Spawn`
    /// materialize an attached non-joining review child without parking (fire-and-forget).
    background: Mutex<Option<Arc<crate::background::BackgroundSpawner>>>,
    /// The per-session live edit-approval policy (shared with `NodeApiImpl::session_modes`), read by
    /// each session's [`ParkingHandler`] to auto-allow / deny without parking a human.
    modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
    /// In-process outbound push sinks keyed by transport instance (daemon-event-io-spec §5.9.3): a
    /// registered sink receives every outbound entry of every session whose `Primary` it owns,
    /// resolved live by the per-session pump (so handover demotion stops/starts delivery for free).
    /// Shared with each pump task; a missing instance simply means no in-process push (pull-only).
    sinks: Arc<DashMap<TransportId, Arc<dyn DeliverySink>>>,
    /// The node-wide event feed (L3), shared from `NodeApiImpl`: the §5 emit hooks here
    /// (`SessionAdvanced` at `MergedLog::append`, `SessionMetaChanged`/`RosterChanged` at
    /// `note_activity`/`ensure`, `ApprovalPending` in the live `ParkingHandler`) push onto it. `None`
    /// until `set_node_events` wires it (a node assembled without a feed leaves it unset).
    node_events: Mutex<Option<Arc<NodeEventFeed>>>,
    /// The auxiliary provider for background session-title generation, when configured: the live
    /// event pump fires one best-effort `generate_title` call after a session's first exchange and
    /// persists the result over the truncation-seeded roster title. `None` keeps seeds only.
    title_aux: Mutex<Option<Arc<dyn Provider>>>,
    /// Sessions this residency already attempted title generation for (once-per-residency guard, so
    /// a failed aux call is not retried on every subsequent turn).
    titled: Arc<DashMap<SessionId, ()>>,
    /// Per-session live model selector (Phase 3): the last-seen `Model` selector a resident foreign
    /// (ACP) session's agent advertised, mirrored from the backend's push feed. Read by `session_get`
    /// (surfaced as `SessionDetail.model_selector`); populated by the per-session watcher spawned in
    /// `ensure` and cleared when the session is shut down. Empty for native + non-advertising sessions.
    selectors: Arc<DashMap<SessionId, daemon_api::ModelSelector>>,
}

/// Record a foreign session's freshly-captured `Model` selector in the sidecar and, when the current
/// selection or choice set actually changed, emit a `SessionMetaChanged` pointer so thin clients
/// refetch `session_get` (dedup keeps a re-report of the same selector event-free). Shared by the
/// per-session watcher and the live set path so both update the surface identically.
fn emit_selector_change(
    selectors: &DashMap<SessionId, daemon_api::ModelSelector>,
    feed: &Option<Arc<NodeEventFeed>>,
    session: &SessionId,
    selector: daemon_api::ModelSelector,
) {
    let changed = selectors
        .get(session)
        .map(|e| *e != selector)
        .unwrap_or(true);
    selectors.insert(session.clone(), selector);
    if changed {
        if let Some(feed) = feed {
            let rev = feed.note_roster_change(session);
            feed.emit(NodeEvent::SessionMetaChanged {
                session: session.clone(),
                rev,
            });
        }
    }
}

impl LiveSessions {
    pub(crate) fn new(
        builder: SessionEngineBuilder,
        modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
        store: Arc<dyn SessionStore>,
    ) -> Self {
        Self {
            sessions: DashMap::new(),
            builder,
            store,
            journal: Mutex::new(None),
            checkpoints: Mutex::new(None),
            background: Mutex::new(None),
            modes,
            sinks: Arc::new(DashMap::new()),
            node_events: Mutex::new(None),
            title_aux: Mutex::new(None),
            titled: Arc::new(DashMap::new()),
            selectors: Arc::new(DashMap::new()),
        }
    }

    /// Wire the auxiliary provider for background session-title generation.
    pub(crate) fn set_title_aux(&self, aux: Arc<dyn Provider>) {
        *self.title_aux.lock().unwrap() = Some(aux);
    }

    /// Wire the node-wide event feed so the emit hooks reach a real ring.
    pub(crate) fn set_node_events(&self, feed: Arc<NodeEventFeed>) {
        *self.node_events.lock().unwrap() = Some(feed);
    }

    /// The node-wide event feed, when wired (cloned out of the mutex for an emit/`bump_rev`).
    pub(crate) fn node_feed(&self) -> Option<Arc<NodeEventFeed>> {
        self.node_events.lock().unwrap().clone()
    }

    pub(crate) fn set_journal(&self, cfg: JournalConfig) {
        *self.journal.lock().unwrap() = Some(cfg);
    }

    pub(crate) fn set_checkpoints(&self, checkpoints: Arc<dyn daemon_core::CheckpointStore>) {
        *self.checkpoints.lock().unwrap() = Some(checkpoints);
    }

    pub(crate) fn set_background(&self, background: Arc<crate::background::BackgroundSpawner>) {
        *self.background.lock().unwrap() = Some(background);
    }

    /// The in-process actor handle for `session` only if it is already resident AND runs the
    /// native engine (does not spawn a new actor). `None` for a foreign-engine session — the
    /// actor-only surfaces (live provider swap, engine-side policy switch) have no foreign
    /// counterpart and no-op/fail explicitly at their call sites.
    pub(crate) fn handle_if_live(&self, session: &SessionId) -> Option<AgentHandle> {
        self.sessions.get(session).and_then(|s| match &s.handle {
            LiveHandle::Core(handle) => Some(handle.clone()),
            LiveHandle::Foreign(_) => None,
        })
    }

    /// Whether `session` is resident on the live surface (either backend kind).
    pub(crate) fn is_resident(&self, session: &SessionId) -> bool {
        self.sessions.contains_key(session)
    }

    /// Whether a *resident* session runs a foreign engine (`None` when not resident). Foreign
    /// sessions have no model provider to swap and are not rewindable.
    pub(crate) fn resident_is_foreign(&self, session: &SessionId) -> Option<bool> {
        self.sessions.get(session).map(|s| s.handle.is_foreign())
    }

    /// The last-seen live `Model` selector for a resident foreign session (Phase 3), or `None` for a
    /// native session, a foreign agent that advertises no Model selector, or a non-resident session.
    /// Read by `session_get` to surface `SessionDetail.model_selector`.
    pub(crate) fn model_selector(&self, session: &SessionId) -> Option<daemon_api::ModelSelector> {
        self.selectors.get(session).map(|s| s.clone())
    }

    /// Route a live model change to a resident foreign session's backend (Phase 3): a foreign ACP
    /// `AgentNative` session issues a `set_config_option`; a gateway-routed `NodeProvider` session
    /// re-binds its per-session token. Refreshes the sidecar + emits `SessionMetaChanged` when the
    /// backend reports a resulting selector. `Unsupported` when the session is not foreign/resident
    /// or the backend cannot select a model (e.g. no advertised selector).
    pub(crate) async fn set_foreign_model(
        &self,
        session: &SessionId,
        model: String,
    ) -> Result<(), ApiError> {
        let LiveHandle::Foreign(backend) = self.existing(session)? else {
            return Err(ApiError::Unsupported(
                "per-session model select targets a foreign-engine session".into(),
            ));
        };
        match backend.set_model(model).await {
            Ok(Some(selector)) => {
                emit_selector_change(&self.selectors, &self.node_feed(), session, selector);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => Err(ApiError::Unsupported(e)),
        }
    }

    /// A resident session's read-only conversation view, obtained by round-tripping an internal
    /// [`AgentCommand::Snapshot`] through the actor: subscribe first, request with a host-internal
    /// [`ReqId`] (the pump swallows the reply from client surfaces), and await the echoed
    /// [`AgentEvent::Snapshot`] under a short deadline. `None` when the session is not resident or
    /// no reply arrives in time (e.g. a long-running mid-turn model call — snapshots are served at
    /// phase boundaries).
    pub(crate) async fn conv_view(&self, session: &SessionId) -> Option<ConvView> {
        let handle = self.sessions.get(session).map(|s| s.handle.clone())?;
        // Subscribe BEFORE requesting so the reply cannot be missed.
        let mut rx = handle.subscribe();
        let req = next_internal_req();
        handle.request_snapshot(req).await;
        tokio::time::timeout(std::time::Duration::from_secs(3), async move {
            loop {
                match rx.recv().await {
                    Ok(AgentEvent::Snapshot {
                        request_id, view, ..
                    }) if request_id == req => return Some(view),
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        })
        .await
        .ok()
        .flatten()
    }

    /// Spawn (or reuse) the backend for `session`, returning its handle. The `profile` selects
    /// which profile bundle a *new* session's engine is built from (the routing agent-selection
    /// seam); a resident session ignores it (the first `ensure` binds the profile — bindings are
    /// sticky).
    ///
    /// The session's persisted [`SessionOverlay`] is read from the store and applied on top of the
    /// bound profile at build time, so a live model/tools/approval override is **restored** when the
    /// actor is (re)spawned (e.g. after a host restart). The first `ensure` also records the bound
    /// profile in the store metadata, so the durable path can later re-resolve the same profile.
    ///
    /// Fallible: a foreign-engine profile resolves its ACP catalog entry at spawn time (the recipe
    /// lives node-side, keyed by name), so a vanished or no-longer-installed agent fails the open
    /// with a clear [`ApiError`] here instead of a dead actor. Native construction cannot fail.
    pub(crate) async fn ensure(
        &self,
        session: &SessionId,
        profile: Option<ProfileRef>,
    ) -> Result<LiveHandle, ApiError> {
        if let Some(s) = self.sessions.get(session) {
            return Ok(s.handle.clone());
        }
        // Read (and, for a new session, establish) the host-level session metadata: the bound
        // profile + persisted overlay. A read-modify-write keeps the overlay intact when we are only
        // stamping the bound profile for the first time.
        let mut meta = self.store.session_meta(session).await.unwrap_or_default();
        if meta.bound_profile.is_none() && profile.is_some() {
            meta.bound_profile = profile.clone();
        }
        // The engine resolves from the STICKY binding (just adopted, or persisted earlier — e.g. a
        // node-authoritative `session_create` that bound a profile before any submit): bindings are
        // authoritative over the caller's `profile` hint, and a bare `ensure(None)` on a bound
        // session must not silently fall back to the node's active default.
        let effective_profile = meta.bound_profile.clone();
        // L2 resync: stamp this activation's epoch and bump the stored generation, so the next
        // activation (including after a daemon restart - SessionMeta is durable, the live MergedLog
        // is not) yields a strictly greater epoch. The first activation is epoch 0 (matching
        // `Snapshot::fresh`). The live `submit` path has no `SessionRecord`, so this sidecar is the
        // durable epoch source. Always persist (the generation changed even when the profile did not).
        let epoch = meta.activation_epoch;
        meta.activation_epoch = epoch + 1;
        let _ = self.store.set_session_meta(session, meta.clone()).await;
        // L3: a session (re)entered the live roster — the roster *set* changed, so a client refetches
        // it (a delta query is L4). Fires on first activation and on re-activation after a restart.
        if let Some(feed) = self.node_feed() {
            let rev = feed.note_roster_change(session);
            feed.emit(NodeEvent::RosterChanged { rev });
        }
        let overlay = decode_overlay(&meta.overlay);
        let backend = (self.builder)(session.clone(), effective_profile, &overlay);
        let drain: Drain = Arc::new(Mutex::new(VecDeque::new()));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let log: Merged = Arc::new(Mutex::new(MergedLog::new(
            session.clone(),
            epoch,
            self.node_feed(),
        )));
        let delivery: Delivery = Arc::new(Mutex::new(Vec::new()));
        // A per-session journal feeder (keyed by SessionId), shared by the event pump and the
        // request handler so the live transcript is sealed per turn into the unified journal.
        let feeder: Option<Arc<JournalFeeder>> = self.journal.lock().unwrap().as_ref().map(|cfg| {
            let sink = JournalSink::new(
                cfg.store.clone(),
                cfg.signer.clone(),
                JournalStreamId::session(session),
            );
            Arc::new(JournalFeeder::new(Arc::new(sink)))
        });
        let host = Arc::new(ParkingHandler {
            drain: drain.clone(),
            pending: pending.clone(),
            log: log.clone(),
            journal: feeder.clone(),
            session: session.clone(),
            background: self.background.lock().unwrap().clone(),
            modes: self.modes.clone(),
            feed: self.node_feed(),
        });
        // Materialize the backend: the native engine runs on the in-process §17 actor; a foreign
        // engine is constructed by the injected factory (which resolves its catalog recipe and can
        // fail with a clear error — the "re-check at spawn time" half of engine validation). Both
        // route their blocking host requests through the SAME ParkingHandler, so approvals park
        // identically, and both feed the same pump below, so the merged log + journal + delivery
        // are byte-for-byte the native shape.
        let handle: LiveHandle = match backend {
            SessionBackend::Core(engine) => LiveHandle::Core(spawn_agent_session(engine, host)),
            SessionBackend::Foreign(factory) => LiveHandle::Foreign(factory(host).await?),
        };

        // Phase 3: mirror a foreign backend's live `Model` selector into the per-session sidecar so
        // `session_get` surfaces it, refreshing on every change (session/new, set_config_option,
        // config_option_update) and emitting `SessionMetaChanged` on a real change. The watcher ends
        // when the backend's selector feed closes (the session dropped).
        if let LiveHandle::Foreign(backend) = &handle {
            if let Some(mut updates) = backend.selector_updates() {
                let selectors = self.selectors.clone();
                let feed = self.node_feed();
                let sess = session.clone();
                tokio::spawn(async move {
                    loop {
                        match updates.recv().await {
                            Ok(selector) => {
                                emit_selector_change(&selectors, &feed, &sess, selector)
                            }
                            Err(broadcast::error::RecvError::Lagged(_)) => continue,
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                });
            }
        }

        // Pump §17 events from the actor broadcast into the destructive drain queue (lossless until
        // polled), record them on the non-destructive merged log (outbound / Context), and feed the
        // verifiable journal (coalesced finished blocks, sealed per turn) when enabled.
        let mut rx = handle.subscribe();
        let pump_drain = drain.clone();
        let pump_log = log.clone();
        let pump_journal = feeder.clone();
        // Clones for the in-process push path (§5.9.3): the pump re-reads the session's *current*
        // delivery targets per event and pushes the just-recorded entry to any registered sink owning
        // a target, so handover (a demoted `Primary`) silently stops one sink and starts the next.
        let pump_delivery = delivery.clone();
        let pump_sinks = self.sinks.clone();
        // Clones for the turn-boundary bookkeeping (FTS indexing + title generation): on every
        // `TurnFinished` the pump requests an internal snapshot from the actor; the reply's
        // `ConvView` feeds `index_session_text` (the live half of the `session_search` surface) and
        // — once, after the first exchange — the background title generator.
        let pump_handle = handle.clone();
        let pump_store = self.store.clone();
        let pump_feed = self.node_feed();
        let pump_aux = self.title_aux.lock().unwrap().clone();
        let pump_titled = self.titled.clone();
        let pump_session = session.clone();
        let pump = tokio::spawn(async move {
            // The internal snapshot request this pump is awaiting a reply to, if any (the latest
            // `TurnFinished` wins; a stale reply is still fresher than nothing and is used as-is).
            let mut pending_index: Option<ReqId> = None;
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        // Host-internal snapshot replies never reach clients: when it answers OUR
                        // pending request, run the index/title bookkeeping off-path; either way
                        // swallow it (a `live_conv_view` caller awaits it on its own subscription).
                        if is_internal_snapshot(&ev) {
                            if let AgentEvent::Snapshot {
                                request_id, view, ..
                            } = ev
                            {
                                if pending_index == Some(request_id) {
                                    pending_index = None;
                                    tokio::spawn(index_and_title_session(
                                        pump_store.clone(),
                                        pump_session.clone(),
                                        view,
                                        pump_aux.clone(),
                                        pump_titled.clone(),
                                        pump_feed.clone(),
                                    ));
                                }
                            }
                            continue;
                        }
                        let turn_finished = matches!(ev, AgentEvent::TurnFinished { .. });
                        // Stamp + record on the merged log, capturing the freshly-stamped entry so the
                        // push path delivers exactly what subscribers see (one seq, one shape).
                        let entry = pump_log.lock().unwrap().append(
                            Direction::Outbound,
                            LogEntryParts {
                                origin: engine_origin(),
                                disposition: Disposition::Context,
                                payload: SessionPayload::Event(ev.clone()),
                            },
                        );
                        let frame = Outbound::Event(ev);
                        pump_drain.lock().unwrap().push_back(frame.clone());
                        if let Some(feeder) = &pump_journal {
                            feeder.feed(&frame).await;
                        }
                        // In-process push: replies post to where the *current* `Primary` points, so
                        // snapshot the live targets (dropping the lock before any await) and push the
                        // just-recorded entry to the registered sink owning each `Primary`. Re-reading
                        // the targets every event is what makes handover free: a demoted matrix
                        // `Primary` falls to `Spectator` (stops receiving) and the new GUI `Primary`
                        // starts, with no work here. Passive `Spectator`s observe via the pull path
                        // (`subscribe`); pull subscribers are unaffected by this additive push.
                        let primaries: Vec<DeliveryTarget> = pump_delivery
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|t| t.kind == SinkKind::Primary)
                            .cloned()
                            .collect();
                        for target in primaries {
                            if let Some(sink) = pump_sinks.get(&target.transport) {
                                let sink = sink.clone();
                                sink.deliver(target, entry.clone()).await;
                            }
                        }
                        // Turn boundary: ask the (now idle) actor for a consistent conversation
                        // view; the internal reply above indexes it + maybe generates a title.
                        if turn_finished {
                            let req = next_internal_req();
                            pending_index = Some(req);
                            pump_handle.request_snapshot(req).await;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        self.sessions.insert(
            session.clone(),
            LiveSession {
                handle: handle.clone(),
                drain,
                pending,
                log,
                delivery,
                pump,
            },
        );
        Ok(handle)
    }

    pub(crate) async fn submit(
        &self,
        auth: &AuthorizedFor<Session>,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        // No external attribution supplied: default to the generic `api` origin.
        self.submit_from(auth, api_origin(), command).await
    }

    pub(crate) async fn submit_from(
        &self,
        auth: &AuthorizedFor<Session>,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        // The target is derived from the ownership proof, never a caller-supplied id (Cluster A).
        let session = auth.session().clone();
        match command {
            AgentCommand::StartTurn { input, request_id } => {
                // Opening command: spawn-if-absent, then run the turn in the background so events
                // (including the terminal `TurnFinished`) flow to the drain queue for `poll`.
                let handle = self.ensure(&session, None).await?;
                // Seed the session's Primary reply sink from the opening origin (where replies post by
                // default), unless one is already in force. Handover re-points it later.
                self.seed_primary(&session, &origin);
                // Record the inbound command on the merged log first, so an observer sees what was
                // submitted ahead of the engine's replies (StartTurn enters the conversation),
                // attributed to the submitting surface's `origin`.
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Context,
                        payload: SessionPayload::Command(AgentCommand::StartTurn {
                            input: input.clone(),
                            request_id,
                        }),
                    },
                );
                match handle {
                    LiveHandle::Core(handle) => {
                        tokio::spawn(async move {
                            let _ = handle.start_turn(input).await;
                        });
                    }
                    // A foreign session backgrounds the turn itself (submit must return promptly);
                    // progress streams out on the same pump.
                    LiveHandle::Foreign(session) => {
                        session
                            .submit(AgentCommand::StartTurn { input, request_id })
                            .await;
                    }
                }
                Ok(())
            }
            AgentCommand::Interrupt { reason } => {
                let handle = self.existing(&session)?;
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Transport,
                        payload: SessionPayload::Command(AgentCommand::Interrupt {
                            reason: reason.clone(),
                        }),
                    },
                );
                match handle {
                    LiveHandle::Core(handle) => handle.interrupt(reason).await,
                    LiveHandle::Foreign(session) => {
                        session.submit(AgentCommand::Interrupt { reason }).await;
                    }
                }
                Ok(())
            }
            AgentCommand::Shutdown => {
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Transport,
                        payload: SessionPayload::Command(AgentCommand::Shutdown),
                    },
                );
                if let Some((_, s)) = self.sessions.remove(&session) {
                    // Drop the live model-selector sidecar for the closing session (Phase 3).
                    self.selectors.remove(&session);
                    match &s.handle {
                        LiveHandle::Core(handle) => handle.shutdown().await,
                        LiveHandle::Foreign(session) => {
                            session.submit(AgentCommand::Shutdown).await;
                        }
                    }
                }
                Ok(())
            }
            AgentCommand::Steer { text, request_id } => {
                // Steer-when-idle opens a fresh turn; mid-turn it is drained at a phase boundary.
                // Either way the ack + any turn events flow to the drain queue via the pump.
                let handle = self.ensure(&session, None).await?;
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Context,
                        payload: SessionPayload::Command(AgentCommand::Steer {
                            text: text.clone(),
                            request_id,
                        }),
                    },
                );
                match handle {
                    LiveHandle::Core(handle) => handle.steer(request_id, text).await,
                    LiveHandle::Foreign(session) => {
                        session
                            .submit(AgentCommand::Steer { text, request_id })
                            .await;
                    }
                }
                Ok(())
            }
            AgentCommand::Observe { input, request_id } => {
                // Context-only append (no turn): spawn-if-absent so the chatter has a conversation to
                // land in, record it as context, then hand it to the actor — which folds it in when
                // idle or queues it for the following turn when busy (event-io §5.9). No turn starts.
                let handle = self.ensure(&session, None).await?;
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Context,
                        payload: SessionPayload::Command(AgentCommand::Observe {
                            input: input.clone(),
                            request_id,
                        }),
                    },
                );
                match handle {
                    LiveHandle::Core(handle) => handle.observe(request_id, input).await,
                    LiveHandle::Foreign(session) => {
                        session
                            .submit(AgentCommand::Observe { input, request_id })
                            .await;
                    }
                }
                Ok(())
            }
            AgentCommand::Snapshot { request_id } => {
                let handle = self.existing(&session)?;
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Transport,
                        payload: SessionPayload::Command(AgentCommand::Snapshot { request_id }),
                    },
                );
                match handle {
                    LiveHandle::Core(handle) => handle.snapshot(request_id).await,
                    LiveHandle::Foreign(session) => {
                        session.submit(AgentCommand::Snapshot { request_id }).await;
                    }
                }
                Ok(())
            }
            AgentCommand::RewindTo { anchor, request_id } => {
                // Conversation rewind (spec §4): the engine interrupts any live turn, truncates +
                // reconstructs + bumps epoch + emits `Rewound`; the host then seals the durable
                // journal and rolls the workspace back to the sealed-off range's earliest checkpoint.
                let LiveHandle::Core(handle) = self.existing(&session)? else {
                    // A foreign (ACP) engine owns its own conversation state and the protocol has
                    // no truncate-at-anchor primitive — surfaced up front as `rewindable = false`;
                    // an explicit submit is refused rather than silently dropped.
                    return Err(ApiError::Unsupported(
                        "conversation rewind is not supported for a foreign-engine (ACP) session"
                            .into(),
                    ));
                };
                self.record_inbound(
                    &session,
                    LogEntryParts {
                        origin,
                        disposition: Disposition::Transport,
                        payload: SessionPayload::Command(AgentCommand::RewindTo {
                            anchor: anchor.clone(),
                            request_id,
                        }),
                    },
                );
                let outcome = handle
                    .rewind_to(request_id, anchor)
                    .await
                    .map_err(|e| ApiError::Other(e.to_string()))?;
                // A bare `RewindTo` command rewinds the conversation *and* rolls the workspace back —
                // the historical behavior. The finer conversation-only rewind is reachable via the
                // unified `ControlApi::rewind` op with `restore_workspace = false`.
                self.seal_and_rollback_after_rewind(&session, &outcome, true)
                    .await;
                Ok(())
            }
            _ => Err(ApiError::Unsupported("unknown agent command".into())),
        }
    }

    /// Apply the durable side-effects of a conversation rewind for this live session: seal the
    /// journal (when journaled) and roll the workspace back to the dropped range's earliest
    /// checkpoint. Delegates to the shared [`apply_rewind_side_effects`] helper so the live path and
    /// the managed/fleet path ([`crate::unit::LiveAgentSession`]) stay byte-for-byte consistent.
    pub(crate) async fn seal_and_rollback_after_rewind(
        &self,
        session: &SessionId,
        outcome: &daemon_core::RewindOutcome,
        restore_workspace: bool,
    ) {
        let journaled = self.journal.lock().unwrap().is_some();
        let checkpoints = self.checkpoints.lock().unwrap().clone();
        apply_rewind_side_effects(RewindSideEffects {
            store: &self.store,
            checkpoints: checkpoints.as_ref(),
            journaled,
            session,
            outcome,
            restore_workspace,
        })
        .await;
    }

    /// Rewind a *resident* session's transcript at `anchor` (in-process engine truncate + epoch bump),
    /// then apply the shared durable side-effects honoring `restore_workspace`. The host-spec unified
    /// rewind seam for the live path; backs [`NodeApiImpl::rewind`] for a live session. A resident
    /// FOREIGN session is refused explicitly (ACP has no truncate-at-anchor primitive).
    pub(crate) async fn rewind_resident(
        &self,
        auth: &AuthorizedFor<Session>,
        anchor: daemon_protocol::RewindAnchor,
        restore_workspace: bool,
    ) -> Result<(), ApiError> {
        let session = auth.session();
        if self.resident_is_foreign(session) == Some(true) {
            return Err(ApiError::Unsupported(
                "conversation rewind is not supported for a foreign-engine (ACP) session".into(),
            ));
        }
        let handle = self
            .handle_if_live(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let outcome = handle
            .rewind_to(daemon_common::ReqId(0), anchor)
            .await
            .map_err(|e| ApiError::Other(e.to_string()))?;
        self.seal_and_rollback_after_rewind(session, &outcome, restore_workspace)
            .await;
        Ok(())
    }

    /// Append an inbound entry to a live session's merged log (no-op if the session is gone),
    /// attributed to `origin` so per-event provenance is preserved on the authoritative log.
    pub(crate) fn record_inbound(&self, session: &SessionId, parts: LogEntryParts) {
        if let Some(s) = self.sessions.get(session) {
            s.log.lock().unwrap().append(Direction::Inbound, parts);
        }
    }

    /// Record an observability-only transport/meta event (`Disposition::Transport`) on the merged log
    /// — the "GUI attached" / presence / receipt channel. It lands on the live log + broadcast only
    /// (never the engine, never the journal), so it is cache-safe by construction.
    pub(crate) fn record_meta(
        &self,
        auth: &AuthorizedFor<Session>,
        args: RecordMetaArgs,
    ) -> Result<(), ApiError> {
        // The proof is the authority for the target session; `args.session` is ignored.
        let session = auth.session();
        let RecordMetaArgs {
            session: _,
            origin,
            kind,
            body,
        } = args;
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        s.log.lock().unwrap().append(
            Direction::Inbound,
            LogEntryParts {
                origin,
                disposition: Disposition::Transport,
                payload: SessionPayload::Meta { kind, body },
            },
        );
        Ok(())
    }

    /// Seed the session's `Primary` reply sink from the opening origin if none is set yet.
    pub(crate) fn seed_primary(&self, session: &SessionId, origin: &Origin) {
        if let Some(s) = self.sessions.get(session) {
            let mut targets = s.delivery.lock().unwrap();
            if !targets.iter().any(|t| t.kind == SinkKind::Primary) {
                targets.push(origin.primary_target());
            }
        }
    }

    /// Seed the session's `Primary` reply sink to an already-resolved `target` if none is set yet —
    /// the routed-submit counterpart of [`Self::seed_primary`], honoring a binding's pinned delivery.
    pub(crate) fn seed_primary_target(&self, session: &SessionId, target: DeliveryTarget) {
        if let Some(s) = self.sessions.get(session) {
            let mut targets = s.delivery.lock().unwrap();
            if !targets.iter().any(|t| t.kind == SinkKind::Primary) {
                targets.push(target);
            }
        }
    }

    /// The session's current delivery targets (empty if the session is gone).
    pub(crate) fn delivery_targets(&self, auth: &AuthorizedFor<Session>) -> Vec<DeliveryTarget> {
        let session = auth.session();
        match self.sessions.get(session) {
            Some(s) => s.delivery.lock().unwrap().clone(),
            None => Vec::new(),
        }
    }

    /// Every distinct `Primary` delivery target across all live sessions — the resolution of a cron
    /// job's `deliver = "all"` (broadcast a run result to every active conversation's reply sink).
    /// Deduplicated by `(transport, route)` so two sessions posting to the same chat deliver once.
    pub(crate) fn all_primary_targets(&self) -> Vec<DeliveryTarget> {
        let mut out: Vec<DeliveryTarget> = Vec::new();
        for s in self.sessions.iter() {
            for t in s.delivery.lock().unwrap().iter() {
                if t.kind == SinkKind::Primary
                    && !out
                        .iter()
                        .any(|e| e.transport == t.transport && e.route == t.route)
                {
                    out.push(t.clone());
                }
            }
        }
        out
    }

    /// Push a synthesized outbound `entry` to the registered sink owning `target`'s transport
    /// (post-settle cron delivery). A no-op when no sink is registered (pull-only transport).
    pub(crate) async fn push_to_target(&self, target: DeliveryTarget, entry: SessionLogEntry) {
        if let Some(sink) = self.sinks.get(&target.transport).map(|s| s.clone()) {
            sink.deliver(target, entry).await;
        }
    }

    /// The live sessions a transport instance owns for delivery (daemon-event-io-spec §5.9.3): every
    /// resident session whose `Primary` [`DeliveryTarget`] names `transport`. An on-demand scan of
    /// the live table (called on (re)connect, not per-event), so O(live sessions) is acceptable.
    pub(crate) fn delivery_sessions(&self, transport: &TransportId) -> Vec<SessionId> {
        self.sessions
            .iter()
            .filter(|s| {
                s.delivery
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|t| t.kind == SinkKind::Primary && &t.transport == transport)
            })
            .map(|s| s.key().clone())
            .collect()
    }

    /// Every live (in-memory, submit/poll) session id — the visibility half of the unified roster
    /// (these never appear in the durable `list_sessions` until `assign`). An on-demand snapshot scan.
    pub(crate) fn live_ids(&self) -> Vec<SessionId> {
        self.sessions.iter().map(|s| s.key().clone()).collect()
    }

    /// Register an in-process push [`DeliverySink`] for `transport` (a live handle, replacing any
    /// prior sink for the instance). The per-session pump picks it up on the next event.
    pub(crate) fn register_delivery_sink(
        &self,
        transport: TransportId,
        sink: Arc<dyn DeliverySink>,
    ) {
        self.sinks.insert(transport, sink);
    }

    /// Drop the in-process push sink for `transport` (delivery for that instance reverts to pull).
    pub(crate) fn unregister_delivery_sink(&self, transport: &TransportId) {
        self.sinks.remove(transport);
    }

    /// Re-point the session's `Primary` to `target`: any prior `Primary` is demoted to `Spectator`,
    /// any existing entry for the same transport+route is replaced, and `target` is installed as the
    /// new `Primary`.
    pub(crate) fn handover(
        &self,
        auth: &AuthorizedFor<Session>,
        target: DeliveryTarget,
    ) -> Result<(), ApiError> {
        let session = auth.session();
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut targets = s.delivery.lock().unwrap();
        for t in targets.iter_mut() {
            if t.kind == SinkKind::Primary {
                t.kind = SinkKind::Spectator;
            }
        }
        targets.retain(|t| !(t.transport == target.transport && t.route == target.route));
        targets.push(DeliveryTarget::new(
            target.transport,
            target.route.0,
            SinkKind::Primary,
        ));
        Ok(())
    }

    /// Non-destructive cursor page of a live session's merged log (empty if the session is gone).
    pub(crate) fn log_after(
        &self,
        auth: &AuthorizedFor<Session>,
        after_seq: u64,
        max: u32,
    ) -> LogPageView {
        let session = auth.session();
        match self.sessions.get(session) {
            Some(s) => s.log.lock().unwrap().page(after_seq, max),
            None => LogPageView::default(),
        }
    }

    /// A live push subscription to a session's merged log (empty stream if the session is gone).
    pub(crate) fn subscribe(&self, auth: &AuthorizedFor<Session>, after_seq: u64) -> LogStream {
        let session = auth.session();
        match self.sessions.get(session) {
            Some(s) => s.log.lock().unwrap().subscribe(after_seq),
            None => stream::empty().boxed(),
        }
    }

    /// The activation epoch of a live session's merged log (0 if the session is not resident).
    pub(crate) fn log_epoch(&self, auth: &AuthorizedFor<Session>) -> u64 {
        let session = auth.session();
        match self.sessions.get(session) {
            Some(s) => s.log.lock().unwrap().epoch,
            None => 0,
        }
    }

    pub(crate) fn existing(&self, session: &SessionId) -> Result<LiveHandle, ApiError> {
        self.sessions
            .get(session)
            .map(|s| s.handle.clone())
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))
    }

    pub(crate) fn poll(
        &self,
        auth: &AuthorizedFor<Session>,
        max: u32,
    ) -> Result<Vec<Outbound>, ApiError> {
        let session = auth.session();
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut q = s.drain.lock().unwrap();
        let take = if max == 0 {
            q.len()
        } else {
            (max as usize).min(q.len())
        };
        Ok(q.drain(..take).collect())
    }

    pub(crate) fn respond(
        &self,
        auth: &AuthorizedFor<Session>,
        response: HostResponse,
    ) -> Result<(), ApiError> {
        let session = auth.session();
        let s = self
            .sessions
            .get(session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let tx = s.pending.lock().unwrap().remove(&response.request_id);
        match tx {
            Some(tx) => {
                // The answer to a raised host request enters the conversation (inbound / Context).
                s.log.lock().unwrap().append(
                    Direction::Inbound,
                    LogEntryParts {
                        origin: api_origin(),
                        disposition: Disposition::Context,
                        payload: SessionPayload::Response(response.clone()),
                    },
                );
                let _ = tx.send(response);
                Ok(())
            }
            None => Err(ApiError::Other(format!(
                "no parked request {:?} on session {}",
                response.request_id, session
            ))),
        }
    }

    pub(crate) async fn interrupt(&self, auth: &AuthorizedFor<Session>) -> bool {
        let session = auth.session();
        let Some(handle) = self.sessions.get(session).map(|s| s.handle.clone()) else {
            return false;
        };
        match handle {
            LiveHandle::Core(handle) => handle.interrupt(Some("control cancel".into())).await,
            LiveHandle::Foreign(session) => {
                session
                    .submit(AgentCommand::Interrupt {
                        reason: Some("control cancel".into()),
                    })
                    .await;
            }
        }
        true
    }
}

/// The inputs to [`apply_rewind_side_effects`], grouped so the seal + workspace-rollback request
/// travels as one value (the live and managed/fleet rewind paths both build it).
pub(crate) struct RewindSideEffects<'a> {
    pub(crate) store: &'a Arc<dyn SessionStore>,
    pub(crate) checkpoints: Option<&'a Arc<dyn daemon_core::CheckpointStore>>,
    pub(crate) journaled: bool,
    pub(crate) session: &'a SessionId,
    pub(crate) outcome: &'a daemon_core::RewindOutcome,
    pub(crate) restore_workspace: bool,
}

/// The durable side-effects of a conversation rewind (conversation-rewind spec §6), factored out so
/// the live path ([`LiveSessions::seal_and_rollback_after_rewind`]) and the managed/fleet path
/// ([`crate::unit::LiveAgentSession`]) apply *exactly* the same seal + rollback. Previously only the
/// live path sealed/rolled-back, so a rewind on a managed engine silently skipped both — this is the
/// shared helper both now call.
///
/// - **Seal** (when `journaled`): append a `JournalSeal` at the journal head so the dropped tail is
///   marked `sealed_after` while the audit log stays complete.
/// - **Rollback** (when `restore_workspace` and there are dropped tool calls): restore the earliest
///   pre-mutation checkpoint among the dropped calls, undoing every later mutation in the sealed
///   range. A read-only rewound range (no checkpoints) leaves the filesystem untouched.
pub(crate) async fn apply_rewind_side_effects(fx: RewindSideEffects<'_>) {
    let RewindSideEffects {
        store,
        checkpoints,
        journaled,
        session,
        outcome,
        restore_workspace,
    } = fx;
    if journaled {
        let stream = JournalStreamId::session(session);
        let head = store.load_journal(&stream, u64::MAX, 1).await.head_cursor;
        let recorded_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let result = store
            .record_journal_seal(
                &stream,
                daemon_store::JournalSeal {
                    seal_cursor: head,
                    retained_turns: outcome.retained_turns as u64,
                    epoch: outcome.epoch.0,
                    recorded_unix,
                },
            )
            .await;
        match result {
            Ok(()) => tracing::info!(
                trace_id = %current_trace(),
                session = %session,
                seal_cursor = head,
                retained_turns = outcome.retained_turns,
                epoch = outcome.epoch.0,
                "rewind.seal"
            ),
            Err(e) => tracing::warn!(
                trace_id = %current_trace(),
                session = %session,
                error = %e,
                "rewind.seal.failed"
            ),
        }
    }

    if restore_workspace && !outcome.dropped_call_ids.is_empty() {
        if let Some(store) = checkpoints {
            let dropped: std::collections::HashSet<&str> = outcome
                .dropped_call_ids
                .iter()
                .map(|s| s.as_str())
                .collect();
            let mut matching: Vec<_> = store
                .list(Some(session.as_str()))
                .await
                .into_iter()
                .filter(|r| dropped.contains(r.call_id.as_str()))
                .collect();
            matching.sort_by_key(|r| r.created_unix);
            if let Some(earliest) = matching.first() {
                match store.restore(earliest).await {
                    Ok(()) => tracing::info!(
                        trace_id = %current_trace(),
                        session = %session,
                        checkpoint_id = %earliest.id,
                        dropped_call_ids = outcome.dropped_call_ids.len(),
                        "rewind.workspace"
                    ),
                    Err(e) => tracing::warn!(
                        trace_id = %current_trace(),
                        session = %session,
                        checkpoint_id = %earliest.id,
                        error = %e,
                        "rewind.workspace.failed"
                    ),
                }
            } else {
                tracing::debug!(
                    trace_id = %current_trace(),
                    session = %session,
                    reason = "no_matching_checkpoints",
                    dropped_call_ids = outcome.dropped_call_ids.len(),
                    "rewind.workspace.skipped"
                );
            }
        } else {
            tracing::debug!(
                trace_id = %current_trace(),
                session = %session,
                reason = "no_checkpoint_store",
                dropped_call_ids = outcome.dropped_call_ids.len(),
                "rewind.workspace.skipped"
            );
        }
    } else {
        tracing::debug!(
            trace_id = %current_trace(),
            session = %session,
            restore_workspace,
            dropped_call_ids = outcome.dropped_call_ids.len(),
            "rewind.workspace.skipped"
        );
    }
}

impl NodeApiImpl {
    /// A resident live session's read-only conversation view (`None` when the session is not
    /// resident on the live surface or the actor does not reply in time). The seam the
    /// `session_search` agent tool's archive uses to read a live session's turns, and the recap
    /// op's live fallback.
    pub async fn live_conv_view(&self, session: &SessionId) -> Option<ConvView> {
        self.live.conv_view(session).await
    }
}

/// Whether a session's roster title is still replaceable by the background generator: unset, or
/// exactly the truncation seed of the conversation's opening user text. A user rename or an earlier
/// generated title differs from the seed and is never clobbered.
fn title_replaceable(meta: &SessionMeta, first_user: &str) -> bool {
    match &meta.title {
        None => true,
        Some(current) => seed_title(Some(first_user)).as_deref() == Some(current.as_str()),
    }
}

/// The turn-boundary bookkeeping task the live event pump spawns with a fresh [`ConvView`]:
///
/// 1. **Index**: coalesce the conversation (user + assistant text + tool names) and replace the
///    session's FTS row — the live half of the `session_search` surface (the durable incarnation
///    indexes the managed path). Best-effort; the store swallows write errors.
/// 2. **Title** (hermes `maybe_auto_title` parity): once per residency, after the first exchange
///    (≤ 2 user turns), while the roster title is still unset/seeded — fire the auxiliary
///    `generate_title` call and persist the cleaned result, then emit `SessionMetaChanged` so
///    roster subscribers refresh, and refresh the FTS row's title column.
///
/// Runs entirely off the turn path; every failure leaves the seed/index as they were.
async fn index_and_title_session(
    store: Arc<dyn SessionStore>,
    session: SessionId,
    view: ConvView,
    aux: Option<Arc<dyn Provider>>,
    titled: Arc<DashMap<SessionId, ()>>,
    feed: Option<Arc<NodeEventFeed>>,
) {
    use crate::session_index::{coalesce_body, turns_from_view, IndexRole};

    let turns = turns_from_view(&view);
    let body = coalesce_body(&turns);
    let meta = store.session_meta(&session).await.unwrap_or_default();
    if !body.trim().is_empty() {
        store
            .index_session_text(&session, meta.title.clone(), &body)
            .await;
    }

    // Title generation: gated exactly like hermes — first exchange only (≤ 2 user turns), both
    // sides present, title still replaceable — plus a once-per-residency guard so a failed aux
    // call is not retried every turn.
    let Some(aux) = aux else { return };
    let user_turns = turns.iter().filter(|t| t.role == IndexRole::User).count();
    if user_turns == 0 || user_turns > 2 {
        return;
    }
    let first_user = turns
        .iter()
        .find(|t| t.role == IndexRole::User && !t.text.trim().is_empty())
        .map(|t| t.text.clone());
    let first_reply = turns
        .iter()
        .find(|t| t.role == IndexRole::Assistant && !t.text.trim().is_empty())
        .map(|t| t.text.clone());
    let (Some(first_user), Some(first_reply)) = (first_user, first_reply) else {
        return;
    };
    if !title_replaceable(&meta, &first_user) {
        return;
    }
    if titled.insert(session.clone(), ()).is_some() {
        return;
    }
    let Some(title) =
        crate::title_gen::generate_title(aux.as_ref(), &first_user, &first_reply).await
    else {
        return;
    };
    // Re-read before writing: a rename may have landed while the aux call ran — never clobber it.
    let mut fresh = store.session_meta(&session).await.unwrap_or_default();
    if !title_replaceable(&fresh, &first_user) {
        return;
    }
    fresh.title = Some(title.clone());
    if store.set_session_meta(&session, fresh).await.is_err() {
        return;
    }
    // Refresh the FTS row so a search by the generated title's words hits immediately (the body is
    // this turn's; a concurrently-finished turn re-replaces it at its own boundary).
    if !body.trim().is_empty() {
        store.index_session_text(&session, Some(title), &body).await;
    }
    if let Some(feed) = &feed {
        let rev = feed.note_roster_change(&session);
        feed.emit(NodeEvent::SessionMetaChanged { session, rev });
    }
}

/// The session sub-surface's host handler: park each blocking §17 request into the drain queue and
/// a pending table, await its `respond`. Events and parked requests thus ride one ordered queue
/// (daemon-ffi-spec §3.3).
pub(crate) struct ParkingHandler {
    drain: Drain,
    pending: Pending,
    /// The session's non-destructive merged log, so a raised request is observable to every surface.
    log: Merged,
    /// The per-session journal feeder, so a raised request graduates into a durable request block.
    journal: Option<Arc<JournalFeeder>>,
    /// This session's id (the parent of any background spawn it raises).
    session: SessionId,
    /// The §4.3 background-spawn materializer, when configured.
    background: Option<Arc<crate::background::BackgroundSpawner>>,
    /// The shared per-session live edit-approval policy, consulted on an `Approval` request to
    /// auto-allow / deny without parking a human (in lockstep with the engine's snapshot policy).
    modes: Arc<DashMap<SessionId, daemon_core::ApprovalPolicy>>,
    /// The node-wide event feed (L3): emit `ApprovalPending` when an approval parks for a human, so a
    /// client badges it without polling `approvals_pending`. `None` => no feed wired.
    feed: Option<Arc<NodeEventFeed>>,
}

#[async_trait]
impl HostRequestHandler for ParkingHandler {
    async fn request(&self, req: HostRequest) -> HostResponse {
        // §4.3 fire-and-forget spawn: materialize the attached non-joining child immediately and
        // return — never park (parking would block the parent turn, defeating fire-and-forget).
        if let HostRequestKind::Spawn { spec } = &req.kind {
            let child = match &self.background {
                Some(bg) => bg
                    .spawn(&self.session, daemon_common::Epoch::ZERO, spec, None)
                    .await
                    .unwrap_or_else(|| self.session.clone()),
                None => self.session.clone(),
            };
            return HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Spawned(child),
            };
        }
        // Live edit-approval gate: an `Approval` reaching the host has already cleared the engine's
        // policy gate as `Ask`, but consult the live session policy as the host-side authority so a
        // GUI auto-allow / deny mode answers inline without parking a human (mirrors hermes' ACP
        // adapter resolving the mode in-process). `Ask`/`AcceptEdits` fall through to parking.
        if let HostRequestKind::Approval { .. } = &req.kind {
            match self.modes.get(&self.session).map(|p| *p) {
                Some(daemon_core::ApprovalPolicy::AutoAllow) => {
                    return HostResponse {
                        request_id: req.request_id,
                        body: HostResponseBody::Approved {
                            approved: true,
                            allow_permanent: false,
                            reason: None,
                        },
                    };
                }
                Some(daemon_core::ApprovalPolicy::Deny) => {
                    return HostResponse {
                        request_id: req.request_id,
                        body: HostResponseBody::Approved {
                            approved: false,
                            allow_permanent: false,
                            reason: None,
                        },
                    };
                }
                _ => {}
            }
        }
        let is_approval = matches!(req.kind, HostRequestKind::Approval { .. });
        let (tx, rx) = oneshot::channel();
        let request_id = req.request_id;
        self.pending.lock().unwrap().insert(request_id, tx);
        // L3: an approval just parked for operator action — badge it on the feed (the client then
        // fetches the detail via `approvals_pending`). Payload-free notification only.
        if is_approval {
            if let Some(feed) = &self.feed {
                feed.emit(NodeEvent::ApprovalPending {
                    session: self.session.clone(),
                    request_id: request_id.0.to_string(),
                });
            }
        }
        // Record the raised request on the merged log (outbound / Context) under the unified seq, so
        // it shares one ordered timeline with events and the eventual inbound response.
        self.log.lock().unwrap().append(
            Direction::Outbound,
            LogEntryParts {
                origin: engine_origin(),
                disposition: Disposition::Context,
                payload: SessionPayload::Request(req.clone()),
            },
        );
        let frame = Outbound::Request(req);
        if let Some(feeder) = &self.journal {
            feeder.feed(&frame).await;
        }
        self.drain.lock().unwrap().push_back(frame);
        match rx.await {
            Ok(resp) => resp,
            // The session was dropped before an answer arrived: decline safely.
            Err(_) => HostResponse {
                request_id,
                body: HostResponseBody::Approved {
                    approved: false,
                    allow_permanent: false,
                    reason: None,
                },
            },
        }
    }
}

#[cfg(test)]
mod node_feed_tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    pub(crate) fn page_resyncs_when_cursor_aged_out_of_the_ring() {
        // A tiny ring (capacity 2) so a few emits push the early cursors out.
        let feed = NodeEventFeed::new(2);
        for rev in 1..=5 {
            feed.emit(NodeEvent::RosterChanged { rev });
        }
        // The ring retains only the last two (cursors 4,5). A reader still at cursor 0 lost 1..=3, so
        // it must be told to re-baseline rather than silently miss them.
        let page = feed.page(0, 0);
        assert_eq!(
            page.events,
            vec![NodeEvent::ResyncNeeded {
                scope: "all".into()
            }],
            "an aged-out cursor must surface ResyncNeeded"
        );
        assert_eq!(page.head_cursor, 5);

        // A reader still within the ring (cursor 3 -> 4,5) reads forward, no resync.
        let page = feed.page(3, 0);
        assert_eq!(
            page.events,
            vec![
                NodeEvent::RosterChanged { rev: 4 },
                NodeEvent::RosterChanged { rev: 5 }
            ]
        );
        assert_eq!(page.next_cursor, 5);
    }

    #[test]
    pub(crate) fn session_advanced_is_coalesced_per_session_in_the_backlog() {
        let feed = NodeEventFeed::new(64);
        let a = SessionId::new("sa");
        let b = SessionId::new("sb");
        for head in 1..=4 {
            feed.emit(NodeEvent::SessionAdvanced {
                session: a.clone(),
                epoch: 0,
                head_seq: head,
            });
        }
        feed.emit(NodeEvent::SessionAdvanced {
            session: b.clone(),
            epoch: 0,
            head_seq: 9,
        });
        // The backlog keeps one SessionAdvanced per session (latest head_seq), not one per append.
        let page = feed.page(0, 0);
        let advanced: Vec<_> = page
            .events
            .iter()
            .filter_map(|e| match e {
                NodeEvent::SessionAdvanced {
                    session, head_seq, ..
                } => Some((session.clone(), *head_seq)),
                _ => None,
            })
            .collect();
        assert_eq!(
            advanced,
            vec![(a, 4), (b, 9)],
            "coalesced to the latest per session"
        );
    }

    #[test]
    pub(crate) fn fleet_changed_is_coalesced_in_the_backlog() {
        let feed = NodeEventFeed::new(64);
        for rev in 1..=4 {
            feed.emit(NodeEvent::FleetChanged { rev });
        }
        feed.emit(NodeEvent::RosterChanged { rev: 1 }); // a different event is untouched
        feed.emit(NodeEvent::FleetChanged { rev: 5 });
        let page = feed.page(0, 0);
        let fleet = page
            .events
            .iter()
            .filter(|e| matches!(e, NodeEvent::FleetChanged { .. }))
            .count();
        assert_eq!(fleet, 1, "the backlog keeps a single (latest) FleetChanged");
        assert!(
            matches!(page.events.last(), Some(NodeEvent::FleetChanged { rev: 5 })),
            "the latest FleetChanged wins"
        );
        assert!(
            page.events
                .iter()
                .any(|e| matches!(e, NodeEvent::RosterChanged { .. })),
            "FleetChanged coalescing must not drop other events"
        );
    }

    #[tokio::test]
    pub(crate) async fn subscribe_backfills_then_delivers_live() {
        let feed = NodeEventFeed::new(64);
        feed.emit(NodeEvent::RosterChanged { rev: 1 });
        let mut stream = feed.subscribe(0);
        // Backlog first.
        let first = stream.next().await.expect("backlog page");
        assert_eq!(first.events, vec![NodeEvent::RosterChanged { rev: 1 }]);
        // Then a live emit arrives on the same stream.
        feed.emit(NodeEvent::ApprovalPending {
            session: SessionId::new("s"),
            request_id: "r".into(),
        });
        let live = stream.next().await.expect("live page");
        assert!(matches!(
            live.events.as_slice(),
            [NodeEvent::ApprovalPending { .. }]
        ));
    }

    /// The background title generator replaces only an unset or still-seeded title: a user rename
    /// (or an earlier generated title) never gets clobbered.
    #[test]
    pub(crate) fn title_replaceable_guards_renames() {
        let first_user = "please help me with docker networking setup on this host over there";
        let unset = SessionMeta::default();
        assert!(title_replaceable(&unset, first_user));
        let seeded = SessionMeta {
            title: seed_title(Some(first_user)),
            ..SessionMeta::default()
        };
        assert!(title_replaceable(&seeded, first_user));
        let renamed = SessionMeta {
            title: Some("my own name".into()),
            ..SessionMeta::default()
        };
        assert!(!title_replaceable(&renamed, first_user));
        let generated = SessionMeta {
            title: Some("Docker Networking Help".into()),
            ..SessionMeta::default()
        };
        assert!(!title_replaceable(&generated, first_user));
    }
}
